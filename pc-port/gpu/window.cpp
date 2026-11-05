// SPDX-License-Identifier: GPL-2.0-or-later
#include <stdexcept>
#include <SDL3/SDL.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <algorithm>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <objidl.h>
namespace Gdiplus { using std::min; using std::max; }
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#include "overlay.h"
#include "runtime_timeline.h"
#include "sdl_window.h"
#include "visual_capture.h"
namespace iss::save_ui { bool OnEvent(const SDL_Event& e); }
// Why the game closes (EXIT_REASON lines): the window messages that end it, who sent a WM_CLOSE
// (this thread: the X button or Alt+F4; another thread or process), the console control events
// and the session end. The game once closed by itself with no crash (2026-10-19 20:36) and the
// log could not tell why.
namespace {
WNDPROC g_exit_next_proc{};
void ExitReason(const char* what,unsigned long long a=0,unsigned long long b=0){
    std::fprintf(stderr,"EXIT_REASON %s a=0x%llx b=0x%llx tick_ms=%llu\n",what,a,b,
                 (unsigned long long)SDL_GetTicks());
    std::fflush(stderr);
}
LRESULT CALLBACK ExitWatchProc(HWND hwnd,UINT msg,WPARAM wp,LPARAM lp){
    switch(msg){
    case WM_CLOSE:{
        const DWORD from=InSendMessageEx(nullptr);
        ExitReason((from&ISMEX_SEND)?"wm_close_sent_by_other_thread_or_process":"wm_close_this_thread",
                   from,(unsigned long long)GetAsyncKeyState(VK_MENU)&0x8000);
        break;}
    case WM_SYSCOMMAND:
        if((wp&0xFFF0)==SC_CLOSE)ExitReason("sc_close",wp,lp);
        break;
    case WM_SYSKEYDOWN:
        if(wp==VK_F4)ExitReason("alt_f4");
        break;
    case WM_QUERYENDSESSION:ExitReason("query_end_session",wp,lp);break;
    case WM_ENDSESSION:ExitReason("end_session",wp,lp);break;
    case WM_DESTROY:ExitReason("wm_destroy");break;
    }
    return CallWindowProcW(g_exit_next_proc,hwnd,msg,wp,lp);
}
BOOL WINAPI ExitConsoleCtrl(DWORD type){
    ExitReason("console_ctrl",type);
    return FALSE; // the handlers before this one (the CRT's SIGINT, SDL's quit) still run
}
}
namespace Frontend {
WindowSDL::WindowSDL(s32 w,s32 h,const char *title):width{w},height{h},base_title{title}{
    if(!SDL_InitSubSystem(SDL_INIT_VIDEO))throw std::runtime_error(SDL_GetError());
    const char* display_mode=std::getenv("ISS_FULLSCREEN");
    const bool fullscreen=display_mode&&display_mode[0]=='1';
    const char* hidden_env=std::getenv("ISS_HYBRID_BENCH_HIDDEN");
    const bool hidden=hidden_env && hidden_env[0]=='1';
    window=SDL_CreateWindow(title,w,h,SDL_WINDOW_VULKAN|SDL_WINDOW_RESIZABLE|
        (hidden?SDL_WINDOW_HIDDEN:(fullscreen?SDL_WINDOW_FULLSCREEN:0)));
    if(!window)throw std::runtime_error(SDL_GetError());
    int pixel_width=w,pixel_height=h;
    SDL_GetWindowSizeInPixels(window,&pixel_width,&pixel_height);
    width=pixel_width;height=pixel_height;
    if(const char* icon_path=std::getenv("ISS_WINDOW_ICON")){
        if(auto* icon=SDL_LoadBMP(icon_path)){
            SDL_SetWindowIcon(window,icon);SDL_DestroySurface(icon);
        }else std::fprintf(stderr,"WINDOW_ICON failed: %s\n",SDL_GetError());
    }
    std::fprintf(stderr,"WINDOW_MODE fullscreen=%d size=%dx%d\n",fullscreen,pixel_width,pixel_height);
    const auto properties=SDL_GetWindowProperties(window);
    window_info.type=WindowSystemType::Windows;
    window_info.render_surface=SDL_GetPointerProperty(properties,SDL_PROP_WINDOW_WIN32_HWND_POINTER,nullptr);
    // EXIT_REASON: after SDL (its console handler is registered first, so this one runs first).
    if(HWND hwnd=static_cast<HWND>(window_info.render_surface);hwnd&&!g_exit_next_proc)
        g_exit_next_proc=reinterpret_cast<WNDPROC>(
            SetWindowLongPtrW(hwnd,GWLP_WNDPROC,reinterpret_cast<LONG_PTR>(&ExitWatchProc)));
    SetConsoleCtrlHandler(&ExitConsoleCtrl,TRUE);
    window_info.display_connection=nullptr; // Vulkan's Windows surface path uses the HWND.
    window_info.render_surface_scale=SDL_GetWindowDisplayScale(window);
}
WindowSDL::~WindowSDL(){if(window)SDL_DestroyWindow(window);}
bool WindowSDL::PollEvents(){SDL_Event e;while(SDL_PollEvent(&e)){
    if(iss::save_ui::OnEvent(e))continue; // the save dialog takes the input while it is up
    if(e.type==SDL_EVENT_KEY_DOWN&&!e.key.repeat){
        if(e.key.key==SDLK_RETURN&&(e.key.mod&SDL_KMOD_ALT)){
            const bool fullscreen=(SDL_GetWindowFlags(window)&SDL_WINDOW_FULLSCREEN)!=0;
            if(!SDL_SetWindowFullscreen(window,!fullscreen))
                std::fprintf(stderr,"WINDOW_MODE toggle failed: %s\n",SDL_GetError());
        }
        if(e.key.key==SDLK_F9){bool show=!iss_show_fps.load();iss_show_fps.store(show);std::fprintf(stderr,"FPS_OVERLAY visible=%d tick_ms=%llu\n",show,(unsigned long long)SDL_GetTicks());}
        if(e.key.key==SDLK_F10){iss::timeline::Emit("user_stutter_mark",0,0,SDL_GetTicks());std::fprintf(stderr,"PERF_MARK tick_ms=%llu\n",(unsigned long long)SDL_GetTicks());std::fflush(stderr);if(iss::visual_capture::Enabled())iss::visual_capture::Start(SDL_GetTicksNS());}}
    if(e.type==SDL_EVENT_QUIT||e.type==SDL_EVENT_WINDOW_CLOSE_REQUESTED){
        if(is_open)ExitReason(e.type==SDL_EVENT_QUIT?"sdl_quit":"sdl_window_close_requested");
        is_open=false;
    }
    if(e.type==SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED){width=e.window.data1;height=e.window.data2;}}
    return is_open;
}

// The loading screen while the renderer starts and the shaders preload (RPCS3 style): the game's
// own art (sce_sys/pic1.png, the crow emblem; pic0.png if missing, from ISS_GAME_DIR, as shadPS4's launcher
// shows it), a shade at the bottom, the status and a progress bar. Drawn with GDI+ straight on the
// window until the game's first frame replaces it (ISS_LOADING_SCREEN=0 off).
namespace {
std::wstring Widen(const std::string& text){
    if(text.empty())return {};
    const int n=MultiByteToWideChar(CP_UTF8,0,text.data(),int(text.size()),nullptr,0);
    std::wstring out(size_t(n),L'\0');
    MultiByteToWideChar(CP_UTF8,0,text.data(),int(text.size()),out.data(),n);
    return out;
}
struct LoadingArt{
    bool tried{},enabled{true};
    ULONG_PTR token{};
    std::unique_ptr<Gdiplus::Bitmap> image;
    std::unique_ptr<Gdiplus::Bitmap> scaled; ///< the art fitted to the current client size
    int scaled_w{},scaled_h{};
};
LoadingArt& Art(){static LoadingArt art;return art;}
}

void WindowSDL::DrawLoading(double fraction,const std::string& status){
    auto& art=Art();
    if(!art.tried){
        art.tried=true;
        if(const char* p=std::getenv("ISS_LOADING_SCREEN");p&&p[0]=='0')art.enabled=false;
        Gdiplus::GdiplusStartupInput input;
        if(art.enabled&&Gdiplus::GdiplusStartup(&art.token,&input,nullptr)!=Gdiplus::Ok)art.enabled=false;
        if(art.enabled){
            if(const char* dir=std::getenv("ISS_GAME_DIR")){
                for(const char* name:{"pic1.png","pic0.png"}){
                    const std::wstring path=Widen(dir)+L"\\sce_sys\\"+Widen(name);
                    auto bitmap=std::make_unique<Gdiplus::Bitmap>(path.c_str());
                    if(bitmap->GetLastStatus()==Gdiplus::Ok&&bitmap->GetWidth()>0){art.image=std::move(bitmap);break;}
                }
            }
            std::fprintf(stderr,"LOADING_SCREEN art=%d\n",art.image?1:0);
        }
    }
    if(!art.enabled)return;
    HWND hwnd=static_cast<HWND>(window_info.render_surface);
    if(!hwnd)return;
    RECT client{};GetClientRect(hwnd,&client);
    const int w=client.right-client.left,h=client.bottom-client.top;
    if(w<=0||h<=0)return;
    using namespace Gdiplus;
    // The art scaled to cover the window, kept until the size changes.
    if(!art.scaled||art.scaled_w!=w||art.scaled_h!=h){
        art.scaled=std::make_unique<Bitmap>(w,h,PixelFormat32bppPARGB);
        Graphics g(art.scaled.get());
        g.Clear(Color(255,8,8,10));
        if(art.image){
            g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
            const double iw=art.image->GetWidth(),ih=art.image->GetHeight();
            const double scale=std::max(w/iw,h/ih);
            const REAL dw=REAL(iw*scale),dh=REAL(ih*scale);
            g.DrawImage(art.image.get(),RectF((w-dw)/2,(h-dh)/2,dw,dh));
        }
        // A shade at the bottom for the bar and the text.
        const REAL shade_h=REAL(h)*0.30f;
        LinearGradientBrush shade(PointF(0,REAL(h)-shade_h),PointF(0,REAL(h)),Color(0,0,0,0),Color(220,0,0,0));
        g.FillRectangle(&shade,REAL(0),REAL(h)-shade_h,REAL(w),shade_h);
        art.scaled_w=w;art.scaled_h=h;
    }
    Bitmap frame(w,h,PixelFormat32bppPARGB);
    {
        Graphics g(&frame);
        g.DrawImage(art.scaled.get(),0,0,w,h);
        g.SetSmoothingMode(SmoothingModeAntiAlias);
        g.SetTextRenderingHint(TextRenderingHintAntiAliasGridFit);
        const REAL margin=REAL(w)*0.06f;
        const REAL bar_w=REAL(w)-2*margin,bar_h=std::max<REAL>(6.f,REAL(h)*0.010f);
        const REAL bar_y=REAL(h)*0.90f;
        SolidBrush track(Color(90,255,255,255));
        g.FillRectangle(&track,margin,bar_y,bar_w,bar_h);
        // The logo's palette: yellow to orange along the bar.
        LinearGradientBrush fill(PointF(margin,0),PointF(margin+bar_w,0),Color(255,255,196,46),Color(255,247,120,22));
        if(fraction>=0){
            const REAL done=bar_w*REAL(std::clamp(fraction,0.0,1.0));
            if(done>0)g.FillRectangle(&fill,margin,bar_y,done,bar_h);
        }else{
            // Unknown length: a segment sweeping across.
            const double t=double(GetTickCount64()%1600)/1600.0;
            const REAL seg=bar_w*0.25f;
            const REAL x=margin+REAL(t)*(bar_w+seg)-seg;
            const REAL left=std::max(margin,x),right=std::min(margin+bar_w,x+seg);
            if(right>left)g.FillRectangle(&fill,left,bar_y,right-left,bar_h);
        }
        const REAL text_size=std::max<REAL>(13.f,REAL(h)*0.026f);
        FontFamily family(L"Segoe UI");
        Font font(&family,text_size,FontStyleRegular,UnitPixel);
        SolidBrush text(Color(235,255,255,255));
        const std::wstring status_w=Widen(status);
        g.DrawString(status_w.c_str(),-1,&font,PointF(margin,bar_y-text_size*1.7f),&text);
        if(fraction>=0){
            wchar_t percent[16];swprintf(percent,16,L"%d%%",int(std::clamp(fraction,0.0,1.0)*100+0.5));
            StringFormat right;right.SetAlignment(StringAlignmentFar);
            g.DrawString(percent,-1,&font,RectF(margin,bar_y-text_size*1.7f,bar_w,text_size*1.5f),&right,&text);
        }
    }
    HDC dc=GetDC(hwnd);
    if(dc){
        Graphics out(dc);
        out.SetCompositingMode(CompositingModeSourceCopy);
        out.DrawImage(&frame,0,0,w,h);
        ReleaseDC(hwnd,dc);
    }
}

void WindowSDL::BeginTextInput(const std::string& initial,const std::string& prompt){std::scoped_lock guard{text_mutex};text=initial;text_prompt=prompt;text_state=0;text_requested=true;}
int WindowSDL::PollTextInput(std::string& value){std::scoped_lock guard{text_mutex};value=text;return text_state;}
void WindowSDL::UpdateTextTitle(){}
void WindowSDL::UpdateCursor(){}
}
