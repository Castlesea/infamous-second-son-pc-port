// SPDX-License-Identifier: GPL-2.0-or-later
// The source presenter draws the game image through ImGui; retain actual rendering.
#include <SDL3/SDL.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <imgui.h>
#include <imgui_internal.h>
#include "sdl_window.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderdoc.h"
#include "core/debug_state.h"
#include "overlay.h"
#include "iss_dlss.h"
namespace iss::dlss { extern std::atomic<unsigned long long> g_overlay_evaluates; extern std::atomic<bool> g_present_stall; }
#include "visual_capture.h"
#include "freeze_watchdog.h"

namespace iss::save_ui {
void LoadFonts(ImGuiIO& io);
void Draw(float width, float height, float dt);
bool Active();
}
namespace ImGui::Core {
// While the save dialog is up the presenter keeps drawing (the game may stop flipping).
bool MustKeepDrawing(){return iss::save_ui::Active();}
static const Frontend::WindowSDL *window;
static uint64_t last_tick;
void Initialize(const ::Vulkan::Instance& instance,const Frontend::WindowSDL& w,u32 count,vk::Format format){
    window=&w;ImGui::CreateContext();auto& io=ImGui::GetIO();
    io.ConfigFlags|=ImGuiConfigFlags_DockingEnable;io.IniFilename=nullptr;io.LogFilename=nullptr;
    io.DisplaySize={(float)w.GetWidth(),(float)w.GetHeight()};io.Fonts->AddFontDefault();iss::save_ui::LoadFonts(io);
    ImGui::Vulkan::InitInfo info{
        .instance=instance.GetInstance(),.physical_device=instance.GetPhysicalDevice(),
        .device=instance.GetDevice(),.queue_family=instance.GetPresentQueueFamilyIndex(),
        .queue=instance.GetPresentQueue(),.image_count=count,.min_allocation_size=1024*1024,
        .pipeline_rendering_create_info{.colorAttachmentCount=1,.pColorAttachmentFormats=&format},
        .check_vk_result_fn=[](vk::Result r){if(r!=vk::Result::eSuccess)std::fprintf(stderr,"ImGui Vulkan error: %d\n",int(r));}
    };
    ImGui::Vulkan::Init(info);TextureManager::StartWorker();last_tick=SDL_GetPerformanceCounter();
}
void OnSurfaceFormatChange(vk::Format f){ImGui::Vulkan::OnSurfaceFormatChange(f);}
void Shutdown(vk::Device d){d.waitIdle();TextureManager::StopWorker();ImGui::Vulkan::Shutdown();ImGui::DestroyContext();}
ImGuiID NewFrame(bool reused){
    static unsigned frames=0;++frames;
    iss::freeze_watchdog::OnPresent();
    if(frames==120){const char *p=std::getenv("ISS_CAPTURE_OVERLAY");if(p&&p[0]=='1')VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::WithOverlays);}
    static uint64_t capture_tick=0;const uint64_t tick=SDL_GetTicks();
    static const bool capture=[] {const char *p=std::getenv("ISS_CAPTURE_FRAMES");return p&&p[0]=='1';}();
    if(frames==1||frames==30||frames==120||tick-capture_tick>=15000){capture_tick=tick;std::fprintf(stdout,"GPU_PRESENT frame=%u reused=%d tick_ms=%llu\n",frames,int(reused),(unsigned long long)tick);std::fflush(stdout);if(capture&&!iss::visual_capture::Enabled())VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);}
    if(iss::visual_capture::Enabled())iss::visual_capture::OnFrame(SDL_GetTicksNS(),frames,reused);
    {
        // ISS_SHOT_AT=s1,s2,... (scripted runs): the game's own frame (no other window, the game
        // in the background or not) captured at these seconds of the session.
        static std::vector<uint64_t> shots=[]{
            std::vector<uint64_t> v;const char *p=std::getenv("ISS_SHOT_AT");
            while(p&&*p){v.push_back(uint64_t(std::strtoull(p,nullptr,10))*1000);const char *c=std::strchr(p,',');p=c?c+1:nullptr;}
            return v;}();
        static size_t next_shot=0;
        if(next_shot<shots.size()&&tick>=shots[next_shot]){++next_shot;VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);}
        // ISS_SHOT_FLIPS=n1,n2,...: at these guest flips instead (the same game frame in an A/B,
        // whatever the startup took).
        static std::vector<uint64_t> flip_shots=[]{
            std::vector<uint64_t> v;const char *p=std::getenv("ISS_SHOT_FLIPS");
            while(p&&*p){v.push_back(uint64_t(std::strtoull(p,nullptr,10)));const char *c=std::strchr(p,',');p=c?c+1:nullptr;}
            return v;}();
        static size_t next_flip_shot=0;
        if(next_flip_shot<flip_shots.size()&&DebugState.flip_count.load(std::memory_order_relaxed)>=flip_shots[next_flip_shot]){
            ++next_flip_shot;VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);}
    }
    auto& io=ImGui::GetIO();uint64_t now=SDL_GetPerformanceCounter();
    io.DeltaTime=std::max(0.000001f,float(double(now-last_tick)/double(SDL_GetPerformanceFrequency())));last_tick=now;
    io.DisplaySize={(float)window->GetWidth(),(float)window->GetHeight()};ImGui::NewFrame();
    return ImGui::DockSpaceOverViewport(0,ImGui::GetMainViewport(),int(ImGuiDockNodeFlags_PassthruCentralNode)|int(ImGuiDockNodeFlags_NoTabBar));
}
void Render(const vk::CommandBuffer& command,const vk::ImageView& view,const vk::Extent2D& extent){
    static uint64_t sample_tick=SDL_GetTicksNS(),sample_flips=0;static double fps=0,ms=0;
    const uint64_t now=SDL_GetTicksNS(),flips=DebugState.flip_count.load(std::memory_order_relaxed);
    if(now-sample_tick>=500000000){const auto count=flips-sample_flips;double seconds=double(now-sample_tick)/1e9;fps=double(count)/seconds;ms=count?seconds*1000/double(count):0;sample_tick=now;sample_flips=flips;}
    // ISS_DLSS=1: whether DLSS evaluated in the last half second (on), the player switched it off
    // (Home / Fn+End), or it is not running.
    static uint64_t dlss_sample_evals=0;static int dlss_state=0; // 0 not running, 1 on, 2 off
    if(now-sample_tick<1000000&&iss::dlss::Mode()==1){const auto e=iss::dlss::g_overlay_evaluates.load(std::memory_order_relaxed);
        dlss_state=!iss::dlss::g_ready.load()?0:!iss::dlss::g_user_on.load()?2:e!=dlss_sample_evals?1:0;dlss_sample_evals=e;}
    if(iss_show_fps.load()){char label[320];
        if(iss::dlss::Mode()==1)std::snprintf(label,sizeof(label),"%.1f FPS  |  %.2f ms\nDLSS: %s  (Home / Fn+End)%s\nF9: show/hide",fps,ms,
            dlss_state==1?"ON (DLAA)":dlss_state==2?"OFF (game's AA)":"not running",
            iss::dlss::g_present_stall.load(std::memory_order_relaxed)
                ?"\nFrame limiter slowing presents: NVIDIA Control Panel ->\nProgram Settings -> iss-probe.exe -> Max Frame Rate: Off":"");
        else std::snprintf(label,sizeof(label),"%.1f FPS  |  %.2f ms\nF9: show/hide",fps,ms);
        auto *draw=ImGui::GetForegroundDrawList();ImVec2 pos{16,16},size=ImGui::CalcTextSize(label);
        draw->AddRectFilled(pos,ImVec2{pos.x+size.x+20,pos.y+size.y+16},IM_COL32(0,0,0,190),5);
        draw->AddText(ImVec2{pos.x+10,pos.y+8},IM_COL32(235,245,255,255),label);}
    iss::save_ui::Draw(ImGui::GetIO().DisplaySize.x,ImGui::GetIO().DisplaySize.y,ImGui::GetIO().DeltaTime);
    ImGui::Render();auto *data=ImGui::GetDrawData();ImGui::Vulkan::UpdateTextures(*data,command);
    if(!data->CmdListsCount)return;
    const vk::RenderingAttachmentInfo color{.imageView=view,.imageLayout=vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp=vk::AttachmentLoadOp::eClear,.storeOp=vk::AttachmentStoreOp::eStore};
    const vk::RenderingInfo info{.renderArea=vk::Rect2D{{0,0},extent},.layerCount=1,.colorAttachmentCount=1,.pColorAttachments=&color};
    command.beginRendering(info);ImGui::Vulkan::RenderDrawData(*data,command);command.endRendering();
}
}
