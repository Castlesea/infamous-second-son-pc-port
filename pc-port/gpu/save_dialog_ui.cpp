// SPDX-License-Identifier: GPL-2.0-or-later
// The PS4 save data dialog, drawn over the game (ImGui draw lists, after the game image) in the
// style of the console's system dialog: a dimmed game, a panel with the operation, the list of
// saved data with their icons, titles and dates, a sliding selection, button prompts; messages
// with Yes / No or OK and an animated progress bar. Driven by the controller (D-pad / left stick,
// A / Cross confirms, B / Circle goes back), the keyboard (arrows, Enter, Esc) and the mouse.
// Layout and behavior follow shadPS4's savedatadialog_ui.cpp (GPL-2.0-or-later); the game side of
// the ABI stays in src/save_dialog.c, which opens this through the exports at the bottom.
#include <SDL3/SDL.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <xinput.h>
#include <imgui.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include "imgui/imgui_texture.h"
#include "iss_dialog.h"

namespace iss::save_ui {
namespace {

enum class Input { Up, Down, Left, Right, Accept, Back, PageUp, PageDown };
struct Pointer {
    float x = -1, y = -1;
    bool clicked = false;
    float wheel = 0;
};

struct State {
    std::mutex mutex;
    bool open = false;      // on screen
    bool finished = false;  // a result is waiting for poll
    IssDialog dialog{};
    std::vector<ImGui::RefCountedTexture> icons;
    IssDialogResult result{};
    int focus = 0;          // list item or message button
    float focus_y = 0;      // animated selection (list rows)
    float scroll = 0;       // animated list scroll
    float appear = 0;       // open animation 0 -> 1
    double opened_at = 0;
    std::deque<Input> inputs;
    Pointer pointer;
    bool pad_used = true;
    // The mouse takes over from the controller only when it really moves (6 pixels from where it
    // was at the last controller press, and not within 0.75 s of it): sensor jitter, a resting
    // cursor and the port's own cursor moves (spray mode, mouse touchpad) left the selection
    // stuck under the cursor and the controller seemingly disconnected (2026-11-01). A row or
    // button under the cursor is selected when the cursor enters it, not on every frame.
    double pad_press_at = -1;
    float anchor_x = -1, anchor_y = -1;
    int hover = -1;
    // Repeats of held directions (controller stick and D-pad, keyboard has its own repeat).
    int held_dir = 0;       // -1 up, +1 down, 0 none
    double held_next = 0;
    float stick_y = 0;
    bool dpad_up = false, dpad_down = false;
};
State& S() {
    static State s;
    return s;
}

ImFont* g_regular = nullptr;
ImFont* g_bold = nullptr;
bool g_playstation = false;  // the game's controller is a DualShock 4 / DualSense: Cross / Circle prompts
std::atomic<bool> g_active{false};

double Now() { return double(SDL_GetTicksNS()) / 1e9; }

std::string DateText(int64_t mtime) {
    if (mtime <= 0) return {};
    std::time_t t = std::time_t(mtime);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y/%m/%d   %H:%M", &tm);
    return buf;
}

// The game the launcher runs (ISS_GAME_NAME), else Second Son.
const char* GameName() {
    const char* name = std::getenv("ISS_GAME_NAME");
    return name && *name ? name : "inFAMOUS\xE2\x84\xA2 Second Son";
}

const char* Heading(const IssDialog& d) {
    switch (d.type) {
    case IssDialogLoad: return "Load";
    case IssDialogDelete: return "Delete";
    default: return "Save";
    }
}

bool IsList(const IssDialog& d) { return d.mode == IssDialogList; }
int Choices(const IssDialog& d) {
    if (IsList(d)) return int(d.count);
    if (d.mode == IssDialogProgress || d.buttons == IssDialogNone) return 0;
    return d.buttons == IssDialogYesNo ? 2 : 1;
}

void Finish(State& s, bool canceled) {
    s.result = {};
    s.result.canceled = canceled ? 1 : 0;
    s.result.index = s.focus;
    if (!IsList(s.dialog)) s.result.button = canceled ? 0 : (s.dialog.buttons == IssDialogYesNo && s.focus == 1 ? 2 : 1);
    s.open = false;
    s.finished = true;
    g_active = false;
    std::fprintf(stderr, "SAVE_DIALOG_UI finished canceled=%d index=%d button=%d\n", s.result.canceled,
                 s.result.index, s.result.button);
}

void Apply(State& s, Input in) {
    const int n = Choices(s.dialog);
    const bool list = IsList(s.dialog);
    switch (in) {
    case Input::Up:
        if (list && n) s.focus = std::max(0, s.focus - 1);
        break;
    case Input::Down:
        if (list && n) s.focus = std::min(n - 1, s.focus + 1);
        break;
    case Input::PageUp:
        if (list && n) s.focus = std::max(0, s.focus - 4);
        break;
    case Input::PageDown:
        if (list && n) s.focus = std::min(n - 1, s.focus + 4);
        break;
    case Input::Left:
        if (!list && n == 2) s.focus = 0;
        break;
    case Input::Right:
        if (!list && n == 2) s.focus = 1;
        break;
    case Input::Accept:
        if (n > 0 && Now() - s.opened_at > 0.2) Finish(s, false); // ignore the press that opened it
        break;
    case Input::Back:
        if (s.dialog.mode != IssDialogProgress) Finish(s, true);
        break;
    }
}

ImU32 Col(int r, int g, int b, float a) { return IM_COL32(r, g, b, int(std::clamp(a, 0.0f, 1.0f) * 255)); }

// A controller button prompt: a filled circle with its letter, then the label.
float Prompt(ImDrawList* dl, ImVec2 at, const char* glyph, const char* label, float k, float alpha) {
    const float r = 13 * k;
    const ImVec2 c(at.x + r, at.y + r);
    dl->AddCircleFilled(c, r, Col(235, 238, 245, alpha), 24);
    const ImU32 ink = Col(12, 18, 28, alpha);
    if (g_playstation && !std::strcmp(glyph, "A")) {  // Cross, drawn (no font has the PS symbols)
        const float d = 5.5f * k;
        dl->AddLine(ImVec2(c.x - d, c.y - d), ImVec2(c.x + d, c.y + d), ink, 2.2f * k);
        dl->AddLine(ImVec2(c.x - d, c.y + d), ImVec2(c.x + d, c.y - d), ink, 2.2f * k);
    } else if (g_playstation && !std::strcmp(glyph, "B")) {  // Circle
        dl->AddCircle(c, 6.5f * k, ink, 20, 2.2f * k);
    } else {
        const ImVec2 gs = g_bold->CalcTextSizeA(18 * k, FLT_MAX, 0, glyph);
        dl->AddText(g_bold, 18 * k, ImVec2(c.x - gs.x / 2, c.y - gs.y / 2), ink, glyph);
    }
    dl->AddText(g_regular, 21 * k, ImVec2(at.x + 2 * r + 9 * k, at.y + r - 12 * k), Col(230, 234, 242, alpha), label);
    return 2 * r + 9 * k + g_regular->CalcTextSizeA(21 * k, FLT_MAX, 0, label).x + 30 * k;
}

void DrawFooter(ImDrawList* dl, State& s, ImVec2 a, ImVec2 b, float k, float alpha) {
    dl->AddLine(ImVec2(a.x + 30 * k, b.y - 64 * k), ImVec2(b.x - 30 * k, b.y - 64 * k), Col(255, 255, 255, 0.10f * alpha));
    const int n = Choices(s.dialog);
    const bool list = IsList(s.dialog);
    const char* accept = list ? "Select" : "OK";
    float x = b.x - 40 * k;
    // Laid out right to left.
    struct P { const char* g; const char* l; };
    std::vector<P> prompts;
    if (s.pad_used) {
        if (s.dialog.mode != IssDialogProgress) prompts.push_back({"B", "Back"});
        if (n > 0) prompts.push_back({"A", accept});
    } else {
        if (s.dialog.mode != IssDialogProgress) prompts.push_back({"Esc", "Back"});
        if (n > 0) prompts.push_back({"Enter", accept}); // Segoe UI has no return arrow
    }
    for (auto& p : prompts) {
        const float w = 26 * k + 9 * k + g_regular->CalcTextSizeA(21 * k, FLT_MAX, 0, p.l).x + 30 * k;
        x -= w;
        if (s.pad_used) {
            Prompt(dl, ImVec2(x, b.y - 46 * k), p.g, p.l, k, alpha);
        } else {
            // Keyboard: a key cap.
            const ImVec2 ks = g_bold->CalcTextSizeA(15 * k, FLT_MAX, 0, p.g);
            const ImVec2 k0(x, b.y - 46 * k), k1(x + std::max(26 * k, ks.x + 14 * k), b.y - 20 * k);
            dl->AddRect(k0, k1, Col(235, 238, 245, 0.85f * alpha), 4 * k, 0, 1.5f * k);
            dl->AddText(g_bold, 15 * k, ImVec2((k0.x + k1.x - ks.x) / 2, (k0.y + k1.y - ks.y) / 2), Col(235, 238, 245, alpha), p.g);
            dl->AddText(g_regular, 21 * k, ImVec2(k1.x + 9 * k, b.y - 45 * k), Col(230, 234, 242, alpha), p.l);
        }
    }
}

void DrawList(ImDrawList* dl, State& s, ImVec2 a, ImVec2 b, float k, float alpha, float dt) {
    const IssDialog& d = s.dialog;
    // Header.
    dl->AddText(g_bold, 40 * k, ImVec2(a.x + 40 * k, a.y + 28 * k), Col(255, 255, 255, alpha), Heading(d));
    dl->AddText(g_regular, 19 * k, ImVec2(a.x + 42 * k, a.y + 78 * k), Col(170, 182, 200, alpha),
                GameName());
    dl->AddLine(ImVec2(a.x + 30 * k, a.y + 114 * k), ImVec2(b.x - 30 * k, a.y + 114 * k), Col(255, 255, 255, 0.10f * alpha));

    const ImVec2 la(a.x + 30 * k, a.y + 126 * k), lb(b.x - 30 * k, b.y - 76 * k);
    const float row_h = 116 * k, gap = 10 * k;
    const float view_h = lb.y - la.y;
    const int n = int(d.count);
    if (n == 0) {
        const char* none = "There is no saved data.";
        const ImVec2 ts = g_regular->CalcTextSizeA(24 * k, FLT_MAX, 0, none);
        dl->AddText(g_regular, 24 * k, ImVec2((la.x + lb.x - ts.x) / 2, (la.y + lb.y - ts.y) / 2), Col(200, 208, 220, alpha), none);
        return;
    }
    // Keep the focused row in view, scrolling smoothly.
    const float focus_top = s.focus * (row_h + gap);
    float target_scroll = s.scroll;
    if (focus_top < target_scroll) target_scroll = focus_top;
    if (focus_top + row_h > target_scroll + view_h) target_scroll = focus_top + row_h - view_h;
    target_scroll = std::clamp(target_scroll, 0.0f, std::max(0.0f, n * (row_h + gap) - gap - view_h));
    const float ease = 1.0f - std::exp(-dt * 16.0f);
    s.scroll += (target_scroll - s.scroll) * ease;
    s.focus_y += (focus_top - s.focus_y) * (1.0f - std::exp(-dt * 20.0f));

    int hovered = -1;
    dl->PushClipRect(la, lb, true);
    // The sliding selection: a glow, a lighter body and a bright border.
    {
        const float y = la.y + s.focus_y - s.scroll;
        const ImVec2 r0(la.x, y), r1(lb.x, y + row_h);
        for (int i = 3; i >= 1; --i) {
            const float e = 3.0f * i * k;
            dl->AddRect(ImVec2(r0.x - e, r0.y - e), ImVec2(r1.x + e, r1.y + e), Col(140, 200, 255, 0.07f * alpha), 8 * k, 0, 3 * k);
        }
        dl->AddRectFilled(r0, r1, Col(255, 255, 255, 0.12f * alpha), 6 * k);
        const float pulse = 0.75f + 0.25f * std::sin(float(Now()) * 4.0f);
        dl->AddRect(r0, r1, Col(235, 245, 255, pulse * alpha), 6 * k, 0, 2.2f * k);
    }
    for (int i = 0; i < n; ++i) {
        const float y = la.y + i * (row_h + gap) - s.scroll;
        if (y > lb.y || y + row_h < la.y) continue;
        const IssDialogItem& it = d.items[i];
        const ImVec2 r0(la.x, y), r1(lb.x, y + row_h);
        if (i != s.focus) dl->AddRectFilled(r0, r1, Col(255, 255, 255, 0.045f * alpha), 6 * k);
        // Icon (16:9).
        const ImVec2 i0(r0.x + 12 * k, r0.y + 12 * k), i1(i0.x + 164 * k, i0.y + 92 * k);
        if (size_t(i) < s.icons.size() && s.icons[i]) {
            dl->AddImage(s.icons[i].GetTexture().im_id, i0, i1, ImVec2(0, 0), ImVec2(1, 1), Col(255, 255, 255, alpha));
        } else {
            dl->AddRectFilled(i0, i1, Col(40, 52, 70, alpha), 4 * k);
            if (it.dir[0] == 0) { // new saved data: a plus sign
                const ImVec2 c((i0.x + i1.x) / 2, (i0.y + i1.y) / 2);
                dl->AddLine(ImVec2(c.x - 18 * k, c.y), ImVec2(c.x + 18 * k, c.y), Col(220, 230, 245, alpha), 4 * k);
                dl->AddLine(ImVec2(c.x, c.y - 18 * k), ImVec2(c.x, c.y + 18 * k), Col(220, 230, 245, alpha), 4 * k);
            }
        }
        dl->AddRect(i0, i1, Col(255, 255, 255, 0.12f * alpha), 4 * k);
        // Texts.
        const float tx = i1.x + 20 * k, tw = r1.x - tx - 16 * k;
        const char* title = it.title[0] ? it.title : (it.dir[0] ? it.dir : "New Saved Data");
        dl->AddText(g_bold, 26 * k, ImVec2(tx, r0.y + 14 * k), Col(255, 255, 255, alpha), title, nullptr, tw);
        if (it.subtitle[0])
            dl->AddText(g_regular, 20 * k, ImVec2(tx, r0.y + 48 * k), Col(214, 222, 236, alpha), it.subtitle, nullptr, tw);
        const std::string date = DateText(it.mtime);
        if (!date.empty())
            dl->AddText(g_regular, 18 * k, ImVec2(tx, r0.y + 78 * k), Col(150, 164, 186, alpha), date.c_str());
        // Mouse: hover focuses, click chooses.
        if (s.pointer.x >= r0.x && s.pointer.x <= r1.x && s.pointer.y >= std::max(r0.y, la.y) &&
            s.pointer.y <= std::min(r1.y, lb.y)) {
            hovered = i;
            if (!s.pad_used && s.hover != i) s.focus = i;
            if (s.pointer.clicked) {
                s.focus = i;
                Finish(s, false);
            }
        }
    }
    dl->PopClipRect();
    s.hover = hovered;
    // A scroll bar when the list does not fit.
    const float content = n * (row_h + gap) - gap;
    if (content > view_h) {
        const float bar_h = view_h * view_h / content, bar_y = la.y + (view_h - bar_h) * (s.scroll / (content - view_h));
        dl->AddRectFilled(ImVec2(lb.x + 10 * k, la.y), ImVec2(lb.x + 14 * k, lb.y), Col(255, 255, 255, 0.06f * alpha), 2 * k);
        dl->AddRectFilled(ImVec2(lb.x + 10 * k, bar_y), ImVec2(lb.x + 14 * k, bar_y + bar_h), Col(255, 255, 255, 0.45f * alpha), 2 * k);
    }
}

void DrawMessage(ImDrawList* dl, State& s, ImVec2 a, ImVec2 b, float k, float alpha) {
    const IssDialog& d = s.dialog;
    const float wrap = (b.x - a.x) - 100 * k;
    const ImVec2 ts = g_regular->CalcTextSizeA(25 * k, FLT_MAX, wrap, d.text);
    const float text_y = d.mode == IssDialogProgress ? (a.y + b.y) / 2 - ts.y - 20 * k : a.y + (b.y - a.y - 64 * k) / 2 - ts.y / 2 - 30 * k;
    dl->AddText(g_regular, 25 * k, ImVec2(a.x + 50 * k, text_y), Col(240, 244, 250, alpha), d.text, nullptr, wrap);
    if (d.mode == IssDialogProgress || d.buttons == IssDialogNone) {
        // Indeterminate progress: a light running along a track.
        const ImVec2 t0(a.x + 50 * k, text_y + ts.y + 34 * k), t1(b.x - 50 * k, t0.y + 6 * k);
        dl->AddRectFilled(t0, t1, Col(255, 255, 255, 0.12f * alpha), 3 * k);
        const float w = (t1.x - t0.x) * 0.28f, phase = std::fmod(float(Now()) * 0.7f, 1.0f);
        const float x0 = t0.x + (t1.x - t0.x + w) * phase - w;
        dl->PushClipRect(t0, t1, true);
        dl->AddRectFilledMultiColor(ImVec2(x0, t0.y), ImVec2(x0 + w, t1.y), Col(120, 190, 255, 0), Col(200, 235, 255, alpha),
                                    Col(200, 235, 255, alpha), Col(120, 190, 255, 0));
        dl->PopClipRect();
        return;
    }
    // Buttons, centered.
    const int n = Choices(d);
    const char* labels[2] = {d.buttons == IssDialogYesNo ? "Yes" : "OK", "No"};
    const float bw = 200 * k, bh = 50 * k, spacing = 24 * k;
    const float total = n * bw + (n - 1) * spacing;
    float x = (a.x + b.x - total) / 2;
    const float y = b.y - 64 * k - bh - 34 * k;
    int hovered_button = -1;
    for (int i = 0; i < n; ++i) {
        const ImVec2 r0(x, y), r1(x + bw, y + bh);
        const bool hover = s.pointer.x >= r0.x && s.pointer.x <= r1.x && s.pointer.y >= r0.y && s.pointer.y <= r1.y;
        if (hover) hovered_button = i;
        if (hover && !s.pad_used && s.hover != i) s.focus = i;
        const bool focused = s.focus == i;
        dl->AddRectFilled(r0, r1, focused ? Col(240, 246, 255, alpha) : Col(255, 255, 255, 0.10f * alpha), bh / 2);
        if (!focused) dl->AddRect(r0, r1, Col(255, 255, 255, 0.25f * alpha), bh / 2, 0, 1.5f * k);
        const ImVec2 ls = g_bold->CalcTextSizeA(22 * k, FLT_MAX, 0, labels[i]);
        dl->AddText(g_bold, 22 * k, ImVec2((r0.x + r1.x - ls.x) / 2, (r0.y + r1.y - ls.y) / 2),
                    focused ? Col(14, 22, 34, alpha) : Col(235, 240, 248, alpha), labels[i]);
        if (hover && s.pointer.clicked) {
            s.focus = i;
            Finish(s, false);
        }
        x += bw + spacing;
    }
    s.hover = hovered_button;
}

// Controllers, read every frame: XInput (Xbox pads, Steam Input, the Steam Deck) and the game's own
// controller from src/pad.c (iss_pad_dialog_state, exported by the exe: this DLL's static SDL never
// opens it, so a DualSense / DualShock 4 without Steam was invisible here until 2026-11-08).
enum PadBit : unsigned { PadUp = 1, PadDown = 2, PadLeft = 4, PadRight = 8, PadA = 16, PadB = 32, PadLb = 64, PadRb = 128 };
unsigned ReadPads() {
    unsigned bits = 0;
    using GetState = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
    static GetState xinput = [] {
        for (const wchar_t* dll : {L"xinput1_4.dll", L"xinput1_3.dll", L"xinput9_1_0.dll"})
            if (HMODULE m = LoadLibraryW(dll))
                if (auto f = reinterpret_cast<GetState>(GetProcAddress(m, "XInputGetState"))) return f;
        return GetState(nullptr);
    }();
    constexpr SHORT dead = 16000;
    for (DWORD i = 0; xinput && i < 4; ++i) {
        XINPUT_STATE st{};
        if (xinput(i, &st) != ERROR_SUCCESS) continue;
        const auto& g = st.Gamepad;
        if ((g.wButtons & XINPUT_GAMEPAD_DPAD_UP) || g.sThumbLY > dead) bits |= PadUp;
        if ((g.wButtons & XINPUT_GAMEPAD_DPAD_DOWN) || g.sThumbLY < -dead) bits |= PadDown;
        if ((g.wButtons & XINPUT_GAMEPAD_DPAD_LEFT) || g.sThumbLX < -dead) bits |= PadLeft;
        if ((g.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT) || g.sThumbLX > dead) bits |= PadRight;
        if (g.wButtons & XINPUT_GAMEPAD_A) bits |= PadA;
        if (g.wButtons & XINPUT_GAMEPAD_B) bits |= PadB;
        if (g.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER) bits |= PadLb;
        if (g.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) bits |= PadRb;
    }
    using PadState = uint64_t (*)(int16_t*, int*);
    static PadState game_pad = reinterpret_cast<PadState>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(nullptr), "iss_pad_dialog_state")));
    if (game_pad) {
        int16_t axes[2] = {};
        int playstation = 0;
        const uint64_t b = game_pad(axes, &playstation);
        g_playstation = playstation != 0;
        auto on = [&](int button) { return ((b >> button) & 1) != 0; };
        if (on(SDL_GAMEPAD_BUTTON_DPAD_UP) || axes[1] < -dead) bits |= PadUp;
        if (on(SDL_GAMEPAD_BUTTON_DPAD_DOWN) || axes[1] > dead) bits |= PadDown;
        if (on(SDL_GAMEPAD_BUTTON_DPAD_LEFT) || axes[0] < -dead) bits |= PadLeft;
        if (on(SDL_GAMEPAD_BUTTON_DPAD_RIGHT) || axes[0] > dead) bits |= PadRight;
        if (on(SDL_GAMEPAD_BUTTON_SOUTH)) bits |= PadA;
        if (on(SDL_GAMEPAD_BUTTON_EAST)) bits |= PadB;
        if (on(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER)) bits |= PadLb;
        if (on(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)) bits |= PadRb;
    }
    return bits;
}

// Presses become inputs; up and down repeat while held (after 0.4 s, every 0.11 s).
void PollPads(State& s) {
    static unsigned previous = ~0u; // the buttons held when the dialog opened do not count
    static double repeat_at = 0;
    const unsigned bits = ReadPads();
    if (s.appear == 0) previous = bits;
    const unsigned pressed = bits & ~previous;
    previous = bits;
    const double now = Now();
    auto fire = [&](unsigned bit, Input in) {
        if (pressed & bit) {
            s.inputs.push_back(in);
            s.pad_used = true;
            s.pad_press_at = Now();
            s.anchor_x = s.pointer.x;
            s.anchor_y = s.pointer.y;
        }
    };
    fire(PadUp, Input::Up);
    fire(PadDown, Input::Down);
    fire(PadLeft, Input::Left);
    fire(PadRight, Input::Right);
    fire(PadA, Input::Accept);
    fire(PadB, Input::Back);
    fire(PadLb, Input::PageUp);
    fire(PadRb, Input::PageDown);
    const unsigned vertical = bits & (PadUp | PadDown);
    if (pressed & (PadUp | PadDown)) repeat_at = now + 0.4;
    else if (vertical && now >= repeat_at) {
        s.inputs.push_back(vertical & PadUp ? Input::Up : Input::Down);
        repeat_at = now + 0.11;
    }
}

} // namespace

// Fonts: Segoe UI on Windows; under Proton, ISS_UI_FONT (the launcher's font), else ImGui's.
void LoadFonts(ImGuiIO& io) {
    static const ImWchar ranges[] = {0x0020, 0x00FF, 0x2010, 0x2027, 0x2122, 0x2122, 0x21B5, 0x21B5, 0};
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    auto load = [&](const char* path) -> ImFont* {
        if (!path || !*path) return nullptr;
        if (FILE* f = std::fopen(path, "rb")) std::fclose(f);
        else return nullptr;
        return io.Fonts->AddFontFromFileTTF(path, 48.0f, &cfg, ranges);
    };
    char windir[260] = "C:\\Windows";
    if (const char* w = std::getenv("WINDIR")) std::snprintf(windir, sizeof(windir), "%s", w);
    const std::string segoe = std::string(windir) + "\\Fonts\\segoeui.ttf";
    const std::string segoe_bold = std::string(windir) + "\\Fonts\\seguisb.ttf";
    g_regular = load(segoe.c_str());
    g_bold = load(segoe_bold.c_str());
    if (!g_regular) g_regular = load(std::getenv("ISS_UI_FONT"));
    if (!g_bold) g_bold = g_regular;
    if (!g_regular) g_regular = g_bold = io.Fonts->Fonts.empty() ? io.Fonts->AddFontDefault() : io.Fonts->Fonts[0];
}

bool Active() { return g_active.load(); }

// SDL events of the game window (window thread): taken while the dialog is on screen.
bool OnEvent(const SDL_Event& e) {
    if (!g_active.load()) return false;
    State& s = S();
    std::lock_guard lock{s.mutex};
    switch (e.type) {
    case SDL_EVENT_KEY_DOWN:
        s.pad_used = false;
        switch (e.key.key) {
        case SDLK_UP: s.inputs.push_back(Input::Up); break;
        case SDLK_DOWN: s.inputs.push_back(Input::Down); break;
        case SDLK_LEFT: s.inputs.push_back(Input::Left); break;
        case SDLK_RIGHT: s.inputs.push_back(Input::Right); break;
        case SDLK_PAGEUP: s.inputs.push_back(Input::PageUp); break;
        case SDLK_PAGEDOWN: s.inputs.push_back(Input::PageDown); break;
        case SDLK_RETURN: case SDLK_KP_ENTER: case SDLK_SPACE: s.inputs.push_back(Input::Accept); break;
        case SDLK_ESCAPE: case SDLK_BACKSPACE: s.inputs.push_back(Input::Back); break;
        default: break;
        }
        return true;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        s.pad_used = true;
        s.pad_press_at = Now();
        s.anchor_x = s.pointer.x;
        s.anchor_y = s.pointer.y;
        switch (e.gbutton.button) {
        case SDL_GAMEPAD_BUTTON_DPAD_UP: s.inputs.push_back(Input::Up); s.dpad_up = true; s.held_dir = -1; s.held_next = Now() + 0.4; break;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: s.inputs.push_back(Input::Down); s.dpad_down = true; s.held_dir = 1; s.held_next = Now() + 0.4; break;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: s.inputs.push_back(Input::Left); break;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: s.inputs.push_back(Input::Right); break;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: s.inputs.push_back(Input::PageUp); break;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: s.inputs.push_back(Input::PageDown); break;
        case SDL_GAMEPAD_BUTTON_SOUTH: s.inputs.push_back(Input::Accept); break;
        case SDL_GAMEPAD_BUTTON_EAST: s.inputs.push_back(Input::Back); break;
        default: break;
        }
        return true;
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_DPAD_UP) s.dpad_up = false;
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_DPAD_DOWN) s.dpad_down = false;
        if (!s.dpad_up && !s.dpad_down && std::fabs(s.stick_y) < 0.5f) s.held_dir = 0;
        return true;
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) {
            const float v = e.gaxis.value / 32767.0f;
            const int dir = v < -0.5f ? -1 : v > 0.5f ? 1 : 0;
            const int was = std::fabs(s.stick_y) > 0.5f ? (s.stick_y < 0 ? -1 : 1) : 0;
            s.stick_y = v;
            if (dir != 0 && dir != was) {
                s.pad_used = true;
                s.inputs.push_back(dir < 0 ? Input::Up : Input::Down);
                s.held_dir = dir;
                s.held_next = Now() + 0.4;
            } else if (dir == 0 && !s.dpad_up && !s.dpad_down) {
                s.held_dir = 0;
            }
        } else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX) {
            static int was_x = 0;
            const float v = e.gaxis.value / 32767.0f;
            const int dir = v < -0.5f ? -1 : v > 0.5f ? 1 : 0;
            if (dir != was_x && dir != 0) s.inputs.push_back(dir < 0 ? Input::Left : Input::Right);
            was_x = dir;
        }
        return true;
    case SDL_EVENT_MOUSE_MOTION: {
        float sx = 1, sy = 1;
        if (SDL_Window* w = SDL_GetWindowFromID(e.motion.windowID)) {
            int pw, ph, ww, wh;
            SDL_GetWindowSizeInPixels(w, &pw, &ph);
            SDL_GetWindowSize(w, &ww, &wh);
            if (ww > 0 && wh > 0) {
                sx = float(pw) / ww;
                sy = float(ph) / wh;
            }
        }
        s.pointer.x = e.motion.x * sx;
        s.pointer.y = e.motion.y * sy;
        if (s.anchor_x < 0) { s.anchor_x = s.pointer.x; s.anchor_y = s.pointer.y; }
        const float mx = s.pointer.x - s.anchor_x, my = s.pointer.y - s.anchor_y;
        if (s.pad_used && Now() - s.pad_press_at > 0.75 && mx * mx + my * my >= 36.0f) {
            s.pad_used = false;
            s.hover = -1; // the row or button now under the cursor takes the selection
        }
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (e.button.button == SDL_BUTTON_LEFT) s.pointer.clicked = true;
        if (e.button.button == SDL_BUTTON_RIGHT) s.inputs.push_back(Input::Back);
        return true;
    case SDL_EVENT_MOUSE_WHEEL:
        s.inputs.push_back(e.wheel.y > 0 ? Input::Up : Input::Down);
        return true;
    default: return false;
    }
}

// Every presented frame (render thread), before ImGui::Render.
void Draw(float width, float height, float dt) {
    State& s = S();
    std::lock_guard lock{s.mutex};
    if (!s.open) return;
    PollPads(s);
    while (!s.inputs.empty() && s.open) {
        const Input in = s.inputs.front();
        s.inputs.pop_front();
        Apply(s, in);
    }
    if (!s.open) return;
    s.appear = std::min(1.0f, s.appear + dt / 0.18f);
    const float t = 1.0f - (1.0f - s.appear) * (1.0f - s.appear); // ease out
    const float k = height / 1080.0f;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(width, height), Col(0, 0, 0, 0.62f * t));

    const bool list = IsList(s.dialog);
    const float pw = list ? std::min(width * 0.66f, 1240 * k) : std::min(width * 0.5f, 860 * k);
    const float ph = list ? height * 0.80f : (s.dialog.mode == IssDialogProgress || s.dialog.buttons == IssDialogNone ? 300 * k : 380 * k);
    const float grow = 0.96f + 0.04f * t;
    const ImVec2 c(width / 2, height / 2 + (1 - t) * 18 * k);
    const ImVec2 a(c.x - pw * grow / 2, c.y - ph * grow / 2), b(c.x + pw * grow / 2, c.y + ph * grow / 2);
    // Panel: a deep blue gradient, a soft shadow and a hairline border.
    for (int i = 4; i >= 1; --i) {
        const float e = 6.0f * i * k;
        dl->AddRectFilled(ImVec2(a.x - e, a.y - e + 8 * k), ImVec2(b.x + e, b.y + e + 8 * k), Col(0, 0, 0, 0.08f * t), 14 * k + e);
    }
    dl->AddRectFilledMultiColor(a, b, Col(20, 33, 52, 0.97f * t), Col(16, 27, 44, 0.97f * t), Col(8, 13, 24, 0.97f * t),
                                Col(10, 16, 28, 0.97f * t));
    dl->AddRect(a, b, Col(255, 255, 255, 0.14f * t), 12 * k, 0, 1.2f * k);

    if (list) DrawList(dl, s, a, b, k, t, dt);
    else DrawMessage(dl, s, a, b, k, t);
    if (!s.open) return;
    DrawFooter(dl, s, a, b, k, t);
    s.pointer.clicked = false;
}

} // namespace iss::save_ui

extern "C" __declspec(dllexport) int iss_gpu_dialog_open(const IssDialog* d) {
    if (!d) return -1;
    auto& s = iss::save_ui::S();
    std::lock_guard lock{s.mutex};
    s.dialog = *d;
    s.dialog.count = std::min<uint32_t>(s.dialog.count, ISS_DIALOG_MAX_ITEMS);
    s.icons.clear();
    for (uint32_t i = 0; i < s.dialog.count; ++i) {
        const IssDialogItem& it = s.dialog.items[i];
        if (it.icon_png && it.icon_png_size) {
            const auto* p = static_cast<const uint8_t*>(it.icon_png);
            s.icons.push_back(ImGui::RefCountedTexture::DecodePngTexture({p, p + it.icon_png_size}));
        } else if (it.icon_path[0]) {
            std::FILE* f = std::fopen(it.icon_path, "rb");
            if (f) {
                std::fclose(f);
                s.icons.push_back(ImGui::RefCountedTexture::DecodePngFile(it.icon_path));
            } else {
                s.icons.emplace_back();
            }
        } else {
            s.icons.emplace_back();
        }
        s.dialog.items[i].icon_png = nullptr; // not kept (the game's buffer)
    }
    const int n = iss::save_ui::Choices(s.dialog);
    s.focus = n ? std::clamp(int(d->focus), 0, n - 1) : 0;
    if (!iss::save_ui::IsList(s.dialog) && d->buttons == IssDialogYesNo && d->default_no) s.focus = 1;
    s.focus_y = s.focus * 126.0f;
    s.scroll = 0;
    s.appear = 0;
    s.opened_at = iss::save_ui::Now();
    s.inputs.clear();
    s.pointer = {};
    s.hover = -1;
    s.anchor_x = s.anchor_y = -1;
    s.pad_press_at = -1;
    s.held_dir = 0;
    s.finished = false;
    s.open = true;
    iss::save_ui::g_active = true;
    std::fprintf(stderr, "SAVE_DIALOG_UI open mode=%u type=%u items=%u\n", d->mode, d->type, s.dialog.count);
    return 0;
}

extern "C" __declspec(dllexport) int iss_gpu_dialog_poll(IssDialogResult* out) {
    auto& s = iss::save_ui::S();
    std::lock_guard lock{s.mutex};
    if (s.finished) {
        if (out) *out = s.result;
        s.finished = false;
        return 1;
    }
    return s.open ? 0 : -1;
}

extern "C" __declspec(dllexport) void iss_gpu_dialog_close(void) {
    auto& s = iss::save_ui::S();
    std::lock_guard lock{s.mutex};
    s.open = false;
    s.finished = false;
    iss::save_ui::g_active = false; // the icons stay until the next open (a frame may still use them)
}

extern "C" __declspec(dllexport) int iss_gpu_dialog_active(void) { return iss::save_ui::Active() ? 1 : 0; }
