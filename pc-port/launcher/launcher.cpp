// SPDX-License-Identifier: GPL-2.0-or-later
// inFAMOUS Second Son PC port launcher (2026-10-18). A portable folder:
//   <root>\inFAMOUS Second Son Launcher.exe   this program
//   <root>\bin\                                 iss-probe.exe, iss-gpu.dll, SDL3.dll
//   <root>\shaders\                             optional: library-v2\ (pipeline library), seed\
//   <root>\user\                                launcher.ini, prepared\, saves\, shader-cache\, logs\
// The game itself is never shipped: on first launch the user picks their own inFAMOUS Second Son
// (CUSA00004 or CUSA00223) folder. The window art comes from that folder too: sce_sys\pic1.png (the
// crow, as on the loading screen) behind the menu, darkened on the left. The logo is <root>\assets\logo.png
// when present, else it is cut out of sce_sys\icon0.png (its black background made transparent).
// PLAY starts bin\iss-probe.exe with the environment of the tested build (play_porte.py, porte-x)
// plus the Settings, hides this window, and comes back when the game exits. The accumulated shader
// cache follows play_v1_guard.py: each session plays on a copy, promoted after a clean session.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <objidl.h>
#include <gdiplus.h>
#include <xinput.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace Gdiplus;

namespace iss_prepare {
std::wstring PrepareGame(const fs::path& game, const fs::path& out_file,
                         const std::function<void(float, const wchar_t*)>& progress);
}

namespace {

// ---------------------------------------------------------------------------------------------
// Paths and settings
// ---------------------------------------------------------------------------------------------
fs::path g_root, g_bin, g_user;

struct Choice {
    const wchar_t* label;
    const char* value;
};
struct Setting {
    const char* key;
    const wchar_t* label;
    const wchar_t* hint;
    std::vector<Choice> choices;
    int tab;
    int index = 0;
};

enum Tab { TabDisplay, TabGraphics, TabGame, TabAdvanced, TabCount };
const wchar_t* kTabNames[TabCount] = {L"DISPLAY", L"GRAPHICS", L"GAME", L"ADVANCED"};

std::vector<Setting> g_settings = {
    {"window_mode", L"Window mode", L"Alt+Enter also switches in game.",
     {{L"Windowed", "0"}, {L"Fullscreen", "1"}}, TabDisplay},
    {"vsync", L"VSync", L"Off: no tearing protection, lowest latency.",
     {{L"On", "Fifo"}, {L"Off", "Immediate"}}, TabDisplay},
    {"fps_limit", L"Frame rate limit", L"Above your monitor's refresh rate it needs VSync Off.",
     {{L"Unlimited", "0"}, {L"30", "30"}, {L"40", "40"}, {L"50", "50"}, {L"60", "60"},
      {L"75", "75"}, {L"90", "90"}, {L"100", "100"}, {L"120", "120"}, {L"144", "144"},
      {L"165", "165"}, {L"240", "240"}},
     TabDisplay},
    {"fsr", L"Upscaling (FSR 1)", L"Upscales the picture on windows larger than it.",
     {{L"Off", "0"}, {L"On", "1"}}, TabDisplay},
    // The picture size, field of view and HUD edges (native/ultrawide.cpp: ISS_RESOLUTION,
    // ISS_FOV, ISS_HUD_ASPECT). Wider than 16:9 keeps the vertical view and widens it.
    {"resolution", L"Resolution", L"Wider than 16:9 shows more to the sides. Desktop: your monitor's.",
     {{L"1920x1080 (16:9)", "1920x1080"}, {L"2560x1440 (16:9)", "2560x1440"},
      {L"3840x2160 (16:9)", "3840x2160"}, {L"2560x1080 (21:9)", "2560x1080"},
      {L"3440x1440 (21:9)", "3440x1440"}, {L"3840x1600 (24:10)", "3840x1600"},
      {L"3840x1080 (32:9)", "3840x1080"}, {L"5120x1440 (32:9)", "5120x1440"}, {L"Desktop", "desktop"}},
     TabGraphics},
    {"fov", L"Field of view", L"100%: the game's own. Higher shows more around the camera.",
     {{L"100%", "100"}, {L"105%", "105"}, {L"110%", "110"}, {L"115%", "115"}, {L"120%", "120"},
      {L"130%", "130"}, {L"90%", "90"}},
     TabGraphics},
    {"hud_width", L"HUD width", L"Where the HUD edges sit on screens wider than it.",
     {{L"16:9", "1.7778"}, {L"21:9", "2.3704"}, {L"Full width", "0"}}, TabGraphics},
    {"shader_mode", L"Shader compilation", L"Wait: no missing objects. Skip: smoother, brief pop-in.",
     {{L"Wait", "0"}, {L"Skip", "1"}}, TabGraphics},
    {"blackbox", L"GPU crash diagnostics", L"Writes a report if the GPU hangs.",
     {{L"On", "1"}, {L"Off", "0"}}, TabAdvanced},
    // ISS_UNIFIED_MEMORY: the GPU reads the game's memory in place (1) or from copies in video
    // memory (0). On an RTX 5070 Ti the in-place reads went over PCIe and the tiled lighting
    // shader (cs ce1675e0) took 33-50 ms instead of 0.9: 22 FPS instead of 117 at 1080p.
    {"gpu_memory", L"GPU memory", L"Auto: copies in video memory on NVIDIA, in place on others.",
     {{L"Auto", "auto"}, {L"In place", "1"}, {L"Video memory", "0"}}, TabGraphics},
    // Its hint and choices are those of the game folder's edition (kEditions, ApplyEdition).
    {"language", L"Language", L"", {}, TabGame},
    // ISS_VOLUME (src/audio.c): the game's sound level, 100% its own; above it the peaks are
    // softened instead of clipped. Index 10 = 100%.
    {"volume", L"Volume", L"100%: the game's own level. Up to 200% for a quiet mix.",
     {{L"50%", "50"}, {L"55%", "55"}, {L"60%", "60"}, {L"65%", "65"}, {L"70%", "70"},
      {L"75%", "75"}, {L"80%", "80"}, {L"85%", "85"}, {L"90%", "90"}, {L"95%", "95"},
      {L"100%", "100"}, {L"105%", "105"}, {L"110%", "110"}, {L"115%", "115"}, {L"120%", "120"},
      {L"125%", "125"}, {L"130%", "130"}, {L"135%", "135"}, {L"140%", "140"}, {L"145%", "145"},
      {L"150%", "150"}, {L"155%", "155"}, {L"160%", "160"}, {L"165%", "165"}, {L"170%", "170"},
      {L"175%", "175"}, {L"180%", "180"}, {L"185%", "185"}, {L"190%", "190"}, {L"195%", "195"},
      {L"200%", "200"}},
     TabGame, 10},
};

// The editions the port runs. Their eboot.bin and sce_module are byte identical (the SHA-256
// prepare_game.cpp checks); only the language packages (art\cache\lang_*) differ. The languages
// are PS4 system language ids (src/services.c ISS_LANGUAGE), which the game uses as the index of
// its own lang_%s_%s%s.xpp name table (english=1, french=2, ..., brazilian=17, latino=20).
struct Edition {
    const char* title_id;
    const wchar_t* language_hint;
    std::vector<Choice> languages;
};
const Edition kEditions[] = {
    // Europe: ten with voices, the four Nordic ones subtitles only (English voices).
    {"CUSA00004", L"Text and voices. Nordic languages: subtitles only.",
     {{L"English", "1"}, {L"French", "2"}, {L"Italian", "5"}, {L"German", "4"}, {L"Spanish", "3"},
      {L"Dutch", "6"}, {L"Portuguese", "7"}, {L"Russian", "8"}, {L"Polish", "16"}, {L"Turkish", "19"},
      {L"Danish", "14"}, {L"Finnish", "12"}, {L"Norwegian", "15"}, {L"Swedish", "13"}}},
    // North America: four, all with voices.
    {"CUSA00223", L"Text and voices.",
     {{L"English", "1"}, {L"French", "2"}, {L"Spanish (Latin America)", "20"},
      {L"Portuguese (Brazil)", "17"}}},
};
void ApplyEdition(const std::string& language);
std::wstring g_game_dir;
std::map<std::string, std::string> g_extra; // other keys of launcher.ini (dev overrides)

// The controller layout (CONTROLLERS): what each button of a controller without a touchpad does
// in game (src/pad.c, ISS_PAD_MAP; the keys are pad.c's names). PlayStation controllers keep
// their own layout.
struct PadName {
    const char* key;
    const wchar_t* label;
};
const PadName kPadSources[] = {
    {"a", L"A"}, {"b", L"B"}, {"x", L"X"}, {"y", L"Y"}, {"lb", L"LB"}, {"rb", L"RB"}, {"lt", L"LT"},
    {"rt", L"RT"}, {"ls", L"Left stick press"}, {"rs", L"Right stick press"}, {"view", L"View"},
    {"menu", L"Menu"}, {"guide", L"Xbox button"}, {"share", L"Share"}, {"up", L"D-pad up"},
    {"down", L"D-pad down"}, {"left", L"D-pad left"}, {"right", L"D-pad right"}};
const PadName kPadTargets[] = {
    {"none", L"Nothing"}, {"cross", L"Cross"}, {"circle", L"Circle"}, {"square", L"Square"},
    {"triangle", L"Triangle"}, {"l1", L"L1"}, {"r1", L"R1"}, {"l2", L"L2"}, {"r2", L"R2"},
    {"l3", L"L3"}, {"r3", L"R3"}, {"options", L"Options"}, {"ps", L"PS button"},
    {"up", L"D-pad up"}, {"down", L"D-pad down"}, {"left", L"D-pad left"}, {"right", L"D-pad right"},
    {"touch", L"Touchpad press"}, {"touch_left", L"Touchpad left"}, {"touch_right", L"Touchpad right"},
    {"swipe_up", L"Swipe up"}, {"swipe_down", L"Swipe down"}, {"swipe_left", L"Swipe left"},
    {"swipe_right", L"Swipe right"}};
constexpr int kPadSourceCount = int(std::size(kPadSources)), kPadTargetCount = int(std::size(kPadTargets));
// Recommended: the PS4 layout one to one; View / Menu press the left / right side of the touchpad.
const int kRecommendedPadMap[kPadSourceCount] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19, 12, 0, 13, 14, 15, 16};
int g_pad_map[kPadSourceCount];

bool PadMapRecommended() { return std::equal(std::begin(g_pad_map), std::end(g_pad_map), std::begin(kRecommendedPadMap)); }

void ReadPadMap() {
    std::copy(std::begin(kRecommendedPadMap), std::end(kRecommendedPadMap), std::begin(g_pad_map));
    const auto it = g_extra.find("pad_map");
    if (it == g_extra.end()) return;
    size_t at = 0;
    const std::string& s = it->second;
    while (at < s.size()) {
        const size_t end = std::min(s.find(',', at), s.size()), eq = s.find('=', at);
        if (eq < end) {
            const std::string key = s.substr(at, eq - at), value = s.substr(eq + 1, end - eq - 1);
            for (int i = 0; i < kPadSourceCount; ++i)
                if (key == kPadSources[i].key)
                    for (int t = 0; t < kPadTargetCount; ++t)
                        if (value == kPadTargets[t].key) g_pad_map[i] = t;
        }
        at = end + 1;
    }
}

// The layout for launcher.ini and ISS_PAD_MAP; empty when it is the recommended one.
std::string PadMapString() {
    if (PadMapRecommended()) return {};
    std::string s;
    for (int i = 0; i < kPadSourceCount; ++i)
        s += std::string(i ? "," : "") + kPadSources[i].key + "=" + kPadTargets[g_pad_map[i]].key;
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}
std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

fs::path ConfigPath() { return g_user / "launcher.ini"; }

void LoadConfig() {
    std::ifstream f(ConfigPath());
    std::string line, language;
    while (std::getline(f, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (key == "game_dir") {
            g_game_dir = Widen(value);
            continue;
        }
        if (key == "language") {
            // Its choices depend on the game folder, matched once all of the file is read.
            language = value;
            continue;
        }
        bool known = false;
        for (auto& s : g_settings) {
            if (key != s.key) continue;
            known = true;
            for (size_t i = 0; i < s.choices.size(); ++i)
                if (value == s.choices[i].value) s.index = int(i);
        }
        if (!known) g_extra[key] = value;
    }
    ApplyEdition(language);
    ReadPadMap();
}

void SaveConfig() {
    fs::create_directories(g_user);
    std::ofstream f(ConfigPath(), std::ios::trunc);
    f << "game_dir=" << Narrow(g_game_dir) << "\n";
    for (const auto& s : g_settings) f << s.key << "=" << s.choices[s.index].value << "\n";
    for (const auto& [k, v] : g_extra) f << k << "=" << v << "\n";
}

const char* SettingValue(const char* key) {
    for (const auto& s : g_settings)
        if (!strcmp(s.key, key)) return s.choices[s.index].value;
    return "";
}

// ISS_RESOLUTION: the chosen WxH, or the primary monitor's current mode for "desktop".
std::string ResolutionValue() {
    const std::string value = SettingValue("resolution");
    if (value != "desktop") return value;
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (!EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) || !mode.dmPelsWidth || !mode.dmPelsHeight)
        return "1920x1080";
    return std::to_string(mode.dmPelsWidth) + "x" + std::to_string(mode.dmPelsHeight);
}

// ISS_UNIFIED_MEMORY: "0" with an NVIDIA display adapter in the system when set to Auto, else "1".
std::string UnifiedMemoryValue() {
    const std::string value = SettingValue("gpu_memory");
    if (value != "auto") return value;
    DISPLAY_DEVICEW device{};
    device.cb = sizeof(device);
    for (DWORD i = 0; EnumDisplayDevicesW(nullptr, i, &device, 0); ++i)
        if (wcsstr(device.DeviceString, L"NVIDIA")) return "0";
    return "1";
}

// An override from launcher.ini (dev and testing), else the default under the root.
fs::path PathOption(const char* key, const fs::path& fallback) {
    const auto it = g_extra.find(key);
    return it != g_extra.end() && !it->second.empty() ? fs::path(Widen(it->second)) : fallback;
}

// ---------------------------------------------------------------------------------------------
// The game folder
// ---------------------------------------------------------------------------------------------
bool FileContains(const fs::path& path, const char* needle) {
    std::ifstream f(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return data.find(needle) != std::string::npos;
}

// The edition of a game folder (its sce_sys\param.sfo title id), else null.
const Edition* GameEdition(const fs::path& dir) {
    for (const auto& e : kEditions)
        if (FileContains(dir / "sce_sys" / "param.sfo", e.title_id)) return &e;
    return nullptr;
}

// The language choices of the game folder's edition (the first one without a game folder),
// keeping language when that edition has it, else its first language (English).
void ApplyEdition(const std::string& language) {
    const Edition* e = g_game_dir.empty() ? nullptr : GameEdition(g_game_dir);
    if (!e) e = &kEditions[0];
    Setting& s = *std::find_if(g_settings.begin(), g_settings.end(),
                               [](const Setting& x) { return !strcmp(x.key, "language"); });
    s.hint = e->language_hint;
    s.choices = e->languages;
    s.index = 0;
    for (size_t i = 0; i < s.choices.size(); ++i)
        if (language == s.choices[i].value) s.index = int(i);
}

// Empty when the folder is an inFAMOUS Second Son game folder (an edition of kEditions), else the
// reason.
std::wstring CheckGameDir(const fs::path& dir) {
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return L"Folder not found.";
    if (!fs::is_regular_file(dir / "eboot.bin", ec)) return L"eboot.bin is missing in this folder.";
    if (!fs::is_regular_file(dir / "sce_sys" / "param.sfo", ec))
        return L"sce_sys\\param.sfo is missing in this folder.";
    if (!GameEdition(dir))
        return L"This is not inFAMOUS Second Son (CUSA00004 or CUSA00223).";
    if (!fs::is_regular_file(dir / "sce_module" / "libc.prx", ec))
        return L"sce_module\\libc.prx is missing in this folder.";
    return {};
}

std::wstring PickFolder(HWND owner) {
    std::wstring result;
    IFileOpenDialog* dialog = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dialog))))
        return result;
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    dialog->SetTitle(L"Select your inFAMOUS Second Son (CUSA00004 or CUSA00223) game folder");
    if (SUCCEEDED(dialog->Show(owner))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                result = path;
                CoTaskMemFree(path);
            }
            item->Release();
        }
    }
    dialog->Release();
    return result;
}

// ---------------------------------------------------------------------------------------------
// Launching the game
// ---------------------------------------------------------------------------------------------
// The environment of the tested build (play_porte.py on porte-vsync1, 2026-10-19), without the
// paths and diagnostics of the development runs. The game's compute shaders are compiled before
// the first session (ISS_AOT_COMPUTE, the loading screen's "Preparing shaders"), and the graphics
// pipelines listed by the shader recipes (shaders/recipes) are rebuilt from the user's own game
// files (ISS_AOT_RECIPES).
const char* const kGameEnvironment[][2] = {
    {"BB_VBLANK_HZ", "240"}, {"ISS_AOT_COMPUTE", "auto"}, {"ISS_AOT_INDEX", "auto"},
    {"ISS_AOT_RECIPES", "1"}, {"ISS_AOT_WAIT", "1"},
    {"ISS_ASYNC_COMPUTE", "0"}, {"ISS_BARRIER_POOL", "1"},
    {"ISS_CAPTURE_FRAMES", "0"}, {"ISS_CAPTURE_OVERLAY", "0"}, {"ISS_CLAMP_CACHE_FIRST", "1"},
    {"ISS_COMMAND_TRACE", "0"}, {"ISS_COMPUTE_FAIRNESS", "1"}, {"ISS_COMPUTE_PHASES", "0"},
    {"ISS_CPU_BUNDLE_INDEX_ONCE", "0"}, {"ISS_CPU_BUNDLE_SPV_VERIFIED", "0"}, {"ISS_CP_CPU", "14"},
    {"ISS_CP_SPLIT", "0"}, {"ISS_DEEP_PROFILE", "0"}, {"ISS_DESCRIPTOR_DELTAS", "0"},
    {"ISS_DESCRIPTOR_FAST_PATH", "1"}, {"ISS_DIAGNOSTIC_SESSION", "0"}, {"ISS_DRAW_PIPE", "0"},
    {"ISS_ENGINE_MARKERS", "0"}, {"ISS_ENGINE_STATS", "0"}, {"ISS_FAST_DRAW", "1"},
    {"ISS_FAST_DYN_EXT", "1"}, {"ISS_FAST_LEAN", "1"}, {"ISS_FAST_PROGRAM_ALIAS", "1"},
    {"ISS_FAST_REBIND", "1"}, {"ISS_FAST_STATE_SWITCH", "1"}, {"ISS_FAST_SWITCH", "1"},
    {"ISS_FAULT_WINDOW_KB", "64"}, {"ISS_FEEDBACK_ARENA", "0"}, {"ISS_FEEDBACK_EARLY_SUBMIT", "0"},
    {"ISS_FEEDBACK_IDENTITY", "0"}, {"ISS_FEEDBACK_JOIN", "0"}, {"ISS_FEEDBACK_PAGE_READBACK", "0"},
    {"ISS_FEEDBACK_TRACE", "0"}, {"ISS_G4_FEEDBACK_PROBE", "0"}, {"ISS_GPU", "1"},
    {"ISS_GPU_HAZARD_DIAG", "0"}, {"ISS_GPU_PROFILE", "0"}, {"ISS_HYBRID_BENCH_HIDDEN", "0"},
    {"ISS_HYBRID_FAMILY_PROFILE", "0"}, {"ISS_INDIRECT_GUARD", "0"}, {"ISS_LAZY_PIPELINES", "1"},
    {"ISS_MAPPING_CACHE", "1"}, {"ISS_MAPPING_INDEX", "1"}, {"ISS_MINIMAL_LOGS", "0"},
    {"ISS_NATIVE_CITY_FETCH", "0"}, {"ISS_NATIVE_CPU_DRAWS", "0"}, {"ISS_NATIVE_DEPTH_RUN", "0"},
    {"ISS_NATIVE_DRAW_STATE", "0"}, {"ISS_NATIVE_DRAW_STATE_VERIFY", "0"},
    {"ISS_NATIVE_FEEDBACK_PROBE", "0"}, {"ISS_NATIVE_GBUFFER_SETS", "1"},
    {"ISS_NATIVE_GEOMETRY", "0"}, {"ISS_NATIVE_GEOMETRY_STATS", "0"},
    {"ISS_NATIVE_INDEXED_BATCH", "0"}, {"ISS_NATIVE_INSTANCED_BATCH", "0"},
    {"ISS_NATIVE_ASC_ASYNC", "1"}, {"ISS_NATIVE_INSTANCED_FEATURE", "1"}, {"ISS_NATIVE_ITEMS", "0"}, {"ISS_NATIVE_ITEM_RANGES", "0"},
    {"ISS_NATIVE_MATERIALS", "1"}, {"ISS_NATIVE_MEMORY_REPORT", "0"}, {"ISS_NATIVE_PASSES", "1"},
    {"ISS_NATIVE_PASSES_ASYNC_SCAN", "1"}, {"ISS_NATIVE_PASSES_DYNAMIC", "1"},
    {"ISS_NATIVE_PASSES_SUBMIT", "1"}, {"ISS_NATIVE_READBACK_PAGES", "0"},
    {"ISS_NATIVE_REDUCTION", "0"}, {"ISS_NATIVE_SHADOW", "0"}, {"ISS_NATIVE_SHADOW_STATS", "0"},
    {"ISS_NATIVE_SHADOW_VERIFY", "0"}, {"ISS_NATIVE_VERIFY", "0"}, {"ISS_PARALLEL_RENDER", "1"},
    {"ISS_PIPELINE_EVENTS", "0"}, {"ISS_PIPELINE_STATE_REUSE", "0"},
    {"ISS_PIPELINE_STATE_REUSE_STATS", "0"}, {"ISS_PIPELINE_STATE_REUSE_VERIFY", "0"},
    {"ISS_PRELOAD_BACKGROUND", "1"}, {"ISS_PREWARM_LARGE_UPLOAD", "0"},
    {"ISS_PROBE_HOST_IMPORT", "0"}, {"ISS_PROFILE_CITY_DRAW_GPU", "0"}, {"ISS_READBACK_DIAG", "0"},
    {"ISS_REAL_OCCLUSION", "1"}, {"ISS_RENDER_REUSE", "1"}, {"ISS_RING_BARRIERS", "1"},
    {"ISS_SECONDARY_BATCH_PROBE", "0"}, {"ISS_SMALL_READ_MEMO", "0"}, {"ISS_SPEC_TRANSLATE", "1"},
    {"ISS_STAGE_MEMO", "1"}, {"ISS_STAGE_MEMO_LOOSE", "1"}, {"ISS_STAGE_MEMO_TEXEL", "1"},
    {"ISS_SUBMIT_EVERY", "1000"}, {"ISS_TEXTURE_LOOKUP_MEMO", "0"},
    {"ISS_TEXTURE_LOOKUP_MEMO_STATS", "0"}, {"ISS_THREAD_PRIORITY", "1"}, {"ISS_TIMELINE", "0"},
    {"ISS_TLS_PATCH", "1"}, {"ISS_TRACE_READBACK", "0"},
    {"ISS_TUNING", "async_written_uploads=1,scout=1,scout_fast=1,bind_prefetch=1"},
    {"ISS_UNIFIED_FLUSH_IRQ", "0"}, {"ISS_UNIFIED_FLUSH_US", "1000"}, {"ISS_UNIFIED_MEMORY", "1"},
    {"ISS_UPLOAD_WORKER", "1"}, {"ISS_VISUAL_SEQUENCE", "0"}, {"SHADPS4_FAST_STAGE_LOOKUP", "1"},
    {"SHADPS4_PARALLEL_PRELOAD", "1"}, {"SHADPS4_REUSE_DESCRIPTOR_STORAGE", "1"},
};

constexpr int MinPlayedFrames = 600; // a session that played (not only the preload and title)

struct Session {
    fs::path log_dir, cache_work;
    bool library = false;
};

// The accumulated shader cache, user\shader-cache (play_v1_guard.prepare_cache / finish_cache):
//   current\    the cache between sessions; current\.unverified while the last session that added
//               to it has not been followed by one that played
//   session\    current, renamed for the time the game runs (ISS_PIPELINE_CACHE_ROOT)
//   base\       the files the game rewrites (profile.bin, the index, the Vulkan pipeline cache),
//               copied at PLAY, and start.txt (when the session began)
//   rollback\   the base of the last session that added to the cache, until it is verified
// The game only adds blobs (.spv, .meta, .key: named by their hash, written once when new) and
// rewrites the base files, so a session is undone by deleting the files created after it began and
// putting the base back. Each session used to copy the whole cache: ~35 000 files after a few
// sessions, and the window stopped responding at PLAY (2026-10-19).
bool RewrittenCacheFile(const fs::path& p) {
    const auto e = p.extension();
    return e == ".bin" || e == ".idx" || e == ".vkcache";
}

// fn(path, creation time) for each file under dir (one FindFirstFileEx listing per folder: the
// 35 000 files in ~20 ms).
void ForEachCacheFile(const fs::path& dir, const std::function<void(const fs::path&, ULONGLONG)>& fn) {
    WIN32_FIND_DATAW d;
    HANDLE h = FindFirstFileExW((dir / L"*").c_str(), FindExInfoBasic, &d, FindExSearchNameMatch, nullptr,
                                FIND_FIRST_EX_LARGE_FETCH);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        const std::wstring name = d.cFileName;
        if (name == L"." || name == L"..") continue;
        if (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ForEachCacheFile(dir / name, fn);
        else fn(dir / name, (ULONGLONG(d.ftCreationTime.dwHighDateTime) << 32) | d.ftCreationTime.dwLowDateTime);
    } while (FindNextFileW(h, &d));
    FindClose(h);
}

// Copies the files the game rewrites, and the time the session begins, into backup.
void SaveBase(const fs::path& cache, const fs::path& backup) {
    std::error_code ec;
    fs::remove_all(backup, ec);
    fs::create_directories(backup / "files", ec);
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    ForEachCacheFile(cache, [&](const fs::path& p, ULONGLONG) {
        if (!RewrittenCacheFile(p)) return;
        const fs::path dst = backup / "files" / p.lexically_relative(cache);
        fs::create_directories(dst.parent_path(), ec);
        fs::copy_file(p, dst, fs::copy_options::overwrite_existing, ec);
    });
    std::ofstream(backup / "start.txt") << ((ULONGLONG(now.dwHighDateTime) << 32) | now.dwLowDateTime);
}

ULONGLONG BaseStart(const fs::path& backup) {
    ULONGLONG start = 0;
    std::ifstream(backup / "start.txt") >> start;
    return start;
}

// The blobs the session of a base added.
size_t CountAdded(const fs::path& cache, const fs::path& backup) {
    const ULONGLONG start = BaseStart(backup);
    size_t n = 0;
    if (start)
        ForEachCacheFile(cache, [&](const fs::path& p, ULONGLONG created) {
            if (created >= start && !RewrittenCacheFile(p) && p.filename() != ".unverified") ++n;
        });
    return n;
}

// Undoes the session of a base: the files created since it began go, the base files come back.
void Rollback(const fs::path& cache, const fs::path& backup) {
    const ULONGLONG start = BaseStart(backup);
    if (!start) return;
    std::error_code ec;
    std::vector<fs::path> created;
    ForEachCacheFile(cache, [&](const fs::path& p, ULONGLONG t) {
        if (t >= start && p.filename() != ".unverified") created.push_back(p);
    });
    for (const auto& p : created) fs::remove(p, ec);
    const fs::path files = backup / "files";
    ForEachCacheFile(files, [&](const fs::path& p, ULONGLONG) {
        const fs::path dst = cache / p.lexically_relative(files);
        fs::create_directories(dst.parent_path(), ec);
        fs::copy_file(p, dst, fs::copy_options::overwrite_existing, ec);
    });
}

// Folders left by older launchers or unfinished cleanups, deleted by the session's watcher.
void RemoveOldCaches(const fs::path& lineage) {
    std::error_code ec;
    for (auto& e : fs::directory_iterator(lineage, ec))
        if (e.path().filename().wstring().rfind(L"old-", 0) == 0) fs::remove_all(e.path(), ec);
}

// PLAY: current becomes the session's cache (renamed, so it takes no time however big it is).
fs::path PrepareCache(bool library, const std::function<void(float)>& progress = {}) {
    const fs::path lineage = g_user / "shader-cache", current = lineage / "current", work = lineage / "session";
    std::error_code ec;
    const std::string tag = std::to_string(GetTickCount64());
    if (fs::exists(work, ec)) {
        if (!fs::exists(current, ec)) {
            // A session nobody finished (the PC went off while playing): undone, back to current.
            Rollback(work, lineage / "base");
            fs::rename(work, current, ec);
        } else {
            fs::rename(work, lineage / ("old-session-" + tag), ec); // an older launcher's full copy
        }
    }
    // An older launcher's full copy of the cache before its last update (now: rollback\).
    if (fs::exists(lineage / "previous", ec)) fs::rename(lineage / "previous", lineage / ("old-previous-" + tag), ec);
    const fs::path seed = PathOption("shader_seed", g_root / "shaders" / "seed");
    if (!fs::is_regular_file(current / "CUSA00004" / "profile.bin", ec)) {
        fs::remove_all(current, ec);
        fs::create_directories(current / "CUSA00004", ec);
        if (fs::is_regular_file(seed / "CUSA00004" / "profile.bin", ec)) {
            if (library)
                fs::copy_file(seed / "CUSA00004" / "profile.bin", current / "CUSA00004" / "profile.bin",
                              fs::copy_options::overwrite_existing, ec);
            else
                fs::copy(seed, current, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        }
    }
    SaveBase(current, lineage / "base");
    if (progress) progress(0.5f);
    fs::rename(current, work, ec);
    if (progress) progress(1.0f);
    return ec ? current : work; // something holds the folder open: the session uses current itself
}

std::string ReadText(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// After the game: keeps what a clean session added (to be verified by the next one), undoes a
// session that did not end cleanly, and undoes an unverified addition when the session after it
// ended before playing (play_v1_guard.finish_cache). Returns a line for the status bar.
std::wstring FinishCache(const Session& s) {
    const fs::path lineage = g_user / "shader-cache", current = lineage / "current", work = s.cache_work,
                   base = lineage / "base", rollback = lineage / "rollback", marker = work / ".unverified";
    std::error_code ec;
    const auto back = [&] {
        if (work != current) fs::rename(work, current, ec);
        RemoveOldCaches(lineage);
    };
    int frames = 0;
    {
        std::ifstream f(s.log_dir / "frames.csv");
        std::string line;
        while (std::getline(f, line)) ++frames;
        frames = std::max(0, frames - 1);
    }
    const std::string err = ReadText(s.log_dir / "stderr.log");
    const bool crashed = err.find("GPU_CRASH") != std::string::npos ||
                         fs::exists(s.log_dir / "gpu-crash.txt", ec);
    const bool played = frames >= MinPlayedFrames;
    if (fs::exists(marker, ec)) {
        if (played) {
            fs::remove(marker, ec);
            fs::remove_all(rollback, ec);
        } else if (fs::is_directory(rollback, ec)) {
            Rollback(work, base);
            Rollback(work, rollback);
            fs::remove(marker, ec);
            fs::remove_all(rollback, ec);
            back();
            return L"The new shader cache was rolled back (the session ended before playing).";
        }
    }
    const bool clean = played && !crashed && err.find("PORT_STOP") != std::string::npos;
    const size_t added = clean ? CountAdded(work, base) : 0;
    if (!clean) Rollback(work, base);
    if (!clean || !added) {
        back();
        return L"";
    }
    fs::remove_all(rollback, ec);
    fs::rename(base, rollback, ec);
    std::ofstream(marker) << Narrow(s.log_dir.wstring());
    back();
    return L"Shader cache updated: " + std::to_wstring(added) + L" new shaders.";
}
std::wstring Timestamp() {
    SYSTEMTIME t;
    GetLocalTime(&t);
    wchar_t buf[32];
    swprintf(buf, 32, L"%04u%02u%02u-%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute,
             t.wSecond);
    return buf;
}

void KeepNewest(const fs::path& dir, size_t keep) {
    std::error_code ec;
    std::vector<fs::path> entries;
    for (auto& e : fs::directory_iterator(dir, ec))
        if (e.is_directory(ec)) entries.push_back(e.path());
    std::sort(entries.begin(), entries.end());
    while (entries.size() > keep) {
        fs::remove_all(entries.front(), ec);
        entries.erase(entries.begin());
    }
}

fs::path PreparedImage() {
    return PathOption("prepared_image", g_user / "prepared" / "boot-linked.bin");
}

// The prepared image exists and was made from the current game folder (or is a dev override).
bool PreparedReady() {
    std::error_code ec;
    const fs::path image = PreparedImage();
    if (!fs::is_regular_file(image, ec)) return false;
    if (g_extra.count("prepared_image")) return true;
    std::ifstream f(image.wstring() + L".source");
    std::string source;
    std::getline(f, source);
    return fs::equivalent(fs::path(Widen(source)), fs::path(g_game_dir), ec);
}

// DLC: the user's .pkg files of the game (entitlement-only packages). Their content id is at 0x40
// of the package header ("\x7fCNT"): EP9000-CUSA00004_00-<16 character label> (UP9000-CUSA00223_00-
// for the North American edition; the game asks for the label only).
std::vector<std::string> ScanDlcLabels(const fs::path& dir) {
    std::vector<std::string> labels;
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return labels;
    for (auto it = fs::recursive_directory_iterator(dir, ec); it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        auto ext = it->path().extension().wstring();
        for (auto& c : ext) c = wchar_t(towlower(c));
        if (ext != L".pkg") continue;
        std::ifstream f(it->path(), std::ios::binary);
        char header[0x64] = {};
        f.read(header, sizeof(header));
        if (memcmp(header, "\x7f" "CNT", 4) != 0) continue;
        const std::string id(header + 0x40, 36);
        if (std::none_of(std::begin(kEditions), std::end(kEditions),
                         [&](const Edition& e) { return id.find(e.title_id) != std::string::npos; }))
            continue;
        const std::string label = id.substr(20, 16);
        if (std::find(labels.begin(), labels.end(), label) == labels.end()) labels.push_back(label);
    }
    return labels;
}

fs::path DlcDir() { return PathOption("dlc_dir", {}); }

// The game window's icon: the crow emblem of the game's own sce_sys\pic1.png (the launcher's
// background), a square around it, brightened to read on a dark taskbar. Written as a BMP for the
// game (ISS_WINDOW_ICON, SDL_LoadBMP); empty when it cannot be made.
fs::path MakeGameIcon() {
    const fs::path out = g_user / "game-icon.bmp";
    Gdiplus::Bitmap pic((fs::path(g_game_dir) / "sce_sys" / "pic1.png").c_str());
    if (pic.GetLastStatus() != Gdiplus::Ok) return {};
    const float w = float(pic.GetWidth()), h = float(pic.GetHeight()), side = h * 0.9f;
    const Gdiplus::RectF src(w * 0.5f - side / 2, h * 0.47f - side / 2, side, side);
    Gdiplus::Bitmap icon(256, 256, PixelFormat24bppRGB);
    {
        Gdiplus::Graphics g(&icon);
        g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
        g.Clear(Gdiplus::Color(255, 0, 0, 0));
        const float k = 2.4f; // the emblem is dark grey on black
        Gdiplus::ColorMatrix cm = {{{k, 0, 0, 0, 0}, {0, k, 0, 0, 0}, {0, 0, k, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}}};
        Gdiplus::ImageAttributes attr;
        attr.SetColorMatrix(&cm);
        g.DrawImage(&pic, Gdiplus::RectF(0, 0, 256, 256), src.X, src.Y, src.Width, src.Height,
                    Gdiplus::UnitPixel, &attr);
    }
    UINT num = 0, size = 0;
    Gdiplus::GetImageEncodersSize(&num, &size);
    std::vector<BYTE> buf(size);
    auto* codecs = reinterpret_cast<Gdiplus::ImageCodecInfo*>(buf.data());
    Gdiplus::GetImageEncoders(num, size, codecs);
    for (UINT i = 0; i < num; ++i)
        if (!wcscmp(codecs[i].MimeType, L"image/bmp")) {
            std::error_code ec;
            fs::create_directories(g_user, ec);
            if (icon.Save(out.c_str(), &codecs[i].Clsid, nullptr) == Gdiplus::Ok) return out;
        }
    return {};
}

HANDLE g_game = nullptr;
Session g_session;
constexpr UINT WM_GAME_EXIT = WM_APP + 1;

// The release's shader recipes (no shader code: pipeline states and hashes); a full library
// (library-v2, development) is used instead when present.
fs::path ShaderLibrary() {
    std::error_code ec;
    return PathOption("shader_library",
                      fs::is_regular_file(g_root / "shaders" / "library-v2" / "manifest.json", ec)
                          ? g_root / "shaders" / "library-v2"
                          : g_root / "shaders" / "recipes");
}

// Starts the game, with the session's shader cache already made (PrepareCache, on PLAY's worker
// thread). Empty on success, else what went wrong.
std::wstring LaunchGame(HWND notify) {
    std::error_code ec;
    if (const auto why = CheckGameDir(g_game_dir); !why.empty()) return why;
    const fs::path exe = g_bin / "iss-probe.exe", image = PreparedImage();
    if (!fs::is_regular_file(exe, ec) || !fs::is_regular_file(g_bin / "iss-gpu.dll", ec))
        return L"bin\\iss-probe.exe or bin\\iss-gpu.dll is missing.";
    if (!fs::is_regular_file(image, ec))
        return L"The game files are not prepared yet (user\\prepared\\boot-linked.bin).";
    const fs::path library = ShaderLibrary();
    g_session.library = fs::is_regular_file(library / "manifest.json", ec);
    fs::create_directories(g_user / "logs", ec);
    KeepNewest(g_user / "logs", 5);
    g_session.log_dir = g_user / "logs" / Timestamp();
    fs::create_directories(g_session.log_dir, ec);
    const fs::path saves = PathOption("saves", g_user / "saves");
    fs::create_directories(saves, ec);

    // The environment: this process's, without other port settings, then the game's.
    std::map<std::wstring, std::wstring, std::less<>> env;
    if (wchar_t* block = GetEnvironmentStringsW()) {
        for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
            std::wstring kv = p;
            const auto eq = kv.find(L'=', 1);
            if (eq == std::wstring::npos) continue;
            std::wstring key = kv.substr(0, eq);
            std::wstring upper = key;
            for (auto& c : upper) c = wchar_t(towupper(c));
            if (upper.rfind(L"ISS_", 0) == 0 || upper.rfind(L"SHADPS4_", 0) == 0 ||
                upper.rfind(L"BB_", 0) == 0)
                continue;
            env[key] = kv.substr(eq + 1);
        }
        FreeEnvironmentStringsW(block);
    }
    for (const auto& kv : kGameEnvironment) env[Widen(kv[0])] = Widen(kv[1]);
    env[L"ISS_GAME_DIR"] = g_game_dir;
    if (const fs::path icon = MakeGameIcon(); !icon.empty()) env[L"ISS_WINDOW_ICON"] = icon.wstring();
    env[L"ISS_SAVE_ROOT"] = saves.wstring();
    env[L"ISS_PIPELINE_CACHE_ROOT"] = g_session.cache_work.wstring();
    if (g_session.library) env[L"ISS_PIPELINE_LIBRARY"] = library.wstring();
    env[L"SHADPS4_FRAME_CSV"] = (g_session.log_dir / "frames.csv").wstring();
    env[L"ISS_STOP_FILE"] = (g_session.log_dir / "stop.request").wstring();
    // Settings.
    env[L"ISS_FULLSCREEN"] = Widen(SettingValue("window_mode"));
    env[L"BB_PRESENT_MODE"] = Widen(SettingValue("vsync"));
    env[L"BB_FPS_LIMIT"] = Widen(SettingValue("fps_limit"));
    // Flips are taken on vblank ticks (240 Hz): a limit that does not divide it would snap to the
    // next tick (75 ran at 80 FPS, 144 at 120), so the vblank runs at a multiple of the limit.
    if (const int limit = atoi(SettingValue("fps_limit")); limit > 0)
        env[L"BB_VBLANK_HZ"] = std::to_wstring(limit * ((240 + limit - 1) / limit));
    env[L"BB_FSR1"] = Widen(SettingValue("fsr"));
    env[L"ISS_RESOLUTION"] = Widen(ResolutionValue());
    env[L"ISS_FOV"] = Widen(SettingValue("fov"));
    env[L"ISS_HUD_ASPECT"] = Widen(SettingValue("hud_width"));
    env[L"ISS_ASYNC_SKIP_DRAWS"] = Widen(SettingValue("shader_mode"));
    env[L"ISS_GPU_BLACKBOX"] = Widen(SettingValue("blackbox"));
    env[L"ISS_UNIFIED_MEMORY"] = Widen(UnifiedMemoryValue());
    env[L"ISS_LANGUAGE"] = Widen(SettingValue("language"));
    env[L"ISS_VOLUME"] = Widen(SettingValue("volume"));
    if (const std::string map = PadMapString(); !map.empty()) env[L"ISS_PAD_MAP"] = Widen(map);
    {
        std::string labels;
        for (const auto& l : ScanDlcLabels(DlcDir())) labels += (labels.empty() ? "" : ",") + l;
        if (!labels.empty()) env[L"ISS_ADDCONT_LABELS"] = Widen(labels);
    }
    std::wstring block;
    for (const auto& [k, v] : env) block += k + L"=" + v + L'\0';
    block += L'\0';

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE out = CreateFileW((g_session.log_dir / "stdout.log").c_str(), GENERIC_WRITE,
                             FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE err = CreateFileW((g_session.log_dir / "stderr.log").c_str(), GENERIC_WRITE,
                             FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = out;
    si.hStdError = err;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + exe.wstring() + L"\" \"" + image.wstring() + L"\" --entry";
    const fs::path cwd = image.parent_path();
    const BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, block.data(),
                                   cwd.c_str(), &si, &pi);
    if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    if (err != INVALID_HANDLE_VALUE) CloseHandle(err);
    if (!ok) return L"Could not start the game (error " + std::to_wstring(GetLastError()) + L").";
    CloseHandle(pi.hThread);
    g_game = pi.hProcess;
    std::thread([notify, process = pi.hProcess] {
        WaitForSingleObject(process, INFINITE);
        DWORD code = 0;
        GetExitCodeProcess(process, &code);
        PostMessageW(notify, WM_GAME_EXIT, code, 0);
    }).detach();
    return {};
}

// ---------------------------------------------------------------------------------------------
// Art
// ---------------------------------------------------------------------------------------------
std::unique_ptr<Bitmap> g_background, g_logo;
HICON g_icon = nullptr, g_icon_small = nullptr;

void LoadArt() {
    g_background.reset();
    g_logo.reset();
    if (!CheckGameDir(g_game_dir).empty()) return;
    const fs::path sys = fs::path(g_game_dir) / "sce_sys";
    auto bg = std::make_unique<Bitmap>((sys / "pic1.png").c_str());
    if (bg->GetLastStatus() == Ok) g_background = std::move(bg);
    auto logo_file = std::make_unique<Bitmap>((g_root / "assets" / "logo.png").c_str());
    const bool has_logo = logo_file->GetLastStatus() == Ok;
    if (has_logo) g_logo = std::move(logo_file);
    Bitmap icon((sys / "icon0.png").c_str());
    if (icon.GetLastStatus() != Ok) return;
    if (!has_logo) {
    // The logo is the bright part of icon0 over its black background: the band with the text,
    // its black made transparent (the faint crows behind fall under the threshold).
    const UINT w = icon.GetWidth(), h = icon.GetHeight();
    const UINT top = h * 25 / 100, bottom = h * 86 / 100, left = w * 6 / 100, right = w * 94 / 100;
    auto logo = std::make_unique<Bitmap>(right - left, bottom - top, PixelFormat32bppARGB);
    BitmapData src{}, dst{};
    Rect all(0, 0, w, h), cut(0, 0, right - left, bottom - top);
    icon.LockBits(&all, ImageLockModeRead, PixelFormat32bppARGB, &src);
    logo->LockBits(&cut, ImageLockModeWrite, PixelFormat32bppARGB, &dst);
    for (UINT y = top; y < bottom; ++y) {
        const auto* in = reinterpret_cast<const UINT32*>(static_cast<BYTE*>(src.Scan0) + y * src.Stride);
        auto* o = reinterpret_cast<UINT32*>(static_cast<BYTE*>(dst.Scan0) + (y - top) * dst.Stride);
        for (UINT x = left; x < right; ++x) {
            const UINT32 p = in[x];
            const int r = (p >> 16) & 255, g = (p >> 8) & 255, b = p & 255;
            const int m = std::max({r, g, b});
            int a = (m - 70) * 255 / 120;
            a = std::clamp(a, 0, 255);
            o[x - left] = (UINT32(a) << 24) | (p & 0xFFFFFF);
        }
    }
    logo->UnlockBits(&dst);
    icon.UnlockBits(&src);
    g_logo = std::move(logo);
    }
    // The window icons from the game's own icon, resampled to the system sizes (the taskbar
    // and the title use the big one, Alt+Tab and lists the small one).
    Bitmap icon_bitmap((sys / "icon0.png").c_str());
    auto scaled = [&](int size) {
        HICON h = nullptr;
        Bitmap b(size, size, PixelFormat32bppARGB);
        {
            // The drawing must be finished (Graphics gone) before the icon is made from it:
            // made earlier, the icon came out empty (no taskbar icon).
            Graphics gb(&b);
            gb.SetInterpolationMode(InterpolationModeHighQualityBicubic);
            gb.SetPixelOffsetMode(PixelOffsetModeHighQuality);
            gb.DrawImage(&icon_bitmap, 0, 0, size, size);
            gb.Flush(FlushIntentionSync);
        }
        b.GetHICON(&h);
        return h;
    };
    if (g_icon) DestroyIcon(g_icon);
    if (g_icon_small) DestroyIcon(g_icon_small);
    g_icon = scaled(std::max(48, GetSystemMetrics(SM_CXICON) * 2));
    g_icon_small = scaled(std::max(16, GetSystemMetrics(SM_CXSMICON) * 2));
}

// The launcher's file icon (in Explorer): the same game icon as its window, written into the .exe
// from the player's own sce_sys\icon0.png once the game folder is known (the release ships no game
// art). A running .exe cannot be written but can be renamed: a copy gets the icon resources
// (UpdateResource) and takes the launcher's name; the old file goes at the next start.
#pragma pack(push, 2)
struct IconGroupHeader {
    WORD reserved, type, count;
};
struct IconGroupEntry {
    BYTE width, height, colors, reserved;
    WORD planes, bits;
    DWORD bytes;
    WORD id;
};
#pragma pack(pop)

std::vector<BYTE> PngOfSize(Bitmap& src, int size) {
    Bitmap b(size, size, PixelFormat32bppARGB);
    {
        Graphics g(&b);
        g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
        g.SetPixelOffsetMode(PixelOffsetModeHighQuality);
        g.DrawImage(&src, 0, 0, size, size);
        g.Flush(FlushIntentionSync);
    }
    UINT num = 0, bytes = 0;
    GetImageEncodersSize(&num, &bytes);
    std::vector<BYTE> buf(bytes);
    auto* codecs = reinterpret_cast<ImageCodecInfo*>(buf.data());
    GetImageEncoders(num, bytes, codecs);
    std::vector<BYTE> data;
    IStream* stream = nullptr;
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) return data;
    for (UINT i = 0; i < num; ++i)
        if (!wcscmp(codecs[i].MimeType, L"image/png") && b.Save(stream, &codecs[i].Clsid, nullptr) == Ok) {
            STATSTG st{};
            stream->Stat(&st, STATFLAG_NONAME);
            data.resize(size_t(st.cbSize.QuadPart));
            const LARGE_INTEGER zero{};
            stream->Seek(zero, STREAM_SEEK_SET, nullptr);
            ULONG read = 0;
            stream->Read(data.data(), ULONG(data.size()), &read);
            data.resize(read);
            break;
        }
    stream->Release();
    return data;
}

void ApplyExeIcon() {
    if (!CheckGameDir(g_game_dir).empty()) return;
    wchar_t self_buf[MAX_PATH];
    GetModuleFileNameW(nullptr, self_buf, MAX_PATH);
    const fs::path self = self_buf, old = self.wstring() + L".old", next = self.wstring() + L".new";
    std::error_code ec;
    fs::remove(old, ec);
    fs::remove(next, ec);
    const fs::path marker = g_user / "exe-icon.txt";
    const std::string source = Narrow(g_game_dir) + "|" + std::to_string(fs::file_size(self, ec));
    if (ReadText(marker) == source) return;
    Bitmap icon((fs::path(g_game_dir) / "sce_sys" / "icon0.png").c_str());
    if (icon.GetLastStatus() != Ok || !CopyFileW(self.c_str(), next.c_str(), FALSE)) return;
    HANDLE update = BeginUpdateResourceW(next.c_str(), FALSE);
    if (!update) {
        fs::remove(next, ec);
        return;
    }
    const int sizes[] = {256, 64, 48, 32, 24, 16};
    const WORD lang = MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL);
    std::vector<BYTE> group(sizeof(IconGroupHeader));
    const IconGroupHeader header{0, 1, WORD(std::size(sizes))};
    memcpy(group.data(), &header, sizeof(header));
    bool ok = true;
    for (size_t i = 0; i < std::size(sizes) && ok; ++i) {
        std::vector<BYTE> png = PngOfSize(icon, sizes[i]);
        ok = !png.empty() && UpdateResourceW(update, RT_ICON, MAKEINTRESOURCEW(i + 1), lang, png.data(), DWORD(png.size()));
        const BYTE side = BYTE(sizes[i] == 256 ? 0 : sizes[i]);
        const IconGroupEntry entry{side, side, 0, 0, 1, 32, DWORD(png.size()), WORD(i + 1)};
        const auto* p = reinterpret_cast<const BYTE*>(&entry);
        group.insert(group.end(), p, p + sizeof(entry));
    }
    ok = ok && UpdateResourceW(update, RT_GROUP_ICON, MAKEINTRESOURCEW(1), lang, group.data(), DWORD(group.size()));
    if (!EndUpdateResourceW(update, !ok) || !ok) {
        fs::remove(next, ec);
        return;
    }
    if (!MoveFileExW(self.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        fs::remove(next, ec);
        return;
    }
    if (!MoveFileExW(next.c_str(), self.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        MoveFileExW(old.c_str(), self.c_str(), 0);
        return;
    }
    SetFileAttributesW(old.c_str(), FILE_ATTRIBUTE_HIDDEN); // until the next start deletes it
    std::ofstream(marker) << Narrow(g_game_dir) << "|" << fs::file_size(self, ec);
    SHChangeNotify(SHCNE_UPDATEITEM, SHCNF_PATHW, self.c_str(), nullptr);
}

// ---------------------------------------------------------------------------------------------
// UI: drawn on a 960x600 canvas scaled to the window (on a Steam Deck or any screen up to
// 1280x800 the launcher opens borderless fullscreen). Mouse, keyboard and controllers (XInput:
// Steam Input on the Deck, Xbox pads, PlayStation pads through Steam) move a focus between items.
// ---------------------------------------------------------------------------------------------
enum class Screen { Main, Settings, Controllers, Setup, Running, Preparing };
Screen g_screen = Screen::Main;
int g_tab = TabDisplay;
std::wstring g_status;
float g_scale = 1.0f;     // canvas units -> pixels
HWND g_hwnd = nullptr;
bool g_fullscreen = false;
bool g_pad_used = false;  // the last input came from a controller: show its button hints
std::unique_ptr<PrivateFontCollection> g_fonts;
std::unique_ptr<FontFamily> g_oswald;

enum Hot {
    HotNone, HotClose, HotMinimize, HotPlay, HotSettings, HotQuit, HotBack, HotBrowse, HotBrowseDlc,
    HotContinue, HotControllers, HotMapReset, HotMapDone, HotHelp, HotTab0 = 20, HotSetting0 = 100, HotAction0 = 200,
    HotMap0 = 300
};
struct Spot {
    RectF rect;
    int id;
};
std::vector<Spot> g_spots;
int g_hover = HotNone;  // under the mouse
int g_focus = HotPlay;  // keyboard / controller focus

enum Action { ActDlcFolder, ActChangeGame, ActOpenSaves, ActOpenLogs, ActClearCache, ActCount };
const wchar_t* kActionNames[ActCount] = {L"DLC folder", L"Change game folder", L"Open saves folder",
                                         L"Open logs folder", L"Clear shader cache"};

const Color kYellow(255, 245, 166, 28), kYellowHot(255, 255, 190, 60), kText(255, 225, 225, 225),
    kDim(255, 135, 135, 135), kBar(255, 12, 12, 12);

float S(float v) { return v * g_scale; }

const FontFamily* Family() {
    static FontFamily fallback(L"Tahoma");
    return g_oswald ? g_oswald.get() : &fallback;
}

void LoadFonts() {
    g_fonts = std::make_unique<PrivateFontCollection>();
    const fs::path font = g_root / "assets" / "fonts" / "Oswald.ttf";
    if (g_fonts->AddFontFile(font.c_str()) != Ok || g_fonts->GetFamilyCount() < 1) return;
    int found = 0;
    auto families = std::make_unique<FontFamily[]>(1);
    g_fonts->GetFamilies(1, families.get(), &found);
    if (found) g_oswald.reset(families[0].Clone());
}

void SpacedText(Graphics& g, const wchar_t* text, const Font& font, const Brush& brush, float x,
                float y, float spacing) {
    StringFormat fmt(StringFormat::GenericTypographic());
    fmt.SetFormatFlags(fmt.GetFormatFlags() | StringFormatFlagsMeasureTrailingSpaces);
    for (const wchar_t* p = text; *p; ++p) {
        g.DrawString(p, 1, &font, PointF(x, y), &fmt, &brush);
        RectF box;
        g.MeasureString(p, 1, &font, PointF(0, 0), &fmt, &box);
        x += box.Width + spacing;
    }
}

float SpacedWidth(Graphics& g, const wchar_t* text, const Font& font, float spacing) {
    StringFormat fmt(StringFormat::GenericTypographic());
    fmt.SetFormatFlags(fmt.GetFormatFlags() | StringFormatFlagsMeasureTrailingSpaces);
    float w = 0;
    for (const wchar_t* p = text; *p; ++p) {
        RectF box;
        g.MeasureString(p, 1, &font, PointF(0, 0), &fmt, &box);
        w += box.Width + spacing;
    }
    return w - spacing;
}

// The items the focus walks through, top to bottom, on the current screen.
std::vector<int> FocusOrder() {
    std::vector<int> order;
    switch (g_screen) {
    case Screen::Main: order = {HotPlay, HotSettings, HotControllers, HotQuit, HotHelp}; break;
    case Screen::Controllers:
        for (int i = 0; i < kPadSourceCount; ++i) order.push_back(HotMap0 + i);
        order.insert(order.end(), {HotMapReset, HotMapDone, HotHelp});
        break;
    case Screen::Setup:
        order = {HotBrowse};
        if (CheckGameDir(g_game_dir).empty()) order.insert(order.end(), {HotBrowseDlc, HotContinue});
        order.push_back(HotQuit);
        break;
    case Screen::Settings:
        for (size_t i = 0; i < g_settings.size(); ++i)
            if (g_settings[i].tab == g_tab) order.push_back(HotSetting0 + int(i));
        if (g_tab == TabAdvanced)
            for (int a = 0; a < ActCount; ++a) order.push_back(HotAction0 + a);
        order.push_back(HotBack);
        order.push_back(HotHelp);
        break;
    case Screen::Running: case Screen::Preparing: break;
    }
    return order;
}

void FixFocus() {
    const auto order = FocusOrder();
    if (order.empty()) return;
    if (std::find(order.begin(), order.end(), g_focus) == order.end()) g_focus = order.front();
}

bool Lit(int id) { return g_hover == id || (g_focus == id && g_hover == HotNone); }

// The names of the game's DLC licenses (content id labels), for the setup page.
std::wstring DlcName(const std::string& label) {
    static const std::map<std::string, const wchar_t*> names = {
        {"UNLOCKCOLEMISSIO", L"Cole's Legacy"},
        {"UNLOCKCOLEJACKET", L"Cole's Jacket"},
        {"UNLOCKVESTHANDPR", L"Cole's Legendary Vest"},
        {"UNLOCKVESTEXIST0", L"EXIST Vest"},
        {"UNLOCKVESTJOS000", L"JOS Vest"},
        {"UNLOCKVESTPENNYA", L"Penny Arcade Vest"},
        {"UNLOCKVESTSKELET", L"iam8bit Vest"},
    };
    const auto it = names.find(label);
    return it != names.end() ? it->second : Widen(label);
}

std::wstring DlcSummary() {
    const fs::path dir = DlcDir();
    if (dir.empty()) return L"Not set";
    const auto n = ScanDlcLabels(dir).size();
    return std::to_wstring(n) + (n == 1 ? L" unlocked" : L" unlocked");
}

// Game preparation (first launch, or a new game folder) on a worker thread.
constexpr UINT WM_PREP_PROGRESS = WM_APP + 2, WM_PREP_DONE = WM_APP + 3, WM_CACHE_READY = WM_APP + 4;
std::atomic<float> g_prep_progress{0.0f};
std::atomic<const wchar_t*> g_prep_step{L""};
bool g_play_after_prep = false;
bool g_prep_starting = false; // the Preparing page shows PLAY's session cache, not the game files
bool g_cache_ready = false;   // the session's shader cache is made: PLAY starts the game
float g_prep_shown = 0.0f; // the drawn bar, easing toward g_prep_progress
void Restart();

void StartPreparation(bool then_play) {
    g_play_after_prep = then_play;
    g_prep_progress = 0.0f;
    g_prep_shown = 0.0f;
    g_prep_step = L"Starting";
    g_screen = Screen::Preparing;
    Restart();
    const fs::path game = g_game_dir, out = g_user / "prepared" / "boot-linked.bin";
    const HWND hwnd = g_hwnd;
    std::thread([game, out, hwnd] {
        const std::wstring result = iss_prepare::PrepareGame(game, out, [hwnd](float p, const wchar_t* step) {
            g_prep_progress = p;
            g_prep_step = step;
            PostMessageW(hwnd, WM_PREP_PROGRESS, 0, 0);
        });
        PostMessageW(hwnd, WM_PREP_DONE, 0, LPARAM(new std::wstring(result)));
    }).detach();
}

// PLAY: the session's shader cache is made on a worker thread (the STARTING page), then the game
// starts (WM_CACHE_READY).
void StartSession() {
    g_prep_starting = true;
    g_prep_progress = 0.0f;
    g_prep_shown = 0.0f;
    g_prep_step = L"Preparing the shader cache";
    g_screen = Screen::Preparing;
    Restart();
    std::error_code ec;
    const bool library = fs::is_regular_file(ShaderLibrary() / "manifest.json", ec);
    const HWND hwnd = g_hwnd;
    std::thread([library, hwnd] {
        const fs::path work = PrepareCache(library, [hwnd](float p) {
            g_prep_progress = p;
            PostMessageW(hwnd, WM_PREP_PROGRESS, 0, 0);
        });
        PostMessageW(hwnd, WM_CACHE_READY, 0, LPARAM(new fs::path(work)));
    }).detach();
}

// Animation state. Every value eases toward its target with an exponential curve on the real
// frame time (smooth at any refresh rate); the 16 ms timer repaints while something still moves.
ULONGLONG g_last_paint = 0;
bool g_animating = false;
bool g_pulsing = false;        // only the PLAY glow moves: 30 FPS is enough
float g_time = 0.0f;           // seconds, for the PLAY pulse
float g_content = 0.0f;        // fade and slide of the page in (0 -> 1) on a screen or tab change
RectF g_bar;                   // the sliding selection bar
float g_bar_alpha = 0.0f;
bool g_bar_placed = false;
std::map<int, float> g_lit;    // per item highlight (0 -> 1)
std::unique_ptr<Bitmap> g_backdrop;
int g_backdrop_w = 0, g_backdrop_h = 0;
bool g_backdrop_fullscreen = false;

float Approach(float cur, float target, float dt, float speed) {
    return target + (cur - target) * std::exp(-speed * dt);
}
Color Mix(const Color& a, const Color& b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    auto m = [t](BYTE x, BYTE y) { return BYTE(x + (y - x) * t + 0.5f); };
    return Color(m(a.GetA(), b.GetA()), m(a.GetR(), b.GetR()), m(a.GetG(), b.GetG()), m(a.GetB(), b.GetB()));
}
void Restart() { g_content = 0.0f; } // a new page fades and slides in

// The background (pic1 and the shading) is rendered once per window size.
void BuildBackdrop(int width, int height, float bar) {
    g_backdrop = std::make_unique<Bitmap>(width, height, PixelFormat32bppPARGB);
    g_backdrop_w = width;
    g_backdrop_h = height;
    g_backdrop_fullscreen = g_fullscreen;
    Graphics g(g_backdrop.get());
    g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
    g.Clear(Color(255, 8, 8, 8));
    if (g_background) {
        const float iw = float(g_background->GetWidth()), ih = float(g_background->GetHeight());
        const float area_h = height - bar, k = std::max(width / iw, area_h / ih);
        const float dw = iw * k, dh = ih * k;
        g.DrawImage(g_background.get(), RectF(width - dw, bar + (area_h - dh) / 2, dw, dh));
    }
    LinearGradientBrush shade(PointF(0, 0), PointF(width * 0.62f, 0), Color(255, 6, 6, 6), Color(0, 6, 6, 6));
    const REAL positions[] = {0.0f, 0.45f, 1.0f};
    const Color colors[] = {Color(250, 6, 6, 6), Color(215, 6, 6, 6), Color(0, 6, 6, 6)};
    shade.SetInterpolationColors(colors, positions, 3);
    shade.SetWrapMode(WrapModeTileFlipXY);
    g.FillRectangle(&shade, RectF(0, bar, width * 0.62f + 1, float(height)));
    LinearGradientBrush floor(PointF(0, height * 0.72f), PointF(0, float(height)), Color(0, 0, 0, 0),
                              Color(190, 0, 0, 0));
    floor.SetWrapMode(WrapModeTileFlipXY);
    g.FillRectangle(&floor, RectF(0, height * 0.72f, float(width), height * 0.28f + 1));
}

// Items the sliding bar can sit behind (rows and text menu entries).
bool BarItem(int id) {
    return id == HotSettings || id == HotQuit || id == HotBack || id == HotControllers ||
           id == HotMapReset || id == HotMapDone || (id >= HotMap0 && id < HotMap0 + kPadSourceCount) ||
           (id >= HotSetting0 && id < HotSetting0 + int(g_settings.size())) ||
           (id >= HotAction0 && id < HotAction0 + ActCount);
}

void Paint(HDC hdc, int width, int height) {
    const ULONGLONG now = GetTickCount64();
    const float dt = g_last_paint ? std::clamp((now - g_last_paint) / 1000.0f, 0.0f, 0.05f) : 0.0f;
    g_last_paint = now;
    g_time += dt;
    g_animating = false;
    g_pulsing = false;
    auto ease = [&](float& v, float target, float speed) {
        v = Approach(v, target, dt, speed);
        if (std::fabs(v - target) > 0.002f) g_animating = true;
        else v = target;
    };

    const float bar = g_fullscreen ? 0.0f : S(30);
    if (!g_backdrop || g_backdrop_w != width || g_backdrop_h != height || g_backdrop_fullscreen != g_fullscreen)
        BuildBackdrop(width, height, bar);
    // Frame buffers kept across frames (allocating them every frame cost most of the CPU).
    static std::unique_ptr<Bitmap> back_buf, layer_buf;
    if (!back_buf || int(back_buf->GetWidth()) != width || int(back_buf->GetHeight()) != height) {
        back_buf = std::make_unique<Bitmap>(width, height, PixelFormat32bppPARGB);
        layer_buf = std::make_unique<Bitmap>(width, height, PixelFormat32bppPARGB);
    }
    Bitmap& back = *back_buf;
    Graphics frame(&back);
    frame.SetCompositingQuality(CompositingQualityHighSpeed);
    frame.DrawImage(g_backdrop.get(), 0, 0, width, height);

    // While a page fades and slides in it is drawn on its own layer; once in place, directly.
    ease(g_content, 1.0f, 7.0f);
    const bool layered = g_content < 1.0f;
    Graphics g(layered ? static_cast<Image*>(layer_buf.get()) : static_cast<Image*>(&back));
    g.SetSmoothingMode(SmoothingModeAntiAlias);
    g.SetTextRenderingHint(TextRenderingHintAntiAliasGridFit);
    g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
    if (layered) g.Clear(Color(0, 0, 0, 0));
    const std::vector<Spot> previous = g_spots;
    g_spots.clear();

    const FontFamily* oswald = Family();
    FontFamily segoe(L"Segoe UI");
    SolidBrush text_brush(kText), dim_brush(kDim), yellow(kYellow), black(Color(255, 10, 10, 10));
    auto lit = [&](int id) {
        float& v = g_lit[id];
        ease(v, Lit(id) ? 1.0f : 0.0f, 16.0f);
        return v;
    };

    // The sliding selection bar, behind the items (placed on last frame's layout).
    {
        const int target = g_hover != HotNone ? g_hover : g_focus;
        const Spot* spot = nullptr;
        for (const auto& s : previous)
            if (s.id == target && BarItem(target)) spot = &s;
        if (spot) {
            if (!g_bar_placed || g_bar_alpha < 0.01f) {
                g_bar = spot->rect;
                g_bar_placed = true;
            }
            ease(g_bar.X, spot->rect.X, 18.0f);
            ease(g_bar.Y, spot->rect.Y, 18.0f);
            ease(g_bar.Width, spot->rect.Width, 18.0f);
            ease(g_bar.Height, spot->rect.Height, 18.0f);
        }
        ease(g_bar_alpha, spot ? 1.0f : 0.0f, 14.0f);
        if (g_bar_alpha > 0.01f) {
            SolidBrush hl(Color(BYTE(48 * g_bar_alpha), 255, 255, 255));
            g.FillRectangle(&hl, g_bar);
            SolidBrush mark(Color(BYTE(255 * g_bar_alpha), kYellow.GetR(), kYellow.GetG(), kYellow.GetB()));
            g.FillRectangle(&mark, RectF(g_bar.X, g_bar.Y, S(3), g_bar.Height));
        }
    }

    // Logo (smaller on the settings page).
    const float left = S(78);
    const bool compact = g_screen == Screen::Settings || g_screen == Screen::Setup || g_screen == Screen::Controllers;
    // The controller page has the most rows: the smallest logo.
    const bool tiny = g_screen == Screen::Controllers;
    float y = bar + S(tiny ? 16 : compact ? 28 : 46);
    {
        const float lw = S(tiny ? 130 : compact ? 210 : 340);
        if (g_logo) {
            const float lh = lw * g_logo->GetHeight() / g_logo->GetWidth();
            g.DrawImage(g_logo.get(), RectF(left - S(14), y, lw, lh));
            y += lh + S(tiny ? 10 : compact ? 18 : 30);
        } else {
            // No game folder yet: the title in type until the game's own logo can be shown.
            Font small_title(oswald, S(22), FontStyleRegular, UnitPixel);
            SpacedText(g, L"inFAMOUS", small_title, text_brush, left + S(2), y, S(3));
            Font big(oswald, S(52), FontStyleBold, UnitPixel);
            g.DrawString(L"SECOND SON", -1, &big, PointF(left - S(2), y + S(22)), &yellow);
            y += S(104);
        }
    }

    Font menu(oswald, S(21), FontStyleRegular, UnitPixel);
    Font play_font(&segoe, S(18), FontStyleBold, UnitPixel); // PLAY keeps its own font
    Font label(oswald, S(17), FontStyleRegular, UnitPixel);
    Font hint(oswald, S(13), FontStyleRegular, UnitPixel);
    Font tab_font(oswald, S(15), FontStyleRegular, UnitPixel);

    auto menu_item = [&](const wchar_t* text, int id, float at) {
        const RectF box(left, at, S(270), S(40));
        const float t = lit(id);
        SolidBrush brush(Mix(kText, kYellow, t));
        SpacedText(g, text, menu, brush, left + S(16) + S(4) * t, at + S(4), S(3));
        g_spots.push_back({box, id});
    };

    if (g_screen == Screen::Main || g_screen == Screen::Running) {
        const bool running = g_screen == Screen::Running;
        const float t = running ? 0.0f : lit(HotPlay);
        const float grow = S(4) * t;
        const RectF play(left - grow / 2, y - grow / 2, S(250) + grow, S(48) + grow);
        if (t > 0.01f) {
            // A soft glow that breathes while PLAY is selected.
            const float pulse = 0.55f + 0.45f * std::sin(g_time * 3.2f);
            for (int i = 3; i >= 1; --i) {
                SolidBrush glow(Color(BYTE(22 * t * pulse * (4 - i)), 255, 190, 60));
                const float e = S(4.0f * i);
                g.FillRectangle(&glow, RectF(play.X - e, play.Y - e, play.Width + 2 * e, play.Height + 2 * e));
            }
            if (GetForegroundWindow() == g_hwnd) g_pulsing = true;
        }
        SolidBrush play_brush(running ? Color(255, 90, 75, 30) : Mix(kYellow, kYellowHot, t));
        g.FillRectangle(&play_brush, play);
        const float ty = play.Y + play.Height / 2;
        PointF tri[3] = {PointF(play.X + S(22), ty - S(9)), PointF(play.X + S(22), ty + S(9)),
                         PointF(play.X + S(38), ty)};
        g.FillPolygon(&black, tri, 3);
        SpacedText(g, running ? L"RUNNING" : L"PLAY", play_font, black, play.X + S(56), ty - S(12), S(3));
        if (!running) {
            g_spots.push_back({play, HotPlay});
            menu_item(L"SETTINGS", HotSettings, y + S(70));
            menu_item(L"CONTROLS", HotControllers, y + S(120));
            menu_item(L"QUIT", HotQuit, y + S(170));
        }
    } else if (g_screen == Screen::Settings) {
        float tx = left;
        const float ty = y;
        static float underline_x = 0, underline_w = 0;
        for (int t = 0; t < TabCount; ++t) {
            const float w = SpacedWidth(g, kTabNames[t], tab_font, S(2.5f));
            const float hl = lit(HotTab0 + t);
            SolidBrush brush(t == g_tab ? kYellow : Mix(kDim, kText, hl));
            SpacedText(g, kTabNames[t], tab_font, brush, tx, ty, S(2.5f));
            if (t == g_tab) {
                if (underline_w == 0) {
                    underline_x = tx;
                    underline_w = w;
                }
                ease(underline_x, tx, 16.0f);
                ease(underline_w, w, 16.0f);
            }
            g_spots.push_back({RectF(tx - S(5), ty - S(5), w + S(10), S(30)), HotTab0 + t});
            tx += w + S(30);
        }
        SolidBrush underline(kYellow);
        g.FillRectangle(&underline, RectF(underline_x, ty + S(25), underline_w, S(2)));
        if (g_pad_used) g.DrawString(L"LB / RB", -1, &hint, PointF(tx + S(4), ty + S(3)), &dim_brush);
        float ry = ty + S(46);
        const float row_w = S(450);
        for (size_t i = 0; i < g_settings.size(); ++i) {
            auto& s = g_settings[i];
            if (s.tab != g_tab) continue;
            const int id = HotSetting0 + int(i);
            const RectF row(left - S(10), ry - S(4), row_w, S(50));
            const float t = lit(id);
            SolidBrush label_brush(Mix(kText, Color(255, 255, 255, 255), t));
            g.DrawString(s.label, -1, &label, PointF(left + S(4) * t, ry - S(2)), &label_brush);
            g.DrawString(s.hint, -1, &hint, PointF(left + S(4) * t, ry + S(23)), &dim_brush);
            const wchar_t* value = s.choices[s.index].label;
            RectF vbox;
            g.MeasureString(value, -1, &label, PointF(0, 0), &vbox);
            const float vx = left + row_w - S(40) - vbox.Width;
            SolidBrush arrows(Mix(kDim, kYellow, t));
            g.DrawString(L"‹", -1, &label, PointF(vx - S(22) - S(3) * t, ry - S(2)), &arrows);
            g.DrawString(value, -1, &label, PointF(vx, ry - S(2)), &yellow);
            g.DrawString(L"›", -1, &label, PointF(left + row_w - S(30) + S(3) * t, ry - S(2)), &arrows);
            g_spots.push_back({row, id});
            ry += S(56);
        }
        if (g_tab == TabAdvanced) {
            for (int a = 0; a < ActCount; ++a) {
                const int id = HotAction0 + a;
                const RectF row(left - S(10), ry - S(4), row_w, S(34));
                const float t = lit(id);
                SolidBrush brush(Mix(kText, kYellow, t));
                g.DrawString(kActionNames[a], -1, &label, PointF(left + S(4) * t, ry), &brush);
                if (a == ActDlcFolder) {
                    const std::wstring summary = DlcSummary();
                    RectF vbox;
                    g.MeasureString(summary.c_str(), -1, &hint, PointF(0, 0), &vbox);
                    g.DrawString(summary.c_str(), -1, &hint, PointF(left + row_w - S(28) - vbox.Width, ry + S(3)), &dim_brush);
                }
                g_spots.push_back({row, id});
                ry += S(36);
            }
        }
        menu_item(L"BACK", HotBack, ry + S(8));
    } else if (g_screen == Screen::Controllers) {
        // The controller layout: two columns of buttons, each with what it does in game.
        Font head(oswald, S(18), FontStyleRegular, UnitPixel);
        SpacedText(g, L"CONTROLLER LAYOUT", head, text_brush, left, y, S(2));
        const float head_w = SpacedWidth(g, L"CONTROLLER LAYOUT", head, S(2));
        const bool recommended = PadMapRecommended();
        SolidBrush badge(recommended ? kYellow : Color(255, 70, 70, 70));
        const wchar_t* badge_text = recommended ? L"RECOMMENDED" : L"CUSTOM";
        const float badge_w = SpacedWidth(g, badge_text, hint, S(1.5f)) + S(14);
        g.FillRectangle(&badge, RectF(left + head_w + S(14), y + S(3), badge_w, S(20)));
        SpacedText(g, badge_text, hint, recommended ? black : text_brush, left + head_w + S(21), y + S(4), S(1.5f));
        g.DrawString(L"For Xbox and other controllers without a touchpad. PlayStation controllers keep their own layout.",
                     -1, &hint, PointF(left, y + S(28)), &dim_brush);
        const float rows_y = y + S(54), row_h = S(25), col_w = S(318);
        Font row_font(oswald, S(15), FontStyleRegular, UnitPixel);
        for (int i = 0; i < kPadSourceCount; ++i) {
            const int id = HotMap0 + i, col = i / 9, row = i % 9;
            const float cx = left + col * (col_w + S(14)), ry = rows_y + row * row_h;
            const float t = lit(id);
            SolidBrush label_brush(Mix(kText, Color(255, 255, 255, 255), t));
            g.DrawString(kPadSources[i].label, -1, &row_font, PointF(cx + S(4) * t, ry), &label_brush);
            const int target = g_pad_map[i];
            const bool changed = target != kRecommendedPadMap[i];
            SolidBrush value_brush(changed ? Color(255, 255, 255, 255) : kYellow);
            const wchar_t* value = kPadTargets[target].label;
            RectF vbox;
            g.MeasureString(value, -1, &row_font, PointF(0, 0), &vbox);
            const float v0 = cx + S(150), v1 = cx + col_w - S(24);
            SolidBrush arrows(Mix(kDim, kYellow, t));
            g.DrawString(L"‹", -1, &row_font, PointF(v0 - S(2) * t, ry), &arrows);
            g.DrawString(value, -1, &row_font, PointF((v0 + v1) / 2 - vbox.Width / 2 + S(6), ry), &value_brush);
            g.DrawString(L"›", -1, &row_font, PointF(v1 + S(2) * t, ry), &arrows);
            g_spots.push_back({RectF(cx - S(10), ry - S(3), col_w, row_h - S(1)), id});
        }
        const float ey = rows_y + 9 * row_h + S(8);
        g.DrawString(L"Recommended: the PS4 layout one to one; View and Menu press the left and right side of the "
                     L"touchpad. The game asks for the touchpad in fast gameplay moments (absorbing powers, finishers, "
                     L"prompts), so it stays under your thumbs instead of on the keyboard. Options (pause): Esc, or give "
                     L"it a button here.\n"
                     L"In game, F5 opens the touchpad & spray window (the ? at the top right tells the keys).",
                     -1, &hint, RectF(left, ey, S(800), S(76)), nullptr, &dim_brush);
        const float by = ey + S(74);
        auto text_button = [&](const wchar_t* text, int id, float x, float w) {
            const float t = lit(id);
            SolidBrush brush(Mix(kText, kYellow, t));
            SpacedText(g, text, menu, brush, x + S(16) + S(4) * t, by + S(4), S(3));
            g_spots.push_back({RectF(x, by, w, S(40)), id});
        };
        text_button(L"RESTORE RECOMMENDED", HotMapReset, left, S(300));
        text_button(L"BACK", HotMapDone, left + S(320), S(140));
    } else if (g_screen == Screen::Setup) {
        Font head(oswald, S(18), FontStyleRegular, UnitPixel);
        const bool has_game = CheckGameDir(g_game_dir).empty();
        auto button = [&](const wchar_t* text, int id, const RectF& r, bool primary) {
            const float t = lit(id);
            if (primary) {
                SolidBrush b(Mix(kYellow, kYellowHot, t));
                g.FillRectangle(&b, r);
            } else {
                SolidBrush b(Color(BYTE(40 + 50 * t), 255, 255, 255));
                g.FillRectangle(&b, r);
                Pen edge(Mix(Color(255, 90, 90, 90), kYellow, t), S(1.2f));
                g.DrawRectangle(&edge, r);
            }
            const float w = SpacedWidth(g, text, menu, S(3));
            SolidBrush tb(primary ? Color(255, 10, 10, 10) : Mix(kText, kYellow, t));
            SpacedText(g, text, menu, tb, r.X + (r.Width - w) / 2, r.Y + (r.Height - S(30)) / 2, S(3));
            g_spots.push_back({r, id});
        };
        auto ellipsize = [&](std::wstring text, float max_w, const Font& f) {
            RectF box;
            g.MeasureString(text.c_str(), -1, &f, PointF(0, 0), &box);
            if (box.Width <= max_w) return text;
            while (text.size() > 4) {
                text.erase(0, 1);
                g.MeasureString((L"…" + text).c_str(), -1, &f, PointF(0, 0), &box);
                if (box.Width <= max_w) break;
            }
            return L"…" + text;
        };
        // Step 1: the game folder.
        SpacedText(g, L"SELECT GAME PATH", head, text_brush, left, y, S(2));
        button(L"BROWSE", HotBrowse, RectF(left, y + S(30), S(170), S(38)), !has_game);
        if (has_game) {
            SolidBrush ok(kYellow);
            g.DrawString(L"✓", -1, &label, PointF(left + S(184), y + S(37)), &ok);
            const std::wstring path = ellipsize(g_game_dir, S(330), hint);
            g.DrawString(path.c_str(), -1, &hint, PointF(left + S(206), y + S(40)), &dim_brush);
        }
        if (has_game) {
            // Step 2: the DLC packages (optional).
            const float y2 = y + S(88);
            SpacedText(g, L"SELECT DLCs", head, text_brush, left, y2, S(2));
            RectF tb;
            g.MeasureString(L"SELECT DLCs", -1, &head, PointF(0, 0), &tb);
            g.DrawString(L"optional", -1, &hint, PointF(left + S(124), y2 + S(4)), &dim_brush);
            button(L"BROWSE", HotBrowseDlc, RectF(left, y2 + S(30), S(170), S(38)), false);
            const auto labels = ScanDlcLabels(DlcDir());
            if (!labels.empty()) {
                SolidBrush ok(kYellow);
                g.DrawString(L"✓", -1, &label, PointF(left + S(184), y2 + S(37)), &ok);
                const std::wstring found = std::to_wstring(labels.size()) + L" DLC found";
                g.DrawString(found.c_str(), -1, &hint, PointF(left + S(206), y2 + S(40)), &text_brush);
                std::wstring list;
                for (const auto& l : labels) list += (list.empty() ? L"" : L"  •  ") + DlcName(l);
                g.DrawString(list.c_str(), -1, &hint, RectF(left, y2 + S(76), S(560), S(40)), nullptr, &dim_brush);
            } else {
                g.DrawString(L"Your DLC .pkg files: unlocks costumes and Cole's Legacy.", -1, &hint,
                             PointF(left + S(184), y2 + S(40)), &dim_brush);
            }
            button(L"CONTINUE", HotContinue, RectF(left, y2 + S(124), S(250), S(44)), true);
            menu_item(L"QUIT", HotQuit, y2 + S(178));
        } else {
            menu_item(L"QUIT", HotQuit, y + S(84));
        }
    }

    if (g_screen == Screen::Preparing) {
        Font head(oswald, S(20), FontStyleRegular, UnitPixel);
        SpacedText(g, g_prep_starting ? L"STARTING" : L"PREPARING GAME FILES", head, text_brush, left, y, S(2));
        g.DrawString(g_prep_starting ? L"The shaders you have built so far, ready for this session.\n"
                                       L"The game opens in a moment."
                                     : L"Only on the first launch and after changing the game folder.\n"
                                       L"Your game folder is only read, never modified.", -1, &label,
                     PointF(left, y + S(36)), &dim_brush);
        ease(g_prep_shown, g_prep_progress.load(), 6.0f);
        const RectF track(left, y + S(108), S(420), S(8));
        SolidBrush track_brush(Color(255, 40, 40, 40));
        g.FillRectangle(&track_brush, track);
        g.FillRectangle(&yellow, RectF(track.X, track.Y, track.Width * g_prep_shown, track.Height));
        // A light sweeping over the filled part while it works.
        const float sweep = std::fmod(g_time * 0.8f, 1.0f) * track.Width * g_prep_shown;
        SolidBrush shine(Color(110, 255, 240, 200));
        g.FillRectangle(&shine, RectF(track.X + std::max(0.0f, sweep - S(30)), track.Y, std::min(S(30), sweep), track.Height));
        g_animating = true;
        const std::wstring step = std::wstring(g_prep_step.load()) + L"…  " +
                                  std::to_wstring(int(g_prep_shown * 100 + 0.5f)) + L"%";
        g.DrawString(step.c_str(), -1, &hint, PointF(left, y + S(124)), &dim_brush);
    }

    // The ? in the corner of every page: the in-game touchpad and spray keys, shown on hover (or focus).
    if (g_screen == Screen::Main || g_screen == Screen::Settings || g_screen == Screen::Controllers) {
        const RectF help(float(width) - S(64), bar + S(18), S(28), S(28));
        const float th = lit(HotHelp);
        Pen ring(Mix(kDim, kYellow, th), S(1.6f));
        g.DrawEllipse(&ring, help);
        SolidBrush mark_brush(Mix(kText, kYellow, th));
        StringFormat centre;
        centre.SetAlignment(StringAlignmentCenter);
        centre.SetLineAlignment(StringAlignmentCenter);
        g.DrawString(L"?", -1, &label, help, &centre, &mark_brush);
        g_spots.push_back({RectF(help.X - S(4), help.Y - S(4), help.Width + S(8), help.Height + S(8)), HotHelp});
        if (g_hover == HotHelp || (g_focus == HotHelp && g_hover == HotNone)) {
            const RectF box(float(width) - S(520), help.Y + S(38), S(490), S(262));
            SolidBrush shade(Color(245, 14, 14, 16));
            g.FillRectangle(&shade, box);
            Pen edge(kYellow, S(1.2f));
            g.DrawRectangle(&edge, box);
            SpacedText(g, L"IN GAME", hint, yellow, box.X + S(16), box.Y + S(12), S(2));
            g.DrawString(
                L"F5   Touchpad & spray window: a touchpad to click and drag, the touchpad sides, click and "
                L"swipes, the spray can and a guide. It stays on top of the game without taking the focus.\n\n"
                L"F6   Mouse as touchpad: in the game window the left button is a finger, the right button "
                L"clicks.\n\n"
                L"F7   Spray can, for the graffiti missions: the mouse aims the can, hold the right button to "
                L"shake it, hold the left button to spray. Shift: fine aim. Middle button: recenter. F7 again "
                L"ends it.\n\n"
                L"Keyboard: Space / Backspace press the left / right side of the touchpad, F8 both, 1-4 swipe.",
                -1, &hint, RectF(box.X + S(16), box.Y + S(36), box.Width - S(32), box.Height - S(44)), nullptr,
                &text_brush);
        }
    }

    // Footer: the status line, and the controls of the input in use.
    if (!g_status.empty()) {
        RectF box(left, float(height) - S(52), float(width) - left - S(30), S(24));
        g.DrawString(g_status.c_str(), -1, &hint, box, nullptr, &dim_brush);
    }
    if (g_screen != Screen::Running && g_screen != Screen::Preparing) {
        const wchar_t* keys = g_pad_used
            ? (g_screen == Screen::Settings || g_screen == Screen::Controllers ? L"Ⓐ Change   Ⓑ Back   ◀ ▶ Value"
                                            : L"Ⓐ Select   Ⓑ Back   START Play")
            : (g_screen == Screen::Settings || g_screen == Screen::Controllers ? L"Enter  Change     Esc  Back     ← →  Value"
                                            : L"Enter  Select     Esc  Back");
        RectF kbox;
        g.MeasureString(keys, -1, &hint, PointF(0, 0), &kbox);
        g.DrawString(keys, -1, &hint, PointF(width - kbox.Width - S(22), float(height) - S(30)), &dim_brush);
    }

    // Composite the page: fade in with a short slide from the left (ease-out).
    if (layered) {
        const float a = g_content, slide = (1.0f - a) * (1.0f - a) * S(22);
        ColorMatrix cm = {{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, a, 0}, {0, 0, 0, 0, 1}}};
        ImageAttributes attr;
        attr.SetColorMatrix(&cm);
        frame.DrawImage(layer_buf.get(), RectF(-slide, 0, float(width), float(height)), 0, 0, float(width),
                        float(height), UnitPixel, &attr);
        // Hit areas follow the drawn position.
        for (auto& s : g_spots) s.rect.X -= slide;
    }

    // Title bar (windowed only), on top and never faded.
    if (!g_fullscreen) {
        frame.SetSmoothingMode(SmoothingModeAntiAlias);
        frame.SetTextRenderingHint(TextRenderingHintAntiAliasGridFit);
        SolidBrush bar_brush(kBar);
        frame.FillRectangle(&bar_brush, RectF(0, 0, float(width), bar));
        Font title_font(oswald, S(13), FontStyleRegular, UnitPixel);
        SpacedText(frame, L"inFAMOUS SECOND SON", title_font, dim_brush, S(14), S(7), S(1.5f));
        const float bw = S(40);
        RectF close(width - bw, 0, bw, bar), minimize(width - 2 * bw, 0, bw, bar);
        const float tc = lit(HotClose), tm = lit(HotMinimize);
        SolidBrush red(Color(BYTE(255 * tc), 196, 43, 28)), grey(Color(BYTE(255 * tm), 40, 40, 40));
        frame.FillRectangle(&red, close);
        frame.FillRectangle(&grey, minimize);
        Pen pen(Color(255, 200, 200, 200), S(1.1f));
        const float cx = close.X + bw / 2, cy = bar / 2, d = S(4.5f);
        frame.DrawLine(&pen, cx - d, cy - d, cx + d, cy + d);
        frame.DrawLine(&pen, cx - d, cy + d, cx + d, cy - d);
        const float mx = minimize.X + bw / 2;
        frame.DrawLine(&pen, mx - d, cy, mx + d, cy);
        g_spots.push_back({close, HotClose});
        g_spots.push_back({minimize, HotMinimize});
    }

    Graphics screen(hdc);
    screen.DrawImage(&back, 0, 0);
}

int HitTest(int x, int y) {
    for (auto it = g_spots.rbegin(); it != g_spots.rend(); ++it)
        if (it->rect.Contains(float(x), float(y))) return it->id;
    return HotNone;
}

void Redraw() { InvalidateRect(g_hwnd, nullptr, FALSE); }

void SetStatus(std::wstring s) {
    g_status = std::move(s);
    Redraw();
}

void OpenFolder(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
    ShellExecuteW(g_hwnd, L"open", p.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void ApplyIcon() {
    if (!g_icon) return;
    SetClassLongPtrW(g_hwnd, GCLP_HICON, LONG_PTR(g_icon));
    SetClassLongPtrW(g_hwnd, GCLP_HICONSM, LONG_PTR(g_icon_small ? g_icon_small : g_icon));
    SendMessageW(g_hwnd, WM_SETICON, ICON_BIG, LPARAM(g_icon));
    SendMessageW(g_hwnd, WM_SETICON, ICON_SMALL, LPARAM(g_icon_small ? g_icon_small : g_icon));
}

void ChooseGameFolder() {
    const std::wstring dir = PickFolder(g_hwnd);
    if (dir.empty()) return;
    if (const auto why = CheckGameDir(dir); !why.empty()) {
        SetStatus(why);
        return;
    }
    const std::string language = SettingValue("language");
    g_game_dir = dir;
    ApplyEdition(language);
    SaveConfig();
    LoadArt();
    ApplyExeIcon();
    g_backdrop.reset();
    ApplyIcon();
    if (g_screen == Screen::Setup) {
        // First launch: next, the DLC (optional), then CONTINUE prepares the game.
        g_focus = HotBrowseDlc;
        Redraw();
        return;
    }
    FixFocus();
    SetStatus(L"Game folder: " + g_game_dir);
    if (!PreparedReady()) StartPreparation(false);
}

void ChooseDlcFolder() {
    const std::wstring dir = PickFolder(g_hwnd);
    if (dir.empty()) return;
    const auto labels = ScanDlcLabels(dir);
    if (labels.empty()) {
        SetStatus(L"No inFAMOUS Second Son DLC (.pkg) found in that folder.");
        return;
    }
    g_extra["dlc_dir"] = Narrow(dir);
    SaveConfig();
    SetStatus(std::to_wstring(labels.size()) + L" DLC unlocked. The base game is not modified.");
}

// Moves the value of a setting by -1 or +1.
void Cycle(int id, int step) {
    auto& s = g_settings[id - HotSetting0];
    const int n = int(s.choices.size());
    s.index = (s.index + step + n) % n;
    SaveConfig();
    Redraw();
}

void SavePadMap() {
    if (const std::string map = PadMapString(); map.empty()) g_extra.erase("pad_map");
    else g_extra["pad_map"] = map;
    SaveConfig();
}

// Moves what a controller button does by -1 or +1.
void CycleMap(int id, int step) {
    int& t = g_pad_map[id - HotMap0];
    t = (t + step + kPadTargetCount) % kPadTargetCount;
    SavePadMap();
    Redraw();
}

void Activate(int id) {
    if (id == HotClose || id == HotQuit) {
        PostMessageW(g_hwnd, WM_CLOSE, 0, 0);
    } else if (id == HotMinimize) {
        ShowWindow(g_hwnd, SW_MINIMIZE);
    } else if (id == HotPlay) {
        if (!PreparedReady()) {
            StartPreparation(true);
            return;
        }
        if (!g_cache_ready) {
            StartSession();
            return;
        }
        g_cache_ready = false;
        const std::wstring why = LaunchGame(g_hwnd);
        if (!why.empty()) {
            SetStatus(why);
            return;
        }
        // The launcher closes with the game started: a hidden copy of it waits for the game to
        // exit and promotes the session's shader cache (FinishCache), then ends.
        wchar_t self[MAX_PATH];
        GetModuleFileNameW(nullptr, self, MAX_PATH);
        std::wstring cmd = L"\"" + std::wstring(self) + L"\" --finish \"" +
                           g_session.log_dir.wstring() + L"\" \"" + g_session.cache_work.wstring() +
                           L"\" " + std::to_wstring(GetProcessId(g_game));
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        if (CreateProcessW(self, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | DETACHED_PROCESS,
                           nullptr, g_root.c_str(), &si, &pi)) {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            CloseHandle(g_game);
            g_game = nullptr;
            DestroyWindow(g_hwnd);
            return;
        }
        // No watcher: stay (minimized) and finish the session here when the game exits.
        g_screen = Screen::Running;
        SetStatus(L"Game running. The launcher comes back when it closes.");
        ShowWindow(g_hwnd, SW_MINIMIZE);
    } else if (id == HotSettings) {
        Restart();
        g_screen = Screen::Settings;
        g_focus = HotNone;
        FixFocus();
        SetStatus(L"");
    } else if (id == HotControllers) {
        Restart();
        g_screen = Screen::Controllers;
        g_focus = HotNone;
        FixFocus();
        SetStatus(L"");
    } else if (id == HotMapReset) {
        std::copy(std::begin(kRecommendedPadMap), std::end(kRecommendedPadMap), std::begin(g_pad_map));
        SavePadMap();
        SetStatus(L"Recommended layout restored.");
    } else if (id == HotMapDone) {
        SavePadMap();
        Restart();
        g_screen = Screen::Main;
        g_focus = HotControllers;
        SetStatus(L"");
    } else if (id >= HotMap0 && id < HotMap0 + kPadSourceCount) {
        CycleMap(id, +1);
    } else if (id == HotBack) {
        SaveConfig();
        Restart();
        g_screen = Screen::Main;
        g_focus = HotSettings;
        SetStatus(L"");
    } else if (id == HotBrowse) {
        ChooseGameFolder();
    } else if (id == HotBrowseDlc) {
        ChooseDlcFolder();
        if (g_screen == Screen::Setup) g_focus = HotContinue;
    } else if (id == HotContinue) {
        Restart();
        g_screen = Screen::Main;
        g_focus = HotPlay;
        SetStatus(L"");
        if (!PreparedReady()) StartPreparation(false);
    } else if (id >= HotTab0 && id < HotTab0 + TabCount) {
        if (g_tab != id - HotTab0) Restart();
        g_tab = id - HotTab0;
        g_focus = HotNone;
        FixFocus();
        Redraw();
    } else if (id >= HotSetting0 && id < HotSetting0 + int(g_settings.size())) {
        Cycle(id, +1);
    } else if (id >= HotAction0 && id < HotAction0 + ActCount) {
        switch (id - HotAction0) {
        case ActDlcFolder: ChooseDlcFolder(); break;
        case ActChangeGame: ChooseGameFolder(); break;
        case ActOpenSaves: OpenFolder(PathOption("saves", g_user / "saves")); break;
        case ActOpenLogs: OpenFolder(g_user / "logs"); break;
        case ActClearCache: {
            if (MessageBoxW(g_hwnd, L"Delete the accumulated shader cache? The next sessions compile "
                                    L"the shaders again.", L"Clear shader cache",
                            MB_YESNO | MB_ICONQUESTION) == IDYES) {
                std::error_code ec;
                fs::remove_all(g_user / "shader-cache", ec);
                SetStatus(L"Shader cache cleared.");
            }
            break;
        }
        }
    }
    Redraw();
}

// Navigation shared by the keyboard and the controllers.
enum class Nav { Up, Down, Left, Right, Accept, Back, Play, PrevTab, NextTab };
void Navigate(Nav nav) {
    if (g_screen == Screen::Running || g_screen == Screen::Preparing) return;
    g_hover = HotNone;
    FixFocus();
    const auto order = FocusOrder();
    const auto it = std::find(order.begin(), order.end(), g_focus);
    const size_t at = it == order.end() ? 0 : size_t(it - order.begin());
    const bool on_setting = g_focus >= HotSetting0 && g_focus < HotSetting0 + int(g_settings.size());
    const bool on_map = g_focus >= HotMap0 && g_focus < HotMap0 + kPadSourceCount;
    switch (nav) {
    case Nav::Up:
        if (!order.empty()) g_focus = order[(at + order.size() - 1) % order.size()];
        break;
    case Nav::Down:
        if (!order.empty()) g_focus = order[(at + 1) % order.size()];
        break;
    case Nav::Left:
        if (on_setting) Cycle(g_focus, -1);
        if (on_map) CycleMap(g_focus, -1);
        break;
    case Nav::Right:
        if (on_setting) Cycle(g_focus, +1);
        if (on_map) CycleMap(g_focus, +1);
        break;
    case Nav::Accept: Activate(g_focus); return;
    case Nav::Back:
        if (g_screen == Screen::Settings) Activate(HotBack);
        if (g_screen == Screen::Controllers) Activate(HotMapDone);
        return;
    case Nav::Play:
        if (g_screen == Screen::Main) Activate(HotPlay);
        return;
    case Nav::PrevTab:
    case Nav::NextTab:
        if (g_screen == Screen::Settings)
            Activate(HotTab0 + (g_tab + (nav == Nav::NextTab ? 1 : TabCount - 1)) % TabCount);
        return;
    }
    Redraw();
}

// Controllers (XInput, loaded at run time: xinput1_4 on Windows and Proton).
using XInputGetStateFn = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
XInputGetStateFn g_xinput = nullptr;
void LoadXInput() {
    for (const wchar_t* dll : {L"xinput1_4.dll", L"xinput1_3.dll", L"xinput9_1_0.dll"})
        if (HMODULE m = LoadLibraryW(dll)) {
            g_xinput = reinterpret_cast<XInputGetStateFn>(GetProcAddress(m, "XInputGetState"));
            if (g_xinput) return;
        }
}

void PollPads() {
    if (!g_xinput || g_screen == Screen::Running || GetForegroundWindow() != g_hwnd) return;
    static WORD previous = 0;
    static ULONGLONG repeat_at = 0;
    WORD buttons = 0;
    for (DWORD i = 0; i < 4; ++i) {
        XINPUT_STATE st{};
        if (g_xinput(i, &st) != ERROR_SUCCESS) continue;
        WORD b = st.Gamepad.wButtons;
        constexpr SHORT dead = 16000;
        if (st.Gamepad.sThumbLY > dead) b |= XINPUT_GAMEPAD_DPAD_UP;
        if (st.Gamepad.sThumbLY < -dead) b |= XINPUT_GAMEPAD_DPAD_DOWN;
        if (st.Gamepad.sThumbLX < -dead) b |= XINPUT_GAMEPAD_DPAD_LEFT;
        if (st.Gamepad.sThumbLX > dead) b |= XINPUT_GAMEPAD_DPAD_RIGHT;
        buttons |= b;
    }
    const WORD pressed = buttons & ~previous;
    const ULONGLONG now = GetTickCount64();
    constexpr WORD dirs = XINPUT_GAMEPAD_DPAD_UP | XINPUT_GAMEPAD_DPAD_DOWN | XINPUT_GAMEPAD_DPAD_LEFT |
                          XINPUT_GAMEPAD_DPAD_RIGHT;
    WORD fire = pressed;
    if (buttons & dirs) {
        if (pressed & dirs) repeat_at = now + 400;           // first repeat after 400 ms
        else if (now >= repeat_at) {
            fire |= buttons & dirs;
            repeat_at = now + 120;                            // then every 120 ms
        }
    }
    previous = buttons;
    if (!fire) return;
    if (!g_pad_used) {
        g_pad_used = true;
        Redraw();
    }
    if (fire & XINPUT_GAMEPAD_DPAD_UP) Navigate(Nav::Up);
    if (fire & XINPUT_GAMEPAD_DPAD_DOWN) Navigate(Nav::Down);
    if (fire & XINPUT_GAMEPAD_DPAD_LEFT) Navigate(Nav::Left);
    if (fire & XINPUT_GAMEPAD_DPAD_RIGHT) Navigate(Nav::Right);
    if (fire & XINPUT_GAMEPAD_A) Navigate(Nav::Accept);
    if (fire & XINPUT_GAMEPAD_B) Navigate(Nav::Back);
    if (fire & XINPUT_GAMEPAD_START) Navigate(Nav::Play);
    if (fire & XINPUT_GAMEPAD_LEFT_SHOULDER) Navigate(Nav::PrevTab);
    if (fire & XINPUT_GAMEPAD_RIGHT_SHOULDER) Navigate(Nav::NextTab);
}

void UpdateScale() {
    RECT rc;
    GetClientRect(g_hwnd, &rc);
    g_scale = std::max(0.5f, std::min(rc.right / 960.0f, rc.bottom / 600.0f));
}

void SetFullscreen(bool on) {
    g_fullscreen = on;
    HMONITOR mon = MonitorFromWindow(g_hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi{sizeof(mi)};
    GetMonitorInfoW(mon, &mi);
    if (on) {
        SetWindowLongW(g_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE | WS_MINIMIZEBOX | WS_SYSMENU);
        SetWindowPos(g_hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                     mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
        const float dpi = GetDpiForWindow(g_hwnd) / 96.0f;
        const int w = int(960 * dpi), h = int(600 * dpi);
        const RECT& wa = mi.rcWork;
        SetWindowLongW(g_hwnd, GWL_STYLE, WS_POPUP | WS_THICKFRAME | WS_VISIBLE | WS_MINIMIZEBOX | WS_SYSMENU);
        SetWindowPos(g_hwnd, nullptr, wa.left + (wa.right - wa.left - w) / 2,
                     wa.top + (wa.bottom - wa.top - h) / 2, w, h, SWP_FRAMECHANGED | SWP_NOZORDER);
    }
    UpdateScale();
    Redraw();
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_NCCALCSIZE:
        if (wp) return 0; // no system frame: the client area is the whole window
        break;
    case WM_NCACTIVATE:
        // Activation would repaint the (hidden) system frame: a white edge on Windows 11.
        return DefWindowProcW(hwnd, msg, wp, -1);
    case WM_NCHITTEST: {
        POINT p{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        ScreenToClient(hwnd, &p);
        RECT rc;
        GetClientRect(hwnd, &rc);
        if (!g_fullscreen) {
            const int edge = int(S(6));
            const bool l = p.x < edge, r = p.x >= rc.right - edge, t = p.y < edge, b = p.y >= rc.bottom - edge;
            if (t && l) return HTTOPLEFT;
            if (t && r) return HTTOPRIGHT;
            if (b && l) return HTBOTTOMLEFT;
            if (b && r) return HTBOTTOMRIGHT;
            if (l) return HTLEFT;
            if (r) return HTRIGHT;
            if (b) return HTBOTTOM;
            if (t) return HTTOP;
            if (HitTest(p.x, p.y) == HotNone && p.y < S(30)) return HTCAPTION;
        }
        return HTCLIENT;
    }
    case WM_GETMINMAXINFO: {
        auto* mm = reinterpret_cast<MINMAXINFO*>(lp);
        mm->ptMinTrackSize = {640, 400};
        return 0;
    }
    case WM_SIZE:
        UpdateScale();
        Redraw();
        return 0;
    case WM_TIMER:
        PollPads();
        {
            static unsigned tick = 0;
            ++tick;
            if (!IsIconic(hwnd) && (g_animating || (g_pulsing && (tick & 1)))) Redraw();
        }
        return 0;
    case WM_MOUSEMOVE: {
        const int id = HitTest(GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        if (g_pad_used) {
            g_pad_used = false;
            Redraw();
        }
        if (id != g_hover) {
            g_hover = id;
            if (id != HotNone && id != HotClose && id != HotMinimize && id != HotHelp &&
                !(id >= HotTab0 && id < HotTab0 + TabCount))
                g_focus = id;
            Redraw();
        }
        TRACKMOUSEEVENT t{sizeof(t), TME_LEAVE, hwnd, 0};
        TrackMouseEvent(&t);
        return 0;
    }
    case WM_MOUSELEAVE:
        g_hover = HotNone;
        Redraw();
        return 0;
    case WM_SETCURSOR:
        if (LOWORD(lp) == HTCLIENT) {
            SetCursor(LoadCursor(nullptr, g_hover != HotNone ? IDC_HAND : IDC_ARROW));
            return TRUE;
        }
        break;
    case WM_LBUTTONUP: {
        const int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp), id = HitTest(x, y);
        if (id >= HotSetting0 && id < HotSetting0 + int(g_settings.size())) {
            // The arrow ‹ (left of the value) lowers it; anywhere else on the row raises it.
            for (const auto& s : g_spots)
                if (s.id == id) Cycle(id, x > s.rect.X + s.rect.Width - S(120) &&
                                              x < s.rect.X + s.rect.Width - S(30) &&
                                              x < s.rect.X + s.rect.Width - S(36) - S(40) ? -1 : +1);
            return 0;
        }
        if (id >= HotMap0 && id < HotMap0 + kPadSourceCount) {
            // The arrow ‹ (left of the value) goes back; anywhere else on the row forward.
            for (const auto& s : g_spots)
                if (s.id == id) CycleMap(id, x < s.rect.X + S(185) && x > s.rect.X + S(140) ? -1 : +1);
            return 0;
        }
        if (id != HotNone) Activate(id);
        return 0;
    }
    case WM_RBUTTONUP: {
        const int id = HitTest(GET_X_LPARAM(lp), GET_Y_LPARAM(lp));
        if (id >= HotMap0 && id < HotMap0 + kPadSourceCount) CycleMap(id, -1);
        if (id >= HotSetting0 && id < HotSetting0 + int(g_settings.size())) Cycle(id, -1);
        return 0;
    }
    case WM_KEYDOWN:
        if (g_pad_used) g_pad_used = false;
        switch (wp) {
        case VK_UP: Navigate(Nav::Up); break;
        case VK_DOWN: Navigate(Nav::Down); break;
        case VK_LEFT: Navigate(Nav::Left); break;
        case VK_RIGHT: Navigate(Nav::Right); break;
        case VK_RETURN: case VK_SPACE: Navigate(Nav::Accept); break;
        case VK_ESCAPE: case VK_BACK: Navigate(Nav::Back); break;
        case VK_TAB: Navigate(GetKeyState(VK_SHIFT) < 0 ? Nav::PrevTab : Nav::NextTab); break;
        case VK_F11: SetFullscreen(!g_fullscreen); break;
        }
        return 0;
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT rc;
        GetClientRect(hwnd, &rc);
        Paint(hdc, rc.right, rc.bottom);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_DPICHANGED: {
        const RECT* r = reinterpret_cast<const RECT*>(lp);
        SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_PREP_PROGRESS:
        Redraw();
        return 0;
    case WM_CACHE_READY: {
        std::unique_ptr<fs::path> work(reinterpret_cast<fs::path*>(lp));
        g_session.cache_work = *work;
        g_prep_starting = false;
        g_screen = Screen::Main;
        g_focus = HotPlay;
        g_cache_ready = true;
        Activate(HotPlay);
        return 0;
    }
    case WM_PREP_DONE: {
        std::unique_ptr<std::wstring> result(reinterpret_cast<std::wstring*>(lp));
        Restart();
        g_screen = Screen::Main;
        g_focus = HotPlay;
        if (!result->empty()) {
            SetStatus(*result);
            return 0;
        }
        SetStatus(L"Game files ready.");
        if (g_play_after_prep) Activate(HotPlay);
        return 0;
    }
    case WM_GAME_EXIT: {
        if (g_game) {
            CloseHandle(g_game);
            g_game = nullptr;
        }
        const std::wstring cache = FinishCache(g_session);
        Restart();
        g_screen = Screen::Main;
        g_focus = HotPlay;
        ShowWindow(hwnd, SW_RESTORE);
        SetForegroundWindow(hwnd);
        std::wstring status = wp == 0 ? L"The game closed." : L"The game ended with error " +
                                                                std::to_wstring(DWORD(wp)) +
                                                                L". Logs: user\\logs.";
        if (!cache.empty()) status += L"  " + cache;
        SetStatus(status);
        return 0;
    }
    case WM_CLOSE:
        if (g_game && MessageBoxW(hwnd, L"The game is still running. Close the launcher anyway?",
                                  L"inFAMOUS Second Son", MB_YESNO | MB_ICONQUESTION) != IDYES)
            return 0;
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// A Steam Deck (Steam sets SteamDeck=1) or a screen up to 1280x800: borderless fullscreen.
bool SmallScreen() {
    if (const char* deck = std::getenv("SteamDeck"); deck && deck[0] == '1') return true;
    return GetSystemMetrics(SM_CXSCREEN) <= 1280 && GetSystemMetrics(SM_CYSCREEN) <= 800;
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    GdiplusStartupInput gdi_input;
    ULONG_PTR gdi_token = 0;
    GdiplusStartup(&gdi_token, &gdi_input, nullptr);

    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    g_root = fs::path(self).parent_path();
    g_bin = g_root / "bin";
    g_user = g_root / "user";
    // launcher.exe --finish <log dir> <cache work> <game pid>: the hidden watcher of a session.
    {
        int argc = 0;
        wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (argv && argc == 5 && !wcscmp(argv[1], L"--finish")) {
            g_session.log_dir = argv[2];
            g_session.cache_work = argv[3];
            if (HANDLE game = OpenProcess(SYNCHRONIZE, FALSE, DWORD(_wtoi(argv[4])))) {
                WaitForSingleObject(game, INFINITE);
                CloseHandle(game);
            }
            FinishCache(g_session);
            LocalFree(argv);
            GdiplusShutdown(gdi_token);
            CoUninitialize();
            return 0;
        }
        if (argv && argc == 4 && !wcscmp(argv[1], L"--prepare")) {
            // launcher.exe --prepare <game folder> <out file>: exit code 0 on success; the reason
            // of a failure goes to <out file>.error.
            const std::wstring why = iss_prepare::PrepareGame(argv[2], argv[3], [](float, const wchar_t*) {});
            if (!why.empty()) std::wofstream(std::wstring(argv[3]) + L".error") << why;
            LocalFree(argv);
            GdiplusShutdown(gdi_token);
            CoUninitialize();
            return why.empty() ? 0 : 1;
        }
        if (argv) LocalFree(argv);
    }
    // One launcher at a time: a second start brings the open one to the front.
    HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\IssLauncherSingleInstance");
    if (single && GetLastError() == ERROR_ALREADY_EXISTS) {
        if (HWND other = FindWindowW(L"IssLauncher", nullptr)) {
            ShowWindow(other, SW_RESTORE);
            SetForegroundWindow(other);
        }
        GdiplusShutdown(gdi_token);
        CoUninitialize();
        return 0;
    }
    const bool first_run = !fs::exists(ConfigPath());
    LoadConfig();
    if (first_run && SmallScreen()) g_settings[0].index = 1; // the game fullscreen on a Deck
    LoadFonts();
    LoadXInput();
    if (CheckGameDir(g_game_dir).empty()) {
        LoadArt();
        ApplyExeIcon();
    } else {
        g_screen = Screen::Setup;
    }
    FixFocus();

    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)); // no white flash at open
    wc.lpszClassName = L"IssLauncher";
    wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(WS_EX_APPWINDOW, wc.lpszClassName, L"inFAMOUS Second Son",
                             WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_SYSMENU, CW_USEDEFAULT,
                             CW_USEDEFAULT, 960, 600, nullptr, nullptr, instance, nullptr);
    const MARGINS shadow{1, 1, 1, 1};
    DwmExtendFrameIntoClientArea(g_hwnd, &shadow);
    const DWM_WINDOW_CORNER_PREFERENCE round = DWMWCP_ROUND;
    DwmSetWindowAttribute(g_hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, &round, sizeof(round));
    // Windows 11 draws a 1 px window border (white when active): the launcher's own dark color,
    // and the dark frame mode (older Windows ignore both attributes).
    const BOOL dark = TRUE;
    DwmSetWindowAttribute(g_hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof(dark));
    const COLORREF border = RGB(12, 12, 12);
    DwmSetWindowAttribute(g_hwnd, 34 /* DWMWA_BORDER_COLOR */, &border, sizeof(border));
    ApplyIcon();
    SetFullscreen(SmallScreen());
    
    ShowWindow(g_hwnd, show);
    UpdateWindow(g_hwnd);
    SetTimer(g_hwnd, 1, 16, nullptr); // controller polling

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    g_background.reset();
    g_logo.reset();
    g_oswald.reset();
    g_fonts.reset();
    GdiplusShutdown(gdi_token);
    CoUninitialize();
    return 0;
}
