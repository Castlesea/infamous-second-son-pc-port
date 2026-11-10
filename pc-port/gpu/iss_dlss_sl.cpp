// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_DLSS=1: DLAA through NVIDIA Streamline with the player's own DLLs (iss_dlss.h). Port of
// axylpup/shadPS4 second-son-v1.1.1 src/video_core/renderer_vulkan/vk_dlss.cpp (GPL-2.0-or-later).
// Only the MIT-licensed Streamline headers (third_party/streamline-sdk) are compiled in; the
// interposer is loaded at run time after its NVIDIA signature is verified.

// sl_security.h defines UNICODE and includes windows.h: it comes first so this file sees one
// setting, and it is included in this one file only (it defines non-inline functions).
#include <sl_security.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "common/logging/log.h"
#include "iss_dlss.h"

#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>

namespace iss::dlss {

/// 30 FPS hunt (2026-11-04): which module a Vulkan function pointer lives in (vk_swapchain.cpp).
void LogFunctionModule(const char* what, const void* fn) {
    HMODULE module{};
    wchar_t path[MAX_PATH]{};
    if (fn && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                     GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                 static_cast<LPCWSTR>(fn), &module)) {
        GetModuleFileNameW(module, path, MAX_PATH);
    }
    std::fprintf(stderr, "VK_FN %s = %p in %ls\n", what, fn, path[0] ? path : L"(unknown)");
}

/// Evaluates done (the overlay's "DLSS on" line; imgui_display.cpp declares it).
std::atomic<unsigned long long> g_overlay_evaluates{0};

/// Present stall (vk_swapchain.cpp sets it, imgui_display.cpp shows it): with Streamline loaded a
/// driver frame limiter (NVIDIA Control Panel "Max Frame Rate") paces inside vkQueuePresentKHR,
/// and with VSync that costs a whole extra refresh per frame (60 -> 30 FPS, 4 Nov 2026).
std::atomic<bool> g_present_stall{false};

namespace {

// ISS_DLSS_STAGE (test, 2026-11-04: 30 FPS with Streamline loaded, even with DLSS off):
// 1 = slInit + instance extensions only, 2 = + device extensions, 3 (default) = everything.
int Stage() {
    static const int stage = [] {
        const char* v = std::getenv("ISS_DLSS_STAGE");
        const int s = v && v[0] >= '1' && v[0] <= '3' ? v[0] - '0' : 3;
        std::fprintf(stderr, "DLSS stage %d (1 = slInit only, 2 = + device extensions, 3 = all)\n", s);
        return s;
    }();
    return stage;
}
// The Windows timer resolution (100 ns units) around slInit: Streamline changes it.
using PFN_NtQueryTimerResolution = LONG(NTAPI*)(PULONG, PULONG, PULONG);
using PFN_NtSetTimerResolution = LONG(NTAPI*)(ULONG, BOOLEAN, PULONG);
ULONG TimerResolution() {
    static const auto query = reinterpret_cast<PFN_NtQueryTimerResolution>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryTimerResolution"));
    ULONG lo = 0, hi = 0, cur = 0;
    if (query) query(&lo, &hi, &cur);
    return cur;
}

struct Api {
    HMODULE module{};
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slIsFeatureSupported* is_feature_supported{};
    PFun_slGetFeatureRequirements* get_feature_requirements{};
    PFun_slSetVulkanInfo* set_vulkan_info{};
    PFun_slGetFeatureFunction* get_feature_function{};
    PFun_slGetNewFrameToken* get_new_frame_token{};
    PFun_slSetConstants* set_constants{};
    PFun_slEvaluateFeature* evaluate_feature{};
    PFun_slFreeResources* free_resources{};
    PFN_vkGetDeviceProcAddr get_device_proc_addr{};
    PFun_slDLSSSetOptions* dlss_set_options{};
    PFun_slDLSSGetOptimalSettings* dlss_get_optimal{};
    PFN_vkQueuePresentKHR present{};
};

Api g_api;
bool g_initialized = false; // slInit succeeded
bool g_device_ok = false;   // slSetVulkanInfo succeeded and the DLSS functions exist
std::vector<const char*> g_device_extensions;
u32 g_options_preset = ~0u; // the preset the options were last set with
u32 g_options_width = 0;     // ... and the output size
u32 g_options_height = 0;
u32 g_options_in_w = 0; // ... and the input size (a live screen percentage changes it)
u32 g_options_in_h = 0;
std::chrono::steady_clock::time_point g_last_eval{};
u64 g_evaluated = 0;
u64 g_refused_key = 0; ///< input / output sizes no mode takes (asked once)
u64 g_failed = 0;

// A fixed id for this project (Streamline uses it with engineVersion instead of an NVIDIA id).
constexpr const char* ProjectId = "7c1e5d3a-2b9f-4e6a-8d41-5f0a9c3b7e12";
constexpr const char* EngineVersion = "castlesea-second-son-1";

std::filesystem::path ExeDir() {
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return {};
        }
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    return std::filesystem::path{buf}.parent_path();
}

void LogMessage(sl::LogType type, const char* msg) {
    std::string text{msg ? msg : ""};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    if (type == sl::LogType::eError) {
        LOG_ERROR(Render_Vulkan, "Streamline: {}", text);
    } else if (type == sl::LogType::eWarn) {
        LOG_WARNING(Render_Vulkan, "Streamline: {}", text);
    } else {
        LOG_INFO(Render_Vulkan, "Streamline: {}", text);
    }
}

template <typename T>
bool Resolve(T*& out, const char* name) {
    out = reinterpret_cast<T*>(GetProcAddress(g_api.module, name));
    if (!out) {
        LOG_WARNING(Render_Vulkan, "DLSS: sl.interposer.dll has no {}", name);
    }
    return out != nullptr;
}

bool Startup() {
    static bool tried = false;
    if (tried) {
        return g_initialized;
    }
    tried = true;
    if (Mode() != 1) {
        return false;
    }
    // bin\dlss next to iss-probe.exe, else dlss in the game folder (one level up).
    std::error_code ec;
    // ISS_DLSS_DIR=<name> (test): another folder name than dlss, in the same places.
    const char* dir_name = std::getenv("ISS_DLSS_DIR");
    const std::string name = dir_name && dir_name[0] ? dir_name : "dlss";
    auto dir = ExeDir() / name;
    if (!std::filesystem::is_regular_file(dir / "sl.interposer.dll", ec)) {
        dir = ExeDir().parent_path() / name;
    }
    const auto interposer = dir / "sl.interposer.dll";
    if (!std::filesystem::is_regular_file(interposer, ec)) {
        LOG_WARNING(Render_Vulkan,
                    "DLSS: {} not found - the game's own anti-aliasing stays (put NVIDIA's "
                    "sl.interposer.dll, sl.common.dll, sl.dlss.dll and nvngx_dlss.dll there)",
                    interposer.string());
        return false;
    }
    // 1.1.1 (GitHub #19, error 193 on an RTX 5080): the Streamline release has three zips with the
    // same file names - streamline-sdk-v2.14.16.zip (x64) and -aarch64 / -arm64ec (ARM PCs) - and the
    // repository keeps its DLLs in Git LFS, so a "Source code" zip holds small text placeholders.
    // The log and the Home notice say which one a player has. sl.interposer.dll of 2.14.16 x64 is
    // 652,928 bytes.
    const u64 interposer_bytes = std::filesystem::file_size(interposer, ec);
    const auto wrong_files = [&](const char* what) {
        LOG_ERROR(Render_Vulkan,
                  "DLSS: {} ({} bytes) is {} - download streamline-sdk-v2.14.16.zip (not the "
                  "-aarch64 / -arm64ec zips, not \"Source code\") and copy the five files from its "
                  "bin\\x64 folder (sl.interposer.dll 652,928 bytes)",
                  interposer.string(), interposer_bytes, what);
    };
    if (!sl::security::verifyEmbeddedSignature(interposer.wstring().c_str())) {
        LOG_ERROR(Render_Vulkan, "DLSS: {} has no valid NVIDIA signature - not loaded",
                  interposer.string());
        if (interposer_bytes < 4096) {
            wrong_files("a small text placeholder (the Git LFS pointer of a \"Source code\" zip)");
        } else {
        }
        return false;
    }
    g_api.module = LoadLibraryW(interposer.wstring().c_str());
    if (!g_api.module) {
        const DWORD error = GetLastError();
        LOG_ERROR(Render_Vulkan, "DLSS: LoadLibrary({}) failed ({})", interposer.string(), error);
        if (error == ERROR_BAD_EXE_FORMAT) {
            wrong_files("not a Windows x64 DLL (error 193: the ARM build of the files)");
        } else {
        }
        return false;
    }
    bool ok = Resolve(g_api.init, "slInit") && Resolve(g_api.shutdown, "slShutdown") &&
              Resolve(g_api.is_feature_supported, "slIsFeatureSupported") &&
              Resolve(g_api.get_feature_requirements, "slGetFeatureRequirements") &&
              Resolve(g_api.set_vulkan_info, "slSetVulkanInfo") &&
              Resolve(g_api.get_feature_function, "slGetFeatureFunction") &&
              Resolve(g_api.get_new_frame_token, "slGetNewFrameToken") &&
              Resolve(g_api.set_constants, "slSetConstants") &&
              Resolve(g_api.evaluate_feature, "slEvaluateFeature") &&
              Resolve(g_api.free_resources, "slFreeResources");
    g_api.get_device_proc_addr =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(GetProcAddress(g_api.module, "vkGetDeviceProcAddr"));
    ok = ok && g_api.get_device_proc_addr != nullptr;
    if (!ok) {
        return false;
    }

    static const std::wstring plugin_dir = dir.wstring();
    static const wchar_t* plugin_paths[] = {plugin_dir.c_str()};
    static const sl::Feature features[] = {sl::kFeatureDLSS};
    sl::Preferences prefs{};
    prefs.showConsole = false;
    // SHADPS4_DLSS_VERBOSE=1: Streamline's verbose log (start-up diagnosis).
    const char* verbose = std::getenv("ISS_DLSS_VERBOSE");
    prefs.logLevel = verbose && verbose[0] == '1' ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    prefs.pathsToPlugins = plugin_paths;
    prefs.numPathsToPlugins = 1;
    prefs.pathToLogsAndData = nullptr; // messages go to our log through the callback
    prefs.logMessageCallback = &LogMessage;
    // No over-the-air downloads: exactly the files the player put in dlss\ are used.
    prefs.flags = sl::PreferenceFlags::eUseManualHooking |
                  sl::PreferenceFlags::eUseFrameBasedResourceTagging |
                  sl::PreferenceFlags::eDisableCLStateTracking;
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = 1;
    prefs.applicationId = 0;
    prefs.engine = sl::EngineType::eCustom;
    prefs.engineVersion = EngineVersion;
    prefs.projectId = ProjectId;
    prefs.renderAPI = sl::RenderAPI::eVulkan;
    // ISS_DLSS_NO_INIT=1 (test): the interposer is loaded and verified, slInit is not called.
    if (const char* v = std::getenv("ISS_DLSS_NO_INIT"); v && v[0] == '1') {
        std::fprintf(stderr, "DLSS test: interposer loaded, slInit NOT called (ISS_DLSS_NO_INIT=1)\n");
        return false;
    }
    const ULONG timer_before = TimerResolution();
    const sl::Result init = g_api.init(prefs, sl::kSDKVersion);
    const ULONG timer_after = TimerResolution();
    std::fprintf(stderr, "DLSS timer resolution: %lu before slInit, %lu after (100 ns units)\n",
                 timer_before, timer_after);
    // ISS_DLSS_RESTORE_TIMER=1 (test): ask for the resolution the process had before slInit.
    if (const char* v = std::getenv("ISS_DLSS_RESTORE_TIMER"); v && v[0] == '1') {
        static const auto set = reinterpret_cast<PFN_NtSetTimerResolution>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetTimerResolution"));
        ULONG now = 0;
        if (set) set(timer_before, TRUE, &now);
        std::fprintf(stderr, "DLSS timer resolution restored: now %lu\n", TimerResolution());
    }
    if (init != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "DLSS: slInit failed ({})", static_cast<int>(init));
        return false;
    }
    g_initialized = true;
    LOG_INFO(Render_Vulkan, "DLSS: Streamline loaded from {}", dir.string());
    return true;
}

} // namespace

void AddInstanceExtensions(std::vector<const char*>& extensions) {
    if (!Startup()) {
        return;
    }
    sl::FeatureRequirements req{};
    const sl::Result r = g_api.get_feature_requirements(sl::kFeatureDLSS, req);
    if (r != sl::Result::eOk || req.vkNumDeviceExtensions == 0) {
        LOG_WARNING(Render_Vulkan, "DLSS: not available on this system (requirements {}, {} device "
                                   "extensions) - the game's own anti-aliasing stays",
                    static_cast<int>(r), req.vkNumDeviceExtensions);
        return;
    }
    // ISS_DLSS_NO_INSTANCE_EXT=1 (test): Streamline's instance extensions are not added.
    const char* no_ext = std::getenv("ISS_DLSS_NO_INSTANCE_EXT");
    for (u32 i = 0; i < req.vkNumInstanceExtensions; ++i) {
        const char* ext = req.vkInstanceExtensions[i];
        std::fprintf(stderr, "DLSS instance extension %s%s\n", ext,
                     no_ext && no_ext[0] == '1' ? " (NOT added, test)" : "");
        if (no_ext && no_ext[0] == '1') continue;
        if (std::find_if(extensions.begin(), extensions.end(), [ext](const char* e) {
                return std::strcmp(e, ext) == 0;
            }) == extensions.end()) {
            extensions.push_back(ext);
        }
    }
    for (u32 i = 0; i < req.vkNumDeviceExtensions; ++i) {
        g_device_extensions.push_back(req.vkDeviceExtensions[i]);
    }
    LOG_INFO(Render_Vulkan, "DLSS: {} instance / {} device extensions requested",
             req.vkNumInstanceExtensions, req.vkNumDeviceExtensions);
}

bool DeviceSupported(vk::PhysicalDevice physical_device) {
    if (!g_initialized || g_device_extensions.empty() || Stage() < 2) {
        return false;
    }
    const auto chain =
        physical_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
    auto id = chain.get<vk::PhysicalDeviceIDProperties>();
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
    if (id.deviceLUIDValid) {
        adapter.deviceLUID = id.deviceLUID.data();
        adapter.deviceLUIDSizeInBytes = VK_LUID_SIZE;
    }
    const sl::Result r = g_api.is_feature_supported(sl::kFeatureDLSS, adapter);
    if (r != sl::Result::eOk) {
        LOG_WARNING(Render_Vulkan, "DLSS: not supported on this GPU / driver ({}) - the game's own "
                                   "anti-aliasing stays",
                    static_cast<int>(r));
        return false;
    }
    return true;
}

const std::vector<const char*>& DeviceExtensions() {
    return g_device_extensions;
}

void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device,
                   u32 graphics_family, u32 graphics_index) {
    if (!g_initialized || Stage() < 3) {
        return;
    }
    sl::VulkanInfo info{};
    info.device = static_cast<VkDevice>(device);
    info.instance = static_cast<VkInstance>(instance);
    info.physicalDevice = static_cast<VkPhysicalDevice>(physical_device);
    info.graphicsQueueFamily = graphics_family;
    info.graphicsQueueIndex = graphics_index;
    info.computeQueueFamily = graphics_family;
    info.computeQueueIndex = graphics_index;
    LOG_INFO(Render_Vulkan, "DLSS: slSetVulkanInfo (graphics family {} queue {})", graphics_family,
             graphics_index);
    const sl::Result r = g_api.set_vulkan_info(info);
    LOG_INFO(Render_Vulkan, "DLSS: slSetVulkanInfo returned {}", static_cast<int>(r));
    if (r != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "DLSS: slSetVulkanInfo failed ({})", static_cast<int>(r));
        return;
    }
    void* fn = nullptr;
    if (g_api.get_feature_function(sl::kFeatureDLSS, "slDLSSSetOptions", fn) != sl::Result::eOk ||
        !fn) {
        LOG_ERROR(Render_Vulkan, "DLSS: the DLSS plugin did not start (nvngx_dlss.dll missing?) - "
                                 "the game's own anti-aliasing stays");
        return;
    }
    g_api.dlss_set_options = reinterpret_cast<PFun_slDLSSSetOptions*>(fn);
    fn = nullptr;
    if (g_api.get_feature_function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", fn) ==
        sl::Result::eOk) {
        g_api.dlss_get_optimal = reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(fn);
    }
    g_api.present = reinterpret_cast<PFN_vkQueuePresentKHR>(
        g_api.get_device_proc_addr(static_cast<VkDevice>(device), "vkQueuePresentKHR"));
    if (!g_api.present) {
        LOG_ERROR(Render_Vulkan, "DLSS: no present proxy - DLSS stays off");
        return;
    }
    g_device_ok = true;
    g_ready.store(true, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "DLSS: ready (DLAA at the game's render size; Home = on / off)");
    std::fprintf(stderr, "DLSS ready: DLAA at the game's render size (Home = on / off)\n");
}

PFN_vkQueuePresentKHR PresentFunction() {
    // ISS_DLSS_PRESENT_PROXY=0 (test, 2026-11-04: frames capped to 30 with Streamline loaded):
    // present as usual, not through Streamline's proxy.
    static const bool proxy = [] {
        const char* v = std::getenv("ISS_DLSS_PRESENT_PROXY");
        const bool on = !v || v[0] != '0';
        std::fprintf(stderr, "DLSS present proxy: %s\n", on ? "on" : "off (ISS_DLSS_PRESENT_PROXY=0)");
        return on;
    }();
    return g_device_ok && proxy ? g_api.present : nullptr;
}

void Shutdown() {
    if (!g_initialized) {
        return;
    }
    g_ready.store(false, std::memory_order_relaxed);
    if (g_device_ok && g_evaluated) {
        g_api.free_resources(sl::kFeatureDLSS, sl::ViewportHandle{0u});
    }
    g_api.shutdown();
    g_initialized = false;
    g_device_ok = false;
    LOG_INFO(Render_Vulkan, "DLSS: shut down after {} evaluates ({} failed)", g_evaluated, g_failed);
}

bool Evaluate(vk::CommandBuffer cmdbuf, u64 frame, const EvalImage& color, const EvalImage& motion,
              const EvalImage& depth, const EvalImage& output, float jitter_x, float jitter_y,
              bool reset) {
    if (!g_device_ok) {
        return false;
    }
    const sl::ViewportHandle viewport{0u};
    const u32 preset = g_preset.load(std::memory_order_relaxed); // 0 = default (K for DLAA)
    if (preset != g_options_preset || output.width != g_options_width ||
        output.height != g_options_height || color.width != g_options_in_w ||
        color.height != g_options_in_h) {
        sl::DLSSOptions options{};
        options.mode = sl::DLSSMode::eDLAA;
        options.outputWidth = output.width;
        options.outputHeight = output.height;
        const u64 key = (u64(color.width) << 48) | (u64(color.height) << 32) |
                        (u64(output.width) << 16) | u64(output.height);
        if (key == g_refused_key) {
            return false;
        }
        if (output.width != color.width || output.height != color.height) {
            // D2 (ISS_DLSS_UPSCALE): the first mode whose render range holds the input (AxylPup
            // vk_dlss.cpp); a mode the driver refuses is skipped; none = not run.
            const sl::DLSSMode modes[] = {sl::DLSSMode::eMaxQuality, sl::DLSSMode::eUltraQuality,
                                          sl::DLSSMode::eBalanced, sl::DLSSMode::eMaxPerformance,
                                          sl::DLSSMode::eUltraPerformance};
            bool fits = false;
            sl::DLSSMode best = sl::DLSSMode::eOff;
            u32 best_gap = ~0u;
            for (const sl::DLSSMode m : modes) {
                options.mode = m;
                sl::DLSSOptimalSettings opt{};
                if (!g_api.dlss_get_optimal ||
                    g_api.dlss_get_optimal(options, opt) != sl::Result::eOk) {
                    continue;
                }
                LOG_INFO(Render_Vulkan, "DLSS: mode {} for {}x{}: render {}x{} (min {}x{}, max {}x{})",
                         static_cast<u32>(m), output.width, output.height, opt.optimalRenderWidth,
                         opt.optimalRenderHeight, opt.renderWidthMin, opt.renderHeightMin,
                         opt.renderWidthMax, opt.renderHeightMax);
                if (opt.renderWidthMin <= color.width && color.width <= opt.renderWidthMax &&
                    opt.renderHeightMin <= color.height && color.height <= opt.renderHeightMax) {
                    // The mode whose own render size is nearest (the launcher's quality choice).
                    const u32 gap = opt.optimalRenderWidth > color.width
                                        ? opt.optimalRenderWidth - color.width
                                        : color.width - opt.optimalRenderWidth;
                    if (gap < best_gap) {
                        best_gap = gap;
                        best = m;
                    }
                    fits = true;
                }
            }
            options.mode = best;
            if (!fits) {
                g_refused_key = key;
                LOG_WARNING(Render_Vulkan, "DLSS: no mode takes {}x{} to {}x{} - not run",
                            color.width, color.height, output.width, output.height);
                std::fprintf(stderr, "DLSS: no mode takes %ux%u to %ux%u - not run\n", color.width,
                             color.height, output.width, output.height);
                ++g_failed;
                return false;
            }
            std::fprintf(stderr, "DLSS upscale mode %u: %ux%u -> %ux%u\n",
                         static_cast<u32>(options.mode), color.width, color.height, output.width,
                         output.height);
        }
        options.qualityPreset = static_cast<sl::DLSSPreset>(preset);
        options.ultraQualityPreset = static_cast<sl::DLSSPreset>(preset);
        options.balancedPreset = static_cast<sl::DLSSPreset>(preset);
        options.performancePreset = static_cast<sl::DLSSPreset>(preset);
        options.ultraPerformancePreset = static_cast<sl::DLSSPreset>(preset);
        options.colorBuffersHDR = sl::Boolean::eFalse; // the tonemapped LDR frame
        options.preExposure = 1.0f;
        options.exposureScale = 1.0f;
        options.useAutoExposure = sl::Boolean::eFalse;
        options.dlaaPreset = static_cast<sl::DLSSPreset>(preset);
        options.alphaUpscalingEnabled = sl::Boolean::eFalse;
        const sl::Result r = g_api.dlss_set_options(viewport, options);
        if (r != sl::Result::eOk) {
            ++g_failed;
            LOG_ERROR(Render_Vulkan, "DLSS: slDLSSSetOptions failed ({})", static_cast<int>(r));
            return false;
        }
        g_options_preset = preset;
        g_options_width = output.width;
        g_options_height = output.height;
        g_options_in_w = color.width;
        g_options_in_h = color.height;
        LOG_INFO(Render_Vulkan, "DLSS: options set: mode {} {}x{} -> {}x{}, preset {}",
                 static_cast<u32>(options.mode), color.width, color.height, output.width,
                 output.height, preset);
        reset = true;
    }

    // A long gap (loading screen, menu, DLSS just switched on) starts a new history.
    const auto now = std::chrono::steady_clock::now();
    if (g_evaluated == 0 || now - g_last_eval > std::chrono::milliseconds{250}) {
        reset = true;
    }
    g_last_eval = now;

    sl::FrameToken* token = nullptr;
    const u32 frame_index = static_cast<u32>(frame);
    if (g_api.get_new_frame_token(token, &frame_index) != sl::Result::eOk || !token) {
        ++g_failed;
        return false;
    }

    sl::Constants c{};
    sl::float4x4 identity{};
    for (u32 r = 0; r < 4; ++r) {
        identity.row[r] = sl::float4(r == 0 ? 1.0f : 0.0f, r == 1 ? 1.0f : 0.0f,
                                     r == 2 ? 1.0f : 0.0f, r == 3 ? 1.0f : 0.0f);
    }
    // Not read by DLSS SR (checklist sec. 4); sane values so the plugin's checks stay quiet.
    c.cameraViewToClip = identity;
    c.clipToCameraView = identity;
    c.clipToLensClip = identity;
    c.clipToPrevClip = identity;
    c.prevClipToClip = identity;
    c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    c.cameraPos = sl::float3(0.0f, 0.0f, 0.0f);
    c.cameraUp = sl::float3(0.0f, 1.0f, 0.0f);
    c.cameraRight = sl::float3(1.0f, 0.0f, 0.0f);
    c.cameraFwd = sl::float3(0.0f, 0.0f, 1.0f);
    c.cameraNear = 0.1f;
    c.cameraFar = 10000.0f;
    c.cameraFOV = 1.0f;
    c.cameraAspectRatio = static_cast<float>(output.width) / static_cast<float>(output.height);
    c.jitterOffset = sl::float2(jitter_x, jitter_y);
    // The game's vectors are screen UV, current minus previous (the resolve samples history at
    // uv - mv): DLSS wants the vector to the previous position, normalised -> scale -1.
    const bool mv_positive = [] {
        const char* v = std::getenv("ISS_DLSS_MV");
        return v && v[0] == '+';
    }();
    c.mvecScale = mv_positive ? sl::float2(1.0f, 1.0f) : sl::float2(-1.0f, -1.0f);
    c.depthInverted = sl::Boolean::eTrue; // reversed Z (GreaterEqual)
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse; // measured: the vectors carry no jitter
    c.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    if (g_api.set_constants(c, *token, viewport) != sl::Result::eOk) {
        ++g_failed;
        return false;
    }

    const auto make = [](const EvalImage& img) {
        sl::Resource res{sl::ResourceType::eTex2d, static_cast<void*>(static_cast<VkImage>(img.image)),
                         nullptr, static_cast<void*>(static_cast<VkImageView>(img.view)),
                         static_cast<u32>(static_cast<VkImageLayout>(img.layout))};
        res.width = img.width;
        res.height = img.height;
        res.nativeFormat = static_cast<u32>(static_cast<VkFormat>(img.format));
        res.mipLevels = 1;
        res.arrayLayers = 1;
        res.flags = 0;
        res.usage = static_cast<u32>(static_cast<VkImageUsageFlags>(img.usage));
        return res;
    };
    sl::Resource res_color = make(color);
    sl::Resource res_motion = make(motion);
    sl::Resource res_depth = make(depth);
    sl::Resource res_output = make(output);
    sl::SubresourceRange depth_range{};
    depth_range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth_range.baseMipLevel = 0;
    depth_range.levelCount = 1;
    depth_range.baseArrayLayer = 0;
    depth_range.layerCount = 1;
    res_depth.next = &depth_range;

    sl::ResourceTag tag_depth{&res_depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate};
    sl::ResourceTag tag_motion{&res_motion, sl::kBufferTypeMotionVectors,
                               sl::ResourceLifecycle::eValidUntilEvaluate};
    sl::ResourceTag tag_color{&res_color, sl::kBufferTypeScalingInputColor,
                              sl::ResourceLifecycle::eValidUntilEvaluate};
    sl::ResourceTag tag_output{&res_output, sl::kBufferTypeScalingOutputColor,
                               sl::ResourceLifecycle::eValidUntilEvaluate};
    const sl::BaseStructure* inputs[] = {&viewport, &tag_depth, &tag_motion, &tag_color, &tag_output};
    const sl::Result r = g_api.evaluate_feature(
        sl::kFeatureDLSS, *token, inputs, static_cast<u32>(std::size(inputs)),
        reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(cmdbuf)));
    if (r != sl::Result::eOk) {
        if (++g_failed <= 8) {
            LOG_ERROR(Render_Vulkan, "DLSS: slEvaluateFeature failed ({})", static_cast<int>(r));
        }
        return false;
    }
    ++g_evaluated;
    g_overlay_evaluates.fetch_add(1, std::memory_order_relaxed);
    return true;
}

} // namespace iss::dlss
