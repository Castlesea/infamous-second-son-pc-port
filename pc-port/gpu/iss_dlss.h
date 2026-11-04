// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// ISS_DLSS (port of axylpup/shadPS4 second-son-v1.1.1 SHADPS4_DLSS, GPL-2.0-or-later).
// Second Son's SMAA T2x temporal resolve (fs_a6d23116) reads exactly DLSS's inputs and writes the
// anti-aliased frame the HUD is drawn onto. Its inputs are read every frame from the draws that use
// them, on the command processor thread (NoteDraw):
//  - SMAA edges fs_abc3c768 samples the tonemap output (its RGBA8 texture) with the scene depth;
//  - the resolve fs_a6d23116 samples the motion vectors (its RG16F texture; its size = the game's
//    render size) and writes the HUD layer (its colour target);
//  - SMAA blend weights fs_961c93ba: its CB float 12 is the T2x subsample index (the jitter phase:
//    1 or 2 = +-0.25 px diagonally, 0 = T2x off).
// D0 (ISS_DLSS_PROBE=1): only finds and logs the inputs at the resolve; rendering is unchanged.
// D1 (ISS_DLSS=1): the resolve becomes DLAA through NVIDIA Streamline (iss_dlss_sl.cpp) at the
//   game's render size, written into the resolve's own target (the HUD is drawn onto it after).
//   The player's own signed NVIDIA DLLs are loaded from a dlss folder (next to iss-probe.exe, or
//   one level up); only the MIT Streamline headers are compiled in. Missing / unsigned DLLs, a GPU
//   without DLSS or any failure: the game's own resolve runs. Home = DLSS on / off (live A/B).
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace iss::dlss {

constexpr u64 ResolvePsHash = 0xa6d23116; ///< T2x temporal resolve
constexpr u64 WeightsPsHash = 0x961c93ba; ///< SMAA blend weights (jitter phase)
constexpr u64 EdgesPsHash = 0xabc3c768;   ///< SMAA edges (colour input + depth)

inline bool ProbeEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("ISS_DLSS_PROBE");
        return v && v[0] == '1';
    }();
    return on;
}
/// ISS_DLSS: 1 = DLAA through Streamline, 0 / unset = off.
inline int Mode() {
    static const int mode = [] {
        const char* v = std::getenv("ISS_DLSS");
        return v && v[0] == '1' ? 1 : 0;
    }();
    return mode;
}
inline bool Enabled() {
    return ProbeEnabled() || Mode() != 0;
}
/// Streamline is up and DLSS exists on this device.
inline std::atomic<bool> g_ready{false};
/// Home key: the player's on / off (the game's own resolve runs while off).
inline std::atomic<bool> g_user_on{true};
/// ISS_DLSS_JITTER: jitter sign combination (bit 0 flips x, bit 1 flips y); 3 = AxylPup's pick.
inline std::atomic<u32> g_jitter_combo{[] {
    const char* v = std::getenv("ISS_DLSS_JITTER");
    return v ? static_cast<u32>(std::atoi(v)) & 3u : 3u;
}()};
/// ISS_DLSS_PRESET: the network (sl::DLSSPreset number; 0 = the DLL's default).
inline std::atomic<u32> g_preset{[] {
    const char* v = std::getenv("ISS_DLSS_PRESET");
    return v ? static_cast<u32>(std::atoi(v)) : 0u;
}()};

/// A tapped pixel shader: where it lives and how its resources are fetched from its user data.
/// Written by the pipeline lookup, read by NoteDraw: both on the command processor thread.
struct Tap {
    std::atomic<VAddr> ps{0};
    Shader::ImageResourceList images;
    Shader::BufferResourceList buffers;
    Shader::PFN_SrtWalker walker{};
    u32 flat_dw{16};
};
inline Tap g_resolve, g_weights, g_edges;

/// What the frame's draws told (NoteDraw), read at the resolve.
struct Inputs {
    VAddr color{}, depth{}, motion{}, hud{};
    u32 game_w{}, game_h{};
    u32 color_w{}, color_h{};
    u8 phase{}; ///< 1 / 2 = subsample index, 3 = T2x off, 0 = unknown
    bool edges_seen{}, weights_seen{};
};
inline Inputs g_inputs;

inline void NoteTap(Tap& tap, const Shader::Info& info, VAddr base) {
    if (tap.ps.load(std::memory_order_relaxed) == base) [[likely]] {
        return;
    }
    tap.images = info.images;
    tap.buffers = info.buffers;
    tap.walker = info.srt_info.walker_func;
    tap.flat_dw = std::max<u32>(info.srt_info.flattened_bufsize_dw, 16u);
    tap.ps.store(base, std::memory_order_release);
}

/// PipelineCache::GetProgram (both returns): remembers where the three shaders live.
inline void NoteProgram(bool fragment, u64 hash, VAddr base, const Shader::Info& info) {
    if (!fragment || !Enabled()) [[likely]] {
        return;
    }
    if (hash == ResolvePsHash) {
        NoteTap(g_resolve, info, base);
    } else if (hash == WeightsPsHash) {
        NoteTap(g_weights, info, base);
    } else if (hash == EdgesPsHash) {
        NoteTap(g_edges, info, base);
    }
}

/// The view GetSharp needs (flattened user data + program base).
struct FetchView {
    std::vector<u32> flattened_ud_buf;
    VAddr pgm_base{};
};

inline const FetchView& Flatten(const Tap& tap, const AmdGpu::Regs& regs, VAddr base) {
    thread_local FetchView view;
    const u32* ud = regs.ps_program.user_data.data();
    view.flattened_ud_buf.assign(tap.flat_dw, 0u);
    std::memcpy(view.flattened_ud_buf.data(), ud, 16 * sizeof(u32));
    if (tap.walker) {
        tap.walker(ud, view.flattened_ud_buf.data());
    }
    view.pgm_base = base;
    return view;
}

/// Every draw (FilterDraw): one address compare unless it is one of the three.
inline void NoteDraw(const AmdGpu::Regs& regs) {
    if (!Enabled()) [[likely]] {
        return;
    }
    const VAddr ps = regs.ps_program.Address<VAddr>();
    if (ps == 0) {
        return;
    }
    if (ps == g_edges.ps.load(std::memory_order_acquire)) {
        const auto& view = Flatten(g_edges, regs, ps);
        for (const auto& res : g_edges.images) {
            const AmdGpu::Image image = res.GetSharp(view);
            if (image.Valid() && image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8) {
                g_inputs.color = image.Address();
                g_inputs.color_w = static_cast<u32>(image.width) + 1;
                g_inputs.color_h = static_cast<u32>(image.height) + 1;
                break;
            }
        }
        g_inputs.depth = regs.depth_buffer.DepthAddress();
        g_inputs.edges_seen = true;
    } else if (ps == g_resolve.ps.load(std::memory_order_acquire)) {
        const auto& view = Flatten(g_resolve, regs, ps);
        for (const auto& res : g_resolve.images) {
            const AmdGpu::Image image = res.GetSharp(view);
            if (image.Valid() && image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
                image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
                g_inputs.motion = image.Address();
                g_inputs.game_w = static_cast<u32>(image.width) + 1;
                g_inputs.game_h = static_cast<u32>(image.height) + 1;
                break;
            }
        }
        g_inputs.hud = regs.color_buffers[0].Address();
    } else if (ps == g_weights.ps.load(std::memory_order_acquire)) {
        const auto& view = Flatten(g_weights, regs, ps);
        g_inputs.phase = 0;
        for (const auto& res : g_weights.buffers) {
            if (res.IsSpecial()) {
                continue;
            }
            const AmdGpu::Buffer cb = res.GetSharp(view);
            if (cb.base_address == 0 || cb.GetSize() < 60) {
                continue;
            }
            float f;
            std::memcpy(&f, reinterpret_cast<const void*>(cb.base_address + 48), sizeof(f));
            g_inputs.phase = (f > 0.5f && f < 1.5f)    ? 1
                             : (f > 1.5f && f < 2.5f)  ? 2
                             : (f > -0.5f && f < 0.5f) ? 3
                                                       : 0;
            break;
        }
        g_inputs.weights_seen = true;
    }
}

/// FilterDraw: is this draw the resolve (the tap is known)?
inline bool IsResolve(const AmdGpu::Regs& regs) {
    const VAddr a = g_resolve.ps.load(std::memory_order_relaxed);
    return a != 0 && regs.ps_program.Address<VAddr>() == a;
}

/// FilterDraw: replace this draw (the resolve) with DLSS now?
inline bool ReplaceResolve(const AmdGpu::Regs& regs) {
    return Mode() == 1 && g_ready.load(std::memory_order_relaxed) &&
           g_user_on.load(std::memory_order_relaxed) && g_inputs.edges_seen && IsResolve(regs);
}

// ---- Streamline (iss_dlss_sl.cpp) ----

/// vk_platform GetInstanceExtensions: loads + verifies the interposer, slInit, and appends the
/// instance extensions DLSS needs. Before vkCreateInstance (Streamline's order).
void AddInstanceExtensions(std::vector<const char*>& extensions);
/// vk_instance, device creation: slIsFeatureSupported for this GPU; false = DLSS stays off.
bool DeviceSupported(vk::PhysicalDevice physical_device);
/// The device extensions DLSS asked for (valid until Shutdown).
const std::vector<const char*>& DeviceExtensions();
/// vk_instance, after the device and its queues exist: slSetVulkanInfo, present proxy, DLSS
/// functions; sets g_ready on success.
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device,
                   u32 graphics_family, u32 graphics_index);
/// vk_swapchain: the present to use (Streamline's proxy); null = present as usual.
PFN_vkQueuePresentKHR PresentFunction();
/// Instance destruction (GPU idle): slShutdown before the device goes.
void Shutdown();

struct EvalImage {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::ImageLayout layout;
    vk::ImageUsageFlags usage;
    u32 width;
    u32 height;
};
/// One DLAA evaluate into `cmdbuf` (outside rendering; the images already in `layout`). jitter in
/// render pixels. False = not run (the caller falls back to the game's frame).
bool Evaluate(vk::CommandBuffer cmdbuf, u64 frame, const EvalImage& color, const EvalImage& motion,
              const EvalImage& depth, const EvalImage& output, float jitter_x, float jitter_y,
              bool reset);

} // namespace iss::dlss
