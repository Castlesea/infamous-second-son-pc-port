// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// ISS_DLSS_UPSCALE=1 with ISS_DLSS=1 (D2; port of axylpup/shadPS4 second-son-v1.1.1
// vk_dlss_upscale, GPL-2.0-or-later): the game renders at its render size (ISS_RESOLUTION, e.g.
// 2560x1440 or 1920x1080); DLSS writes the window size (at most 2x the render size) into a host
// image instead of the T2x resolve's target. That target is cleared to 0 and the HUD draws into it
// also write alpha: the layer ends as HUD colour + transmittance (cleared to 0,0,0,1). At the flip the presenter composites
// HUD over DLSS at that size and its post pass draws it to the window; the game's own display
// buffer is not shown. Off (DLAA in place, D1) while the window is not larger than the render size.
#include <atomic>
#include <cstdlib>
#include <deque>
#include <memory>
#include "common/types.h"
#include "iss_dlss.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
class Instance;
}

namespace iss::dlss {

inline bool UpscaleWanted() {
    static const bool wanted = [] {
        const char* v = std::getenv("ISS_DLSS_UPSCALE");
        return Mode() == 1 && v && v[0] == '1';
    }();
    return wanted;
}

/// The resolve's target (the HUD layer), learned per frame by NoteDraw.
inline VAddr HudLayerAddr() {
    return g_inputs.hud;
}

/// The HUD-layer mode is on for this frame's draws (the presenter decides it each flip).
inline std::atomic<bool> g_hud_layer{false};

inline bool HudLayerActive() {
    return g_hud_layer.load(std::memory_order_relaxed) &&
           g_user_on.load(std::memory_order_relaxed);
}

/// Draws into the HUD layer also write alpha (the layer's transmittance, see the pipeline key).
/// Diagnostic counts per path (0 = UpdateColorBlendingState, 1 = native passes): draws into the
/// HUD layer seen, and those that got alpha; printed by the presenter every ~10 s.
inline std::atomic<u64> g_hud_draws[2]{}, g_hud_alpha[2]{}, g_key_fixed{}, g_key_seen{};
template <typename Masks>
inline void ApplyHudLayerMask(Masks& masks, VAddr cb0_address, int path = 0) {
    if (cb0_address != 0 && cb0_address == HudLayerAddr()) {
        g_hud_draws[path].fetch_add(1, std::memory_order_relaxed);
        if (masks[0] && HudLayerActive()) {
            masks[0] |= vk::ColorComponentFlagBits::eA;
            g_hud_alpha[path].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

struct CompositeJob {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::ImageView hud_view;   ///< the HUD layer (render size), shader-read layout
    vk::ImageView scene_view; ///< DLSS output, shader-read layout
    vk::Image out_image;
    vk::ImageView out_view;
    vk::Extent2D size;
};
void RecordComposite(vk::CommandBuffer cmdbuf, const CompositeJob& job);

class Upscaler {
public:
    explicit Upscaler(const Vulkan::Instance& instance);
    ~Upscaler();

    /// Presenter, each flip: the window size; turns the HUD-layer mode on / off.
    void SetOutputSize(u32 width, u32 height);
    vk::Extent2D OutputSize() const {
        return output_size;
    }
    /// Resolve (command processor): the DLSS output image at the output size (GENERAL while the
    /// evaluate runs). Its barriers go into the recorded closure: ToGeneral before, ToRead after.
    EvalImage SceneTarget();
    static void ToGeneral(vk::CommandBuffer cmdbuf, vk::Image image);
    static void ToRead(vk::CommandBuffer cmdbuf, vk::Image image);
    /// Resolve, the evaluate recorded: the next flip may composite.
    void SceneWritten() {
        ready.store(true, std::memory_order_release);
    }
    bool Ready() const {
        return ready.load(std::memory_order_acquire);
    }
    CompositeJob TakeJob(vk::ImageView hud_view);

private:
    struct Target {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
    };
    void Resize(vk::Extent2D size);

    const Vulkan::Instance& instance;
    vk::Device device;
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    std::unique_ptr<Target> scene; ///< DLSS output, RGBA8 UNORM (storage)
    std::unique_ptr<Target> out;   ///< composite, RGBA16F linear
    vk::Extent2D output_size{};
    vk::Extent2D image_size{};
    std::atomic<bool> ready{};
    u64 flips{};
    std::deque<std::pair<u64, std::unique_ptr<Target>>> retired; ///< freed some flips later
};

/// The one upscaler (created on first use; nullptr when not wanted).
Upscaler* GetUpscaler(const Vulkan::Instance& instance);

/// F9 box / log: the last sizes (0 = not running).
inline std::atomic<u32> g_up_in_w{0}, g_up_in_h{0}, g_up_out_w{0}, g_up_out_h{0};

} // namespace iss::dlss
