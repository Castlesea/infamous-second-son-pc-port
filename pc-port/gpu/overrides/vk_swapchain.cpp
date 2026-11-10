// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "iss_dlss.h"

namespace iss::dlss { extern std::atomic<bool> g_present_stall; } // iss_dlss_sl.cpp

namespace {
// ISS_DLSS=1: a driver frame limiter that paces inside vkQueuePresentKHR (NVIDIA Control Panel
// "Max Frame Rate" with Streamline loaded) costs a whole refresh per frame on top of VSync
// (4 Nov 2026: 60 -> 30 FPS). Seen as presents that block >= 10 ms on average while frames take
// >= 25 ms (below 40 FPS), over 120 presents: the F9 box says what to change, the log says once.
void NotePresentTime(u64 begin_ns, u64 end_ns) {
    static u64 count = 0, block_ns = 0, first_ns = 0;
    static bool logged = false;
    if (!count) first_ns = begin_ns;
    block_ns += end_ns - begin_ns;
    if (++count < 120) return;
    const double block_ms = double(block_ns) / 1e6 / double(count);
    const double frame_ms = double(end_ns - first_ns) / 1e6 / double(count - 1);
    const bool stall = block_ms >= 10.0 && frame_ms >= 25.0;
    if (stall != iss::dlss::g_present_stall.load(std::memory_order_relaxed)) {
        iss::dlss::g_present_stall.store(stall, std::memory_order_relaxed);
        std::fprintf(stderr, "DLSS present stall %s: present %.1f ms, frame %.1f ms\n",
                     stall ? "on" : "off", block_ms, frame_ms);
        if (stall && !logged) {
            logged = true;
            LOG_WARNING(Render_Vulkan,
                        "DLSS: presents block {:.1f} ms each ({:.1f} FPS): a driver frame limiter "
                        "is pacing frames. NVIDIA Control Panel -> Manage 3D settings -> Program "
                        "Settings -> iss-probe.exe -> Max Frame Rate: Off (the launcher's frame "
                        "rate limit does the same without the stall)",
                        block_ms, 1000.0 / frame_ms);
        }
    }
    count = block_ns = 0;
}
u64 PresentNowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}
} // namespace

namespace Vulkan {

static constexpr vk::SurfaceFormatKHR SURFACE_FORMAT_HDR = {
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window_)
    : instance{instance_}, window{window_}, surface{CreateSurface(instance.GetInstance(), window)} {
    FindPresentFormat();
    FindPresentMode();

    Create(window.GetWidth(), window.GetHeight());
    ImGui::Core::Initialize(instance, window, image_count, surface_format.format);
}

Swapchain::~Swapchain() {
    Destroy();
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    Destroy();

    SetSurfaceProperties();

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
        exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const auto format = needs_hdr ? SURFACE_FORMAT_HDR : surface_format;
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices = queue_family_indices.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    auto [swapchain_result, chain] = instance.GetDevice().createSwapchainKHR(swapchain_info);
    ASSERT_MSG(swapchain_result == vk::Result::eSuccess, "Failed to create swapchain: {}",
               vk::to_string(swapchain_result));
    swapchain = chain;

    SetupImages();
    RefreshSemaphores();
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    LOG_DEBUG(Render_Vulkan, "Recreate the swapchain: width={} height={} HDR={}", width_, height_,
              needs_hdr);
    Create(width_, height_);
}

void Swapchain::SetHDR(bool hdr) {
    if (needs_hdr == hdr) {
        return;
    }

    auto result = instance.GetDevice().waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on mode change: {}",
                    vk::to_string(result));
    }

    needs_hdr = hdr;
    Recreate(width, height);
    ImGui::Core::OnSurfaceFormatChange(needs_hdr ? SURFACE_FORMAT_HDR.format
                                                 : surface_format.format);
}

// iss: the present submit shares the one graphics queue with the game's work. Its wait on the
// acquire semaphore holds that queue until the presentation engine gives the image back (up to a
// display refresh), and the game's compute queued behind it waits too (the label the game thread
// spins on: the 26 ms vsync misses at 75 Hz). The present thread waits for the image instead (an
// acquire fence), so the semaphore is already signaled when the submit reaches the queue
// (ISS_PRESENT_ACQUIRE_FENCE=0 off). The image comes back ~10.5 ms after the acquire returns;
// city at 75 Hz, frames >25 ms: 17.9 -> 2.7 a minute, p99 21.9 -> 16.8 ms.
// PRESENT_ACQUIRE lines (every 15 s) give the CPU time spent in each part.
namespace {
struct AcquireStats {
    u64 window_start = 0, count = 0, call_ns = 0, fence_ns = 0, fence_max = 0;
    u64 over1 = 0, over5 = 0, over10 = 0;
};
AcquireStats acquire_stats;
vk::Fence acquire_fence;
u64 NowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}
} // namespace

bool Swapchain::AcquireNextImage() {
    static const bool use_fence = [] {
        const char* p = std::getenv("ISS_PRESENT_ACQUIRE_FENCE");
        return !p || p[0] != '0';
    }();
    vk::Device device = instance.GetDevice();
    if (use_fence && !acquire_fence) {
        auto [fence_result, fence] = device.createFence({});
        ASSERT_MSG(fence_result == vk::Result::eSuccess, "Failed to create the acquire fence");
        acquire_fence = fence;
    }
    const u64 start = NowNs();
    vk::Result result = device.acquireNextImageKHR(
        swapchain, std::numeric_limits<u64>::max(), image_acquired[frame_index],
        use_fence ? acquire_fence : VK_NULL_HANDLE, &image_index);
    const u64 called = NowNs();
    u64 fenced = called;
    if (use_fence &&
        (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR)) {
        const auto wait_result =
            device.waitForFences(acquire_fence, true, std::numeric_limits<u64>::max());
        ASSERT_MSG(wait_result == vk::Result::eSuccess, "Acquire fence wait failed: {}",
                   vk::to_string(wait_result));
        const auto reset_result = device.resetFences(acquire_fence);
        ASSERT_MSG(reset_result == vk::Result::eSuccess, "Acquire fence reset failed");
        fenced = NowNs();
    }
    {
        auto& s = acquire_stats;
        if (!s.window_start) {
            s.window_start = start;
        }
        const u64 waited = fenced - called;
        ++s.count;
        s.call_ns += called - start;
        s.fence_ns += waited;
        s.fence_max = std::max(s.fence_max, waited);
        s.over1 += waited > 1'000'000;
        s.over5 += waited > 5'000'000;
        s.over10 += waited > 10'000'000;
        if (fenced - s.window_start >= 15'000'000'000ull) {
            std::fprintf(stdout,
                         "PRESENT_ACQUIRE fence=%d count=%llu call_ms=%.3f fence_avg_ms=%.3f "
                         "fence_max_ms=%.3f over1=%llu over5=%llu over10=%llu\n",
                         int(use_fence), (unsigned long long)s.count,
                         double(s.call_ns) / 1e6 / double(s.count),
                         double(s.fence_ns) / 1e6 / double(s.count), double(s.fence_max) / 1e6,
                         (unsigned long long)s.over1, (unsigned long long)s.over5,
                         (unsigned long long)s.over10);
            std::fflush(stdout);
            s = AcquireStats{};
            s.window_start = fenced;
        }
    }

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Swapchain acquire returned unknown result {}",
                     vk::to_string(result));
        UNREACHABLE();
        break;
    }

    return !needs_recreation;
}

bool Swapchain::Present() {
    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };

    // ISS_DLSS=1: through Streamline's present proxy (its per-frame bookkeeping), else as usual.
    vk::Result result;
    if (const auto dlss_present = iss::dlss::PresentFunction()) [[unlikely]] {
        const VkPresentInfoKHR raw = present_info;
        const u64 begin = PresentNowNs();
        result = static_cast<vk::Result>(
            dlss_present(static_cast<VkQueue>(instance.GetPresentQueue()), &raw));
        NotePresentTime(begin, PresentNowNs());
    } else {
        result = instance.GetPresentQueue().presentKHR(present_info);
    }
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        needs_recreation = true;
    } else {
        ASSERT_MSG(result == vk::Result::eSuccess, "Swapchain presentation failed: {}",
                   vk::to_string(result));
    }

    frame_index = (frame_index + 1) % image_count;

    return !needs_recreation;
}

void Swapchain::FindPresentFormat() {
    const auto [formats_result, formats] =
        instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);
    ASSERT_MSG(formats_result == vk::Result::eSuccess, "Failed to query surface formats: {}",
               vk::to_string(formats_result));

    // Check if the device supports HDR formats. Here we care of Rec.2020 PQ only as it is expected
    // game output. Other variants as e.g. linear Rec.2020 will require additional color space
    // rotation
    supports_hdr =
        std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& format) {
            return format == SURFACE_FORMAT_HDR;
        }) != formats.end();
    // Also make sure that user allowed us to use HDR
    supports_hdr &= EmulatorSettings.IsHdrAllowed();

    // If there is a single undefined surface format, the device doesn't care, so we'll just use
    // RGBA sRGB.
    if (formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eR8G8B8A8Unorm;
        surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        return;
    }

    // Try to find a suitable format.
    for (const vk::SurfaceFormatKHR& sformat : formats) {
        vk::Format format = sformat.format;
        if (format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eB8G8R8A8Unorm) {
            continue;
        }

        surface_format.format = format;
        surface_format.colorSpace = sformat.colorSpace;
        return;
    }

    UNREACHABLE_MSG("Unable to find required swapchain format!");
}

void Swapchain::FindPresentMode() {
    const auto [modes_result, modes] =
        instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    if (modes_result != vk::Result::eSuccess) {
        LOG_ERROR(Render, "Failed to query available present modes, falling back to Fifo as "
                          "guaranteed supported option.");
        present_mode = vk::PresentModeKHR::eFifo;
        return;
    }

    const auto requested_mode = EmulatorSettings.GetPresentMode();
    if (requested_mode == "Mailbox") {
        present_mode = vk::PresentModeKHR::eMailbox;
    } else if (requested_mode == "Fifo") {
        present_mode = vk::PresentModeKHR::eFifo;
    } else if (requested_mode == "Immediate") {
        present_mode = vk::PresentModeKHR::eImmediate;
    } else {
        LOG_ERROR(Render_Vulkan, "Unknown present mode {}, defaulting to Mailbox.",
                  EmulatorSettings.GetPresentMode());
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    if (std::ranges::find(modes, present_mode) == modes.cend()) {
        // FIFO is guaranteed to be supported by the Vulkan spec.
        constexpr auto fallback = vk::PresentModeKHR::eFifo;
        LOG_WARNING(Render, "Requested present mode {} is not supported, falling back to {}.",
                    vk::to_string(present_mode), vk::to_string(fallback));
        present_mode = fallback;
    }
}

void Swapchain::SetSurfaceProperties() {
    const auto [capabilities_result, capabilities] =
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    ASSERT_MSG(capabilities_result == vk::Result::eSuccess,
               "Failed to query surface capabilities: {}", vk::to_string(capabilities_result));

    extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width, width));
        extent.height = std::max(capabilities.minImageExtent.height,
                                 std::min(capabilities.maxImageExtent.height, height));
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }

    // Opaque is not supported everywhere.
    composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    }
}

void Swapchain::Destroy() {
    vk::Device device = instance.GetDevice();
    const auto wait_result = device.waitIdle();
    if (wait_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to wait for device to become idle: {}",
                    vk::to_string(wait_result));
    }

    for (auto& image_view : images_view) {
        device.destroyImageView(image_view);
    }
    images_view.clear();

    if (swapchain) {
        device.destroySwapchainKHR(swapchain);
    }

    for (const auto& sem : image_acquired) {
        device.destroySemaphore(sem);
    }
    for (const auto& sem : present_ready) {
        device.destroySemaphore(sem);
    }

    image_acquired.clear();
    present_ready.clear();
}

void Swapchain::RefreshSemaphores() {
    const vk::Device device = instance.GetDevice();
    image_acquired.resize(image_count);
    present_ready.resize(image_count);

    for (vk::Semaphore& semaphore : image_acquired) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create image acquired semaphore: {}",
                   vk::to_string(semaphore_result));
        semaphore = sem;
    }
    for (vk::Semaphore& semaphore : present_ready) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create present ready semaphore: {}", vk::to_string(semaphore_result));
        semaphore = sem;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, image_acquired[i], "Swapchain Semaphore: image_acquired {}", i);
        SetObjectName(device, present_ready[i], "Swapchain Semaphore: present_ready {}", i);
    }
}

void Swapchain::SetupImages() {
    vk::Device device = instance.GetDevice();
    auto [images_result, imgs] = device.getSwapchainImagesKHR(swapchain);
    ASSERT_MSG(images_result == vk::Result::eSuccess, "Failed to create swapchain images: {}",
               vk::to_string(images_result));
    images = std::move(imgs);
    image_count = static_cast<u32>(images.size());
    images_view.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        if (images_view[i]) {
            device.destroyImageView(images_view[i]);
        }
        auto [im_view_result, im_view] = device.createImageView(vk::ImageViewCreateInfo{
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = needs_hdr ? SURFACE_FORMAT_HDR.format : surface_format.format,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        });
        ASSERT_MSG(im_view_result == vk::Result::eSuccess, "Failed to create image view: {}",
                   vk::to_string(im_view_result));
        images_view[i] = im_view;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, images[i], "Swapchain Image {}", i);
        SetObjectName(device, images_view[i], "Swapchain ImageView {}", i);
    }
}

} // namespace Vulkan
