// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "gpu_blackbox.h"
#include <cstdio>
#include <cstdlib>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_stats.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

// iss: the image collector's bookkeeping for its periodic TEXTURE_GC line: the bytes of the
// registered images (guest sizes) and the images the collector deleted that the game created
// again (same address, size and format) within RecreateWindow gc ticks: textures it should
// have kept.
namespace {
constexpr u64 RecreateWindow = 1800;
struct GcStats {
    std::mutex mutex;
    std::unordered_map<u64, u64> deleted; // image key -> gc tick of its deletion by the collector
    u64 recreated = 0, recreated_bytes = 0, deleted_bytes = 0;
    std::atomic<u64> texture_bytes{0}, images{0};
    u64 ticks_trigger = 0, ticks_pressured = 0, ticks_critical = 0;
} g_gc;

u64 GcImageKey(const ImageInfo& info) {
    const u64 words[] = {info.guest_address, info.guest_size, u64(info.pixel_format),
                         (u64(info.size.width) << 32) | info.size.height};
    return XXH3_64bits(words, sizeof(words));
}

// ISS_GC_HEADROOM_MB=trigger:pressure:critical: the collector starts, presses and turns
// aggressive when the free room left in the Vulkan heap budget (VK_EXT_memory_budget, queried
// each tick) falls below these. Default 1536:1024:512 on every GPU; 0 turns it off (the
// thresholds from the start-up budget). On the Steam Deck the start-up
// budget was max(budget - usage - 8 GB, 2 GB) = 2 GB: trigger 0, critical 1.5 GB, against a
// device usage of 7.8 GB (both heaps: 1 GB VRAM + 8 GB GTT, the imported unified heap included),
// so every tick ran the aggressive pass and deleted up to 60 images idle for 80 ticks (86 000 in
// a 13-minute session, 2026-10-21 16:48), which the game then created and uploaded again.
// On an RTX 4060 (8 GB) the start-up thresholds were trigger 3217 / pressure 2515 / critical
// 5031 MB of a 6289 MB budget: at the 4.85 GB the game uses every tick ran the pressured pass,
// ~625 000 textures deleted in 31 minutes (2026-10-22). Discrete GPUs count the device-local
// heaps only (Instance::valid_heaps); the driver's budget already leaves out other processes,
// so even the critical threshold (512 MB below it) keeps the process inside its VRAM share.
struct Headroom {
    bool enabled = false;
    u64 trigger = 0, pressure = 0, critical = 0;
};
Headroom ParseHeadroom() {
    Headroom h{};
    const char* p = std::getenv("ISS_GC_HEADROOM_MB");
    if (!p) {
        h = {true, 1536_MB, 1024_MB, 512_MB};
        return h;
    }
    unsigned long long t = 0, pr = 0, c = 0;
    if (std::sscanf(p, "%llu:%llu:%llu", &t, &pr, &c) == 3 && t != 0)
        h = {true, u64(t) << 20, u64(pr) << 20, u64(c) << 20};
    return h;
}
} // namespace

void TextureCache::AssertNoAscTextureAccessSlow() const {
    if (liverpool && liverpool->CpSplitActive() && liverpool->IsCurrentCpThread()) {
        ASSERT_MSG(liverpool->CurrentQueue() == 0,
                   "ASC attempted direct G-owned TextureCache lookup/mutation");
    }
}

iss::g4::NegativeImageSnapshot TextureCache::ProbeNoRegisteredImage(VAddr address,
                                                                     size_t size) const noexcept {
    return negative_image_index.Probe(address, size, Traits::PAGE_BITS, NumImagePages,
                                      [this](u64 page) {
        return page_image_counts[page].load(std::memory_order_acquire);
    });
}

bool TextureCache::NegativeImageSnapshotCurrent(
    const iss::g4::NegativeImageSnapshot& snapshot) const noexcept {
    return negative_image_index.StillCurrent(snapshot);
}

bool TextureCache::WithNegativeSnapshotsLocked(
    std::span<const iss::g4::NegativeImageSnapshot> snapshots,
    const std::function<bool()>& publish) {
    std::scoped_lock lock{negative_index_writer_mutex};
    for (const auto& snapshot : snapshots) {
        if (!negative_image_index.StillCurrent(snapshot)) return false;
    }
    return publish();
}

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()},
      texture_lookup_memo_enabled{[] {
          const char* value = std::getenv("ISS_TEXTURE_LOOKUP_MEMO");
          return value && value[0] == '1';
      }()},
      texture_lookup_stats_enabled{[] {
          const char* value = std::getenv("ISS_TEXTURE_LOOKUP_MEMO_STATS");
          return value && value[0] == '1';
      }()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    const s64 min_pressure_floor =
        std::clamp<s64>(device_local_memory / 4, 256_MB, DEFAULT_PRESSURE_GC_MEMORY);
    const s64 min_critical_floor =
        std::clamp<s64>(device_local_memory / 2, 512_MB, DEFAULT_CRITICAL_GC_MEMORY);
    pressure_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      min_pressure_floor));
    critical_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      min_critical_floor));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    // ISS: the collector starts at 60% of the budget (at most the budget less 3 GB), not at
    // (budget - 8 GB) / 2. With a 14.4 GB budget it started at 3.1 GB while the game used 4.3:
    // every frame it deleted the textures not drawn for 16 frames (~250 a second, anything
    // behind the camera), and turning around created and uploaded them again (the +12 ms
    // spikes of 2026-10-17). Smaller cards keep a trigger below their budget.
    // ISS_GC_TRIGGER_MB sets it.
    const s64 relaxed = std::min<s64>(device_local_memory * 6 / 10, device_local_memory - 3_GB);
    trigger_gc_memory = std::max<u64>(trigger_gc_memory, u64(std::max<s64>(relaxed, 0)));
    if (const char* p = std::getenv("ISS_GC_TRIGGER_MB"))
        trigger_gc_memory = u64(std::strtoull(p, nullptr, 10)) << 20;
    std::fprintf(stderr, "TEXTURE_GC budget=%lld MB trigger=%llu MB pressure=%llu MB critical=%llu MB\n",
                 (long long)(device_local_memory >> 20), (unsigned long long)(trigger_gc_memory >> 20),
                 (unsigned long long)(pressure_gc_memory >> 20),
                 (unsigned long long)(critical_gc_memory >> 20));
    if (const Headroom h = ParseHeadroom(); h.enabled)
        std::fprintf(stderr, "TEXTURE_GC headroom trigger=%llu MB pressure=%llu MB critical=%llu MB "
                             "below the heap budget of each tick\n",
                     (unsigned long long)(h.trigger >> 20), (unsigned long long)(h.pressure >> 20),
                     (unsigned long long)(h.critical >> 20));
}

TextureCache::~TextureCache() {
    if (texture_lookup_memo_enabled && texture_lookup_stats_enabled) ReportLookupMemo();
}

void TextureCache::ProcessDownloadImages() {
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

bool TextureCache::DumpImageToFile(VAddr address, std::string_view prefix, u32 width,
                                   u32 height) {
    ImageId image_id{};
    {
        std::scoped_lock lock{mutex};
        // Among the images starting at the address (else those containing it): the one of the
        // given size, or the one used most recently (several images alias an address over a
        // frame, and old ones linger).
        u64 best_tick = 0;
        const auto consider = [&](ImageId id, Image& image) {
            if (width != 0 &&
                (image.info.size.width != width || (height != 0 && image.info.size.height != height))) {
                return;
            }
            if (!image_id || image.tick_accessed_last > best_tick) {
                image_id = id;
                best_tick = image.tick_accessed_last;
            }
        };
        if (const auto it = image_address_index.find(address); it != image_address_index.end()) {
            for (const ImageId id : it->second) {
                consider(id, slot_images[id]);
            }
        }
        if (!image_id) {
            ForEachImageInRegion(address, 1, [&](ImageId id, Image& image) {
                if (image.info.guest_address <= address &&
                    address < image.info.guest_address + image.info.guest_size) {
                    consider(id, image);
                }
            });
        }
    }
    if (!image_id) {
        LOG_INFO(Render_Vulkan, "DUMP image {:x}: no image", address);
        return false;
    }
    Image& image = slot_images[image_id];
    const u32 row_bytes = image.info.pitch * (image.info.num_bits / 8);
    const u32 download_size =
        row_bytes * image.info.size.height * image.info.size.depth * image.info.resources.layers;
    if (download_size == 0 || download_size > (256u << 20)) {
        return false;
    }
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, false);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    scheduler.Finish();
    download.Invalidate();
    using namespace Common::FS;
    const auto dir = GetUserPath(PathType::LogDir) / "xray";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto name = fmt::format("{}_{:x}_{}x{}_p{}_{}_{}.raw", prefix, image.info.guest_address,
                                  image.info.size.width, image.info.size.height, image.info.pitch,
                                  image.info.num_bits, vk::to_string(image.info.pixel_format));
    const auto file = IOFile{dir / name, FileAccessMode::Create};
    file.WriteRaw<u8>(download.mapped, download_size);
    LOG_INFO(Render_Vulkan,
             "DUMP image {:x} written: {} ({} bytes, {} layers, gpu modified {}, last used tick {})",
             address, name, download_size, image.info.resources.layers,
             True(image.flags & ImageFlagBits::GpuModified), image.tick_accessed_last);
    return true;
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, !sync);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
        VideoCore::NoteHostWrite();
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                download.Invalidate();
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download.mapped, download_size);
                VideoCore::NoteHostWrite();
                runtime.GetStagingPool().FreeDeferred(download);
            });
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

bool TextureCache::HasImagesInRange(VAddr addr, size_t size) const {
    const u64 first_page = addr >> Traits::PAGE_BITS;
    const u64 last_page = (addr + size - 1) >> Traits::PAGE_BITS;
    if (last_page >= NumImagePages) {
        return true;
    }
    for (u64 page = first_page; page <= last_page; ++page) {
        if (page_image_counts[page].load(std::memory_order_acquire) != 0) {
            return true;
        }
    }
    return false;
}

static std::atomic<u64> tex_streak_images{};

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    // Every guest write fault comes here, mostly for pages without images: answered without the
    // lock the command processor takes for every texture it binds. An image registered after
    // this check protects its pages after registering, so its writes fault again.
    if (!HasImagesInRange(pages_start, pages_end - pages_start)) {
        Common::Perf::Add(Common::Perf::TexInvalidateSkips);
        return;
    }
    Common::Perf::Add(Common::Perf::TexInvalidateLocked);
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        // Every branch changes the image (dirty, or pages untracked): its next binding checks it.
        BumpImageGeneration(image_id);
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
    // ISS_TEX_FAULT_STREAK_KB (default 256, 0 off): a guest thread writing on through image
    // pages it faults on one after the other gets the images of the next window marked maybe
    // dirty now (hashed, untracked: re-uploaded at their next use only if their bytes changed),
    // as their faults would do one by one. Every ~2 s the game copies ~11 MB over a pool of
    // small textures at 0x2093xxxxx-0x209Fxxxxx: ~2850 faults, one per image, a 45 ms frame
    // (the "mini frame skips", 2026-10-17).
    static const u64 streak_bytes = [] {
        const char* setting = std::getenv("ISS_TEX_FAULT_STREAK_KB");
        const u64 kb = setting ? std::strtoull(setting, nullptr, 10) : 256;
        return std::min<u64>(kb, 4096) * 1024;
    }();
    if (size <= 8 && streak_bytes != 0) {
        thread_local u64 last_page = 0, next_page = 0, streak = 0;
        const u64 page = addr >> 12; // 4 KB pages
        if (page == last_page + 1 || (next_page != 0 && page == next_page)) ++streak;
        else if (page != last_page) streak = 0;
        last_page = page;
        if (streak >= 3) {
            const VAddr ahead = pages_end;
            const VAddr ahead_end = ahead + streak_bytes;
            ImageIds marked;
            ForEachImageInRegion(ahead, streak_bytes, [&](ImageId image_id, Image& image) {
                if (image.info.guest_address >= ahead &&
                    image.info.guest_address + image.info.guest_size <= ahead_end &&
                    image.IsTracked())
                    marked.push_back(image_id);
            });
            for (const ImageId image_id : marked) {
                BumpImageGeneration(image_id);
                MarkAsMaybeDirty(image_id, slot_images[image_id]);
            }
            next_page = ahead_end >> 12;
            tex_streak_images.fetch_add(marked.size(), std::memory_order_relaxed);
        }
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    AssertNoAscTextureAccess();
    if (max_size == 0 || !HasImagesInRange(address, max_size)) {
        Common::Perf::Add(Common::Perf::TexInvalidateSkips);
        return;
    }
    Common::Perf::Add(Common::Perf::TexInvalidateLocked);
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
        BumpImageGeneration(image_id);
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    AssertNoAscTextureAccess();
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    AssertNoAscTextureAccess();
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    BumpGeneration();
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    AssertNoAscTextureAccess();
    const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    runtime.CopyImage(&src_image, &new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
        BumpGeneration();
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::ShrinkImage(const ImageInfo& info, ImageId image_id) {
    AssertNoAscTextureAccess();
    const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    // The content of the common region moves to the new image (the padding is dropped).
    runtime.CopyImage(&src_image, &new_image);
    new_image.flags |= ImageFlagBits::GpuModified;
    new_image.flags &= ~ImageFlagBits::CpuDirty;

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
        BumpGeneration();
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

AmdGpu::CbDbExtent TextureCache::LearnedTargetExtent(VAddr address, u32 pitch) const {
    AssertNoAscTextureAccess();
    if (!Common::Perf::GetTuning().rt_size_learn.load(std::memory_order_relaxed)) {
        return {};
    }
    const auto it = learned_extents.find(address);
    if (it == learned_extents.end() || it->second.pitch != pitch) {
        return {};
    }
    return it->second.extent;
}

ImageId TextureCache::LearnTargetSize(ImageId image_id, const ImageInfo& tsharp_info) {
    AssertNoAscTextureAccess();
    if (!image_id || !Common::Perf::GetTuning().rt_size_learn.load(std::memory_order_relaxed)) {
        return image_id;
    }
    const Image& image = slot_images[image_id];
    const auto& info = image.info;
    if (!info.props.size_from_pitch || info.guest_address != tsharp_info.guest_address ||
        info.pitch != tsharp_info.pitch || info.resources.levels != 1 ||
        tsharp_info.resources.levels != 1 || info.resources.layers != 1 ||
        tsharp_info.resources.layers != 1 || tsharp_info.type != AmdGpu::ImageType::Color2D ||
        info.num_samples != tsharp_info.num_samples || info.num_bits != tsharp_info.num_bits) {
        return image_id;
    }
    const u32 width = tsharp_info.size.width;
    const u32 height = tsharp_info.size.height;
    if (width == 0 || height == 0 || width > info.size.width || height > info.size.height ||
        (width == info.size.width && height == info.size.height) || width > 0xFFFF ||
        height > 0xFFFF) {
        return image_id;
    }
    AmdGpu::CbDbExtent extent{};
    extent.width = static_cast<u16>(width);
    extent.height = static_cast<u16>(height);
    learned_extents[info.guest_address] = LearnedExtent{.pitch = info.pitch, .extent = extent};
    LOG_INFO(Render_Vulkan,
             "Render target {:x} (pitch {}, {}x{}) shrunk to its texture size {}x{} (format {})",
             info.guest_address, info.pitch, info.size.width, info.size.height, width, height,
             vk::to_string(info.pixel_format));
    ImageInfo new_info = info;
    new_info.size.width = width;
    new_info.size.height = height;
    new_info.props.size_from_pitch = false;
    new_info.mips_layout[0].height = height;
    return ShrinkImage(new_info, image_id);
}

std::optional<TextureCache::PreparedTextureToken> TextureCache::CapturePreparedTexture(
    const ImageDesc& desc, ImageId expected_id, const ImageViewInfo& resolved_view,
    vk::ImageView expected_view) {
    AssertNoAscTextureAccess();
    enum ProofReject : size_t {
        InvalidInput, Depth, SizeFromPitch, NoAddressIndex, Selection, Lifetime, Dirty,
        Tracking, Binding, SubresourceState, LayoutAccess, ViewRemap, ViewMissing,
        GenerationRace, NumRejects
    };
    // This API is called only by opt-in native material preparation/validation.
    // Counters are CP-thread-local; log no addresses, shaders or resource contents.
    static thread_local std::array<u64, NumRejects> rejects{};
    static thread_local u64 failures{};
    const auto reject = [&](ProofReject reason) -> std::optional<PreparedTextureToken> {
        ++rejects[reason];
        if (++failures == 1 || (failures & 16383) == 0) {
            static constexpr const char* names[]{
                "invalid_input", "depth", "size_from_pitch", "no_address_index", "selection",
                "lifetime", "dirty", "tracking", "binding", "subresource_state", "layout_access",
                "view_remap", "view_missing", "generation_race"};
            std::fprintf(stderr, "NATIVE_TEXTURE_PROOF failures=%llu", static_cast<unsigned long long>(failures));
            for (size_t i = 0; i < NumRejects; ++i)
                std::fprintf(stderr, " %s=%llu", names[i], static_cast<unsigned long long>(rejects[i]));
            std::fputc('\n', stderr);
        }
        return std::nullopt;
    };
    if (!expected_id || !expected_view || desc.type != BindingType::Texture ||
        !desc.info.guest_address || !desc.info.guest_size) return reject(InvalidInput);
    if (desc.info.props.is_depth) return reject(Depth);
    if (desc.info.props.size_from_pitch) return reject(SizeFromPitch);
    std::scoped_lock lock{mutex};
    // Same last-perfect-match choice as FindImage. Lookup live registered IDs before
    // dereferencing the old token: a SlotVector index alone does not prove lifetime.
    ImageId selected{};
    const auto found = image_address_index.find(desc.info.guest_address);
    if (found == image_address_index.end()) return reject(NoAddressIndex);
    for (const auto id : found->second) {
        const auto& info = slot_images[id].info;
        if (info.guest_address == desc.info.guest_address &&
            info.guest_size == desc.info.guest_size && info.size == desc.info.size &&
            IsVulkanFormatCompatible(info.pixel_format, desc.info.pixel_format) &&
            (info.type == desc.info.type || desc.info.size == Extent3D{1, 1, 1})) selected = id;
    }
    if (selected != expected_id) return reject(Selection);
    const u32 image_generation = ImageGeneration(selected);
    const auto& image = slot_images[selected];
    if (!image.backing) return reject(Lifetime);
    if (image.depth_id || image.info.props.is_depth) return reject(Depth);
    if (image.info.props.size_from_pitch) return reject(SizeFromPitch);
    if (image.info.resources < desc.info.resources) return reject(InvalidInput);
    if (True(image.flags & ImageFlagBits::Dirty)) return reject(Dirty);
    if (False(image.flags & ImageFlagBits::Registered)) return reject(Lifetime);
    if (image.binding.needs_rebind || image.binding.is_target || image.binding.force_general)
        return reject(Binding);
    if (image.track_addr != image.info.guest_address ||
        image.track_addr_end != image.info.guest_address + image.info.guest_size)
        return reject(Tracking);
    const auto& backing = *image.backing;
    // A uniform exact state makes GetBarriers a no-op, including stage tracking.
    if (!backing.subresource_states.empty()) return reject(SubresourceState);
    if (backing.state.layout != vk::ImageLayout::eShaderReadOnlyOptimal ||
        backing.state.access_mask != vk::AccessFlagBits2::eShaderRead ||
        backing.state.pl_stage != vk::PipelineStageFlagBits2::eAllCommands)
        return reject(LayoutAccess);
    // Any remapped view requires the original overlap/subresource resolver.
    if (resolved_view != desc.view_info) return reject(ViewRemap);
    const auto it = std::ranges::find(backing.image_view_infos, resolved_view);
    if (it == backing.image_view_infos.end()) return reject(ViewMissing);
    const auto view_id = backing.image_view_ids[std::distance(backing.image_view_infos.begin(), it)];
    if (!view_id || *slot_image_views[view_id].image_view != expected_view) return reject(ViewMissing);
    if (image_generation != ImageGeneration(selected)) return reject(GenerationRace);
    return PreparedTextureToken{.desc = desc, .resolved_view = resolved_view,
        .image_id = selected, .view_id = view_id,
        .image_uid = image.image_uid, .image_generation = image_generation,
        .backing = image.backing, .view = expected_view, .layout = backing.state.layout};
}

bool TextureCache::ValidatePreparedTexture(const PreparedTextureToken& token) {
    const auto current = CapturePreparedTexture(token.desc, token.image_id,
                                                token.resolved_view, token.view);
    return current && current->image_uid == token.image_uid &&
           current->image_generation == token.image_generation &&
           current->backing == token.backing && current->view_id == token.view_id &&
           current->layout == token.layout;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    AssertNoAscTextureAccess();
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};
    const auto is_perfect_match = [&](const Image& cache_image) {
        if (cache_image.info.guest_address != info.guest_address) {
            return false;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            return false;
        }
        if (cache_image.info.size != info.size) {
            return false;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            return false;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            return false;
        }
        return true;
    };

    ImageId image_id{};

    // Fast path: a perfect match starts at the same address, look it up directly instead of
    // scanning every image overlapping the range. Same pick as the scan below: the last perfect
    // match in registration order.
    if (const auto it = image_address_index.find(info.guest_address);
        it != image_address_index.end()) {
        for (const ImageId cache_id : it->second) {
            if (is_perfect_match(slot_images[cache_id])) {
                image_id = cache_id;
            }
        }
    }

    ImageIds image_ids;
    if (!image_id) {
        ForEachImageInRegion(
            info.guest_address, info.guest_size,
            [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

        // Check for a perfect match first
        for (const auto& cache_id : image_ids) {
            if (is_perfect_match(slot_images[cache_id])) {
                image_id = cache_id;
            }
        }
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id && desc.type == BindingType::Texture) {
        // A texture descriptor of a render target created without size hint: the real size.
        image_id = LearnTargetSize(image_id, info);
    }
    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.insert(instance, runtime, slot_image_views, info);
        RegisterImage(image_id);
        if (info.props.is_tiled) {
            // Tiled layouts decoded differently since the upstream detiling fix (#5196): 128 bit
            // elements in display micro tiling, macro tiled images with micro tiled small mips,
            // thick arrays. Logged once per created image, with the block coded arrays (smoke
            // flipbooks), to tell which game textures the fix changes.
            const auto micro_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
            const u32 thickness = AmdGpu::GetMicroTileThickness(info.array_mode);
            const bool display128 =
                micro_mode == AmdGpu::MicroTileMode::Display && info.num_bits == 128;
            const bool micro_mips = info.micro_tiled_mips != 0;
            const bool thick_array = thickness > 1 && info.resources.layers > 1;
            if (display128 || micro_mips || thick_array ||
                (info.props.is_block && info.resources.layers > 1)) {
                LOG_INFO(Render_Vulkan,
                         "TEXLAYOUT {:x}+{:x} {}x{}x{} layers {} mips {} {} bits {} tile {} "
                         "array {} micro {} micro_mips {:x} changed by #5196: {}{}{}",
                         info.guest_address, info.guest_size, info.size.width, info.size.height,
                         info.size.depth, info.resources.layers, info.resources.levels,
                         vk::to_string(info.pixel_format), info.num_bits,
                         static_cast<u32>(info.tile_mode), static_cast<u32>(info.array_mode),
                         static_cast<u32>(micro_mode), info.micro_tiled_mips,
                         display128 ? "display128 " : "", micro_mips ? "small_mips " : "",
                         thick_array ? "thick_array" : "");
            }
        }
        if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
            LOG_INFO(Render_Vulkan, "DUMP new image {:x}+{:x} {}x{}x{} {} samples {} mips {}",
                     info.guest_address, info.guest_size, info.size.width, info.size.height,
                     info.size.depth, vk::to_string(info.pixel_format), info.num_samples,
                     info.resources.levels);
        }
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    AssertNoAscTextureAccess();
    if (size == 0) {
        return {};
    }
    // Only images starting exactly at the address are candidates: use the address index
    // instead of scanning every image overlapping the range (hot path, once per bound texel
    // buffer and draw).
    ImageIds image_ids;
    if (const auto it = image_address_index.find(address); it != image_address_index.end()) {
        for (const ImageId image_id : it->second) {
            Image& image = slot_images[image_id];
            if (ensure_valid && !image.SafeToDownload()) {
                continue;
            }
            image_ids.push_back(image_id);
        }
    }
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

void TextureCache::ReportLookupMemo() const {
    std::fprintf(stderr,
        "TEXTURE_LOOKUP_MEMO gc_tick=%llu meta=%llu/%llu view=%llu/%llu stencil=%llu/%llu\n",
        static_cast<unsigned long long>(gc_tick),
        static_cast<unsigned long long>(lookup_counts.meta_hits),
        static_cast<unsigned long long>(lookup_counts.meta_queries),
        static_cast<unsigned long long>(lookup_counts.view_hits),
        static_cast<unsigned long long>(lookup_counts.view_queries),
        static_cast<unsigned long long>(lookup_counts.stencil_hits),
        static_cast<unsigned long long>(lookup_counts.stencil_queries));
}

void TextureCache::RegisterSurfaceMeta(VAddr address, const MetaDataInfo& info) {
    // Value-only IsMeta observations cannot outlive a structural mutation. Never
    // cache clear_mask: ConsumeMetaClear/TouchMeta/ClearMeta keep their live lookup.
    if (texture_lookup_memo_enabled) surface_meta_revision.fetch_add(1, std::memory_order_acq_rel);
    surface_metas.emplace(address, info);
    if (texture_lookup_memo_enabled) surface_meta_revision.fetch_add(1, std::memory_order_release);
}

void TextureCache::EraseSurfaceMeta(VAddr address) {
    if (texture_lookup_memo_enabled) surface_meta_revision.fetch_add(1, std::memory_order_acq_rel);
    surface_metas.erase(address);
    if (texture_lookup_memo_enabled) surface_meta_revision.fetch_add(1, std::memory_order_release);
}

ImageView& TextureCache::FindViewMemoized(ImageId image_id, const ImageViewInfo& info,
                                        bool ensure_guest_samples) {
    Image& image = slot_images[image_id];
    if (!texture_lookup_memo_enabled) return image.FindView(info, ensure_guest_samples);
    ++lookup_counts.view_queries;
    // Image::FindView can convert between guest/host MSAA backing. A hit must
    // never bypass that conversion or return a view from the previous backing.
    const bool requires_samples = ensure_guest_samples &&
        ((image.backing->num_samples > 1) != (image.info.num_samples > 1));
    const ViewLookupKey key{image.image_uid, Generation(), ImageGeneration(image_id),
                           scheduler.CurrentTick(), image.backing, info};
    if (!requires_samples) {
        if (const auto* cached = view_lookup_memo.Find(image_id.index, key);
            cached && slot_image_views.is_allocated(*cached)) {
            ++lookup_counts.view_hits;
            return slot_image_views[*cached];
        }
    }
    ImageView& result = image.FindView(info, ensure_guest_samples);
    const auto& infos = image.backing->image_view_infos;
    const auto found = std::ranges::find(infos, info);
    if (found != infos.end()) {
        const ViewLookupKey resolved{image.image_uid, Generation(), ImageGeneration(image_id),
                                     scheduler.CurrentTick(), image.backing, info};
        view_lookup_memo.Store(image_id.index, resolved,
            image.backing->image_view_ids[std::distance(infos.begin(), found)]);
    }
    return result;
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    AssertNoAscTextureAccess();
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return FindViewMemoized(image_id, desc.view_info, true);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    AssertNoAscTextureAccess();
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8)) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer (once: the image remembers it)
    if (desc.info.meta_info.cmask_addr &&
        image.info.meta_info.cmask_addr != desc.info.meta_info.cmask_addr) {
        RegisterSurfaceMeta(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr &&
        image.info.meta_info.fmask_addr != desc.info.meta_info.fmask_addr) {
        RegisterSurfaceMeta(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return FindViewMemoized(image_id, desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    AssertNoAscTextureAccess();
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer (once: the image remembers it)
    if (desc.info.meta_info.htile_addr &&
        image.info.meta_info.htile_addr != desc.info.meta_info.htile_addr) {
        RegisterSurfaceMeta(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        // The registry (page table, slots) is shared with the guest threads' fault and unmap
        // handlers: without the lock an unmap could collect an image twice and free it twice.
        std::scoped_lock lock{mutex};
        // Keep identity by value: inserting a stencil can grow slot_images and
        // invalidate Image references. FindViewMemoized reacquires the depth slot.
        const auto depth_uid = image.image_uid;
        const StencilLookupKey key{depth_uid, Generation(), scheduler.CurrentTick(),
                                   desc.info.stencil_addr, desc.info.stencil_size};
        ImageId stencil_id{};
        if (texture_lookup_memo_enabled) {
            ++lookup_counts.stencil_queries;
            if (const auto* cached = stencil_lookup_memo.Find(image_id.index, key);
                cached && slot_images.is_allocated(cached->id)) {
                const auto& stencil = slot_images[cached->id];
                if (stencil.image_uid == cached->uid &&
                    True(stencil.flags & ImageFlagBits::Registered)) {
                    stencil_id = cached->id;
                    ++lookup_counts.stencil_hits;
                }
            }
        }
        if (!stencil_id) ForEachImageInRegion(
            desc.info.stencil_addr, desc.info.stencil_size, [&](ImageId image_id, Image& image) {
                if (image.info.guest_address != desc.info.stencil_addr) {
                    return;
                }
                if (image.info.pixel_format == vk::Format::eUndefined ||
                    Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                    stencil_id = image_id;
                }
            });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id = slot_images.insert(instance, runtime, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, depth_uid);
        if (texture_lookup_memo_enabled) {
            const StencilLookupKey resolved{depth_uid, Generation(), scheduler.CurrentTick(),
                                            desc.info.stencil_addr, desc.info.stencil_size};
            stencil_lookup_memo.Store(image_id.index, resolved, {stencil_id, stencil_image.image_uid});
        }
    }

    return FindViewMemoized(image_id, desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    AssertNoAscTextureAccess();
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    Common::Perf::ScopedTimer refresh_timer{Common::Perf::ImageRefreshes,
                                            Common::Perf::ImageRefreshNs};
    if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
        LOG_INFO(Render_Vulkan, "DUMP upload image {:x}+{:x} {}x{} from guest memory",
                 image.info.guest_address, image.info.guest_size, image.info.size.width,
                 image.info.size.height);
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    // ISS_BIG_UPLOAD_LOG=1 (diagnosis): every upload of an image of 4 MB or more, with why it was
    // dirty (the ~17 MB uploads of the city's slow frames, play-porte-s 2026-10-18).
    static const bool big_upload_log = [] {
        const char* p = std::getenv("ISS_BIG_UPLOAD_LOG");
        return p && p[0] == '1';
    }();
    const bool log_big = big_upload_log && image.info.guest_size >= (4u << 20);
    const u64 t0 = log_big ? Common::Perf::NowNs() : 0;
    Common::Perf::Add(Common::Perf::ImageUploadBytes, image.info.guest_size);
    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const u64 t1 = log_big ? Common::Perf::NowNs() : 0;
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    const u64 t2 = log_big ? Common::Perf::NowNs() : 0;
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
    if (log_big) [[unlikely]] {
        const u64 t3 = Common::Perf::NowNs();
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 400)
            std::fprintf(stderr,
                         "BIG_UPLOAD frame=%llu addr=%llx size=%u %ux%ux%u mips=%u layers=%u fmt=%s "
                         "tile=%u flags=%x copies=%zu uid=%llu target=%d depth=%d obtain_us=%llu "
                         "detile_us=%llu upload_us=%llu\n",
                         (unsigned long long)Common::Perf::GuestFrameCounter().load(),
                         (unsigned long long)image.info.guest_address, u32(image.info.guest_size),
                         image.info.size.width, image.info.size.height, image.info.size.depth,
                         num_mips, num_layers, vk::to_string(image.info.pixel_format).c_str(),
                         u32(image.info.tile_mode), u32(image.flags), image_copies.size(),
                         (unsigned long long)image.image_uid, int(image.binding.is_target),
                         int(image.info.props.is_depth), (unsigned long long)((t1 - t0) / 1000),
                         (unsigned long long)((t2 - t1) / 1000), (unsigned long long)((t3 - t2) / 1000));
    }
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    AssertNoAscTextureAccess();
    // Compare and plain uses of one S# need separate samplers.
    const u64 hash = HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), is_depth);

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] =
        samplers.try_emplace(hash, instance, sampler, border_color_base, is_depth);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    std::scoped_lock writer{negative_index_writer_mutex};
    negative_image_index.BeginRegistryMutation();
    SCOPE_EXIT { negative_image_index.EndRegistryMutation(); };
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    BumpGeneration();
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    g_gc.texture_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    g_gc.images.fetch_add(1, std::memory_order_relaxed);
    {
        std::scoped_lock gc_lock{g_gc.mutex};
        if (!g_gc.deleted.empty()) {
            if (const auto it = g_gc.deleted.find(GcImageKey(image.info)); it != g_gc.deleted.end()) {
                if (gc_tick - it->second <= RecreateWindow) {
                    ++g_gc.recreated;
                    g_gc.recreated_bytes += image.info.guest_size;
                }
                g_gc.deleted.erase(it);
            }
        }
    }
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        page_table[page].push_back(image_id);
        page_image_counts[page].fetch_add(1, std::memory_order_release);
    });
    image_address_index[image.info.guest_address].push_back(image_id);
    iss::blackbox::Event(iss::blackbox::Image, "imagem +%u uid %llu em %llx %ux%u %s (%u bytes)",
                         image_id.index, (unsigned long long)image.image_uid,
                         (unsigned long long)image.info.guest_address,
                         image.info.size.width, image.info.size.height,
                         vk::to_string(image.info.pixel_format).c_str(), u32(image.info.guest_size));
}

void TextureCache::UnregisterImage(ImageId image_id) {
    std::scoped_lock writer{negative_index_writer_mutex};
    negative_image_index.BeginRegistryMutation();
    SCOPE_EXIT { negative_image_index.EndRegistryMutation(); };
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    BumpGeneration();
    image.flags &= ~ImageFlagBits::Registered;
    iss::blackbox::Event(iss::blackbox::Image, "imagem -%u uid %llu em %llx %ux%u %s",
                         image_id.index, (unsigned long long)image.image_uid,
                         (unsigned long long)image.info.guest_address, image.info.size.width,
                         image.info.size.height, vk::to_string(image.info.pixel_format).c_str());
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    g_gc.texture_bytes.fetch_sub(image.info.guest_size, std::memory_order_relaxed);
    g_gc.images.fetch_sub(1, std::memory_order_relaxed);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
        page_image_counts[page].fetch_sub(1, std::memory_order_release);
    });
    if (auto it = image_address_index.find(image.info.guest_address);
        it != image_address_index.end()) {
        auto& ids = it.value();
        if (const auto id_it = std::ranges::find(ids, image_id); id_it != ids.end()) {
            ids.erase(id_it);
        }
        if (ids.empty()) {
            image_address_index.erase(it);
        }
    }
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Track);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers(addr, size, PageOp::Track);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Untrack);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::GarbageCollectImages() {
    static std::atomic<u64> spared{}, deleted{};
    static const Headroom headroom =
        instance.CanReportMemoryUsage() ? ParseHeadroom() : Headroom{};
    static u64 heap_budget = 0;
    if (gc_tick % 600 == 0) {
        std::scoped_lock gc_lock{g_gc.mutex};
        std::fprintf(stderr,
                     "TEXTURE_GC spared=%llu deleted=%llu streak_images=%llu used=%llu MB trigger=%llu MB"
                     " pressure=%llu MB critical=%llu MB heap_budget=%llu MB textures=%llu MB images=%llu"
                     " gc_deleted=%llu MB recreated=%llu (%llu MB) ticks[trigger=%llu pressured=%llu"
                     " critical=%llu]\n",
                     (unsigned long long)spared.load(), (unsigned long long)deleted.load(),
                     (unsigned long long)tex_streak_images.load(),
                     (unsigned long long)(total_used_memory >> 20),
                     (unsigned long long)(trigger_gc_memory >> 20),
                     (unsigned long long)(pressure_gc_memory >> 20),
                     (unsigned long long)(critical_gc_memory >> 20),
                     (unsigned long long)(heap_budget >> 20),
                     (unsigned long long)(g_gc.texture_bytes.load() >> 20),
                     (unsigned long long)g_gc.images.load(),
                     (unsigned long long)(g_gc.deleted_bytes >> 20), (unsigned long long)g_gc.recreated,
                     (unsigned long long)(g_gc.recreated_bytes >> 20),
                     (unsigned long long)g_gc.ticks_trigger, (unsigned long long)g_gc.ticks_pressured,
                     (unsigned long long)g_gc.ticks_critical);
        // Deletions older than the window can no longer count as recreated.
        if (g_gc.deleted.size() > 65536) {
            std::erase_if(g_gc.deleted, [this](const auto& entry) {
                return gc_tick - entry.second > RecreateWindow;
            });
        }
    }
    if (headroom.enabled) {
        // Usage and budget over the heaps of Instance::valid_heaps (integrated GPUs: all of
        // them; discrete: the device-local ones), the thresholds the given room below the
        // budget of now.
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        vk::PhysicalDeviceMemoryProperties2 props{.pNext = &budget};
        instance.GetPhysicalDevice().getMemoryProperties2(&props);
        const bool integrated = instance.IsIntegrated();
        u64 usage = 0;
        heap_budget = 0;
        for (u32 heap = 0; heap < props.memoryProperties.memoryHeapCount; ++heap) {
            if (!integrated && !(props.memoryProperties.memoryHeaps[heap].flags &
                                 vk::MemoryHeapFlagBits::eDeviceLocal))
                continue;
            usage += budget.heapUsage[heap];
            heap_budget += budget.heapBudget[heap];
        }
        const auto below = [](u64 b, u64 room) { return b > room ? b - room : 0; };
        total_used_memory = usage;
        trigger_gc_memory = below(heap_budget, headroom.trigger);
        pressure_gc_memory = below(heap_budget, headroom.pressure);
        critical_gc_memory = below(heap_budget, headroom.critical);
    } else if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    {
        std::scoped_lock gc_lock{g_gc.mutex};
        ++(total_used_memory >= critical_gc_memory   ? g_gc.ticks_critical
           : total_used_memory >= pressure_gc_memory ? g_gc.ticks_pressured
                                                     : g_gc.ticks_trigger);
    }
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    // ISS_GC_RECENT_USE (default on): an image bound since the oldest gc tick the collector may
    // delete is in use, whatever its LRU position says. The draw paths that skip FindImage (the
    // texture lookup and stage memos, native passes, material templates, native items) set
    // tick_accessed_last but never touched the LRU: under memory pressure the collector deleted
    // textures every frame used, and the guest created and uploaded them again (bursts of 5-30
    // new images a frame, +12 ms frame time spikes while moving, 2026-10-17). Such an image is
    // moved to the LRU's end instead, except when memory is critical (the aggressive pass).
    static const bool recent_use = [] {
        const char* p = std::getenv("ISS_GC_RECENT_USE");
        return !p || p[0] != '0';
    }();
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        if (recent_use && !aggresive && gc_tick >= ticks_to_destroy &&
            ticks_to_destroy < gc_scheduler_ticks.size() && slot_images.is_allocated(image_id)) {
            const u64 used_since =
                gc_scheduler_ticks[(gc_tick - ticks_to_destroy) % gc_scheduler_ticks.size()];
            auto& recent = slot_images[image_id];
            if (used_since != 0 && True(recent.flags & ImageFlagBits::Registered) &&
                recent.tick_accessed_last >= used_since) {
                spared.fetch_add(1, std::memory_order_relaxed);
                TouchImage(recent);
                return false;
            }
        }
        deleted.fetch_add(1, std::memory_order_relaxed);
        --num_deletions;
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            DownloadImageMemory(image_id);
        }
        {
            std::scoped_lock gc_lock{g_gc.mutex};
            g_gc.deleted[GcImageKey(image.info)] = gc_tick;
            g_gc.deleted_bytes += image.info.guest_size;
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const size_t lru_id = samplers.at(hash).lru_id;
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        // Handles remembered outside the cache (Rasterizer sampler memo) are stale.
        samplers_generation.fetch_add(1, std::memory_order_release);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    gc_scheduler_ticks[gc_tick % gc_scheduler_ticks.size()] = scheduler.CurrentTick();

    GarbageCollectImages();
    GarbageCollectSamplers();
    if (texture_lookup_memo_enabled && texture_lookup_stats_enabled &&
        gc_tick != 0 && gc_tick % 1800 == 0) ReportLookupMemo();
}

void TextureCache::TouchImage(const Image& image) {
    const size_t id = image.lru_id;
    if (id < lru_touched.size()) {
        if (lru_touched[id] == gc_tick + 1) {
            // Touched this frame already: the LRU item has this tick (Touch would return).
            return;
        }
    } else {
        lru_touched.resize(std::max<size_t>(id + 1, lru_touched.size() * 2 + 64));
    }
    lru_touched[id] = gc_tick + 1;
    lru_cache.Touch(id, gc_tick);
}

namespace {
struct FreeSite {
    const char* function{};
    u32 line{};
    u64 thread{};
    u64 tick{};
    VAddr address{};
    u64 size{};
};
std::mutex free_sites_mutex;
std::unordered_map<u32, FreeSite> free_sites; // by slot
std::atomic<u32> double_frees{};
u64 ThisThread() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFF;
}
} // Anonymous namespace

void TextureCache::NoteFree(ImageId image_id, const std::source_location& where) {
    const auto& image = slot_images[image_id];
    std::scoped_lock lk{free_sites_mutex};
    free_sites[image_id.index] = {where.function_name(), where.line(), ThisThread(),
                                  scheduler.CurrentTick(), image.info.guest_address,
                                  image.info.guest_size};
}

void TextureCache::ReportDoubleFree(ImageId image_id, const std::source_location& where) {
    const u32 n = double_frees.fetch_add(1, std::memory_order_relaxed);
    if (n >= 64) return;
    const bool allocated = slot_images.is_allocated(image_id);
    FreeSite first{};
    {
        std::scoped_lock lk{free_sites_mutex};
        if (const auto it = free_sites.find(image_id.index); it != free_sites.end()) first = it->second;
    }
    std::fprintf(stderr,
                 "TEXTURE_DOUBLE_FREE #%u image %u allocated=%d now: %s line %u thread %llx tick %llu"
                 " | first free: %s line %u thread %llx tick %llu image 0x%llx+0x%llx\n",
                 n, image_id.index, int(allocated), where.function_name(), where.line(),
                 (unsigned long long)ThisThread(), (unsigned long long)scheduler.CurrentTick(),
                 first.function ? first.function : "?", first.line,
                 (unsigned long long)first.thread, (unsigned long long)first.tick,
                 (unsigned long long)first.address, (unsigned long long)first.size);
}

void TextureCache::DeleteImage(ImageId image_id) {
    // A remembered (id, generation) must not match the image a reused slot holds next: the
    // native recorders check it lock free before handing remembered view handles to the driver
    // (NativeMaterial::HandlesAlive). Its views are destroyed below, a GPU tick later.
    BumpImageGeneration(image_id);
    Image& image = slot_images[image_id];
    iss::blackbox::ImageDeleted(image.image_uid, image.info.guest_address);
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        EraseSurfaceMeta(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        EraseSurfaceMeta(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        EraseSurfaceMeta(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
