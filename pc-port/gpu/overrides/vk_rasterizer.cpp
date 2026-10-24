#include <shared_mutex>
#include <algorithm>
#include <execution>
#include "background_free.h"
#include "native_item_identity.h"
#include "repeat_census.h"
#include "unified_memory.h"
#include "native_shadow.h"
#include "native_gbuffer.h"
#include "native_passes.h"
#include "pause_guard.h"
#include "cp_path_cycles.h"
#include "common/scope_exit.h"
#include "deep_profile.h"
#include "feedback_join.h"
#include "infamous_renderer/secondary_batch_probe.h"
#include <unordered_set>
#include <cstdio>
#include <intrin.h>
#include <span>
#include <unordered_map>
#include "draw_pipe.h"
#include "g4_qid1_owner.h"
#include "g4_copy_hle_plan.h"
#include "hazard_probe.h"
#include "infamous_renderer/feedback_identity_registry.h"
#include "infamous_renderer/native_draw_run.h"
#include "infamous_renderer/native_geometry_batch_vulkan.h"
#include "native_backing_alias_cache.h"
#include "infamous_renderer/native_instanced_batch_vulkan.h"
#include "infamous_renderer/native_instanced_run.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/memory_tracker.h"
namespace VideoCore {void RecordReadbackDma(VAddr,VAddr,u64,u64,u32);}
namespace iss::diagnostics {void RecordFeedbackBinding(u64 address,u64 size,u64 hash,u64 tick);}
// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include "common/debug.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_stats.h"
#include "runtime_timeline.h"
#include "compute_phase.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "srt_walker_capture.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/renderer_vulkan/descriptor_write_storage.h"

namespace iss::command_trace {
extern const bool enabled;
void Submit();
void Stage(const char* kind, int queue, const Shader::Info& info, u64 code,
           u32 x, u32 y, u32 z, u64 tick, bool barrier, u64 args, u64 count_address);
void VertexFetch(int queue, const Shader::Info& info,
                 const Shader::Gcn::FetchShaderData& fetch, u64 pipeline_identity,
                 u64 context_changes, u32 primitive_type, u32 instance_count,
                 u32 index_count, u64 index_address, u32 index_type, u32 index_offset,
                 const std::array<u8, 3>& attribute_ownership, u8 index_ownership,
                 u64 mapping_epoch, u64 buffer_mapping_generation);
}

namespace iss::asc_stats {
// Defined in buffer_cache.cpp (same enum).
enum Cause:u32 {WaitGraphicsSubmitted,ComputeSubmit,GdsFlushGraphics,GdsFlushCompute,BdaFlush,
                RangeFlushCompute,RangeFlushGraphics,FirstTouchFlushGraphics,ReadbackWriterFlush,Num};
extern std::atomic<u64> count[Num],ns[Num];
inline constexpr const char* names[Num]{"wait_gfx_submitted","compute_submit","gds_flush_gfx",
    "gds_flush_compute","bda_flush","range_flush_compute","range_flush_gfx","first_touch_flush_gfx",
    "readback_writer_flush"};
// UseComputeQueue fallbacks: images, samplers, dma, no push descriptors.
std::atomic<u64> fallback[4]{};
}
namespace iss::asc_stats_api {void Print();}
namespace iss::ring_barriers {
// ISS_RING_BARRIERS=1: the particle update dispatches of the guest's compute ring (phase 2,
// shaders A8) run on the PS4 back to back with no barrier between them inside a segment of the
// ring (the guest's barriers are RELEASE_MEM/WAIT_REG_MEM/ACQUIRE_MEM packets, which end the run
// as before). The shaders listed here were audited (evidence/mapping/shader-access-proof-09.json,
// CONTRATOS-IMPLEMENTACAO.md section 1): no load of the global record array, every store of it at
// an index returned by an atomic add, GDS and feedback by atomics only, the particle state of
// their own emitter. Only hazards between dispatches of these shaders inside one run are not
// flushed; any other dispatch, upload, copy or packet makes the next hazard flush. The dependent
// pair E902349D -> 03E4B72E (0x3F49E0, phase 3) is not listed and keeps its barriers.
static const bool enabled = [] {
    const char* p = std::getenv("ISS_RING_BARRIERS");
    const bool on = p && p[0] == '1';
    std::fprintf(stderr, "PORT_TUNING ring_barriers=%d\n", int(on));
    return on;
}();
constexpr u32 audited[]{0x3839CBEA, 0x29F590F3, 0xB12F0F9D, 0x7EB39747, 0x4274ADD5,
                        0x1FC9E69C, 0x5563EA54, 0x187E6FAF, 0x21E3C997};
bool Audited(u64 hash) {
    for (const u32 h : audited) {
        if (u32(hash) == h) {
            return true;
        }
    }
    return false;
}
}
namespace iss::asc_family {extern u32 family;extern thread_local u32 pool_family;}
uint64_t IssImageBase(); // bridge.cpp

namespace Vulkan {

thread_local Rasterizer::SplitThreadState Rasterizer::split_thread_state{};
#include "draw_pipe_rasterizer.inl"

namespace {

bool EarlyFeedbackEnabled() {
    static const bool enabled=[] {const char *p=std::getenv("ISS_FEEDBACK_EARLY_SUBMIT");return p&&p[0]=='1';}();
    return enabled;
}
// Only the graphics command processor owns this state. A tick change means an
// existing fence/readback already submitted the batch; do not submit it twice.
thread_local u64 pending_feedback_tick{};
void SubmitFeedbackBatch(Scheduler& scheduler) {
    if(!EarlyFeedbackEnabled()||!pending_feedback_tick)return;
    const auto tick=std::exchange(pending_feedback_tick,0);
    if(tick!=scheduler.CurrentTick())return;
    scheduler.Flush();
    static u64 count{};
    if(++count==1||(count&(count-1))==0)
        LOG_INFO(Render_Vulkan,"PORT_FEEDBACK early submissions {}",count);
}
// ISS_SUBMIT_EVERY=N: the batch is submitted every N draws of one submission, so the GPU runs
// the frame while the command processor records the rest of it. Without it the GPU only starts
// at the readbacks (about 8 per frame), and the command processor then waits for all the work
// recorded before each of them (CP recording and GPU execution do not overlap).
// ISS_SUBMIT_INTERVAL_US=N: also once the batch has been recorded for N us. A native segment counts
// as one draw, so with ~150 segments a frame the batches went out ~13 times a frame, up to 2.4 ms
// apart: the GPU waited for the next one, then fell behind on a heavy one (a1-jobwait 2026-10-18).
thread_local u64 periodic_tick{};
thread_local u32 periodic_draws{};
thread_local u64 periodic_start_ns{};
void PeriodicSubmit(Scheduler& scheduler) {
    static const u32 every = [] {
        const char* p = std::getenv("ISS_SUBMIT_EVERY");
        return p ? static_cast<u32>(std::strtoul(p, nullptr, 10)) : 0u;
    }();
    static const u64 interval_ns = [] {
        const char* p = std::getenv("ISS_SUBMIT_INTERVAL_US");
        return p ? std::strtoull(p, nullptr, 10) * 1000ull : 0ull;
    }();
    if (every == 0 && interval_ns == 0) return;
    const u64 tick = scheduler.CurrentTick();
    if (tick != periodic_tick) {
        periodic_tick = tick;
        periodic_draws = 0;
        periodic_start_ns = interval_ns ? Common::Perf::NowNs() : 0;
    }
    const bool due = interval_ns && Common::Perf::NowNs() - periodic_start_ns >= interval_ns;
    if (!due && (every == 0 || ++periodic_draws < every)) return;
    scheduler.Flush();
    periodic_tick = scheduler.CurrentTick();
    periodic_draws = 0;
    periodic_start_ns = interval_ns ? Common::Perf::NowNs() : 0;
}
bool WritesFeedbackRecord(const Shader::Info& stage) {
    if(!EarlyFeedbackEnabled())return false;
    return std::ranges::any_of(stage.buffers,[&](const auto& desc){
        return desc.is_written&&!desc.IsSpecial()&&desc.GetSharp(stage).GetSize()==48;
    });
}

/// GPU black box (gpu_blackbox.h): the targets a draw writes with the registers as they are.
GpuCrashDiag::Context BlackboxContext(const AmdGpu::Regs& regs, u32 path) {
    GpuCrashDiag::Context ctx{};
    ctx.path = path;
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS && i < 8; ++i) {
        const auto& cb = regs.color_buffers[i];
        if (!cb || regs.color_target_mask.GetMask(i) == 0) continue;
        ctx.color[i] = cb.Address();
        ctx.color_format[i] = cb.info.format;
        if (regs.blend_control[i].enable) ctx.blend_mask |= 1u << i;
    }
    if (regs.depth_buffer.DepthValid()) ctx.depth = regs.depth_buffer.DepthAddress();
    return ctx;
}

/// GPU black box: the texture cache images a command samples and draws into (uids, addresses).
template <typename Images, typename Cache, typename Targets, typename Depth>
void BlackboxImages(GpuCrashDiag::Context& ctx, const Images& images, Cache& cache,
                    const Targets& targets, const Depth& depth) {
    u32 n = 0;
    for (const auto id : images) {
        if (n >= 8) break;
        if (!id) continue;
        const auto& image = cache.GetImage(id);
        bool seen = false;
        for (u32 i = 0; i < n; ++i) seen |= ctx.tex_uid[i] == image.image_uid;
        if (seen) continue;
        ctx.tex_uid[n] = image.image_uid;
        ctx.tex_addr[n] = image.info.guest_address;
        ++n;
    }
    for (u32 i = 0; i < 8 && i < targets.size(); ++i)
        if (targets[i].first && ctx.color[i]) ctx.rt_uid[i] = cache.GetImage(targets[i].first).image_uid;
    if (depth.first && ctx.depth) ctx.depth_uid = cache.GetImage(depth.first).image_uid;
}

/// Leaves a GPU crash diagnostics checkpoint for a draw (see GpuCrashDiag).
void MarkDraw(const Scheduler& scheduler, const CommandRecorder& cmdbuf,
              const GraphicsPipeline& pipeline, GpuCrashDiag::Kind kind, u32 count, u32 instances,
              const RenderState& state, VAddr args, const GpuCrashDiag::Context& ctx) {
    GpuCrashDiag::Entry entry{
        .kind = kind,
        .count = count,
        .instances = instances,
        .dims = {state.width, state.height, state.num_layers},
        .address = args,
    };
    for (const auto* stage : pipeline.GetStages()) {
        if (!stage) {
            continue;
        }
        if (stage->sw_stage == Shader::SwStage::Fragment) {
            entry.hashes[1] = stage->pgm_hash;
        } else if (stage->sw_stage == Shader::SwStage::Vertex ||
                   stage->sw_stage == Shader::SwStage::TessellationEval) {
            entry.hashes[0] = stage->pgm_hash;
        }
    }
    if (GpuCrashDiag::Blackbox())
        GpuCrashDiag::MarkBox(&scheduler, cmdbuf, scheduler.CurrentTick(), entry, ctx);
    if (GpuCrashDiag::Enabled()) GpuCrashDiag::Mark(cmdbuf, scheduler.CurrentTick(), entry);
}

/// Leaves a GPU crash diagnostics checkpoint for a dispatch (see GpuCrashDiag).
void MarkDispatch(const Scheduler& scheduler, const CommandRecorder& cmdbuf, u64 cs_hash, u32 x,
                  u32 y, u32 z, VAddr args) {
    const GpuCrashDiag::Entry entry{
        .kind = args != 0 ? GpuCrashDiag::Kind::DispatchIndirect : GpuCrashDiag::Kind::Dispatch,
        .dims = {x, y, z},
        .hashes = {cs_hash, 0},
        .address = args,
    };
    if (GpuCrashDiag::Blackbox())
        GpuCrashDiag::MarkBox(&scheduler, cmdbuf, scheduler.CurrentTick(), entry,
                              GpuCrashDiag::Context{});
    if (GpuCrashDiag::Enabled()) GpuCrashDiag::Mark(cmdbuf, scheduler.CurrentTick(), entry);
}

/// Frame x-ray state (Tuning::dump_frames), only used by the command processor thread.
struct FrameDumpState {
    u32 last_request{};
    bool started{}; ///< The value found at startup is not a request.
    u32 frames_left{};
    u32 frame{};
    u32 seq{};
    std::array<u32, 8> after_counts{}; ///< Draws and dispatches seen per dump_after entry.
};
FrameDumpState g_dump;

} // Anonymous namespace

bool Rasterizer::IsDumping() const noexcept {
    return g_dump.frames_left != 0;
}

void Rasterizer::DumpNote(std::string_view kind, std::string_view text) {
    if (g_dump.frames_left == 0) {
        return;
    }
    LOG_INFO(Render_Vulkan, "DUMP f{} #{} {} q{} {}", g_dump.frame, g_dump.seq++, kind,
             DrawQueue(), text);
}

void Rasterizer::DumpAfter(const Pipeline* pipeline) {
    if (g_dump.frames_left == 0 || !pipeline) {
        return;
    }
    const auto& tuning = Common::Perf::GetTuning();
    for (size_t i = 0; i < tuning.dump_after.size(); ++i) {
        const auto& entry = tuning.dump_after[i];
        const u64 shader = entry.shader.load(std::memory_order_relaxed);
        if (shader == 0) {
            break;
        }
        bool match = false;
        for (const Shader::Info* stage : pipeline->GetStages()) {
            if (stage && (stage->hw_stage == Shader::HwStage::Fragment ||
                          stage->hw_stage == Shader::HwStage::Compute) &&
                stage->pgm_hash == shader) {
                match = true;
            }
        }
        if (!match) {
            continue;
        }
        if (++g_dump.after_counts[i] != entry.count.load(std::memory_order_relaxed)) {
            continue;
        }
        const u64 address = entry.address.load(std::memory_order_relaxed);
        if (address == 1) {
            // "hash:vb": the vertex buffers of the draw, as the GPU reads them and as guest
            // memory has them.
            u32 index = 0;
            for (const auto& range : dump_vertex_ranges) {
                DumpBufferToFile(fmt::format("f{}_s{}_vb{}", g_dump.frame, g_dump.seq - 1, index++),
                                 range.buffer, range.offset, range.base, range.size);
            }
            continue;
        }
        if (address == 2) {
            // "hash:srt": the guest buffers bound to the shaders of the draw or dispatch.
            u32 index = 0;
            for (const auto& bound : bound_buffers) {
                if (bound.address != 0) {
                    DumpBufferToFile(
                        fmt::format("f{}_s{}_srt{}", g_dump.frame, g_dump.seq - 1, index),
                        bound.buffer, bound.offset, bound.address, bound.size);
                }
                ++index;
            }
            continue;
        }
        texture_cache.DumpImageToFile(address, fmt::format("f{}_s{}", g_dump.frame, g_dump.seq - 1),
                                      entry.width.load(std::memory_order_relaxed),
                                      entry.height.load(std::memory_order_relaxed));
    }
}

void Rasterizer::DumpBufferToFile(std::string_view name, const VideoCore::Buffer* buffer,
                                  u64 offset, VAddr address, u64 size) {
    constexpr u64 MaxBytes = 64_MB;
    if (!buffer || size == 0 || size > MaxBytes) {
        return;
    }
    const auto download =
        runtime.GetStagingPool().Request(size, VideoCore::MemoryType::HostCached, 16, false);
    const vk::BufferCopy copy = {
        .srcOffset = offset,
        .dstOffset = download.offset,
        .size = size,
    };
    runtime.CopyBuffer(buffer, download.buffer, std::span{&copy, 1});
    scheduler.Finish();
    download.Invalidate();
    using namespace Common::FS;
    const auto dir = GetUserPath(PathType::LogDir) / "xray";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto gpu_name = fmt::format("{}_{:x}_{}.gpu.raw", name, address, size);
    const auto host_name = fmt::format("{}_{:x}_{}.host.raw", name, address, size);
    {
        const auto file = IOFile{dir / gpu_name, FileAccessMode::Create};
        file.WriteRaw<u8>(download.mapped, size);
    }
    {
        std::vector<u8> host(size);
        memory->CopySparseMemory(address, host.data(), size);
        const auto file = IOFile{dir / host_name, FileAccessMode::Create};
        file.WriteRaw<u8>(host.data(), size);
    }
    LOG_INFO(Render_Vulkan, "DUMP buffer {:x}+{:x} written: {} / {} [{}]", address, size, gpu_name,
             host_name, buffer_cache.DescribePages(address, static_cast<u32>(size)));
}

void Rasterizer::ProbeBuffer(std::string_view what, const VideoCore::Buffer* buffer, u64 offset,
                             u32 size, VAddr address) {
    constexpr u64 ProbeBytes = 2_MB;
    constexpr u32 ProbeStride = 32;
    if (g_dump.frames_left == 0 || size == 0 || size > ProbeStride || address == 0) {
        return;
    }
    if (!probe_staging.buffer) {
        probe_staging = runtime.GetStagingPool().Request(ProbeBytes, VideoCore::MemoryType::HostCached,
                                                         0, true);
        probe_used = 0;
    }
    if (probe_used + ProbeStride > ProbeBytes) {
        return;
    }
    Probe probe{
        .frame = g_dump.frame,
        .seq = g_dump.seq - 1,
        .what = std::string{what},
        .address = address,
        .size = size,
        .offset = probe_used,
        .gpu = buffer != nullptr,
        .host = {},
        .state = buffer_cache.DescribePages(address, size),
    };
    memory->CopySparseMemory(address, reinterpret_cast<u8*>(probe.host.data()), size);
    if (buffer) {
        const vk::BufferCopy copy = {
            .srcOffset = offset,
            .dstOffset = probe_staging.offset + probe_used,
            .size = size,
        };
        runtime.CopyBuffer(buffer, probe_staging.buffer, std::span{&copy, 1});
    }
    probe_used += ProbeStride;
    probes.push_back(std::move(probe));
}

void Rasterizer::ProbeMemory(std::string_view what, VAddr address, u32 size) {
    if (g_dump.frames_left == 0) {
        return;
    }
    const auto [arena, offset] = buffer_cache.PeekArena(address, size);
    ProbeBuffer(what, arena, offset, size, address);
}

void Rasterizer::FinishProbes() {
    if (probe_staging.buffer && !probes.empty()) {
        scheduler.Finish();
        probe_staging.Invalidate();
        const auto words = [](const u32* data, u32 size) {
            std::string text;
            for (u32 i = 0; i < (size + 3) / 4; ++i) {
                text += fmt::format("{}{:x}", i ? "," : "", data[i]);
            }
            return text;
        };
        for (const Probe& probe : probes) {
            std::array<u32, 8> gpu{};
            std::string verdict = "host-only";
            if (probe.gpu) {
                std::memcpy(gpu.data(), probe_staging.mapped + probe.offset, probe.size);
                verdict = std::memcmp(gpu.data(), probe.host.data(), probe.size) == 0 ? "same"
                                                                                       : "DIFF";
            }
            LOG_INFO(Render_Vulkan, "DUMPV f{} #{} {} {:x}+{} host {} gpu {} {} [{}]", probe.frame,
                     probe.seq, probe.what, probe.address, probe.size,
                     words(probe.host.data(), probe.size),
                     probe.gpu ? words(gpu.data(), probe.size) : "-", verdict, probe.state);
        }
    }
    probes.clear();
    if (probe_staging.buffer) {
        runtime.GetStagingPool().FreeDeferred(probe_staging);
        probe_staging = {};
    }
    probe_used = 0;
}

void Rasterizer::DumpCommand(std::string_view kind, const Pipeline* pipeline,
                             std::string_view params) {
    if (g_dump.frames_left == 0) {
        return;
    }
    const auto& regs = DrawRegs();
    std::string line = fmt::format("DUMP f{} #{} {} q{} {}", g_dump.frame, g_dump.seq++, kind,
                                   DrawQueue(), params);
    if (pipeline) {
        for (const Shader::Info* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            line += fmt::format(" | s{}:{:x}", u32(stage->hw_stage), stage->pgm_hash);
            for (const auto& desc : stage->buffers) {
                if (desc.IsSpecial()) {
                    continue;
                }
                const auto vsharp = desc.GetSharp(*stage);
                line += fmt::format(" b{:x}+{:x}{}", vsharp.base_address, vsharp.GetSize(),
                                    desc.is_written ? "w" : "");
            }
            for (const auto& desc : stage->images) {
                const auto tsharp = desc.GetSharp(*stage);
                line += fmt::format(" t{:x}:{}x{}x{}:f{}n{}{}", tsharp.Address(),
                                    tsharp.width + 1, tsharp.height + 1, tsharp.depth + 1,
                                    u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()),
                                    desc.is_written ? "w" : "");
            }
        }
    }
    if (!pipeline || !pipeline->IsCompute()) {
        for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
            const auto& cb = regs.color_buffers[i];
            if (!cb || regs.color_target_mask.GetMask(i) == 0) {
                continue;
            }
            const auto& hint = DrawCbExtents()[i];
            line += fmt::format(" rt{}:{:x}:{}x{}:f{}n{}s{}:h{}x{}", i, cb.Address(), cb.Pitch(),
                                cb.Height(), u32(cb.GetDataFmt()), u32(cb.GetNumberFmt()),
                                cb.NumSamples(), hint.width, hint.height);
            const auto& blend = regs.blend_control[i];
            if (blend.enable) {
                line += fmt::format(" blend{}:{}:{}", u32(blend.color_func),
                                    u32(blend.color_src_factor), u32(blend.color_dst_factor));
            }
        }
        const auto& db = regs.depth_buffer;
        if (db.DepthValid()) {
            line += fmt::format(" z:{:x}{}{}f{}", db.DepthAddress(),
                                regs.depth_control.depth_enable ? "t" : "",
                                regs.depth_control.depth_write_enable ? "w" : "",
                                u32(regs.depth_control.depth_func));
        }
    }
    LOG_INFO(Render_Vulkan, "{}", line);
}

OcclusionQueries::OcclusionQueries(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const vk::QueryPoolCreateInfo pool_info = {
        .queryType = vk::QueryType::eOcclusion,
        .queryCount = NumQueries,
    };
    auto [result, query_pool] = instance.GetDevice().createQueryPoolUnique(pool_info);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to create occlusion query pool: {}",
               vk::to_string(result));
    pool = std::move(query_pool);
    if (instance.IsOcclusionQueryPreciseSupported()) {
        control_flags = vk::QueryControlFlagBits::ePrecise;
    }
    scheduler.SetRenderingObserver(this);
}

OcclusionQueries::~OcclusionQueries() {
    scheduler.SetRenderingObserver(nullptr);
}

void OcclusionQueries::OnBeginRendering(const CommandRecorder& cmdbuf) {
    if (!open.empty() && !active) {
        BeginSegment(cmdbuf);
    }
}

void OcclusionQueries::OnEndRendering(const CommandRecorder& cmdbuf) {
    if (active) {
        EndSegment(cmdbuf);
    }
}

bool OcclusionQueries::Begin(VAddr address, u32 num_pairs) {
    // A query begun again at the same address starts over.
    for (auto it = open.begin(); it != open.end(); ++it) {
        if (it->address == address) {
            open.erase(it);
            break;
        }
    }
    if (open.size() >= MaxOpenQueries) {
        return false;
    }
    Cut();
    open.push_back(OpenQuery{
        .address = address,
        .max_segments = std::clamp<u32>(num_pairs, 1, MaxCounterPairs),
        .first_segment = static_cast<u32>(segments.size()),
        .frame = frame,
        .incomplete = false,
    });
    Resume();
    return true;
}

bool OcclusionQueries::End(VAddr begin_address,
                           boost::container::static_vector<u32, MaxCounterPairs>& out_queries) {
    const auto it = std::find_if(open.begin(), open.end(), [begin_address](const OpenQuery& q) {
        return q.address == begin_address;
    });
    if (it == open.end()) {
        return false;
    }
    Cut();
    const OpenQuery query = *it;
    open.erase(it);
    const u32 num_segments = static_cast<u32>(segments.size()) - query.first_segment;
    const bool counted = !query.incomplete && num_segments <= query.max_segments;
    if (counted) {
        out_queries.assign(segments.begin() + query.first_segment, segments.end());
    }
    DropOldSegments();
    return counted;
}

void OcclusionQueries::DropOldSegments() {
    u32 oldest = static_cast<u32>(segments.size());
    for (const OpenQuery& q : open) {
        oldest = std::min(oldest, q.first_segment);
    }
    if (oldest == 0) {
        return;
    }
    if (active && oldest == segments.size()) {
        // The active segment counts for no open query.
        --oldest;
    }
    segments.erase(segments.begin(), segments.begin() + oldest);
    for (OpenQuery& q : open) {
        q.first_segment -= oldest;
    }
}

void OcclusionQueries::OnFrameEnd() {
    ++frame;
    const auto old_size = open.size();
    for (auto it = open.begin(); it != open.end();) {
        if (frame - it->frame > 2) {
            it = open.erase(it);
        } else {
            ++it;
        }
    }
    if (open.size() == old_size) {
        return;
    }
    if (open.empty()) {
        Cut();
        segments.clear();
        return;
    }
    DropOldSegments();
}

void OcclusionQueries::AbandonAll() {
    Cut();
    open.clear();
    segments.clear();
}

void OcclusionQueries::MarkUsed(std::span<const u32> queries) {
    const u64 tick = scheduler.CurrentTick();
    for (const u32 query : queries) {
        query_ticks[query] = tick;
    }
}

void OcclusionQueries::Cut() {
    if (active) {
        EndSegment(scheduler.Recorder());
    }
}

void OcclusionQueries::Resume() {
    if (!open.empty() && !active && scheduler.IsRendering()) {
        BeginSegment(scheduler.Recorder());
    }
}

void OcclusionQueries::BeginSegment(const CommandRecorder& cmdbuf) {
    const auto not_counted = [this] {
        // The samples of the open queries until the next segment are lost.
        for (OpenQuery& q : open) {
            q.incomplete = true;
        }
        Common::Perf::Add(Common::Perf::OcclusionDropped);
    };
    if (segments.size() >= MaxSegments || !instance.IsHostQueryResetSupported()) {
        not_counted();
        return;
    }
    const u32 query = next_query;
    const u64 last_use = query_ticks[query];
    if (last_use != 0 && !scheduler.IsFree(last_use)) {
        // Its last use (thousands of queries ago) is still executing.
        not_counted();
        return;
    }
    next_query = (next_query + 1) % NumQueries;
    // Reset from the host, as this is inside a rendering scope.
    instance.GetDevice().resetQueryPool(*pool, query, 1);
    query_ticks[query] = scheduler.CurrentTick();
    cmdbuf.beginQuery(*pool, query, control_flags);
    segments.push_back(query);
    active = true;
}

void OcclusionQueries::EndSegment(const CommandRecorder& cmdbuf) {
    cmdbuf.endQuery(*pool, segments.back());
    active = false;
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, active_runtime{&runtime_},
      active_scheduler{&scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      occlusion{instance, scheduler} {
    if (const char* value = std::getenv("ISS_CP_SPLIT_PREPARE_SERIAL");
        value && value[0] == '1') {
        // Exercise the thread-local queue selection and owner scratch on the existing
        // single CP before any second worker is allowed to touch shared caches.
        buffer_cache.SetCpSplitEnabled(true);
        LOG_INFO(Render_Vulkan, "CP split owner scratch: serial preparation enabled");
    }
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
        ASSERT_MSG(liverpool->BindGraphicsFallbackRecorder(
                       [this](const iss::g4::FallbackPacket& packet,
                              const iss::g4::QidTick& prefix) {
                           return RecordFallbackSnapshot(packet, prefix);
                       }),
                   "G4 graphics fallback recorder was not installed before CP startup");
        // The scout follows the command stream from the start, so that its registers match
        // the command processor's whenever Tuning::scout turns it on.
        liverpool->StartScout(pipeline_cache);
    }
    memory->SetRasterizer(this);
    // Only the command processor records with this scheduler (the presenter records its frame
    // copy on that thread too): its commands may be recorded by a thread of their own.
    scheduler.AllowThreadedRecording();

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        // ISS_UNIFIED_MEMORY: the deferred fences recorded so far, handed over with one wake up.
        scheduler.DeferPriorityOperationsAt(unified_ops);
        if (unified_barrier_pending) {
            // ISS_UNIFIED_MEMORY: the GPU writes of a command buffer that ends deferred fences
            // are visible to the host when their labels are written.
            unified_barrier_pending = false;
            buffer_cache.UnifiedHostBarrier();
        }
        buffer_cache.SubmitPendingArenaBinds(info);
    });
    scheduler.SetPostSubmitCallback([this](u64 tick) {
        if (liverpool->CpSplitActive()) {
            [[maybe_unused]] const bool published =
                liverpool->PublishCpSubmitted(iss::g4::CpLane::Graphics, tick);
        }
    });
    InitializeDrawPipe();
}

bool Rasterizer::Qid1RecordContextEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_CP_RECORD_CONTEXT_QID1_STAGE2");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool Rasterizer::PrepareCpSplit(std::string& reason) {
    if (!instance.HasComputeQueue()) {
        reason = "no_compute_queue";
        return false;
    }
    if (instance.MaxPushDescriptors() == 0) {
        reason = "no_push_descriptors";
        return false;
    }
    if (!Common::Perf::GetTuning().asc_queue.load(std::memory_order_relaxed)) {
        reason = "async_compute_disabled";
        return false;
    }
    if (Common::Perf::GetTuning().hot_page_faults.load(std::memory_order_relaxed) != 0) {
        reason = "hot_page_faults_enabled";
        return false;
    }
    if (draw_pipe_state) {
        reason = "draw_pipe_enabled";
        return false;
    }
    if (Common::Perf::GetTuning().async_pipelines.load(std::memory_order_relaxed) != 0) {
        reason = "async_pipeline_skips_enabled";
        return false;
    }
    if (EarlyFeedbackEnabled()) {
        reason = "feedback_early_submit_enabled";
        return false;
    }
    if (g_dump.frames_left != 0 ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed) ||
        Common::Perf::GetTuning().dump_frames.load(std::memory_order_relaxed) != 0) {
        reason = "frame_dump_enabled";
        return false;
    }
    // This is called on G before C exists. Both Vulkan recording contexts and the
    // BufferCache cross-queue submission hook must be installed before publishing C.
    try {
        EnsureComputeQueueInitialized();
    } catch (const std::exception& error) {
        LOG_ERROR(Render_Vulkan, "CP split ACE startup failed: {}", error.what());
        reason = "compute_context_initialization_failed";
        return false;
    }
    if (!compute_scheduler || !compute_runtime || !compute_stream ||
        compute_context.scheduler != compute_scheduler.get() ||
        compute_context.runtime != compute_runtime.get() ||
        compute_context.stream_buffer != compute_stream.get()) {
        reason = "compute_context_unavailable";
        return false;
    }
    // Concurrent recording must wait for the packet-level resource reservation and
    // generation checks. Never start a second CP against the shared mutable caches before it.
    reason = "packet_coherence_unavailable";
    return false;
}

bool Rasterizer::SealCpPrefix(iss::g4::CpLane lane, u64 through_tick) {
    if (!cp_split_enabled) return false;
    Scheduler* owner = lane == iss::g4::CpLane::Graphics ? &scheduler : compute_scheduler.get();
    if (!owner || owner->CurrentTick() != through_tick) return false;
    owner->Flush();
    owner->WaitSubmitted(through_tick);
    return true;
}

bool Rasterizer::WaitAscOnGraphicsToken(const iss::g4::QidTick& sealed_g) {
    if (!cp_split_enabled || !compute_scheduler || !liverpool ||
        sealed_g.lane != iss::g4::CpLane::Graphics || sealed_g.qid != 0 ||
        sealed_g.generation != liverpool->CpGraphicsGeneration() ||
        sealed_g.tick == 0) return false;
    compute_scheduler->AddCrossWait(scheduler.GetWorkSemaphore()->Handle(), sealed_g.tick);
    return compute_scheduler->CrossWaitTick(scheduler.GetWorkSemaphore()->Handle()) >=
           sealed_g.tick;
}

bool Rasterizer::AscSchedulerIdleAfterSeal(u64 sealed_tick) const noexcept {
    return compute_scheduler && sealed_tick != 0 &&
           compute_scheduler->SubmittedTick() >= sealed_tick &&
           compute_scheduler->CurrentTick() > sealed_tick &&
           !compute_scheduler->HasWork();
}

bool Rasterizer::WaitAscGpuComplete(u64 sealed_tick) {
    if (!AscSchedulerIdleAfterSeal(sealed_tick)) return false;
    compute_scheduler->Wait(sealed_tick);
    return compute_scheduler->IsFree(sealed_tick);
}

Rasterizer::AscArenaControlResult Rasterizer::PrepareAscArenaOnGraphics(
    std::span<const std::pair<VAddr,u32>> ranges) {
    AscArenaControlResult out{};
    if (!cp_split_enabled || !liverpool->CpSplitActive() ||
        !liverpool->AscOwnerPausedForGraphics() || ranges.empty() || DrawQueue() != 0)
        return out;
    const u64 mapping_generation = buffer_cache.MappingGeneration();
    for (const auto [address,size] : ranges) {
        if (!address || !size || buffer_cache.HasGpuData(address,size) ||
            texture_cache.ProbeNoRegisteredImage(address,size).result !=
                iss::g4::NegativeImageResult::NoRegisteredPage) return out;
    }
    auto admission = TryPrepareGraphicsGpuPacket();
    if (admission.status == GraphicsHostWriteStatus::Busy) {
        out.status = AscArenaControlStatus::Busy;
        return out;
    }
    if (admission.status != GraphicsHostWriteStatus::Ready) return out;
    if (!ResolveGraphicsGpuPacket(admission.guard) ||
        !PublishGraphicsGpuPacket(admission.guard)) {
        FinishGraphicsGpuPacket(admission.guard);
        out.status = AscArenaControlStatus::Busy;
        return out;
    }
    u64 last_tick{};
    for (const auto [address,size] : ranges) {
        const auto prepared = buffer_cache.PrepareArenaForAsc(address,size);
        // A partial G upload is already recorded and cannot be rolled back.
        ASSERT_MSG(prepared && prepared->mapping_generation == mapping_generation,
                   "G4 ASC arena preflight failed after graphics emission");
        last_tick = std::max(last_tick,prepared->graphics_tick);
    }
    FinishGraphicsGpuPacket(admission.guard);
    ASSERT_MSG(mapping_generation == buffer_cache.MappingGeneration() && last_tick,
               "G4 ASC arena mapping changed after graphics preflight");
    out.status = AscArenaControlStatus::Ready;
    out.graphics_tick = admission.guard.final_tick;
    out.mapping_generation = mapping_generation;
    return out;
}

u64 Rasterizer::CurrentSubmissionGeneration() const noexcept {
    return liverpool ? liverpool->CurrentSubmissionGeneration() : 0;
}

Rasterizer::~Rasterizer() {
    ReportNativeItems();
    draw_pipe_state.reset();
    liverpool->StopScout();
    if (compute_scheduler) {
        buffer_cache.SetCrossQueueTracking(nullptr);
        compute_scheduler->Finish();
    }
}

void Rasterizer::EnsureComputeQueueInitialized() {
    if (!compute_scheduler) {
        static constexpr u64 ComputeStreamSize = 64_MB;
        // A compute-only family queue (iss::asc_family) needs a command pool of that family.
        iss::asc_family::pool_family = iss::asc_family::family;
        struct RestorePoolFamily {
            ~RestorePoolFamily() { iss::asc_family::pool_family = ~0u; }
        } restore_pool_family;
        auto new_scheduler = std::make_unique<Scheduler>(instance, instance.GetComputeQueue(),
                                                         "compute");
        new_scheduler->SetCpRecordingLane(iss::g4::CpLane::Asc);
        new_scheduler->SetPostSubmitCallback([this](u64 tick) {
            if (liverpool->CpSplitActive()) {
                [[maybe_unused]] const bool published =
                    liverpool->PublishCpSubmitted(iss::g4::CpLane::Asc, tick);
            }
        });
        iss::asc_family::pool_family = ~0u;
        auto new_runtime = std::make_unique<Runtime>(instance, *new_scheduler);
        auto new_stream = std::make_unique<VideoCore::StreamBuffer>(
            instance, *new_scheduler, VideoCore::MemoryType::Stream, ComputeStreamSize);
        new_scheduler->SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
            compute_runtime->FlushBarriers();
            buffer_cache.SubmitPendingArenaBinds(info, false);
            buffer_cache.WaitArenaBinds(info);
        });
        compute_scheduler = std::move(new_scheduler);
        compute_runtime = std::move(new_runtime);
        compute_stream = std::move(new_stream);
        if (AscQueueEnabled()) {
            // ISS_ASC_NATIVE_QUEUE: a graphics session runs after the compute work recorded
            // before its submission (the unified heap and readback stamps of that work carry the
            // graphics tick of the session recorded with it).
            scheduler.SetBeforeSubmitCallback([this] { AscFlush(); });
        }
        compute_context = VideoCore::QueueContext{
            .scheduler = compute_scheduler.get(),
            .runtime = compute_runtime.get(),
            .stream_buffer = compute_stream.get(),
            .index = 1,
        };
        LOG_INFO(Render_Vulkan, "asc_queue: compute work on a queue of its own");
    }
}

bool Rasterizer::UseComputeQueue(const ComputePipeline* pipeline) {
    if (!DrawAsync() || !instance.HasComputeQueue() ||
        !Common::Perf::GetTuning().asc_queue.load(std::memory_order_relaxed)) {
        return false;
    }
    // Buffers only: the images (layouts, barriers) are tracked for the graphics queue; the
    // descriptors must be pushed (sets are recycled with the graphics timeline).
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (!cs.images.empty() || !cs.samplers.empty() || cs.uses_dma ||
        !pipeline->UsesPushDescriptors()) {
        Common::Perf::Add(Common::Perf::AscQueueFallbacks);
        iss::asc_stats::fallback[!cs.images.empty() ? 0 : !cs.samplers.empty() ? 1
                                 : cs.uses_dma ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    EnsureComputeQueueInitialized();
    if (!compute_queue_routing_enabled) {
        buffer_cache.SetCrossQueueTracking(&compute_context);
        compute_queue_routing_enabled = true;
    }
    return true;
}

void Rasterizer::FlushComputeQueue() {
    DrainDrawPipe(5);
    if (compute_queue_routing_enabled) {
        buffer_cache.FlushComputeQueue();
    }
}

bool Rasterizer::FilterDraw() {
    const auto& regs = DrawRegs();
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::PrepareRender};
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = DrawRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    // Render target memo: a slot whose registers and size hint (given by the commands, not
    // learned by the cache) are those its description was built from keeps its image while the
    // cache and the image do not change. The generation is read before the search: a change
    // during it leaves the entry invalid until the next draw finds the image again.
    const bool use_rt_memo = Common::Perf::GetTuning().rt_memo.load(std::memory_order_relaxed);
    const auto found_again = [&](VideoCore::ImageId image_id, u64& touched_gc_tick) {
        bound_images.emplace_back(image_id);
        auto& image = texture_cache.GetImage(image_id);
        image.tick_accessed_last = scheduler.CurrentTick();
        if (touched_gc_tick != texture_cache.GcTick()) {
            // As FindImage does, for the garbage collector (once per frame).
            texture_cache.TouchImageLocked(image_id);
            touched_gc_tick = texture_cache.GcTick();
        }
        image.binding.is_target = 1u;
        Common::Perf::Add(Common::Perf::RtMemoHits);
    };
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        auto hint = DrawCbExtents()[cb];
        const bool memo_candidate = use_rt_memo && hint.Valid();
        auto& memo = rt_memos[cb];
        if (memo_candidate && memo.valid && image_id && memo.image_id == image_id &&
            memo.hint == hint.raw &&
            std::memcmp(memo.regs.data(), &col_buf, sizeof(memo.regs)) == 0 &&
            memo.generation == texture_cache.Generation() &&
            memo.image_generation == texture_cache.ImageGeneration(image_id)) {
            found_again(image_id, memo.touched_gc_tick);
            continue;
        }
        if (!hint.Valid()) {
            hint = texture_cache.LearnedTargetExtent(col_buf.Address(), col_buf.Pitch());
        }
        const u64 generation = memo_candidate ? texture_cache.Generation() : 0;
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        if (memo_candidate) {
            Common::Perf::Add(Common::Perf::RtMemoMisses);
            memo.valid = generation == texture_cache.Generation();
            memo.hint = hint.raw;
            memo.generation = generation;
            memo.image_generation = texture_cache.ImageGeneration(image_id);
            memo.image_id = image_id;
            memo.touched_gc_tick = texture_cache.GcTick();
            std::memcpy(memo.regs.data(), &col_buf, sizeof(memo.regs));
        } else {
            memo.valid = false;
        }
    }

    // The PS4 depth block counts the samples of occlusion queries even when the depth test is
    // off (a query without depth test measures the full size of the tested shape). Host GPUs
    // count them in their depth unit: the depth buffer stays attached while a query is open.
    const bool depth_for_query = occlusion.HasOpen() && regs.depth_buffer.DepthValid();
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid()) ||
        depth_for_query) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        auto hint = DrawDbExtent();
        auto& [image_id, desc] = db_desc;
        static_assert(sizeof(regs.depth_view) == sizeof(u32) &&
                      sizeof(regs.depth_control) == sizeof(u32));
        u32 depth_view;
        u32 depth_control;
        std::memcpy(&depth_view, &regs.depth_view, sizeof(u32));
        std::memcpy(&depth_control, &regs.depth_control, sizeof(u32));
        const bool memo_candidate = use_rt_memo && hint.Valid();
        auto& memo = db_memo;
        if (memo_candidate && memo.valid && image_id && memo.image_id == image_id &&
            memo.hint == hint.raw && memo.htile_address == htile_address &&
            memo.depth_view == depth_view && memo.depth_control == depth_control &&
            std::memcmp(memo.regs.data(), &regs.depth_buffer, sizeof(memo.regs)) == 0 &&
            memo.generation == texture_cache.Generation() &&
            memo.image_generation == texture_cache.ImageGeneration(image_id)) {
            found_again(image_id, memo.touched_gc_tick);
        } else {
            if (!hint.Valid()) {
                hint = texture_cache.LearnedTargetExtent(regs.depth_buffer.DepthAddress(),
                                                         regs.depth_buffer.Pitch());
            }
            const u64 generation = memo_candidate ? texture_cache.Generation() : 0;
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            auto& image = texture_cache.GetImage(image_id);
            image.binding.is_target = 1u;
            if (memo_candidate) {
                Common::Perf::Add(Common::Perf::RtMemoMisses);
                memo.valid = generation == texture_cache.Generation();
                memo.hint = hint.raw;
                memo.generation = generation;
                memo.image_generation = texture_cache.ImageGeneration(image_id);
                memo.image_id = image_id;
                memo.touched_gc_tick = texture_cache.GcTick();
                memo.htile_address = htile_address;
                memo.depth_view = depth_view;
                memo.depth_control = depth_control;
                std::memcpy(memo.regs.data(), &regs.depth_buffer, sizeof(memo.regs));
            } else {
                memo.valid = false;
            }
        }
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    BumpBindEpoch();
    auto& col_buf = DrawRegs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, DrawCbExtents()[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

static bool FastDrawEnabled();
static void NoteDrawCensus(const RenderState& state, u32 path, const GraphicsPipeline* pipeline,
                           const AmdGpu::Regs& regs);

namespace {
// ISS_DRAW_CYCLES=1: time stamp counter cycles of each phase of the direct draw path, and of the
// command processor work between draws, reported per draw every 65536 draws.
bool LazyPipelinesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_LAZY_PIPELINES");
        return value && value[0] == '1';
    }();
    return enabled;
}
// ISS_ASYNC_SKIP_DRAWS=1 (gow3_pc style, off by default): a graphics draw whose pipeline is still
// compiling is skipped instead of waited for (no stall on new shaders; the object or effect
// appears a few frames later). Dispatches still wait: their results are read afterwards.
bool AsyncSkipDrawsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_ASYNC_SKIP_DRAWS");
        const bool on = value && value[0] == '1';
        if (on) std::fprintf(stderr, "ASYNC_SKIP_DRAWS on: draws of compiling pipelines are skipped\n");
        return on;
    }();
    return enabled;
}
bool SkipUnreadyDraws() {
    return !LazyPipelinesEnabled() || AsyncSkipDrawsEnabled();
}
// Whether this draw of a compiling pipeline is skipped. With ISS_ASYNC_SKIP_DRAWS a skip must not
// break the whole frame: a full screen pass (a few vertices: lighting, sky, composite, post
// effects) waits for its pipeline. Skipping them when the world spawns showed the raw G-buffer, a
// flat yellow-green screen. ISS_ASYNC_SKIP_FRAME_CAP=N (off by default) also makes every draw
// after N skips in a frame wait: with a cold cache the spawn frame then waited ~500 pipelines (6 s).
bool SkipUnreadyDraw(u32 num_vertices, u32 num_instances) {
    if (!SkipUnreadyDraws()) return false;
    if (!AsyncSkipDrawsEnabled()) return true;
    static const u32 frame_cap = [] {
        const char* value = std::getenv("ISS_ASYNC_SKIP_FRAME_CAP");
        return value ? static_cast<u32>(std::atoi(value)) : ~0u;
    }();
    if (num_vertices <= 6 && num_instances <= 1) return false;
    thread_local u64 frame = ~0ull;
    thread_local u32 skips = 0;
    const u64 now = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    if (now != frame) {
        frame = now;
        skips = 0;
    }
    if (skips >= frame_cap) return false;
    ++skips;
    return true;
}
struct DrawCycles {
    bool enabled = [] {
        const char* value = std::getenv("ISS_DRAW_CYCLES");
        return value && value[0] == '1';
    }();
    std::array<u64, 16> cycles{};
    u64 draws{};
    u64 last{};
    void Mark(u32 phase) {
        const u64 now = __rdtsc();
        if (last != 0) {
            cycles[phase] += now - last;
        }
        last = now;
    }
    void Report() {
        static constexpr std::array names{"prologue", "pipeline", "prepare_rt", "bind_res",
                                          "begin_rend", "vertex", "index", "flush_barr",
                                          "push_desc", "dyn_state", "sched_begin", "record",
                                          "reset", "between", "skipped", "unused"};
        std::string line = "DRAW_CYCLES per_draw";
        u64 total = 0;
        for (u32 i = 0; i < cycles.size(); ++i) {
            total += cycles[i];
            line += fmt::format(" {}={:.0f}", names[i], double(cycles[i]) / double(draws));
        }
        line += fmt::format(" total={:.0f} draws={}", double(total) / double(draws), draws);
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
        cycles = {};
        draws = 0;
    }
};
DrawCycles g_draw_cycles;
} // namespace

// ISS_LAZY_PIPELINES=1: a new graphics pipeline is compiled by the compiler workers while the
// command processor goes on; its bind waits for it on the recording thread, when the commands
// run in order (a burst of new pipelines compiles in parallel instead of one after another).
static std::atomic<u64> g_lazy_binds{}, g_lazy_wait_ns{}, g_lazy_steals{};
extern "C" void IssPreciseSleepUs(unsigned us); // bridge.cpp
static void WaitPipelineReady(const Pipeline* pipeline, PipelineCache* cache = nullptr) {
    if (pipeline->IsReady()) {
        return;
    }
    const u64 start = Common::Perf::NowNs();
    // Not started by a worker yet (they are busy with the rest of the burst): compile it here.
    if (cache && cache->CompileNowIfQueued(pipeline)) {
        g_lazy_steals.fetch_add(1, std::memory_order_relaxed);
    }
    for (u32 n = 0; !pipeline->IsReady(); ++n) {
        if (n < 256) {
            _mm_pause();
        } else if (n < 2048) {
            std::this_thread::yield();
        } else {
            // Was std::this_thread::sleep_for(50 us): Sleep(1) on Windows, ~15.6 ms without a
            // 1 ms timer resolution, added to every compile the recording thread waited for.
            IssPreciseSleepUs(100);
        }
    }
    const u64 waited = g_lazy_wait_ns.fetch_add(Common::Perf::NowNs() - start) +
                       (Common::Perf::NowNs() - start);
    (void)waited;
}

void Rasterizer::BindGraphicsPipeline(const GraphicsPipeline* pipeline) {
    BindPipelineLazy(scheduler, vk::PipelineBindPoint::eGraphics, pipeline);
}

void Rasterizer::BindPipelineLazy(Scheduler& on, vk::PipelineBindPoint bind_point,
                                  const Pipeline* pipeline) {
    const auto cmdbuf = on.Recorder();
    if (pipeline->IsReady()) {
        cmdbuf.bindPipeline(bind_point, pipeline->Handle());
        return;
    }
    const u64 n = g_lazy_binds.fetch_add(1) + 1;
    if ((n & 63) == 0) {
        std::fprintf(stderr, "LAZY_PIPELINES binds=%llu recorder_wait_ms=%.1f steals=%llu\n",
                     (unsigned long long)n, double(g_lazy_wait_ns.load()) / 1e6,
                     (unsigned long long)g_lazy_steals.load());
    }
    if (on.DirectCommandBuffer()) {
        // Recording on this thread: the pipeline is needed now.
        WaitPipelineReady(pipeline, &pipeline_cache);
        cmdbuf.bindPipeline(bind_point, pipeline->Handle());
        return;
    }
    // The bind cache must not take a later bind of the previous pipeline for redundant.
    auto& binds = on.GetDynamicState().binds;
    binds.pipelines[bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0] = vk::Pipeline{};
    ++binds.epoch;
    on.Stream().Record([pipeline, bind_point, cache = &pipeline_cache](vk::CommandBuffer c) {
        WaitPipelineReady(pipeline, cache);
        c.bindPipeline(bind_point, pipeline->Handle());
    });
}

std::size_t Rasterizer::Draw(bool is_indexed, u32 index_offset,
                             std::span<const u32> following_commands, std::size_t native_packet_budget) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::Draw};
    ObserveNativeVisualDraw();
    if (g_draw_cycles.enabled) g_draw_cycles.Mark(13);
    const bool pipelined=UseDrawPipe() && FilterDrawPasses();
    // ISS_NATIVE_PASSES learn request: the full path (DrawRecord), which keeps the template.
    const bool pass_learn = native_pass_learn != 0 && !pipelined;
    // Pause guard (pause_guard.h, FastDraw): in the pause the native/fast paths below are off.
    const bool native_off = pass_learn || iss::pause_guard::Blocked(iss::pause_guard::FastDraw);
    if(!pipelined){DrainDrawPipe(5);SubmitFeedbackBatch(scheduler);PeriodicSubmit(scheduler);}
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Draw);
    RENDERER_TRACE;

    if(!pipelined)scheduler.PopPendingOperations();

    if (!pipelined && !FilterDraw()) {
        return 0;
    }

    const auto& regs = DrawRegs();
    if (regs.num_indices == 0) [[unlikely]] {
        // No vertex to draw: nothing the GPU would do (skip_empty_draws), before looking the
        // pipeline and the resources up.
        Common::Perf::Add(Common::Perf::EmptyDraws);
        if (Common::Perf::GetTuning().skip_empty_draws.load(std::memory_order_relaxed)) {
            return 0;
        }
    }
    if (!native_off && !iss::native_passes::Enabled() && !pipelined && is_indexed &&
        index_offset == 0 && !following_commands.empty() && native_packet_budget &&
        NativeDepthRunEnabled()) {
        // Before the fast draw, which would take this draw alone: a depth family draw following
        // one of the same programs starts a run (its pipeline looked up as Draw does).
        if (const auto* candidate = DepthRunCandidate()) {
            if (const auto consumed = TryNativeDepthRun(candidate, following_commands,
                                                        native_packet_budget))
                return consumed;
        }
    }
    if (!native_off && !pipelined && FastDrawEnabled()) {
        const u64 t0 = g_draw_cycles.enabled ? __rdtsc() : 0;
        const u64 rebinds = fast_draw_stats.rebinds;
        const u64 leans = fast_draw_stats.leans;
        const u64 switches = fast_draw_stats.switches;
        const bool fast = TryFastDraw(is_indexed, index_offset);
        if (g_draw_cycles.enabled) {
            const u64 dt = __rdtsc() - t0;
            auto& s = fast_draw_stats;
            if (!fast) {
                s.cyc_fail += dt;
            } else if (s.switches != switches) {
                s.cyc_switch += dt;
            } else if (s.leans != leans) {
                s.cyc_lean += dt;
            } else if (s.rebinds != rebinds) {
                s.cyc_rebind += dt;
            } else {
                s.cyc_plain += dt;
            }
        }
        if (fast) {
            if (g_draw_cycles.enabled) g_draw_cycles.Mark(14);
            return 0;
        }
    }
    if (!native_off && !pipelined && TryNativeItemDraw(is_indexed, index_offset)) return 0;
    if (!pipelined && Common::Perf::GetTuning().bind_prefetch.load(std::memory_order_relaxed)) {
        // Read while the pipeline is looked up: the bindings find their memo entries cached.
        PrefetchDrawLookups(is_indexed, index_offset);
    }
    if (g_draw_cycles.enabled) g_draw_cycles.Mark(0);
    const GraphicsPipeline* pipeline =
        fast_lookup_set ? fast_lookup : LookupNativeGraphicsPipeline();
    fast_lookup_set = false;
    if (g_draw_cycles.enabled) g_draw_cycles.Mark(1);
    if (!pipeline) {
        return 0;
    }
    // Diagnostic only (wrong image): ISS_DIAG_SKIP_DRAWS=1 drops the dominant city draw family,
    // 2 drops every direct draw, to measure how the frame rate follows command processor work.
    static const int diag_skip = [] {
        const char* value = std::getenv("ISS_DIAG_SKIP_DRAWS");
        return value ? std::atoi(value) : 0;
    }();
    if (diag_skip != 0) [[unlikely]] {
        const auto* vs = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Vertex)];
        const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
        if (diag_skip == 2 || (vs && ps && vs->pgm_hash == 0x0BEB7156 &&
                               ps->pgm_hash == 0x6353E48B)) {
            return 0;
        }
    }
    if (!pipeline->IsReady() &&
        SkipUnreadyDraw(DrawRegs().num_indices, DrawRegs().num_instances.NumInstances())) {
        // Still compiled by a worker thread, the draw is skipped instead of stalling.
        Common::Perf::Add(Common::Perf::PipelineSkips);
        if (AsyncSkipDrawsEnabled()) pipeline_cache.PrioritizeCompile(pipeline);
        RecordSkippedWrites(*pipeline);
        if (g_dump.frames_left != 0) [[unlikely]] {
            DumpCommand("draw-skipped", pipeline, "");
        }
        return 0;
    }

    if(pipelined && PostDrawPipe(pipeline,false,is_indexed,index_offset))return 0;
    if(pipelined){DrainDrawPipe(5);SubmitFeedbackBatch(scheduler);scheduler.PopPendingOperations();}
    if (!native_off && !iss::native_passes::Enabled() && !pipelined && is_indexed && index_offset == 0 && !following_commands.empty() && native_packet_budget) {
        if (const auto consumed = TryNativeCpuRun(pipeline, following_commands, native_packet_budget)) return consumed;
        if (const auto consumed = TryNativeIndexedRun(pipeline, following_commands, native_packet_budget)) return consumed;
        if (const auto consumed = TryNativeInstancedRun(pipeline, following_commands, native_packet_budget)) return consumed;
    }
    DrawRecord(pipeline,is_indexed,index_offset);
    return 0;
}

#include "native_draw_run_rasterizer.inl"
#include "native_cpu_draw_run.inl"
#include "native_instanced_run_rasterizer.inl"

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline,bool is_indexed,u32 index_offset) {
    if (NativeVisualCaptureEnabled()) native_visual_latch = {};
    std::optional<iss::diagnostics::Scope> worker_scope;
    if(OnDrawWorker())worker_scope.emplace(iss::diagnostics::Draw);
    const auto& regs=DrawRegs();
    // Statistics: draws a fast path for unchanged render state could take.
    if (const u64 changes = DrawContextChanges(); changes == last_draw_context_changes) {
        Common::Perf::Add(Common::Perf::DrawsSameContext);
        if (pipeline == last_draw_pipeline) {
            Common::Perf::Add(Common::Perf::DrawsSamePipeline);
        }
    } else {
        last_draw_context_changes = changes;
    }
    last_draw_pipeline = pipeline;

    const bool cycles = g_draw_cycles.enabled && !OnDrawWorker();
    fast_draw.valid = false;
    PrepareRenderState(pipeline);
    if (cycles) g_draw_cycles.Mark(2);
    // ISS_NATIVE_PASSES: the template of a learn request's draw, under the key of its registers
    // (indexed or DRAW_INDEX_AUTO: the same pipeline and resources).
    const u64 pass_learn_key = native_pass_learn && !OnDrawWorker()
                                   ? NativePassLearnKey(native_pass_learn) : 0;
    native_pass_learn = 0;
    std::vector<NativeImageCapture> pass_capture;
    if (pass_learn_key) native_image_capture = &pass_capture;
    const bool bound_resources = BindResources(pipeline);
    if (pass_learn_key) native_image_capture = nullptr;
    if (!bound_resources) {
        return;
    }
    if (is_indexed && native_visual_latch.material_verified) CompleteNativeVisualPlans(pipeline);
    if (cycles) g_draw_cycles.Mark(3);
    if (iss::command_trace::enabled) {
        for (const auto* stage : pipeline->GetStages()) if (stage)
            iss::command_trace::Stage("draw", DrawQueue(), *stage,
                                      stage->ProgramBase(), 0, 0, 0, scheduler.CurrentTick(),
                                      needs_barrier, 0, 0);
    }
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpCommand(is_indexed ? "drawidx" : "draw", pipeline,
                    fmt::format("n{} i{} prim{}", regs.num_indices,
                                regs.num_instances.NumInstances(), u32(regs.primitive_type)));
    }
    auto state = BeginRendering(pipeline);
    if (native_recording) native_recording->render_state = state;
    if (cycles) g_draw_cycles.Mark(4);

    iss::native_geometry::ProbeSecondaryDraw(
        *pipeline, regs, *memory, buffer_cache,
        DrawQueue(), index_offset, scheduler.CurrentTick());
    BindVertexBuffers(pipeline);
    if (cycles) g_draw_cycles.Mark(5);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }
    if (cycles) g_draw_cycles.Mark(6);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    if (cycles) g_draw_cycles.Mark(7);

    pipeline->BindResources(set_writes, push_data);
    if (cycles) g_draw_cycles.Mark(8);
    UpdateDynamicState(pipeline, is_indexed);
    if (cycles) g_draw_cycles.Mark(9);
    scheduler.BeginRendering(state);
    if (cycles) g_draw_cycles.Mark(10);
    if (pass_learn_key && !attachment_feedback_loop) {
        native_image_capture = &pass_capture;
        // ISS_LEARN_STATS=1 (diagnosis): the cost of building templates, every 4096 learns.
        static const bool learn_stats = [] {
            const char* p = std::getenv("ISS_LEARN_STATS");
            return p && p[0] == '1';
        }();
        const u64 learn_start = learn_stats ? Common::Perf::NowNs() : 0;
        NativePassLearn(pass_learn_key, pipeline, state);
        if (learn_stats) {
            static u64 learns = 0, learn_ns = 0;
            learn_ns += Common::Perf::NowNs() - learn_start;
            if ((++learns & 4095) == 0)
                std::fprintf(stderr, "LEARN_STATS learns=%llu avg_us=%.1f\n",
                             (unsigned long long)learns, double(learn_ns) / 1000.0 / double(learns));
        }
        native_image_capture = nullptr;
    }

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    if (iss::command_trace::enabled && vs_info.pgm_hash == 0x0BEB7156 &&
        pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash == 0x6353E48B) {
        std::array<u8, 3> attribute_ownership{};
        for (u32 i = 0; i < std::min<u32>(3, fetch_shader.attributes.size()); ++i) {
            const auto sharp = fetch_shader.attributes[i].GetSharp(vs_info);
            if (!sharp.base_address || !sharp.GetSize()) continue;
            attribute_ownership[i] =
                (buffer_cache.HasGpuData(sharp.base_address, sharp.GetSize()) ? 1 : 0) |
                (buffer_cache.IsRegionCpuModified(sharp.base_address, sharp.GetSize()) ? 2 : 0) |
                (memory->ClampRangeSize(sharp.base_address, sharp.GetSize()) == sharp.GetSize()
                     ? 4 : 0);
        }
        const bool index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = index16 ? sizeof(u16) : sizeof(u32);
        const VAddr index_address = regs.index_base_address.Address<VAddr>() +
                                    u64{index_offset} * index_size;
        const u64 index_bytes = u64{regs.num_indices} * index_size;
        const u8 index_ownership = index_bytes && index_bytes <= std::numeric_limits<u32>::max()
            ? (buffer_cache.HasGpuData(index_address, index_bytes) ? 1 : 0) |
              (buffer_cache.IsRegionCpuModified(index_address, index_bytes) ? 2 : 0) |
              (memory->ClampRangeSize(index_address, index_bytes) == index_bytes ? 4 : 0)
            : 0;
        iss::command_trace::VertexFetch(
            DrawQueue(), vs_info, fetch_shader, reinterpret_cast<u64>(pipeline),
            DrawContextChanges(), u32(regs.primitive_type),
            regs.num_instances.NumInstances(), regs.num_indices,
            index_address, regs.index_buffer_type.raw, index_offset,
            attribute_ownership, index_ownership, memory->MappingEpoch(),
            buffer_cache.MappingGeneration());
    }
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.Recorder();
    BindGraphicsPipeline(pipeline);
    if (GpuCrashDiag::Enabled() || GpuCrashDiag::Blackbox()) {
        MarkDraw(scheduler, cmdbuf, *pipeline,
                 is_indexed ? GpuCrashDiag::Kind::DrawIndexed : GpuCrashDiag::Kind::Draw,
                 regs.num_indices, regs.num_instances.NumInstances(), state, 0, [&] {
                     auto ctx = BlackboxContext(regs, 0);
                     if (GpuCrashDiag::Blackbox())
                         BlackboxImages(ctx, bound_images, texture_cache, cb_descs, db_desc);
                     return ctx;
                 }());
    }

    if (is_indexed) {
        if (native_visual_latch.eligible)
            ProbeNativeVisual(pipeline, state, s32(vertex_offset), instance_offset);
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    iss::feedback_join::Recorded();
    DebugState.IncDrawCall();

    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpAfter(pipeline);
    }
    if (cycles) g_draw_cycles.Mark(11);
    if (FastDrawEnabled()) {
        CaptureFastDraw(pipeline, is_indexed, state);
    }
    NoteDrawCensus(state, 4, pipeline, DrawRegs());
    ResetBindings(false);
    if (FastDrawEnabled()) {
        FinishFastDraw();
    }
    if (cycles) {
        g_draw_cycles.Mark(12);
        if (++g_draw_cycles.draws == 65536) {
            g_draw_cycles.Report();
        }
    }
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::DrawIndirect};
    BumpBindEpoch();
    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    // ISS_DRAW_PIPE_DEFER: queued to the draw pipe worker behind the draws before it (the
    // arguments are read there, in order), instead of draining the pipe.
    if (draw_pipe_state && draw_pipe_state->defer && UseDrawPipe() && FilterDrawPasses() &&
        g_dump.frames_left == 0 &&
        Common::Perf::GetTuning().trace_draw.load(std::memory_order_relaxed) == 0) {
        const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
        if (!pipeline) {
            return;
        }
        const std::array<u64, 5> indirect{arg_address, offset, stride, max_count, count_address};
        if (PostDrawPipe(pipeline, false, is_indexed, 0, &indirect)) {
            return;
        }
    }
    DrainDrawPipe(5);
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }
    if (const u64 trace_ps = Common::Perf::GetTuning().trace_draw.load(std::memory_order_relaxed);
        trace_ps != 0) [[unlikely]] {
        // Particle trace: the arguments of this draw as guest memory has them, each frame.
        for (const Shader::Info* stage : pipeline->GetStages()) {
            if (stage && stage->hw_stage == Shader::HwStage::Fragment &&
                stage->pgm_hash == trace_ps) {
                std::array<u32, 4> args{};
                memory->CopySparseMemory(arg_address + offset, reinterpret_cast<u8*>(args.data()),
                                         sizeof(args));
                LOG_INFO(Render_Vulkan,
                         "PTRACE draw f{} ps {:x} args {:x} {},{},{},{} max {} count {:x}{}",
                         Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed),
                         trace_ps, arg_address + offset, args[0], args[1], args[2], args[3],
                         max_count, count_address, pipeline->IsReady() ? "" : " not-ready");
                break;
            }
        }
    }
    DrawIndirectRecord(pipeline, is_indexed, arg_address, offset, stride, max_count,
                       count_address);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    VAddr arg_address, u32 offset, u32 stride, u32 max_count,
                                    VAddr count_address) {
    // The next fast draw compares with a draw recorded by the full path.
    fast_draw.valid = false;
    // The vertex count is in GPU memory: only the frame cap of SkipUnreadyDraw applies.
    if ((!pipeline->IsReady() && SkipUnreadyDraw(~0u, ~0u)) ||
        ReadsSkippedWrites(arg_address + offset, u64{stride} * max_count) ||
        (count_address != 0 && ReadsSkippedWrites(count_address, 4))) {
        Common::Perf::Add(Common::Perf::PipelineSkips);
        if (AsyncSkipDrawsEnabled() && !pipeline->IsReady()) pipeline_cache.PrioritizeCompile(pipeline);
        RecordSkippedWrites(*pipeline);
        if (g_dump.frames_left != 0) [[unlikely]] {
            DumpCommand("drawind-skipped", pipeline,
                        fmt::format("args{:x} max{} count{:x}", arg_address + offset, max_count,
                                    count_address));
        }
        return;
    }

    PrepareRenderState(pipeline);
    // ISS_NATIVE_INDIRECT: a learn request at this indexed indirect draw keeps its template.
    const u64 pass_learn_key = native_pass_learn && native_pass_learn_indirect && is_indexed &&
                                       !OnDrawWorker()
                                   ? NativePassLearnKey(native_pass_learn) : 0;
    native_pass_learn = 0;
    native_pass_learn_indirect = 0;
    std::vector<NativeImageCapture> pass_capture;
    if (pass_learn_key) native_image_capture = &pass_capture;
    const bool bound_resources = BindResources(pipeline);
    if (pass_learn_key) native_image_capture = nullptr;
    if (!bound_resources) {
        return;
    }
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpCommand(is_indexed ? "drawidxind" : "drawind", pipeline,
                    fmt::format("args{:x} stride{} max{} count{:x} prim{}", arg_address + offset,
                                stride, max_count, count_address,
                                u32(DrawRegs().primitive_type)));
    }
    if (iss::command_trace::enabled) {
        for (const auto* stage : pipeline->GetStages()) if (stage)
            iss::command_trace::Stage("draw_indirect", DrawQueue(), *stage,
                                      stage->ProgramBase(), max_count, stride, is_indexed,
                                      scheduler.CurrentTick(), needs_barrier,
                                      arg_address + offset, count_address);
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);
    if (g_dump.frames_left != 0) [[unlikely]] {
        ProbeBuffer("args", buffer, base, std::min<u32>(stride * max_count, 32),
                    arg_address + offset);
    }

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);
    if (pass_learn_key && !attachment_feedback_loop) {
        native_image_capture = &pass_capture;
        NativePassLearn(pass_learn_key, pipeline, state);
        native_image_capture = nullptr;
    }

    const auto cmdbuf = scheduler.Recorder();
    BindGraphicsPipeline(pipeline);
    if (GpuCrashDiag::Enabled() || GpuCrashDiag::Blackbox()) {
        MarkDraw(scheduler, cmdbuf, *pipeline,
                 is_indexed ? GpuCrashDiag::Kind::DrawIndexedIndirect
                            : GpuCrashDiag::Kind::DrawIndirect,
                 max_count, stride, state, arg_address + offset, [&] {
                     auto ctx = BlackboxContext(DrawRegs(), 0);
                     if (GpuCrashDiag::Blackbox())
                         BlackboxImages(ctx, bound_images, texture_cache, cb_descs, db_desc);
                     return ctx;
                 }());
    }

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_offset,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }
    iss::feedback_join::Recorded();

    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpAfter(pipeline);
    }
    ResetBindings(false);
}

std::optional<Rasterizer::PreparedComputeInputs> Rasterizer::PrepareComputeInputs(
    const ComputePipeline* pipeline, std::optional<iss::g4::IndirectArgsRef> indirect_args) {
    if (!pipeline) return std::nullopt;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    const auto& cs = DrawCsRegs();
    // This HLE route uses DrawRegs and CPU-visible control data not in the packet.
    if (stage.pgm_hash == 0xfefebf9f || stage.uses_dma || !stage.images.empty() ||
        !stage.fmasks.empty() || !stage.samplers.empty()) return std::nullopt;
    if (!std::ranges::equal(stage.user_data, cs.user_data)) return std::nullopt;

    iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
    auto srt = iss::g4::CaptureSrtSnapshot(stage.srt_info, cs.user_data, provider);
    if (srt.proof.coverage != iss::g4::SrtCoverage::Complete ||
        srt.flattened != stage.flattened_ud_buf ||
        !iss::g4::RevalidateSrtSnapshot(srt, provider)) return std::nullopt;

    PreparedComputeInputs prepared{};
    prepared.cs = cs;
    // Compute shaders only use the buffer-offset portion of PushData. The
    // viewport fields are consumed by vertex clip-space conversion; reading
    // G's live viewport registers from the C worker would be a data race.
    prepared.push_data = {};
    prepared.mapping_generation = buffer_cache.MappingGeneration();
    prepared.native_mapping_epoch = memory->MappingEpoch();
    prepared.srt = std::move(srt);
    prepared.indirect_args = indirect_args;
    for (const auto& read : prepared.srt.reads) {
        if (!buffer_cache.PlanGuestAccess(read.guest_address, read.bytes.size(), false,
                                          prepared.mapping_generation, prepared.accesses))
            return std::nullopt;
    }
    for (const auto& desc : stage.buffers) {
        switch (desc.buffer_type) {
        case Shader::BufferType::Guest: {
            const auto sharp = desc.GetSharp(stage);
            if (sharp.base_address == 0 || sharp.GetSize() == 0) break;
            const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
            if (!size || !buffer_cache.PlanGuestAccess(sharp.base_address, size,
                                                        desc.is_written,
                                                        prepared.mapping_generation,
                                                        prepared.accesses)) return std::nullopt;
            break;
        }
        case Shader::BufferType::GdsBuffer:
            VideoCore::BufferCache::PlanGdsAccess(desc.is_written, prepared.accesses);
            break;
        case Shader::BufferType::BdaPagetable:
            VideoCore::BufferCache::PlanBdaAccess(prepared.accesses);
            break;
        case Shader::BufferType::Flatbuf:
        case Shader::BufferType::SharedMemory:
            break; // owning flat words or owner-local stream storage
        default:
            return std::nullopt; // BDA/fault/other shared resources need a separate proof
        }
    }
    if (indirect_args) {
        const auto& ref = *indirect_args;
        if (!ref.gpu_address || !ref.size ||
            ref.backing_generation != prepared.mapping_generation ||
            !buffer_cache.PlanGuestAccess(ref.gpu_address, ref.size, false,
                                          prepared.mapping_generation, prepared.accesses))
            return std::nullopt;
    }
    if (!prepared.native_mapping_epoch ||
        memory->MappingEpoch() != prepared.native_mapping_epoch ||
        buffer_cache.MappingGeneration() != prepared.mapping_generation ||
        !iss::g4::RevalidateSrtSnapshot(prepared.srt, provider)) return std::nullopt;
    return prepared;
}

std::optional<Rasterizer::PreparedComputeInputs> Rasterizer::PrepareCopyHleInputs(
    const ComputePipeline* pipeline) {
    if (!pipeline) return std::nullopt;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    const auto& cs = DrawCsRegs();
    if (stage.pgm_hash != 0xfefebf9f || stage.buffers.size() < 3 ||
        !stage.images.empty() || !stage.fmasks.empty() || !stage.samplers.empty())
        return std::nullopt;
    if (stage.buffers[0].buffer_type != Shader::BufferType::Guest ||
        stage.buffers[1].buffer_type != Shader::BufferType::Guest ||
        stage.buffers[2].buffer_type != Shader::BufferType::Guest ||
        stage.buffers[0].is_written || stage.buffers[1].is_written ||
        !stage.buffers[2].is_written) return std::nullopt;
    const auto& ctl = stage.buffers[0].GetSharp(stage);
    const auto& src = stage.buffers[1].GetSharp(stage);
    const auto& dst = stage.buffers[2].GetSharp(stage);
    const u64 control_bytes = u64{cs.dim_x} * sizeof(iss::g4::CopyHleControl);
    if (ctl.GetStride() != sizeof(iss::g4::CopyHleControl) ||
        !ctl.base_address || !src.base_address || !dst.base_address ||
        src.GetStride() == 0 || src.GetStride() != dst.GetStride() ||
        control_bytes == 0 || control_bytes > 128_KB ||
        control_bytes > ctl.GetSize()) return std::nullopt;
    const std::array extra{iss::g4::SrtExtraRead{ctl.base_address,
                                                 static_cast<u32>(control_bytes)}};
    iss::g4::TrackedSrtMemoryProvider provider{*memory,page_manager,buffer_cache};
    auto capture = iss::g4::CaptureSrtSnapshot(stage.srt_info,cs.user_data,provider,extra);
    if (capture.proof.coverage != iss::g4::SrtCoverage::Complete ||
        capture.flattened != stage.flattened_ud_buf ||
        capture.reads.empty() || capture.reads.back().guest_address != ctl.base_address ||
        capture.reads.back().bytes.size() != control_bytes ||
        !iss::g4::RevalidateSrtSnapshot(capture,provider)) return std::nullopt;
    std::vector<iss::g4::CopyHleControl> controls(cs.dim_x);
    std::memcpy(controls.data(),capture.reads.back().bytes.data(),control_bytes);
    auto copy_plan = iss::g4::BuildCopyHlePlan(controls,src.GetStride(),src.GetSize(),
                                               dst.GetSize());
    if (!copy_plan) return std::nullopt;
    PreparedComputeInputs prepared{};
    prepared.cs = cs;
    prepared.push_data = {};
    prepared.mapping_generation = buffer_cache.MappingGeneration();
    prepared.native_mapping_epoch = memory->MappingEpoch();
    prepared.srt = std::move(capture);
    for (const auto& read : prepared.srt.reads) {
        if (!buffer_cache.PlanGuestAccess(read.guest_address,read.bytes.size(),false,
                                          prepared.mapping_generation,prepared.accesses))
            return std::nullopt;
    }
    PreparedResourceBindings bindings{};
    bindings.hle_copies.reserve(copy_plan->batches.size());
    for (const auto& batch : copy_plan->batches) {
        if (src.base_address > std::numeric_limits<VAddr>::max()-batch.src_begin ||
            dst.base_address > std::numeric_limits<VAddr>::max()-batch.dst_begin)
            return std::nullopt;
        const VAddr src_addr = src.base_address+batch.src_begin;
        const VAddr dst_addr = dst.base_address+batch.dst_begin;
        const u64 src_size = batch.src_end-batch.src_begin;
        const u64 dst_size = batch.dst_end-batch.dst_begin;
        if (!src_size || !dst_size || src_size > std::numeric_limits<u32>::max() ||
            dst_size > std::numeric_limits<u32>::max() ||
            !buffer_cache.PlanGuestAccess(src_addr,src_size,false,
                                          prepared.mapping_generation,prepared.accesses) ||
            !buffer_cache.PlanGuestAccess(dst_addr,dst_size,true,
                                          prepared.mapping_generation,prepared.accesses))
            return std::nullopt;
        auto src_negative = texture_cache.ProbeNoRegisteredImage(src_addr,src_size);
        if (src_negative.result != iss::g4::NegativeImageResult::NoRegisteredPage)
            return std::nullopt;
        auto negative = texture_cache.ProbeNoRegisteredImage(dst_addr,dst_size);
        if (negative.result != iss::g4::NegativeImageResult::NoRegisteredPage)
            return std::nullopt;
        bindings.negative_images.push_back(src_negative);
        bindings.negative_images.push_back(negative);
        auto src_binding = buffer_cache.PrepareGuestBufferBinding(
            src_addr,static_cast<u32>(src_size),false);
        auto dst_binding = buffer_cache.PrepareGuestBufferBinding(
            dst_addr,static_cast<u32>(dst_size),true);
        const auto needs_upload = [&](VAddr address,u64 size) {
            if (buffer_cache.HasGpuData(address,size)) return false;
            prepared.upload_ranges.emplace_back(address,static_cast<u32>(size));
            return true;
        };
        if (!src_binding && !needs_upload(src_addr,src_size)) return std::nullopt;
        if (!dst_binding && !needs_upload(dst_addr,dst_size)) return std::nullopt;
        if (!src_binding || !dst_binding) continue;
        bindings.guest.push_back(*src_binding);
        bindings.guest.push_back(*dst_binding);
        bindings.max_residency_tick = std::max({bindings.max_residency_tick,
                                                src_binding->residency_tick,
                                                dst_binding->residency_tick});
        bindings.max_graphics_bda_tick = std::max({bindings.max_graphics_bda_tick,
                                                  src_binding->graphics_bda_tick,
                                                  dst_binding->graphics_bda_tick});
        PreparedResourceBindings::CopyBatch ready{*src_binding,*dst_binding,{}};
        ready.copies.reserve(batch.count);
        for (std::size_t i=batch.first;i<batch.first+batch.count;++i) {
            const auto& op = copy_plan->copies[i];
            ready.copies.push_back(vk::BufferCopy{
                .srcOffset = src_binding->offset+op.src_offset-batch.src_begin,
                .dstOffset = dst_binding->offset+op.dst_offset-batch.dst_begin,
                .size = op.size});
        }
        bindings.hle_copies.push_back(std::move(ready));
    }
    if (!prepared.upload_ranges.empty()) return prepared;
    if (bindings.hle_copies.size() != copy_plan->batches.size() ||
        prepared.mapping_generation != buffer_cache.MappingGeneration() ||
        prepared.native_mapping_epoch != memory->MappingEpoch() ||
        !iss::g4::RevalidateSrtSnapshot(prepared.srt,provider)) return std::nullopt;
    prepared.bindings = std::move(bindings);
    return prepared;
}

bool Rasterizer::PrepareFallbackCode(PreparedComputeInputs& prepared,
                                     const ComputePipeline* pipeline) {
    if (!pipeline || prepared.mapping_generation != buffer_cache.MappingGeneration()) return false;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    const auto& cs = prepared.cs;
    // SearchBinaryInfo inspects up to 0x4000 dwords. This bounded window is copied
    // only for a fallback, never for the common C dispatch path.
    constexpr u32 ShaderCaptureBytes = 0x4000 * sizeof(u32);
    const VAddr shader_base = reinterpret_cast<VAddr>(cs.Address<u32*>());
    if (!shader_base || shader_base > std::numeric_limits<VAddr>::max() - ShaderCaptureBytes ||
        stage.pgm_base != shader_base) return false;
    const std::array extra_reads{iss::g4::SrtExtraRead{shader_base, ShaderCaptureBytes}};
    iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
    auto captured = iss::g4::CaptureSrtSnapshot(stage.srt_info, cs.user_data, provider,
                                                extra_reads);
    if (captured.proof.coverage != iss::g4::SrtCoverage::Complete ||
        captured.flattened != prepared.srt.flattened ||
        captured.reads.size() != prepared.srt.reads.size() + 1 ||
        !iss::g4::RevalidateSrtSnapshot(captured, provider)) return false;
    for (std::size_t i = 0; i < prepared.srt.reads.size(); ++i) {
        const auto& old = prepared.srt.reads[i];
        const auto& now = captured.reads[i];
        if (old.guest_address != now.guest_address || old.bytes != now.bytes ||
            old.memory_generation != now.memory_generation) return false;
    }
    const auto& code_read = captured.reads.back();
    if (code_read.guest_address != shader_base ||
        code_read.bytes.size() != ShaderCaptureBytes) return false;
    AmdGpu::BinaryInfo binary_info{};
    bool found{};
    for (std::size_t offset = 0;
         offset + sizeof(binary_info) <= code_read.bytes.size(); offset += sizeof(u32)) {
        std::memcpy(&binary_info, code_read.bytes.data() + offset, sizeof(binary_info));
        if (binary_info.Valid()) { found = true; break; }
    }
    if (!found || binary_info.shader_hash != stage.pgm_hash || binary_info.length == 0 ||
        binary_info.length % sizeof(u32) != 0 || binary_info.length > ShaderCaptureBytes)
        return false;
    if (!buffer_cache.PlanGuestAccess(shader_base, ShaderCaptureBytes, false,
                                      prepared.mapping_generation, prepared.accesses)) return false;
    prepared.shader_hash = binary_info.shader_hash;
    prepared.shader_guest_base = shader_base;
    prepared.shader_code.resize(binary_info.length / sizeof(u32));
    std::memcpy(prepared.shader_code.data(), code_read.bytes.data(), binary_info.length);
    prepared.srt = std::move(captured);
    return buffer_cache.MappingGeneration() == prepared.mapping_generation &&
           iss::g4::RevalidateSrtSnapshot(prepared.srt, provider);
}

std::optional<Rasterizer::PreparedResourceBindings> Rasterizer::PrepareResourceBindings(
    const ComputePipeline* pipeline, const PreparedComputeInputs& inputs) {
    if (!pipeline || !pipeline->IsReady() || !pipeline->UsesPushDescriptors() ||
        inputs.mapping_generation != buffer_cache.MappingGeneration()) return std::nullopt;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (stage.flattened_ud_buf != inputs.srt.flattened || !stage.images.empty() ||
        !stage.fmasks.empty() || !stage.samplers.empty() || stage.uses_dma)
        return std::nullopt;
    PreparedResourceBindings out{};
    out.push = inputs.push_data;
    out.buffers.reserve(stage.buffers.size());
    out.infos.reserve(stage.buffers.size());
    out.writes.resize(stage.buffers.size());
    const u64 alignment = instance.StorageMinAlignment();
    for (const auto& desc : stage.buffers) {
        PreparedResourceBindings::BufferItem item{};
        if (desc.buffer_type == Shader::BufferType::Guest) {
            const auto sharp = desc.GetSharp(stage);
            if (sharp.base_address && sharp.GetSize()) {
                const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
                if (!size || size > std::numeric_limits<u32>::max()) return std::nullopt;
                auto guest = buffer_cache.PrepareGuestBufferBinding(
                    sharp.base_address, static_cast<u32>(size), desc.is_written);
                if (!guest || !buffer_cache.RevalidateGuestBufferBinding(*guest))
                    return std::nullopt;
                if (desc.is_written) {
                    auto negative = texture_cache.ProbeNoRegisteredImage(sharp.base_address, size);
                    if (negative.result != iss::g4::NegativeImageResult::NoRegisteredPage)
                        return std::nullopt;
                    out.negative_images.push_back(negative);
                }
                item = {guest->buffer, guest->offset, size, sharp.base_address, desc.is_written};
                out.max_residency_tick = std::max(out.max_residency_tick,
                                                  guest->residency_tick);
                out.max_graphics_bda_tick = std::max(out.max_graphics_bda_tick,
                                                     guest->graphics_bda_tick);
                out.guest.push_back(*guest);
            }
        } else if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
            const auto* buffer = buffer_cache.GetGdsBuffer();
            item = {buffer, 0, buffer->SizeBytes(), 0, desc.is_written};
        } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
            const auto* buffer = buffer_cache.GetBdaPageTableBuffer();
            out.bda_graphics_tick_snapshot = buffer_cache.PublishedBdaGraphicsTick();
            out.max_graphics_bda_tick = std::max(out.max_graphics_bda_tick,
                                                  out.bda_graphics_tick_snapshot);
            out.uses_bda = true;
            item = {buffer, 0, buffer->SizeBytes(), 0, false};
        } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
            auto& stream = buffer_cache.GetStreamBuffer();
            const u64 bytes = inputs.srt.flattened.size() * sizeof(u32);
            if (!bytes) return std::nullopt;
            const u64 offset = stream.Copy(inputs.srt.flattened.data(), bytes, alignment);
            item = {&stream, offset, bytes, 0, false};
        } else {
            return std::nullopt;
        }
        const u32 descriptor_index = static_cast<u32>(out.buffers.size());
        out.buffers.push_back(item);
        if (!item.buffer) {
            out.infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
        } else {
            const u64 aligned = Common::AlignDown(item.offset, alignment);
            const u64 adjust = item.offset - aligned;
            if (adjust % sizeof(u32)) return std::nullopt;
            if (desc.buffer_type == Shader::BufferType::Guest)
                out.push.AddOffset(descriptor_index, adjust);
            out.infos.emplace_back(item.buffer->Handle(), aligned, item.size + adjust);
        }
    }
    if (inputs.indirect_args) {
        const auto& ref = *inputs.indirect_args;
        if (ref.backing_generation != inputs.mapping_generation || ref.size < 12)
            return std::nullopt;
        out.indirect = buffer_cache.PrepareGuestBufferBinding(ref.gpu_address, ref.size, false);
        if (!out.indirect || !buffer_cache.RevalidateGuestBufferBinding(*out.indirect))
            return std::nullopt;
        out.max_residency_tick = std::max(out.max_residency_tick,
                                          out.indirect->residency_tick);
        out.max_graphics_bda_tick = std::max(out.max_graphics_bda_tick,
                                             out.indirect->graphics_bda_tick);
    }
    for (u32 i = 0; i < out.infos.size(); ++i)
        WriteBufferDescriptor(out.writes[i], i, &out.infos[i]);
    if ((out.uses_bda && buffer_cache.PublishedBdaGraphicsTick() !=
                             out.bda_graphics_tick_snapshot) ||
        inputs.mapping_generation != buffer_cache.MappingGeneration() ||
        out.max_residency_tick > buffer_cache.PublishedResidencyTick()) return std::nullopt;
    return out;
}

bool Rasterizer::RecordPreparedCompute(const ComputePipeline* pipeline,
                                       const PreparedComputeInputs& inputs,
                                       PreparedResourceBindings& bindings) {
    const bool hle_copy = !bindings.hle_copies.empty();
    if (!pipeline || !pipeline->IsReady() ||
        (!hle_copy && !pipeline->UsesPushDescriptors()) ||
        bindings.max_residency_tick > buffer_cache.PublishedResidencyTick() ||
        bindings.buffers.size() != bindings.infos.size() ||
        bindings.writes.size() != bindings.infos.size()) return false;
    const bool asc = ActiveCpOwner() == iss::g4::CpOwner::Asc1;
    const ComputeQueueScope queue_scope{*this, asc};
    if (hle_copy) {
        for (const auto& batch : bindings.hle_copies)
            Rt().CopyBuffer(batch.source.buffer,batch.destination.buffer,batch.copies);
        return true;
    }
    bool needs_barrier{};
    for (const auto& item : bindings.buffers) {
        if (!item.buffer || !item.size || item.buffer == &buffer_cache.GetStreamBuffer())
            continue;
        needs_barrier |= Rt().IsBufferAccessed(item.buffer, item.offset, item.size, item.write);
    }
    if (bindings.indirect) {
        const auto& args = *bindings.indirect;
        needs_barrier |= Rt().IsBufferAccessed(args.buffer, args.offset, args.size, false);
    }
    if (needs_barrier) Rt().FlushBarriers();
    if (!asc) scheduler.EndRendering();
    pipeline->BindResources(Sched(), bindings.writes, bindings.push);
    const auto cmdbuf = Sched().Recorder();
    BindPipelineLazy(Sched(), vk::PipelineBindPoint::eCompute, pipeline);
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    Sched().ProfileDispatchBegin(stage.pgm_hash);
    if (bindings.indirect) {
        cmdbuf.dispatchIndirect(bindings.indirect->buffer->Handle(), bindings.indirect->offset);
    } else {
        cmdbuf.dispatch(inputs.cs.dim_x, inputs.cs.dim_y, inputs.cs.dim_z);
    }
    Sched().ProfileDispatchEnd();
    DebugState.IncDispatch();
    for (const auto& item : bindings.buffers) {
        if (!item.buffer || !item.size || item.buffer == &buffer_cache.GetStreamBuffer())
            continue;
        const auto access = vk::AccessFlagBits2::eShaderRead |
                            (item.write ? vk::AccessFlagBits2::eShaderWrite
                                        : vk::AccessFlagBits2::eNone);
        Rt().AccessBuffer(item.buffer, item.offset, item.size,
                          vk::PipelineStageFlagBits2::eComputeShader, access);
    }
    if (bindings.indirect) {
        const auto& args = *bindings.indirect;
        Rt().AccessBuffer(args.buffer, args.offset, args.size,
                          vk::PipelineStageFlagBits2::eDrawIndirect,
                          vk::AccessFlagBits2::eIndirectCommandRead);
    }
    return true;
}

Rasterizer::AscDispatchPlan Rasterizer::PrepareAscDispatch(const AscDispatchInput& input) {
    AscDispatchPlan out{};
    out.guest = input.guest;
    if (!cp_split_enabled || input.guest.qid == 0 || input.guest.qid >= 57 ||
        input.guest.generation == 0 || input.guest.packet_sequence == 0 ||
        input.guest.qid != static_cast<u32>(DrawQueue()) ||
        input.guest.generation != CurrentSubmissionGeneration() ||
        input.cs.user_data != DrawCsRegs().user_data ||
        input.cs.Address<u32*>() != DrawCsRegs().Address<u32*>()) return out;
    std::optional<iss::g4::IndirectArgsRef> indirect;
    if (input.indirect) {
        const auto [address, size] = *input.indirect;
        if (!address || size < 12 || address > std::numeric_limits<VAddr>::max() - size)
            return out;
        indirect = iss::g4::IndirectArgsRef{address, buffer_cache.MappingGeneration(), size};
    }
    const auto* pipeline = pipeline_cache.GetComputePipeline(iss::g4::CpOwner::Asc1);
    if (!pipeline) return out;
    out.pipeline = pipeline;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (!stage.images.empty()) {
        // The image registry, refresh/upload path, and layouts belong to G. C only
        // captures immutable frontend inputs; Liverpool must pause C before G runs
        // this packet under an unpublished global lease.
        const bool supported = !indirect && pipeline->IsReady() &&
            stage.pgm_hash != 0xfefebf9f && !stage.uses_dma &&
            stage.fmasks.empty() &&
            // Meta-clear HLE can otherwise consume a buffer write before the
            // image dispatch; this route records the shader itself.
            (stage.has_bitwise_xor ||
             std::ranges::none_of(stage.buffers, [](const auto& desc) {
                 return desc.buffer_type == Shader::BufferType::Guest && desc.is_written;
             })) &&
            std::ranges::none_of(stage.buffers, [](const auto& desc) {
                return desc.buffer_type != Shader::BufferType::Guest &&
                       desc.buffer_type != Shader::BufferType::Flatbuf;
            });
        if (!supported) {
            static std::atomic<u64> rejected{0};
            const u64 count = rejected.fetch_add(1, std::memory_order_relaxed) + 1;
            if (count == 1 || (count & (count - 1)) == 0)
                LOG_WARNING(Render_Vulkan,
                            "G4 serial image fallback rejected count={} indirect={} hle={} "
                            "dma={} fmask={} sampler={} ready={}",
                            count, bool(indirect), stage.pgm_hash == 0xfefebf9f,
                            stage.uses_dma, stage.fmasks.size(), stage.samplers.size(),
                            pipeline->IsReady());
            return out;
        }
        if (!std::ranges::equal(stage.user_data, input.cs.user_data)) return out;
        iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
        PreparedComputeInputs prepared{};
        prepared.cs = input.cs;
        prepared.push_data = {};
        prepared.mapping_generation = buffer_cache.MappingGeneration();
        prepared.native_mapping_epoch = memory->MappingEpoch();
        prepared.srt = iss::g4::CaptureSrtSnapshot(stage.srt_info, input.cs.user_data,
                                                   provider);
        if (prepared.srt.proof.coverage != iss::g4::SrtCoverage::Complete ||
            prepared.srt.flattened != stage.flattened_ud_buf ||
            !iss::g4::RevalidateSrtSnapshot(prepared.srt, provider) ||
            !PrepareFallbackCode(prepared, pipeline)) return out;
        auto capture = std::make_shared<iss::g4::SrtCaptureResult>(std::move(prepared.srt));
        out.fallback = iss::g4::CaptureTypedFallbackPacket(
            input.guest, input.cs, input.cs.user_data, capture->flattened,
            capture->reads, capture->proof, {}, false, capture->watch_lease,
            iss::g4::ShaderCodeSnapshot{prepared.shader_guest_base,
                                        prepared.shader_hash,
                                        std::move(prepared.shader_code)},
            // Sampler border colors are resolved on G from G's current register
            // state inside the exclusive serial action. C must not read it.
            capture, prepared.push_data, 0, true);
        if (out.fallback->Gate() != iss::g4::FallbackGate::Ready) {
            out.fallback.reset();
            return out;
        }
        out.route = AscDispatchRoute::FallbackGraphics;
        return out;
    }
    if (stage.pgm_hash == 0xfefebf9f) {
        // The copy HLE derives its work count from the direct dispatch X dimension.
        // GPU-written indirect dimensions cannot be inspected during C prepare.
        if (indirect) return out;
        if (input.cs.dim_x == 0) {
            out.route = AscDispatchRoute::NoOp;
            return out;
        }
        if (!UseComputeQueue(pipeline)) return out;
        const ComputeQueueScope queue_scope{*this,true};
        out.prepared = PrepareCopyHleInputs(pipeline);
        if (!out.prepared) return out;
        if (!out.prepared->upload_ranges.empty()) {
            out.upload_ranges = out.prepared->upload_ranges;
            out.route = AscDispatchRoute::RetryNeedUpload;
            return out;
        }
        if (!out.prepared->bindings) return out;
        out.route = AscDispatchRoute::Compute;
        return out;
    }
    out.prepared = PrepareComputeInputs(pipeline, indirect);
    if (!out.prepared || !UseComputeQueue(pipeline)) return out;
    {
        const ComputeQueueScope queue_scope{*this, true};
        const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
        for (const auto& desc : stage.buffers) {
            if (desc.buffer_type != Shader::BufferType::Guest) continue;
            const auto sharp = desc.GetSharp(stage);
            if (!sharp.base_address || !sharp.GetSize()) continue;
            const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
            if (!size || size > std::numeric_limits<u32>::max()) return out;
            if (buffer_cache.PrepareGuestBufferBinding(
                    sharp.base_address, static_cast<u32>(size), desc.is_written)) continue;
            // A G-only upload is valid only for CPU-owned bytes with no registered image.
            // GPU-owned or image-backed ranges require a separate slow path.
            if (buffer_cache.HasGpuData(sharp.base_address,size) ||
                texture_cache.ProbeNoRegisteredImage(sharp.base_address,size).result !=
                    iss::g4::NegativeImageResult::NoRegisteredPage) return out;
            out.upload_ranges.emplace_back(sharp.base_address,static_cast<u32>(size));
        }
        if (!out.upload_ranges.empty()) {
            out.route = AscDispatchRoute::RetryNeedUpload;
            return out;
        }
        out.prepared->bindings = PrepareResourceBindings(pipeline, *out.prepared);
    }
    if (!out.prepared->bindings) return out;
    out.route = AscDispatchRoute::Compute;
    return out;
}

Rasterizer::AscCommitResult Rasterizer::CommitAscDispatch(AscDispatchPlan& plan) {
    if (!cp_split_enabled || plan.route != AscDispatchRoute::Compute || !plan.pipeline ||
        !plan.prepared || !plan.prepared->bindings ||
        plan.guest.qid == 0 || plan.guest.qid >= 57 ||
        plan.guest.generation != CurrentSubmissionGeneration() ||
        plan.guest.qid != static_cast<u32>(DrawQueue())) return AscCommitResult::Failed;
    auto& inputs = *plan.prepared;
    auto& bindings = *inputs.bindings;
    const ComputeQueueScope queue_scope{*this, true};
    const u64 tick = Sched().CurrentTick();
    if (tick == 0 || tick != liverpool->CpOpenTick(iss::g4::CpLane::Asc))
        return AscCommitResult::Retry;
    const iss::g4::PacketLeaseKey key{iss::g4::CpLane::Asc, plan.guest.qid,
                                      plan.guest.generation, plan.guest.packet_sequence};
    const auto reservation = buffer_cache.TryReservePacket(key, inputs.accesses,
                                                           inputs.mapping_generation);
    if (reservation.status == iss::g4::PacketReserveStatus::Busy ||
        reservation.status == iss::g4::PacketReserveStatus::Stale)
        return AscCommitResult::Retry;
    if (reservation.status != iss::g4::PacketReserveStatus::Reserved)
        return AscCommitResult::Failed;
    const iss::g4::CpTick consumer{iss::g4::CpLane::Asc, tick};
    const auto cancel = [&] {
        [[maybe_unused]] const bool cancelled = buffer_cache.CancelPacket(reservation.lease_id);
        liverpool->CancelCpConsumer(consumer);
    };
    for (const auto& producer : buffer_cache.PacketDependencies(reservation.lease_id)) {
        auto seal = liverpool->SealDependency(consumer, producer);
        if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Asc) !=
                         iss::g4::SealState::Done ||
            (producer.lane == iss::g4::CpLane::Graphics &&
             !WaitAscOnGraphicsToken(producer))) {
            cancel();
            return AscCommitResult::Retry;
        }
    }
    u64 gpu_owned_g_writer_tick{};
    for (const auto& guest : bindings.guest) {
        if (guest.gpu_owned)
            gpu_owned_g_writer_tick = std::max(gpu_owned_g_writer_tick,
                                               guest.writer_ticks[0]);
    }
    if (bindings.indirect && bindings.indirect->gpu_owned)
        gpu_owned_g_writer_tick = std::max(gpu_owned_g_writer_tick,
                                           bindings.indirect->writer_ticks[0]);
    if (gpu_owned_g_writer_tick) {
        const iss::g4::QidTick gpu_producer{iss::g4::CpLane::Graphics, 0,
                                             liverpool->CpGraphicsGeneration(),
                                             gpu_owned_g_writer_tick};
        auto seal = liverpool->SealDependency(consumer, gpu_producer);
        if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Asc) !=
                         iss::g4::SealState::Done ||
            !WaitAscOnGraphicsToken(gpu_producer)) {
            cancel();
            return AscCommitResult::Retry;
        }
    }
    if (bindings.max_graphics_bda_tick) {
        const u64 graphics_generation = liverpool->CpGraphicsGeneration();
        if (!graphics_generation) { cancel(); return AscCommitResult::Retry; }
        const iss::g4::QidTick bda_producer{iss::g4::CpLane::Graphics, 0,
                                             graphics_generation,
                                             bindings.max_graphics_bda_tick};
        auto seal = liverpool->SealDependency(consumer, bda_producer);
        if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Asc) !=
                         iss::g4::SealState::Done ||
            !WaitAscOnGraphicsToken(bda_producer)) {
            cancel();
            return AscCommitResult::Retry;
        }
    }
    iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
    if ((bindings.uses_bda && buffer_cache.PublishedBdaGraphicsTick() !=
                                  bindings.bda_graphics_tick_snapshot) ||
        inputs.mapping_generation != buffer_cache.MappingGeneration() ||
        inputs.native_mapping_epoch != memory->MappingEpoch() ||
        !iss::g4::RevalidateSrtSnapshot(inputs.srt, provider) ||
        std::ranges::any_of(bindings.guest, [&](const auto& guest) {
            return !buffer_cache.RevalidateGuestBufferBinding(guest);
        }) ||
        (bindings.indirect &&
         !buffer_cache.RevalidateGuestBufferBinding(*bindings.indirect))) {
        cancel();
        return AscCommitResult::Retry;
    }
    if (!liverpool->BeginCpRecord(iss::g4::CpLane::Asc, tick)) {
        cancel();
        return AscCommitResult::Retry;
    }
    const iss::g4::QidTick token{iss::g4::CpLane::Asc, plan.guest.qid,
                                  plan.guest.generation, tick};
    const auto publish = [&] {
        return buffer_cache.PublishPacketWithBufferWrites(
            reservation.lease_id, token, true, bindings.guest, bindings.negative_images);
    };
    bool published{};
    // Current lock graph has no coherence->PageManager page-lock edge. This short
    // publication takes native mapping shared -> SRT page -> coherence -> texture.
    if (inputs.srt.watch_lease) {
        published = inputs.srt.watch_lease->PublishIfValid([&] {
            return memory->MappingEpoch() == inputs.native_mapping_epoch && publish();
        });
    } else if (inputs.srt.proof.no_guest_reads && inputs.srt.reads.empty()) {
        published = memory->WithMappingReadLock([&](u64 epoch) {
            return epoch == inputs.native_mapping_epoch && publish();
        });
    }
    if (!published) {
        [[maybe_unused]] const bool ended = liverpool->EndCpRecord(iss::g4::CpLane::Asc, tick);
        cancel();
        return AscCommitResult::Retry;
    }
    // Publication is irreversible: a late failure is a contract violation, never a retry.
    ASSERT_MSG(RecordPreparedCompute(plan.pipeline, inputs, bindings),
               "G4 packet failed after coherence publication");
    ASSERT_MSG(buffer_cache.FinishPacketRecord(reservation.lease_id),
               "G4 packet lease did not finish after recording");
    ASSERT_MSG(liverpool->EndCpRecord(iss::g4::CpLane::Asc, tick),
               "G4 packet record token did not close");
    return AscCommitResult::Recorded;
}

std::optional<iss::g4::GraphicsRecordingProof> Rasterizer::RecordFallbackSnapshot(
    const iss::g4::FallbackPacket& packet,
    const iss::g4::QidTick& sealed_asc_prefix) {
    if (!cp_split_enabled || packet.Gate() != iss::g4::FallbackGate::Ready ||
        !packet.srt_capture || !packet.srt_lease || !compute_scheduler ||
        packet.srt_lease != packet.srt_capture->watch_lease ||
        !packet.srt_lease->Covers(packet.srt_reads) ||
        sealed_asc_prefix.lane != iss::g4::CpLane::Asc ||
        sealed_asc_prefix.qid != packet.guest.qid ||
        sealed_asc_prefix.generation != packet.guest.generation ||
        sealed_asc_prefix.tick == 0 ||
        packet.srt_capture->flattened != packet.flattened_srt ||
        packet.srt_capture->reads.size() != packet.srt_reads.size()) return std::nullopt;
    for (std::size_t i = 0; i < packet.srt_reads.size(); ++i) {
        const auto& a = packet.srt_capture->reads[i];
        const auto& b = packet.srt_reads[i];
        if (a.guest_address != b.guest_address ||
            a.memory_generation != b.memory_generation || a.bytes != b.bytes)
            return std::nullopt;
    }
    if (packet.compute_state.size() != sizeof(AmdGpu::ComputeProgram) ||
        packet.srt_capture->read_versions.empty()) return std::nullopt;
    PreparedComputeInputs inputs{};
    std::memcpy(&inputs.cs, packet.compute_state.data(), sizeof(inputs.cs));
    inputs.srt = *packet.srt_capture;
    inputs.push_data = packet.push_data;
    inputs.indirect_args = packet.indirect_args;
    inputs.mapping_generation = inputs.srt.read_versions.front().mapping_generation;
    inputs.native_mapping_epoch = inputs.srt.read_versions.front().native_mapping_epoch;
    inputs.shader_code = packet.shader.code;
    inputs.shader_hash = packet.shader.hash;
    inputs.shader_guest_base = packet.shader.guest_base;
    if (!inputs.mapping_generation || !inputs.native_mapping_epoch ||
        inputs.mapping_generation != buffer_cache.MappingGeneration() ||
        inputs.native_mapping_epoch != memory->MappingEpoch()) return std::nullopt;
    iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
    if (!iss::g4::RevalidateSrtSnapshot(inputs.srt, provider)) return std::nullopt;
    const auto* pipeline = pipeline_cache.GetComputePipelineFromSnapshot(packet);
    if (!pipeline || !pipeline->IsReady()) return std::nullopt;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (stage.pgm_hash != packet.shader.hash || stage.flattened_ud_buf != inputs.srt.flattened ||
        stage.pgm_hash == 0xfefebf9f || stage.uses_dma || !stage.images.empty() ||
        !stage.fmasks.empty() || !stage.samplers.empty()) return std::nullopt;
    for (const auto& read : inputs.srt.reads) {
        if (!buffer_cache.PlanGuestAccess(read.guest_address, read.bytes.size(), false,
                                          inputs.mapping_generation, inputs.accesses))
            return std::nullopt;
    }
    for (const auto& desc : stage.buffers) {
        if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
            VideoCore::BufferCache::PlanGdsAccess(desc.is_written, inputs.accesses);
        } else if (desc.buffer_type == Shader::BufferType::Guest) {
            const auto sharp = desc.GetSharp(stage);
            if (sharp.base_address && sharp.GetSize()) {
                const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
                if (!size || !buffer_cache.PlanGuestAccess(sharp.base_address, size,
                                                            desc.is_written,
                                                            inputs.mapping_generation,
                                                            inputs.accesses)) return std::nullopt;
            }
        } else if (desc.buffer_type != Shader::BufferType::Flatbuf) {
            return std::nullopt;
        }
    }
    if (inputs.indirect_args &&
        !buffer_cache.PlanGuestAccess(inputs.indirect_args->gpu_address,
                                      inputs.indirect_args->size, false,
                                      inputs.mapping_generation, inputs.accesses))
        return std::nullopt;
    inputs.bindings = PrepareResourceBindings(pipeline, inputs);
    if (!inputs.bindings) return std::nullopt;
    // A GPU-owned byte written by C needs its exact qid/generation, which the
    // coarse per-lane MemoryTracker tick does not carry. Do not guess it here.
    if (std::ranges::any_of(inputs.bindings->guest, [](const auto& guest) {
            return guest.gpu_owned && guest.writer_ticks[1] != 0;
        }) ||
        (inputs.bindings->indirect && inputs.bindings->indirect->gpu_owned &&
         inputs.bindings->indirect->writer_ticks[1] != 0)) return std::nullopt;
    const u64 tick = scheduler.CurrentTick();
    const u64 generation = liverpool->CpGraphicsGeneration();
    if (!tick || !generation || tick != liverpool->CpOpenTick(iss::g4::CpLane::Graphics))
        return std::nullopt;
    const u64 sequence = graphics_fallback_sequence.fetch_add(1, std::memory_order_relaxed);
    const iss::g4::PacketLeaseKey key{iss::g4::CpLane::Graphics, 0, generation, sequence};
    const auto reserve = buffer_cache.TryReservePacket(key, inputs.accesses,
                                                       inputs.mapping_generation);
    if (reserve.status != iss::g4::PacketReserveStatus::Reserved) return std::nullopt;
    const iss::g4::CpTick consumer{iss::g4::CpLane::Graphics, tick};
    const auto cancel = [&] {
        [[maybe_unused]] const bool cancelled = buffer_cache.CancelPacket(reserve.lease_id);
        liverpool->CancelCpConsumer(consumer);
    };
    const auto wait_asc = [&](const iss::g4::QidTick& producer) {
        if (compute_scheduler->SubmittedTick() < producer.tick) {
            auto seal = liverpool->SealDependency(consumer, producer);
            if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Graphics) !=
                             iss::g4::SealState::Done ||
                compute_scheduler->SubmittedTick() < producer.tick) return false;
        }
        scheduler.AddCrossWait(compute_scheduler->GetWorkSemaphore()->Handle(), producer.tick);
        return true;
    };
    if (!wait_asc(sealed_asc_prefix)) { cancel(); return std::nullopt; }
    for (const auto& producer : buffer_cache.PacketDependencies(reserve.lease_id)) {
        if (producer.lane == iss::g4::CpLane::Asc && !wait_asc(producer)) {
            cancel();
            return std::nullopt;
        }
    }
    if (!iss::g4::RevalidateSrtSnapshot(inputs.srt, provider) ||
        std::ranges::any_of(inputs.bindings->guest, [&](const auto& guest) {
            return !buffer_cache.RevalidateGuestBufferBinding(guest);
        }) ||
        (inputs.bindings->indirect &&
         !buffer_cache.RevalidateGuestBufferBinding(*inputs.bindings->indirect)) ||
        !liverpool->BeginCpRecord(iss::g4::CpLane::Graphics, tick)) {
        cancel();
        return std::nullopt;
    }
    const iss::g4::QidTick token{iss::g4::CpLane::Graphics, 0, generation, tick};
    const bool published = packet.PublishSrtIfValid([&] {
        return buffer_cache.PublishPacketWithBufferWrites(
            reserve.lease_id, token, true, inputs.bindings->guest,
            inputs.bindings->negative_images);
    });
    if (!published) {
        [[maybe_unused]] const bool ended = liverpool->EndCpRecord(iss::g4::CpLane::Graphics,
                                                                     tick);
        cancel();
        return std::nullopt;
    }
    ASSERT_MSG(RecordPreparedCompute(pipeline, inputs, *inputs.bindings),
               "G4 fallback failed after graphics metadata publication");
    ASSERT_MSG(buffer_cache.FinishPacketRecord(reserve.lease_id),
               "G4 fallback lease did not finish after recording");
    ASSERT_MSG(liverpool->EndCpRecord(iss::g4::CpLane::Graphics, tick),
               "G4 fallback graphics record token did not close");
    return iss::g4::GraphicsRecordingProof{tick, stage.srt_info.flattened_bufsize_dw,
                                            true, true, true};
}

std::optional<iss::g4::GraphicsRecordingProof> Rasterizer::RecordSerialImageFallback(
    const iss::g4::FallbackPacket& packet,
    const iss::g4::QidTick& sealed_asc_prefix) {
    // This is an exclusive serial action, not the prepared buffer-only packet path.
    // Liverpool must pause all ASC translation and wait for the preceding C work to
    // finish on the GPU before invoking us at a G safe point.
    if (!cp_split_enabled || !liverpool->CpSplitActive() ||
        !liverpool->AscOwnerPausedForGraphics() || DrawQueue() != 0 ||
        !compute_scheduler || !packet.serial_image ||
        packet.Gate() != iss::g4::FallbackGate::Ready ||
        !packet.srt_capture || !packet.srt_lease ||
        packet.srt_lease != packet.srt_capture->watch_lease ||
        sealed_asc_prefix.lane != iss::g4::CpLane::Asc ||
        sealed_asc_prefix.qid != packet.guest.qid ||
        sealed_asc_prefix.generation != packet.guest.generation ||
        sealed_asc_prefix.tick == 0 ||
        !compute_scheduler->IsFree(sealed_asc_prefix.tick) ||
        packet.compute_state.size() != sizeof(AmdGpu::ComputeProgram)) return std::nullopt;
    AmdGpu::ComputeProgram cs{};
    std::memcpy(&cs, packet.compute_state.data(), sizeof(cs));
    if (cs.user_data != packet.user_data ||
        reinterpret_cast<VAddr>(cs.Address<u32*>()) != packet.shader.guest_base ||
        packet.srt_capture->flattened != packet.flattened_srt ||
        !packet.srt_lease->Covers(packet.srt_reads)) return std::nullopt;
    iss::g4::TrackedSrtMemoryProvider provider{*memory, page_manager, buffer_cache};
    if (!iss::g4::RevalidateSrtSnapshot(*packet.srt_capture, provider)) return std::nullopt;
    const auto* pipeline = pipeline_cache.GetComputePipelineFromSnapshot(packet);
    if (!pipeline || !pipeline->IsReady()) return std::nullopt;
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (stage.pgm_hash != packet.shader.hash ||
        stage.flattened_ud_buf != packet.flattened_srt ||
        stage.images.empty() || !stage.fmasks.empty() ||
        stage.uses_dma || stage.pgm_hash == 0xfefebf9f ||
        (!stage.has_bitwise_xor &&
         std::ranges::any_of(stage.buffers, [](const auto& desc) {
             return desc.buffer_type == Shader::BufferType::Guest && desc.is_written;
         })) ||
        std::ranges::any_of(stage.buffers, [](const auto& desc) {
            return desc.buffer_type != Shader::BufferType::Guest &&
                   desc.buffer_type != Shader::BufferType::Flatbuf;
        })) return std::nullopt;

    auto admission = TryPrepareGraphicsGpuPacket();
    if (admission.status != GraphicsHostWriteStatus::Ready ||
        !ResolveGraphicsGpuPacket(admission.guard) ||
        !iss::g4::RevalidateSrtSnapshot(*packet.srt_capture, provider) ||
        !PublishGraphicsGpuPacket(admission.guard)) return std::nullopt;

    // The global lease remains unpublished through every lazy image lookup,
    // refresh/upload, descriptor allocation, transition, and Vulkan emission.
    // A failure from here cannot be retried without duplicating guest work.
    const ComputeQueueScope queue_scope{*this, false};
    ASSERT_MSG(BindResources(pipeline, &packet.push_data, true),
               "G4 serial image fallback failed after graphics emission began");
    if (needs_barrier) Rt().FlushBarriers();
    scheduler.EndRendering();
    pipeline->BindResources(Sched(), set_writes, push_data);
    const auto cmdbuf = Sched().Recorder();
    BindPipelineLazy(Sched(), vk::PipelineBindPoint::eCompute, pipeline);
    Sched().ProfileDispatchBegin(stage.pgm_hash);
    cmdbuf.dispatch(cs.dim_x, cs.dim_y, cs.dim_z);
    Sched().ProfileDispatchEnd();
    DebugState.IncDispatch();
    ResetBindings(true);
    FinishGraphicsGpuPacket(admission.guard);
    ASSERT_MSG(admission.guard.final_tick != 0,
               "G4 serial image fallback recorded no graphics tick");
    return iss::g4::GraphicsRecordingProof{admission.guard.final_tick,
                                            stage.srt_info.flattened_bufsize_dw,
                                            true, false, false};
}

Rasterizer::GraphicsHostWriteGuard& Rasterizer::GraphicsHostWriteGuard::operator=(
    GraphicsHostWriteGuard&& other) noexcept {
    if (this != &other) {
        if (owner) owner->FinishGraphicsHostWrite(*this);
        owner = std::exchange(other.owner, nullptr);
        lease_id = other.lease_id;
        tick = other.tick;
        generation = other.generation;
        final_tick = other.final_tick;
        address = other.address;
        size = other.size;
        gpu_packet = other.gpu_packet;
        record_started = other.record_started;
        published = other.published;
    }
    return *this;
}

Rasterizer::GraphicsHostWriteGuard::~GraphicsHostWriteGuard() {
    if (owner) owner->FinishGraphicsHostWrite(*this);
}

Rasterizer::GraphicsHostWriteAdmission Rasterizer::TryPrepareGraphicsHostWrite(
    VAddr address, u64 size) {
    if (!cp_split_enabled || !liverpool->CpSplitActive() || DrawQueue() != 0 ||
        !size || address > std::numeric_limits<VAddr>::max() - size)
        return {GraphicsHostWriteStatus::Failed, {}};
    const u64 tick = scheduler.CurrentTick();
    if (tick == 0 || tick != liverpool->CpOpenTick(iss::g4::CpLane::Graphics))
        return {GraphicsHostWriteStatus::Busy, {}};
    const u64 sequence = graphics_host_write_sequence.fetch_add(1, std::memory_order_relaxed);
    if (sequence == 0) return {GraphicsHostWriteStatus::Failed, {}};
    const u64 generation = liverpool->CpGraphicsGeneration();
    if (!generation) return {GraphicsHostWriteStatus::Failed, {}};
    const iss::g4::PacketLeaseKey key{iss::g4::CpLane::Graphics, 0, generation, sequence};
    const iss::g4::PacketResourceAccess global{iss::g4::GlobalResourceKey, 0, 1, true};
    const auto reserve = buffer_cache.TryReservePacket(key, std::span{&global, 1},
                                                       buffer_cache.MappingGeneration());
    if (reserve.status == iss::g4::PacketReserveStatus::Busy ||
        reserve.status == iss::g4::PacketReserveStatus::Stale)
        return {GraphicsHostWriteStatus::Busy, {}};
    if (reserve.status != iss::g4::PacketReserveStatus::Reserved)
        return {GraphicsHostWriteStatus::Failed, {}};
    return {GraphicsHostWriteStatus::Ready,
            GraphicsHostWriteGuard{this, reserve.lease_id, tick, generation, address, size}};
}

bool Rasterizer::ResolveGraphicsHostWrite(GraphicsHostWriteGuard& guard) {
    if (guard.owner != this || !guard.lease_id || !compute_scheduler ||
        scheduler.CurrentTick() != guard.tick) return false;
    const iss::g4::CpTick consumer{iss::g4::CpLane::Graphics, guard.tick};
    for (const auto& producer : buffer_cache.PacketDependencies(guard.lease_id)) {
        if (producer.lane != iss::g4::CpLane::Asc) continue;
        if (compute_scheduler->SubmittedTick() < producer.tick) {
            auto seal = liverpool->SealDependency(consumer, producer);
            if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Graphics) !=
                             iss::g4::SealState::Done ||
                compute_scheduler->SubmittedTick() < producer.tick) return false;
        }
        compute_scheduler->Wait(producer.tick); // host write needs GPU completion
    }
    // A direct CPU store also follows G's own previously recorded GPU writes.
    // Flush the exact open prefix while the global logical lease blocks new C work,
    // then wait for GPU completion before the guest VA can be modified.
    scheduler.Flush();
    scheduler.WaitSubmitted(guard.tick);
    scheduler.Wait(guard.tick);
    return buffer_cache.RevalidatePacket(guard.lease_id, true);
}

void Rasterizer::FinishGraphicsHostWrite(GraphicsHostWriteGuard& guard) noexcept {
    if (guard.owner != this) return;
    if (guard.gpu_packet && guard.record_started) {
        FinishGraphicsGpuPacket(guard);
        return;
    }
    if (guard.lease_id) {
        if (guard.published) {
            [[maybe_unused]] const bool finished =
                buffer_cache.FinishPacketRecord(guard.lease_id);
            [[maybe_unused]] const bool ended =
                liverpool->EndCpRecord(iss::g4::CpLane::Graphics, guard.tick);
        } else {
            [[maybe_unused]] const bool cancelled = buffer_cache.CancelPacket(guard.lease_id);
        }
        liverpool->CancelCpConsumer({iss::g4::CpLane::Graphics, guard.tick});
    }
    guard.owner = nullptr;
    guard.lease_id = 0;
    guard.record_started = false;
    guard.published = false;
}

Rasterizer::GraphicsHostWriteAdmission Rasterizer::TryPrepareGraphicsGpuPacket() {
    auto admission = TryPrepareGraphicsHostWrite(0, 1);
    if (admission.status == GraphicsHostWriteStatus::Ready)
        admission.guard.gpu_packet = true;
    return admission;
}

bool Rasterizer::ResolveGraphicsGpuPacket(GraphicsHostWriteGuard& guard) {
    if (guard.owner != this || !guard.gpu_packet || !guard.lease_id ||
        scheduler.CurrentTick() != guard.tick || !compute_scheduler) return false;
    const iss::g4::CpTick consumer{iss::g4::CpLane::Graphics, guard.tick};
    for (const auto& producer : buffer_cache.PacketDependencies(guard.lease_id)) {
        if (producer.lane != iss::g4::CpLane::Asc) continue;
        if (compute_scheduler->SubmittedTick() < producer.tick) {
            auto seal = liverpool->SealDependency(consumer, producer);
            if (!seal || liverpool->WaitCpSeal(seal, iss::g4::CpLane::Graphics) !=
                             iss::g4::SealState::Done ||
                compute_scheduler->SubmittedTick() < producer.tick) return false;
        }
        scheduler.AddCrossWait(compute_scheduler->GetWorkSemaphore()->Handle(), producer.tick);
    }
    return buffer_cache.RevalidatePacket(guard.lease_id, true);
}

bool Rasterizer::PublishGraphicsGpuPacket(GraphicsHostWriteGuard& guard) {
    if (guard.owner != this || !guard.gpu_packet || guard.record_started || !guard.lease_id ||
        scheduler.CurrentTick() != guard.tick ||
        liverpool->CpOpenTick(iss::g4::CpLane::Graphics) != guard.tick ||
        !liverpool->BeginCpRecord(iss::g4::CpLane::Graphics, guard.tick)) return false;
    // GlobalResourceKey is admission metadata, not a guest byte snapshot. Keep the
    // lease unpublished (and conflicting with every C reservation) across the action.
    // Its writer token is published only after the action's final scheduler tick is known.
    guard.record_started = true;
    return true;
}

void Rasterizer::FinishGraphicsGpuPacket(GraphicsHostWriteGuard& guard) noexcept {
    if (guard.owner != this) return;
    ASSERT_MSG(guard.gpu_packet && guard.record_started && guard.lease_id,
               "G4 global graphics lease was not recording");
    const u64 current_tick = scheduler.CurrentTick();
    ASSERT_MSG(current_tick >= guard.tick, "G4 graphics scheduler tick regressed");
    // A Flush inside Draw/Fill/Copy may have submitted one or more ticks. Wait for
    // their actual host submissions before the timeline bridge reconciles them.
    if (current_tick > guard.tick) scheduler.WaitSubmitted(current_tick - 1);
    const bool has_current_work = scheduler.HasWork();
    const u64 final_tick = has_current_work ? current_tick
                                            : (current_tick > guard.tick ? current_tick - 1 : 0);
    guard.final_tick = final_tick;
    ASSERT_MSG(liverpool->EndCpRecord(iss::g4::CpLane::Graphics, guard.tick, true),
               "G4 multi-tick graphics record did not close");
    guard.record_started = false;
    if (final_tick != 0) {
        const iss::g4::QidTick final_token{iss::g4::CpLane::Graphics, 0,
                                            guard.generation, final_tick};
        ASSERT_MSG(buffer_cache.PublishPacket(guard.lease_id, final_token, true),
                   "G4 global graphics final token failed after emission");
        guard.published = true;
        ASSERT_MSG(buffer_cache.FinishPacketRecord(guard.lease_id),
                   "G4 global graphics final lease did not finish");
    } else {
        ASSERT_MSG(buffer_cache.CancelPacket(guard.lease_id),
                   "G4 empty graphics action retained a global lease");
    }
    liverpool->CancelCpConsumer({iss::g4::CpLane::Graphics, guard.tick});
    guard.owner = nullptr;
    guard.lease_id = 0;
}

Rasterizer::AscHostWriteGuard& Rasterizer::AscHostWriteGuard::operator=(
    AscHostWriteGuard&& other) noexcept {
    if (this != &other) {
        if (owner) owner->FinishAscHostWrite(*this);
        owner = std::exchange(other.owner, nullptr);
        lease_id = other.lease_id;
        tick = other.tick;
        generation = other.generation;
        qid = other.qid;
        address = other.address;
        size = other.size;
    }
    return *this;
}

Rasterizer::AscHostWriteGuard::~AscHostWriteGuard() {
    if (owner) owner->FinishAscHostWrite(*this);
}

Rasterizer::AscHostWriteAdmission Rasterizer::TryPrepareAscHostWrite(VAddr address,u64 size) {
    const s32 selected_qid = DrawQueue();
    if (!cp_split_enabled || !liverpool->CpSplitActive() || !compute_scheduler ||
        selected_qid <= 0 || selected_qid >= 57 || !size ||
        address > std::numeric_limits<VAddr>::max() - size)
        return {GraphicsHostWriteStatus::Failed, {}};
    const u32 qid = static_cast<u32>(selected_qid);
    const u64 generation = CurrentSubmissionGeneration();
    const u64 tick = compute_scheduler->CurrentTick();
    if (!generation || !tick || tick != liverpool->CpOpenTick(iss::g4::CpLane::Asc))
        return {GraphicsHostWriteStatus::Busy, {}};
    const u64 sequence = asc_host_write_sequence.fetch_add(1, std::memory_order_relaxed);
    if (!sequence) return {GraphicsHostWriteStatus::Failed, {}};
    const iss::g4::PacketLeaseKey key{iss::g4::CpLane::Asc,qid,generation,sequence};
    const iss::g4::PacketResourceAccess global{iss::g4::GlobalResourceKey,0,1,true};
    const auto reserve = buffer_cache.TryReservePacket(key,std::span{&global,1},
                                                       buffer_cache.MappingGeneration());
    if (reserve.status == iss::g4::PacketReserveStatus::Busy ||
        reserve.status == iss::g4::PacketReserveStatus::Stale)
        return {GraphicsHostWriteStatus::Busy, {}};
    if (reserve.status != iss::g4::PacketReserveStatus::Reserved)
        return {GraphicsHostWriteStatus::Failed, {}};
    return {GraphicsHostWriteStatus::Ready,
            AscHostWriteGuard{this,reserve.lease_id,tick,qid,generation,address,size}};
}

bool Rasterizer::ResolveAscHostWrite(AscHostWriteGuard& guard) {
    if (guard.owner != this || !guard.lease_id || !compute_scheduler ||
        guard.qid != static_cast<u32>(DrawQueue()) ||
        guard.generation != CurrentSubmissionGeneration() ||
        compute_scheduler->CurrentTick() != guard.tick) return false;
    const iss::g4::CpTick consumer{iss::g4::CpLane::Asc,guard.tick};
    for (const auto& producer : buffer_cache.PacketDependencies(guard.lease_id)) {
        if (producer.lane != iss::g4::CpLane::Graphics) continue;
        auto seal = liverpool->SealDependency(consumer,producer);
        if (!seal || liverpool->WaitCpSeal(seal,iss::g4::CpLane::Asc) !=
                         iss::g4::SealState::Done ||
            scheduler.SubmittedTick() < producer.tick) return false;
        scheduler.Wait(producer.tick); // CPU store follows actual G GPU completion
    }
    // Same-lane C GPU writers/readers must complete too. No cache/page lock is held.
    compute_scheduler->Flush();
    compute_scheduler->WaitSubmitted(guard.tick);
    compute_scheduler->Wait(guard.tick);
    return buffer_cache.RevalidatePacket(guard.lease_id,true);
}

void Rasterizer::FinishAscHostWrite(AscHostWriteGuard& guard) noexcept {
    if (guard.owner != this) return;
    if (guard.lease_id) {
        [[maybe_unused]] const bool cancelled = buffer_cache.CancelPacket(guard.lease_id);
        liverpool->CancelCpConsumer({iss::g4::CpLane::Asc,guard.tick});
    }
    guard.owner = nullptr;
    guard.lease_id = 0;
}

// ISS_DISPATCH_CYCLES=1: where a direct dispatch spends its cycles (command processor thread).
struct DispatchCycles {
    bool enabled = [] {
        const char* value = std::getenv("ISS_DISPATCH_CYCLES");
        return value && value[0] == '1';
    }();
    bool active{};
    std::array<u64, 8> cycles{};
    u64 count{}, hle{}, last{};
    void Start() {
        active = true;
        last = __rdtsc();
    }
    void Mark(u32 phase) {
        if (!active) {
            return;
        }
        const u64 now = __rdtsc();
        cycles[phase] += now - last;
        last = now;
    }
    void Done() {
        if (!active) {
            return;
        }
        active = false;
        if ((++count & 0x3FFF) != 0) {
            return;
        }
        static constexpr std::array names{"pre",    "lookup", "hle_queue", "bind_res",
                                          "barrier", "record", "reset",     "other"};
        std::string line = fmt::format("DISPATCH_CYCLES n={} hle={}", count, hle);
        for (u32 i = 0; i < cycles.size(); ++i) {
            line += fmt::format(" {}={:.0f}", names[i], double(cycles[i]) / double(count));
        }
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
    }
};
static DispatchCycles g_dispatch_cycles;

void Rasterizer::DispatchDirect() {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::Dispatch};
    auto& dc = g_dispatch_cycles;
    if (dc.enabled && !OnDrawWorker()) dc.Start();
    BumpBindEpoch();
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Dispatch);
    RENDERER_TRACE;

    const bool pipelined=UseDrawPipe();
    if(!pipelined){DrainDrawPipe(5);scheduler.PopPendingOperations();}

    const auto& cs_program = DrawCsRegs();
    if (cs_program.dim_x == 0 || cs_program.dim_y == 0 || cs_program.dim_z == 0) [[unlikely]] {
        // No workgroup to run (skip_empty_draws).
        Common::Perf::Add(Common::Perf::EmptyDispatches);
        if (Common::Perf::GetTuning().skip_empty_draws.load(std::memory_order_relaxed)) {
            return;
        }
    }
    if (dc.active) dc.Mark(0);
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(ActiveCpOwner());
    if (dc.active) dc.Mark(1);
    if (!pipeline) {
        dc.active = false;
        return;
    }

    // Prepare the complete SRT read set before a split packet can touch BindResources.
    // Startup still denies CP_SPLIT until reserve/seal/publish/record is wired for all kinds.
    [[maybe_unused]] std::optional<PreparedComputeInputs> prepared;
    if (cp_split_enabled) {
        prepared = PrepareComputeInputs(pipeline);
        ASSERT_MSG(prepared.has_value(), "G4 compute packet has no complete SRT/resource plan");
    }

    if(pipelined && PostDrawPipe(pipeline,true,false,0)){dc.active=false;return;}
    if(pipelined){DrainDrawPipe(5);scheduler.PopPendingOperations();}
    DispatchRecord(pipeline);
    dc.Done();
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    std::optional<iss::diagnostics::Scope> worker_scope;
    if(OnDrawWorker())worker_scope.emplace(iss::diagnostics::Dispatch);
    const auto& cs_program=DrawCsRegs();
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    const bool feedback=WritesFeedbackRecord(cs);
    if(!feedback)SubmitFeedbackBatch(scheduler);
    if (ExecuteShaderHLE(cs, DrawRegs(), cs_program, *this)) {
        if (g_dump.frames_left != 0) [[unlikely]] {
            DumpCommand("dispatch-hle", pipeline, "");
        }
        ++g_dispatch_cycles.hle;
        return;
    }

    // asc_queue: the dispatch, its bindings and uploads go to the compute queue.
    const ComputeQueueScope queue_scope{*this, UseComputeQueue(pipeline)};
    auto& set_writes = ComputeBindingsActive() ? compute_bindings.set_writes : this->set_writes;
    auto& push_data = ComputeBindingsActive() ? compute_bindings.push_data : this->push_data;
    auto& needs_barrier = ComputeBindingsActive() ? compute_bindings.needs_barrier
                                                   : this->needs_barrier;
    const bool hazard_diag=iss::hazard_probe::Enabled();
    if(hazard_diag)iss::hazard_probe::BeginDispatch(cs.pgm_hash,0,Sched().CurrentTick(),u32(DrawQueue()),u32(cs.hw_stage),needs_barrier);
    auto& dc = g_dispatch_cycles;
    dc.Mark(2);
    // ISS_NATIVE_COMPUTE: a learn request's dispatch keeps what BindResources resolves.
    const u64 compute_learn =
        native_compute_learn && !OnDrawWorker() && !ComputeBindingsActive() ? native_compute_learn : 0;
    std::vector<NativeImageCapture> compute_capture;
    if (compute_learn) native_image_capture = &compute_capture;
    const bool bound_resources = BindResources(pipeline);
    if (compute_learn) native_image_capture = nullptr;
    if (!bound_resources) {
        if(hazard_diag)iss::hazard_probe::AbortDispatch(1);
        return;
    }
    if (compute_learn) {
        native_compute_learn = 0;
        NativeComputeLearn(compute_learn, pipeline, compute_capture);
    }
    dc.Mark(3);
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpCommand("dispatch", pipeline,
                    fmt::format("g{}x{}x{}{}", cs_program.dim_x, cs_program.dim_y,
                                cs_program.dim_z, OnComputeRecordQueue() ? " cq" : ""));
    }

    // Compute phases (see compute_phase.h): a dispatch of the compute queues following the
    // dispatches of an open run with nothing but register writes in between needs no barrier
    // for hazards with them only.
    if (iss::command_trace::enabled) {
        iss::command_trace::Stage("dispatch", DrawQueue(), cs,
                                  cs_program.Address<u64>(), cs_program.dim_x, cs_program.dim_y,
                                  cs_program.dim_z, Sched().CurrentTick(), needs_barrier, 0, 0);
    }
    const bool phase = (iss::phase::Enabled() && DrawAsync() && !OnComputeRecordQueue()) ||
                       (iss::ring_barriers::enabled && DrawAsync() &&
                        iss::ring_barriers::Audited(cs.pgm_hash));
    if (phase) {
        iss::phase::g_owner.store(&Rt(), std::memory_order_relaxed);
        if (iss::phase::g_boundary.exchange(false, std::memory_order_relaxed)) {
            iss::phase::g_open.store(false, std::memory_order_relaxed);
        }
    }
    bool hazard_flushed=false;
    if (needs_barrier) {
        if (phase && iss::phase::g_open.load(std::memory_order_relaxed) &&
            !iss::phase::g_foreign.load(std::memory_order_relaxed)) {
            iss::phase::g_skipped.fetch_add(1, std::memory_order_relaxed);
        } else {
            Rt().FlushBarriers();
            hazard_flushed=true;
            if (phase) {
                iss::phase::g_flushed.fetch_add(1, std::memory_order_relaxed);
                iss::phase::g_open.store(true, std::memory_order_relaxed);
                iss::phase::g_foreign.store(false, std::memory_order_relaxed);
            }
        }
    }

    if(hazard_diag)iss::hazard_probe::EndDispatch(needs_barrier,hazard_flushed);
    dc.Mark(4);
    if (!OnComputeRecordQueue()) {
        scheduler.EndRendering();
    }
    pipeline->BindResources(Sched(), set_writes, push_data);

    const auto cmdbuf = Sched().Recorder();
    BindPipelineLazy(Sched(), vk::PipelineBindPoint::eCompute, pipeline);
    if (GpuCrashDiag::Enabled() || GpuCrashDiag::Blackbox()) {
        MarkDispatch(Sched(), cmdbuf, cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                     cs_program.dim_z, 0);
    }
    Sched().ProfileDispatchBegin(cs.pgm_hash);
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    iss::feedback_join::Recorded();
    Sched().ProfileDispatchEnd();
    DebugState.IncDispatch();

    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpAfter(pipeline);
    }
    // The accesses of a dispatch of the open run belong to the run.
    const bool run_access = phase && iss::phase::g_open.load(std::memory_order_relaxed);
    iss::phase::g_run_access.store(run_access, std::memory_order_relaxed);
    dc.Mark(5);
    ResetBindings(true);
    iss::phase::g_run_access.store(false, std::memory_order_relaxed);
    dc.Mark(6);
    if(feedback&&!OnComputeRecordQueue())pending_feedback_tick=scheduler.CurrentTick();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::Dispatch};
    BumpBindEpoch();
    DrainDrawPipe(5);
    SubmitFeedbackBatch(scheduler);
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = DrawCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(ActiveCpOwner());
    if (!pipeline) {
        return;
    }
    [[maybe_unused]] std::optional<PreparedComputeInputs> prepared;
    if (cp_split_enabled) {
        const u64 args_address = address + offset;
        ASSERT_MSG(args_address >= address, "G4 indirect argument address overflow");
        prepared = PrepareComputeInputs(
            pipeline, iss::g4::IndirectArgsRef{args_address, buffer_cache.MappingGeneration(), size});
        ASSERT_MSG(prepared.has_value(), "G4 indirect packet has no complete SRT/resource plan");
    }
    if (ReadsSkippedWrites(address + offset, size)) {
        Common::Perf::Add(Common::Perf::PipelineSkips);
        RecordSkippedWrites(*pipeline);
        if (g_dump.frames_left != 0) [[unlikely]] {
            DumpCommand("dispatchind-skipped", pipeline, fmt::format("args{:x}", address + offset));
        }
        return;
    }

    // asc_queue: the dispatch, its bindings and uploads go to the compute queue.
    const ComputeQueueScope queue_scope{*this, UseComputeQueue(pipeline)};
    auto& set_writes = ComputeBindingsActive() ? compute_bindings.set_writes : this->set_writes;
    auto& push_data = ComputeBindingsActive() ? compute_bindings.push_data : this->push_data;
    auto& needs_barrier = ComputeBindingsActive() ? compute_bindings.needs_barrier
                                                   : this->needs_barrier;
    if (!BindResources(pipeline)) {
        return;
    }
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpCommand("dispatchind", pipeline,
                    fmt::format("args{:x}{}", address + offset, OnComputeRecordQueue() ? " cq" : ""));
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= Rt().IsBufferAccessed(buffer, base, size);
    if (iss::command_trace::enabled) {
        iss::command_trace::Stage("dispatch_indirect", DrawQueue(),
                                  pipeline->GetStage(Shader::SwStage::Compute),
                                  cs_program.Address<u64>(), 0, 0, 0, Sched().CurrentTick(),
                                  needs_barrier, address + offset, 0);
    }

    if (needs_barrier) {
        Rt().FlushBarriers();
    }

    if (!OnComputeRecordQueue()) {
        scheduler.EndRendering();
    }
    pipeline->BindResources(Sched(), set_writes, push_data);

    const auto cmdbuf = Sched().Recorder();
    BindPipelineLazy(Sched(), vk::PipelineBindPoint::eCompute, pipeline);
    if (GpuCrashDiag::Enabled() || GpuCrashDiag::Blackbox()) {
        MarkDispatch(Sched(), cmdbuf, pipeline->GetStage(Shader::SwStage::Compute).pgm_hash, 0, 0,
                     0, address + offset);
    }
    Sched().ProfileDispatchBegin(pipeline->GetStage(Shader::SwStage::Compute).pgm_hash);
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    iss::feedback_join::Recorded();
    Sched().ProfileDispatchEnd();
    DebugState.IncDispatch();

    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpAfter(pipeline);
    }
    ResetBindings(true);
}

u64 Rasterizer::Flush() {
    DrainDrawPipe(5);
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    DrainDrawPipe(5);
    scheduler.Finish();
}

#include "occlusion_diag.inl"

void Rasterizer::OnSubmit() {
    BumpBindEpoch();
    DrainDrawPipe(5);
    if (iss::command_trace::enabled) iss::command_trace::Submit();
    if (Common::Perf::GetTuning().asc_queue.load(std::memory_order_relaxed)) {
        // asc_queue: cumulative cross-queue counters every 5 s (stderr, CP thread only).
        static u64 last_print{};
        if (const u64 now = Common::Perf::NowNs(); now - last_print >= 5'000'000'000ull) {
            last_print = now;
            using namespace Common::Perf;
            std::fprintf(stderr, "ASC_STATS frames=%llu asc_dispatches=%llu fallbacks=%llu "
                         "(img=%llu smp=%llu dma=%llu nopush=%llu) compute_submits=%llu "
                         "waits_gfx=%llu waits_compute=%llu flushes=%llu first_touches=%llu",
                         u64(GuestFrameCounter().load()), Total(AscQueueDispatches), Total(AscQueueFallbacks),
                         iss::asc_stats::fallback[0].load(), iss::asc_stats::fallback[1].load(),
                         iss::asc_stats::fallback[2].load(), iss::asc_stats::fallback[3].load(),
                         Total(ComputeQueueSubmits), Total(CrossWaitsGraphics),
                         Total(CrossWaitsCompute), Total(CrossQueueFlushes),
                         Total(CrossQueueFirstTouches));
            for (u32 i = 0; i < iss::asc_stats::Num; ++i) {
                std::fprintf(stderr, " %s=%llu/%.1fms", iss::asc_stats::names[i],
                             iss::asc_stats::count[i].load(),
                             double(iss::asc_stats::ns[i].load()) / 1e6);
            }
            std::fputc('\n', stderr);
            iss::asc_stats_api::Print();
        }
    }
    if (iss::ring_barriers::enabled) {
        static u64 last_ring_print{};
        if (const u64 now = Common::Perf::NowNs(); now - last_ring_print >= 5'000'000'000ull) {
            last_ring_print = now;
            std::fprintf(stderr, "RING_BARRIERS frames=%llu skipped=%llu flushed=%llu\n",
                         u64(Common::Perf::GuestFrameCounter().load()),
                         iss::phase::g_skipped.load(), iss::phase::g_flushed.load());
        }
    }
    // Frame x-ray: a new request dumps the next 2 guest frames.
    if (g_dump.frames_left != 0) {
        LOG_INFO(Render_Vulkan, "DUMP f{} end of frame ({} commands)", g_dump.frame, g_dump.seq);
        const auto& tuning = Common::Perf::GetTuning();
        for (size_t i = 0; i < tuning.dump_images.size(); ++i) {
            const u64 address = tuning.dump_images[i].load(std::memory_order_relaxed);
            if (address != 0) {
                const u64 size = tuning.dump_images_size[i].load(std::memory_order_relaxed);
                texture_cache.DumpImageToFile(address, fmt::format("f{}", g_dump.frame),
                                              static_cast<u32>(size >> 32),
                                              static_cast<u32>(size));
            }
        }
        g_dump.after_counts.fill(0);
        --g_dump.frames_left;
        ++g_dump.frame;
        g_dump.seq = 0;
        Common::Perf::FrameDumpActive().store(g_dump.frames_left != 0, std::memory_order_relaxed);
        if (g_dump.frames_left == 0) {
            FinishProbes();
        }
    }
    const u32 dump_request =
        Common::Perf::GetTuning().dump_frames.load(std::memory_order_relaxed);
    if (!g_dump.started) {
        g_dump.started = true;
        g_dump.last_request = dump_request;
    }
    if (dump_request != g_dump.last_request) {
        g_dump.last_request = dump_request;
        if (dump_request != 0 && g_dump.frames_left == 0) {
            // The value is the number of frames (2 to 30).
            g_dump.frames_left = std::clamp<u32>(dump_request, 2, 30);
            g_dump.frame = 0;
            g_dump.seq = 0;
            Common::Perf::FrameDumpActive().store(true, std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan, "DUMP request {}: next 2 guest frames", dump_request);
        }
    }
    occlusion.OnFrameEnd();
    if (occlusion_diag::enabled) [[unlikely]] {
        occlusion_diag::Collect(instance, scheduler, occlusion.Pool());
        occlusion_diag::Report();
    }
    iss::unified::MaybeReport();
    scheduler.CollectProfile();
    buffer_cache.TickFrame();
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
    if (compute_queue_routing_enabled) {
        compute_runtime->TickFrame();
    }
}

void Rasterizer::OnFence() {
    DrainDrawPipe(5);
    texture_cache.ProcessDownloadImages();
    buffer_cache.PrefetchReadbacks();
}

void Rasterizer::RetireReadbacks() {
    // An early retire is only an optimization (a guest fault completes the readback anyway):
    // with ISS_DRAW_PIPE_LAZY it waits for a moment when stage B is idle.
    if(DrawPipeBusyLazy(draw_pipe_state.get()))return;
    DrainDrawPipe(5);
    buffer_cache.RetireReadbacks();
}

void Rasterizer::PrefetchDrawLookups(bool is_indexed, u32 index_offset) {
    // The entries of the current draw were prefetched with the draw before, unless it had none.
    if (const auto* draw = liverpool->CurrentScoutDraw(); draw != nullptr) {
        if (draw != last_prefetched_draw) {
            PrefetchScoutLookups(*draw);
        }
        // The memory its small buffers will copy, now that their stream entries are cached.
        const u32 num_stages = std::min<u32>(draw->num_stages, AmdGpu::ScoutDraw::MaxStages);
        for (u32 s = 0; s < num_stages; ++s) {
            const auto& stage = draw->stages[s];
            if (!stage.valid) {
                continue;
            }
            for (u32 i = 0; i < stage.num_buffer_ranges; ++i) {
                buffer_cache.PrefetchStreamSource(stage.buffer_addresses[i],
                                                  stage.buffer_sizes[i]);
            }
        }
    }
    if (const auto* next = liverpool->PeekNextScoutDraw(); next != nullptr) {
        PrefetchScoutLookups(*next);
        last_prefetched_draw = next;
    }
    if (is_indexed) {
        // As BindIndexBuffer, which runs last.
        const auto& regs = DrawRegs();
        const u32 index_size =
            regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
        buffer_cache.PrefetchLookup(
            regs.index_base_address.Address<VAddr>() + u64{index_offset} * index_size,
            u64{regs.num_indices} * index_size);
    }
}

void Rasterizer::PrefetchScoutLookups(const AmdGpu::ScoutDraw& draw) {
    Common::Perf::Add(Common::Perf::PrefetchedDraws);
    const u32 num_stages = std::min<u32>(draw.num_stages, AmdGpu::ScoutDraw::MaxStages);
    for (u32 s = 0; s < num_stages; ++s) {
        const auto& stage = draw.stages[s];
        if (!stage.valid) {
            continue;
        }
        for (u32 i = 0; i < stage.num_buffer_ranges; ++i) {
            buffer_cache.PrefetchLookup(stage.buffer_addresses[i], stage.buffer_sizes[i]);
        }
        for (u32 i = 0; i < stage.num_image_keys; ++i) {
            PrefetchTexMemo(stage.image_keys[i]);
        }
    }
}

void Rasterizer::PrefetchTexMemo(u64 key) const noexcept {
    if (!tex_memos) {
        return;
    }
    const auto* entry = &(*tex_memos)[(key * 0x9E3779B97F4A7C15ULL) >> (64 - NumTexMemoBits)];
    VideoCore::PrefetchLine(entry);
    VideoCore::PrefetchLine(reinterpret_cast<const u8*>(entry) + sizeof(TexMemo) - 1);
}

static bool StageMemoEnabled();
static bool StageMemoVerify();
static bool StageMemoCycles();

bool Rasterizer::BindResources(const Pipeline* pipeline,
                               const Shader::PushData* captured_push_data,
                               bool skip_compute_hle) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::BindRes};
    iss::feedback_join::BeginBindings();
    ++bind_resources_seq;
    auto& set_write_index = ComputeBindingsActive() ? compute_bindings.set_write_index
                                                    : this->set_write_index;
    auto& set_writes = ComputeBindingsActive() ? compute_bindings.set_writes : this->set_writes;
    auto& buffer_infos = ComputeBindingsActive() ? compute_bindings.buffer_infos : this->buffer_infos;
    auto& image_infos = ComputeBindingsActive() ? compute_bindings.image_infos : this->image_infos;
    auto& push_data = ComputeBindingsActive() ? compute_bindings.push_data : this->push_data;
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Resources);
    if (!skip_compute_hle &&
        (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
         IsComputeImageClear(pipeline))) {
        return false;
    }
    if (!pipeline->IsReady() && !LazyPipelinesEnabled()) {
        // Still compiled by a worker thread (only the emulated dispatches above can run).
        Common::Perf::Add(Common::Perf::PipelineSkips);
        RecordSkippedWrites(*pipeline);
        return false;
    }

    set_write_index = 0;
    static const bool reuse_descriptor_storage = [] {
        const char* value = std::getenv("SHADPS4_REUSE_DESCRIPTOR_STORAGE");
        const bool enabled = value && value[0] == '1';
        LOG_INFO(Render_Vulkan, "Descriptor storage reuse: {}", enabled);
        return enabled;
    }();
    size_t write_count = 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (stage) {
            write_count += stage->buffers.size() + stage->images.size() + stage->samplers.size();
        }
    }
    PrepareDescriptorWrites(set_writes, write_count, reuse_descriptor_storage);
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = captured_push_data ? *captured_push_data : MakeUserData(DrawRegs());
    // ISS_NATIVE_MATERIALS learns from the descriptors resolved here, not from a replay.
    const bool stage_memo = StageMemoEnabled() && !captured_push_data && !pipeline->IsCompute() &&
                            !native_image_capture && StageMemoContextEligible();
    if (StageMemoEnabled() && !pipeline->IsCompute() && ++stage_memo_stats.report >= 262144) {
        stage_memo_stats.report = 0;
        ReportStageMemo();
    }
    const auto stages = pipeline->GetStages();
    for (u32 slot = 0; slot < stages.size(); ++slot) {
        const auto* stage = stages[slot];
        if (!stage) {
            continue;
        }
        StageMemoCapture capture{};
        bool verify_hit = false;
        const bool cycles = stage_memo && StageMemoCycles();
        u64 t0 = cycles ? __rdtsc() : 0;
        if (stage_memo) {
            const bool verify = StageMemoVerify();
            if (ReplayStageMemo(slot, *stage, pipeline, binding, push_data, verify)) {
                if (!verify) {
                    uses_dma |= stage->uses_dma;
                    if (cycles) stage_memo_stats.cyc_hit += __rdtsc() - t0;
                    continue;
                }
                verify_hit = true;
            }
            if (cycles) {
                const u64 t1 = __rdtsc();
                stage_memo_stats.cyc_check += t1 - t0;
                t0 = t1;
            }
            if (verify_hit || StageMemoStageEligible(*stage)) {
                capture = {
                    .active = true,
                    .stamp = CaptureStageMemoStamp(),
                    .tick = scheduler.CurrentTick(),
                    .barrier_before = needs_barrier,
                    .binding_in = binding,
                    .buffer_infos = static_cast<u32>(buffer_infos.size()),
                    .image_infos = static_cast<u32>(image_infos.size()),
                    .writes = set_write_index,
                    .bound_buffers = static_cast<u32>(bound_buffers.size()),
                    .bound_images = static_cast<u32>(bound_images.size()),
                };
                needs_barrier = false;
            }
        }
        if (!pipeline->IsCompute() && slot == u32(Shader::SwStage::Fragment) &&
            BindNativeMaterial(pipeline, *stage, binding, push_data)) {
            if (capture.active) {
                RememberStageMemo(slot, *stage, pipeline, capture, binding, push_data, false);
                needs_barrier |= capture.barrier_before;
            }
            uses_dma |= stage->uses_dma;
            continue;
        }
        if (cycles) t0 = __rdtsc();
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
        if (cycles) {
            const u64 t1 = __rdtsc();
            stage_memo_stats.cyc_original += t1 - t0;
            ++stage_memo_stats.originals;
            stage_memo_stats.slot_cyc_original[slot & 7] += t1 - t0;
            ++stage_memo_stats.slot_originals[slot & 7];
            t0 = t1;
        }
        if (capture.active) {
            RememberStageMemo(slot, *stage, pipeline, capture, binding, push_data, verify_hit);
            needs_barrier |= capture.barrier_before;
            if (cycles) stage_memo_stats.cyc_remember += __rdtsc() - t0;
        }
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }

    ASSERT(set_write_index == set_writes.size());

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::BindVertex};
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Vertices);
    const auto& regs = DrawRegs();
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.Recorder();
        if (!keep_vertex_input) {
            cmdbuf.setVertexInputEXT(bindings, attributes);
        }
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    // The city renderer's dominant indexed draw uses three CPU-owned V# streams. Upload
    // them as one native host-visible allocation instead of resolving three guest ranges
    // through the general buffer cache. Keep the ordinary path for GPU-produced data,
    // unmapped ranges, and every other shader or vertex layout.
    static const bool native_city_fetch = [] {
        const char* value = std::getenv("ISS_NATIVE_CITY_FETCH");
        return value && value[0] == '1';
    }();
    if (native_city_fetch && DrawQueue() == 0 && guest_buffers.size() == 3 &&
        pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash == 0x0BEB7156 &&
        pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash == 0x6353E48B) {
        struct NativeRange {
            VAddr base;
            VAddr end;
            u64 stream_offset;
        };
        std::array<NativeRange, 3> source_ranges{};
        bool cpu_owned = true;
        for (u32 i = 0; i < 3; ++i) {
            const auto& guest = guest_buffers[i];
            const u64 size = guest.GetSize();
            if (!guest.base_address || !size || size > 16 * 1024 ||
                memory->ClampRangeSize(guest.base_address, size) != size ||
                buffer_cache.HasGpuData(guest.base_address, size)) {
                cpu_owned = false;
                break;
            }
            source_ranges[i] = {guest.base_address, guest.base_address + size, 0};
        }
        if (cpu_owned) {
            std::ranges::sort(source_ranges, [](const NativeRange& a, const NativeRange& b) {
                return a.base < b.base;
            });
            std::array<NativeRange, 3> merged{};
            u32 merged_count = 0;
            for (const auto& range : source_ranges) {
                if (merged_count && range.base <= merged[merged_count - 1].end) {
                    merged[merged_count - 1].end =
                        std::max(merged[merged_count - 1].end, range.end);
                } else {
                    merged[merged_count++] = range;
                }
            }
            u64 total_size = 0;
            for (u32 i = 0; i < merged_count; ++i) {
                merged[i].stream_offset = total_size;
                total_size += merged[i].end - merged[i].base;
            }
            auto& stream = buffer_cache.GetStreamBuffer();
            const auto [mapped, stream_offset] = stream.Map(total_size, 4);
            if (mapped) {
                VertexInputs<vk::Buffer> host_buffers;
                VertexInputs<vk::DeviceSize> host_offsets;
                VertexInputs<vk::DeviceSize> host_sizes;
                VertexInputs<vk::DeviceSize> host_strides;
                for (u32 i = 0; i < merged_count; ++i) {
                    const auto& range = merged[i];
                    memory->CopySparseMemory(range.base, mapped + range.stream_offset,
                                             range.end - range.base);
                }
                for (const auto& guest : guest_buffers) {
                    const u64 size = guest.GetSize();
                    const auto found = std::ranges::find_if(merged.begin(),
                        merged.begin() + merged_count, [&](const NativeRange& range) {
                            return guest.base_address >= range.base &&
                                   guest.base_address + size <= range.end;
                        });
                    ASSERT(found != merged.begin() + merged_count);
                    host_buffers.emplace_back(stream.Handle());
                    host_offsets.push_back(stream_offset + found->stream_offset +
                                           guest.base_address - found->base);
                    host_sizes.push_back(size);
                    host_strides.push_back(guest.GetStride());
                }
                stream.Commit();
                const auto cmdbuf = scheduler.Recorder();
                if (instance.IsVertexInputDynamicState()) {
                    cmdbuf.bindVertexBuffers(0, 3, host_buffers.data(), host_offsets.data());
                } else {
                    cmdbuf.bindVertexBuffers2(0, 3, host_buffers.data(), host_offsets.data(),
                                              host_sizes.data(), host_strides.data());
                }
                return;
            }
        }
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    if (g_dump.frames_left != 0) [[unlikely]] {
        dump_vertex_ranges.clear();
    }
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        if (iss::repeat_census::enabled) [[unlikely]]
            iss::repeat_census::Note(iss::repeat_census::VertexRange, range.base_address, size);
        bool known_unwritten = false;
        const bool reuse_check = Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false, false,
                                     reuse_check ? &known_unwritten : nullptr);
        if (range.buffer != &buffer_cache.GetStreamBuffer()) {
            if (known_unwritten) Common::Perf::Add(Common::Perf::BufferChecksReused);
            else needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        }
        if (g_dump.frames_left != 0) [[unlikely]] {
            if (dump_vertex_ranges.size() < dump_vertex_ranges.capacity()) {
                dump_vertex_ranges.push_back(BoundVertexRange{
                    .base = range.base_address,
                    .size = size,
                    .buffer = range.buffer,
                    .offset = range.offset,
                });
            }
            DumpNote("vb", fmt::format("{:x}+{:x} {} [{}]", range.base_address, size,
                                       range.buffer == &buffer_cache.GetStreamBuffer()
                                           ? "stream"
                                           : "arena",
                                       buffer_cache.DescribePages(range.base_address, size)));
        }
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.Recorder();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::BindIndex};
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Indices);
    const auto& regs = DrawRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    if (iss::repeat_census::enabled) [[unlikely]]
        iss::repeat_census::Note(iss::repeat_census::IndexRange, index_address, index_buffer_size);
    bool known_unwritten = false;
    const bool reuse_check = Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false, false,
                                 reuse_check ? &known_unwritten : nullptr);
    if (buffer != &buffer_cache.GetStreamBuffer()) {
        if (known_unwritten) Common::Perf::Add(Common::Perf::BufferChecksReused);
        else needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    }
    const auto cmdbuf = scheduler.Recorder();
    cmdbuf.bindIndexBuffer(buffer->Handle(), offset, index_type);
}

void Rasterizer::ResetBindings(bool is_compute) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::Reset};
    auto& bound_images = ComputeBindingsActive() ? compute_bindings.bound_images
                                                 : this->bound_images;
    auto& bound_buffers = ComputeBindingsActive() ? compute_bindings.bound_buffers
                                                  : this->bound_buffers;
    auto& needs_barrier = ComputeBindingsActive() ? compute_bindings.needs_barrier
                                                   : this->needs_barrier;
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written, address] : bound_buffers) {
        if (is_written) {
            BumpBindEpoch();
        }
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        Rt().AccessBuffer(buffer, offset, size, dst_stage,
                          vk::AccessFlagBits2::eShaderRead | write_flag);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

void Rasterizer::RecordSkippedWrites(const Pipeline& pipeline) {
    for (const auto* stage : pipeline.GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& desc : stage->buffers) {
            if (!desc.is_written || desc.IsSpecial()) {
                continue;
            }
            const auto vsharp = desc.GetSharp(*stage);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                continue;
            }
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            if (size == 0) {
                continue;
            }
            if (skipped_writes.size() == skipped_writes.capacity()) {
                skipped_writes.erase(skipped_writes.begin());
            }
            skipped_writes.push_back(SkippedWrite{
                .start = vsharp.base_address,
                .end = vsharp.base_address + size,
                .producer = &pipeline,
            });
        }
    }
}

bool Rasterizer::ReadsSkippedWrites(VAddr address, u64 size) {
    if (skipped_writes.empty()) {
        return false;
    }
    // The data stays stale only until the producer is compiled and runs again, and it may never
    // run again (a one-shot pass): entries expire once their pipeline is ready.
    for (auto it = skipped_writes.begin(); it != skipped_writes.end();) {
        if (it->producer->IsReady()) {
            it = skipped_writes.erase(it);
        } else {
            ++it;
        }
    }
    return std::ranges::any_of(skipped_writes, [address, size](const SkippedWrite& range) {
        return address < range.end && range.start < address + size;
    });
}

void Rasterizer::ClearSkippedWrites(VAddr address, u64 size) {
    for (auto it = skipped_writes.begin(); it != skipped_writes.end();) {
        if (address < it->end && it->start < address + size) {
            it = skipped_writes.erase(it);
        } else {
            ++it;
        }
    }
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Two data buffers, optionally followed by the flattened user-data buffer.
    const auto& cs_pgm = DrawCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Two data buffers, optionally followed by the flattened user-data buffer.
    const auto& cs_pgm = DrawCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

namespace {
struct TracedRange {
    VAddr start;
    VAddr end;
};
std::array<TracedRange, 64> g_traced_ranges{};
u32 g_num_traced_ranges = 0;
std::array<u64, 32> g_traced_writers{};
u32 g_num_traced_writers = 0;
} // Anonymous namespace

#include "stage_bind_memo.inl"
#include "fast_draw.inl"
#include "native_items.inl"
#include "native_visual_probe.inl"
#include "native_depth_run.inl"
#include "native_shadow_rasterizer.inl"
#include "native_material_rasterizer.inl"
#include "native_recorder.inl"
#include "native_gbuffer_rasterizer.inl"
#include "native_passes_rasterizer.inl"
#include "native_compute_rasterizer.inl"

void Rasterizer::TraceBufferWriters(const Shader::Info& stage) {
    if (g_num_traced_ranges == 0 || g_num_traced_writers >= g_traced_writers.size()) {
        return;
    }
    for (const auto& desc : stage.buffers) {
        if (desc.IsSpecial() || !desc.is_written) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        const VAddr start = vsharp.base_address;
        const VAddr end = start + vsharp.GetSize();
        for (u32 r = 0; r < g_num_traced_ranges; ++r) {
            const auto& range = g_traced_ranges[r];
            if (start >= range.end || end <= range.start) {
                continue;
            }
            const u64 key = stage.pgm_hash ^ (u64(r) << 48);
            if (std::find(g_traced_writers.begin(), g_traced_writers.begin() + g_num_traced_writers,
                          key) != g_traced_writers.begin() + g_num_traced_writers) {
                continue;
            }
            g_traced_writers[g_num_traced_writers++] = key;
            const auto& cs = DrawCsRegs();
            LOG_INFO(
                Render_Vulkan,
                "Trace writer: {} shader {:#x} writes {:#x}..{:#x} (traced range {:#x}..{:#x}) "
                "dispatch {}x{}x{}",
                stage.hw_stage == Shader::HwStage::Compute ? "compute" : "graphics", stage.pgm_hash,
                start, end, range.start, range.end, cs.dim_x, cs.dim_y, cs.dim_z);
            if (g_num_traced_writers >= g_traced_writers.size()) {
                return;
            }
        }
    }
}

void Rasterizer::TraceShaderResources(const Shader::Info& stage) {
    static std::atomic<u32> traced{0};
    if (traced.fetch_add(1, std::memory_order_relaxed) >= 6) {
        return;
    }
    const auto& regs = DrawRegs();
    LOG_INFO(Render_Vulkan, "Trace shader {:#x}: {} buffers, {} images, screen scissor {}x{}",
             stage.pgm_hash, stage.buffers.size(), stage.images.size(),
             regs.screen_scissor.GetWidth(), regs.screen_scissor.GetHeight());
    for (u32 i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        std::string data;
        const u64 size = vsharp.GetSize();
        if (vsharp.base_address != 0 && size != 0 && g_num_traced_ranges < g_traced_ranges.size()) {
            const TracedRange range{vsharp.base_address, vsharp.base_address + size};
            if (std::none_of(g_traced_ranges.begin(), g_traced_ranges.begin() + g_num_traced_ranges,
                             [&](const TracedRange& r) {
                                 return r.start == range.start && r.end == range.end;
                             })) {
                g_traced_ranges[g_num_traced_ranges++] = range;
            }
        }
        if (vsharp.base_address != 0 && size != 0 && size <= 4096 &&
            !buffer_cache.IsRegionGpuModified(vsharp.base_address, size)) {
            const auto* words = reinterpret_cast<const u32*>(vsharp.base_address);
            for (u32 w = 0; w < std::min<u64>(size / 4, 32); ++w) {
                data += fmt::format(" {:08x}", words[w]);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "  buffer {}: base {:#x} size {:#x} stride {} records {} dfmt {} nfmt {} "
                 "written {} formatted {}{}{}",
                 i, vsharp.base_address, size, vsharp.GetStride(), vsharp.num_records,
                 u32(vsharp.GetDataFmt()), u32(vsharp.GetNumberFmt()), desc.is_written,
                 desc.is_formatted, data.empty() ? "" : " data", data);
    }
    for (u32 i = 0; i < stage.images.size(); ++i) {
        const auto tsharp = stage.images[i].GetSharp(stage);
        LOG_INFO(Render_Vulkan, "  image {}: base {:#x} {}x{}x{} dfmt {} nfmt {} type {}", i,
                 tsharp.Address(), tsharp.width + 1, tsharp.height + 1, tsharp.depth + 1,
                 u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()), u32(tsharp.GetType()));
    }
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    auto& buffer_infos = ComputeBindingsActive() ? compute_bindings.buffer_infos : this->buffer_infos;
    auto& bound_buffers = ComputeBindingsActive() ? compute_bindings.bound_buffers
                                                  : this->bound_buffers;
    auto& set_writes = ComputeBindingsActive() ? compute_bindings.set_writes : this->set_writes;
    auto& set_write_index = ComputeBindingsActive() ? compute_bindings.set_write_index
                                                    : this->set_write_index;
    auto& needs_barrier = ComputeBindingsActive() ? compute_bindings.needs_barrier
                                                   : this->needs_barrier;
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Buffers);
    const u64 alignment = instance.StorageMinAlignment();
    const bool hazard_diag=iss::hazard_probe::Enabled();
    // Stage 1 is a serial assertion adapter, not a second command processor.
    static const bool qid1_owner_stage1 = [] {
        const char* value = std::getenv("ISS_CP_OWNER_QID1_STAGE1");
        return value && value[0] == '1';
    }();
    const bool explicit_qid1 = qid1_owner_stage1 && !OnDrawWorker() && DrawQueue() == 1;
    for (const auto& trace : Common::Perf::GetTuning().trace_shaders) {
        const u64 hash = trace.load(std::memory_order_relaxed);
        if (hash == 0) {
            break;
        }
        if (hash == stage.pgm_hash) [[unlikely]] {
            TraceShaderResources(stage);
            break;
        }
    }
    if (g_num_traced_ranges != 0) [[unlikely]] {
        TraceBufferWriters(stage);
    }
    if (Common::Perf::GetTuning().bind_prefetch.load(std::memory_order_relaxed) &&
        stage.buffers.size() > 1) {
        // The lookups of the stage's buffers start together instead of one after the other,
        // unless the scout's entry of the draw had them prefetched already.
        const auto* scouted = (OnDrawWorker() ? nullptr : pipeline_cache.TakenScoutStage(stage));
        if (scouted == nullptr || scouted->num_buffer_ranges == 0) {
            for (const auto& desc : stage.buffers) {
                if (desc.IsSpecial() || desc.is_written) {
                    continue;
                }
                const auto vsharp = desc.GetSharp(stage);
                if (vsharp.base_address != 0 && vsharp.GetSize() != 0) {
                    buffer_cache.PrefetchLookup(vsharp.base_address, vsharp.GetSize());
                }
            }
        }
    }
    for (const auto& desc : stage.buffers) {
        iss::hazard_probe::BufferCheck check{};
        const u32 native_index = static_cast<u32>(&desc - stage.buffers.data());
        const u32 native_bound_start = bound_buffers.size();
        NativeBufferAccess* native_access = native_access_audit
            ? &native_access_audit->buffers[native_index] : nullptr;
        if (native_access) {
            native_access->hazard_before = runtime.HazardSeq();
            const auto& token = native_access_audit->record->buffer_tokens[native_index];
            native_access->cached = token && buffer_cache.ValidatePreparedRead(*token);
        }
        if(hazard_diag){check.descriptor_index=u32(&desc-stage.buffers.data());check.buffer_type=u32(desc.buffer_type);check.is_special=desc.IsSpecial();check.is_written=desc.is_written;check.needs_before=needs_barrier;}
        if (!desc.IsSpecial() && !desc.is_written && !desc.is_formatted &&
            stage.hw_stage == Shader::HwStage::Fragment && NativeContext()) {
            const bool hazard = PrepareNativeReadOnlyBuffer(desc.GetSharp(stage), stage.pgm_hash,
                                                            binding.buffer, push_data);
            if (native_access) native_access->hazard = hazard;
        } else if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                const bool hit=Rt().IsBufferAccessed(gds_buf,0,gds_buf->SizeBytes(),desc.is_written);
                needs_barrier |= hit;
                if(hazard_diag){check.effective_size=gds_buf->SizeBytes();check.backing_id=reinterpret_cast<u64>(gds_buf);check.tracked_size=gds_buf->SizeBytes();check.check_performed=true;check.hit=hit;}
                buffer_cache.NoteGdsAccess(desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                if (NativeContext() && stage.hw_stage == Shader::HwStage::Fragment) {
                    buffer_infos.push_back(PrepareNativeFlatBuffer(stage));
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    const u32 ubo_size = stage.FlatUserData().size() * sizeof(u32);
                    const u64 offset =
                        vk_buffer.Copy(stage.FlatUserData().data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (DrawRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = DrawRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
                buffer_cache.NoteBdaAccess();
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = DrawCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            if (iss::repeat_census::enabled) [[unlikely]]
                iss::repeat_census::Note(iss::repeat_census::StageBuffer, vsharp.base_address,
                                         vsharp.GetSize(), u32(stage.hw_stage));
            if(hazard_diag){AmdGpu::Buffer raw{};desc.sharp_fetch.Fetch(stage.FlatUserData().data(),&raw);std::memcpy(check.raw_vsharp.data(),&raw,sizeof(raw));check.effective_address=vsharp.base_address;check.effective_size=vsharp.GetSize();}
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                // Correlate the game's 48-byte feedback descriptor with the
                // EBOOT call-site observation. This is passive: recycled guest
                // addresses and untracked aliases still require generic GPU
                // binding/readback. No descriptor or synchronization changes.
                static const bool observe_feedback_identity = [] {
                    const char* setting = std::getenv("ISS_FEEDBACK_IDENTITY");
                    return setting && setting[0] == '1';
                }();
                if (observe_feedback_identity &&
                    stage.hw_stage == Shader::HwStage::Compute && size == 48) {
                    const auto observed = iss::game::feedback::IdentityRegistry::Instance()
                                              .Latest(vsharp.base_address);
                    static std::atomic<u64> candidates{}, matches{};
                    const auto candidate = ++candidates;
                    const auto matched = observed ? ++matches : matches.load();
                    if (observed) {
                        thread_local std::unordered_map<u64, u64> hashes;
                        const auto hash_count = ++hashes[stage.pgm_hash];
                        if (hash_count == 1 || (hash_count & (hash_count - 1)) == 0) {
                            std::fprintf(stderr,
                                         "FB_IDENTITY_HASH hash=0x%llx count=%llu "
                                         "descriptor=%u record=0x%llx write=%u stride=%u records=%u\n",
                                         static_cast<unsigned long long>(stage.pgm_hash),
                                         static_cast<unsigned long long>(hash_count),
                                         unsigned(&desc - stage.buffers.data()),
                                         static_cast<unsigned long long>(vsharp.base_address),
                                         unsigned(desc.is_written), unsigned(vsharp.GetStride()),
                                         unsigned(vsharp.num_records));
                        }
                    }
                    if ((observed && matched <= 256) || candidate <= 8 ||
                        (candidate & (candidate - 1)) == 0) {
                        std::fprintf(stderr,
                                     "FB_IDENTITY_BIND candidate=%llu match=%u matches=%llu "
                                     "hash=0x%llx descriptor=%u record=0x%llx "
                                     "write=%u stride=%u records=%u "
                                     "owner=0x%llx shader_object=0x%llx "
                                     "sequence=%llu reuses=%llu\n",
                                     static_cast<unsigned long long>(candidate),
                                     unsigned(observed.has_value()),
                                     static_cast<unsigned long long>(matched),
                                     static_cast<unsigned long long>(stage.pgm_hash),
                                     unsigned(&desc - stage.buffers.data()),
                                     static_cast<unsigned long long>(vsharp.base_address),
                                     unsigned(desc.is_written), unsigned(vsharp.GetStride()),
                                     unsigned(vsharp.num_records),
                                     static_cast<unsigned long long>(observed ? observed->owner_object : 0),
                                     static_cast<unsigned long long>(observed ? observed->shader_object : 0),
                                     static_cast<unsigned long long>(observed ? observed->sequence : 0),
                                     static_cast<unsigned long long>(observed ? observed->address_reuses : 0));
                    }
                }
                bool known_unwritten = false;
                const bool reuse_check = Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
                const auto [buffer, offset] = [&] {
                    if (!explicit_qid1) {
                        return buffer_cache.ObtainBuffer(
                            vsharp.base_address, size, desc.is_written, desc.is_formatted,
                            reuse_check ? &known_unwritten : nullptr);
                    }
                    const auto& selected = buffer_cache.ActiveQueue();
                    const iss::g4::OwnerContext owner{
                        .owner = selected.index == 1 ? iss::g4::CpOwner::Asc1
                                                     : iss::g4::CpOwner::Graphics,
                        .qid = 1,
                        .generation = active_record_context &&
                                              active_record_context->rasterizer == this
                                          ? active_record_context->generation : 0,
                        .scheduler = &Sched(),
                        .runtime = &Rt(),
                        .stream = selected.stream_buffer,
                        .queue_index = selected.index,
                    };
                    return iss::g4::ObtainForSerialOwner(
                        buffer_cache, owner, vsharp.base_address, size, desc.is_written,
                        desc.is_formatted, reuse_check ? &known_unwritten : nullptr);
                }();
                if(hazard_diag){check.effective_size=size;check.backing_id=reinterpret_cast<u64>(buffer);check.buffer_offset=offset;check.tracked_size=size;}
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                if (desc.is_written) {
                    if (iss::feedback_join::enabled && size == 48) {
                        iss::feedback_join::Binding(
                            vsharp.base_address, stage.pgm_hash, u32(stage.hw_stage),
                            DrawQueue(), buffer_cache.OnComputeQueue() ? 1 : 0,
                            buffer_cache.MappingGeneration(), Sched());
                    }
                    iss::diagnostics::RecordFeedbackBinding(vsharp.base_address, size,
                                                           stage.pgm_hash, Sched().CurrentTick());
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                    if (!skipped_writes.empty()) {
                        ClearSkippedWrites(vsharp.base_address, size);
                    }
                }
                // The stream buffer is only written by the CPU before submission, its reads
                // never need a barrier. Skipping it avoids tracking most bindings of a draw.
                if (buffer != &buffer_cache.GetStreamBuffer()) {
                    bound_buffers.emplace_back(buffer, offset, size, desc.is_written,
                                               vsharp.base_address);
                    if (known_unwritten) Common::Perf::Add(Common::Perf::BufferChecksReused);
                    else {const bool hit=Rt().IsBufferAccessed(buffer,offset,size,desc.is_written);needs_barrier|=hit;if(hazard_diag){check.check_performed=true;check.hit=hit;}}
                }
            }
        }

        if (native_access) {
            native_access->hazard_after = runtime.HazardSeq();
            native_access->bound.assign(bound_buffers.begin() + native_bound_start, bound_buffers.end());
        }
        if(hazard_diag){check.needs_after=needs_barrier;iss::hazard_probe::Buffer(check);}
        auto& set_write = set_writes[set_write_index++];
        WriteBufferDescriptor(set_write, binding.unified++, &buffer_infos.back());
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::BindTex};
    auto& image_bindings = ComputeBindingsActive() ? compute_bindings.image_bindings
                                                   : this->image_bindings;
    auto& image_sources = ComputeBindingsActive() ? compute_bindings.image_sources
                                                  : this->image_sources;
    auto& image_infos = ComputeBindingsActive() ? compute_bindings.image_infos : this->image_infos;
    auto& bound_images = ComputeBindingsActive() ? compute_bindings.bound_images
                                                 : this->bound_images;
    auto& set_writes = ComputeBindingsActive() ? compute_bindings.set_writes : this->set_writes;
    auto& set_write_index = ComputeBindingsActive() ? compute_bindings.set_write_index
                                                    : this->set_write_index;
    auto& tex_memos = ComputeBindingsActive() ? compute_bindings.tex_memos : this->tex_memos;
    auto& sampler_memos = ComputeBindingsActive() ? compute_bindings.sampler_memos
                                                  : this->sampler_memos;
    auto& needs_barrier = ComputeBindingsActive() ? compute_bindings.needs_barrier
                                                   : this->needs_barrier;
    iss::diagnostics::Scope phase_scope(iss::diagnostics::Textures);
    image_bindings.clear();
    image_sources.clear();
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    // Texture binding memo (Tuning::tex_memo): the image and view found for a descriptor,
    // reused while the texture cache does not change (no image created, deleted or replaced)
    // and the image was not written: no search, no lock, no refresh check.
    const bool use_memo = Common::Perf::GetTuning().tex_memo.load(std::memory_order_relaxed);
    if (use_memo && !tex_memos) {
        tex_memos = std::make_unique<std::array<TexMemo, NumTexMemos>>();
    }
    const u64 cache_generation = use_memo ? texture_cache.Generation() : 0;

    // bind_prefetch: the memo keys of the first binding of the image resources, known before
    // the loop. Those of a stage taken from the scout are the scout's (computed from the same
    // flattened user data, their memo entries prefetched a draw ahead); otherwise they are
    // hashed first, so that the memo lookups of the stage start together.
    std::array<u64, AmdGpu::ScoutStage::MaxImageKeys> stage_keys;
    const u64* known_keys = nullptr;
    u32 num_known_keys = 0;
    bool keys_from_scout = false;
    if (use_memo && Common::Perf::GetTuning().bind_prefetch.load(std::memory_order_relaxed)) {
        if (const auto* scouted = (OnDrawWorker() ? nullptr : pipeline_cache.TakenScoutStage(stage));
            scouted != nullptr && scouted->num_image_keys != 0) {
            known_keys = scouted->image_keys.data();
            num_known_keys = scouted->num_image_keys;
            keys_from_scout = true;
        } else if (stage.images.size() > 1) {
            num_known_keys = std::min<u32>(static_cast<u32>(stage.images.size()),
                                           static_cast<u32>(stage_keys.size()));
            for (u32 k = 0; k < num_known_keys; ++k) {
                const auto& desc = stage.images[k];
                stage_keys[k] = TexMemoKey(desc.GetSharp(stage), 0, desc);
                PrefetchTexMemo(stage_keys[k]);
            }
            known_keys = stage_keys.data();
        }
    }

    u32 image_desc_index = 0;
    if (native_access_audit) {
        // Production makes this whole-section decision after all buffer preparation.
        // Capture it here, before the reference can refresh or transition any image.
        for (u32 i = 0; i < native_access_audit->images.size(); ++i) {
            auto& access = native_access_audit->images[i];
            const auto& token = native_access_audit->record->image_tokens[i];
            access.cached = token && texture_cache.ValidatePreparedTexture(*token);
            if (token && !access.cached) native_access_audit->images_cached = false;
        }
    }
    for (const auto& image_desc : stage.images) {
        const u32 desc_index = image_desc_index++;
        if (native_access_audit)
            native_access_audit->images[desc_index].hazard_before = runtime.HazardSeq();
        const auto tsharp = image_desc.GetSharp(stage);
        if (iss::repeat_census::enabled) [[unlikely]]
            iss::repeat_census::Note(iss::repeat_census::Texture, tsharp.Address(),
                                     TexMemoKey(tsharp, 0, image_desc));
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_sources.push_back(ImageSource{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_sources.push_back(ImageSource{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            TexMemo* memo = nullptr;
            u64 memo_key = 0;
            if (use_memo) {
                // What ImageDesc(tsharp, image_desc) depends on (TexMemoKey).
                if (i == 0 && desc_index < num_known_keys) {
                    memo_key = known_keys[desc_index];
                    if (keys_from_scout) {
                        Common::Perf::Add(Common::Perf::ScoutTexKeys);
                    }
                } else {
                    memo_key = TexMemoKey(tsharp, static_cast<u32>(i), image_desc);
                }
                memo = &(*tex_memos)[(memo_key * 0x9E3779B97F4A7C15ULL) >> (64 - NumTexMemoBits)];
                if (memo->key == memo_key && memo->generation == cache_generation &&
                    memo->image_id &&
                    memo->image_generation == texture_cache.ImageGeneration(memo->image_id)) {
                    Common::Perf::Add(Common::Perf::TexMemoHits);
                    auto& [image_id, desc] = image_bindings.emplace_back(
                        std::piecewise_construct, std::tuple{memo->image_id}, std::tuple{});
                    desc.view_info = memo->view_info;
                    desc.type = memo->type;
                    image_sources.push_back(ImageSource{
                        .image_desc = &image_desc,
                        .tsharp = tsharp,
                        .index = static_cast<u32>(i),
                        .fresh = true,
                        .view_backing = memo->view_backing,
                        .view_id = memo->view_id,
                    });
                    auto& image = texture_cache.GetImage(memo->image_id);
                    image.tick_accessed_last = scheduler.CurrentTick();
                    if (image.binding.is_bound) {
                        image.binding.force_general |= image_desc.is_written;
                    }
                    image.binding.is_bound = 1u;
                    continue;
                }
            }
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});
            image_sources.push_back(ImageSource{
                .image_desc = &image_desc,
                .tsharp = tsharp,
                .index = static_cast<u32>(i),
                .fresh = false,
            });

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
            if (memo) {
                // The memo entry is completed in the second pass (view found): the cache
                // generation read after the search, an image it created counts as seen.
                Common::Perf::Add(Common::Perf::TexMemoMisses);
                memo->key = memo_key;
                memo->generation = 0;
                memo->image_id = image_id;
                image_sources.back().memo = memo;
            }
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (size_t binding_idx = 0; binding_idx < image_bindings.size(); ++binding_idx) {
        auto& [image_id, desc] = image_bindings[binding_idx];
        auto& source = image_sources[binding_idx];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            if (native_image_capture) native_image_capture->push_back(NativeImageCapture{});
        } else {
            bool fresh = source.fresh && !is_storage;
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                if (source.fresh) {
                    // The memo gave the view only: the full descriptor for the search.
                    desc = VideoCore::TextureCache::ImageDesc{source.tsharp, *source.image_desc};
                    if (source.image_desc->mip_fallback_mode ==
                        Shader::MipStorageFallbackMode::ConstantIndex) {
                        desc.view_info.range.base.level += source.image_desc->constant_mip_index;
                        desc.view_info.range.extent.levels = 1;
                    } else if (source.image_desc->mip_fallback_mode ==
                               Shader::MipStorageFallbackMode::DynamicIndex) {
                        desc.view_info.range.base.level += source.index;
                        desc.view_info.range.extent.levels = 1;
                    }
                    source.memo = nullptr;
                }
                image_id = texture_cache.FindImage(desc);
                fresh = false;
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            // A memoized texture is tracked and up to date: its view is enough (FindTexture
            // would lock the cache to check that again), and the memo remembers it.
            VideoCore::ImageView* image_view = nullptr;
            if (fresh) {
                if (source.view_backing == image.backing && source.view_id) {
                    image_view = &texture_cache.GetImageView(source.view_id);
                } else {
                    image_view = &image.FindView(desc.view_info);
                }
            } else {
                image_view = &texture_cache.FindTexture(image_id, desc);
                if (source.memo != nullptr && !is_storage) {
                    const auto& views = image.backing->image_view_ids;
                    const auto& infos = image.backing->image_view_infos;
                    const auto it = std::ranges::find(infos, desc.view_info);
                    *source.memo = TexMemo{
                        .key = source.memo->key,
                        .generation = texture_cache.Generation(),
                        .image_generation = texture_cache.ImageGeneration(image_id),
                        .image_id = image_id,
                        .view_id = it != infos.end()
                                       ? views[std::distance(infos.begin(), it)]
                                       : VideoCore::ImageViewId{},
                        .view_backing = image.backing,
                        .view_info = desc.view_info,
                        .type = desc.type,
                    };
                }
            }
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    const bool transition = runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                    needs_barrier |= transition;
                    if (native_access_audit)
                        native_access_audit->images[binding_idx].transition = transition;
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view->image_view,
                                     image.backing->state.layout);
            if (native_image_capture) {
                native_image_capture->push_back(NativeImageCapture{
                    .id = image_id,
                    .uid = image.image_uid,
                    .backing = image.backing,
                    .view = *image_view->image_view,
                    .layout = image.backing->state.layout,
                    .range = desc.view_info.range,
                    .generation = texture_cache.ImageGeneration(image_id),
                    .plain = !is_storage && !binding.force_general && !binding.is_target,
                });
            }
        }
        if (native_access_audit)
            native_access_audit->images[binding_idx].hazard_after = runtime.HazardSeq();
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        WriteImageDescriptor(set_write, binding.unified, array_size,
                             is_storage ? vk::DescriptorType::eStorageImage
                                        : vk::DescriptorType::eSampledImage,
                             &image_infos[image_info_idx]);

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                DrawRegs().ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        DrawRegs().ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        // Sampler memo: the handle found for an S# (and border color base) while no sampler
        // was deleted, without the cache lock.
        vk::Sampler vk_sampler{};
        SamplerMemo* sampler_memo = nullptr;
        u64 sampler_key = 0;
        if (use_memo) {
            struct SamplerKeyData {
                AmdGpu::Sampler ssharp;
                u64 border_color_base;
                u32 is_depth;
                u32 padding;
            } key_data{
                .ssharp = ssharp,
                .border_color_base = DrawRegs().ta_bc_base.Address(),
                .is_depth = sampler.is_depth ? 1u : 0u,
                .padding = 0,
            };
            static_assert(sizeof(SamplerKeyData) == sizeof(AmdGpu::Sampler) + 16);
            sampler_key = XXH3_64bits(&key_data, sizeof(key_data)) | 1;
            sampler_memo = &sampler_memos[(sampler_key * 0x9E3779B97F4A7C15ULL) >> (64 - 10)];
            if (sampler_memo->key == sampler_key &&
                sampler_memo->generation == texture_cache.SamplersGeneration()) {
                vk_sampler = sampler_memo->sampler;
            }
        }
        if (!vk_sampler) {
            vk_sampler =
                texture_cache.GetSampler(ssharp, DrawRegs().ta_bc_base, sampler.is_depth);
            if (sampler_memo != nullptr) {
                *sampler_memo = SamplerMemo{
                    .key = sampler_key,
                    .generation = texture_cache.SamplersGeneration(),
                    .sampler = vk_sampler,
                };
            }
        }
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        WriteImageDescriptor(set_write, binding.unified++, 1, vk::DescriptorType::eSampler,
                             &image_infos.back());
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::BeginRender};
    iss::diagnostics::Scope phase_scope(iss::diagnostics::RenderTargets);
    attachment_feedback_loop = false;
    const auto& regs = DrawRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.ConsumeMetaClear(col_buf.CmaskAddress(), slice);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            texture_cache.ConsumeMetaClear(htile_address, slice) ||
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    BumpBindEpoch();
    const auto& mrt0_hint = DrawCbExtents()[0];
    const auto& mrt1_hint = DrawCbExtents()[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{DrawRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{DrawRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 DrawRegs().color_buffers[0].Address(),
                                 DrawRegs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    BumpBindEpoch();
    auto& regs = DrawRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DrawDbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DrawDbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = DrawRegs().depth_view.slice_start;
    sub_range.extent.layers = DrawRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

// ---- ISS_ASC_NATIVE_QUEUE (see vk_rasterizer.h) -------------------------------------------
namespace {
struct AscQueueState {
    bool unflushed = false;               ///< compute work recorded since the last submission
    std::array<u64, 64> graphics_tick{};  ///< per ring: graphics work its next compute waits for
    std::array<u8, 64> last{};            ///< per ring: where its last work went (1 compute, 2 graphics)
    u64 segments = 0, flushes = 0, releases = 0, graphics_works = 0, compute_waits = 0,
        label_waits = 0, copies = 0;
    std::array<u64, 6> why{};
    std::array<u64, 4> seg_why{}; ///< segments left on the graphics queue: images, uploads, GDS
    u64 last_flush_ns = 0;
};
AscQueueState g_asc;
} // namespace

namespace {
// ISS_ASC_NATIVE_QUEUE: a transfer of a compute ring on the compute queue, ordered with that
// queue's dispatches and transfers.
void AscTransferBarrier(vk::CommandBuffer cmd, bool before) {
    const vk::MemoryBarrier2 barrier = before
        ? vk::MemoryBarrier2{
              .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                              vk::PipelineStageFlagBits2::eTransfer,
              .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead |
                               vk::AccessFlagBits2::eTransferWrite,
              .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
              .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
          }
        : vk::MemoryBarrier2{
              .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
              .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
              .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                              vk::PipelineStageFlagBits2::eTransfer,
              .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                               vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
          };
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}
} // namespace

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    BumpBindEpoch();
    if(is_gds && PostDrawPipeGdsFill(address,num_bytes,value))return;
    DrainDrawPipe(5);
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        ForgetCountedResults(address, num_bytes);
    }
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpNote("fill", fmt::format("{}{:x}+{:x} value {:x} gpu_modified {}", is_gds ? "gds" : "",
                                     address, num_bytes, value,
                                     !is_gds && buffer_cache.HasGpuData(address, num_bytes)));
    }
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.HasGpuData(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            BeforeGpuThreadWrite(address, num_bytes);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            buffer_cache.OnGpuThreadWrite(address, num_bytes);
            return;
        }
    }
    // ISS_ASC_NATIVE_QUEUE: a compute ring's fill runs on the compute queue.
    if (asc_copy_vqid != ~0u) {
        const u32 vqid = std::exchange(asc_copy_vqid, ~0u);
        if (compute_scheduler && AscQueueEnabled()) {
            const u64 uploads_before = Common::Perf::Total(Common::Perf::BufferUploads);
            const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
                if (is_gds) return {buffer_cache.GetGdsBuffer(), address};
                return buffer_cache.ObtainBuffer(address, num_bytes, true);
            }();
            if (Common::Perf::Total(Common::Perf::BufferUploads) != uploads_before) {
                AscBeforeGraphicsWork(vqid, 4);
                Rt().FillBuffer(buffer, offset, num_bytes, value);
                return;
            }
            AscBeforeComputeWork(vqid);
            vk::CommandBuffer cmd = compute_scheduler->CommandBuffer();
            AscTransferBarrier(cmd, true);
            cmd.fillBuffer(buffer->Handle(), offset, num_bytes, value);
            AscTransferBarrier(cmd, false);
            ++g_asc.copies;
            return;
        }
        AscBeforeGraphicsWork(vqid, 5);
    }
    // asc_queue: a fill of one of the guest's compute queues runs on the compute queue.
    const ComputeQueueScope queue_scope{*this, UseComputeQueueForCopy()};
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            buffer_cache.NoteGdsAccess(true);
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    Rt().FillBuffer(buffer, offset, num_bytes, value);
}


void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    if (!dst_gds) {
        ForgetCountedResults(dst, num_bytes);
    }
    if (g_dump.frames_left != 0) [[unlikely]] {
        DumpNote("copy", fmt::format("dst {}{:x} src {}{:x} size {:x} gpu_modified dst {} src {}",
                                     dst_gds ? "gds" : "", dst, src_gds ? "gds" : "", src,
                                     num_bytes,
                                     !dst_gds && buffer_cache.HasGpuData(dst, num_bytes),
                                     !src_gds && buffer_cache.HasGpuData(src, num_bytes)));
    }
    if (g_dump.frames_left != 0 && !src_gds) [[unlikely]] {
        ProbeMemory("copy-src", src, num_bytes);
    }
    if (!dst_gds && !buffer_cache.HasGpuData(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.HasGpuData(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            BeforeGpuThreadWrite(dst, num_bytes);
            // The whole destination invalidated at once, as its write faults would one page at a
            // time (images over it dirty and untracked, tracked pages CPU modified): the game's
            // DMA copies of ~5 MB into its texture pool every ~2 s faulted once per page on this
            // thread (~1240 Windows exceptions, part of the periodic 45 ms frames, 2026-10-17).
            static const bool whole_invalidate = [] {
                const char* p = std::getenv("ISS_CP_COPY_INVALIDATE");
                return !p || p[0] != '0';
            }();
            // ISS_BIG_UPLOAD_LOG=1 (diagnosis): the host copies of 256 KB or more.
            static const bool log_copies = [] {
                const char* p = std::getenv("ISS_BIG_UPLOAD_LOG");
                return p && p[0] == '1';
            }();
            static const u32 log_min = [] {
                const char* p = std::getenv("ISS_BIG_UPLOAD_LOG_MIN");
                return p ? u32(std::strtoul(p, nullptr, 10)) : (256u << 10);
            }();
            const bool log_copy = log_copies && num_bytes >= log_min;
            const u64 copy_t0 = log_copy ? Common::Perf::NowNs() : 0;
            if (whole_invalidate) {
                InvalidateMemory(dst, num_bytes, true);
            }
            const u64 copy_t1 = log_copy ? Common::Perf::NowNs() : 0;
            // Copies of 4 MB or more split over the parallel algorithms' threads
            // (ISS_PARALLEL_UPLOAD_COPY=0 off, as the large uploads).
            static const bool parallel_copy = [] {
                const char* p = std::getenv("ISS_PARALLEL_UPLOAD_COPY");
                return !p || p[0] != '0';
            }();
            if (parallel_copy && num_bytes >= (4u << 20)) {
                constexpr u32 Chunk = 1u << 20;
                std::vector<u32> chunks((num_bytes + Chunk - 1) / Chunk);
                for (u32 i = 0; i < chunks.size(); ++i) chunks[i] = i * Chunk;
                std::for_each(std::execution::par, chunks.begin(), chunks.end(), [&](u32 offset) {
                    std::memcpy(std::bit_cast<u8*>(dst) + offset, std::bit_cast<const u8*>(src) + offset,
                                std::min(Chunk, num_bytes - offset));
                });
            } else
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            const u64 copy_t2 = log_copy ? Common::Perf::NowNs() : 0;
            buffer_cache.OnGpuThreadWrite(dst, num_bytes);
            if (log_copy) [[unlikely]] {
                static std::atomic<u32> logged{0};
                static const u64 log_from = [] {
                    const char* p = std::getenv("ISS_BIG_UPLOAD_LOG_FROM_FRAME");
                    return p ? std::strtoull(p, nullptr, 10) : 0ull;
                }();
                if (Common::Perf::GuestFrameCounter().load() >= log_from &&
                    logged.fetch_add(1, std::memory_order_relaxed) < 20000)
                    std::fprintf(stderr,
                                 "BIG_DMA frame=%llu dst=%llx src=%llx size=%u invalidate_us=%llu "
                                 "copy_us=%llu written_us=%llu\n",
                                 (unsigned long long)Common::Perf::GuestFrameCounter().load(),
                                 (unsigned long long)dst, (unsigned long long)src, num_bytes,
                                 (unsigned long long)((copy_t1 - copy_t0) / 1000),
                                 (unsigned long long)((copy_t2 - copy_t1) / 1000),
                                 (unsigned long long)((Common::Perf::NowNs() - copy_t2) / 1000));
            }
            if (g_dump.frames_left != 0) [[unlikely]] {
                ProbeBuffer("copy-dst-host", nullptr, 0, num_bytes, dst);
            }
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    // ISS_ASC_NATIVE_QUEUE: a compute ring's copy between buffers runs on the compute queue,
    // after that ring's earlier compute work (none from an image or GDS; CPU data uploaded on
    // the graphics queue sends it there).
    if (asc_copy_vqid != ~0u) {
        const u32 vqid = std::exchange(asc_copy_vqid, ~0u);
        if (compute_scheduler && AscQueueEnabled() &&
            (src_gds || !texture_cache.FindImageFromRange(src, num_bytes))) {
            const u64 uploads_before = Common::Perf::Total(Common::Perf::BufferUploads);
            if (!dst_gds && !src_gds)
                VideoCore::RecordReadbackDma(dst, src, num_bytes, scheduler.CurrentTick(), 0);
            // A source only the CPU wrote: its bytes into the compute queue's stream buffer.
            const bool cpu_source = !src_gds && num_bytes <= (1u << 20) &&
                                    !buffer_cache.HasGpuData(src, num_bytes);
            const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
                if (src_gds) return {buffer_cache.GetGdsBuffer(), src};
                if (cpu_source) return {compute_stream.get(), compute_stream->Copy(src, num_bytes, 4)};
                return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
            }();
            const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
                if (dst_gds) return {buffer_cache.GetGdsBuffer(), dst};
                return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
            }();
            const vk::BufferCopy copy = {.srcOffset = src_offset, .dstOffset = dst_offset, .size = num_bytes};
            if (Common::Perf::Total(Common::Perf::BufferUploads) != uploads_before) {
                AscBeforeGraphicsWork(vqid, 4);
                if (cpu_source) {
                    // The copy reads the compute queue's stream buffer: back to a graphics one.
                    const auto [src2, off2] = buffer_cache.ObtainBuffer(src, num_bytes, false, true);
                    const vk::BufferCopy copy2 = {.srcOffset = off2, .dstOffset = dst_offset, .size = num_bytes};
                    Rt().CopyBuffer(src2, dst_buffer, std::span{&copy2, 1});
                } else {
                    Rt().CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
                }
                return;
            }
            AscBeforeComputeWork(vqid);
            vk::CommandBuffer cmd = compute_scheduler->CommandBuffer();
            AscTransferBarrier(cmd, true);
            cmd.copyBuffer(src_buffer->Handle(), dst_buffer->Handle(), copy);
            AscTransferBarrier(cmd, false);
            ++g_asc.copies;
            return;
        }
        AscBeforeGraphicsWork(vqid, 1);
    }
    // asc_queue: a copy of one of the guest's compute queues runs on the compute queue (unless
    // the source may come from an image, tracked for the graphics queue).
    const ComputeQueueScope queue_scope{
        *this, UseComputeQueueForCopy() &&
                   (src_gds || !texture_cache.FindImageFromRange(src, num_bytes))};
    if(!dst_gds && !src_gds)VideoCore::RecordReadbackDma(dst,src,num_bytes,Sched().CurrentTick(),OnComputeRecordQueue()?1:0);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            buffer_cache.NoteGdsAccess(false);
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            buffer_cache.NoteGdsAccess(true);
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    Rt().CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
    if (g_dump.frames_left != 0 && !dst_gds) [[unlikely]] {
        ProbeBuffer("copy-dst", dst_buffer, dst_offset, num_bytes, dst);
    }
}

bool Rasterizer::CopyBufferHostBatch(std::span<const HostCopy> copies) {
    // The game's texture streamer sends ~1500 DMA copies of 4-16 KB in one frame every few
    // seconds (~10 MB into its texture pool): one by one, the invalidation and the per packet
    // work cost the command processor ~6 ms of a 24 ms frame (city at 75 Hz, 2026-10-19).
    static const bool whole_invalidate = [] {
        const char* p = std::getenv("ISS_CP_COPY_INVALIDATE");
        return !p || p[0] != '0';
    }();
    if (!whole_invalidate || g_dump.frames_left != 0) {
        return false;
    }
    for (const auto& copy : copies) {
        if (buffer_cache.HasGpuData(copy.dst, copy.size) ||
            buffer_cache.HasGpuData(copy.src, copy.size) ||
            texture_cache.FindImageFromRange(copy.src, copy.size)) {
            return false;
        }
    }
    // Merged destination ranges; a source inside any destination would depend on the order.
    std::vector<std::pair<VAddr, VAddr>> ranges;
    ranges.reserve(copies.size());
    for (const auto& copy : copies) {
        ranges.emplace_back(copy.dst, copy.dst + copy.size);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            return false; // overlapping destinations: the last write must win
        }
    }
    for (const auto& copy : copies) {
        const auto it = std::upper_bound(ranges.begin(), ranges.end(),
                                         std::pair<VAddr, VAddr>{copy.src + copy.size, 0});
        if (it != ranges.begin() && std::prev(it)->second > copy.src) {
            return false;
        }
    }
    size_t merged = 0;
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first == ranges[merged].second) {
            ranges[merged].second = ranges[i].second;
        } else {
            ranges[++merged] = ranges[i];
        }
    }
    ranges.resize(merged + 1);

    BumpBindEpoch();
    DrainDrawPipe(5);
    for (const auto& copy : copies) {
        ForgetCountedResults(copy.dst, copy.size);
    }
    for (const auto& [begin, end] : ranges) {
        BeforeGpuThreadWrite(begin, end - begin);
        InvalidateMemory(begin, end - begin, true);
    }
    std::for_each(std::execution::par, copies.begin(), copies.end(), [](const HostCopy& copy) {
        std::memcpy(std::bit_cast<void*>(copy.dst), std::bit_cast<const void*>(copy.src),
                    copy.size);
    });
    for (const auto& [begin, end] : ranges) {
        buffer_cache.OnGpuThreadWrite(begin, end - begin);
    }
    static std::atomic<u64> batches{0}, batched{0}, merged_ranges{0};
    batched.fetch_add(copies.size(), std::memory_order_relaxed);
    merged_ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
    if (const u64 n = batches.fetch_add(1, std::memory_order_relaxed) + 1; (n & (n - 1)) == 0) {
        std::fprintf(stderr, "CP_DMA_BATCH batches=%llu copies=%llu ranges=%llu\n",
                     (unsigned long long)n, (unsigned long long)batched.load(),
                     (unsigned long long)merged_ranges.load());
    }
    return true;
}

OcclusionDumpResult Rasterizer::OcclusionDump(VAddr address, u32 num_pairs) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    if (!Common::Perf::GetTuning().real_occlusion.load(std::memory_order_relaxed)) {
        // Real queries just turned off: a query begun counted (nothing written at its begin
        // dump) gets consistent emulated counters at both dumps.
        const bool begun_counted =
            address % 16 == sizeof(u64) && occlusion.IsOpen(address - sizeof(u64));
        if (occlusion.HasOpen()) {
            occlusion.AbandonAll();
        }
        counted_results.clear();
        if (occlusion_diag::enabled && address % 16 == sizeof(u64)) [[unlikely]] {
            occlusion_diag::Note(address - sizeof(u64), occlusion_diag::ROff, num_pairs);
        }
        if (IsDumping()) [[unlikely]] {
            DumpNote("zpass", fmt::format("{:x} emulated (real occlusion off{})", address,
                                          begun_counted ? ", with begin" : ""));
        }
        return begun_counted ? OcclusionDumpResult::EmulatedWithBegin
                             : OcclusionDumpResult::Emulated;
    }
    num_pairs = std::clamp<u32>(num_pairs, 1, OcclusionQueries::MaxCounterPairs);
    const u32 size = num_pairs * 2 * sizeof(u64);
    const auto emulated = [this, address, num_pairs](OcclusionDumpResult result,
                                                     std::string_view why) {
        Common::Perf::Add(Common::Perf::OcclusionEmulated);
        if (occlusion_diag::enabled && address % 16 == sizeof(u64)) [[unlikely]] {
            const u32 reason = why == "unaligned"     ? occlusion_diag::RUnaligned
                               : why == "not counted" ? occlusion_diag::RNotCounted
                               : why == "unmapped"    ? occlusion_diag::RUnmapped
                                                      : occlusion_diag::RTooMany;
            occlusion_diag::Note(address - sizeof(u64), reason, num_pairs);
        }
        if (IsDumping()) [[unlikely]] {
            DumpNote("zpass", fmt::format("{:x} emulated ({}) open {}", address, why,
                                          occlusion.NumOpen()));
        }
        return result;
    };
    if (address % 16 == 0) {
        // Begin dump: nothing is written, the GPU writes the whole query at the end dump.
        ForgetCountedResults(address, size);
        if (!IsMapped(address, size)) {
            return emulated(OcclusionDumpResult::Emulated, "unmapped");
        }
        if (!occlusion.Begin(address, num_pairs)) {
            return emulated(OcclusionDumpResult::Emulated, "too many open");
        }
        if (IsDumping()) [[unlikely]] {
            DumpNote("zpass", fmt::format("{:x} begin open {}", address, occlusion.NumOpen()));
        }
        return OcclusionDumpResult::Counted;
    }
    if (address % 16 != sizeof(u64)) {
        return emulated(OcclusionDumpResult::Emulated, "unaligned");
    }
    // End dump. The begin dump of the query was not written: when the GPU did not count the
    // query, both are emulated, so that the query sees all its samples visible.
    const VAddr results = address - sizeof(u64);
    ForgetCountedResults(results, size);
    boost::container::static_vector<u32, OcclusionQueries::MaxCounterPairs> queries;
    if (!occlusion.End(results, queries)) {
        // The other open queries go on counting.
        occlusion.Resume();
        return emulated(OcclusionDumpResult::EmulatedWithBegin, "not counted");
    }
    texture_cache.InvalidateMemoryFromGPU(results, size);
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(results, size, true);
    const std::span<const u32> used{queries.data(), queries.size()};
    runtime.WriteOcclusionResults(buffer, offset, num_pairs, occlusion.Pool(), used);
    occlusion.MarkUsed(used);
    if (occlusion_diag::enabled) [[unlikely]] {
        occlusion_diag::NoteCounted(results, num_pairs, used, scheduler.CurrentTick());
    }
    if (counted_results.size() == MaxCountedResults) {
        counted_results.erase(counted_results.begin());
    }
    counted_results.push_back(CountedResults{
        .address = results,
        .num_pairs = num_pairs,
        .num_counts = std::min<u32>(static_cast<u32>(queries.size()), num_pairs),
    });
    if (IsDumping()) [[unlikely]] {
        // Frame x-ray: the samples counted, read once the GPU is done (slow, dumps only).
        scheduler.Finish();
        u64 samples = 0;
        for (const u32 query : queries) {
            u64 value = 0;
            const auto result = instance.GetDevice().getQueryPoolResults(
                occlusion.Pool(), query, 1, sizeof(value), &value, sizeof(value),
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            if (result == vk::Result::eSuccess) {
                samples += value;
            }
        }
        DumpNote("zpass", fmt::format("{:x} end counted segments {} samples {} open {}", address,
                                      queries.size(), samples, occlusion.NumOpen()));
    }
    // Writing the results ended the rendering scope: the other open queries go on counting from
    // the next one.
    occlusion.Resume();
    Common::Perf::Add(Common::Perf::OcclusionQueries);
    Common::Perf::Add(Common::Perf::OcclusionSegments, queries.size());
    return OcclusionDumpResult::Counted;
}

bool Rasterizer::IsCountedOcclusionResult(VAddr address, std::optional<u32>& value) const {
    // Written by Runtime::WriteOcclusionResults: every counter is valid (bit 63), begin counters
    // are 0 and the low half of an end counter holds the samples of its segment, if any.
    static constexpr u32 ValidBit = 0x80000000U;
    value.reset();
    for (auto it = counted_results.rbegin(); it != counted_results.rend(); ++it) {
        if (address < it->address || address >= it->address + it->num_pairs * 2 * sizeof(u64)) {
            continue;
        }
        const u64 offset = address - it->address;
        const u32 pair = static_cast<u32>(offset / (2 * sizeof(u64)));
        switch ((offset % (2 * sizeof(u64))) / sizeof(u32)) {
        case 0: // Begin counter, low half.
            value = 0U;
            break;
        case 2: // End counter, low half: the samples counted, unknown until the GPU is done.
            if (pair >= it->num_counts) {
                value = 0U;
            }
            break;
        default: // High halves.
            value = ValidBit;
            break;
        }
        return true;
    }
    return false;
}

void Rasterizer::ForgetCountedResults(VAddr address, u64 size) {
    if (counted_results.empty()) [[likely]] {
        return;
    }
    const auto overlaps = [address, size](const CountedResults& r) {
        return address < r.address + r.num_pairs * 2 * sizeof(u64) && r.address < address + size;
    };
    counted_results.erase(std::remove_if(counted_results.begin(), counted_results.end(), overlaps),
                          counted_results.end());
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    DrainDrawPipe(5);
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

static bool RunPausedGuestMemoryAction(AmdGpu::Liverpool* liverpool,
                                       const std::function<bool()>& action) {
    using State = iss::g4::GraphicsControlMailbox::Control::State;
    auto result = std::make_shared<std::atomic<u8>>(0);
    for (;;) {
        // The C seal and owner-wide pause are admitted atomically. A separate
        // seal followed by a pause would let C record a newer open tick.
        auto control = liverpool->EnqueueSealingPausedGuestGraphicsControl(
            [action, result] {
                result->store(action() ? 2 : 1, std::memory_order_release);
            });
        if (!control) {
            ASSERT_MSG(liverpool->CpSplitActive(),
                       "G4 guest memory control was cancelled during shutdown");
            std::this_thread::yield();
            continue;
        }
        while (control->Status() == State::Queued ||
               control->Status() == State::Running) {
            std::this_thread::yield();
        }
        ASSERT_MSG(control->Status() == State::Done,
                   "G4 guest memory control failed before readback completed");
        const u8 completed = result->load(std::memory_order_acquire);
        ASSERT_MSG(completed == 1 || completed == 2,
                   "G4 guest memory control returned without a result");
        return completed == 2;
    }
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    BumpBindEpoch();
    if (cp_split_enabled && liverpool->CpSplitActive()) {
        if (!assume_locks) {
            return RunPausedGuestMemoryAction(
                liverpool, [this, addr, size] { return InvalidateMemory(addr, size, true); });
        }
        ASSERT_MSG(liverpool->CurrentQueue() == 0 &&
                       liverpool->AscOwnerPausedForGraphics(),
                   "G4 GPU worker write fault lacks a paused graphics readback path");
    }
    DrainDrawPipe(5);
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::BeforeGpuThreadWrite(VAddr addr, u64 size) {
    BumpBindEpoch();
    if (!size) return;
    page_manager.InvalidateSrtWriteWatches(addr, size);
    buffer_cache.InvalidatePreparedPacketAccess(addr, size);
}

void Rasterizer::OnGpuThreadWrite(VAddr addr, u64 size) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    page_manager.InvalidateSrtWriteWatches(addr, size);
    buffer_cache.OnGpuThreadWrite(addr, size);
}

bool Rasterizer::DeferUnifiedFence(VAddr dst, u32 bytes, std::optional<u32> value, bool irq,
                                   Common::UniqueFunction<void>&& write) {
    // ISS_UNIFIED_FENCES=early (experiment): labels written when the command processor reads
    // them, as without unified memory; the guest may then reuse heap memory the GPU still reads.
    static const bool early_fences = [] {
        const char* p = std::getenv("ISS_UNIFIED_FENCES");
        return p && std::string_view{p} == "early";
    }();
    if (!iss::unified::enabled || early_fences ||
        !buffer_cache.IsUnified(iss::unified::RangeBase(), sizeof(u32)) || draw_pipe_state) {
        return false;
    }
    if (bytes != 0 && !buffer_cache.IsUnified(dst, bytes)) {
        iss::unified::Add(iss::unified::FencesOutside);
        return false;
    }
    // The guest reuses the heap's memory once it sees the label: the label is written when the
    // GPU has finished the work before it, made visible to the host by the barrier.
    DrainDrawPipe(5);
    unified_barrier_pending = true;
    const u64 tick = scheduler.CurrentTick();
    if (iss::timeline::enabled) {
        // fence_defer / fence_write: from the command processor reading the fence to the label
        // reaching guest memory.
        const u64 defer_ns = Common::Perf::NowNs();
        iss::timeline::Emit("fence_defer", reinterpret_cast<u64>(this), tick, dst, value ? *value : 0, irq);
        unified_ops.emplace_back(tick, [w = std::move(write), dst, tick, defer_ns, this]() mutable {
            w();
            iss::timeline::Emit("fence_write", reinterpret_cast<u64>(this), tick, dst, 0, 0, 0,
                                Common::Perf::NowNs() - defer_ns);
        });
    } else {
        unified_ops.emplace_back(tick, std::move(write));
    }
    if (bytes != 0 && value) {
        std::erase_if(unified_labels, [&](const UnifiedLabel& label) {
            return label.address == dst ||
                   (label.compute ? compute_scheduler && compute_scheduler->IsFree(label.tick)
                                  : scheduler.IsFree(label.tick));
        });
        if (unified_labels.size() < 256) {
            unified_labels.push_back({dst, *value, tick});
        }
    }
    static const u64 flush_interval_ns = [] {
        const char* p = std::getenv("ISS_UNIFIED_FLUSH_US");
        return (p ? std::strtoull(p, nullptr, 10) : 1000ull) * 1000ull;
    }();
    static const bool flush_irq = [] {
        const char* p = std::getenv("ISS_UNIFIED_FLUSH_IRQ");
        return !p || p[0] != '0';
    }();
    const u64 now = Common::Perf::NowNs();
    if ((irq && flush_irq) || now - unified_last_flush_ns >= flush_interval_ns) {
        unified_last_flush_ns = now;
        Flush();
        iss::unified::Add(iss::unified::FencesFlushed);
    } else if (unified_unsubmitted_tick != tick) {
        // FlushUnifiedFences submits it: before the command processor idles, or once it waited
        // the flush interval.
        unified_unsubmitted_tick = tick;
        unified_unsubmitted_ns = now;
    }
    iss::unified::Add(iss::unified::FencesDeferred);
    return true;
}

void Rasterizer::FlushUnifiedFences(bool idle) {
    if (unified_unsubmitted_tick == 0) {
        return;
    }
    if (unified_unsubmitted_tick != scheduler.CurrentTick()) {
        unified_unsubmitted_tick = 0; // submitted with its command buffer meanwhile
        return;
    }
    static const u64 flush_interval_ns = [] {
        const char* p = std::getenv("ISS_UNIFIED_FLUSH_US");
        return (p ? std::strtoull(p, nullptr, 10) : 1000ull) * 1000ull;
    }();
    const u64 now = Common::Perf::NowNs();
    if (!idle && now - unified_unsubmitted_ns < flush_interval_ns) {
        return;
    }
    unified_unsubmitted_tick = 0;
    unified_last_flush_ns = now;
    Flush();
    iss::unified::Add(iss::unified::FencesFlushed);
}


bool Rasterizer::AscQueueEnabled() const noexcept {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_ASC_NATIVE_QUEUE");
        return p && p[0] == '1';
    }();
    return enabled && instance.HasComputeQueue() && iss::asc_family::family != ~0u &&
           !cp_split_enabled && !draw_pipe_state;
}

void Rasterizer::AscFlush() {
    if (!g_asc.unflushed) return;
    g_asc.unflushed = false;
    g_asc.last_flush_ns = Common::Perf::NowNs();
    const u64 tick = compute_scheduler->CurrentTick();
    compute_scheduler->Flush();
    // The graphics session being recorded (whose tick the unified heap writes of the compute work
    // were stamped with) and every later one run after this compute work.
    scheduler.AddCrossWait(compute_scheduler->GetWorkSemaphore()->Handle(), tick);
    if ((++g_asc.flushes & 8191) == 1)
        std::fprintf(stderr,
                     "ASC_QUEUE segments=%llu copies=%llu flushes=%llu releases=%llu "
                     "graphics_works=%llu[seg=%llu dma=%llu dispatch=%llu patched=%llu upload=%llu "
                     "fill=%llu] seg[images=%llu uploads=%llu gds=%llu] compute_waits=%llu "
                     "label_waits=%llu\n",
                     (unsigned long long)g_asc.segments, (unsigned long long)g_asc.copies,
                     (unsigned long long)g_asc.flushes,
                     (unsigned long long)g_asc.releases, (unsigned long long)g_asc.graphics_works,
                     (unsigned long long)g_asc.why[0], (unsigned long long)g_asc.why[1],
                     (unsigned long long)g_asc.why[2], (unsigned long long)g_asc.why[3],
                     (unsigned long long)g_asc.why[4], (unsigned long long)g_asc.why[5],
                     (unsigned long long)g_asc.seg_why[0], (unsigned long long)g_asc.seg_why[1],
                     (unsigned long long)g_asc.seg_why[2],
                     (unsigned long long)g_asc.compute_waits, (unsigned long long)g_asc.label_waits);
}

void Rasterizer::AscBeforeGraphicsWork(u32 vqid, u32 why) {
    if (!compute_scheduler || !AscQueueEnabled()) return;
    // The graphics session this work goes into waits, when submitted, for every compute work
    // recorded before its submission (the before-submit callback, AscFlush).
    if (vqid < g_asc.last.size()) {
        g_asc.last[vqid] = 2;
        g_asc.graphics_tick[vqid] = scheduler.CurrentTick();
    }
    ++g_asc.graphics_works;
    ++g_asc.why[std::min<u32>(why, 5)];
}

void Rasterizer::AscSegmentOnGraphics(u32 why) {
    ++g_asc.seg_why[std::min<u32>(why, 3)];
}

void Rasterizer::AscBeforeComputeWork(u32 vqid) {
    if (vqid < g_asc.graphics_tick.size()) {
        const u64 t = g_asc.graphics_tick[vqid];
        if (t && !scheduler.IsFree(t)) {
            if (t >= scheduler.CurrentTick()) scheduler.Flush();
            // A timeline wait for a graphics session the recording thread submits soon (wait
            // before signal is allowed): the command processor goes on.
            compute_scheduler->AddCrossWait(scheduler.GetWorkSemaphore()->Handle(), t);
            ++g_asc.compute_waits;
        }
        g_asc.graphics_tick[vqid] = 0;
        g_asc.last[vqid] = 1;
    }
    g_asc.unflushed = true;
    ++g_asc.segments;
}

bool Rasterizer::AscRelease(u32 vqid, VAddr dst, u32 bytes, std::optional<u32> value, bool irq,
                            Common::UniqueFunction<void>&& write) {
    if (!compute_scheduler || !AscQueueEnabled() || vqid >= g_asc.last.size() ||
        g_asc.last[vqid] != 1)
        return false;
    // The compute queue's writes to the unified heap reach the host before the label.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    compute_scheduler->CommandBuffer().pipelineBarrier2(
        vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    const u64 tick = compute_scheduler->CurrentTick();
    compute_scheduler->DeferPriorityOperation(std::move(write));
    if (bytes != 0 && value) {
        std::erase_if(unified_labels, [&](const UnifiedLabel& label) {
            return label.address == dst ||
                   (label.compute ? compute_scheduler->IsFree(label.tick)
                                  : scheduler.IsFree(label.tick));
        });
        if (unified_labels.size() < 256) unified_labels.push_back({dst, *value, tick, true});
    }
    g_asc.unflushed = true;
    static const u64 flush_interval_ns = [] {
        const char* p = std::getenv("ISS_UNIFIED_FLUSH_US");
        return (p ? std::strtoull(p, nullptr, 10) : 1000ull) * 1000ull;
    }();
    if (irq || Common::Perf::NowNs() - g_asc.last_flush_ns >= flush_interval_ns) AscFlush();
    ++g_asc.releases;
    return true;
}

void Rasterizer::AscWaitedLabel(u32 vqid, VAddr addr) {
    if (!compute_scheduler || !AscQueueEnabled() || vqid >= g_asc.graphics_tick.size()) return;
    for (auto it = unified_labels.rbegin(); it != unified_labels.rend(); ++it) {
        if (it->address != addr) continue;
        if (!it->compute && !scheduler.IsFree(it->tick)) {
            g_asc.graphics_tick[vqid] = std::max(g_asc.graphics_tick[vqid], it->tick);
            ++g_asc.label_waits;
        }
        return;
    }
}

bool Rasterizer::PendingUnifiedLabel(VAddr addr, u32& value) {
    for (auto it = unified_labels.rbegin(); it != unified_labels.rend(); ++it) {
        if (it->address == addr) {
            if (it->compute ? compute_scheduler && compute_scheduler->IsFree(it->tick)
                            : scheduler.IsFree(it->tick)) {
                return false; // written to guest memory already
            }
            value = it->value;
            return true;
        }
    }
    return false;
}

void Rasterizer::SyncUnifiedRead(VAddr addr, u64 size) {
    if (iss::unified::enabled) [[unlikely]] {
        DrainDrawPipe(5);
        buffer_cache.SyncUnifiedRead(addr, size);
    }
}

void Rasterizer::InvalidateSrtWriteWatches(VAddr addr, u64 size) {
    page_manager.InvalidateSrtWriteWatches(addr, size);
}

void Rasterizer::WithSrtDirectWrite(const std::function<void()>& write) {
    page_manager.WithSrtDirectWrite(write);
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks, u64 guest_rip) {
    BumpBindEpoch();
    if (cp_split_enabled && liverpool->CpSplitActive()) {
        if (!assume_locks) {
            return RunPausedGuestMemoryAction(liverpool, [this, addr, size, guest_rip] {
                return ReadMemory(addr, size, true, guest_rip);
            });
        }
        ASSERT_MSG(liverpool->CurrentQueue() == 0 &&
                       liverpool->AscOwnerPausedForGraphics(),
                   "G4 GPU worker read fault lacks a paused graphics readback path");
    }
    DrainDrawPipe(5);
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, assume_locks, guest_rip);
    if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
        // Frame x-ray: the GPU results the guest reads back (first dwords at the address).
        std::array<u32, 4> values{};
        std::memcpy(values.data(), reinterpret_cast<const void*>(addr & ~VAddr{0xF}),
                    sizeof(values));
        LOG_INFO(Render_Vulkan, "DUMP readback {:x} values {:08x} {:08x} {:08x} {:08x}", addr,
                 values[0], values[1], values[2], values[3]);
    }
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    buffer_cache.NoteMappingChange(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    buffer_cache.NoteMappingChange(addr, size);
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    BumpBindEpoch();
    DrainDrawPipe(5);
    buffer_cache.NoteMappingChange(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::Dynamic};
    iss::diagnostics::Scope phase_scope(iss::diagnostics::DynamicState);
    auto& dynamic_state = scheduler.GetDynamicState();
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_DRAW_STATE");
        return value && value[0] == '1';
    }();
    static const bool verify = [] {
        const char* value = std::getenv("ISS_NATIVE_DRAW_STATE_VERIFY");
        return value && value[0] == '1';
    }();
    auto& native = native_draw_state;
    const bool eligible = enabled && !cp_split_enabled && !native.failed;
    const u64 context_changes = DrawContextChanges();
    const bool reuse = eligible && native.pipeline == pipeline &&
                       native.context_changes == context_changes &&
                       native.indexed == is_indexed &&
                       native.feedback_loop == attachment_feedback_loop;
    const auto convert = [&] {
        UpdateViewportScissorState();
        UpdateDepthStencilState();
        UpdatePrimitiveState(is_indexed);
        UpdateRasterizationState();
        UpdateColorBlendingState(pipeline);
    };
    if (!reuse) {
        convert();
        if (eligible) {
            native.context_changes = context_changes;
            native.pipeline = pipeline;
            native.indexed = is_indexed;
            native.feedback_loop = attachment_feedback_loop;
            ++native.misses;
        }
    } else {
        ++native.hits;
        if (verify) {
            // Recompute from the original inputs and let the existing setters identify a
            // difference. Preserve dirty bits from command-buffer/blit invalidation too.
            auto previous_dirty = dynamic_state.dirty_state;
            std::memset(&dynamic_state.dirty_state, 0, sizeof(dynamic_state.dirty_state));
            convert();
            auto* actual = reinterpret_cast<u8*>(&dynamic_state.dirty_state);
            const auto* previous = reinterpret_cast<const u8*>(&previous_dirty);
            bool differs = false;
            for (size_t i = 0; i < sizeof(dynamic_state.dirty_state); ++i) {
                differs |= actual[i] != 0;
                actual[i] |= previous[i];
            }
            ++native.verified;
            if (differs) {
                ++native.mismatches;
                native.failed = true;
                std::fprintf(stderr, "NATIVE_DRAW_STATE mismatch; full conversion restored\n");
            }
        }
    }
    if (eligible && ((native.hits + native.misses) & 0x3ffff) == 0) {
        std::fprintf(stderr, "NATIVE_DRAW_STATE hits=%llu misses=%llu verified=%llu mismatches=%llu\n",
                     static_cast<unsigned long long>(native.hits),
                     static_cast<unsigned long long>(native.misses),
                     static_cast<unsigned long long>(native.verified),
                     static_cast<unsigned long long>(native.mismatches));
    }
    // Invalidate() keeps the cached values and marks them dirty after a blit or a new command
    // buffer, so Commit remains mandatory even when the guest state conversion was reused.
    dynamic_state.Commit(instance, scheduler.Recorder());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = DrawRegs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            // This limitation can occur on hundreds of draws per frame. Preserve the diagnostic
            // without formatting and enqueueing the same warning on every affected draw.
            static std::atomic_flag warned = ATOMIC_FLAG_INIT;
            if (!warned.test(std::memory_order_relaxed) &&
                !warned.test_and_set(std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                           "op_val; the stencil test will use op_val (reported once)");
            }
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    DrainDrawPipe(5);
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.Recorder();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    DrainDrawPipe(5);
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.Recorder();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    DrainDrawPipe(5);
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.Recorder();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    DrainDrawPipe(5);
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.Recorder();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}

bool Rasterizer::IsGpuCommandProcessorThreadId(u32 thread_id) {
    return liverpool->IsCpThreadId(thread_id);
}
#endif

} // namespace Vulkan

void IssEngineMarker(const u32* payload) {
    Vulkan::EngineMarker(payload);
}
bool IssPassCostEnabled() {
    return Vulkan::PassCostEnabled();
}
void IssPassCostTick(u32 packet_class) {
    Vulkan::PassCostTick(packet_class);
}
void IssEngineFrame() {
    Vulkan::g_iss_frame.fetch_add(1, std::memory_order_relaxed);
}

