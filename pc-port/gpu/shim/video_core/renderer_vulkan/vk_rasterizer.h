// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "g4_queue_record_context.h"
#include "g4_cp_coordination.h"
#include "srt_walker_capture.h"
#include "common/perf_stats.h"
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace iss::native_shadow {
struct Batch;
}
namespace iss::native_gbuffer {
struct Batch;
}
namespace iss::native_passes {
struct Segment;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;
struct DrawPipeState;

/**
 * Guest occlusion queries counted by the GPU. The guest dumps the depth block sample counters
 * (ZPASS_DONE) before the draws it tests, at a 16 byte aligned address, and after them, 8 bytes
 * further; the sum of the differences of the counter pairs is the number of samples that passed
 * the depth test. The counters run all the time on the GPU, so queries may overlap or nest (a
 * query around a whole pass containing smaller queries). Here Vulkan occlusion queries count the
 * samples of consecutive segments of the command stream, shared by all the open guest queries: a
 * segment ends at each dump and at the end of each rendering scope (Vulkan queries begun in a
 * rendering scope must end in it, and only one may be active). A guest query reports the
 * segments between its two dumps, one per counter pair.
 */
class OcclusionQueries final : public RenderingObserver {
public:
    static constexpr u32 MaxCounterPairs = 16;
    static constexpr u32 MaxOpenQueries = 16;

    explicit OcclusionQueries(const Instance& instance, Scheduler& scheduler);
    ~OcclusionQueries() override;

    void OnBeginRendering(const CommandRecorder& cmdbuf) override;
    void OnEndRendering(const CommandRecorder& cmdbuf) override;

    /// Begin dump: counts the samples of the following draws for the query at this address.
    /// Returns false when it cannot be counted (too many open queries).
    bool Begin(VAddr address, u32 num_pairs);

    /// End dump of the query that began at begin_address: stops counting it and fills the Vulkan
    /// queries that counted its samples, oldest first. Returns false when no counted query began
    /// there, or when some of its samples were not counted (too many segments).
    bool End(VAddr begin_address,
             boost::container::static_vector<u32, MaxCounterPairs>& out_queries);

    /// Stops counting all open queries (real queries turned off).
    void AbandonAll();

    /// Starts counting again for the open queries after an end dump, if a rendering scope is
    /// active (otherwise the next one starts counting).
    void Resume();

    /// End of a guest frame: queries open for several frames are forgotten (never ended), so
    /// that they do not keep segments and open slots forever.
    void OnFrameEnd();

    /// The queries are read by the current command buffer: not reused before it completes.
    void MarkUsed(std::span<const u32> queries);

    [[nodiscard]] bool HasOpen() const noexcept {
        return !open.empty();
    }

    /// Whether a counted query begun at this address is open.
    [[nodiscard]] bool IsOpen(VAddr address) const noexcept {
        return std::any_of(open.begin(), open.end(),
                           [address](const OpenQuery& q) { return q.address == address; });
    }

    [[nodiscard]] u32 NumOpen() const noexcept {
        return static_cast<u32>(open.size());
    }

    [[nodiscard]] vk::QueryPool Pool() const noexcept {
        return *pool;
    }

private:
    struct OpenQuery {
        VAddr address;
        u32 max_segments;  ///< Counter pairs of the query: one segment each.
        u32 first_segment; ///< Index in segments of its first segment.
        u32 frame;         ///< Guest frame of its begin dump.
        bool incomplete;   ///< Some of its samples were not counted.
    };

    /// Forgets the segments that no open query needs any more.
    void DropOldSegments();

    void BeginSegment(const CommandRecorder& cmdbuf);
    void EndSegment(const CommandRecorder& cmdbuf);
    /// Ends the active segment: the samples before and after a dump are counted apart.
    void Cut();

    static constexpr u32 NumQueries = 4096;
    static constexpr u32 MaxSegments = 256;

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueQueryPool pool;
    vk::QueryControlFlags control_flags{};
    /// GPU tick of the command buffer that last used each query (reused when it completed).
    std::array<u64, NumQueries> query_ticks{};
    u32 next_query{};
    bool active{}; ///< The last segment is counting in the current rendering scope.
    u32 frame{};   ///< Guest frames ended.
    boost::container::static_vector<OpenQuery, MaxOpenQueries> open;
    /// Vulkan queries of the segments since the oldest open query began, in order.
    boost::container::static_vector<u32, MaxSegments> segments;
};

/// What the command processor writes for a guest occlusion counter dump.
enum class OcclusionDumpResult {
    Emulated,          ///< Emulated counters at the dump address (all samples visible).
    EmulatedWithBegin, ///< Emulated counters at the begin address of the query, then at the end
                       ///< address: the begin dump was left to the GPU, which did not count.
    Counted,           ///< The GPU writes the counters.
};

class Rasterizer {
public:
    struct PreparedResourceBindings {
        struct CopyBatch {
            VideoCore::BufferCache::PreparedGuestBufferBinding source{};
            VideoCore::BufferCache::PreparedGuestBufferBinding destination{};
            std::vector<vk::BufferCopy> copies{};
        };
        struct BufferItem {
            const VideoCore::Buffer* buffer{};
            u64 offset{}, size{};
            VAddr guest_address{};
            bool write{};
        };
        std::vector<BufferItem> buffers{};
        std::vector<VideoCore::BufferCache::PreparedGuestBufferBinding> guest{};
        std::vector<iss::g4::NegativeImageSnapshot> negative_images{};
        std::optional<VideoCore::BufferCache::PreparedGuestBufferBinding> indirect{};
        std::vector<vk::DescriptorBufferInfo> infos{};
        Pipeline::DescriptorWrites writes{};
        Shader::PushData push{};
        u64 max_residency_tick{};
        u64 max_graphics_bda_tick{};
        u64 bda_graphics_tick_snapshot{};
        bool uses_bda{};
        std::vector<CopyBatch> hle_copies{};
    };
    struct PreparedComputeInputs {
        iss::g4::SrtCaptureResult srt{};
        std::vector<iss::g4::PacketResourceAccess> accesses{};
        std::vector<std::pair<VAddr,u32>> upload_ranges{};
        std::optional<PreparedResourceBindings> bindings{};
        AmdGpu::ComputeProgram cs{};
        Shader::PushData push_data{};
        std::vector<u32> shader_code{};
        u64 shader_hash{};
        VAddr shader_guest_base{};
        u64 mapping_generation{};
        u64 native_mapping_epoch{};
        std::optional<iss::g4::IndirectArgsRef> indirect_args{};
    };
    struct AscDispatchInput {
        AmdGpu::ComputeProgram cs{};
        iss::g4::GuestPacketIdentity guest{};
        std::optional<std::pair<VAddr, u32>> indirect{};
    };
    enum class AscDispatchRoute : u8 { Compute, NoOp, RetryNeedUpload, FallbackGraphics, Failed };
    struct AscDispatchPlan {
        AscDispatchRoute route{AscDispatchRoute::Failed};
        iss::g4::GuestPacketIdentity guest{};
        const ComputePipeline* pipeline{};
        std::optional<PreparedComputeInputs> prepared{};
        std::vector<std::pair<VAddr,u32>> upload_ranges{};
        std::optional<iss::g4::FallbackPacket> fallback{};
    };
    [[nodiscard]] AscDispatchPlan PrepareAscDispatch(const AscDispatchInput& input);
    enum class AscCommitResult : u8 { Recorded, Retry, Failed };
    [[nodiscard]] AscCommitResult CommitAscDispatch(AscDispatchPlan& plan);
    enum class GraphicsHostWriteStatus : u8 { Ready, Busy, Failed };
    struct GraphicsHostWriteGuard {
        Rasterizer* owner{};
        u64 lease_id{}, tick{}, generation{}, final_tick{};
        VAddr address{};
        u64 size{};
        bool gpu_packet{}, record_started{}, published{};
        GraphicsHostWriteGuard() = default;
        GraphicsHostWriteGuard(Rasterizer* owner_, u64 lease_id_, u64 tick_,
                               u64 generation_, VAddr address_, u64 size_, bool gpu_packet_=false)
            : owner{owner_}, lease_id{lease_id_}, tick{tick_}, generation{generation_},
              address{address_}, size{size_}, gpu_packet{gpu_packet_} {}
        GraphicsHostWriteGuard(const GraphicsHostWriteGuard&) = delete;
        GraphicsHostWriteGuard& operator=(const GraphicsHostWriteGuard&) = delete;
        GraphicsHostWriteGuard(GraphicsHostWriteGuard&& other) noexcept
            : owner{std::exchange(other.owner, nullptr)}, lease_id{other.lease_id},
              tick{other.tick}, generation{other.generation},
              final_tick{other.final_tick}, address{other.address},
              size{other.size}, gpu_packet{other.gpu_packet},
              record_started{other.record_started}, published{other.published} {}
        GraphicsHostWriteGuard& operator=(GraphicsHostWriteGuard&& other) noexcept;
        ~GraphicsHostWriteGuard();
    };
    struct GraphicsHostWriteAdmission {
        GraphicsHostWriteStatus status{GraphicsHostWriteStatus::Failed};
        GraphicsHostWriteGuard guard{};
    };
    [[nodiscard]] GraphicsHostWriteAdmission TryPrepareGraphicsHostWrite(VAddr address,u64 size);
    [[nodiscard]] bool ResolveGraphicsHostWrite(GraphicsHostWriteGuard& guard);
    void FinishGraphicsHostWrite(GraphicsHostWriteGuard& guard) noexcept;
    [[nodiscard]] GraphicsHostWriteAdmission TryPrepareGraphicsGpuPacket();
    [[nodiscard]] bool ResolveGraphicsGpuPacket(GraphicsHostWriteGuard& guard);
    [[nodiscard]] bool PublishGraphicsGpuPacket(GraphicsHostWriteGuard& guard);
    void FinishGraphicsGpuPacket(GraphicsHostWriteGuard& guard) noexcept;
    struct AscHostWriteGuard {
        Rasterizer* owner{};
        u64 lease_id{}, tick{}, generation{};
        u32 qid{};
        VAddr address{};
        u64 size{};
        AscHostWriteGuard() = default;
        AscHostWriteGuard(Rasterizer* owner_, u64 lease_id_, u64 tick_,
                          u32 qid_, u64 generation_, VAddr address_, u64 size_)
            : owner{owner_}, lease_id{lease_id_}, tick{tick_}, generation{generation_},
              qid{qid_}, address{address_}, size{size_} {}
        AscHostWriteGuard(const AscHostWriteGuard&) = delete;
        AscHostWriteGuard& operator=(const AscHostWriteGuard&) = delete;
        AscHostWriteGuard(AscHostWriteGuard&& other) noexcept
            : owner{std::exchange(other.owner,nullptr)}, lease_id{other.lease_id},
              tick{other.tick}, generation{other.generation}, qid{other.qid},
              address{other.address}, size{other.size} {}
        AscHostWriteGuard& operator=(AscHostWriteGuard&& other) noexcept;
        ~AscHostWriteGuard();
    };
    struct AscHostWriteAdmission {
        GraphicsHostWriteStatus status{GraphicsHostWriteStatus::Failed};
        AscHostWriteGuard guard{};
    };
    [[nodiscard]] AscHostWriteAdmission TryPrepareAscHostWrite(VAddr address,u64 size);
    [[nodiscard]] bool ResolveAscHostWrite(AscHostWriteGuard& guard);
    void FinishAscHostWrite(AscHostWriteGuard& guard) noexcept;
    [[nodiscard]] std::optional<iss::g4::GraphicsRecordingProof> RecordFallbackSnapshot(
        const iss::g4::FallbackPacket& packet,
        const iss::g4::QidTick& sealed_asc_prefix);
    [[nodiscard]] std::optional<iss::g4::GraphicsRecordingProof> RecordSerialImageFallback(
        const iss::g4::FallbackPacket& packet,
        const iss::g4::QidTick& sealed_asc_prefix);

    // Prepare-only read set for a compute packet. It owns every SRT byte and records
    // physical hazards before BindResources or any Vulkan emission. Unsupported resource
    // kinds fail closed; the caller must not record a split packet on nullopt.
    [[nodiscard]] std::optional<PreparedComputeInputs> PrepareComputeInputs(
        const ComputePipeline* pipeline,
        std::optional<iss::g4::IndirectArgsRef> indirect_args = {});
    // Only the ASC→G fallback needs owning shader code. Re-arms the SRT and code
    // read set under one lease; false keeps fallback routing closed before emission.
    [[nodiscard]] bool PrepareFallbackCode(PreparedComputeInputs& prepared,
                                           const ComputePipeline* pipeline);
    [[nodiscard]] std::optional<PreparedComputeInputs> PrepareCopyHleInputs(
        const ComputePipeline* pipeline);
    [[nodiscard]] std::optional<PreparedResourceBindings> PrepareResourceBindings(
        const ComputePipeline* pipeline, const PreparedComputeInputs& inputs);
    [[nodiscard]] bool RecordPreparedCompute(const ComputePipeline* pipeline,
                                             const PreparedComputeInputs& inputs,
                                             PreparedResourceBindings& bindings);
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    std::size_t Draw(bool is_indexed, u32 index_offset = 0,
                     std::span<const u32> following_commands = {}, std::size_t native_packet_budget = 0);
    std::size_t TryNativeIndexedRun(const GraphicsPipeline* pipeline,
                                    std::span<const u32> following_commands, std::size_t packet_budget);
    std::size_t TryNativeCpuRun(const GraphicsPipeline* pipeline,
                                std::span<const u32> following_commands, std::size_t packet_budget);
    std::size_t TryNativeInstancedRun(const GraphicsPipeline* pipeline,
                                      std::span<const u32> following_commands,
                                      std::size_t packet_budget);
    /// ISS_NATIVE_DEPTH_RUN: a run of the instanced depth family as one indirect draw.
    std::size_t TryNativeDepthRun(const GraphicsPipeline* pipeline,
                                  std::span<const u32> following_commands,
                                  std::size_t packet_budget);
    static bool NativeDepthRunEnabled() noexcept;
    const GraphicsPipeline* DepthRunCandidate();
    void VerifyDepthRunPair(const GraphicsPipeline* pipeline, RenderState& render, u32 ssbo_write,
                            u32 ssbo_index, u64 command_offset,
                            const VideoCore::Buffer* vertex_buffer,
                            const VideoCore::Buffer* index_buffer, u32 mode);
    void RecordDepthRunMode(u32 mode, const GraphicsPipeline* pipeline, u32 ssbo_write,
                            u32 ssbo_index, u64 command_offset,
                            const VideoCore::Buffer* vertex_buffer,
                            const VideoCore::Buffer* index_buffer);
    /// ISS_NATIVE_SHADOW (native_shadow_rasterizer.inl): the draws of an engine shadow batch
    /// recorded from the batch. Returns how many of them, in order, were recorded: the command
    /// processor interprets the range and skips those (0 when a render state is still unknown).
    u32 NativeShadowExecute(const iss::native_shadow::Batch& batch);
    /// Interpretation of a range: the pipeline of each draw whose render state is unknown is
    /// learned (the draw takes the full path).
    void NativeShadowLearnBegin(const iss::native_shadow::Batch& batch);
    void NativeShadowBeforeDraw(u32 draw);
    void NativeShadowAfterDraw(u32 draw);
    void NativeShadowLearnEnd();
    std::optional<RenderState> NativeShadowVerifyStart(const RenderState& render);
    RenderState NativeShadowVerifyTargets(const RenderState& render);
    void NativeShadowVerifyNative(const iss::native_shadow::Batch& batch);
    void NativeShadowVerifyFinish(bool complete);
    bool NativeShadowLearning() const { return shadow_learn.batch != nullptr; }
    /// ISS_NATIVE_GBUFFER (native_gbuffer_rasterizer.inl): the draws of an engine G-buffer range
    /// recorded from the range. Returns how many of them, in order, were recorded; begin_draw and
    /// end_draw bracket each draw (the scout entry of its packet).
    u32 NativeGbufferExecute(const iss::native_gbuffer::Batch& batch,
                             const std::function<void(const u32*)>& begin_draw,
                             const std::function<void()>& end_draw);
    void NativeGbufferInterpretedDraw(const iss::native_gbuffer::Batch& batch, bool last);
    void NativeGbufferRangeEnd(const iss::native_gbuffer::Batch* batch);
    void NativeGbufferVerifyFinish(bool complete);
    void NativeGbufferVerifyNative();
    RenderState NativeGbufferVerifyTargets(const RenderState& render);
    /// ISS_NATIVE_MATERIALS (native_material_rasterizer.inl): a texture BindTextures resolved for
    /// a draw whose material is being learned (an empty id: a null descriptor).
    struct NativeImageCapture {
        VideoCore::ImageId id{};
        u64 uid{};
        const void* backing{};
        vk::ImageView view{};
        vk::ImageLayout layout{};
        VideoCore::SubresourceRange range{};
        u32 generation{};
        bool plain{}; ///< sampled only: not a target, not storage
    };
    std::vector<NativeImageCapture>* native_image_capture{};
    struct NativeMaterialTemplate;
    /// The material template of the draw at the current registers (stage user data refreshed),
    /// nullptr when none was learned. state_key: the range context and render state version.
    const NativeMaterialTemplate* NativeMaterialFind(u64 state_key);
    /// Binds a found template's resources for the draw (textures validated and transitioned,
    /// buffers resolved, descriptors pushed). False: the template is stale, take the full path.
    bool NativeMaterialBind(const NativeMaterialTemplate& t);
    void NativeMaterialDrop(u64 key);
    const GraphicsPipeline* NativeMaterialPipeline(const NativeMaterialTemplate& t) const;
    /// Learns the template of the draw the full path just bound (BindResources ran with
    /// native_image_capture set).
    void NativeMaterialLearn(u64 state_key, const GraphicsPipeline* pipeline,
                             const RenderState& render);
    /// ISS_NATIVE_RECORDER (native_recorder.inl): lets the engine threads record ranges once the
    /// command processor has started; executes a range's recording when what it assumed holds
    /// (returns its draw count, 0: take the other paths).
    void NativeRecorderSetup();
    /// ISS_NATIVE_PASSES (native_passes_rasterizer.inl): the engine threads record whole passes
    /// (native_passes.h); the command processor executes a segment at its first draw (false:
    /// interpret it) or learns the template of a draw an engine thread asked for.
    void NativePassesSetup();
    bool NativePassExecute(iss::native_passes::Segment& segment,
                           const std::function<void(const u32*)>& begin_draw,
                           const std::function<void()>& end_draw, u64& tick);
    u64 NativePassLearnKey(u64 engine_key);
    void NativePassLearn(u64 key, const GraphicsPipeline* pipeline, const RenderState& render);
    /// Set by the command processor for the draw of a learn request (its key), 0 otherwise.
    u64 native_pass_learn{};
    /// ISS_NATIVE_INDIRECT: the learn request is at an indexed indirect draw: 1 << 32 | base vertex
    /// location << 16 | start instance location (0: a direct draw).
    u64 native_pass_learn_indirect{};
    /// ISS_NATIVE_COMPUTE (native_compute_rasterizer.inl): a compute segment executed at its
    /// first dispatch packet; a learn request's key for the dispatch (DispatchRecord learns its
    /// template, or the command processor counts the attempt failed).
    /// asc: a segment of an asynchronous compute queue (ISS_NATIVE_ASC; recorded at the DingDong).
    bool NativeComputeExecute(iss::native_passes::Segment& segment, u64& tick, bool asc = false);
    void NativeComputeLearn(u64 key, const ComputePipeline* pipeline,
                            const std::vector<NativeImageCapture>& caps);
    void NativeComputeLearnFailed(u64 key);
    u64 native_compute_learn{};
    std::vector<u32> native_pass_learn_mask;
    bool native_material_allow_clip{}; ///< NativeMaterialLearn: clip planes (native passes upload them)
    /// The end of the command buffer span being processed (set by the command processor before
    /// a range), and the chained ranges the last execution also drew, in order (the command
    /// processor applies the registers up to each and resumes after the last).
    const u32* native_span_end{};
    std::vector<const iss::native_gbuffer::Batch*> native_stretch;
    u32 NativeRecorderExecute(const iss::native_gbuffer::Batch& batch, u64 context,
                              const std::function<void(u32)>& set_registers,
                              const std::function<void(const u32*)>& begin_draw,
                              const std::function<void()>& end_draw);
    struct ShadowLearn {
        const iss::native_shadow::Batch* batch{};
        u64 begin_hash{};
        u64 key{};
        bool learn{};
    } shadow_learn;
    /// True when the packet was queued to the draw pipe worker (the caller skips it).
    bool BeforeDrawPipePacket(u32 opcode,const u32* words,u32 count,int asc_pipe_id=-1);
    void DrainDrawPipe(u32 reason=5);
    bool PrepareDrawPipeFault(bool is_write);
    void DrainDrawPipeForIdle();
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    /// The command processor translates the commands of one of the guest's asynchronous compute
    /// queues (asc_queue: their dispatches, copies and fills run on the compute queue).
    void SetAsyncCompute(bool active) noexcept {
        if (cp_split_enabled) {
            split_thread_state = {this, active};
        } else {
            async_compute_translation = active;
        }
    }

    /// asc_queue: submits the work recorded on the compute queue (at the fences of the compute
    /// queues and the end of their submissions).
    void FlushComputeQueue();
    /// Only called before admitting split work. A false result keeps the serial CP path.
    bool PrepareCpSplit(std::string& reason);
    [[nodiscard]] bool CpSplitEnabled() const noexcept { return cp_split_enabled; }
    /// Called by the owning CP worker at a safe point, after recording through this tick.
    bool SealCpPrefix(iss::g4::CpLane lane, u64 through_tick);
    // C records a GPU timeline wait for a G fallback that was actually sealed.
    [[nodiscard]] bool WaitAscOnGraphicsToken(const iss::g4::QidTick& sealed_g);
    // Called on the C owner thread immediately before it parks for a G control.
    [[nodiscard]] bool AscSchedulerIdleAfterSeal(u64 sealed_tick) const noexcept;
    [[nodiscard]] bool WaitAscGpuComplete(u64 sealed_tick);
    enum class AscArenaControlStatus : u8 { Ready, Busy, Failed };
    struct AscArenaControlResult {
        AscArenaControlStatus status{AscArenaControlStatus::Failed};
        u64 graphics_tick{}, mapping_generation{};
    };
    [[nodiscard]] AscArenaControlResult PrepareAscArenaOnGraphics(
        std::span<const std::pair<VAddr,u32>> ranges);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    struct HostCopy {
        VAddr dst;
        VAddr src;
        u32 size;
    };
    /// Consecutive memory to memory DMA packets copied in host memory together (the invalidation
    /// once per merged destination range, the copies in parallel). False when one of them cannot
    /// take the host path or they overlap: the caller then copies them one by one.
    bool CopyBufferHostBatch(std::span<const HostCopy> copies);
    /// Guest occlusion counter dump (ZPASS_DONE), counted by real occlusion queries when they are
    /// turned on. Otherwise, or when the query cannot be counted, the caller writes emulated
    /// counters, consistent for the whole query (see OcclusionDumpResult).
    OcclusionDumpResult OcclusionDump(VAddr address, u32 num_pairs);
    /// Whether this address (a dword) is in the results of an occlusion query counted by the GPU
    /// and, if so, the value the GPU writes there when it does not depend on the samples counted
    /// (begin counters and valid bits). A command processor wait for it is then satisfied without
    /// waiting for the GPU: the commands that follow it run after the GPU wrote the results anyway.
    [[nodiscard]] bool IsCountedOcclusionResult(VAddr address, std::optional<u32>& value) const;

    /// Frame x-ray (Tuning::dump_frames): whether the commands of this guest frame are logged.
    [[nodiscard]] bool IsDumping() const noexcept;
    /// Logs one event of the dumped frames (guest markers...).
    void DumpNote(std::string_view kind, std::string_view text);
    /// Frame x-ray: records what the GPU reads in `size` bytes (at most 32) of `buffer` at
    /// `offset`, the copy of guest address `address`, to log it with the guest memory content
    /// when the dump ends (null buffer: guest memory only).
    void ProbeBuffer(std::string_view what, const VideoCore::Buffer* buffer, u64 offset, u32 size,
                     VAddr address);
    /// The same for the GPU copy (arena) of guest memory, when resident.
    void ProbeMemory(std::string_view what, VAddr address, u32 size);
    /// Waits for the GPU and logs the probes of the dump.
    void FinishProbes();
    /// Frame x-ray: after a draw or dispatch, the image dumps Tuning::dump_after asks for.
    void DumpAfter(const Pipeline* pipeline);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// The command processor wrote guest memory directly (see BufferCache::OnGpuThreadWrite).
    void BeforeGpuThreadWrite(VAddr addr, u64 size);
    void OnGpuThreadWrite(VAddr addr, u64 size);
    /// ISS_UNIFIED_MEMORY (unified_memory.h): runs a fence's label write and interrupt when the
    /// GPU work recorded before it completes (a label of `bytes` at dst in the unified heap, or
    /// an interrupt only when bytes is 0). False when the fence is not deferred (heap not
    /// imported, or a label outside it): the caller writes it now as before.
    /// The command processor sees the label value at once (PendingUnifiedLabel): its own waits
    /// on it are ordered by the queue. irq: the fence raises an interrupt (submitted at once);
    /// other fences are submitted at the next submission, at most ISS_UNIFIED_FLUSH_US later.
    bool DeferUnifiedFence(VAddr dst, u32 bytes, std::optional<u32> value, bool irq,
                           Common::UniqueFunction<void>&& write);
    /// The value of a deferred label at addr not yet written to guest memory.
    bool PendingUnifiedLabel(VAddr addr, u32& value);
    struct UnifiedLabel {
        VAddr address;
        u32 value;
        u64 tick;
        bool compute{}; ///< ISS_ASC_NATIVE_QUEUE: written when the compute queue reaches tick
    };

    // ---- ISS_ASC_NATIVE_QUEUE=1 (vk_rasterizer.cpp, "AscQueue") ----
    // The asynchronous compute rings' native segments that use the unified heap only run on a
    // queue of a compute-only family (the GPU's asynchronous compute engines), concurrently with
    // the rendering, as on the PS4: the game's thread waits each frame for its particle compute,
    // which on the graphics queue waited behind the rendering submitted before it.
    // Ordering: the graphics work recorded after compute work waits for it (a cross queue wait at
    // each compute submission); compute work waits for graphics work only where the guest's
    // queue waits for a graphics label, or after its own work that ran on the graphics queue.
    [[nodiscard]] bool AscQueueEnabled() const noexcept;
    /// The compute ring (vqid) whose packets the command processor is running.
    void AscSetVqid(u32 vqid) noexcept { asc_vqid = vqid; }
    /// Work of compute ring vqid about to be recorded on the graphics queue: after its compute
    /// work. why: 0 segment, 1 DMA, 2 full path dispatch, 3 patched (indirect) dispatch, 4 upload,
    /// 5 fill (counted apart in the ASC_QUEUE report).
    void AscBeforeGraphicsWork(u32 vqid, u32 why = 0);
    /// The compute ring whose DMA packet runs now (~0u: none): CopyBuffer may record it on the
    /// compute queue.
    u32 asc_copy_vqid{~0u};
    /// A fence of compute ring vqid: deferred on the compute queue when its last work is there.
    bool AscRelease(u32 vqid, VAddr dst, u32 bytes, std::optional<u32> value, bool irq,
                    Common::UniqueFunction<void>&& write);
    /// A compute ring's segment left on the graphics queue (0 images, 1 uploads, 2 GDS).
    void AscSegmentOnGraphics(u32 why);
    /// Compute ring vqid's wait was satisfied by a graphics label not written yet.
    void AscWaitedLabel(u32 vqid, VAddr addr);
    /// Submits the compute work recorded so far; the graphics work recorded from now on waits for it.
    void AscFlush();
    /// Compute work of ring vqid about to be recorded on the compute queue: after the graphics
    /// work it must follow.
    void AscBeforeComputeWork(u32 vqid);
    u32 asc_vqid{};
    std::vector<UnifiedLabel> unified_labels;
    u64 unified_last_flush_ns{};
    bool unified_barrier_pending{};
    std::vector<std::pair<u64, Common::UniqueFunction<void>>> unified_ops; ///< deferred fences
    u64 unified_unsubmitted_tick{}; ///< tick of deferred fences not submitted yet (0 none)
    u64 unified_unsubmitted_ns{};
    /// Submits deferred fences still in the current command buffer: always when the command
    /// processor is about to idle (idle), otherwise once they waited ISS_UNIFIED_FLUSH_US.
    void FlushUnifiedFences(bool idle);
    /// Command processor reads of guest memory: waits for the GPU writes recorded into the
    /// unified heap that cover it.
    void SyncUnifiedRead(VAddr addr, u64 size);
    void InvalidateSrtWriteWatches(VAddr addr, u64 size);
    void WithSrtDirectWrite(const std::function<void()>& write);
    /// guest_rip: the faulting instruction of a guest read (readback_relaxed_sites), 0 if none.
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false, u64 guest_rip = 0);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();
    void RetireReadbacks();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
    bool IsGpuCommandProcessorThreadId(u32 thread_id);
#endif

private:
    void InitializeDrawPipe();
    bool UseDrawPipe() const;
    bool FilterDrawPasses() const;
    bool PostDrawPipe(const Pipeline* pipeline,bool compute,bool indexed,u32 index_offset,
                      const std::array<u64,5>* indirect=nullptr);
    void DrawIndirectRecord(const GraphicsPipeline* pipeline,bool is_indexed,VAddr arg_address,
                            u32 offset,u32 stride,u32 max_count,VAddr count_address);
    bool PostDrawPipeGdsFill(VAddr address,u32 size,u32 value);
    static void RunDrawPipePacket(void* context,const u8* data,u32 size);
    void DrawRecord(const GraphicsPipeline* pipeline,bool indexed,u32 index_offset);
    void DispatchRecord(const ComputePipeline* pipeline);
    AmdGpu::Regs& DrawRegs() const;
    const AmdGpu::ComputeProgram& DrawCsRegs() const;
    const std::array<AmdGpu::CbDbExtent,AmdGpu::NUM_COLOR_BUFFERS>& DrawCbExtents() const;
    const AmdGpu::CbDbExtent& DrawDbExtent() const;
    int DrawQueue() const;
    bool DrawAsync() const;
    u64 DrawContextChanges() const;
    bool OnDrawWorker() const;
    std::unique_ptr<DrawPipeState> draw_pipe_state;
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    /// ISS_DLSS=1 (iss_dlss.h): the game's T2x resolve as DLAA. False = not done (the game's
    /// own resolve draw runs).
    bool DlssResolve();

    void TraceShaderResources(const Shader::Info& stage);
    void TraceBufferWriters(const Shader::Info& stage);
    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline,
                       const Shader::PushData* captured_push_data = nullptr,
                       bool skip_compute_hle = false);

    /// Stage binding memo (ISS_STAGE_MEMO=1): what BindBuffers/BindTextures produced for one
    /// stage of a graphics pipeline, replayed for a later draw of the same pipeline whose stage
    /// reads the same V#/T#/S# while nothing that could change those results happened: same
    /// upload epoch and emulator writes, mappings, arenas, published GPU writes, CPU modified
    /// pages, recorded buffer writes and image transitions, texture and sampler caches, and no
    /// rasterizer event (dispatch, copy, fill, invalidation, readback...). Stream buffer copies
    /// are only replayed within the command buffer they were recorded in.
    struct StageMemoStamp {
        u64 upload_epoch{}, host_write_gen{}, mapping_gen{}, arena_gen{}, gpu_write_seq{},
            cpu_dirty_seq{}, hazard_seq{}, bind_epoch{}, tex_gen{}, sampler_gen{};
        bool operator==(const StageMemoStamp&) const = default;
    };
    struct StageMemo {
        static constexpr u32 MaxBuffers = 16;
        static constexpr u32 MaxImages = 16;
        static constexpr u32 MaxSamplers = 16;
        static constexpr u32 MaxImageInfos = 32;
        static constexpr u32 MaxWrites = 48;
        struct Write {
            u32 binding;
            u32 count;
            vk::DescriptorType type;
            u16 info_index;
            bool is_buffer;
        };
        bool valid{};
        bool uses_stream{};
        const Shader::Info* info{};
        const Pipeline* pipeline{};
        VAddr program_base{};
        u64 tick{};
        Shader::Backend::Bindings binding_in{};
        Shader::Backend::Bindings binding_out{};
        StageMemoStamp stamp{};
        u64 ta_bc_base{};
        u32 clip_enable{};
        u8 num_buffers{};
        u8 num_images{};
        u8 num_samplers{};
        u8 num_image_infos{};
        u8 num_writes{};
        u8 num_bound_buffers{};
        u8 num_bound_images{};
        std::array<u8, MaxBuffers> flatbuf{};
        /// Per buffer: its entry in bound_buffers (0xFF: none) and whether it was a stream
        /// copy; partial_ok when every bound entry belongs to one buffer (partial replays).
        std::array<u8, MaxBuffers> bound_index{};
        u32 stream_mask{};
        bool partial_ok{};
        std::array<AmdGpu::Buffer, MaxBuffers> vsharps{};
        std::array<AmdGpu::Image, MaxImages> tsharps{};
        std::array<AmdGpu::Sampler, MaxSamplers> ssharps{};
        std::array<vk::DescriptorBufferInfo, MaxBuffers> buffer_infos{};
        std::array<u8, MaxBuffers> push_offsets{};
        std::array<vk::DescriptorImageInfo, MaxImageInfos> image_infos{};
        std::array<Write, MaxWrites> writes{};
        struct Bound {
            const VideoCore::Buffer* buffer;
            u64 offset;
            u32 size;
            VAddr address;
        };
        std::array<Bound, MaxBuffers> bound_buffers{};
        std::array<VideoCore::ImageId, MaxImageInfos> bound_images{};
    };
    struct StageMemoCapture {
        bool active{};
        StageMemoStamp stamp{};
        u64 tick{};
        bool barrier_before{};
        Shader::Backend::Bindings binding_in{};
        u32 buffer_infos{}, image_infos{}, writes{}, bound_buffers{}, bound_images{};
    };
    [[nodiscard]] StageMemoStamp CaptureStageMemoStamp() const noexcept;
    [[nodiscard]] bool StageMemoContextEligible() const noexcept;
    [[nodiscard]] static bool StageMemoStageEligible(const Shader::Info& stage) noexcept;
    bool ReplayStageMemo(u32 slot, const Shader::Info& stage, const Pipeline* pipeline,
                         Shader::Backend::Bindings& binding, Shader::PushData& push_data,
                         bool verify_only);
    void RememberStageMemo(u32 slot, const Shader::Info& stage, const Pipeline* pipeline,
                           const StageMemoCapture& capture, const Shader::Backend::Bindings& binding,
                           const Shader::PushData& push_data, bool verify);
    void ReportStageMemo();

    /// Fast draw (ISS_FAST_DRAW=1): a direct draw whose state is the one the previous draw left
    /// in the command buffer, except its geometry (fetch V# list, vertex/instance offsets, index
    /// range). Same context registers, programs, pixel stage user data, vertex stage
    /// descriptors and fetch formats, render targets, command buffer and bound state, and no
    /// event since (StageMemoStamp). Only the vertex and index buffers and the draw are
    /// recorded; pipeline, descriptors, push constants, dynamic state and render pass stay.
    bool TryFastDraw(bool is_indexed, u32 index_offset);
    /// Called by DrawRecord: what a following fast draw replays (before ResetBindings), then
    /// the stamps after it.
    void CaptureFastDraw(const GraphicsPipeline* pipeline, bool is_indexed,
                         const RenderState& state);
    void FinishFastDraw();
    void ReportFastDraw();
    void BumpBindEpoch() noexcept {
        bind_epoch.fetch_add(1, std::memory_order_release);
    }

    /// Tuning::bind_prefetch: prefetches the memo entries the bindings of a draw read first,
    /// from its scout entry (the current draw if not done yet, and the next one), and the
    /// index buffer lookup of the current draw.
    void PrefetchDrawLookups(bool is_indexed, u32 index_offset);
    void PrefetchScoutLookups(const AmdGpu::ScoutDraw& draw);
    void PrefetchTexMemo(u64 key) const noexcept;

    /// Logs a draw or dispatch of the dumped frames: shaders and their buffers and images, render
    /// targets, blending and depth state.
    void DumpCommand(std::string_view kind, const Pipeline* pipeline, std::string_view params);

    /// Work skipped while its pipeline compiles leaves the buffers it would write stale. Indirect
    /// arguments read from them could be garbage, so indirect draws and dispatches reading them
    /// are skipped too, until something writes those buffers.
    void RecordSkippedWrites(const Pipeline& pipeline);
    bool ReadsSkippedWrites(VAddr address, u64 size);
    void ClearSkippedWrites(VAddr address, u64 size);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;
    std::atomic<u64> graphics_host_write_sequence{1};
    std::atomic<u64> asc_host_write_sequence{1};
    std::atomic<u64> graphics_fallback_sequence{1};

    /// Create the ACE scheduler, runtime, and stream on the graphics CP before split admission.
    /// The ordinary serial ASC path also uses this when it first needs the compute queue.
    void EnsureComputeQueueInitialized();
    /// asc_queue: whether the compute queue can run this dispatch (buffers only, push
    /// descriptors).
    bool UseComputeQueue(const ComputePipeline* pipeline);
    /// asc_queue: whether a copy or fill of the compute queue being translated runs there.
    [[nodiscard]] bool UseComputeQueueForCopy() const noexcept {
        return DrawAsync() && compute_queue_routing_enabled &&
               Common::Perf::GetTuning().asc_queue.load(std::memory_order_relaxed);
    }
    using RecordContext = iss::g4::QueueRecordContext<Scheduler, Runtime,
        VideoCore::StreamBuffer, VideoCore::QueueContext>;
    inline static thread_local RecordContext* active_record_context{};
    /// asc_queue: records the dispatch, copy or fill in scope on the compute queue when told
    /// to (the buffer cache binds, uploads and tracks the dependencies for that queue).
    class ComputeQueueScope {
    public:
        ComputeQueueScope(Rasterizer& rasterizer_, bool compute) : rasterizer{rasterizer_} {
            if (rasterizer.cp_split_enabled) {
                split_context = true;
                previous_record_context = active_record_context;
                previous_buffer_queue = const_cast<VideoCore::QueueContext*>(
                    &rasterizer.buffer_cache.ActiveQueue());
                rasterizer.buffer_cache.SetQueue(compute ? &rasterizer.compute_context : nullptr);
                const auto& selected = rasterizer.buffer_cache.ActiveQueue();
                context.emplace(RecordContext{
                    .rasterizer = &rasterizer,
                    .owner = selected.index == 1 ? iss::g4::CpOwner::Asc1
                                                 : iss::g4::CpOwner::Graphics,
                    .qid = static_cast<u32>(rasterizer.DrawQueue()),
                    .generation = rasterizer.CurrentSubmissionGeneration(),
                    .scheduler = selected.scheduler,
                    .runtime = selected.runtime,
                    .stream = selected.stream_buffer,
                    .buffer_queue = const_cast<VideoCore::QueueContext*>(&selected),
                    .physical_compute = selected.index == 1,
                });
                ASSERT_MSG(context->ValidForSplitRecord(), "G4 split record context mismatch");
                active_record_context = &*context;
                if (compute) Common::Perf::Add(Common::Perf::AscQueueDispatches);
                return;
            }
            if (rasterizer.Qid1RecordContextEnabled() && rasterizer.DrawQueue() == 1 &&
                !rasterizer.OnDrawWorker()) {
                explicit_context = true;
                previous_compute = rasterizer.on_compute_queue;
                previous_runtime = rasterizer.active_runtime;
                previous_scheduler = rasterizer.active_scheduler;
                previous_record_context = active_record_context;
                if (compute) {
                    rasterizer.on_compute_queue = true;
                    rasterizer.active_runtime = rasterizer.compute_runtime.get();
                    rasterizer.active_scheduler = rasterizer.compute_scheduler.get();
                    rasterizer.buffer_cache.SetQueue(&rasterizer.compute_context);
                    Common::Perf::Add(Common::Perf::AscQueueDispatches);
                }
                const auto& selected = rasterizer.buffer_cache.ActiveQueue();
                context.emplace(RecordContext{
                    .rasterizer = &rasterizer,
                    .owner = selected.index == 1 ? iss::g4::CpOwner::Asc1
                                                 : iss::g4::CpOwner::Graphics,
                    .qid = 1,
                    .generation = rasterizer.CurrentSubmissionGeneration(),
                    .scheduler = selected.scheduler,
                    .runtime = selected.runtime,
                    .stream = selected.stream_buffer,
                    .buffer_queue = const_cast<VideoCore::QueueContext*>(&selected),
                    .physical_compute = selected.index == 1,
                });
                ASSERT_MSG(context->ValidForSerialRecord(), "G4 qid1 record context mismatch");
                previous_producer_identity = rasterizer.buffer_cache.SetSerialProducerIdentity(
                    context->owner, context->qid, context->generation);
                active_record_context = &*context;
                return;
            }
            if (compute) {
                rasterizer.on_compute_queue = true;
                rasterizer.active_runtime = rasterizer.compute_runtime.get();
                rasterizer.active_scheduler = rasterizer.compute_scheduler.get();
                rasterizer.buffer_cache.SetQueue(&rasterizer.compute_context);
                Common::Perf::Add(Common::Perf::AscQueueDispatches);
            }
        }
        ~ComputeQueueScope() {
            if (split_context) {
                active_record_context = previous_record_context;
                rasterizer.buffer_cache.SetQueue(previous_buffer_queue);
                return;
            }
            if (explicit_context) {
                rasterizer.buffer_cache.RestoreSerialProducerIdentity(*previous_producer_identity);
                active_record_context = previous_record_context;
                rasterizer.on_compute_queue = previous_compute;
                rasterizer.active_runtime = previous_runtime;
                rasterizer.active_scheduler = previous_scheduler;
                rasterizer.buffer_cache.SetQueue(previous_compute ? &rasterizer.compute_context
                                                                   : nullptr);
                return;
            }
            if (rasterizer.on_compute_queue) {
                rasterizer.on_compute_queue = false;
                rasterizer.active_runtime = &rasterizer.runtime;
                rasterizer.active_scheduler = &rasterizer.scheduler;
                rasterizer.buffer_cache.SetQueue(nullptr);
            }
        }

    private:
        Rasterizer& rasterizer;
        bool split_context{};
        bool explicit_context{};
        bool previous_compute{};
        Runtime* previous_runtime{};
        Scheduler* previous_scheduler{};
        RecordContext* previous_record_context{};
        VideoCore::QueueContext* previous_buffer_queue{};
        std::optional<RecordContext> context{};
        std::optional<VideoCore::BufferCache::SerialProducerIdentity> previous_producer_identity{};
    };
    /// asc_queue: the runtime of the queue the current dispatch, copy or fill is recorded on.
    [[nodiscard]] Runtime& Rt() noexcept {
        if (active_record_context && active_record_context->rasterizer == this)
            return *active_record_context->runtime;
        return *active_runtime;
    }
    /// asc_queue: the scheduler of that queue.
    [[nodiscard]] Scheduler& Sched() noexcept {
        if (active_record_context && active_record_context->rasterizer == this)
            return *active_record_context->scheduler;
        return *active_scheduler;
    }

    static bool Qid1RecordContextEnabled() noexcept;
    u64 CurrentSubmissionGeneration() const noexcept;
    [[nodiscard]] bool OnComputeRecordQueue() const noexcept {
        if (cp_split_enabled && active_record_context &&
            active_record_context->rasterizer == this)
            return active_record_context->physical_compute;
        return on_compute_queue;
    }
    [[nodiscard]] iss::g4::CpOwner ActiveCpOwner() const noexcept {
        return active_record_context && active_record_context->rasterizer == this
                   ? active_record_context->owner : iss::g4::CpOwner::Graphics;
    }

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    /// asc_queue: the compute queue objects (created when first used) and the active ones.
    std::unique_ptr<Scheduler> compute_scheduler;
    std::unique_ptr<Runtime> compute_runtime;
    std::unique_ptr<VideoCore::StreamBuffer> compute_stream;
    VideoCore::QueueContext compute_context{};
    bool compute_queue_routing_enabled{}; ///< The serial path has admitted its first ACE dispatch.
    Runtime* active_runtime;
    Scheduler* active_scheduler;
    bool async_compute_translation{};
    struct SplitThreadState {
        const Rasterizer* rasterizer{};
        bool async{};
    };
    static thread_local SplitThreadState split_thread_state;
    bool on_compute_queue{}; ///< The current dispatch, copy or fill goes to the compute queue.
    bool cp_split_enabled{};
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct Probe {
        u32 frame;
        u32 seq;
        std::string what;
        VAddr address;
        u32 size;
        u32 offset; ///< In the probe staging buffer.
        bool gpu;
        std::array<u32, 8> host;
        std::string state;
    };
    std::vector<Probe> probes;
    StagingBufferRef probe_staging{};
    u32 probe_used{};
    /// Frame x-ray: the vertex buffer ranges bound by the last draw (dump_after "hash:vb").
    struct BoundVertexRange {
        VAddr base;
        u64 size;
        const VideoCore::Buffer* buffer;
        u64 offset;
    };
    boost::container::static_vector<BoundVertexRange, 32> dump_vertex_ranges;
    /// Writes the GPU and guest copies of a range to the x-ray folder (waits for the GPU).
    void DumpBufferToFile(std::string_view name, const VideoCore::Buffer* buffer, u64 offset,
                          VAddr address, u64 size);

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    /// Render target memo (Tuning::rt_memo): the registers and size hint a slot's description
    /// (cb_descs, db_desc) was last built from, and the cache generations its image was found
    /// with. The same registers find the same image while neither generation changes.
    struct RtMemo {
        bool valid{};
        u32 hint{};
        u64 generation{};
        u32 image_generation{};
        VideoCore::ImageId image_id{};
        u64 touched_gc_tick{};
        std::array<u32, sizeof(AmdGpu::ColorBuffer) / sizeof(u32)> regs{};
    };
    std::array<RtMemo, AmdGpu::NUM_COLOR_BUFFERS> rt_memos{};
    struct DbMemo {
        bool valid{};
        u32 hint{};
        u64 generation{};
        u32 image_generation{};
        VideoCore::ImageId image_id{};
        u64 touched_gc_tick{};
        VAddr htile_address{};
        u32 depth_view{};
        u32 depth_control{};
        std::array<u32, sizeof(AmdGpu::DepthBuffer) / sizeof(u32)> regs{};
    };
    DbMemo db_memo{};
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;
    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
        VAddr address{}; ///< Guest address (frame x-ray), 0 for emulator buffers.
    };
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    /// Texture binding memo (Tuning::tex_memo): the image and view found for a texture
    /// descriptor of a shader, valid while the texture cache generation and the image
    /// generation do not change. Small entries: a lookup is a cache miss at most.
    struct TexMemo {
        u64 key{};
        u64 generation{};
        u32 image_generation{};
        VideoCore::ImageId image_id{};
        VideoCore::ImageViewId view_id{};
        const void* view_backing{}; ///< The image backing the view belongs to.
        VideoCore::ImageViewInfo view_info{};
        VideoCore::TextureCache::BindingType type{};
    };
    static constexpr u32 NumTexMemoBits = 12;
    static constexpr size_t NumTexMemos = size_t{1} << NumTexMemoBits;
    std::unique_ptr<std::array<TexMemo, NumTexMemos>> tex_memos;
    /// Per binding of BindTextures: what the binding was made from, whether the memo gave it
    /// (no refresh check needed), its memo entry to complete, and the view it remembers.
    struct ImageSource {
        const Shader::ImageResource* image_desc{};
        AmdGpu::Image tsharp{};
        u32 index{};
        bool fresh{};
        TexMemo* memo{};
        const void* view_backing{};
        VideoCore::ImageViewId view_id{};
    };
    boost::container::static_vector<ImageSource, Shader::NUM_IMAGES> image_sources;
    /// Sampler memo (with tex_memo): the sampler handle found for an S#, valid while no sampler
    /// was deleted (TextureCache::SamplersGeneration).
    struct SamplerMemo {
        u64 key{};
        u64 generation{};
        vk::Sampler sampler{};
    };
    std::array<SamplerMemo, 1024> sampler_memos{};
    // The ASC recorder must not overwrite descriptors or memo entries that the graphics
    // recorder is still using. The graphics members above retain the serial layout; the
    // compute owner selects this separate scratch through its thread-local record context.
    struct ComputeBindingScratch {
        boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
        boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
        boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;
        boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;
        u32 set_write_index{};
        Pipeline::DescriptorWrites set_writes;
        Shader::PushData push_data;
        boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
        std::unique_ptr<std::array<TexMemo, NumTexMemos>> tex_memos;
        boost::container::static_vector<ImageSource, Shader::NUM_IMAGES> image_sources;
        std::array<SamplerMemo, 1024> sampler_memos{};
        bool needs_barrier{};
    };
    ComputeBindingScratch compute_bindings{};
    [[nodiscard]] bool ComputeBindingsActive() const noexcept {
        return active_record_context && active_record_context->rasterizer == this &&
               active_record_context->owner == iss::g4::CpOwner::Asc1;
    }
    /// bind_prefetch: the last scout entry whose lookups were prefetched.
    const AmdGpu::ScoutDraw* last_prefetched_draw{};
    /// Statistics (DrawsSameContext): the context register changes and pipeline of the last
    /// draw.
    u64 last_draw_context_changes{~u64{0}};
    const GraphicsPipeline* last_draw_pipeline{};
    // Native dynamic state is derived once per context change. Shader resources and geometry
    // retain their own lifetimes; a new Vulkan command buffer still commits all dirty state.
    struct NativeDrawState {
        u64 context_changes{~u64{0}};
        const GraphicsPipeline* pipeline{};
        bool indexed{};
        bool feedback_loop{};
        bool failed{};
        u64 hits{}, misses{}, verified{}, mismatches{};
    };
    mutable NativeDrawState native_draw_state{};
    bool attachment_feedback_loop{};
    bool needs_barrier{};
    /// Entries per stage type, indexed by shader, user data (the material) and first bindings.
    static constexpr u32 StageMemoBits = 11;
    static constexpr u32 StageMemoEntries = 1u << StageMemoBits;
    /// Zero filled on first use (an all zero StageMemo is an invalid entry); pages are only
    /// committed when touched.
    StageMemo* stage_memo_table{};
    StageMemo& StageMemoEntry(u32 slot, const Shader::Info& stage,
                              const Shader::Backend::Bindings& binding);
    /// The entry the last lookup found (the store after a miss uses it, same stage and draw).
    const Shader::Info* memo_key_stage{};
    Shader::Backend::Bindings memo_key_binding{};
    StageMemo* memo_key_entry{};
    StageMemo stage_memo_scratch{}; ///< Verification: the result of the original lookups.
    // Engine-keyed material preparation. Per-object buffers are resolved for each draw.
    struct NativeDrawRecord {
        bool valid{}, disabled{};
        const Pipeline* pipeline{};
        const Shader::Info* stage{};
        std::shared_ptr<PipelineCache::PreparedGraphicsState> prepared_pipeline;
        Shader::Backend::Bindings first{}, last{};
        u64 border_base{};
        std::vector<AmdGpu::Buffer> buffers;
        std::vector<AmdGpu::Image> images;
        std::vector<AmdGpu::Sampler> samplers;
        std::vector<std::optional<VideoCore::BufferCache::PreparedReadToken>> buffer_tokens;
        std::vector<std::optional<VideoCore::TextureCache::PreparedTextureToken>> image_tokens;
        std::vector<vk::DescriptorBufferInfo> buffer_infos;
        std::vector<vk::DescriptorImageInfo> image_infos;
        std::vector<u8> offsets;
        RenderState render_state{};
        Shader::PushData captured_push{};
    };
    struct NativeIdentityHash {
        size_t operator()(const std::array<u64, 4>& v) const noexcept {
            u64 h = 0x9e3779b97f4a7c15ULL;
            for (const auto x : v) h = (h ^ x) * 0xc2b2ae3d27d4eb4fULL;
            return h ^ (h >> 32);
        }
    };
    std::unordered_map<std::array<u64, 4>, std::unique_ptr<NativeDrawRecord>, NativeIdentityHash>
        native_items;
    NativeDrawRecord* native_recording{};
    struct NativeVisualPlan {
        std::vector<vk::WriteDescriptorSet> writes;
        std::vector<vk::DescriptorBufferInfo> buffers;
        std::vector<vk::DescriptorImageInfo> images;
        std::vector<u32> buffer_starts, image_starts;
        Shader::PushData push{};
        NativeVisualPlan() = default;
        NativeVisualPlan(const NativeVisualPlan& other) { *this = other; }
        NativeVisualPlan& operator=(const NativeVisualPlan& other) {
            if (this != &other) {
                writes = other.writes; buffers = other.buffers; images = other.images;
                buffer_starts = other.buffer_starts; image_starts = other.image_starts;
                push = other.push; Rebind();
            }
            return *this;
        }
        NativeVisualPlan(NativeVisualPlan&&) noexcept = default;
        NativeVisualPlan& operator=(NativeVisualPlan&&) noexcept = default;
        void Rebind();
        bool Capture(std::span<const vk::WriteDescriptorSet> source, const Shader::PushData& data);
    };
    struct NativeVisualLatch {
        bool eligible{}, material_verified{};
        std::array<u64, 4> identity{}; // technique, geometry, material, item
        u32 ps_first{}, ps_count{}, prepared_resources{};
        NativeVisualPlan candidate_ps, reference_ps;
        NativeVisualPlan candidate, reference;
    } native_visual_latch;
    bool NativeVisualCaptureEnabled() const;
    void CompleteNativeVisualPlans(const GraphicsPipeline* pipeline);
    void ProbeNativeVisual(const GraphicsPipeline* pipeline, RenderState& state,
                           s32 vertex_offset, u32 instance_offset);
    void ObserveNativeVisualDraw();
    bool NativeVisualFailed() const;
    struct NativeBufferAccess {
        bool cached{}, hazard{};
        u64 hazard_before{}, hazard_after{};
        std::vector<BoundBuffer> bound;
    };
    struct NativeImageAccess {
        bool cached{}, transition{};
        u64 hazard_before{}, hazard_after{};
    };
    struct NativeAccessAudit {
        NativeDrawRecord* record{};
        std::vector<NativeBufferAccess> buffers;
        std::vector<NativeImageAccess> images;
        bool images_cached{true};
    };
    NativeAccessAudit* native_access_audit{}; // sampled canonical preparation only
    // Shared production preparation for the admitted ordinary read-only PS subset.
    // Emits one descriptor's resource intent, never a draw; returns its own hazard result.
    bool PrepareNativeReadOnlyBuffer(const AmdGpu::Buffer& sharp, u64 shader_hash,
                                    u32 binding, Shader::PushData& push);
    vk::DescriptorBufferInfo PrepareNativeFlatBuffer(const Shader::Info& stage);
    struct NativeItemStats {
        u64 candidates{}, captures{}, replays{}, verified{}, mismatches{}, draws{},
            pipeline_hits{}, pipeline_verified{}, verify_candidates{};
        std::array<u64, 12> fallback{};
    } native_item_stats;
    NativeDrawRecord* NativeRecord(bool create);
    bool NativeContext() const;
    const GraphicsPipeline* LookupNativeGraphicsPipeline();
    bool TryNativeItemDraw(bool indexed, u32 index_offset);
    bool BindNativeMaterial(const Pipeline* pipeline, const Shader::Info& stage,
                            Shader::Backend::Bindings& binding, Shader::PushData& push);
    void ReportNativeItems();
    std::atomic<u64> bind_epoch{1};
    struct StageMemoStats {
        u64 checks{}, hits{}, stores{}, verified{}, mismatches{}, concurrent{}, report{};
        u64 no_memo{}, other_info{}, other_binding{}, stamp{}, tick{}, sharps{}, target{},
            capacity{};
        u64 reject_stamp{}, reject_barrier{}, reject_target{}, reject_capacity{}, reject_clip{};
        u64 cyc_hit{}, cyc_check{}, cyc_original{}, cyc_remember{}, originals{};
        u64 partial_hits{}, resolved_buffers{};
        /// Stamp misses: how many times each field differed (upload, host write, mapping, arena,
        /// gpu write, cpu dirty, hazard, bind epoch, texture, sampler).
        std::array<u64, 10> stamp_fields{};
        u64 stamp_only_bind_epoch{};
        std::array<u64, 8> slot_checks{}, slot_hits{}, slot_info{}, slot_stamp{}, slot_sharps{},
            slot_cyc_original{}, slot_originals{};
    };
    StageMemoStats stage_memo_stats{};
    struct FastDrawState {
        static constexpr u32 MaxSharps = 16;
        static constexpr u32 MaxFetch = 16;
        bool valid{};
        bool captured{};
        bool indexed{};
        const GraphicsPipeline* pipeline{};
        u64 context_changes{};
        u64 tick{};
        u64 bind_epoch{};
        u64 flush_seq{};
        StageMemoStamp stamp{};
        u32 stage_enable{};
        u32 index_type{};
        u32 num_instances{};
        std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> cb_extents{};
        u32 db_extent{};
        std::array<u8, sizeof(AmdGpu::ShaderProgram)> ps_program{};
        std::array<u8, offsetof(AmdGpu::ShaderProgram, user_data)> vs_program{};
        u8 num_vs_buffers{};
        u8 num_vs_images{};
        u8 num_vs_samplers{};
        u8 num_fetch{};
        std::array<AmdGpu::Buffer, MaxSharps> vs_vsharps{};
        std::array<AmdGpu::Image, MaxSharps> vs_tsharps{};
        std::array<AmdGpu::Sampler, MaxSharps> vs_ssharps{};
        std::array<std::array<u32, 2>, MaxFetch> fetch_formats{};
        RenderState state{};
        boost::container::static_vector<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS + 1> targets;
        boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;
        /// The descriptor set the last full or rebind draw pushed (verification).
        struct PushedWrite {
            u32 binding, count, type, first;
        };
        /// Registers of dynamic state only (fast draws re-derive dynamic state when only
        /// these changed): polygon control, depth control, stencil front/back ref masks.
        std::array<u32, 4> dyn_regs{};
        /// The pixel stage permutation the reference draw's lookup took from its scout entry.
        bool force_rebind{};
        const void* ps_scout_program{};
        u32 ps_perm{};
        bool writes_known{};
        u32 num_writes{};
        u32 num_infos{};
        std::array<PushedWrite, 64> writes{};
        std::array<std::array<u64, 3>, 160> infos{};
        /// Lean rebind: set_writes/buffer_infos still hold this draw's descriptors
        /// (bind_resources_seq unchanged) and where the vertex stage buffers sit in them.
        u64 bind_seq{};
        /// The push constants of the last draw of this state (pipeline cache re-entry).
        Shader::PushData push{};
        bool lean_ok{};
        u32 vs_write_base{};
        u32 vs_binding_base{};
        /// ISS_ENGINE_FAST: the engine draw (technique, material, kind) this state drew, and
        /// the pixel stage sharps it resolved (compared instead of the pixel user data, which
        /// the engine copies again for every draw).
        u64 engine_key{};
        bool ps_sharps_ok{};
        bool lean_ps_ok{};
        u8 num_ps_buffers{};
        u8 num_ps_images{};
        u8 num_ps_samplers{};
        std::array<AmdGpu::Buffer, MaxSharps> ps_vsharps{};
        std::array<AmdGpu::Image, MaxSharps> ps_tsharps{};
        std::array<AmdGpu::Sampler, MaxSharps> ps_ssharps{};
    };
    FastDrawState fast_draw{};
    /// ISS_FAST_CACHE: the states of the pipelines drawn last in this render pass; a shader
    /// switch back to one of them restores it instead of looking up and resolving again.
    std::array<FastDrawState, 32> fast_cache{};
    u32 fast_cache_next{};
    bool TryReenterPipeline(const AmdGpu::Regs& regs, u64 engine_key);
    /// Binds a graphics pipeline; one still compiled is bound by the recording thread once
    /// ready (ISS_LAZY_PIPELINES).
    void BindGraphicsPipeline(const GraphicsPipeline* pipeline);
    void BindPipelineLazy(Scheduler& on, vk::PipelineBindPoint bind_point,
                          const Pipeline* pipeline);
    void CapturePsSharps(const GraphicsPipeline* pipeline);
    /// Incremented by every BindResources (descriptor arrays rebuilt).
    u64 bind_resources_seq{};
    /// -1 not possible, 0 verification mismatch, 1 done (or verified).
    int TryLeanRebind(const GraphicsPipeline* pipeline, Shader::Info& vs, u32 changed_vs,
                      const Shader::PushData* verify_push, u32* written, u32& num_written,
                      Shader::Info* ps = nullptr, u32 changed_ps = 0);
    void NoteLeanLayout(const GraphicsPipeline* pipeline);
    /// Fast draw after a shader change: the new pipeline keeps the render pass.
    bool FastSwitchDraw(bool is_indexed, u32 index_offset);
    /// The pipeline a failed fast draw already looked up (Draw uses it instead of a second lookup).
    const GraphicsPipeline* fast_lookup{};
    /// Draw pipe worker (ISS_DRAW_PIPE_FAST): the packet's pipeline and changed registers.
    const GraphicsPipeline* pipe_pipeline{};
    std::array<u32, 32> pipe_changed_regs{};
    u32 pipe_num_changed_regs{};
    /// Fast draws of the same pipeline and fetch formats: the vertex input state recorded by
    /// the previous draw is still set (BindVertexBuffers only binds the buffers).
    bool keep_vertex_input{};
    bool fast_lookup_set{};
    void RememberPushedWrites();
    /// 0 same, 1 another stream copy of the same range only, 2 different.
    int ComparePushedWrites() const;
    struct FastDrawStats {
        u64 cyc_plain{}, cyc_rebind{}, cyc_fail{}, cyc_lean{}, leans{}, lean_fallbacks{}, cyc_switch{}, switches{}, switch_fails{}, reentries{}, reentry_misses{},
            engine_ps{}, engine_ps_fails{};
        u64 checks{}, hits{}, rebinds{}, ps_rebinds{}, dyn_updates{}, invalid{}, state{}, program{}, stamp{}, occlusion{}, vertex{},
            fetch{}, context{}, report{};
        u64 capture_rejects{}, verified{}, mismatches{}, stream_equiv{};
        std::array<u64, 16> reject_why{};
    };
    FastDrawStats fast_draw_stats{};
    struct SkippedWrite {
        VAddr start;
        VAddr end;
        const Pipeline* producer; ///< Entry expires once this pipeline is compiled.
    };
    boost::container::static_vector<SkippedWrite, 32> skipped_writes;
    OcclusionQueries occlusion;

    /// Forgets the results of counted occlusion queries in this range (written again).
    void ForgetCountedResults(VAddr address, u64 size);
    struct CountedResults {
        VAddr address;
        u32 num_pairs;  ///< Counter pairs written by the GPU.
        u32 num_counts; ///< Pairs whose end counter holds samples counted by the GPU.
    };
    static constexpr size_t MaxCountedResults = 64;
    /// Results of the last occlusion queries counted by the GPU, most recent last.
    boost::container::static_vector<CountedResults, MaxCountedResults> counted_results;
};

} // namespace Vulkan
