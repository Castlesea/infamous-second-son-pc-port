// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bitset>
#include <boost/preprocessor/stringize.hpp>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <array>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

#include "common/assert.h"
#include "common/debug.h"
#include "common/perf_stats.h"
#include "common/scope_exit.h"
#include "runtime_timeline.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "compute_phase.h"
#include "g4_record_ownership.h"
#include "native_item_identity.h"
#include "native_shadow.h"
#include "native_gbuffer.h"
#include "native_passes.h"
#include "pause_guard.h"
#include "unified_memory.h"
namespace Vulkan {
u32 ExecuteNativeShadowSuffix(Rasterizer&, const iss::native_shadow::Batch&, u32 first);
}

static bool SkipShadowPrefixEnabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_SHADOW_SKIP_PREFIX");
        return p && p[0] == '1';
    }();
    return enabled;
}
// State exactly at a recorded DRAW_INDEX_2. Trailing setters stay in the command
// stream; inherited fields retain their current value from ordinary interpretation.
static void CommitShadowDraw(AmdGpu::Regs& regs, const iss::native_shadow::Batch& batch, u32 index) {
    namespace ns = iss::native_shadow;
    const auto& draw = batch.draws[index];
    const auto& state = batch.states[draw.state];
    for (u32 k = 0; k < batch.state_regs.size(); ++k)
        if (!((state.inherit >> k) & 1))
            regs.reg_array[batch.state_regs[k]] = batch.state_values[state.first_value + k];
    for (u32 k = 0; k < ns::DrawUserWords; ++k)
        if (!((draw.user_inherit >> k) & 1)) regs.reg_array[ns::VsUserData2 + k] = draw.user[k];
    if (!draw.instances_inherit) regs.num_instances.num_instances = draw.instances;
    if (!draw.index_type_inherit) regs.index_buffer_type.raw = draw.index_type;
    regs.max_index_size = draw.max_index;
    regs.index_base_address.base_addr_lo = draw.index_lo;
    regs.index_base_address.base_addr_hi = draw.index_hi;
    regs.num_indices = draw.index_count;
    regs.draw_initiator = draw.initiator;
}

// ISS_DRAW_ORIGIN_CENSUS=1 (diagnosis): the draws the command processor interprets, by origin
// (0 outside any engine range, 1 a shadow range, 2 a G-buffer range), those outside by their
// VS/PS shader hashes (the names of the shader dumps and of the technique join).
static bool DrawOriginCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_DRAW_ORIGIN_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
// The span of the last G-buffer marker whose batch was found (origin 4: a draw outside the
// tracking but inside that span).
static const u32* g_census_lo{};
static const u32* g_census_hi{};
static void NoteDrawOrigin(const AmdGpu::Regs& regs, u32 origin) {
    static std::array<u64, 5> origins{};
    static std::unordered_map<u64, u64> hashes; // program address -> shader hash
    static std::map<std::pair<u64, u64>, u64> pairs;
    static u64 total{};
    ++origins[origin];
    const auto hash_of = [&](const auto& program) -> u64 {
        const VAddr address = program.template Address<VAddr>();
        if (!address) return 0;
        auto [it, inserted] = hashes.try_emplace(address, 0);
        if (inserted) it->second = AmdGpu::GetParams(program).hash;
        return it->second;
    };
    if (origin == 0) ++pairs[{hash_of(regs.vs_program), hash_of(regs.ps_program)}];
    if ((++total & ((1u << 21) - 1)) != 0) return;
    std::vector<std::pair<u64, std::pair<u64, u64>>> top;
    for (const auto& [key, n] : pairs) top.emplace_back(n, key);
    std::sort(top.rbegin(), top.rend());
    std::string text;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 24); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %llx/%llx=%llu", (unsigned long long)top[i].second.first,
                      (unsigned long long)top[i].second.second, (unsigned long long)top[i].first);
        text += item;
    }
    std::fprintf(stderr, "DRAW_ORIGIN total=%llu outside=%llu shadow=%llu gbuffer=%llu lost=%llu "
                 "in_span=%llu pairs=%zu top[%s ]\n",
                 (unsigned long long)total, (unsigned long long)origins[0],
                 (unsigned long long)origins[1], (unsigned long long)origins[2],
                 (unsigned long long)origins[3], (unsigned long long)origins[4], pairs.size(),
                 text.c_str());
}

// ISS_NATIVE_RECORDER chains: the register writes between two chained ranges (only SET_*_REG
// packets without render target registers, checked by the engine thread that linked them).
static void ApplyPlainRegisterWrites(AmdGpu::Regs& regs, const u32* p, const u32* end) {
    while (p < end) {
        const u32 header = *p;
        if ((header >> 30) == 2) {
            ++p;
            continue;
        }
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        const u32 base = opcode == 0x68 ? 0x2000 : opcode == 0x69 ? 0xA000
                       : opcode == 0x76 ? 0x2C00 : opcode == 0x79 ? 0xC000 : 0;
        if (base && count > 1)
            std::memcpy(&regs.reg_array[base + p[1]], p + 2, (count - 1) * sizeof(u32));
        else if (opcode == 0x2A) regs.index_buffer_type.raw = p[1];
        else if (opcode == 0x2F) regs.num_instances.num_instances = p[1];
        p += 1 + count;
    }
}

// The registers at the end of a range the command processor did not interpret.
static void CommitNativeBatch(AmdGpu::Regs& regs, const iss::native_gbuffer::Batch& batch) {
    for (const auto& [reg, value] : batch.final_regs) regs.reg_array[reg] = value;
    if (batch.has_instances) regs.num_instances.num_instances = batch.final_instances;
    if (batch.has_index_type) regs.index_buffer_type.raw = batch.final_index_type;
    const auto& last = batch.draws.back();
    regs.max_index_size = last.max_index;
    regs.index_base_address.base_addr_lo = last.index_lo;
    regs.index_base_address.base_addr_hi = last.index_hi;
    regs.num_indices = last.index_count;
    regs.draw_initiator = last.initiator;
}

// ISS_NATIVE_GBUFFER: the registers exactly at a recorded draw of a G-buffer range.
static void CommitGbufferDraw(AmdGpu::Regs& regs, const iss::native_gbuffer::Batch& batch, u32 index) {
    namespace ng = iss::native_gbuffer;
    const auto& draw = batch.draws[index];
    const auto& state = batch.states[draw.state];
    for (u32 k = 0; k < batch.state_regs.size(); ++k)
        if (!((state.inherit >> k) & 1))
            regs.reg_array[batch.state_regs[k]] = batch.state_values[state.first_value + k];
    for (u32 k = 0; k < ng::UserWords; ++k) {
        if (!((draw.vs_inherit >> k) & 1)) regs.reg_array[ng::VsUser0 + k] = draw.vs_user[k];
        if (!((draw.ps_inherit >> k) & 1)) regs.reg_array[ng::PsUser0 + k] = draw.ps_user[k];
    }
    if (!draw.instances_inherit) regs.num_instances.num_instances = draw.instances;
    if (!draw.index_type_inherit) regs.index_buffer_type.raw = draw.index_type;
    regs.max_index_size = draw.max_index;
    regs.index_base_address.base_addr_lo = draw.index_lo;
    regs.index_base_address.base_addr_hi = draw.index_hi;
    regs.num_indices = draw.index_count;
    regs.draw_initiator = draw.initiator;
}

void IssPinCommandProcessor(); // bridge.cpp, ISS_CP_CPU
void IssEngineMarker(const u32* payload); // vk_rasterizer.cpp, ISS_ENGINE_MARKERS
bool IssPassCostEnabled();                 // vk_rasterizer.cpp, ISS_PASS_COST
void IssPassCostTick(u32 packet_class);
extern u64 g_iss_dcb_lo, g_iss_dcb_hi;    // buffer_cache.cpp, ISS_OBTAIN_CENSUS

namespace iss::command_trace {
extern const bool enabled;
void Packet(int queue, u32 opcode, const u32* words, size_t count);
}


#include "hybrid_family_profile.inl"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    if(rasterizer)rasterizer->DrainDrawPipe(3|0x100); /* 0x100: ISS_DRAW_PIPE_LAZY skips */   \
    FIBER_EXIT;                                                                                    \
    if (iss::hybrid_profile::Enabled()) ++iss::hybrid_profile::switches;               \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])
// A yield of a compute queue task: the graphics queue runs meanwhile (asc_queue routing off).
#define YIELD_ASC_QUEUE(id)                                                                        \
    do {                                                                                           \
        if (rasterizer) {                                                                          \
            rasterizer->SetAsyncCompute(false);                                                    \
        }                                                                                          \
        YIELD_ASC(id);                                                                             \
        if (rasterizer) {                                                                          \
            rasterizer->SetAsyncCompute(true);                                                     \
        }                                                                                          \
    } while (0)

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    if (iss::hybrid_profile::Enabled()) ++iss::hybrid_profile::switches;               \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

// A G emission call may retry admission before its first Vulkan command. The
// nested task yields without a lease, allowing C to seal conflicting work.
#define RUN_GFX_GPU(action)                                                                         \
    do {                                                                                            \
        auto g4_gpu_task = ProcessGraphicsGpuCall(action);                                          \
        RESUME_GFX(g4_gpu_task);                                                                    \
        while (!g4_gpu_task.handle.done()) {                                                        \
            YIELD_GFX();                                                                            \
            RESUME_GFX(g4_gpu_task);                                                                \
        }                                                                                           \
        g4_gpu_task.handle.destroy();                                                               \
    } while (0)

#define RUN_GFX_HOST(address, size, action)                                                         \
    do {                                                                                            \
        auto g4_host_task = ProcessGraphicsHostWrite(address, size, action);                        \
        RESUME_GFX(g4_host_task);                                                                   \
        while (!g4_host_task.handle.done()) {                                                       \
            YIELD_GFX();                                                                            \
            RESUME_GFX(g4_host_task);                                                               \
        }                                                                                           \
        g4_host_task.handle.destroy();                                                              \
    } while (0)

#define RUN_GFX_HOST_WHEN(address, size, action, ready)                                              \
    do {                                                                                            \
        auto g4_host_task = ProcessGraphicsHostWrite(address, size, action, ready);                 \
        RESUME_GFX(g4_host_task);                                                                   \
        while (!g4_host_task.handle.done()) {                                                       \
            YIELD_GFX();                                                                            \
            RESUME_GFX(g4_host_task);                                                               \
        }                                                                                           \
        g4_host_task.handle.destroy();                                                              \
    } while (0)

#define RUN_CE_HOST(address, size, action)                                                          \
    do {                                                                                            \
        auto g4_host_task = ProcessGraphicsHostWrite(address, size, action);                        \
        RESUME_CE(g4_host_task);                                                                    \
        while (!g4_host_task.handle.done()) {                                                       \
            YIELD_CE();                                                                             \
            RESUME_CE(g4_host_task);                                                                \
        }                                                                                           \
        g4_host_task.handle.destroy();                                                              \
    } while (0)

#define RUN_ASC_HOST(id, address, size, action)                                                     \
    do {                                                                                            \
        auto g4_asc_host_task = ProcessAscHostWrite(id, address, size, action);                    \
        RESUME_ASC(g4_asc_host_task, id);                                                           \
        while (!g4_asc_host_task.handle.done()) {                                                  \
            YIELD_ASC_QUEUE(id);                                                                   \
            RESUME_ASC(g4_asc_host_task, id);                                                       \
        }                                                                                           \
        g4_asc_host_task.handle.destroy();                                                         \
    } while (0)

#define RUN_ASC_HOST_WHEN(id, address, size, action, ready)                                         \
    do {                                                                                            \
        auto g4_asc_host_task = ProcessAscHostWrite(id, address, size, action, ready);             \
        RESUME_ASC(g4_asc_host_task, id);                                                           \
        while (!g4_asc_host_task.handle.done()) {                                                  \
            YIELD_ASC_QUEUE(id);                                                                   \
            RESUME_ASC(g4_asc_host_task, id);                                                       \
        }                                                                                           \
        g4_asc_host_task.handle.destroy();                                                         \
    } while (0)

#define RUN_OCCLUSION_HOST(address)                                                                 \
    do {                                                                                            \
        const VAddr g4_counter_address = address;                                                   \
        const u64 g4_counter_bytes = num_counter_pairs * 2 * sizeof(u64);                          \
        if (CpSplitActive()) {                                                                       \
            const auto write = [this, g4_counter_address] {                                        \
                WriteEmulatedOcclusionCounters(g4_counter_address);                                \
            };                                                                                     \
            RUN_GFX_HOST(g4_counter_address, g4_counter_bytes, write);                             \
        } else {                                                                                    \
            if (rasterizer)                                                                        \
                rasterizer->BeforeGpuThreadWrite(g4_counter_address, g4_counter_bytes);            \
            WriteEmulatedOcclusionCounters(g4_counter_address);                                    \
            if (rasterizer)                                                                        \
                rasterizer->OnGpuThreadWrite(g4_counter_address, g4_counter_bytes);                \
        }                                                                                           \
    } while (0)

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;
thread_local Liverpool::CpThreadContext Liverpool::thread_context{};

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    asc_process_thread.request_stop();
    {
        std::scoped_lock lk{submit_mutex};
        cp_stopping.store(true, std::memory_order_release);
        cp_coordination.Stop();
        asc_owner_pause.Stop();
        for (const auto& waiter : command_waiters) {
            u32 pending = 0;
            if (waiter->state.compare_exchange_strong(pending, 3u)) waiter->sem.release();
        }
    }
    graphics_controls.Stop();
    fallback_mailbox.Stop();
    submit_cv.notify_all(); // Wake an ASC unmap waiter so shutdown can cancel it.
    if (asc_process_thread.joinable()) asc_process_thread.join();
    process_thread.join();
    for (auto& queue : mapped_queues) {
        std::scoped_lock lk{queue.m_access};
        while (!queue.submits.empty()) {
            queue.submits.front().handle.destroy();
            queue.submits.pop();
        }
    }
}

void Liverpool::DecideCpSplit() {
    bool has_fallback_recorder{};
    {
        std::scoped_lock lk{submit_mutex};
        ASSERT_MSG(!cp_split_decided.exchange(true, std::memory_order_acq_rel),
                   "CP split startup gate was run twice");
        has_fallback_recorder = static_cast<bool>(graphics_fallback_recorder);
    }
    if (!cp_split_requested) return;
    std::string reason;
    if (!rasterizer) {
        reason = "rasterizer_unbound";
    } else if (!Common::Perf::GetTuning().asc_queue.load(std::memory_order_acquire)) {
        reason = "async_compute_disabled";
    } else if (iss::command_trace::enabled) {
        reason = "legacy_command_trace";
    } else if (iss::phase::Enabled()) {
        reason = "compute_phases";
    } else if (!rasterizer->PrepareCpSplit(reason)) {
        if (reason.empty()) reason = "rasterizer_unavailable";
    } else if (!has_fallback_recorder) {
        reason = "fallback_recorder_unavailable";
    } else {
        // Packet fallbacks now have a G mailbox route, but a page fault raised by a
        // recording worker can still re-enter shared cache state without a safe owner
        // handoff. Keep this independent gate after rasterizer packet preflight.
        reason = "worker_page_fault_route_unavailable";
    }
    if (!reason.empty()) {
        std::fprintf(stderr, "CP_SPLIT disabled reason=%s\n", reason.c_str());
        LOG_INFO(Lib_GnmDriver, "CP_SPLIT disabled reason={}", reason);
        return;
    }
    // The C thread is created only after every backend precondition passed. It cannot see
    // queued work until cp_split_active is released and the condition variable is signalled.
    asc_process_thread = std::jthread{std::bind_front(&Liverpool::ProcessAsc, this)};
    cp_split_active.store(true, std::memory_order_release);
    submit_cv.notify_all();
    std::fprintf(stderr, "CP_SPLIT enabled owners=G,C asc_qids=1..56\n");
    LOG_INFO(Lib_GnmDriver, "CP_SPLIT enabled owners=G,C asc_qids=1..56");
}

bool Liverpool::BindGraphicsFallbackRecorder(GraphicsFallbackRecorder recorder) {
    std::scoped_lock lk{submit_mutex};
    if (!recorder || cp_stopping.load(std::memory_order_acquire) ||
        cp_split_decided.load(std::memory_order_acquire)) return false;
    graphics_fallback_recorder = std::move(recorder);
    return true;
}

Liverpool::FallbackRequest Liverpool::EnqueueAscFallback(
    iss::g4::FallbackPacket packet, const SealRequest& sealed_asc_prefix) {
    if (!CpSplitActive() || !sealed_asc_prefix || thread_context.owner != this ||
        thread_context.qid <= 0 ||
        packet.guest.qid != static_cast<u32>(thread_context.qid) ||
        packet.guest.generation != thread_context.token.generation ||
        packet.guest.submission_sequence != thread_context.token.sequence) return {};
    auto request = fallback_mailbox.Enqueue(std::move(packet), *sealed_asc_prefix);
    if (request) {
        fallback_pending.fetch_add(1, std::memory_order_release);
        submit_cv.notify_all();
    }
    return request;
}

Liverpool::Task Liverpool::ProcessPreparedAscFallback(iss::g4::FallbackPacket packet,
                                                       u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    const u32 qid = vqid + 1;
    ASSERT_MSG(CpSplitActive() && CurrentQueue() == static_cast<s32>(qid) &&
                   packet.guest.qid == qid &&
                   packet.guest.generation == CurrentSubmissionGeneration(),
               "G4 fallback has no active ASC owner or stale guest identity");
    // All earlier C commands must be submitted before G may record this packet. The
    // self-seal runs at a resolve safe point without a cache lock or pending record.
    const QidTick asc_prefix{CpLane::Asc, qid, packet.guest.generation,
                             CpOpenTick(CpLane::Asc)};
    auto c_seal = SealAtLeast(asc_prefix);
    ASSERT_MSG(WaitCpSeal(c_seal, CpLane::Asc) == iss::g4::SealState::Done,
               "G4 fallback could not seal preceding ASC prefix");
    auto request = EnqueueAscFallback(std::move(packet), c_seal);
    ASSERT_MSG(request, "G4 fallback packet rejected after ASC seal");

    // Do not block the C worker: other ASC qids and owner seal controls can advance
    // while this guest qid is suspended at the fallback packet.
    while (request->State() == iss::g4::FallbackState::Queued ||
           request->State() == iss::g4::FallbackState::RecordingGraphics ||
           request->State() == iss::g4::FallbackState::GraphicsRecorded) {
        YIELD_ASC_QUEUE(vqid);
    }
    ASSERT_MSG(request->State() == iss::g4::FallbackState::GraphicsSealed,
               "G4 graphics fallback failed or was cancelled");
    const auto graphics_token = request->GraphicsResult();
    ASSERT_MSG(graphics_token, "G4 fallback returned no sealed G token");
    // CPU-side DAG bookkeeping alone cannot order the two Vulkan queues. Install the
    // actual G timeline wait on C before resuming the next PM4 packet of this qid.
    ASSERT_MSG(AddCpDependency(CpTick{CpLane::Asc, CpOpenTick(CpLane::Asc)},
                               *graphics_token),
               "G4 fallback dependency would form a cycle");
    ASSERT_MSG(rasterizer && rasterizer->WaitAscOnGraphicsToken(*graphics_token),
               "G4 fallback G token was not inserted into ASC Vulkan waits");
    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessAscDmaFallback(
    u32 vqid, iss::g4::DmaFallbackPacket packet) {
    FIBER_ENTER(acb_task_name[vqid]);
    const u32 qid = vqid + 1;
    ASSERT_MSG(CpSplitActive() && rasterizer && packet.Valid() &&
                   packet.guest_qid == qid && CurrentQueue() == static_cast<s32>(qid) &&
                   packet.guest_generation == CurrentSubmissionGeneration() &&
                   packet.submission_sequence == thread_context.token.sequence,
               "G4 DMA fallback lacks a live owning ASC packet");
    for (;;) {
        // No packet lease survives a yield. Complete C's prior GPU work before G
        // reads a source buffer or writes a destination through the legacy path.
        const QidTick c_prefix{CpLane::Asc, qid, packet.guest_generation,
                               CpOpenTick(CpLane::Asc)};
        auto c_seal = SealAtLeast(c_prefix);
        ASSERT_MSG(WaitCpSeal(c_seal, CpLane::Asc) == iss::g4::SealState::Done &&
                       rasterizer->WaitAscGpuComplete(c_prefix.tick),
                   "G4 DMA fallback could not finish the preceding C prefix");

        auto result = std::make_shared<iss::g4::DmaFallbackResult>();
        auto control = EnqueuePausedGraphicsControl(
            [this, packet, result] {
                ASSERT_MSG(AscOwnerPausedForGraphics() && CurrentQueue() == 0,
                           "G4 DMA graphics action ran outside the paused G safe point");
                auto admission = rasterizer->TryPrepareGraphicsGpuPacket();
                if (admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Busy) {
                    result->status = iss::g4::DmaFallbackStatus::Busy;
                    return;
                }
                if (admission.status != Vulkan::Rasterizer::GraphicsHostWriteStatus::Ready) {
                    result->status = iss::g4::DmaFallbackStatus::Failed;
                    return;
                }
                if (!rasterizer->ResolveGraphicsGpuPacket(admission.guard) ||
                    !rasterizer->PublishGraphicsGpuPacket(admission.guard)) {
                    rasterizer->FinishGraphicsHostWrite(admission.guard);
                    result->status = iss::g4::DmaFallbackStatus::Busy;
                    return;
                }
                // Everything the G action reads from this packet is a scalar copy.
                // The live global lease covers internal Flush calls as well.
                if (packet.kind == iss::g4::DmaFallbackKind::Fill) {
                    rasterizer->FillBuffer(packet.dst, packet.size, packet.value,
                                           packet.dst_gds);
                    if (packet.record_label)
                        RecordCpLabel(packet.dst, &packet.value,
                                      std::min<u32>(packet.size, sizeof(packet.value)));
                } else {
                    rasterizer->CopyBuffer(packet.dst, packet.src, packet.size,
                                           packet.dst_gds, packet.src_gds);
                }
                rasterizer->FinishGraphicsGpuPacket(admission.guard);
                result->graphics_tick = admission.guard.final_tick;
                result->status = iss::g4::DmaFallbackStatus::Ready;
            }, c_prefix.tick);
        ASSERT_MSG(control, "G4 DMA fallback could not pause the C owner");
        while (control->Status() == iss::g4::GraphicsControlMailbox::Control::State::Queued ||
               control->Status() == iss::g4::GraphicsControlMailbox::Control::State::Running) {
            YIELD_ASC_QUEUE(vqid);
        }
        ASSERT_MSG(control->Status() == iss::g4::GraphicsControlMailbox::Control::State::Done,
                   "G4 DMA graphics control failed or was cancelled");
        if (result->status == iss::g4::DmaFallbackStatus::Busy) {
            YIELD_ASC_QUEUE(vqid);
            continue;
        }
        ASSERT_MSG(result->status == iss::g4::DmaFallbackStatus::Ready,
                   "G4 DMA graphics action failed");
        if (result->graphics_tick != 0) {
            const QidTick g_token{CpLane::Graphics, 0, CpGraphicsGeneration(),
                                  result->graphics_tick};
            auto g_seal = SealAtLeast(g_token);
            ASSERT_MSG(WaitCpSeal(g_seal, CpLane::Asc) == iss::g4::SealState::Done,
                       "G4 DMA final G tick was not submitted");
            ASSERT_MSG(AddCpDependency(CpTick{CpLane::Asc, CpOpenTick(CpLane::Asc)}, g_token) &&
                           rasterizer->WaitAscOnGraphicsToken(g_token),
                       "G4 DMA final G tick was not installed in C's Vulkan waits");
        }
        break;
    }
    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphicsGpuCall(std::function<void()> action) {
    FIBER_ENTER(dcb_task_name);
    ASSERT_MSG(CpSplitActive() && rasterizer && action,
               "G4 graphics packet admission requires a live G owner");
    for (;;) {
        auto admission = rasterizer->TryPrepareGraphicsGpuPacket();
        if (admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Busy) {
            YIELD_GFX();
            continue;
        }
        ASSERT_MSG(admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Ready,
                   "G4 graphics packet admission failed");
        if (!rasterizer->ResolveGraphicsGpuPacket(admission.guard) ||
            !rasterizer->PublishGraphicsGpuPacket(admission.guard)) {
            rasterizer->FinishGraphicsGpuPacket(admission.guard);
            YIELD_GFX();
            continue;
        }
        // Publish is irreversible: a failure after this point ends the diagnostic run.
        action();
        // The live global lease blocks C reservations across any internal Flush.
        // Finish publishes the action's actual final scheduler tick before release.
        rasterizer->FinishGraphicsGpuPacket(admission.guard);
        break;
    }
    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphicsHostWrite(VAddr address, u64 size,
                                                    std::function<void()> action,
                                                    std::function<bool()> ready) {
    FIBER_ENTER(dcb_task_name);
    ASSERT_MSG(CpSplitActive() && rasterizer && address && size && action,
               "G4 host write requires a live G owner and complete value snapshot");
    for (;;) {
        auto admission = rasterizer->TryPrepareGraphicsHostWrite(address, size);
        if (admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Busy) {
            YIELD_GFX();
            continue;
        }
        ASSERT_MSG(admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Ready,
                   "G4 graphics host write admission failed");
        if (!rasterizer->ResolveGraphicsHostWrite(admission.guard)) {
            rasterizer->FinishGraphicsHostWrite(admission.guard);
            YIELD_GFX();
            continue;
        }
        if (ready && !ready()) {
            rasterizer->FinishGraphicsHostWrite(admission.guard);
            YIELD_GFX();
            continue;
        }
        // The C producer is submitted and GPU-complete. No cache/packet mutex is held
        // while this guest VA write can fault. The guard is never carried over a yield.
        rasterizer->BeforeGpuThreadWrite(address, size);
        action();
        rasterizer->OnGpuThreadWrite(address, size);
        rasterizer->FinishGraphicsHostWrite(admission.guard);
        break;
    }
    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessAscHostWrite(u32 vqid, VAddr address, u64 size,
                                               std::function<void()> action,
                                               std::function<bool()> ready) {
    FIBER_ENTER(acb_task_name[vqid]);
    ASSERT_MSG(CpSplitActive() && rasterizer && vqid < NumComputeRings &&
                   CurrentQueue() == static_cast<s32>(vqid + 1) && address && size && action,
               "G4 ASC host write requires a live C owner and owning value snapshot");
    for (;;) {
        auto admission = rasterizer->TryPrepareAscHostWrite(address,size);
        if (admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Busy) {
            YIELD_ASC_QUEUE(vqid);
            continue;
        }
        ASSERT_MSG(admission.status == Vulkan::Rasterizer::GraphicsHostWriteStatus::Ready,
                   "G4 ASC host write admission failed");
        if (!rasterizer->ResolveAscHostWrite(admission.guard)) {
            rasterizer->FinishAscHostWrite(admission.guard);
            YIELD_ASC_QUEUE(vqid);
            continue;
        }
        if (ready && !ready()) {
            rasterizer->FinishAscHostWrite(admission.guard);
            YIELD_ASC_QUEUE(vqid);
            continue;
        }
        rasterizer->BeforeGpuThreadWrite(address,size);
        action();
        rasterizer->OnGpuThreadWrite(address,size);
        rasterizer->FinishAscHostWrite(admission.guard);
        break;
    }
    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessSplitAscDispatch(
    u32 vqid, ComputeProgram cs, std::optional<std::pair<VAddr, u32>> indirect) {
    FIBER_ENTER(acb_task_name[vqid]);
    const u32 qid = vqid + 1;
    ASSERT_MSG(CpSplitActive() && rasterizer && CurrentQueue() == static_cast<s32>(qid),
               "G4 dispatch has no active ASC owner");
    const auto submission = thread_context.token;
    ASSERT_MSG(submission.qid == qid && submission.generation != 0 &&
                   submission.sequence != 0,
               "G4 dispatch has no live guest submission token");
    const iss::g4::GuestPacketIdentity guest{
        qid, submission.generation, submission.sequence,
        ++mapped_queues[qid].packet_sequence};
    const Vulkan::Rasterizer::AscDispatchInput input{cs, guest, indirect};
    u32 image_fallback_retries{};

    for (;;) {
        // Prepare can fault, allocate, and retry. Nothing has been emitted when it
        // returns Failed or Commit returns Retry; discard its snapshots before yielding.
        auto plan = rasterizer->PrepareAscDispatch(input);
        if (plan.route == Vulkan::Rasterizer::AscDispatchRoute::Compute) {
            const auto result = rasterizer->CommitAscDispatch(plan);
            if (result == Vulkan::Rasterizer::AscCommitResult::Recorded) break;
            ASSERT_MSG(result == Vulkan::Rasterizer::AscCommitResult::Retry,
                       "G4 compute packet failed before emission");
            plan = {};
            YIELD_ASC_QUEUE(vqid);
            continue;
        }
        if (plan.route == Vulkan::Rasterizer::AscDispatchRoute::RetryNeedUpload) {
            ASSERT_MSG(!plan.upload_ranges.empty(), "G4 upload retry has no owning ranges");
            auto ranges = std::move(plan.upload_ranges);
            plan = {}; // release SRT/page watches before the owner-wide pause
            const QidTick c_prefix{CpLane::Asc,qid,guest.generation,CpOpenTick(CpLane::Asc)};
            auto c_seal = SealAtLeast(c_prefix);
            ASSERT_MSG(WaitCpSeal(c_seal,CpLane::Asc) == iss::g4::SealState::Done &&
                           rasterizer->WaitAscGpuComplete(c_prefix.tick),
                       "G4 upload control lacks a submitted and GPU-complete C prefix");
            auto result = std::make_shared<Vulkan::Rasterizer::AscArenaControlResult>();
            auto control = EnqueuePausedGraphicsControl(
                [this,result,ranges=std::move(ranges)] {
                    *result = rasterizer->PrepareAscArenaOnGraphics(ranges);
                }, c_prefix.tick);
            ASSERT_MSG(control,
                       "G4 upload control could not pause the C owner");
            while (control->Status() == iss::g4::GraphicsControlMailbox::Control::State::Queued ||
                   control->Status() == iss::g4::GraphicsControlMailbox::Control::State::Running) {
                YIELD_ASC_QUEUE(vqid);
            }
            ASSERT_MSG(control->Status() ==
                           iss::g4::GraphicsControlMailbox::Control::State::Done,
                       "G4 upload graphics control failed or was cancelled");
            if (result->status == Vulkan::Rasterizer::AscArenaControlStatus::Busy) {
                YIELD_ASC_QUEUE(vqid);
                continue;
            }
            ASSERT_MSG(result->status == Vulkan::Rasterizer::AscArenaControlStatus::Ready,
                       "G4 upload graphics control could not prepare the arena");
            if (result->graphics_tick) {
                const QidTick g_token{CpLane::Graphics,0,CpGraphicsGeneration(),
                                       result->graphics_tick};
                auto g_seal = SealAtLeast(g_token);
                ASSERT_MSG(WaitCpSeal(g_seal,CpLane::Asc) == iss::g4::SealState::Done &&
                               rasterizer->WaitAscOnGraphicsToken(g_token),
                           "G4 upload graphics tick was not sealed and waited by C");
            }
            // Re-snapshot SRT and CPU versions from scratch after the G upload.
            continue;
        }
        if (plan.route == Vulkan::Rasterizer::AscDispatchRoute::NoOp) {
            // A completed HLE operation with no GPU command still consumed this
            // guest dispatch in order. No fallback or scheduler token is needed.
            break;
        }
        if (plan.route == Vulkan::Rasterizer::AscDispatchRoute::FallbackGraphics) {
            ASSERT_MSG(plan.fallback &&
                           plan.fallback->Gate() == iss::g4::FallbackGate::Ready,
                       "G4 graphics fallback lacks a complete packet");
            if (plan.fallback->serial_image) {
                auto packet = std::make_shared<iss::g4::FallbackPacket>(
                    std::move(*plan.fallback));
                plan = {};
                // The image cache may lazily read guest bytes, refresh images and
                // emit Vulkan commands. Give that entire legacy action exclusive
                // ownership of G after the preceding C work is GPU-complete.
                const QidTick c_prefix{CpLane::Asc, qid, guest.generation,
                                       CpOpenTick(CpLane::Asc)};
                auto c_seal = SealAtLeast(c_prefix);
                ASSERT_MSG(WaitCpSeal(c_seal, CpLane::Asc) == iss::g4::SealState::Done &&
                               rasterizer->WaitAscGpuComplete(c_prefix.tick),
                           "G4 image fallback could not complete its C prefix");
                auto result = std::make_shared<
                    std::optional<iss::g4::GraphicsRecordingProof>>();
                auto control = EnqueuePausedGraphicsControl(
                    [this, packet, c_prefix, result] {
                        *result = rasterizer->RecordSerialImageFallback(*packet, c_prefix);
                    }, c_prefix.tick);
                if (!control) {
                    ASSERT_MSG(++image_fallback_retries < 256,
                               "G4 image fallback could not acquire an ASC pause");
                    YIELD_ASC_QUEUE(vqid);
                    continue;
                }
                while (control->Status() ==
                           iss::g4::GraphicsControlMailbox::Control::State::Queued ||
                       control->Status() ==
                           iss::g4::GraphicsControlMailbox::Control::State::Running) {
                    YIELD_ASC_QUEUE(vqid);
                }
                ASSERT_MSG(control->Status() ==
                               iss::g4::GraphicsControlMailbox::Control::State::Done,
                           "G4 image fallback control failed or was cancelled");
                if (!*result) {
                    ASSERT_MSG(++image_fallback_retries < 256,
                               "G4 image fallback repeatedly rejected the same packet");
                    YIELD_ASC_QUEUE(vqid);
                    continue;
                }
                const QidTick g_token{CpLane::Graphics, 0, CpGraphicsGeneration(),
                                      (*result)->graphics_tick};
                auto g_seal = SealAtLeast(g_token);
                ASSERT_MSG(WaitCpSeal(g_seal, CpLane::Asc) == iss::g4::SealState::Done &&
                               AddCpDependency(
                                   CpTick{CpLane::Asc, CpOpenTick(CpLane::Asc)},
                                   g_token) &&
                               rasterizer->WaitAscOnGraphicsToken(g_token),
                           "G4 image fallback final G tick was not waited by C");
                break;
            }
            auto fallback = ProcessPreparedAscFallback(std::move(*plan.fallback), vqid);
            plan = {};
            RESUME_ASC(fallback, vqid);
            while (!fallback.handle.done()) {
                YIELD_ASC_QUEUE(vqid);
                RESUME_ASC(fallback, vqid);
            }
            fallback.handle.destroy();
            break;
        }
        UNREACHABLE_MSG("G4 ASC dispatch has no safe prepared route");
    }
    FIBER_EXIT;
}

Liverpool::GraphicsControlHandle Liverpool::EnqueueGraphicsControl(
    std::function<void()> operation) {
    if (!operation || !CpSplitActive() || thread_context.owner != this ||
        thread_context.qid <= 0 || thread_context.token.generation == 0) return {};
    if (cp_stopping.load(std::memory_order_acquire)) return {};
    const u32 guest_qid = static_cast<u32>(thread_context.qid);
    const u64 guest_generation = thread_context.token.generation;
    auto request = graphics_controls.Enqueue(
        [operation = std::move(operation), guest_qid, guest_generation] {
            iss::g4::RecordingOwnerScope owner{
                {CpLane::Graphics, guest_qid, guest_generation}};
            operation();
        });
    if (request) submit_cv.notify_all();
    return request;
}

Liverpool::GraphicsControlHandle Liverpool::EnqueueGuestGraphicsControl(
    std::function<void()> operation) {
    if (!operation || !CpSplitActive() || cp_stopping.load(std::memory_order_acquire)) return {};
    auto request = graphics_controls.Enqueue(std::move(operation));
    if (request) submit_cv.notify_all();
    return request;
}

Liverpool::GraphicsControlHandle Liverpool::EnqueuePausedGraphicsControl(
    std::function<void()> operation, u64 sealed_tick) {
    if (!operation || !CpSplitActive() || thread_context.owner != this ||
        thread_context.qid <= 0 || thread_context.token.generation == 0 ||
        !sealed_tick || !rasterizer ||
        !rasterizer->AscSchedulerIdleAfterSeal(sealed_tick)) return {};
    const u32 qid = static_cast<u32>(thread_context.qid);
    const u64 generation = thread_context.token.generation;
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(CpLane::Asc);
    if (cp_stopping.load(std::memory_order_acquire) || asc_owner_pause.Active() ||
        cp_coordination.SubmittedTick(CpLane::Asc) < sealed_tick) return {};
    auto request = graphics_controls.Enqueue(
        [operation=std::move(operation),qid,generation] {
            iss::g4::RecordingOwnerScope owner{{CpLane::Graphics,qid,generation}};
            operation();
        });
    if (!request) return {};
    ASSERT_MSG(asc_owner_pause.Begin(qid,request,sealed_tick),
               "G4 paused C control was queued without owner pause");
    submit_cv.notify_all();
    return request;
}

Liverpool::GraphicsControlHandle Liverpool::EnqueuePausedGuestGraphicsControl(
    std::function<void()> operation, u64 sealed_tick) {
    if (!operation || !CpSplitActive() || !sealed_tick ||
        (thread_context.owner == this && thread_context.qid != 0)) return {};
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(CpLane::Asc);
    if (cp_stopping.load(std::memory_order_acquire) || asc_owner_pause.Active() ||
        cp_coordination.SubmittedTick(CpLane::Asc) < sealed_tick) return {};
    const u64 generation = g4_ledger.Generation(GfxQueueId);
    auto request = graphics_controls.Enqueue(
        [operation=std::move(operation),generation] {
            iss::g4::RecordingOwnerScope owner{{CpLane::Graphics,0,generation}};
            operation();
        });
    if (!request) return {};
    ASSERT_MSG(asc_owner_pause.BeginGuest(request,sealed_tick),
               "G4 guest control was queued without C owner pause");
    submit_cv.notify_all();
    return request;
}

Liverpool::GraphicsControlHandle Liverpool::EnqueueSealingPausedGuestGraphicsControl(
    std::function<void()> operation) {
    if (!operation || !CpSplitActive() ||
        (thread_context.owner == this && thread_context.qid != 0)) return {};
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(CpLane::Asc);
    if (cp_stopping.load(std::memory_order_acquire) || asc_owner_pause.Active()) return {};
    // Queue the G operation and its C seal while holding the same admission
    // mutex. C cannot begin another packet between choosing the prefix and
    // installing the pause; C itself flushes its open scheduler at a safe point.
    const u64 generation = g4_ledger.Generation(GfxQueueId);
    auto request = graphics_controls.Enqueue(
        [operation = std::move(operation), generation] {
            iss::g4::RecordingOwnerScope owner{{CpLane::Graphics, GfxQueueId, generation}};
            operation();
        });
    if (!request) return {};
    auto seal = cp_coordination.SealGuestAscPrefix();
    ASSERT_MSG(seal && seal->State() == iss::g4::SealState::Pending &&
                   asc_owner_pause.BeginGuest(request, seal->target.tick),
               "G4 guest readback could not pin C's open prefix");
    guest_pause_control = request;
    guest_pause_seal = seal;
    cp_seal_requests[1].push_back(std::move(seal));
    cp_control_pending[1].store(true, std::memory_order_release);
    submit_cv.notify_all();
    return request;
}

bool Liverpool::PauseAscForGraphicsControl(const GraphicsControlHandle& request,
                                          u64 sealed_tick) {
    if (!CpSplitActive() || thread_context.owner != this || thread_context.qid <= 0 ||
        thread_context.token.generation == 0 || !request || !sealed_tick ||
        !rasterizer || !rasterizer->AscSchedulerIdleAfterSeal(sealed_tick)) return false;
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(CpLane::Asc);
    const bool paused = !cp_stopping.load(std::memory_order_acquire) &&
                        cp_coordination.SubmittedTick(CpLane::Asc) >= sealed_tick &&
                        asc_owner_pause.Begin(static_cast<u32>(thread_context.qid), request,
                                              sealed_tick);
    if (paused) submit_cv.notify_all();
    return paused;
}

bool Liverpool::PauseAscForGuestGraphicsControl(const GraphicsControlHandle& request,
                                               u64 sealed_tick) {
    if (!CpSplitActive() || (thread_context.owner == this && thread_context.qid != 0) ||
        !request || !sealed_tick) return false;
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(CpLane::Asc);
    const bool paused = !cp_stopping.load(std::memory_order_acquire) &&
                        cp_coordination.SubmittedTick(CpLane::Asc) >= sealed_tick &&
                        asc_owner_pause.BeginGuest(request,sealed_tick);
    if (paused) submit_cv.notify_all();
    return paused;
}

bool Liverpool::AscOwnerPausedForGraphics() {
    std::scoped_lock lk{submit_mutex};
    return CpSplitActive() && asc_owner_pause.Acknowledged() &&
           cp_coordination.SubmittedTick(CpLane::Asc) >= asc_owner_pause.PrefixTick();
}

std::optional<u32> Liverpool::MapAsc(VAddr map_addr, u32* read_addr, u32 ring_size_dw,
                                    u32 pipe_id) {
    std::scoped_lock lk{submit_mutex};
    if (asc_queues.size() >= NumComputeRings) return std::nullopt;
    const auto slot = asc_queues.insert(map_addr, read_addr, ring_size_dw, pipe_id);
    const u32 qid = slot.index + 1;
    u64 generation{};
    if (qid >= NumTotalQueues || !g4_ledger.Map(qid, generation)) {
        asc_queues.erase(slot);
        return std::nullopt;
    }
    u64 coordinated_generation{};
    ASSERT_MSG(cp_coordination.Map(qid, coordinated_generation) &&
                   coordinated_generation == generation,
               "G4 coordination mapping diverged from admission ledger");
    mapped_queues[qid].packet_sequence = 0;
    RefreshCpControlPendingLocked(CpLane::Asc);
    return qid;
}

std::optional<Liverpool::AscQueueSnapshot> Liverpool::DescribeAsc(u32 qid) {
    std::scoped_lock lk{submit_mutex};
    if (qid == 0 || qid >= NumTotalQueues || !g4_ledger.CanIssue(qid) ||
        !asc_queues.is_allocated(Common::SlotId{qid - 1})) return std::nullopt;
    const auto& info = asc_queues[Common::SlotId{qid - 1}];
    return AscQueueSnapshot{info.map_addr, info.ring_size_dw, g4_ledger.Generation(qid)};
}

std::optional<u64> Liverpool::BeginUnmapAsc(u32 qid) {
    std::scoped_lock lk{submit_mutex};
    if (qid == 0 || qid >= NumTotalQueues ||
        !asc_queues.is_allocated(Common::SlotId{qid - 1})) return std::nullopt;
    u64 generation{};
    if (!g4_ledger.BeginClose(qid, generation)) return std::nullopt;
    u64 coordinated_generation{};
    ASSERT_MSG(cp_coordination.BeginClose(qid, coordinated_generation) &&
                   coordinated_generation == generation,
               "G4 coordination close diverged from admission ledger");
    return generation;
}

bool Liverpool::WaitAscDrained(u32 qid, u64 generation) {
    std::unique_lock lk{submit_mutex};
    submit_cv.wait(lk, [&] {
        return g4_ledger.Drained(qid, generation) ||
               process_thread.get_stop_token().stop_requested();
    });
    return g4_ledger.Drained(qid, generation);
}

bool Liverpool::FinishUnmapAsc(u32 qid, u64 generation) {
    std::scoped_lock lk{submit_mutex};
    if (!g4_ledger.Drained(qid, generation) ||
        !asc_queues.is_allocated(Common::SlotId{qid - 1})) return false;
    if (!g4_ledger.FinishClose(qid, generation)) return false;
    ASSERT_MSG(cp_coordination.FinishClose(qid, generation),
               "G4 coordination unmap diverged from admission ledger");
    RefreshCpControlPendingLocked(CpLane::Asc);
    asc_queues.erase(Common::SlotId{qid - 1});
    return true;
}

void Liverpool::AdvanceAscReadPointer(AscQueueInfo& queue, u32 dwords) {
    // Called only inside ProcessAscHostWrite's global logical lease.
    *queue.read_addr += dwords;
    *queue.read_addr %= queue.ring_size_dw;
}

void Liverpool::ProcessCommands() {
    // Legacy SendCommand callbacks may touch graphics cache state and are owned by G.
    if (CpSplitActive() && thread_context.owner == this &&
        thread_context.qid != GfxQueueId) return;
    // Process incoming commands with high priority
    while (num_commands) {
        if(rasterizer)rasterizer->DrainDrawPipe(1);
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        const auto previous = std::exchange(current_submission_token, {});
        const auto previous_context = thread_context;
        if (thread_context.owner == this) {
            thread_context.token = {};
        }
        try {
            callback();
        } catch (...) {
            current_submission_token = previous;
            thread_context = previous_context;
            throw;
        }
        current_submission_token = previous;
        thread_context = previous_context;
        ++processed_packets;
    }
}

bool Liverpool::BeginCpRecord(CpLane lane, u64 tick) {
    std::scoped_lock lk{submit_mutex};
    return CpSplitActive() && cp_timeline.Reconcile(lane) &&
           cp_coordination.BeginRecord(lane, tick);
}

bool Liverpool::EndCpRecord(CpLane lane, u64 tick, bool allow_submits) {
    std::scoped_lock lk{submit_mutex};
    ASSERT_MSG(allow_submits || cp_timeline.ObservedTick(lane) < tick,
               "G4 scheduler submitted a tick while its packet record was pending");
    const bool ended = cp_coordination.EndRecord(lane, tick);
    if (ended) ASSERT_MSG(cp_timeline.Reconcile(lane),
                          "G4 scheduler/coordinator timeline diverged after record");
    if (ended) submit_cv.notify_all();
    return ended;
}

bool Liverpool::AddCpDependency(CpTick consumer, QidTick producer) {
    std::scoped_lock lk{submit_mutex};
    return CpSplitActive() && cp_coordination.AddDependency(consumer, producer);
}

bool Liverpool::PublishCpSubmitted(CpLane lane, u64 through_tick) {
    std::scoped_lock lk{submit_mutex};
    if (!cp_split_requested || cp_stopping.load(std::memory_order_acquire) ||
        through_tick == 0) return false;
    // The post-submit callback may run on a recorder worker while an explicit
    // seal is in flight. Observe the real signal now; the coordinator publishes
    // its prefix after FinishSeal releases that pinned open tick.
    cp_timeline.ObserveSubmitted(lane, through_tick);
    RefreshCpControlPendingLocked(lane);
    submit_cv.notify_all();
    return true;
}

u64 Liverpool::CpOpenTick(CpLane lane) {
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(lane);
    return cp_coordination.OpenTick(lane);
}

u64 Liverpool::CpGraphicsGeneration() {
    std::scoped_lock lk{submit_mutex};
    return g4_ledger.Generation(GfxQueueId);
}

void Liverpool::CancelCpConsumer(CpTick consumer) {
    std::scoped_lock lk{submit_mutex};
    cp_coordination.CancelConsumer(consumer);
    submit_cv.notify_all();
}

Liverpool::SealRequest Liverpool::SealAtLeast(QidTick target) {
    std::scoped_lock lk{submit_mutex};
    // A disabled split has no separate owner whose scheduler prefix could be sealed here.
    // The serial backend must continue through its existing Flush/OnFence path.
    if (!CpSplitActive()) {
        auto failed = std::make_shared<iss::g4::SealRequest>(target);
        failed->Resolve(iss::g4::SealState::Failed);
        return failed;
    }
    cp_timeline.Reconcile(target.lane);
    auto request = cp_coordination.SealAtLeast(target);
    if (request->State() == iss::g4::SealState::Pending) {
        const size_t lane_index = target.lane == CpLane::Graphics ? 0 : 1;
        cp_seal_requests[lane_index].push_back(request);
        cp_control_pending[lane_index] = true;
    }
    submit_cv.notify_all();
    return request;
}

Liverpool::SealRequest Liverpool::SealDependency(CpTick consumer, QidTick producer) {
    std::scoped_lock lk{submit_mutex};
    cp_timeline.Reconcile(consumer.lane);
    cp_timeline.Reconcile(producer.lane);
    if (!CpSplitActive() || !cp_coordination.AddDependency(consumer, producer)) {
        auto failed = std::make_shared<iss::g4::SealRequest>(producer);
        failed->Resolve(iss::g4::SealState::Failed);
        return failed;
    }
    auto request = cp_coordination.SealAtLeast(producer);
    if (request->State() == iss::g4::SealState::Pending) {
        const size_t lane_index = producer.lane == CpLane::Graphics ? 0 : 1;
        cp_seal_requests[lane_index].push_back(request);
        cp_control_pending[lane_index].store(true, std::memory_order_release);
    }
    submit_cv.notify_all();
    return request;
}

iss::g4::SealState Liverpool::WaitCpSeal(const SealRequest& request, CpLane caller) {
    if (!request) return iss::g4::SealState::Failed;
    while (request->State() == iss::g4::SealState::Pending) {
        if (thread_context.owner == this &&
            (thread_context.qid == 0) == (caller == CpLane::Graphics))
            ServiceCpSeals(caller);
        std::unique_lock lk{submit_mutex};
        submit_cv.wait_for(lk, std::chrono::milliseconds{1}, [&] {
            return request->State() != iss::g4::SealState::Pending ||
                   process_thread.get_stop_token().stop_requested();
        });
        if (process_thread.get_stop_token().stop_requested()) return iss::g4::SealState::Cancelled;
    }
    return request->State();
}

void Liverpool::ServiceCpSeals(CpLane lane) {
    if (!CpSplitActive() || !rasterizer) return;
    std::optional<CpCoordination::SealPlan> plan;
    {
        std::scoped_lock lk{submit_mutex};
        cp_timeline.Reconcile(lane);
        RefreshCpControlPendingLocked(lane);
        plan = cp_coordination.PlanSeal(lane, false);
    }
    if (!plan) return;
    // This call is restricted to an owner safe point. It must never enter PM4 processing,
    // guest reads, resource preparation, or a cache wait. A failed backend seal fails the
    // request; it is never acknowledged as a submitted tick.
    const bool submitted = rasterizer->SealCpPrefix(lane, plan->through_tick);
    {
        std::scoped_lock lk{submit_mutex};
        const bool finished = cp_coordination.FinishSeal(*plan, submitted);
        ASSERT_MSG(!submitted || finished, "G4 seal completion lost its owner tick");
        if (submitted) ASSERT_MSG(cp_timeline.Reconcile(lane),
                                  "G4 submitted scheduler tick did not reconcile after seal");
        RefreshCpControlPendingLocked(lane);
    }
    submit_cv.notify_all();
}

void Liverpool::ServiceGraphicsFallbacks() {
    if (!CpSplitActive() || fallback_pending.load(std::memory_order_acquire) == 0) return;
    // Only the G process loop calls this method, between coroutine resumes. A fallback
    // never runs from ProcessCommands or from inside suspended G packet scratch.
    auto request = fallback_mailbox.TakeAtGraphicsSafePoint(true);
    if (!request) return;
    fallback_pending.fetch_sub(1, std::memory_order_acq_rel);
    const auto& packet = request->Packet();
    const auto& guest = packet.guest;
    u64 current_generation{};
    {
        std::scoped_lock lk{submit_mutex};
        if (guest.qid < NumTotalQueues && g4_ledger.Mapped(guest.qid))
            current_generation = g4_ledger.Generation(guest.qid);
    }
    if (!graphics_fallback_recorder || current_generation != guest.generation ||
        packet.Gate() != iss::g4::FallbackGate::Ready) {
        fallback_mailbox.Fail(request);
        submit_cv.notify_all();
        return;
    }

    // The recorder must consume only immutable packet fields and already prepared resources.
    // It may publish metadata under the SRT lease, but must release those locks before Vulkan
    // recording. The proof is checked before any G seal can be acknowledged.
    iss::g4::RecordingOwnerScope recording_owner{
        {CpLane::Graphics, guest.qid, guest.generation}};
    auto proof = graphics_fallback_recorder(packet, request->AscPrefix());
    if (!proof || !fallback_mailbox.MarkGraphicsRecorded(request, current_generation, *proof)) {
        fallback_mailbox.Fail(request);
        submit_cv.notify_all();
        return;
    }
    u64 graphics_generation{};
    {
        std::scoped_lock lk{submit_mutex};
        graphics_generation = g4_ledger.Generation(GfxQueueId);
    }
    const QidTick gfx_token{CpLane::Graphics, GfxQueueId, graphics_generation,
                            proof->graphics_tick};
    auto seal = SealAtLeast(gfx_token);
    if (WaitCpSeal(seal, CpLane::Graphics) != iss::g4::SealState::Done) {
        fallback_mailbox.Fail(request);
        submit_cv.notify_all();
        return;
    }
    {
        std::scoped_lock lk{submit_mutex};
        current_generation = g4_ledger.Generation(guest.qid);
    }
    if (!fallback_mailbox.PublishGraphicsSeal(request, *seal, current_generation))
        fallback_mailbox.Fail(request);
    submit_cv.notify_all();
}

void Liverpool::ServiceGraphicsControls() {
    if (!CpSplitActive()) return;
    {
        std::scoped_lock lk{submit_mutex};
        if (guest_pause_control &&
            guest_pause_control->Status() !=
                iss::g4::GraphicsControlMailbox::Control::State::Queued &&
            guest_pause_control->Status() !=
                iss::g4::GraphicsControlMailbox::Control::State::Running) {
            guest_pause_control.reset();
            guest_pause_seal.reset();
        }
        if (asc_owner_pause.Next().gate == iss::g4::AscOwnerPause::Gate::Waiting &&
            !asc_owner_pause.Acknowledged()) {
            if (guest_pause_seal &&
                (guest_pause_seal->State() == iss::g4::SealState::Failed ||
                 guest_pause_seal->State() == iss::g4::SealState::Cancelled)) {
                ASSERT_MSG(graphics_controls.CancelPending(guest_pause_control),
                           "G4 failed guest prefix left an uncancellable G control");
                guest_pause_control.reset();
                guest_pause_seal.reset();
                submit_cv.notify_all();
            }
            return;
        }
    }
    if (graphics_controls.ServeOne(true)) submit_cv.notify_all();
}

void Liverpool::RefreshCpControlPendingLocked(CpLane lane) {
    const size_t lane_index = lane == CpLane::Graphics ? 0 : 1;
    std::erase_if(cp_seal_requests[lane_index], [](const SealRequest& request) {
        return request->State() != iss::g4::SealState::Pending;
    });
    cp_control_pending[lane_index].store(!cp_seal_requests[lane_index].empty(),
                                         std::memory_order_release);
}

bool Liverpool::OnlyGraphicsSubmitsPending() {
    std::scoped_lock lk{submit_mutex};
    const u64 gfx_pending = g4_ledger.Issued(GfxQueueId) -
                            g4_ledger.Completed(GfxQueueId);
    return num_submits.load(std::memory_order_relaxed) == gfx_pending;
}

bool Liverpool::GraphicsSubmitsPending() {
    std::scoped_lock lk{submit_mutex};
    return g4_ledger.Issued(GfxQueueId) != g4_ledger.Completed(GfxQueueId);
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    ::IssPinCommandProcessor();
    thread_context = CpThreadContext{this, GfxQueueId, {}};
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid.store(gettid(), std::memory_order_release);
#endif

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return num_commands || num_submits || submit_done ||
                       graphics_controls.HasPending() ||
                       fallback_pending.load(std::memory_order_acquire) != 0 ||
                       cp_control_pending[0].load(std::memory_order_acquire);
            });
        }
        if (stoken.stop_requested()) {
            break;
        }
        if (!cp_split_decided.load(std::memory_order_acquire)) DecideCpSplit();
        // Awake time is split between processing packets and spinning on unsatisfied waits
        // (WAIT_REG_MEM, semaphores...). It is accumulated locally and published at least every
        // millisecond, as the thread can stay awake for many frames.
        u64 time_mark = Common::Perf::NowNs();
        u64 packets_mark = processed_packets;
        u64 work_ns = 0;
        u64 spin_ns = 0;
        const auto account_time = [&](bool publish) {
            const u64 now = Common::Perf::NowNs();
            // Sleeping on a flip label inside a packet is waiting, not work.
            const u64 blocked = std::min(blocked_ns, now - time_mark);
            blocked_ns = 0;
            spin_ns += blocked;
            (processed_packets != packets_mark ? work_ns : spin_ns) += now - time_mark - blocked;
            time_mark = now;
            packets_mark = processed_packets;
            if (publish || work_ns + spin_ns >= 1'000'000) {
                Common::Perf::Add(Common::Perf::GpuBusyNs, work_ns);
                Common::Perf::Add(Common::Perf::GpuSpinNs, spin_ns);
                work_ns = 0;
                spin_ns = 0;
            }
        };

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands ||
               graphics_controls.HasPending() ||
               fallback_pending.load(std::memory_order_acquire) != 0 ||
               cp_control_pending[0].load(std::memory_order_acquire)) {
            if (stoken.stop_requested()) break;
            ProcessCommands();
            ServiceCpSeals(CpLane::Graphics);
            ServiceGraphicsControls();
            ServiceGraphicsFallbacks();
            if (rasterizer) {
                // Readbacks the GPU has finished are written as soon as possible, so their pages
                // are readable before the guest touches them.
                rasterizer->RetireReadbacks();
                // ISS_UNIFIED_MEMORY: deferred fences waiting in the current command buffer.
                if (iss::unified::enabled) rasterizer->FlushUnifiedFences(false);
            }

            if (CpSplitActive() && !GraphicsSubmitsPending()) {
                // ASC owns its coroutines now. Wait for G work or C completion without
                // repeatedly polling empty queue slots or stealing a C coroutine.
                std::unique_lock lk{submit_mutex};
                submit_cv.wait_for(lk, std::chrono::milliseconds{1}, [&] {
                    return stoken.stop_requested() || num_commands ||
                           graphics_controls.HasPending() ||
                           fallback_pending.load(std::memory_order_acquire) != 0 ||
                           (num_submits == 0 &&
                            !graphics_controls.HasPending() &&
                            fallback_pending.load(std::memory_order_acquire) == 0 &&
                            !cp_control_pending[0].load(std::memory_order_acquire)) ||
                           g4_ledger.Issued(GfxQueueId) != g4_ledger.Completed(GfxQueueId);
                });
                if (stoken.stop_requested()) break;
                continue;
            }
            curr_qid = CpSplitActive()
                           ? GfxQueueId
                           : (curr_qid + 1) % num_mapped_queues.load(std::memory_order_acquire);

            auto& queue = mapped_queues[curr_qid];

            GpuQueue::Submission submission{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                submission = queue.submits.front();
            }
            const auto task = submission.handle;
            current_submission_token = submission.token;
            const auto previous_context = std::exchange(
                thread_context, CpThreadContext{this, curr_qid, submission.token});
            if (curr_qid != GfxQueueId) {
                const u64 resume_start = Common::Perf::NowNs();
                if (CpSplitActive()) {
                    iss::g4::RecordingOwnerScope recording_owner{
                        {CpLane::Asc, static_cast<u32>(curr_qid),
                         submission.token.generation}};
                    task.resume();
                } else {
                    task.resume();
                }
                Common::Perf::Add(Common::Perf::AscBusyNs, Common::Perf::NowNs() - resume_start);
            } else {
                if (CpSplitActive()) {
                    iss::g4::RecordingOwnerScope recording_owner{
                        {CpLane::Graphics, GfxQueueId, submission.token.generation}};
                    task.resume();
                } else {
                    task.resume();
                }
            }
            thread_context = previous_context;
            account_time(false);
            current_submission_token = {};

            if (task.done()) {
                iss::timeline::Emit("cp_complete",reinterpret_cast<u64>(this),
                    submission.token.sequence,submission.token.qid,submission.token.generation,
                    reinterpret_cast<u64>(task.address()));
                // Optional with ISS_DRAW_PIPE_LAZY: queued packets own copies of what they read
                // from the task; OnSubmit/Flush/GpuIdle below still drain.
                if(rasterizer)rasterizer->DrainDrawPipe(2|0x100);
                task.destroy();

                {
                    std::unique_lock submit_lock{submit_mutex};
                    std::unique_lock queue_lock{queue.m_access};
                    ASSERT_MSG(!queue.submits.empty() &&
                               queue.submits.front().token == submission.token,
                               "G4 stage-1 submission front changed");
                    queue.submits.pop();
                    ASSERT_MSG(g4_ledger.Complete(submission.token) &&
                                   cp_coordination.Complete(submission.token),
                               "G4 stage-1 completion not FIFO or stale");
                    if (curr_qid != GfxQueueId) {
                        asc_submits.fetch_sub(1, std::memory_order_relaxed);
                    }
                    --num_submits;
                    submits_completed.fetch_add(1, std::memory_order_release);
                }
                submit_cv.notify_all();
            }
        }

        if (stoken.stop_requested()) break;
        // Nothing left to process: the guest may be waiting for a deferred fence.
        if (iss::unified::enabled && rasterizer) rasterizer->FlushUnifiedFences(true);

        if (submit_done) {
            if(rasterizer)rasterizer->DrainDrawPipe(2);
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
            ++processed_packets;
        }
        account_time(true);
        Common::Perf::GpuThreadCpu().store(Common::Perf::CurrentCpu(), std::memory_order_relaxed);

        if(rasterizer)rasterizer->DrainDrawPipeForIdle();
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

void Liverpool::ProcessAsc(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:AscCP");
    thread_context = CpThreadContext{this, 1, {}};
#ifdef __linux__
    asc_gpu_tid.store(gettid(), std::memory_order_release);
#endif
    u32 next_qid = 1;
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken, [this] {
                return CpSplitActive() &&
                       (asc_submits.load(std::memory_order_acquire) != 0 ||
                        asc_owner_pause.Active() ||
                        cp_control_pending[1].load(std::memory_order_acquire));
            });
        }
        if (stoken.stop_requested()) break;
        ServiceCpSeals(CpLane::Asc);
        iss::g4::AscOwnerPause::Decision pause;
        {
            std::scoped_lock lk{submit_mutex};
            pause = asc_owner_pause.Next();
        }
        if (pause.gate == iss::g4::AscOwnerPause::Gate::Waiting) {
            // An owning G upload is pending: no ASC qid may advance PM4 or open
            // another C scheduler tick. C still services safe prefix seals.
            const bool idle = rasterizer &&
                rasterizer->AscSchedulerIdleAfterSeal(pause.prefix_tick);
            {
                std::scoped_lock lk{submit_mutex};
                [[maybe_unused]] const bool acknowledged = asc_owner_pause.AcknowledgePaused(
                    pause.prefix_tick, cp_coordination.SubmittedTick(CpLane::Asc), idle);
                submit_cv.notify_all();
            }
            std::unique_lock lk{submit_mutex};
            submit_cv.wait_for(lk, std::chrono::milliseconds{1});
            continue;
        }
        if (pause.gate == iss::g4::AscOwnerPause::Gate::ReleaseGuest) {
            std::scoped_lock lk{submit_mutex};
            ASSERT_MSG(asc_owner_pause.AcknowledgeGuestRelease(),
                       "G4 guest readback pause did not release");
            submit_cv.notify_all();
            continue;
        }
        if (asc_submits.load(std::memory_order_acquire) == 0) {
            std::unique_lock lk{submit_mutex};
            submit_cv.wait_for(lk, std::chrono::milliseconds{1});
            continue;
        }

        const u32 upper = num_mapped_queues.load(std::memory_order_acquire);
        const bool resume_origin = pause.gate == iss::g4::AscOwnerPause::Gate::ResumeOrigin;
        bool resumed = false;
        for (u32 inspected = 0; inspected < (resume_origin ? 1u : upper - 1); ++inspected) {
            if (stoken.stop_requested()) break;
            if (!resume_origin && next_qid >= upper) next_qid = 1;
            const u32 qid = resume_origin ? pause.qid : next_qid++;
            auto& queue = mapped_queues[qid];
            GpuQueue::Submission submission{};
            {
                std::scoped_lock lk{queue.m_access};
                if (queue.submits.empty()) {
                    ASSERT_MSG(!resume_origin,
                               "G4 ASC upload control lost its originating submission");
                    continue;
                }
                submission = queue.submits.front();
            }
            resumed = true;
            if (resume_origin) {
                // The origin is chosen before releasing the pause. Its coroutine
                // may enqueue another owning control during this same resume.
                std::scoped_lock lk{submit_mutex};
                ASSERT_MSG(asc_owner_pause.AcknowledgeOriginResume(qid),
                           "G4 ASC upload control did not resume its originating qid first");
            }
            const auto task = submission.handle;
            const auto previous_context = std::exchange(
                thread_context, CpThreadContext{this, static_cast<s32>(qid), submission.token});
            const u64 resume_start = Common::Perf::NowNs();
            {
                iss::g4::RecordingOwnerScope recording_owner{
                    {CpLane::Asc, qid, submission.token.generation}};
                task.resume();
            }
            Common::Perf::Add(Common::Perf::AscBusyNs, Common::Perf::NowNs() - resume_start);
            thread_context = previous_context;
            ServiceCpSeals(CpLane::Asc);
            if (!task.done()) break; // round robin after each yielded ASC prefix
            task.destroy();
            {
                std::unique_lock submit_lock{submit_mutex};
                std::unique_lock queue_lock{queue.m_access};
                ASSERT_MSG(!queue.submits.empty() &&
                               queue.submits.front().token == submission.token,
                           "G4 ASC submission front changed");
                queue.submits.pop();
                ASSERT_MSG(g4_ledger.Complete(submission.token) &&
                               cp_coordination.Complete(submission.token),
                           "G4 ASC completion not FIFO or stale");
                asc_submits.fetch_sub(1, std::memory_order_release);
                --num_submits;
                submits_completed.fetch_add(1, std::memory_order_release);
            }
            submit_cv.notify_all();
            break;
        }
        if (!resumed) {
            std::unique_lock lk{submit_mutex};
            submit_cv.wait_for(lk, std::chrono::milliseconds{1});
        }
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();
        ++processed_packets;

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            if(rasterizer)rasterizer->DrainDrawPipe(0);
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            const VAddr dst = dump_const->Address<VAddr>();
            const u32 size = dump_const->Size();
            const u32 offset = dump_const->Offset();
            if (CpSplitActive() && size != 0) {
                std::vector<u8> bytes(size);
                std::memcpy(bytes.data(), cblock.constants_heap.data() + offset, size);
                const auto write = [dst, bytes = std::move(bytes)] {
                    std::memcpy(reinterpret_cast<void*>(dst), bytes.data(), bytes.size());
                };
                RUN_CE_HOST(dst, size, write);
            } else {
                if (rasterizer) rasterizer->BeforeGpuThreadWrite(dst, size);
                memcpy(reinterpret_cast<void*>(dst), cblock.constants_heap.data() + offset,
                       size);
                if (rasterizer) rasterizer->OnGpuThreadWrite(dst, size);
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            task.handle.destroy();
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

// ISS_SEGMENT_CENSUS=1: how much register state each command buffer segment (a submitted
// command buffer or an indirect buffer) inherits from the segments before it: the registers its
// draws see that it did not write itself. Segments recorded by different guest threads that
// inherit nothing could be translated in parallel.
struct SegmentCensus {
    std::bitset<0x400> ctx_written, sh_written;
    bool drew{};
    u32 draws{};
};
static bool SegmentCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SEGMENT_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
static SegmentCensus* g_segment;
static struct {
    u64 segments{}, drawing_segments{}, draws{}, self_contained{}, inherit_sum{};
    u64 key_inherited[14]{};
    u64 draws_hist[6]{}; // 1, 2-8, 9-32, 33-128, 129-512, >512
    u64 seg_ends{};
} g_seg_stats;
static constexpr std::array<std::pair<u32, const char*>, 14> SegmentKeyRegs{{
    {0x318, "cb0_base"},   {0x010, "db_z_info"},     {0x200, "db_depth_ctl"},
    {0x1E0, "blend0"},     {0x205, "su_sc_mode"},    {0x08E, "cb_target_mask"},
    {0x10F, "vport_xscale"}, {0x202, "cb_color_ctl"}, {0x00C, "screen_scissor"},
    {0x191, "ps_input0"},  {0x000, "db_render_ctl"}, {0x1B3, "spi_ps_in_ena"},
    {0x1008, "sh_ps_pgm"}, {0x1048, "sh_vs_pgm"},
}};
static void NoteSegmentWrite(u32 reg_addr, u32 words) {
    if (!g_segment) {
        return;
    }
    for (u32 i = 0; i < words; ++i) {
        const u32 r = reg_addr + i;
        if (r >= Regs::ContextRegWordOffset && r < Regs::ContextRegWordOffset + 0x400) {
            g_segment->ctx_written.set(r - Regs::ContextRegWordOffset);
        } else if (r >= Regs::ShRegWordOffset && r < Regs::ShRegWordOffset + 0x400) {
            g_segment->sh_written.set(r - Regs::ShRegWordOffset);
        }
    }
}
static void NoteSegmentDraw(const Regs& regs) {
    if (!g_segment) {
        return;
    }
    auto& seg = *g_segment;
    ++seg.draws;
    if (seg.drew) {
        return;
    }
    seg.drew = true;
    auto& st = g_seg_stats;
    ++st.drawing_segments;
    u32 inherited = 0;
    for (u32 r = 0; r < 0x400; ++r) {
        inherited += !seg.ctx_written[r] && regs.reg_array[Regs::ContextRegWordOffset + r] != 0;
    }
    st.inherit_sum += inherited;
    bool any_key = false;
    for (u32 k = 0; k < SegmentKeyRegs.size(); ++k) {
        const u32 r = SegmentKeyRegs[k].first;
        const bool written = r >= 0x1000 ? seg.sh_written[r - 0x1000] : seg.ctx_written[r];
        if (!written) {
            ++st.key_inherited[k];
            any_key = true;
        }
    }
    st.self_contained += any_key ? 0 : 1;
}
static void EndSegment(const SegmentCensus& seg) {
    auto& st = g_seg_stats;
    ++st.segments;
    st.draws += seg.draws;
    if (seg.draws) {
        const u32 d = seg.draws;
        ++st.draws_hist[d == 1 ? 0 : d <= 8 ? 1 : d <= 32 ? 2 : d <= 128 ? 3 : d <= 512 ? 4 : 5];
    }
    if ((++st.seg_ends & 0x3FFF) != 0) {
        return;
    }
    std::fprintf(stderr,
                 "SEGMENT_CENSUS segments=%llu drawing=%llu draws=%llu self_contained=%llu "
                 "avg_inherited_nonzero_ctx=%.1f hist[1=%llu 2-8=%llu 9-32=%llu 33-128=%llu "
                 "129-512=%llu >512=%llu]\n",
                 (unsigned long long)st.segments, (unsigned long long)st.drawing_segments,
                 (unsigned long long)st.draws, (unsigned long long)st.self_contained,
                 st.drawing_segments ? double(st.inherit_sum) / st.drawing_segments : 0.0,
                 (unsigned long long)st.draws_hist[0], (unsigned long long)st.draws_hist[1],
                 (unsigned long long)st.draws_hist[2], (unsigned long long)st.draws_hist[3],
                 (unsigned long long)st.draws_hist[4], (unsigned long long)st.draws_hist[5]);
    std::string keys;
    for (u32 k = 0; k < SegmentKeyRegs.size(); ++k) {
        keys += fmt::format(" {}={}", SegmentKeyRegs[k].second, st.key_inherited[k]);
    }
    std::fprintf(stderr, "SEGMENT_CENSUS inherited_keys%s\n", keys.c_str());
}

// ISS_PASS_COST: the class of a graphics packet (fast_draw.inl PassCostTick): 1 draw,
// 2 dispatch, 3.. packets that write or wait on memory, 0 the rest.
static u32 PassCostClass(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti: return 1;
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect: return 2;
    case PM4ItOpcode::EventWrite: return 3;
    case PM4ItOpcode::EventWriteEos: return 4;
    case PM4ItOpcode::EventWriteEop: return 5;
    case PM4ItOpcode::WaitRegMem: return 6;
    case PM4ItOpcode::WriteData: return 7;
    case PM4ItOpcode::DmaData: return 8;
    case PM4ItOpcode::CopyData:
    case PM4ItOpcode::MemSemaphore: return 9;
    case PM4ItOpcode::AcquireMem: return 10;
    default: return 0;
    }
}

// ISS_NATIVE_SHADOW: the engine shadow batch whose range is being interpreted (command processor
// thread): its first `skip` draws were recorded natively, the others are drawn as usual and
// teach the rasterizer the pipelines of unknown render states.
static const iss::native_shadow::Batch* g_shadow_range{};
static u32 g_shadow_index{}, g_shadow_skip{};
// ISS_NATIVE_GBUFFER: the G-buffer range being interpreted (its draws from g_gbuffer_index on).
static const iss::native_gbuffer::Batch* g_gbuffer_range{};
static u32 g_gbuffer_index{};

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           u64 scout_seq) {
    FIBER_ENTER(dcb_task_name);
    iss::hybrid_profile::Segment hybrid_profile;
    iss::native_item::Match native_item_match{}; // owned by this coroutine/DCB only
    SegmentCensus segment_census{};
    if (SegmentCensusEnabled()) {
        g_segment = &segment_census;
    }
    const u64 segment_lo = reinterpret_cast<u64>(dcb.data());
    const u64 segment_hi = segment_lo + dcb.size_bytes();
    g_iss_dcb_lo = segment_lo;
    g_iss_dcb_hi = segment_hi;
    iss::timeline::Emit("cp_gfx_begin",reinterpret_cast<u64>(this),thread_context.token.sequence,
        thread_context.token.qid,thread_context.token.generation,
        reinterpret_cast<u64>(dcb.data()),dcb.size_bytes());

    cblock.Reset();
    // ISS_NATIVE_RECORDER: the registers this command buffer starts from, for the engine thread
    // that writes it next frame.
    bool published_technique_start = false;
    if (iss::native_gbuffer::RecorderEnabled()) {
        iss::native_gbuffer::PublishSpanStart(dcb.data(), regs.reg_array.data(),
                                              regs.index_buffer_type.raw,
                                              regs.num_instances.num_instances);
    }
    // ISS_NATIVE_PASSES: the same for the engine lanes that start at this address.
    const bool native_passes = iss::native_passes::Enabled() && rasterizer;
    if (IssPassCostEnabled()) IssPassCostTick(11); // the idle time before this command buffer
    if (native_passes) {
        rasterizer->NativePassesSetup();
        iss::native_passes::PublishSpanStart(dcb.data(), regs.reg_array.data());
        // Submit mode: its segments are recorded (they are found by their draws below).
        iss::native_passes::WaitRecorded(dcb.data());
    }

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
        rasterizer->DumpNote("gfx", fmt::format("commands {:#x} {} dwords", base_addr, dcb.size()));
    }
    if (scout_seq != 0 && scout) {
        // The scout read this submission ahead (or is reading it): its entries, draw by draw.
        scout_submission = scout->Begin(scout_seq, regs);
    }
    // bind_prefetch: the commands a few draws ahead are prefetched (the hardware prefetchers do
    // not cross into the next 4 KB page, where each page of commands missed the caches).
    const bool prefetch_commands =
        Common::Perf::GetTuning().bind_prefetch.load(std::memory_order_relaxed);
    static const size_t CommandPrefetchDwords = [] {
        const char* value = std::getenv("ISS_PM4_PREFETCH_DW");
        const long dwords = value ? std::strtol(value, nullptr, 10) : 128;
        return static_cast<size_t>(std::clamp<long>(dwords, 16, 4096));
    }();
    const bool pass_cost = IssPassCostEnabled();
    u32 pass_cost_class = 12; // class of the packet processed since the last tick (12: setup)
    while (!dcb.empty()) {
        if (pass_cost) {
            IssPassCostTick(pass_cost_class);
            pass_cost_class = 0;
        }
        ProcessCommands();
        ++processed_packets;
        if (prefetch_commands && dcb.size() > CommandPrefetchDwords) {
            VideoCore::PrefetchLine(dcb.data() + CommandPrefetchDwords);
        }

        // The PS4 runs asynchronous compute queues alongside the graphics queue: the guest may
        // read their results (particle counts...) well before this queue would wait for them.
        // Give way to queued compute work regularly instead of only when blocked.
        if (const u32 yield_packets =
                Common::Perf::GetTuning().asc_yield_packets.load(std::memory_order_relaxed);
            yield_packets != 0 && asc_submits.load(std::memory_order_relaxed) != 0 &&
            ++gfx_packets_since_yield >= yield_packets) {
            gfx_packets_since_yield = 0;
            Common::Perf::Add(Common::Perf::GfxYields);
            YIELD_GFX();
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        // ISS_NATIVE_PASSES_JOB_WAIT: the job recording this packet is done before it is read.
        if (native_passes) iss::native_passes::WaitRecordedAt(dcb.data());

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            if (pass_cost) {
                pass_cost_class = PassCostClass(opcode);
            }
            const bool pipe_deferred = rasterizer && rasterizer->BeforeDrawPipePacket(
                u32(opcode), reinterpret_cast<const u32*>(header), count + 1);
            if (iss::command_trace::enabled) {
                iss::command_trace::Packet(CurrentQueue(), u32(opcode),
                                          reinterpret_cast<const u32*>(header),
                                          std::min<size_t>(count + 1, dcb.size()));
            }
            // ISS_FULL_PATH_STATS=1 (diagnosis): draw and dispatch packets the command processor
            // reaches (not jumped over inside a native segment), by opcode, every 600 frames.
            {
                static const bool full_path_stats = [] {
                    const char* p = std::getenv("ISS_FULL_PATH_STATS");
                    return p && p[0] == '1';
                }();
                if (full_path_stats) {
                    static std::array<u64, 256> reached{};
                    static u64 last_frame = 0;
                    ++reached[u32(opcode) & 0xFF];
                    const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
                    if (frame >= last_frame + 600) {
                        const double frames = double(frame - last_frame);
                        std::string text;
                        for (const u32 op : {0x15u, 0x16u, 0x24u, 0x25u, 0x27u, 0x2Du, 0x30u, 0x35u, 0x38u, 0x2Cu})
                            text += fmt::format(" {:02x}={:.1f}", op, double(reached[op]) / frames);
                        std::fprintf(stderr, "FULL_PATH_STATS per_frame[%s ]\n", text.c_str());
                        reached.fill(0);
                        last_frame = frame;
                    }
                }
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }
                // An engine shadow batch (native_shadow.h): recorded from the batch and jumped
                // over, or interpreted (the rasterizer learns its render states).
                if (nop->header.count.Value() == iss::native_shadow::MarkerDwords - 2 &&
                    (nop->data_block[0] & 0xFFFFFF00u) == iss::native_shadow::MarkerMagic) {
                    namespace ns = iss::native_shadow;
                    g_shadow_range = nullptr;
                    if (!rasterizer) break;
                    rasterizer->NativeShadowLearnEnd();
                    const auto* marker = reinterpret_cast<const u32*>(header);
                    const ns::Batch* batch = ns::Find(marker);
                    // Pause guard: in the pause the batch is interpreted (original engine path).
                    if (!batch || batch->end > dcb.data() + dcb.size() || CpSplitActive() ||
                        iss::pause_guard::Blocked(iss::pause_guard::Shadows)) {
                        ns::NoteRange(ns::Mismatch, 0);
                        break;
                    }
                    const u32 done = rasterizer->NativeShadowExecute(*batch);
                    if (done == batch->draws.size()) {
                        // The registers as the range leaves them. The native recording already
                        // wrote some, so the change is counted unconditionally.
                        for (const auto& [reg, value] : batch->final_regs) {
                            regs.reg_array[reg] = value;
                        }
                        ++context_reg_changes;
                        if (batch->has_instances) {
                            regs.num_instances.num_instances = batch->final_instances;
                        }
                        if (batch->has_index_type) {
                            regs.index_buffer_type.raw = batch->final_index_type;
                        }
                        const auto& last = batch->draws.back();
                        regs.max_index_size = last.max_index;
                        regs.index_base_address.base_addr_lo = last.index_lo;
                        regs.index_base_address.base_addr_hi = last.index_hi;
                        regs.num_indices = last.index_count;
                        regs.draw_initiator = last.initiator;
                        for (const auto& d : batch->draws) {
                            BeginScoutDraw(d.packet);
                            EndScoutDraw();
                        }
                        processed_packets += batch->draws.size();
                        dcb = dcb.subspan(batch->end - dcb.data());
                        continue;
                    }
                    g_shadow_range = batch;
                    g_shadow_index = 0;
                    g_shadow_skip = done;
                    if (done && SkipShadowPrefixEnabled()) {
                        CommitShadowDraw(regs, *batch, done - 1);
                        ++context_reg_changes;
                        for (u32 i = 0; i < done; ++i) {
                            BeginScoutDraw(batch->draws[i].packet); EndScoutDraw();
                        }
                        processed_packets += done;
                        g_shadow_index = done;
                        rasterizer->NativeShadowLearnBegin(*batch);
                        const auto* after = batch->draws[done - 1].packet + 6;
                        dcb = dcb.subspan(after - dcb.data());
                        continue;
                    }
                    rasterizer->NativeShadowLearnBegin(*batch);
                    break;
                }
                // An engine G-buffer range (native_gbuffer.h): recorded from the range and jumped
                // over, or interpreted from the first draw the native path did not take.
                if (nop->header.count.Value() == iss::native_gbuffer::MarkerDwords - 2 &&
                    (nop->data_block[0] & 0xFFFFFF00u) == iss::native_gbuffer::MarkerMagic) {
                    namespace ng = iss::native_gbuffer;
                    if (rasterizer && g_gbuffer_range) rasterizer->NativeGbufferRangeEnd(g_gbuffer_range);
                    g_gbuffer_range = nullptr;
                    if (!rasterizer) break;
                    const auto* marker = reinterpret_cast<const u32*>(header);
                    const ng::Batch* batch = ng::Find(marker);
                    // Pause guard: in the pause the range is interpreted (original engine path).
                    if (!batch || batch->end > dcb.data() + dcb.size() || CpSplitActive() ||
                        iss::pause_guard::Blocked(iss::pause_guard::GBuffer)) {
                        ng::NoteRange(ng::Mismatch, 0);
                        break;
                    }
                    g_census_lo = marker;
                    g_census_hi = batch->end;
                    // ISS_NATIVE_RECORDER_TECH_START: the registers at this command buffer's
                    // first range, for the engine thread that starts over with its technique.
                    if (!published_technique_start && ng::TechStartEnabled()) {
                        published_technique_start = true;
                        ng::PublishTechniqueStart(batch->technique, regs.reg_array.data(),
                                                  regs.index_buffer_type.raw,
                                                  regs.num_instances.num_instances);
                    }
                    rasterizer->native_span_end = dcb.data() + dcb.size();
                    rasterizer->native_stretch.clear();
                    const u32 done = rasterizer->NativeGbufferExecute(
                        *batch, [this](const u32* packet) { BeginScoutDraw(packet); },
                        [this] { EndScoutDraw(); });
                    if (done == batch->draws.size()) {
                        for (const auto& [reg, value] : batch->final_regs) {
                            regs.reg_array[reg] = value;
                        }
                        ++context_reg_changes;
                        if (batch->has_instances) {
                            regs.num_instances.num_instances = batch->final_instances;
                        }
                        if (batch->has_index_type) {
                            regs.index_buffer_type.raw = batch->final_index_type;
                        }
                        const auto& last = batch->draws.back();
                        regs.max_index_size = last.max_index;
                        regs.index_base_address.base_addr_lo = last.index_lo;
                        regs.index_base_address.base_addr_hi = last.index_hi;
                        regs.num_indices = last.index_count;
                        regs.draw_initiator = last.initiator;
                        processed_packets += batch->draws.size();
                        rasterizer->NativeGbufferRangeEnd(batch);
                        // ISS_NATIVE_RECORDER chains: the plain register writes up to each
                        // chained range, its registers, then on after the last.
                        const u32* resume = batch->end;
                        for (const auto* member : rasterizer->native_stretch) {
                            ApplyPlainRegisterWrites(regs, resume, member->marker);
                            ++context_reg_changes;
                            CommitNativeBatch(regs, *member);
                            processed_packets += member->draws.size();
                            rasterizer->NativeGbufferRangeEnd(member);
                            resume = member->end;
                        }
                        rasterizer->native_stretch.clear();
                        dcb = dcb.subspan(resume - dcb.data());
                        continue;
                    }
                    g_gbuffer_range = batch;
                    g_gbuffer_index = done;
                    if (done) {
                        CommitGbufferDraw(regs, *batch, done - 1);
                        ++context_reg_changes;
                        processed_packets += done;
                        const auto* after = batch->draws[done - 1].packet + 6;
                        dcb = dcb.subspan(after - dcb.data());
                        continue;
                    }
                    break;
                }
                // An engine draw marker (native/engine_draw.cpp): nine payload dwords.
                if ((nop->data_block[0] & 0xFFFFFF00u) == 0x49535300u &&
                    nop->header.count.Value() == 8) {
                    const u32 kind = nop->data_block[0] & 255;
                    if (iss::native_item::Enabled()) {
                        if (kind == 1 && iss::pause_guard::Blocked(iss::pause_guard::FastDraw)) {
                            native_item_match = {}; // pause guard: original engine path
                        } else if (kind == 1) {
                            native_item_match = iss::native_item::Scan(dcb, true);
                            iss::native_item::NoteScan(native_item_match.reject);
                        }
                        else if (kind != 4) native_item_match = {};
                    }
                    if (kind != 5) {
                        hybrid_profile.Marker(&nop->data_block[0]);
                        IssEngineMarker(&nop->data_block[0]);
                    }
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{
                        reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
                        rasterizer->DumpNote("marker+", label.substr(0, label.find('\0')));
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{
                        reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                    if (guest_markers_enabled) {
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
                        rasterizer->DumpNote("marker*", label.substr(0, label.find('\0')));
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
                        rasterizer->DumpNote("marker-", "");
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                ++context_reg_changes;
                num_changed_regs = static_cast<u32>(changed_regs.size()) + 1;
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                if (std::memcmp(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32)) !=
                    0) {
                    ++context_reg_changes;
                    if (note_changed_regs) {
                        NoteChangedRegs(reg_addr, &regs.reg_array[reg_addr], payload, count - 1);
                    }
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                if (std::memcmp(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32)) !=
                    0) {
                    ++context_reg_changes;
                    if (note_changed_regs) {
                        NoteChangedRegs(reg_addr, &regs.reg_array[reg_addr], payload, count - 1);
                    }
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                NoteSegmentWrite(reg_addr, count - 1);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    NoteSegmentWrite(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                if (std::memcmp(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                                header + 2, (count - 1) * sizeof(u32)) != 0) {
                    ++context_reg_changes;
                    if (note_changed_regs) {
                        NoteChangedRegs(Regs::UconfigRegWordOffset + set_data->reg_offset,
                                        &regs.reg_array[Regs::UconfigRegWordOffset +
                                                        set_data->reg_offset],
                                        reinterpret_cast<const u32*>(header + 2), count - 1);
                    }
                }
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                // ISS_NATIVE_PASSES: a segment the engine recorded from this draw on, executed
                // whole and jumped over, or a learn request for this draw.
                if (native_passes) {
                    namespace np = iss::native_passes;
                    if (np::Segment* seg = np::Claim(reinterpret_cast<const u32*>(header))) {
                        if (seg->kind == np::Kind::Learn) {
                            np::NoteDivergence(*seg, regs.reg_array.data());
                            rasterizer->native_pass_learn = seg->first_context;
                            rasterizer->native_pass_learn_mask = seg->mask;
                            np::Finish(*seg, false, 0);
                        } else {
                            u64 tick = 0;
                            const bool fits = seg->end > reinterpret_cast<const u32*>(header) &&
                                              seg->end <= dcb.data() + dcb.size();
                            if (fits && !CpSplitActive() && !iss::pause_guard::Blocked(iss::pause_guard::Passes) &&
                                rasterizer->NativePassExecute(
                                    *seg, [this](const u32* packet) { BeginScoutDraw(packet); },
                                    [this] { EndScoutDraw(); }, tick)) {
                                for (const auto& [reg, value] : seg->final_regs)
                                    regs.reg_array[reg] = value;
                                ++context_reg_changes;
                                if (seg->has_instances)
                                    regs.num_instances.num_instances = seg->final_instances;
                                if (seg->has_index_type)
                                    regs.index_buffer_type.raw = seg->final_index_type;
                                if (seg->has_last_direct) {
                                    regs.max_index_size = seg->last_max_index;
                                    regs.index_base_address.base_addr_lo = seg->last_index_lo;
                                    regs.index_base_address.base_addr_hi = seg->last_index_hi;
                                }
                                if (seg->has_indirect_base) indirect_args_addr = seg->final_indirect_base;
                                regs.num_indices = seg->last_index_count;
                                regs.draw_initiator = seg->last_initiator;
                                processed_packets += seg->draw_packets.size();
                                const u32* resume = seg->end;
                                np::Finish(*seg, true, tick);
                                dcb = dcb.subspan(resume - dcb.data());
                                continue;
                            }
                            np::Finish(*seg, false, 0);
                        }
                    }
                }
                SCOPE_EXIT { if (native_passes) rasterizer->native_pass_learn = 0; };
                const bool native_item_owned = native_item_match.draw == dcb.data();
                const auto native_item_identity = native_item_match.identity;
                native_item_match = {}; // exactly one use, even on fallback or no rasterizer
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(regs.num_indices, 0); };
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                NoteSegmentDraw(regs);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                // ISS_NATIVE_SHADOW: a draw of the batch range being interpreted.
                u32 shadow_draw = ~0u;
                if (g_shadow_range) {
                    const u32 i = g_shadow_index++;
                    if (i < g_shadow_range->draws.size() &&
                        g_shadow_range->draws[i].packet == reinterpret_cast<const u32*>(header)) {
                        shadow_draw = i;
                    } else {
                        g_shadow_range = nullptr;
                        if (rasterizer) rasterizer->NativeShadowLearnEnd();
                    }
                }
                // ISS_NATIVE_GBUFFER: an interpreted draw of the G-buffer range, noted after it
                // was recorded (the verified reference completes with the range's last draw).
                const iss::native_gbuffer::Batch* gbuffer_range = nullptr;
                bool gbuffer_last = false;
                bool census_lost = false;
                if (g_gbuffer_range) {
                    const u32 i = g_gbuffer_index++;
                    const auto* range = g_gbuffer_range;
                    if (i < range->draws.size() &&
                        range->draws[i].packet == reinterpret_cast<const u32*>(header)) {
                        gbuffer_range = range;
                        gbuffer_last = i + 1 == range->draws.size();
                        if (gbuffer_last) g_gbuffer_range = nullptr;
                    } else {
                        g_gbuffer_range = nullptr;
                        if (rasterizer) rasterizer->NativeGbufferRangeEnd(range);
                        census_lost = true;
                    }
                }
                SCOPE_EXIT {
                    if (rasterizer && gbuffer_range) {
                        rasterizer->NativeGbufferInterpretedDraw(*gbuffer_range, gbuffer_last);
                        if (gbuffer_last) rasterizer->NativeGbufferRangeEnd(gbuffer_range);
                    }
                };
                if (DrawOriginCensusEnabled()) {
                    const auto* at = reinterpret_cast<const u32*>(header);
                    NoteDrawOrigin(regs, shadow_draw != ~0u ? 1 : gbuffer_range ? 2
                                       : census_lost ? 3
                                       : at >= g_census_lo && at < g_census_hi ? 4 : 0);
                }
                const bool shadow_last =
                    shadow_draw != ~0u && shadow_draw + 1 == g_shadow_range->draws.size();
                const auto end_shadow_range = [&] {
                    if (!shadow_last) return;
                    g_shadow_range = nullptr;
                    if (rasterizer) rasterizer->NativeShadowLearnEnd();
                };
                if (rasterizer && shadow_draw != ~0u && shadow_draw > 0 && shadow_draw >= g_shadow_skip &&
                    !iss::pause_guard::Blocked(iss::pause_guard::Shadows)) {
                    const u32 done = Vulkan::ExecuteNativeShadowSuffix(*rasterizer, *g_shadow_range, shadow_draw);
                    if (done) {
                        g_shadow_skip = shadow_draw + done;
                        if (SkipShadowPrefixEnabled()) {
                            const auto* range = g_shadow_range;
                            const u32 last = shadow_draw + done - 1;
                            CommitShadowDraw(regs, *range, last);
                            ++context_reg_changes;
                            for (u32 i = shadow_draw; i <= last; ++i) {
                                BeginScoutDraw(range->draws[i].packet); EndScoutDraw();
                            }
                            processed_packets += done - 1; // current packet was counted already
                            g_shadow_index = last + 1;
                            const auto* after = range->draws[last].packet + 6;
                            if (last + 1 == range->draws.size()) {
                                g_shadow_range = nullptr;
                                rasterizer->NativeShadowLearnEnd();
                            }
                            dcb = dcb.subspan(after - dcb.data());
                            continue;
                        }
                    }
                }
                if (shadow_draw != ~0u && shadow_draw < g_shadow_skip) {
                    // Recorded natively from the batch.
                    BeginScoutDraw(header);
                    EndScoutDraw();
                    end_shadow_range();
                    break;
                }
                if (!rasterizer) {
                    end_shadow_range();
                    break;
                }
                if (shadow_draw != ~0u) rasterizer->NativeShadowBeforeDraw(shadow_draw);
                SCOPE_EXIT {
                    if (shadow_draw != ~0u) {
                        rasterizer->NativeShadowAfterDraw(shadow_draw);
                        end_shadow_range();
                    }
                };
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto record = [this, cmd_address, native_item_owned, native_item_identity] {
                    iss::native_item::Scope identity_scope(native_item_owned ? &native_item_identity : nullptr);
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                            [&] { rasterizer->Draw(true); });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else {
                    const auto following = dcb.subspan(count + 1);
                    const auto yield_limit = Common::Perf::GetTuning().asc_yield_packets.load(std::memory_order_relaxed);
                    // Never consume the packet at which the ordinary loop could yield to ASC,
                    // including when an ASC submission arrives while the group is prepared.
                    // ISS_NATIVE_DEPTH_RUN: a run may span more packets than the fairness slice
                    // while no ASC submission waits (the yield then follows the run).
                    const bool open_budget = yield_limit == 0 ||
                        (Vulkan::Rasterizer::NativeDepthRunEnabled() &&
                         asc_submits.load(std::memory_order_relaxed) == 0);
                    // Draws of a shadow batch range are taken one by one (learning).
                    const std::size_t packet_budget = iss::native_item::Enabled() || shadow_draw != ~0u ? 0 :
                        (open_budget ? 4096 :
                        (gfx_packets_since_yield < yield_limit ? yield_limit - gfx_packets_since_yield - 1 : 0));
                    std::size_t batched_words = 0;
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                        [&] {
                            iss::native_item::Scope identity_scope(native_item_owned ? &native_item_identity : nullptr);
                            batched_words = rasterizer->Draw(true, 0, following, packet_budget);
                        });
                    EndScoutDraw();
                    if (batched_words) {
                        // The CPU collector has already proved and recorded this bounded run.
                        // Advance scout ordinals for its extra draws, then resume interpretation
                        // at the first untouched packet. The final registers were committed by
                        // the collector only after the native draw was successfully recorded.
                        std::size_t cursor = 0;
                        while (cursor < batched_words) {
                            const auto* skipped = reinterpret_cast<const PM4Header*>(following.data() + cursor);
                            const auto words = skipped->type == 2 ? 1 : skipped->type3.NumWords() + 1;
                            if (skipped->type == 3 && skipped->type3.opcode == PM4ItOpcode::DrawIndex2) {
                                BeginScoutDraw(skipped);
                                EndScoutDraw();
                            }
                            cursor += words;
                            ++processed_packets;
                            if (yield_limit && asc_submits.load(std::memory_order_relaxed)) ++gfx_packets_since_yield;
                        }
                        ASSERT(cursor == batched_words);
                        dcb = NextPacket(dcb, batched_words);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(regs.num_indices, 0); };
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                NoteSegmentDraw(regs);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u32 index_offset = draw_index_off->index_offset;
                const auto record = [this, cmd_address, index_offset] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndexOffset2",
                                            fmt::make_format_args(cmd_address),
                                            [&] { rasterizer->Draw(true, index_offset); });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                // ISS_NATIVE_PASSES: as DRAW_INDEX_2 (non-indexed draws record natively too).
                if (native_passes) {
                    namespace np = iss::native_passes;
                    if (np::Segment* seg = np::Claim(reinterpret_cast<const u32*>(header))) {
                        if (seg->kind == np::Kind::Learn) {
                            np::NoteDivergence(*seg, regs.reg_array.data());
                            rasterizer->native_pass_learn = seg->first_context;
                            rasterizer->native_pass_learn_mask = seg->mask;
                            np::Finish(*seg, false, 0);
                        } else {
                            u64 tick = 0;
                            const bool fits = seg->end > reinterpret_cast<const u32*>(header) &&
                                              seg->end <= dcb.data() + dcb.size();
                            if (fits && !CpSplitActive() && !iss::pause_guard::Blocked(iss::pause_guard::Passes) &&
                                rasterizer->NativePassExecute(
                                    *seg, [this](const u32* packet) { BeginScoutDraw(packet); },
                                    [this] { EndScoutDraw(); }, tick)) {
                                for (const auto& [reg, value] : seg->final_regs)
                                    regs.reg_array[reg] = value;
                                ++context_reg_changes;
                                if (seg->has_instances)
                                    regs.num_instances.num_instances = seg->final_instances;
                                if (seg->has_index_type)
                                    regs.index_buffer_type.raw = seg->final_index_type;
                                if (seg->has_last_direct) {
                                    regs.max_index_size = seg->last_max_index;
                                    regs.index_base_address.base_addr_lo = seg->last_index_lo;
                                    regs.index_base_address.base_addr_hi = seg->last_index_hi;
                                }
                                if (seg->has_indirect_base) indirect_args_addr = seg->final_indirect_base;
                                regs.num_indices = seg->last_index_count;
                                regs.draw_initiator = seg->last_initiator;
                                processed_packets += seg->draw_packets.size();
                                const u32* resume = seg->end;
                                np::Finish(*seg, true, tick);
                                dcb = dcb.subspan(resume - dcb.data());
                                continue;
                            }
                            np::Finish(*seg, false, 0);
                        }
                    }
                }
                SCOPE_EXIT { if (native_passes) rasterizer->native_pass_learn = 0; };
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(regs.num_indices, 1); };
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                NoteSegmentDraw(regs);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto record = [this, cmd_address] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto",
                                            fmt::make_format_args(cmd_address),
                                            [&] { rasterizer->Draw(false); });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(~u32{0}, 2); };
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const u16 base_vtx = draw_indirect->base_vtx_loc;
                const u16 start_inst = draw_indirect->start_inst_loc;
                const auto record = [this, cmd_address, args_addr, offset, stride, base_vtx,
                                     start_inst] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndirect",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, args_addr, offset, stride, 1, 0,
                                                 base_vtx, start_inst);
                    });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(~u32{0}, 2); };
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const u32 packet_stride = draw_indirect->stride;
                const u32 packet_count = draw_indirect->count;
                const u16 base_vtx = draw_indirect->base_vtx_loc;
                const u16 start_inst = draw_indirect->start_inst_loc;
                const auto record = [this, cmd_address, args_addr, offset, packet_stride,
                                     packet_count, base_vtx, start_inst] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndirectMulti",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, args_addr, offset, packet_stride,
                                                 packet_count, 0, base_vtx, start_inst);
                    });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                // ISS_NATIVE_INDIRECT: a segment recorded from this draw on, or a learn request.
                if (native_passes && iss::native_passes::IndirectEnabled()) {
                    namespace np = iss::native_passes;
                    const u32* first = reinterpret_cast<const u32*>(header);
                    if (np::Segment* seg = np::Claim(first)) {
                        if (seg->kind == np::Kind::Learn) {
                            np::NoteDivergence(*seg, regs.reg_array.data());
                            rasterizer->native_pass_learn = seg->first_context;
                            rasterizer->native_pass_learn_mask = seg->mask;
                            rasterizer->native_pass_learn_indirect =
                                (1ull << 32) | (u64(first[2] & 0xFFFF) << 16) | (first[3] & 0xFFFF);
                            np::Finish(*seg, false, 0);
                        } else {
                            u64 tick = 0;
                            const bool fits = seg->end > first && seg->end <= dcb.data() + dcb.size();
                            if (fits && !CpSplitActive() && !iss::pause_guard::Blocked(iss::pause_guard::Passes) &&
                                rasterizer->NativePassExecute(
                                    *seg, [this](const u32* packet) { BeginScoutDraw(packet); },
                                    [this] { EndScoutDraw(); }, tick)) {
                                for (const auto& [reg, value] : seg->final_regs)
                                    regs.reg_array[reg] = value;
                                ++context_reg_changes;
                                if (seg->has_instances)
                                    regs.num_instances.num_instances = seg->final_instances;
                                if (seg->has_index_type)
                                    regs.index_buffer_type.raw = seg->final_index_type;
                                if (seg->has_last_direct) {
                                    regs.max_index_size = seg->last_max_index;
                                    regs.index_base_address.base_addr_lo = seg->last_index_lo;
                                    regs.index_base_address.base_addr_hi = seg->last_index_hi;
                                }
                                if (seg->has_indirect_base) indirect_args_addr = seg->final_indirect_base;
                                regs.num_indices = seg->last_index_count;
                                regs.draw_initiator = seg->last_initiator;
                                processed_packets += seg->draw_packets.size();
                                const u32* resume = seg->end;
                                np::Finish(*seg, true, tick);
                                dcb = dcb.subspan(resume - dcb.data());
                                continue;
                            }
                            np::Finish(*seg, false, 0);
                        }
                    }
                }
                SCOPE_EXIT {
                    if (native_passes) {
                        rasterizer->native_pass_learn = 0;
                        rasterizer->native_pass_learn_indirect = 0;
                    }
                };
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(~u32{0}, 2); };
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const u16 base_vtx = draw_index_indirect->base_vtx_loc;
                const u16 start_inst = draw_index_indirect->start_inst_loc;
                const auto record = [this, cmd_address, args_addr, offset, stride, base_vtx,
                                     start_inst] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndexIndirect",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, args_addr, offset, stride, 1, 0,
                                                 base_vtx, start_inst);
                    });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(~u32{0}, 2); };
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const u32 packet_stride = draw_index_indirect->stride;
                const u32 packet_count = draw_index_indirect->count;
                const u16 base_vtx = draw_index_indirect->base_vtx_loc;
                const u16 start_inst = draw_index_indirect->start_inst_loc;
                const auto record = [this, cmd_address, args_addr, offset, packet_stride,
                                     packet_count, base_vtx, start_inst] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndexIndirectMulti",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, args_addr, offset, packet_stride,
                                                 packet_count, 0, base_vtx, start_inst);
                    });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                hybrid_profile.BeginDraw();
                SCOPE_EXIT { hybrid_profile.EndDraw(~u32{0}, 2); };
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const u32 packet_stride = draw_index_indirect->stride;
                const u32 packet_count = draw_index_indirect->count;
                const u64 count_addr = draw_index_indirect->count_indirect_enable.Value()
                                           ? draw_index_indirect->count_addr : 0;
                const u16 base_vtx = draw_index_indirect->base_vtx_loc;
                const u16 start_inst = draw_index_indirect->start_inst_loc;
                const auto record = [this, cmd_address, args_addr, offset, packet_stride,
                                     packet_count, count_addr, base_vtx, start_inst] {
                    BeginScoutDraw(cmd_address);
                    rasterizer->ScopeMarker("gfx:{}:DrawIndexIndirectCountMulti",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, args_addr, offset, packet_stride,
                                                 packet_count, count_addr, base_vtx, start_inst);
                    });
                    EndScoutDraw();
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                // ISS_NATIVE_COMPUTE: a compute segment recorded at the submission from this
                // dispatch on, executed whole and jumped over, or a learn request for it.
                if (native_passes) {
                    namespace np = iss::native_passes;
                    if (np::Segment* seg = np::Claim(reinterpret_cast<const u32*>(header))) {
                        if (seg->kind == np::Kind::Learn) {
                            rasterizer->native_compute_learn = seg->first_context;
                            np::Finish(*seg, false, 0);
                        } else if (seg->kind == np::Kind::Compute) {
                            u64 tick = 0;
                            const bool fits = seg->end > reinterpret_cast<const u32*>(header) &&
                                              seg->end <= dcb.data() + dcb.size();
                            if (fits && !CpSplitActive() && !iss::pause_guard::Blocked(iss::pause_guard::Compute) &&
                                rasterizer->NativeComputeExecute(*seg, tick)) {
                                auto& cs = GetCsRegs();
                                auto* cs_words = reinterpret_cast<u32*>(&cs);
                                for (const auto& [reg, value] : seg->final_regs) {
                                    const u32 word = reg - np::CsBase;
                                    if (reg >= np::CsBase && word < sizeof(ComputeProgram) / 4)
                                        cs_words[word] = value;
                                    else
                                        regs.reg_array[reg] = value;
                                }
                                cs.dim_x = seg->last_dims[0];
                                cs.dim_y = seg->last_dims[1];
                                cs.dim_z = seg->last_dims[2];
                                cs.dispatch_initiator = seg->last_dispatch_initiator;
                                processed_packets += seg->draw_packets.size();
                                const u32* resume = seg->end;
                                np::Finish(*seg, true, tick);
                                dcb = dcb.subspan(resume - dcb.data());
                                continue;
                            }
                            np::Finish(*seg, false, 0);
                        } else {
                            np::Finish(*seg, false, 0);
                        }
                    }
                }
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    if (rasterizer && rasterizer->native_compute_learn) {
                        rasterizer->NativeComputeLearnFailed(rasterizer->native_compute_learn);
                        rasterizer->native_compute_learn = 0;
                    }
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto record = [this, cmd_address] {
                    rasterizer->ScopeMarker("gfx:{}:DispatchDirect",
                                            fmt::make_format_args(cmd_address),
                                            [&] { rasterizer->DispatchDirect(); });
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                if (rasterizer->native_compute_learn) {
                    // Not learned (an emulated or skipped dispatch, or its bindings failed).
                    rasterizer->NativeComputeLearnFailed(rasterizer->native_compute_learn);
                    rasterizer->native_compute_learn = 0;
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u64 args_addr = indirect_args_addr;
                const auto record = [this, cmd_address, args_addr, offset, size] {
                    rasterizer->ScopeMarker("gfx:{}:DispatchIndirect",
                                            fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DispatchIndirect(args_addr, offset, size);
                    });
                };
                if (CpSplitActive()) RUN_GFX_GPU(record);
                else record();
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        // Real occlusion queries: the GPU counts the samples of the draws between
                        // the begin and end dumps (light flares behind walls are not visible).
                        const VAddr address = event->Address<VAddr>();
                        auto result = Vulkan::OcclusionDumpResult::Emulated;
                        if (rasterizer) {
                            const u32 pairs = num_counter_pairs;
                            const auto record = [this, address, pairs, &result] {
                                result = rasterizer->OcclusionDump(address, pairs);
                            };
                            if (CpSplitActive()) RUN_GFX_GPU(record);
                            else record();
                        }
                        if (result == Vulkan::OcclusionDumpResult::EmulatedWithBegin) {
                            RUN_OCCLUSION_HOST(address - sizeof(u64));
                        }
                        if (result != Vulkan::OcclusionDumpResult::Counted) {
                            RUN_OCCLUSION_HOST(address);
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                if (pipe_deferred) {
                    break; // Queued to the draw pipe worker (RunDeferredFence).
                }
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                if (CpSplitActive() &&
                    event_eos->command == PM4CmdEventWriteEos::Command::SignalFence) {
                    const PM4CmdEventWriteEos packet = *event_eos;
                    const VAddr dst = reinterpret_cast<VAddr>(packet.Address());
                    const auto write = [packet] {
                        packet.SignalFence([](void* address, u64 data, u32 num_bytes) {
                            ASSERT(Core::Memory::Instance()->TryWriteBacking(
                                address, &data, num_bytes));
                        });
                    };
                    RUN_GFX_HOST(dst, sizeof(u32), write);
                    const u32 value = packet.DataDWord();
                    NoteCpWrite(dst, &value, sizeof(value));
                } else if (event_eos->command == PM4CmdEventWriteEos::Command::SignalFence &&
                           rasterizer && [&] {
                               // ISS_UNIFIED_MEMORY: written when the GPU work before it is done.
                               const VAddr dst = reinterpret_cast<VAddr>(event_eos->Address());
                               const u32 value = event_eos->DataDWord();
                               if (!rasterizer->DeferUnifiedFence(dst, sizeof(u32), value, false, [dst, value] {
                                       Core::Memory::Instance()->TryWriteBacking(
                                           reinterpret_cast<void*>(dst), &value, sizeof(value));
                                   })) {
                                   return false;
                               }
                               RecordCpLabel(dst, &value, sizeof(value));
                               return true;
                           }()) {
                } else {
                    event_eos->SignalFence([this](void* address, u64 data, u32 num_bytes) {
                        auto* memory = Core::Memory::Instance();
                        if (rasterizer) {
                            rasterizer->BeforeGpuThreadWrite(reinterpret_cast<VAddr>(address),
                                                             num_bytes);
                        }
                        ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                        if (rasterizer) {
                            rasterizer->OnGpuThreadWrite(reinterpret_cast<VAddr>(address),
                                                         num_bytes);
                        }
                        NoteCpWrite(reinterpret_cast<VAddr>(address), &data, num_bytes);
                    });
                }
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        const VAddr dst = reinterpret_cast<VAddr>(event_eos->Address());
                        if (CpSplitActive()) {
                            const u16 gds_index = event_eos->gds_index;
                            auto value = std::make_shared<u32>();
                            const auto write = [this, dst, gds_index, value] {
                                rasterizer->Finish();
                                *value = rasterizer->ReadDataFromGds(gds_index);
                                *reinterpret_cast<u32*>(dst) = *value;
                            };
                            RUN_GFX_HOST(dst, sizeof(u32), write);
                            NoteCpWrite(dst, value.get(), sizeof(u32));
                        } else {
                            rasterizer->Finish();
                            const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                            rasterizer->BeforeGpuThreadWrite(dst, sizeof(value));
                            *reinterpret_cast<u32*>(dst) = value;
                            rasterizer->OnGpuThreadWrite(dst, sizeof(value));
                            NoteCpWrite(dst, &value, sizeof(value));
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                if (pipe_deferred) {
                    break; // Queued to the draw pipe worker (RunDeferredFence).
                }
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                const auto selection = event_eop->data_sel.Value();
                const u32 bytes = selection == DataSelect::Data32Low ? sizeof(u32)
                                  : (selection == DataSelect::Data64 ||
                                     selection == DataSelect::GpuClock64 ||
                                     selection == DataSelect::PerfCounter) ? sizeof(u64) : 0;
                if (CpSplitActive() && bytes != 0) {
                    const PM4CmdEventWriteEop packet = *event_eop;
                    const VAddr dst = reinterpret_cast<VAddr>(packet.Address<u8>());
                    auto written = std::make_shared<u64>();
                    auto irq = std::make_shared<bool>(false);
                    const auto write = [packet, written, irq] {
                        packet.SignalFence(
                            [written](void* address, u64 data, u32 num_bytes) {
                                *written = data;
                                ASSERT(Core::Memory::Instance()->TryWriteBacking(
                                    address, &data, num_bytes));
                            },
                            [irq] { *irq = true; });
                    };
                    RUN_GFX_HOST(dst, bytes, write);
                    NoteCpWrite(dst, written.get(), bytes);
                    if (*irq) Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                } else if (rasterizer &&
                           (bytes != 0 ||
                            event_eop->int_sel.Value() != InterruptSelect::None) &&
                           [&] {
                               // ISS_UNIFIED_MEMORY: label and interrupt when the GPU is done.
                               const PM4CmdEventWriteEop packet = *event_eop;
                               const VAddr dst = reinterpret_cast<VAddr>(packet.Address<u8>());
                               const auto selection = packet.data_sel.Value();
                               const std::optional<u32> value =
                                   selection == DataSelect::Data32Low ||
                                           selection == DataSelect::Data64
                                       ? std::optional<u32>{packet.DataDWord()}
                                       : std::nullopt;
                               const bool irq =
                                   packet.int_sel.Value() != InterruptSelect::None;
                               return rasterizer->DeferUnifiedFence(dst, bytes, value, irq,
                                                                    [packet] {
                                   packet.SignalFence(
                                       [](void* address, u64 data, u32 num_bytes) {
                                           Core::Memory::Instance()->TryWriteBacking(
                                               address, &data, num_bytes);
                                       },
                                       [] {
                                           Platform::IrqC::Instance()->Signal(
                                               Platform::InterruptId::GfxEop);
                                       });
                               });
                           }()) {
                } else {
                    event_eop->SignalFence(
                        [this](void* address, u64 data, u32 num_bytes) {
                            auto* memory = Core::Memory::Instance();
                            if (rasterizer) {
                                rasterizer->BeforeGpuThreadWrite(reinterpret_cast<VAddr>(address),
                                                                 num_bytes);
                            }
                            ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                            if (rasterizer) {
                                rasterizer->OnGpuThreadWrite(reinterpret_cast<VAddr>(address),
                                                             num_bytes);
                            }
                            NoteCpWrite(reinterpret_cast<VAddr>(address), &data, num_bytes);
                        },
                        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
                }
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                const PM4DmaData command = *dma_data;
                const auto record = [this, command] {
                    if (command.src_sel == DmaDataSrc::Data &&
                        command.dst_sel == DmaDataDst::Gds) {
                        rasterizer->FillBuffer(command.dst_addr_lo, command.NumBytes(),
                                               command.data, true);
                    } else if ((command.src_sel == DmaDataSrc::Memory ||
                                command.src_sel == DmaDataSrc::MemoryUsingL2) &&
                               command.dst_sel == DmaDataDst::Gds) {
                        rasterizer->CopyBuffer(command.dst_addr_lo, command.SrcAddress<VAddr>(),
                                               command.NumBytes(), true, false);
                    } else if (command.src_sel == DmaDataSrc::Data &&
                               (command.dst_sel == DmaDataDst::Memory ||
                                command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->FillBuffer(command.DstAddress<VAddr>(), command.NumBytes(),
                                               command.data, false);
                        RecordCpLabel(command.DstAddress<VAddr>(), &command.data,
                                      std::min<u32>(command.NumBytes(), sizeof(u32)));
                    } else if (command.src_sel == DmaDataSrc::Gds &&
                               (command.dst_sel == DmaDataDst::Memory ||
                                command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(command.DstAddress<VAddr>(), command.src_addr_lo,
                                               command.NumBytes(), false, true);
                    } else if ((command.src_sel == DmaDataSrc::Memory ||
                                command.src_sel == DmaDataSrc::MemoryUsingL2) &&
                               (command.dst_sel == DmaDataDst::Memory ||
                                command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(command.DstAddress<VAddr>(),
                                               command.SrcAddress<VAddr>(), command.NumBytes(),
                                               false, false);
                    } else {
                        UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                        u32(command.src_sel), u32(command.dst_sel));
                    }
                };
                if (CpSplitActive()) {
                    RUN_GFX_GPU(record);
                    break;
                }
                // iss: a run of memory to memory DMA packets (the texture streamer's ~1500 small
                // copies in one frame) copied together; ISS_CP_DMA_BATCH=0 off.
                static const bool dma_batch = [] {
                    const char* p = std::getenv("ISS_CP_DMA_BATCH");
                    return !p || p[0] != '0';
                }();
                const auto mem_to_mem = [](const PM4DmaData& d) {
                    return (d.src_sel == DmaDataSrc::Memory ||
                            d.src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (d.dst_sel == DmaDataDst::Memory ||
                            d.dst_sel == DmaDataDst::MemoryUsingL2);
                };
                if (dma_batch && mem_to_mem(command)) {
                    const u32 yield_limit = Common::Perf::GetTuning().asc_yield_packets.load(
                        std::memory_order_relaxed);
                    // Never past the packet at which the ordinary loop would yield to ASC.
                    const size_t packet_budget =
                        yield_limit == 0 || asc_submits.load(std::memory_order_relaxed) == 0
                            ? 4096
                            : (gfx_packets_since_yield < yield_limit
                                   ? yield_limit - gfx_packets_since_yield - 1
                                   : 0);
                    boost::container::small_vector<Vulkan::Rasterizer::HostCopy, 64> copies;
                    copies.push_back({command.DstAddress<VAddr>(), command.SrcAddress<VAddr>(),
                                      command.NumBytes()});
                    const auto following = dcb.subspan(count + 1);
                    size_t cursor = 0;
                    while (copies.size() <= packet_budget && cursor < following.size()) {
                        const auto* next = reinterpret_cast<const PM4Header*>(following.data() + cursor);
                        if (next->type != 3 || next->type3.opcode != PM4ItOpcode::DmaData) break;
                        const auto words = next->type3.NumWords() + 1;
                        if (cursor + words > following.size()) break;
                        const auto* next_dma = reinterpret_cast<const PM4DmaData*>(next);
                        if (next_dma->dst_addr_lo == 0x3022C || next_dma->command.das != 0 ||
                            !mem_to_mem(*next_dma)) {
                            break;
                        }
                        if (native_passes) iss::native_passes::WaitRecordedAt(following.data() + cursor);
                        copies.push_back({next_dma->DstAddress<VAddr>(),
                                          next_dma->SrcAddress<VAddr>(), next_dma->NumBytes()});
                        cursor += words;
                    }
                    if (copies.size() > 1 && rasterizer->CopyBufferHostBatch(copies)) {
                        const size_t extra = copies.size() - 1;
                        processed_packets += extra;
                        if (yield_limit && asc_submits.load(std::memory_order_relaxed)) {
                            gfx_packets_since_yield += u32(extra);
                        }
                        dcb = NextPacket(dcb, cursor);
                        break;
                    }
                }
                record();
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    if (rasterizer) rasterizer->OnFence();
                    const VAddr dst = reinterpret_cast<VAddr>(address);
                    if (CpSplitActive() && data_size != 0) {
                        auto bytes = std::make_shared<std::vector<u8>>(data_size);
                        std::memcpy(bytes->data(), write_data->data, data_size);
                        const auto write = [dst, bytes] {
                            std::memcpy(reinterpret_cast<void*>(dst), bytes->data(),
                                        bytes->size());
                        };
                        RUN_GFX_HOST(dst, data_size, write);
                        RecordCpLabel(dst, bytes->data(), data_size);
                    } else {
                        if (rasterizer) rasterizer->BeforeGpuThreadWrite(dst, data_size);
                        std::memcpy(address, write_data->data, data_size);
                        if (rasterizer) rasterizer->OnGpuThreadWrite(dst, data_size);
                        RecordCpLabel(dst, write_data->data, data_size);
                    }
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                const PM4CmdMemSemaphore packet = *mem_semaphore;
                const VAddr sem_address = packet.Address<VAddr>();
                if (packet.IsSignaling()) {
                    if (CpSplitActive()) {
                        const auto write = [packet] { packet.Signal(); };
                        RUN_GFX_HOST(sem_address, sizeof(u64), write);
                    } else {
                        if (rasterizer)
                            rasterizer->BeforeGpuThreadWrite(sem_address, sizeof(u64));
                        packet.Signal();
                        if (rasterizer)
                            rasterizer->OnGpuThreadWrite(sem_address, sizeof(u64));
                    }
                    VideoCore::NoteHostWrite();
                } else {
                    if (rasterizer) rasterizer->SyncUnifiedRead(sem_address, sizeof(u64));
                    while (!packet.Signaled()) {
                        YIELD_GFX();
                    }
                    if (CpSplitActive()) {
                        const auto write = [packet] { packet.Decrement(); };
                        const auto ready = [packet] { return packet.Signaled(); };
                        RUN_GFX_HOST_WHEN(sem_address, sizeof(u64), write, ready);
                    } else {
                        if (rasterizer)
                            rasterizer->BeforeGpuThreadWrite(sem_address, sizeof(u64));
                        packet.Decrement();
                        if (rasterizer)
                            rasterizer->OnGpuThreadWrite(sem_address, sizeof(u64));
                    }
                    VideoCore::NoteHostWrite();
                    // The guest may have written memory before signaling.
                    Common::Perf::Add(Common::Perf::EpochSemaphoreWaits);
                    VideoCore::NextUploadEpoch();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                // The guest CPU validated the commands that follow.
                Common::Perf::Add(Common::Perf::EpochRewinds);
                VideoCore::NextUploadEpoch();
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                if (pipe_deferred) {
                    break; // Holds in the draw pipe's order (RunDeferredFence).
                }
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                if (SkipOcclusionWait(*wait_reg_mem)) {
                    break;
                }
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (rasterizer && wait_reg_mem->mem_space.Value() ==
                                      PM4CmdWaitRegMem::MemSpace::Memory) {
                    rasterizer->SyncUnifiedRead(reinterpret_cast<VAddr>(wait_addr), sizeof(u32));
                    u32 pending;
                    if (rasterizer->PendingUnifiedLabel(reinterpret_cast<VAddr>(wait_addr),
                                                        pending) &&
                        wait_reg_mem->TestValue(pending)) {
                        // A label this queue wrote before (deferred for the guest).
                        iss::unified::Add(iss::unified::CpLabelWaits);
                        OnWaitRegMemSatisfied(*wait_reg_mem);
                        break;
                    }
                }
                if (vo_port->IsVoLabel(wait_addr) && OnlyGraphicsSubmitsPending()) {
                    const u64 wait_start = Common::Perf::NowNs();
                    // Commands from guest threads (readbacks waiting for this thread) are still
                    // served while the label is not written. Compute work submitted meanwhile
                    // runs at once (the PS4 runs it alongside): wait by yielding to it then.
                    const bool yield_to_compute =
                        Common::Perf::GetTuning().asc_yield_packets.load(
                            std::memory_order_relaxed) != 0;
                    bool satisfied = false;
                    while (!(satisfied = vo_port->WaitVoLabelFor(
                                 [&] { return wait_reg_mem->Test(regs.reg_array); },
                                 std::chrono::milliseconds{1}))) {
                        ProcessCommands();
                        if (yield_to_compute && asc_submits.load(std::memory_order_relaxed) != 0) {
                            break;
                        }
                    }
                    blocked_ns += Common::Perf::NowNs() - wait_start;
                    if (satisfied) {
                        // Written by the presentation, not by the guest CPU: no new upload epoch.
                        Common::Perf::Add(Common::Perf::EpochVoWaitsSkipped);
                        break;
                    }
                }
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    YIELD_GFX();
                }
                OnWaitRegMemSatisfied(*wait_reg_mem);
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                task.handle.destroy();
                if (SegmentCensusEnabled()) {
                    g_segment = &segment_census;
                }
                g_iss_dcb_lo = segment_lo;
                g_iss_dcb_hi = segment_hi;
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                if (rasterizer) {
                    rasterizer->SyncUnifiedRead(reinterpret_cast<VAddr>(cond_exec->Address()),
                                                sizeof(u32));
                }
                const auto skip = *cond_exec->Address() == false;
                if (scout_submission) {
                    scout_submission->CheckCondition(header, skip);
                }
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }
    if (pass_cost) {
        IssPassCostTick(pass_cost_class);
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }
    if (scout_seq != 0 && scout) {
        scout_submission.reset();
        scout->Finish(scout_seq);
    }

    if (SegmentCensusEnabled()) {
        EndSegment(segment_census);
        g_segment = nullptr;
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid, u64 submit_ns) {
    FIBER_ENTER(acb_task_name[vqid]);
    if constexpr (!is_indirect) iss::timeline::Emit("cp_asc_begin",reinterpret_cast<u64>(this),
        thread_context.token.sequence,thread_context.token.qid,thread_context.token.generation,
        reinterpret_cast<u64>(acb.data()),acb.size_bytes(),
        submit_ns?Common::Perf::NowNs()-submit_ns:0);
    auto& queue = asc_queues[{vqid}];
    Regs& compute_regs = CpSplitActive() ? mapped_queues[vqid + 1].regs : regs;
    if constexpr (!is_indirect) {
        // ISS_NATIVE_ASC_ASYNC: the pool records this submission's dispatches; the graphics queue
        // runs meanwhile.
        if (rasterizer && iss::native_passes::AscEnabled()) {
            while (!iss::native_passes::AscRecorded(acb.data())) {
                YIELD_ASC_QUEUE(vqid);
            }
        }
    }
    if (submit_ns != 0) {
        // First resumed: how long the work waited for the command processor.
        Common::Perf::Add(Common::Perf::AscStarts);
        Common::Perf::Add(Common::Perf::AscStartDelayNs, Common::Perf::NowNs() - submit_ns);
    }
    if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
        rasterizer->DumpNote("asc", fmt::format("queue {} {} dwords{}", vqid, acb.size(),
                                                is_indirect ? " indirect" : ""));
    }

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    // ISS_PASS_COST: the time before this queue's work (idle), then its packets (compute).
    const bool pass_cost = IssPassCostEnabled();
    u32 pass_cost_class = 11;
    while (!acb.empty()) {
        if (pass_cost) {
            IssPassCostTick(pass_cost_class);
            pass_cost_class = 13;
        }
        ProcessCommands();
        if (CpSplitActive()) ++asc_processed_packets;
        else ++processed_packets;
        // asc_queue: the dispatches, copies and fills of this queue run on the compute queue.
        if (rasterizer) {
            rasterizer->SetAsyncCompute(true);
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                if (CpSplitActive()) {
                    const u32 advance = static_cast<u32>(acb.size());
                    const auto write = [this,&queue,advance] {
                        AdvanceAscReadPointer(queue,advance);
                    };
                    RUN_ASC_HOST(vqid,reinterpret_cast<VAddr>(queue.read_addr),
                                 sizeof(*queue.read_addr),write);
                } else {
                    *queue.read_addr += acb.size();
                    *queue.read_addr %= queue.ring_size_dw;
                }
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                if (CpSplitActive()) {
                    const u32 advance = next_dw_off;
                    const auto write = [this,&queue,advance] {
                        AdvanceAscReadPointer(queue,advance);
                    };
                    RUN_ASC_HOST(vqid,reinterpret_cast<VAddr>(queue.read_addr),
                                 sizeof(*queue.read_addr),write);
                } else {
                    *queue.read_addr += next_dw_off;
                    *queue.read_addr %= queue.ring_size_dw;
                }
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const bool pipe_deferred = rasterizer && rasterizer->BeforeDrawPipePacket(
            u32(opcode), reinterpret_cast<const u32*>(header), header->type3.NumWords() + 1,
            static_cast<int>(queue.pipe_id));
        // Compute phases (pc-port): any packet but register writes and direct dispatches ends
        // the run of dispatches that need no barrier between them.
        if (opcode != PM4ItOpcode::SetShReg && opcode != PM4ItOpcode::Nop &&
            opcode != PM4ItOpcode::DispatchDirect) {
            iss::phase::g_boundary.store(true, std::memory_order_relaxed);
        }
        if (iss::command_trace::enabled) {
            iss::command_trace::Packet(CurrentQueue(), u32(opcode),
                                      reinterpret_cast<const u32*>(header),
                                      header->type3.NumWords() + 1);
        }

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC_QUEUE(vqid);
                RESUME_ASC(task, vqid);
            }
            task.handle.destroy();
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (CpSplitActive()) {
                const PM4DmaData command = *dma_data;
                const u32 num_bytes = command.NumBytes();
                // This DMA patches a following dispatch's arguments; its source is
                // a GPU reference, so keep the existing indirect-dispatch route.
                if ((command.src_sel == DmaDataSrc::Memory ||
                     command.src_sel == DmaDataSrc::MemoryUsingL2) &&
                    (command.dst_sel == DmaDataDst::Memory ||
                     command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                    const VAddr dst = command.DstAddress<VAddr>();
                    const PM4Header* target =
                        reinterpret_cast<const PM4Header*>(dst - sizeof(PM4Header));
                    if (dst >= base_addr && dst < base_addr + acb_size &&
                        num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                        target->type == 3 &&
                        target->type3.opcode == PM4ItOpcode::DispatchDirect) {
                        indirect_patches.emplace_back(target, command.SrcAddress<VAddr>());
                        break;
                    }
                }
                if (num_bytes == 0) break;
                const u32 qid = vqid + 1;
                const auto submission = thread_context.token;
                iss::g4::DmaFallbackPacket packet{};
                packet.guest_qid = qid;
                packet.guest_generation = submission.generation;
                packet.submission_sequence = submission.sequence;
                packet.packet_sequence = ++mapped_queues[qid].packet_sequence;
                packet.size = num_bytes;
                if (command.src_sel == DmaDataSrc::Data &&
                    command.dst_sel == DmaDataDst::Gds) {
                    packet.kind = iss::g4::DmaFallbackKind::Fill;
                    packet.dst = command.dst_addr_lo;
                    packet.value = command.data;
                    packet.dst_gds = true;
                } else if ((command.src_sel == DmaDataSrc::Memory ||
                            command.src_sel == DmaDataSrc::MemoryUsingL2) &&
                           command.dst_sel == DmaDataDst::Gds) {
                    packet.kind = iss::g4::DmaFallbackKind::Copy;
                    packet.dst = command.dst_addr_lo;
                    packet.src = command.SrcAddress<VAddr>();
                    packet.dst_gds = true;
                } else if (command.src_sel == DmaDataSrc::Data &&
                           (command.dst_sel == DmaDataDst::Memory ||
                            command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                    packet.kind = iss::g4::DmaFallbackKind::Fill;
                    packet.dst = command.DstAddress<VAddr>();
                    packet.value = command.data;
                    packet.record_label = true;
                } else if (command.src_sel == DmaDataSrc::Gds &&
                           (command.dst_sel == DmaDataDst::Memory ||
                            command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                    packet.kind = iss::g4::DmaFallbackKind::Copy;
                    packet.dst = command.DstAddress<VAddr>();
                    packet.src = command.src_addr_lo;
                    packet.src_gds = true;
                } else if ((command.src_sel == DmaDataSrc::Memory ||
                            command.src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (command.dst_sel == DmaDataDst::Memory ||
                            command.dst_sel == DmaDataDst::MemoryUsingL2)) {
                    packet.kind = iss::g4::DmaFallbackKind::Copy;
                    packet.dst = command.DstAddress<VAddr>();
                    packet.src = command.SrcAddress<VAddr>();
                } else {
                    UNREACHABLE_MSG("Unsupported split DMA src_sel={}, dst_sel={}",
                                    u32(command.src_sel), u32(command.dst_sel));
                }
                ASSERT_MSG(packet.Valid(), "G4 DMA scalar packet is incomplete");
                auto fallback = ProcessAscDmaFallback(vqid, packet);
                RESUME_ASC(fallback, vqid);
                while (!fallback.handle.done()) {
                    YIELD_ASC_QUEUE(vqid);
                    RESUME_ASC(fallback, vqid);
                }
                fallback.handle.destroy();
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->asc_copy_vqid = vqid; // ISS_ASC_NATIVE_QUEUE (FillBuffer)
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
                rasterizer->asc_copy_vqid = ~0u;
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->asc_copy_vqid = vqid; // ISS_ASC_NATIVE_QUEUE (CopyBuffer)
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
                rasterizer->asc_copy_vqid = ~0u;
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->asc_copy_vqid = vqid; // ISS_ASC_NATIVE_QUEUE (FillBuffer)
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
                rasterizer->asc_copy_vqid = ~0u;
                RecordCpLabel(dma_data->DstAddress<VAddr>(), &dma_data->data,
                              std::min<u32>(dma_data->NumBytes(), sizeof(u32)));
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->asc_copy_vqid = vqid; // ISS_ASC_NATIVE_QUEUE (CopyBuffer)
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
                rasterizer->asc_copy_vqid = ~0u;
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->asc_copy_vqid = vqid; // ISS_ASC_NATIVE_QUEUE (CopyBuffer)
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                    rasterizer->asc_copy_vqid = ~0u;
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC_QUEUE(vqid);
            }
            // The guest CPU validated the commands that follow.
            Common::Perf::Add(Common::Perf::EpochRewinds);
            VideoCore::NextUploadEpoch();
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&compute_regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->AscBeforeGraphicsWork(vqid, 3); // ISS_ASC_NATIVE_QUEUE
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            // ISS_NATIVE_ASC: a compute segment the pool recorded at the DingDong from this
            // dispatch on (native_passes.cpp BeforeAscSubmit), executed whole and jumped over, or
            // a learn request for it. Only inside indirect buffers (the ring's read pointer
            // advances packet by packet).
            if constexpr (is_indirect) {
                namespace np = iss::native_passes;
                if (rasterizer && np::AscEnabled()) {
                    const u32* first = reinterpret_cast<const u32*>(header);
                    if (np::Segment* seg = np::Claim(first)) {
                        if (seg->kind == np::Kind::Learn) {
                            rasterizer->native_compute_learn = seg->first_context;
                            np::Finish(*seg, false, 0);
                        } else if (seg->kind == np::Kind::Compute) {
                            bool usable = seg->end > first && seg->end <= acb.data() + acb.size() &&
                                          !CpSplitActive() && !iss::pause_guard::Blocked(iss::pause_guard::Compute);
                            // A dispatch whose dimensions a DMA patched is an indirect one.
                            for (const auto& patch : indirect_patches) {
                                const u32* at = reinterpret_cast<const u32*>(patch.header);
                                usable &= !(at >= first && at < seg->end);
                            }
                            u64 tick = 0;
                            rasterizer->AscSetVqid(vqid);
                            if (usable && rasterizer->NativeComputeExecute(*seg, tick, true)) {
                                auto& cs = GetCsRegs();
                                auto* cs_words = reinterpret_cast<u32*>(&cs);
                                for (const auto& [reg, value] : seg->final_regs) {
                                    const u32 word = reg - np::CsBase;
                                    if (reg >= np::CsBase && word < sizeof(ComputeProgram) / 4)
                                        cs_words[word] = value;
                                    else
                                        compute_regs.reg_array[reg] = value;
                                }
                                cs.dim_x = seg->last_dims[0];
                                cs.dim_y = seg->last_dims[1];
                                cs.dim_z = seg->last_dims[2];
                                cs.dispatch_initiator = seg->last_dispatch_initiator;
                                processed_packets += seg->draw_packets.size();
                                const u32* resume = seg->end;
                                np::Finish(*seg, true, tick);
                                // The next full path dispatch is not in a run with these.
                                iss::phase::g_boundary.store(true, std::memory_order_relaxed);
                                acb = acb.subspan(resume - acb.data());
                                continue;
                            }
                            np::Finish(*seg, false, 0);
                        } else {
                            np::Finish(*seg, false, 0);
                        }
                    }
                }
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            // ISS_ASC_STATS=1 (diagnosis): what the compute rings dispatch, every 5 s.
            static const bool asc_stats = [] {
                const char* p = std::getenv("ISS_ASC_STATS");
                return p && p[0] == '1';
            }();
            if (asc_stats) {
                static std::unordered_map<u64, u64> programs;
                static u64 total = 0, empty = 0, tiny = 0, groups = 0;
                static auto last = std::chrono::steady_clock::now();
                const u64 n = u64(dispatch_direct->dim_x) * dispatch_direct->dim_y * dispatch_direct->dim_z;
                ++total;
                empty += n == 0;
                tiny += n > 0 && n <= 4;
                groups += n;
                ++programs[reinterpret_cast<u64>(cs_program.Address<u32*>())];
                const auto now = std::chrono::steady_clock::now();
                if (now - last > std::chrono::seconds(5)) {
                    std::vector<std::pair<u64, u64>> top(programs.begin(), programs.end());
                    std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
                    std::string text;
                    for (size_t i = 0; i < top.size() && i < 8; ++i)
                        text += fmt::format(" {:x}:{}", top[i].first, top[i].second);
                    std::fprintf(stderr, "ASC_STATS dispatches=%llu empty=%llu tiny=%llu groups=%llu programs=%zu initiator_off=%d top[%s ]\n",
                                 (unsigned long long)total, (unsigned long long)empty,
                                 (unsigned long long)tiny, (unsigned long long)groups,
                                 programs.size(), int((cs_program.dispatch_initiator & 1) == 0),
                                 text.c_str());
                    total = empty = tiny = groups = 0;
                    programs.clear();
                    last = now;
                }
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                if (rasterizer && rasterizer->native_compute_learn) {
                    rasterizer->NativeComputeLearnFailed(rasterizer->native_compute_learn);
                    rasterizer->native_compute_learn = 0;
                }
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->AscBeforeGraphicsWork(vqid, 2); // ISS_ASC_NATIVE_QUEUE
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            if (rasterizer->native_compute_learn) {
                // Not learned (an emulated or skipped dispatch, or its bindings failed).
                rasterizer->NativeComputeLearnFailed(rasterizer->native_compute_learn);
                rasterizer->native_compute_learn = 0;
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->AscBeforeGraphicsWork(vqid, 3); // ISS_ASC_NATIVE_QUEUE
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) rasterizer->OnFence();
                const VAddr dst = write_data->Address<VAddr>();
                if (CpSplitActive() && data_size) {
                    auto bytes = std::make_shared<std::vector<u8>>(data_size);
                    std::memcpy(bytes->data(),write_data->data,data_size);
                    const auto write = [dst,bytes] {
                        std::memcpy(reinterpret_cast<void*>(dst),bytes->data(),bytes->size());
                    };
                    RUN_ASC_HOST(vqid,dst,data_size,write);
                    RecordCpLabel(dst,bytes->data(),data_size);
                } else {
                    if (rasterizer) rasterizer->BeforeGpuThreadWrite(dst,data_size);
                    std::memcpy(write_data->Address<void*>(),write_data->data,data_size);
                    if (rasterizer) rasterizer->OnGpuThreadWrite(dst,data_size);
                    RecordCpLabel(dst,write_data->data,data_size);
                }
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            const PM4CmdMemSemaphore packet = *mem_semaphore;
            const VAddr semaphore_address = packet.Address<VAddr>();
            if (packet.IsSignaling()) {
                if (CpSplitActive()) {
                    const auto write = [packet] { packet.Signal(); };
                    RUN_ASC_HOST(vqid,semaphore_address,sizeof(u64),write);
                } else {
                    if (rasterizer)
                        rasterizer->BeforeGpuThreadWrite(semaphore_address,sizeof(u64));
                    packet.Signal();
                    if (rasterizer)
                        rasterizer->OnGpuThreadWrite(semaphore_address,sizeof(u64));
                }
                VideoCore::NoteHostWrite();
            } else {
                while (!packet.Signaled()) {
                    YIELD_ASC_QUEUE(vqid);
                }
                if (CpSplitActive()) {
                    const auto write = [packet] { packet.Decrement(); };
                    const auto ready = [packet] { return packet.Signaled(); };
                    RUN_ASC_HOST_WHEN(vqid,semaphore_address,sizeof(u64),write,ready);
                } else {
                    if (rasterizer)
                        rasterizer->BeforeGpuThreadWrite(semaphore_address,sizeof(u64));
                    packet.Decrement();
                    if (rasterizer)
                        rasterizer->OnGpuThreadWrite(semaphore_address,sizeof(u64));
                }
                VideoCore::NoteHostWrite();
                Common::Perf::Add(Common::Perf::EpochSemaphoreWaits);
                VideoCore::NextUploadEpoch();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            if (pipe_deferred) {
                break; // Holds in the draw pipe's order (RunDeferredFence).
            }
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (SkipOcclusionWait(*wait_reg_mem)) {
                break;
            }
            if (rasterizer &&
                wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
                rasterizer->SyncUnifiedRead(wait_reg_mem->Address<VAddr>(), sizeof(u32));
                u32 pending;
                if (rasterizer->PendingUnifiedLabel(wait_reg_mem->Address<VAddr>(), pending) &&
                    wait_reg_mem->TestValue(pending)) {
                    iss::unified::Add(iss::unified::CpLabelWaits);
                    rasterizer->AscWaitedLabel(vqid, wait_reg_mem->Address<VAddr>());
                    OnWaitRegMemSatisfied(*wait_reg_mem);
                    break;
                }
            }
            while (!wait_reg_mem->Test(compute_regs.reg_array)) {
                YIELD_ASC_QUEUE(vqid);
            }
            OnWaitRegMemSatisfied(*wait_reg_mem);
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            if (pipe_deferred) {
                break; // Queued to the draw pipe worker (RunDeferredFence).
            }
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                // asc_queue: the work before the fence is submitted now, so the guest (and the
                // readbacks at this fence) wait for it only.
                rasterizer->FlushComputeQueue();
                rasterizer->OnFence();
            }
            const auto selection = release_mem->data_sel.Value();
            const u32 direct_write_bytes =
                selection == DataSelect::Data32Low ? sizeof(u32)
                : (selection == DataSelect::Data64 || selection == DataSelect::GpuClock64 ||
                   selection == DataSelect::PerfCounter) ? sizeof(u64) : 0;
            const PM4CmdReleaseMem packet = *release_mem;
            const VAddr dst = packet.Address<VAddr>();
            iss::timeline::Emit("asc_release_mem", reinterpret_cast<u64>(this), vqid, dst,
                                packet.DataDWord(), u32(selection));
            if (CpSplitActive() && direct_write_bytes) {
                const auto write = [packet,pipe_id=queue.pipe_id] {
                    packet.SignalFence(
                        [pipe_id] { Platform::IrqC::Instance()->Signal(
                            static_cast<Platform::InterruptId>(pipe_id)); },
                        [](VAddr,u16,u16) {
                            UNREACHABLE_MSG("GDS copy inside direct ASC host-write guard");
                        });
                };
                RUN_ASC_HOST(vqid,dst,direct_write_bytes,write);
            } else if (rasterizer && direct_write_bytes != 0 &&
                       [&](Common::UniqueFunction<void>&& write) {
                           // ISS_ASC_NATIVE_QUEUE: on the compute queue when the ring's last work
                           // is there, otherwise on the graphics queue as before.
                           const auto value =
                               selection == DataSelect::Data32Low || selection == DataSelect::Data64
                                   ? std::optional<u32>{packet.DataDWord()} : std::nullopt;
                           const bool irq = packet.int_sel.Value() != InterruptSelect::None;
                           if (rasterizer->AscRelease(vqid, dst, direct_write_bytes, value, irq,
                                                      std::move(write)))
                               return true;
                           return rasterizer->DeferUnifiedFence(dst, direct_write_bytes, value, irq,
                                                                std::move(write));
                       }(
                                                     [packet, pipe_id = queue.pipe_id] {
                           // ISS_UNIFIED_MEMORY: label and interrupt when the GPU is done.
                           u64 value = 0;
                           u32 num_bytes = sizeof(u64);
                           switch (packet.data_sel.Value()) {
                           case DataSelect::Data32Low:
                               value = packet.DataDWord();
                               num_bytes = sizeof(u32);
                               break;
                           case DataSelect::Data64:
                               value = packet.DataQWord();
                               break;
                           case DataSelect::GpuClock64:
                               value = GetGpuClock64();
                               break;
                           default:
                               value = GetGpuPerfCounter();
                               break;
                           }
                           Core::Memory::Instance()->TryWriteBacking(
                               packet.Address<void*>(), &value, num_bytes);
                           if (packet.int_sel.Value() != InterruptSelect::None) {
                               Platform::IrqC::Instance()->Signal(
                                   static_cast<Platform::InterruptId>(pipe_id));
                           }
                       })) {
            } else {
                if (rasterizer && direct_write_bytes)
                    rasterizer->BeforeGpuThreadWrite(dst,direct_write_bytes);
                packet.SignalFence(
                    [pipe_id=queue.pipe_id] { Platform::IrqC::Instance()->Signal(
                        static_cast<Platform::InterruptId>(pipe_id)); },
                    [this](VAddr address,u16 gds_index,u16 num_dwords) {
                        rasterizer->CopyBuffer(address,gds_index,num_dwords*sizeof(u32),
                                               false,true);
                    });
                if (rasterizer && direct_write_bytes)
                    rasterizer->OnGpuThreadWrite(dst,direct_write_bytes);
            }
            switch (release_mem->data_sel.Value()) {
            case DataSelect::Data32Low:
                NoteCpWrite(release_mem->Address<VAddr>(), release_mem->Address<const void*>(),
                            sizeof(u32));
                break;
            case DataSelect::Data64:
            case DataSelect::GpuClock64:
            case DataSelect::PerfCounter:
                NoteCpWrite(release_mem->Address<VAddr>(), release_mem->Address<const void*>(),
                            sizeof(u64));
                break;
            default:
                break;
            }
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            if (CpSplitActive()) {
                const u32 advance = next_dw_off;
                const auto write = [this,&queue,advance] {
                    AdvanceAscReadPointer(queue,advance);
                };
                RUN_ASC_HOST(vqid,reinterpret_cast<VAddr>(queue.read_addr),
                             sizeof(*queue.read_addr),write);
            } else {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
        }
    }

    if (rasterizer) {
        rasterizer->SetAsyncCompute(false);
        rasterizer->FlushComputeQueue();
        if constexpr (!is_indirect) rasterizer->AscFlush(); // ISS_ASC_NATIVE_QUEUE
    }
    if (pass_cost) IssPassCostTick(13);
    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::RecordCpLabel(VAddr address, const void* data, u32 num_bytes) {
    // Labels are one or two dwords. Longer writes are data: only their first dwords are kept.
    const u32 num_dwords = std::min<u32>(num_bytes / sizeof(u32), 8);
    std::array<CpLabel, 8> labels{};
    for (u32 i = 0; i < num_dwords; ++i) {
        u32 value;
        std::memcpy(&value, static_cast<const u8*>(data) + i * sizeof(u32), sizeof(value));
        const VAddr dword_address = address + i * sizeof(u32);
        labels[i] = {dword_address, value};
    }
    std::scoped_lock lk{cp_labels_mutex};
    for (u32 i = 0; i < num_dwords; ++i) {
        const CpLabel& label = labels[i];
        cp_labels[(label.address / sizeof(u32)) % NumCpLabels] = label;
    }
}

void Liverpool::WriteEmulatedOcclusionCounters(VAddr address) {
    static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
    static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
    u64* results = reinterpret_cast<u64*>(address);
    for (u32 i = 0; i < num_counter_pairs; ++i, results += 2) {
        *results = pixel_counter | OcclusionCounterValidMask;
    }
    pixel_counter += OcclusionCounterStep;
}

void Liverpool::RunDeferredFence(u32 opcode, const u32* words, u32 pipe_id) {
    const auto write_backing = [this](void* address, u64 data, u32 num_bytes) {
        auto* memory = Core::Memory::Instance();
        if (rasterizer) {
            rasterizer->BeforeGpuThreadWrite(reinterpret_cast<VAddr>(address), num_bytes);
        }
        ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
        if (rasterizer) {
            rasterizer->OnGpuThreadWrite(reinterpret_cast<VAddr>(address), num_bytes);
        }
        NoteCpWrite(reinterpret_cast<VAddr>(address), &data, num_bytes);
    };
    switch (static_cast<PM4ItOpcode>(opcode)) {
    case PM4ItOpcode::EventWriteEos: {
        const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(words);
        if (rasterizer) {
            rasterizer->OnFence();
        }
        event_eos->SignalFence(write_backing);
        if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
            ASSERT(event_eos->size == 1);
            if (rasterizer) {
                const VAddr dst = reinterpret_cast<VAddr>(event_eos->Address());
                rasterizer->Finish();
                const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                rasterizer->BeforeGpuThreadWrite(dst, sizeof(value));
                *reinterpret_cast<u32*>(dst) = value;
                rasterizer->OnGpuThreadWrite(dst, sizeof(value));
                NoteCpWrite(dst, &value, sizeof(value));
            }
        }
        break;
    }
    case PM4ItOpcode::EventWriteEop: {
        const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(words);
        if (rasterizer) {
            rasterizer->OnFence();
        }
        event_eop->SignalFence(
            write_backing,
            [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
        break;
    }
    case PM4ItOpcode::ReleaseMem: {
        const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(words);
        if (rasterizer) {
            rasterizer->FlushComputeQueue();
            rasterizer->OnFence();
        }
        const auto selection = release_mem->data_sel.Value();
        const u32 direct_write_bytes =
            selection == DataSelect::Data32Low ? sizeof(u32)
            : (selection == DataSelect::Data64 || selection == DataSelect::GpuClock64 ||
               selection == DataSelect::PerfCounter) ? sizeof(u64) : 0;
        const PM4CmdReleaseMem packet = *release_mem;
        const VAddr dst = packet.Address<VAddr>();
        if (rasterizer && direct_write_bytes)
            rasterizer->BeforeGpuThreadWrite(dst, direct_write_bytes);
        packet.SignalFence(
            [pipe_id] { Platform::IrqC::Instance()->Signal(
                static_cast<Platform::InterruptId>(pipe_id)); },
            [this](VAddr address, u16 gds_index, u16 num_dwords) {
                rasterizer->CopyBuffer(address, gds_index, num_dwords * sizeof(u32), false, true);
            });
        if (rasterizer && direct_write_bytes)
            rasterizer->OnGpuThreadWrite(dst, direct_write_bytes);
        switch (selection) {
        case DataSelect::Data32Low:
            NoteCpWrite(release_mem->Address<VAddr>(), release_mem->Address<const void*>(),
                        sizeof(u32));
            break;
        case DataSelect::Data64:
        case DataSelect::GpuClock64:
        case DataSelect::PerfCounter:
            NoteCpWrite(release_mem->Address<VAddr>(), release_mem->Address<const void*>(),
                        sizeof(u64));
            break;
        default:
            break;
        }
        break;
    }
    case PM4ItOpcode::WaitRegMem:
        // Held when queued: what the command processor does once a wait is satisfied.
        OnWaitRegMemSatisfied(*reinterpret_cast<const PM4CmdWaitRegMem*>(words));
        break;
    default:
        UNREACHABLE_MSG("Deferred packet {:#x} is not a fence", opcode);
    }
}

void Liverpool::NoteCpWrite(VAddr address, const void* data, u32 num_bytes) {
    RecordCpLabel(address, data, num_bytes);
    VideoCore::NoteHostWrite();
}

bool Liverpool::IsWrittenByCp(VAddr address) const {
    CpLabel label;
    {
        std::scoped_lock lk{cp_labels_mutex};
        label = cp_labels[(address / sizeof(u32)) % NumCpLabels];
    }
    return label.address == address && label.value == *reinterpret_cast<const u32*>(address);
}

bool Liverpool::SkipOcclusionWait(const PM4CmdWaitRegMem& wait) {
    if (wait.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        return false;
    }
    const VAddr address = wait.Address<VAddr>();
    if (wait.IsAlwaysSatisfied()) {
        // Nothing to read: reading memory the GPU wrote (the fill of occlusion counters this
        // waits for) would wait for the GPU. The commands that follow run after it anyway.
        Common::Perf::Add(Common::Perf::WaitsTrivial);
        if (rasterizer && rasterizer->IsDumping()) [[unlikely]] {
            rasterizer->DumpNote("wait", fmt::format("{:x} func {} mask {:x} ref {:x} always "
                                                     "satisfied, skipped",
                                                     address,
                                                     static_cast<u32>(wait.function.Value()),
                                                     wait.mask, wait.ref));
        }
        return true;
    }
    if (!rasterizer) {
        return false;
    }
    std::optional<u32> known;
    const bool counted = rasterizer->IsCountedOcclusionResult(address, known);
    const bool skip = known && wait.TestValue(*known);
    if (rasterizer->IsDumping()) [[unlikely]] {
        rasterizer->DumpNote("wait", fmt::format("{:x} func {} mask {:x} ref {:x} {}", address,
                                                 static_cast<u32>(wait.function.Value()), wait.mask,
                                                 wait.ref,
                                                 !counted ? "polled"
                                                 : skip   ? "occlusion result known, skipped"
                                                 : known  ? "occlusion result known, polled"
                                                          : "occlusion samples, polled"));
    }
    if (counted) {
        Common::Perf::Add(skip ? Common::Perf::OcclusionWaitsSkipped
                               : Common::Perf::OcclusionWaitsRead);
    }
    return skip;
}

void Liverpool::OnWaitRegMemSatisfied(const PM4CmdWaitRegMem& wait) {
    if (wait.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        // Registers are only written by commands.
        return;
    }
    const VAddr address = wait.Address<VAddr>();
    if (vo_port && vo_port->IsVoLabel(reinterpret_cast<const u64*>(address))) {
        // Written by the presentation: the commands that follow were submitted before.
        Common::Perf::Add(Common::Perf::EpochVoWaitsSkipped);
        return;
    }
    if (IsWrittenByCp(address)) {
        // Released by the GPU itself (end of pipe fence of another queue or of earlier work):
        // the guest wrote the data of the commands that follow before submitting them.
        Common::Perf::Add(Common::Perf::EpochCpWaitsSkipped);
        return;
    }
    // The guest CPU writes the data before the value waited for.
    Common::Perf::Add(Common::Perf::EpochCpuWaits);
    VideoCore::NextUploadEpoch();
}

void Liverpool::SubmitDone() noexcept {
    const auto trace_start=iss::timeline::enabled?Common::Perf::NowNs():0;
    iss::timeline::Emit("guest_submit_done_begin",reinterpret_cast<u64>(this),
        Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed));
    SCOPE_EXIT {if(trace_start)iss::timeline::Emit("guest_submit_done_end",reinterpret_cast<u64>(this),
        Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed),0,0,0,0,
        Common::Perf::NowNs()-trace_start);};
    Common::Perf::GuestFrameCounter().fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lk{submit_mutex};
    mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
    mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
    submit_done = true;

    // Guest frames whose submissions the command processor has not finished processing yet.
    const auto mark_slot = frame_index % NumFrameMarks;
    frame_marks[mark_slot] = submits_issued.load(std::memory_order_relaxed);
    g4_frame_marks[mark_slot] = g4_ledger.Capture(frame_index);
    ++frame_index;
    const u64 completed = submits_completed.load(std::memory_order_acquire);
    u64 in_flight = 0;
    for (u64 i = 0; i < std::min<u64>(frame_index, NumFrameMarks); ++i) {
        const auto slot = (frame_index - 1 - i) % NumFrameMarks;
        if (CpSplitActive() ? !g4_ledger.Ready(g4_frame_marks[slot])
                               : frame_marks[slot] > completed) {
            ++in_flight;
        } else if (!g4_ledger.Ready(g4_frame_marks[slot])) {
            ++g4_shadow_global_ready_gap;
        }
    }
    Common::Perf::Add(Common::Perf::GpuLagFrames, in_flight);
    Common::Perf::Add(Common::Perf::GpuLagSamples);
    submit_cv.notify_all();

    // Optionally, the guest does not start a new frame before the command processor has
    // processed all the frames but the last max_lag ones (bounded wait).
    const u32 max_lag =
        Common::Perf::GetTuning().max_gpu_lag_frames.load(std::memory_order_relaxed);
    if (max_lag != 0 && frame_index > max_lag && max_lag < NumFrameMarks) {
        const auto target_slot = (frame_index - 1 - max_lag) % NumFrameMarks;
        const u64 target = frame_marks[target_slot];
        const auto ready = [&] {
            return CpSplitActive()
                       ? g4_ledger.Ready(g4_frame_marks[target_slot])
                       : submits_completed.load(std::memory_order_acquire) >= target;
        };
        if (!ready()) {
            const u64 wait_start = Common::Perf::NowNs();
            submit_cv.wait_for(lk, std::chrono::milliseconds{50}, ready);
            Common::Perf::Add(Common::Perf::GpuLagWaitNs, Common::Perf::NowNs() - wait_start);
        }
    }

    // Optionally, the guest waits until the command processor has processed the work submitted
    // so far: compute queues (1), or all queues (2). On the PS4 the GPU keeps up with the frame,
    // so the results of this work exist when the guest reads them back during its next frame;
    // here the command processor may still be a frame behind and the guest would read the
    // results of an older frame (bounded wait).
    const u32 sync = Common::Perf::GetTuning().gpu_sync_frame_end.load(std::memory_order_relaxed);
    if (sync != 0) {
        const u64 target = submits_issued.load(std::memory_order_relaxed);
        const auto target_mark = g4_ledger.Capture(frame_index);
        const auto done = [&] {
            return sync >= 2 ? (CpSplitActive()
                                    ? g4_ledger.Ready(target_mark)
                                    : submits_completed.load(std::memory_order_acquire) >= target)
                             : asc_submits.load(std::memory_order_acquire) == 0;
        };
        if (!done()) {
            const u64 wait_start = Common::Perf::NowNs();
            submit_cv.wait_for(lk, std::chrono::milliseconds{50}, done);
            Common::Perf::Add(Common::Perf::GpuLagWaitNs, Common::Perf::NowNs() - wait_start);
        }
    }
}

void Liverpool::StartScout(ScoutClient& client) {
    scout = std::make_unique<Scout>(client);
}

void Liverpool::BeginScoutDraw(const void* packet) {
    if (scout_submission) {
        current_scout_draw = scout_submission->Next(packet);
    }
}

void Liverpool::EndScoutDraw() {
    current_scout_draw = nullptr;
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    // The guest wrote the memory this submission reads before submitting it: hot pages are
    // uploaded again from now on (submission order is the guest thread's program order).
    Common::Perf::Add(Common::Perf::EpochGfxSubmits);
    VideoCore::NextUploadEpoch();

    // Scout and coroutine preparation may allocate. Serialize graphics admission
    // without holding either queue.m_access or submit_mutex across that work.
    std::scoped_lock admission_lock{gfx_admission_mutex};
    u64 scout_seq = 0;
    if (scout && ccb.empty()) {
        scout_seq = ++gfx_submit_seq;
        if (Common::Perf::GetTuning().scout.load(std::memory_order_relaxed)) {
            scout->Submit(scout_seq, dcb);
        } else {
            scout->Skip(scout_seq);
        }
    }
    const auto task = ProcessGraphics(dcb, ccb, scout_seq);
    try {
        std::unique_lock submit_lock{submit_mutex};
        std::unique_lock queue_lock{queue.m_access};
        queue.submits.emplace(GpuQueue::Submission{task.handle, {}});
        const bool issued = g4_ledger.Issue(GfxQueueId, queue.submits.back().token);
        ASSERT_MSG(issued, "G4 stage-1 graphics admission failed");
        const auto& trace_token=queue.submits.back().token;
        iss::timeline::Emit("guest_queue_submit",reinterpret_cast<u64>(this),trace_token.sequence,
            trace_token.qid,trace_token.generation,reinterpret_cast<u64>(task.handle.address()),dcb.size_bytes());
        CpCoordination::SubmissionToken coordinated_token{};
        ASSERT_MSG(cp_coordination.Issue(GfxQueueId, coordinated_token) &&
                       coordinated_token == queue.submits.back().token,
                   "G4 coordination graphics admission diverged");
        submits_issued.fetch_add(1, std::memory_order_relaxed);
        ++num_submits;
    } catch (...) {
        task.handle.destroy();
        throw;
    }
    submit_cv.notify_all();
}

bool Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb, u64 generation) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    Common::Perf::Add(Common::Perf::EpochAscSubmits);
    VideoCore::NextUploadEpoch();
    // ISS_NATIVE_ASC: this submission's dispatches recorded before the command processor gets it.
    if (rasterizer && !CpSplitActive() && iss::native_passes::AscEnabled()) {
        iss::native_passes::BeforeAscSubmit(
            gnm_vqid, acb.data(), u32(acb.size()),
            reinterpret_cast<const u32*>(&queue.cs_state), u32(sizeof(ComputeProgram) / 4));
    }
    const auto task = ProcessCompute(acb, vqid, Common::Perf::NowNs());
    bool published = false;
    try {
        std::unique_lock submit_lock{submit_mutex};
        std::unique_lock queue_lock{queue.m_access};
        if (g4_ledger.CanIssue(gnm_vqid) && g4_ledger.Generation(gnm_vqid) == generation) {
            queue.submits.emplace(GpuQueue::Submission{task.handle, {}});
            const bool issued = g4_ledger.Issue(gnm_vqid, queue.submits.back().token);
            ASSERT_MSG(issued, "G4 stage-1 ASC admission failed");
            const auto& trace_token=queue.submits.back().token;
            iss::timeline::Emit("guest_queue_submit",reinterpret_cast<u64>(this),trace_token.sequence,
                trace_token.qid,trace_token.generation,reinterpret_cast<u64>(task.handle.address()),acb.size_bytes());
            CpCoordination::SubmissionToken coordinated_token{};
            ASSERT_MSG(cp_coordination.Issue(gnm_vqid, coordinated_token) &&
                           coordinated_token == queue.submits.back().token,
                       "G4 coordination ASC admission diverged");
            asc_submits.fetch_add(1, std::memory_order_relaxed);
            submits_issued.fetch_add(1, std::memory_order_relaxed);
            num_mapped_queues.store(
                std::max(num_mapped_queues.load(std::memory_order_relaxed), gnm_vqid + 1),
                std::memory_order_release);
            ++num_submits;
            published = true;
        }
    } catch (...) {
        task.handle.destroy();
        throw;
    }
    if (published) submit_cv.notify_all();
    else task.handle.destroy(); // Stale DingDong snapshot must not keep the slot alive.
    return published;
}

} // namespace AmdGpu
