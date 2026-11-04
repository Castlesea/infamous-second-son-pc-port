#include "deep_profile.h"
#include "gpu_blackbox.h"
#include "pipeline_library.h"
// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <cctype>
#include <optional>
#include <future>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <ranges>
#include <thread>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <xxhash.h>

#include "common/hash.h"
#include "common/scope_exit.h"
#include "g4_fallback_packet.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_stats.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/driver_cache_data.h"
#include "shader_salt.h"
#include "iss_dlss.h"
#include "iss_dlss_upscale.h"

namespace iss {
extern thread_local bool assert_throws;
extern thread_local unsigned long long recipe_hash;
extern std::atomic<unsigned> g_preload_entries, g_preload_total;
extern std::atomic<int> g_preload_phase;
}

namespace Vulkan {

/// ISS_AOT_COMPUTE state (StartAotCompute, below).
struct AotCompute {
    struct Entry {
        u64 hash;
        u32 rsrc1, rsrc2, threads_x, threads_y, threads_z, dwords;
        std::string path;
        std::string pack; ///< Relative to the game folder, lower case, '/' separators.
        u64 offset;
    };

    /// The game folder relative name of a pack as Prioritize compares them.
    static std::string PackKey(std::string_view path) {
        std::string key;
        key.reserve(path.size());
        for (const char c : path) {
            key.push_back(c == '\\' ? '/' : char(std::tolower(static_cast<unsigned char>(c))));
        }
        while (key.starts_with("/")) key.erase(0, 1);
        if (key.starts_with("app0/")) key.erase(0, 5);
        return key;
    }

    /// The shaders of a pack the game just opened go first: it opens a pack well before it
    /// dispatches its shaders (~20 s, median).
    void Prioritize(const std::string& key) {
        const auto it = packs.find(key);
        if (it == packs.end()) return;
        std::scoped_lock lock{urgent_mutex};
        urgent.push_back(it->second);
        urgent_count.fetch_add(1, std::memory_order_release);
    }

    /// The next entry to compile: the opened packs' first, then the rest in order.
    size_t Take() {
        if (urgent_count.load(std::memory_order_acquire) != 0) {
            std::scoped_lock lock{urgent_mutex};
            while (!urgent.empty()) {
                auto& [begin, end] = urgent.front();
                while (begin < end) {
                    const u32 i = begin++;
                    if (taken[i].exchange(1, std::memory_order_relaxed) == 0) {
                        urgent_taken.fetch_add(1, std::memory_order_relaxed);
                        return i;
                    }
                }
                urgent.pop_front();
                urgent_count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= entries.size()) return SIZE_MAX;
            if (taken[i].exchange(1, std::memory_order_relaxed) == 0) return i;
        }
    }

    std::unordered_map<std::string, std::pair<u32, u32>> packs; ///< Range of entries.
    std::vector<std::atomic<u8>> taken;
    /// Per entry: 0 waiting, 1 being compiled, 2 left to the game (stateful), 3 compiled,
    /// 4 failed, 5 reached by the game first. For the misses report (NoteMiss).
    std::vector<std::atomic<u8>> state;
    std::unordered_map<u64, u32> entry_of;
    std::array<std::atomic<u64>, 7> misses{}; // by state, 6 = not in the index
    std::atomic<u32> miss_logs{0};
    void NoteMiss(u64 hash) {
        const auto it = entry_of.find(hash);
        const u32 st = it == entry_of.end() ? 6u : state[it->second].load(std::memory_order_relaxed);
        misses[st].fetch_add(1, std::memory_order_relaxed);
        if (miss_logs.fetch_add(1, std::memory_order_relaxed) < 48) {
            std::fprintf(stderr, "AOT_MISS %llx state=%u t=%.1f s\n", (unsigned long long)hash, st,
                         double(Common::Perf::NowNs() - start_ns) / 1e9);
        }
    }
    std::mutex urgent_mutex;
    std::deque<std::pair<u32, u32>> urgent;
    std::atomic<u32> urgent_count{0};
    std::atomic<u64> urgent_taken{0};
    struct Result {
        std::unique_ptr<Program> program;
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
        std::vector<u32> spv;
        std::unique_ptr<ComputePipeline> pipeline;
        ComputePipelineKey key;
    };

    vk::Device device;
    std::vector<Entry> entries;
    std::atomic<size_t> next{0};
    std::mutex mutex;
    std::vector<Result> ready;
    std::atomic<u32> num_ready{0};
    std::atomic<u64> compiled{0}, failed{0}, known{0}, adopted{0}, dropped{0}, stateful{0};
    std::array<std::atomic<u64>, 5> why{}; // stateful: images, samplers, fmasks, formatted, unknown V#
    std::atomic<u64> translate_ns{0}, pipeline_ns{0};
    std::atomic<u32> running{0};
    u64 start_ns{};
    bool reported{};
    std::vector<std::jthread> workers;

    ~AotCompute() {
        Stop();
        for (auto& r : ready) {
            r.pipeline.reset();
            if (r.module) device.destroyShaderModule(r.module);
        }
    }
    void Stop();
};

/// The running AOT compiler, for the packs the game opens (iss_gpu_file_opened); packs opened
/// before it starts are kept for it.
static std::mutex g_aot_mutex;
static AotCompute* g_aot{};
static std::vector<std::string> g_early_packs;

void AotCompute::Stop() {
    {
        std::scoped_lock lock{g_aot_mutex};
        if (g_aot == this) g_aot = nullptr;
    }
    for (auto& w : workers) w.request_stop();
    workers.clear();
}

} // namespace Vulkan

extern "C" __declspec(dllexport) void iss_gpu_file_opened(const char* guest_path) {
    if (!guest_path) return;
    const auto key = Vulkan::AotCompute::PackKey(guest_path);
    std::scoped_lock lock{Vulkan::g_aot_mutex};
    if (Vulkan::g_aot) {
        Vulkan::g_aot->Prioritize(key);
    } else if (Vulkan::g_early_packs.size() < 4096) {
        Vulkan::g_early_packs.push_back(key);
    }
}

namespace Vulkan {


/**
 * Worker threads creating the Vulkan pipeline objects whose compilation was deferred, so the
 * command processor does not stall for the driver (up to a few hundred milliseconds for each new
 * pipeline, which froze the game when an effect was seen for the first time).
 */
struct PipelineCache::PreTranslation {
    u64 hash{};
    std::unique_ptr<Program> program;
    Shader::RuntimeInfo initial_runtime_info{};
    Shader::RuntimeInfo runtime_info{};
    Shader::Pools pools{};
    std::optional<Shader::IR::Program> ir;
    std::future<void> done;
};

static bool ParallelTranslateEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_PARALLEL_TRANSLATE");
        return value && value[0] == '1';
    }();
    return enabled;
}
static bool ParallelTranslateVerify() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_PARALLEL_TRANSLATE_VERIFY");
        return value && value[0] == '1';
    }();
    return enabled;
}
static std::atomic<u64> g_pre_started{}, g_pre_used{}, g_pre_unused{}, g_pre_verified{},
    g_pre_mismatches{};
static void NotePreTranslation() {
    const u64 n = g_pre_started.load(std::memory_order_relaxed);
    if (n == 0 || (n != 1 && (n & 7) != 0)) {
        return;
    }
    std::fprintf(stderr,
                 "PARALLEL_TRANSLATE started=%llu used=%llu unused=%llu verified=%llu "
                 "mismatches=%llu\n",
                 (unsigned long long)n, (unsigned long long)g_pre_used.load(),
                 (unsigned long long)g_pre_unused.load(), (unsigned long long)g_pre_verified.load(),
                 (unsigned long long)g_pre_mismatches.load());
}


// ISS_SPEC_TRANSLATE=1: translations of new programs started by the scout (see SpecTranslator).
static bool SpecTranslateEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SPEC_TRANSLATE");
        return value && value[0] == '1';
    }();
    return enabled;
}
static bool SpecTranslateVerify() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SPEC_TRANSLATE_VERIFY");
        return value && value[0] == '1';
    }();
    return enabled;
}
static std::atomic<u64> g_spec_requested{}, g_spec_translated{}, g_spec_adopted{},
    g_spec_waited{}, g_spec_cancelled{}, g_spec_evicted{}, g_spec_verified{},
    g_spec_mismatches{}, g_spec_missed{};
static void NoteSpec() {
    static std::atomic<u64> calls{};
    if ((calls.fetch_add(1, std::memory_order_relaxed) & 63) != 63) {
        return;
    }
    std::fprintf(stderr,
                 "SPEC_TRANSLATE requested=%llu translated=%llu adopted=%llu waited=%llu "
                 "cancelled=%llu missed=%llu evicted=%llu verified=%llu mismatches=%llu\n",
                 (unsigned long long)g_spec_requested.load(),
                 (unsigned long long)g_spec_translated.load(),
                 (unsigned long long)g_spec_adopted.load(), (unsigned long long)g_spec_waited.load(),
                 (unsigned long long)g_spec_cancelled.load(),
                 (unsigned long long)g_spec_missed.load(), (unsigned long long)g_spec_evicted.load(),
                 (unsigned long long)g_spec_verified.load(),
                 (unsigned long long)g_spec_mismatches.load());
}

struct SpecTranslation {
    enum State : u32 { Queued, Running, Done, Failed, Taken };
    u64 key{};
    std::span<const u32> code;
    std::unique_ptr<Program> program;
    Shader::RuntimeInfo initial_runtime_info{};
    Shader::RuntimeInfo runtime_info{};
    Shader::Pools pools{};
    std::optional<Shader::IR::Program> ir;
    std::atomic<u32> state{Queued};
    u64 created_ns{};
};

/// Worker threads translating the programs the scout found new, keyed by program hash and
/// runtime info. The command processor takes an entry when it meets the program: done, it
/// only emits it; running, it waits; not started, it translates it itself as before.
class SpecTranslator {
public:
    SpecTranslator(const Shader::Profile& profile_, u32 num_threads) : profile{profile_} {
        for (u32 i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i](std::stop_token stop) { Run(stop, i); });
        }
    }
    ~SpecTranslator() {
        for (auto& worker : workers) {
            worker.request_stop();
        }
        work_cv.notify_all();
        workers.clear();
    }
    bool Contains(u64 key) {
        std::scoped_lock lk{mutex};
        return entries.contains(key);
    }
    void Request(std::shared_ptr<SpecTranslation> entry) {
        {
            std::scoped_lock lk{mutex};
            if (!entries.emplace(entry->key, entry).second) {
                return;
            }
            Evict();
            queue.push_back(std::move(entry));
        }
        g_spec_requested.fetch_add(1, std::memory_order_relaxed);
        work_cv.notify_one();
    }
    /// A queued translation whose draw was just skipped goes to the front.
    void Prioritize(u64 key) {
        std::scoped_lock lk{mutex};
        const auto it = std::find_if(queue.begin(), queue.end(),
                                     [key](const auto& entry) { return entry->key == key; });
        if (it == queue.end() || it == queue.begin()) {
            return;
        }
        auto entry = std::move(*it);
        queue.erase(it);
        queue.push_front(std::move(entry));
    }
    /// The state of the entry for this key (-1: none), left in place.
    int Peek(u64 key) {
        std::scoped_lock lk{mutex};
        const auto it = entries.find(key);
        return it == entries.end() ? -1 : int(it->second->state.load(std::memory_order_acquire));
    }
    std::shared_ptr<SpecTranslation> Take(u64 key) {
        std::scoped_lock lk{mutex};
        const auto it = entries.find(key);
        if (it == entries.end()) {
            return nullptr;
        }
        auto entry = std::move(it->second);
        entries.erase(it);
        return entry;
    }
    /// Done (true) or not usable (false): a queued entry is cancelled, a running one awaited.
    bool WaitDone(SpecTranslation& entry) {
        u32 expected = SpecTranslation::Queued;
        if (entry.state.compare_exchange_strong(expected, SpecTranslation::Taken)) {
            g_spec_cancelled.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (entry.state.load(std::memory_order_acquire) == SpecTranslation::Running) {
            g_spec_waited.fetch_add(1, std::memory_order_relaxed);
            std::unique_lock lk{mutex};
            done_cv.wait(lk, [&] {
                return entry.state.load(std::memory_order_acquire) != SpecTranslation::Running;
            });
        }
        return entry.state.load(std::memory_order_acquire) == SpecTranslation::Done;
    }

private:
    /// Entries the command processor never took (another runtime info at the draw): the
    /// oldest go once there are many.
    void Evict() {
        constexpr size_t MaxEntries = 256;
        if (entries.size() <= MaxEntries) {
            return;
        }
        u64 oldest_key = 0;
        u64 oldest_ns = ~0ULL;
        for (const auto& [key, entry] : entries) {
            const u32 state = entry->state.load(std::memory_order_acquire);
            if (state != SpecTranslation::Running && entry->created_ns < oldest_ns) {
                oldest_ns = entry->created_ns;
                oldest_key = key;
            }
        }
        if (oldest_ns != ~0ULL) {
            auto it = entries.find(oldest_key);
            u32 expected = SpecTranslation::Queued;
            it->second->state.compare_exchange_strong(expected, SpecTranslation::Taken);
            entries.erase(it);
            g_spec_evicted.fetch_add(1, std::memory_order_relaxed);
        }
    }
    void Run(std::stop_token stop, u32 index) {
        Common::SetCurrentThreadName(fmt::format("iss:SpecTranslate{}", index).c_str());
        while (!stop.stop_requested()) {
            std::shared_ptr<SpecTranslation> entry;
            {
                std::unique_lock lk{mutex};
                Common::CondvarWait(work_cv, lk, stop, [this] { return !queue.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                entry = std::move(queue.front());
                queue.pop_front();
            }
            u32 expected = SpecTranslation::Queued;
            if (!entry->state.compare_exchange_strong(expected, SpecTranslation::Running)) {
                continue;
            }
            u32 result = SpecTranslation::Done;
            try {
                entry->ir.emplace(Shader::TranslateProgram(entry->code,
                                                           entry->pools, entry->program->info,
                                                           entry->runtime_info, profile));
                g_spec_translated.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                result = SpecTranslation::Failed;
            }
            {
                std::scoped_lock lk{mutex};
                entry->state.store(result, std::memory_order_release);
            }
            done_cv.notify_all();
        }
    }

    const Shader::Profile& profile;
    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable done_cv;
    std::deque<std::shared_ptr<SpecTranslation>> queue;
    std::unordered_map<u64, std::shared_ptr<SpecTranslation>> entries;
    std::vector<std::jthread> workers;
};

// ISS_ASYNC_SKIP_DRAWS=1 (gow3_pc style): with draws of compiling pipelines skipped, a new vertex
// or pixel program is not translated on the command processor either: the speculative translator
// does it and the draws using it are skipped until it is done (cold cache: the command processor
// translated 47 shaders, 134 ms, in one frame; cold2 2026-10-18).
static bool AsyncSkipDraws() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_ASYNC_SKIP_DRAWS");
        // ISS_ASYNC_TRANSLATE=1 too (off by default): in a cold standing scene it doubled the
        // pipelines (762 -> 1659) and the >50 ms frames (5 -> 23), cturn-a/b 2026-10-18.
        const char* translate = std::getenv("ISS_ASYNC_TRANSLATE");
        return value && value[0] == '1' && translate && translate[0] == '1';
    }();
    return enabled;
}
static std::atomic<u64> g_async_pending{}, g_async_requested{};

static u64 SpecKey(u64 program_hash, const Shader::RuntimeInfo& runtime_info) {
    return HashCombine(program_hash, XXH3_64bits(&runtime_info, sizeof(runtime_info)));
}

void PipelineCache::RequestSpecTranslation(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                                           const Shader::ShaderParams& params,
                                           const Shader::RuntimeInfo& runtime_info) {
    if (!spec_translator) {
        return;
    }
    const u64 key = SpecKey(params.hash, runtime_info);
    if (spec_translator->Contains(key)) {
        return;
    }
    auto entry = std::make_shared<SpecTranslation>();
    entry->key = key;
    entry->code = params.code;
    entry->program = std::make_unique<Program>(hw_stage, sw_stage, params);
    // The scout's registers move on: the user data the translation reads is a copy.
    auto& replay = entry->program->replay_user_data;
    std::memcpy(replay.data(), params.user_data.data(),
                std::min(sizeof(replay), params.user_data.size_bytes()));
    entry->program->info.user_data = replay;
    entry->initial_runtime_info = runtime_info;
    entry->runtime_info = runtime_info;
    entry->created_ns = Common::Perf::NowNs();
    spec_translator->Request(std::move(entry));
}

static void BuildComputeRuntimeInfoFrom(Shader::RuntimeInfo& info,
                                        const AmdGpu::ComputeProgram& cs_pgm);
static bool SpecComputeEnabled();

void PipelineCache::RequestComputeSpecTranslation(const AmdGpu::ComputeProgram& cs) {
    if (!spec_translator || !SpecComputeEnabled()) {
        return;
    }
    if (!cs.Address<u32*>()) {
        return;
    }
    const auto params = AmdGpu::GetParams(cs);
    if (FindProgram(params.hash) != nullptr) {
        return; // Translated already (only new programs take a speculative translation).
    }
    Shader::RuntimeInfo runtime_info{};
    BuildComputeRuntimeInfoFrom(runtime_info, cs);
    RequestSpecTranslation(Shader::HwStage::Compute, Shader::SwStage::Compute, params, runtime_info);
}

static bool LazyPipelines() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_LAZY_PIPELINES");
        return value && value[0] == '1';
    }();
    return enabled;
}

class PipelineCompiler {
public:
    explicit PipelineCompiler(u32 num_threads) {
        workers.reserve(num_threads);
        for (u32 i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i](std::stop_token stoken) { WorkerLoop(stoken, i); });
        }
    }

    ~PipelineCompiler() {
        for (auto& worker : workers) {
            worker.request_stop();
        }
        workers.clear();
    }

    void Submit(Pipeline* pipeline) {
        {
            std::scoped_lock lk{mutex};
            queue.push_back(pipeline);
            ++pending;
        }
        work_cv.notify_one();
    }

    /// Background work (ISS_TIERED_GRAPHICS optimized variants): taken only when no pipeline the
    /// game waits for is queued. Not counted in `pending` (WaitIdle and WaitFor ignore it).
    void SubmitLow(Pipeline* pipeline) {
        {
            std::scoped_lock lk{mutex};
            low_queue.push_back(pipeline);
        }
        work_cv.notify_one();
    }

    /// Waits until the pipeline is compiled or the timeout expires, returns true if it is ready.
    /// Does not wait when every worker is busy: the pipeline would not even start in time.
    bool WaitFor(const Pipeline* pipeline, std::chrono::microseconds timeout) {
        if (pipeline->IsReady()) {
            return true;
        }
        std::unique_lock lk{mutex};
        if (pending > workers.size()) {
            return false;
        }
        return done_cv.wait_for(lk, timeout, [pipeline] { return pipeline->IsReady(); });
    }

    /// A queued pipeline whose draw was just skipped goes to the front (ISS_ASYNC_SKIP_DRAWS):
    /// what is on screen appears first.
    void Prioritize(const Pipeline* pipeline) {
        std::scoped_lock lk{mutex};
        const auto it = std::find(queue.begin(), queue.end(), pipeline);
        if (it == queue.end() || it == queue.begin()) {
            return;
        }
        Pipeline* found = *it;
        queue.erase(it);
        queue.push_front(found);
    }

    /// Takes a pipeline that no worker started out of the queue and compiles it on the calling
    /// thread (the one that needs it now).
    bool RunIfQueued(const Pipeline* pipeline) {
        Pipeline* found{};
        {
            std::scoped_lock lk{mutex};
            const auto it = std::find(queue.begin(), queue.end(), pipeline);
            if (it == queue.end()) {
                return false;
            }
            found = *it;
            queue.erase(it);
        }
        found->Compile();
        {
            std::scoped_lock lk{mutex};
            --pending;
        }
        done_cv.notify_all();
        return true;
    }

    /// Waits until every submitted pipeline is compiled.
    void WaitIdle() {
        std::unique_lock lk{mutex};
        done_cv.wait(lk, [this] { return pending == 0; });
    }

private:
    void WorkerLoop(std::stop_token stoken, u32 index) {
        const auto name = fmt::format("shadPS4:PipelineCompiler{}", index);
        Common::SetCurrentThreadName(name.c_str());
        // With lazy binds the recording thread waits for these pipelines: not below the game.
        Common::SetCurrentThreadPriority(LazyPipelines() ? Common::ThreadPriority::Normal
                                                         : Common::ThreadPriority::Low);
        while (!stoken.stop_requested()) {
            Pipeline* pipeline{};
            bool low = false;
            {
                std::unique_lock lk{mutex};
                Common::CondvarWait(work_cv, lk, stoken,
                                    [this] { return !queue.empty() || !low_queue.empty(); });
                if (stoken.stop_requested()) {
                    return;
                }
                if (!queue.empty()) {
                    pipeline = queue.front();
                    queue.pop_front();
                } else {
                    pipeline = low_queue.front();
                    low_queue.pop_front();
                    low = true;
                }
            }
            pipeline->Compile();
            if (low) continue;
            {
                std::scoped_lock lk{mutex};
                --pending;
            }
            done_cv.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable done_cv;
    std::deque<Pipeline*> queue;
    std::deque<Pipeline*> low_queue;
    u32 pending{};
    std::vector<std::jthread> workers;
};

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

static void BuildSwRuntimeInfo(Shader::RuntimeInfo& info, const AmdGpu::Regs& regs, SwStage l_stage,
                               const DrawIndirectParams& indirect_params) {
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = indirect_params.instance_sgpr_offset;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
}

// The compute stage's runtime info from these registers (BuildRuntimeInfo's compute case, any
// thread): the command processor's and the speculative request's are built the same way.
static void BuildComputeRuntimeInfoFrom(Shader::RuntimeInfo& info,
                                        const AmdGpu::ComputeProgram& cs_pgm) {
    info.Initialize(HwStage::Compute, SwStage::Compute);
    info.props.num_user_data = cs_pgm.settings.num_user_regs;
    info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
    info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
    info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
    info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
    info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
    info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                 cs_pgm.num_thread_z.full};
    info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                              cs_pgm.IsTgidEnabled(2)};
    info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
}

static bool SpecComputeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SPEC_COMPUTE");
        return !value || value[0] != '0';
    }();
    return enabled;
}

void PipelineCache::FillColorBuffers(
    const AmdGpu::Regs& regs,
    std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers>& color_buffers) {
    color_buffers.fill({});
    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }
}

bool PipelineCache::BuildRuntimeInfoFrom(
    Shader::RuntimeInfo& info, const AmdGpu::Regs& regs, HwStage stage, SwStage l_stage,
    const std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers>& color_buffers,
    const DrawIndirectParams& indirect_params) const {
    const auto BuildCommon = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local:
    case HwStage::Hull:
        // Read constants from memory through the hull shader: BuildRuntimeInfo.
        return false;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            switch (regs.tess_config.topology) {
            case AmdGpu::TessellationTopology::Point:
                info.hw.gs.in_primitive = AmdGpu::PrimitiveType::PointList;
                break;
            case AmdGpu::TessellationTopology::Line:
                info.hw.gs.in_primitive = AmdGpu::PrimitiveType::LineList;
                break;
            case AmdGpu::TessellationTopology::TriangleCw:
            case AmdGpu::TessellationTopology::TriangleCcw:
                info.hw.gs.in_primitive = AmdGpu::PrimitiveType::TriangleList;
                break;
            default:
                return false;
            }
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = color_buffers[i];
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case HwStage::Compute:
        return false;
    default:
        return false;
    }
    BuildSwRuntimeInfo(info, regs, l_stage, indirect_params);
    return true;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(
    HwStage stage, SwStage l_stage, iss::g4::CpOwner owner) {
    auto& info = owner == iss::g4::CpOwner::Graphics ? runtime_infos[u32(l_stage)]
                                                     : asc_lookup.runtime_infos[u32(l_stage)];
    const auto& color_buffers = owner == iss::g4::CpOwner::Graphics
                                    ? graphics_key.color_buffers : asc_lookup.color_buffers;
    const auto& indirect_params = owner == iss::g4::CpOwner::Graphics
                                      ? draw_indirect_params : asc_lookup.indirect_params;
    const auto& regs = liverpool->regs;
    if (BuildRuntimeInfoFrom(info, regs, stage, l_stage, color_buffers, indirect_params)) {
        if (stage == HwStage::Geometry) {
            DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                       "copy.bin");
        }
        return info;
    }
    const auto BuildCommon = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(SwStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Geometry:
        UNREACHABLE();
        break;
    case HwStage::Compute: {
        const auto& cs_pgm = active_snapshot_cs ? *active_snapshot_cs : liverpool->GetCsRegs();
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    BuildSwRuntimeInfo(info, regs, l_stage, indirect_params);
    return info;
}

namespace {
template <typename InfoT>
u64 SpecializationKey(const InfoT& info, const Shader::RuntimeInfo& runtime_info,
                      const Shader::Backend::Bindings& binding);
template <typename InfoT>
bool VertexAttributesMatch(const InfoT& info, const Shader::RuntimeInfo& runtime_info,
                           const Shader::StageSpecialization& spec);
} // Anonymous namespace

Program* PipelineCache::FindProgram(u64 hash) {
    std::shared_lock lock{program_cache_mutex};
    const auto it = program_cache.find(hash);
    return it != program_cache.end() ? it->second.get() : nullptr;
}

namespace {

/// What the specialization key reads of a stage, for the scout thread (Shader::Info is the
/// command processor's, with its own user data and flattened buffer).
struct ScoutInfo {
    std::span<const u32> user_data;
    std::span<u32> flattened_ud_buf;
    VAddr pgm_base;
    bool has_fetch_shader;
    u32 fetch_shader_sgpr_base;
    Shader::SwStage sw_stage;
    const Shader::BufferResourceList& buffers;
    const Shader::ImageResourceList& images;
    const Shader::FMaskResourceList& fmasks;
    const Shader::SamplerResourceList& samplers;

    template <typename T>
    T ReadUdSharp(u32 sharp_idx) const noexcept {
        T data;
        std::memcpy(&data, &flattened_ud_buf[sharp_idx], sizeof(T));
        return data;
    }

    template <typename T>
    T ReadUdReg(u32 ptr_index, u32 dword_offset) const noexcept {
        T data;
        const u32* base = user_data.data();
        if (ptr_index != Shader::IR::NumScalarRegs) {
            std::memcpy(&base, &user_data[ptr_index], sizeof(base));
            base = reinterpret_cast<const u32*>(VAddr(base) & 0xFFFFFFFFFFFFULL);
        }
        std::memcpy(&data, base + dword_offset, sizeof(T));
        return data;
    }
};

} // Anonymous namespace

void PipelineCache::ScoutDraw(const AmdGpu::Regs& regs, AmdGpu::ScoutDraw& draw,
                              AmdGpu::ScoutArena& arena) {
    draw.num_stages = 0;
    std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers> color_buffers;
    FillColorBuffers(regs, color_buffers);
    const DrawIndirectParams indirect_params{
        .vertex_sgpr_offset = draw.vertex_sgpr_offset,
        .instance_sgpr_offset = draw.instance_sgpr_offset,
    };
    Shader::Backend::Bindings binding{};
    // Mirrors RefreshGraphicsStages: the stages in the order they are bound, the bindings
    // accumulating. Returns false when the stages after this one cannot be precomputed.
    const auto scout_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        if (!regs.stage_enable.IsStageEnabled(static_cast<u32>(stage_in))) {
            return true;
        }
        const auto* pgm = regs.ProgramForStage(static_cast<u32>(stage_in));
        if (!pgm || !pgm->Address<u32*>()) {
            return true;
        }
        if (draw.num_stages >= AmdGpu::ScoutDraw::MaxStages) {
            return false;
        }
        const auto params = AmdGpu::GetParams(*pgm);
        auto& out = draw.stages[draw.num_stages++];
        // Entries are reused without being cleared: every field read is set below.
        out.valid = false;
        out.resolved = false;
        out.num_image_keys = 0;
        out.num_buffer_ranges = 0;
        out.hw_stage = static_cast<u8>(stage_in);
        out.sw_stage = static_cast<u8>(stage_out);
        out.pgm_base = params.Base();
        out.start = binding;
        out.ud_hash = XXH3_64bits(params.user_data.data(), params.user_data.size_bytes());
        Program* program = FindProgram(params.hash);
        if (program == nullptr) {
            // Not compiled yet: the command processor compiles it, and only it knows the
            // bindings of the stages after it. Its translation (independent of the bindings)
            // can start now (ISS_SPEC_TRANSLATE).
            if (SpecTranslateEnabled() &&
                (stage_in == HwStage::Fragment || stage_in == HwStage::Vertex)) {
                Shader::RuntimeInfo spec_runtime_info{};
                if (BuildRuntimeInfoFrom(spec_runtime_info, regs, stage_in, stage_out,
                                         color_buffers, indirect_params)) {
                    RequestSpecTranslation(stage_in, stage_out, params, spec_runtime_info);
                }
            }
            return false;
        }
        const auto& pinfo = program->info;
        // The scout fills every field it needs below. Copying the graphics lookup scratch
        // races the graphics command processor while it refreshes another stage.
        Shader::RuntimeInfo runtime_info{};
        if (!BuildRuntimeInfoFrom(runtime_info, regs, stage_in, stage_out, color_buffers,
                                  indirect_params)) {
            return false;
        }
        out.SetRuntimeInfo(runtime_info);
        // The user data, flattened with the tables it points to (as Info::RefreshFlatBuf).
        const u32 flat_size = pinfo.srt_info.flattened_bufsize_dw;
        u32* flat = arena.Allocate(std::max<u32>(flat_size, 1));
        if (flat == nullptr) {
            return false;
        }
        std::memcpy(flat, params.user_data.data(),
                    std::min<size_t>(params.user_data.size_bytes(), flat_size * sizeof(u32)));
        if (pinfo.srt_info.walker_func) {
            // Tables the command processor itself writes before this draw are not readable
            // yet: no entry for the stage.
            const auto* scout = liverpool->GetScout();
            for (u32 i = 0; i + 1 < params.user_data.size(); ++i) {
                VAddr pointer;
                std::memcpy(&pointer, &params.user_data[i], sizeof(pointer));
                pointer &= 0xFFFFFFFFFFFFULL;
                if (pointer != 0 && scout != nullptr && scout->MayBeWrittenLater(pointer, 4_KB)) {
                    binding.buffer += static_cast<u32>(pinfo.buffers.size());
                    binding.unified += static_cast<u32>(pinfo.buffers.size() +
                                                        pinfo.images.size() +
                                                        pinfo.samplers.size());
                    return true;
                }
            }
            pinfo.srt_info.walker_func(params.user_data.data(), flat);
        }
        const ScoutInfo shim{
            .user_data = params.user_data,
            .flattened_ud_buf = std::span<u32>{flat, flat_size},
            .pgm_base = params.Base(),
            .has_fetch_shader = pinfo.has_fetch_shader,
            .fetch_shader_sgpr_base = pinfo.fetch_shader_sgpr_base,
            .sw_stage = stage_out,
            .buffers = pinfo.buffers,
            .images = pinfo.images,
            .fmasks = pinfo.fmasks,
            .samplers = pinfo.samplers,
        };
        out.memo_key = SpecializationKey(shim, runtime_info, binding);
        out.flat = flat;
        out.flat_size = flat_size;
        // Lookups of the stage's resources (Tuning::bind_prefetch), as the command processor
        // makes them from the same flattened user data: texture memo keys of the image
        // resources in order, ranges of the buffers read only.
        out.num_image_keys = 0;
        out.num_buffer_ranges = 0;
        if (Common::Perf::GetTuning().bind_prefetch.load(std::memory_order_relaxed)) {
            for (const auto& image_desc : pinfo.images) {
                if (out.num_image_keys == AmdGpu::ScoutStage::MaxImageKeys) {
                    break;
                }
                out.image_keys[out.num_image_keys++] =
                    TexMemoKey(image_desc.GetSharp(shim), 0, image_desc);
            }
            for (const auto& desc : pinfo.buffers) {
                if (out.num_buffer_ranges == AmdGpu::ScoutStage::MaxBufferRanges) {
                    break;
                }
                if (desc.IsSpecial() || desc.is_written) {
                    continue;
                }
                const auto vsharp = desc.GetSharp(shim);
                if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                    continue;
                }
                out.buffer_addresses[out.num_buffer_ranges] = vsharp.base_address;
                out.buffer_sizes[out.num_buffer_ranges] = static_cast<u32>(vsharp.GetSize());
                ++out.num_buffer_ranges;
            }
        }
        out.program = program;
        out.params_hash = params.hash;
        out.code_dwords = static_cast<u32>(params.code.size());
        std::memcpy(out.user_data.data(), params.user_data.data(), sizeof(out.user_data));
        // The permutation the program's memo has for the key, checked as the command
        // processor would (Tuning::scout_fast: the command processor then takes it as is).
        u32 perm_idx;
        if (program->ReadMemo(out.memo_key, perm_idx)) {
            // The permutations change under the command processor's exclusive lock.
            std::shared_lock lock{program_cache_mutex};
            if (perm_idx < program->modules.size() &&
                VertexAttributesMatch(shim, runtime_info, program->modules[perm_idx].spec)) {
                out.perm_idx = perm_idx;
                out.resolved = true;
            }
        }
        out.valid = true;
        binding.buffer += static_cast<u32>(pinfo.buffers.size());
        binding.unified +=
            static_cast<u32>(pinfo.buffers.size() + pinfo.images.size() + pinfo.samplers.size());
        return true;
    };
    if (!scout_stage(HwStage::Fragment, SwStage::Fragment)) {
        return;
    }
    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        scout_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (scout_stage(HwStage::Export, SwStage::Vertex)) {
            scout_stage(HwStage::Geometry, SwStage::Geometry);
        }
        break;
    default:
        // Tessellation: the stages read constants from memory, not precomputed.
        break;
    }
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .lds_barriers_multi_wave =
            Common::Perf::GetTuning().lds_barriers_multi_wave.load(std::memory_order_relaxed),
        .wave_memory_barriers =
            Common::Perf::GetTuning().wave_memory_barriers.load(std::memory_order_relaxed),
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    // Each recompiler option changing compute shaders has its own bit: cached shaders compiled
    // with other options are compiled again once.
    const bool lds_multi_wave = profile.needs_lds_barriers && profile.lds_barriers_multi_wave;
    // ISS_DYNAMIC_STRIDE=1 (and ISS_AOT_COMPUTE, which needs it): compute shaders read buffer
    // strides from their V#. Modules cached with a compiled-in stride are compiled again once.
    {
        const char* dynamic = std::getenv("ISS_DYNAMIC_STRIDE");
        const char* aot = std::getenv("ISS_AOT_COMPUTE");
        const char* stats = std::getenv("ISS_AOT_FORMAT_STATS"); // Reads the cache as it is.
        Shader::dynamic_buffer_stride = (dynamic && dynamic[0] == '1') ||
                                        (aot && aot[0] != '\0' && !(stats && stats[0] == '1'));
    }
    SetComputeShaderVariant((lds_multi_wave ? 1u : 0u) | (profile.wave_memory_barriers ? 2u : 0u) |
                            (Shader::dynamic_buffer_stride ? 4u : 0u));
    LOG_INFO(Render_Vulkan,
             "Compute shader LDS barriers for multi wave workgroups: {}, buffer memory barriers "
             "within split GCN waves: {}",
             lds_multi_wave, profile.wave_memory_barriers);
    // Populate the actual Vulkan cache before preloading known pipelines, so both the
    // preload and later new variations can reuse driver compilation data.
    driver_cache_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
        (std::filesystem::path{Common::ElfInfo::Instance().GameSerial()}.filename().string() + ".vkcache");
    std::vector<u8> stored;
    std::span<const u8> initial;
    const auto properties = instance.GetPhysicalDevice().getProperties();
    if (EmulatorSettings.IsPipelineCacheEnabled() && std::filesystem::exists(driver_cache_path)) {
        Common::FS::IOFile file{driver_cache_path, Common::FS::FileAccessMode::Read};
        if (file.IsOpen() && file.GetSize() <= 256ULL * 1024 * 1024) {
            stored.resize(file.GetSize());
            file.Read(stored);
            initial = UnpackDriverCache(stored, properties.vendorID, properties.deviceID,
                {properties.pipelineCacheUUID.data(), properties.pipelineCacheUUID.size()});
        }
        LOG_INFO(Render_Vulkan, "Vulkan driver cache: {} bytes accepted for this device", initial.size());
    }
    const vk::PipelineCacheCreateInfo cache_info{
        .initialDataSize = initial.size(), .pInitialData = initial.empty() ? nullptr : initial.data()};
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique(cache_info);
    if (cache_result != vk::Result::eSuccess && !initial.empty()) {
        LOG_WARNING(Render_Vulkan, "Stored Vulkan cache rejected; starting a fresh cache");
        auto fallback = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = fallback.result;
        cache = std::move(fallback.value);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    u32 num_compilers = std::clamp(std::thread::hardware_concurrency() / 4, 2U, 8U);
    if (LazyPipelines()) {
        // A burst of new pipelines compiles in parallel (ISS_PIPELINE_COMPILERS overrides).
        num_compilers = std::clamp(std::thread::hardware_concurrency() * 5 / 8, 2U, 12U);
    }
    if (const char* value = std::getenv("ISS_PIPELINE_COMPILERS"); value && *value) {
        num_compilers = std::clamp<u32>(std::atoi(value), 1U, 32U);
    }
    compiler = std::make_unique<PipelineCompiler>(num_compilers);
    if (SpecTranslateEnabled()) {
        spec_translator = std::make_unique<SpecTranslator>(profile, 4);
    }
    const char* preload_option = std::getenv("SHADPS4_PARALLEL_PRELOAD");
    parallel_preload = preload_option && preload_option[0] == '1';
    WarmUp();
    LOG_INFO(Render_Vulkan, "Pipelines can be compiled by {} worker threads", num_compilers);
    if (Storage::DataBase::Instance().IsOpened()) {
        Storage::DataBase::Instance().SetBeforeClose([this] { SaveDriverCache(); });
    }
    if (const char* aot_index = std::getenv("ISS_AOT_COMPUTE"); aot_index && *aot_index) {
        StartAotCompute(aot_index);
    }
}

PipelineCache::~PipelineCache() {
    aot.reset(); // Its threads use the device, the profile and the Vulkan pipeline cache.
    spec_translator.reset();
    if (graphics_reuse_checks) {
        std::fprintf(stderr, "PIPELINE_STATE_REUSE checks=%llu hits=%llu verified=%llu\n",
                     static_cast<unsigned long long>(graphics_reuse_checks),
                     static_cast<unsigned long long>(graphics_reuse_hits),
                     static_cast<unsigned long long>(graphics_reuse_verified));
    }
    SaveDriverCache();
    Storage::DataBase::Instance().SetBeforeClose({});
    // Workers reference the pipelines, stop them first.
    compiler.reset();
}

// ---------------------------------------------------------------------------------------------
// ISS_AOT_COMPUTE: ahead-of-time compute shaders (Claude, 2026-10-19).
//
// The index (scripts/aot_index.py, made from the player's own game files) gives every compute
// shader of the game: its code (pack and offset) and the registers the game dispatches it with.
// Worker threads at low priority translate each shader the way GetProgram would for a new
// program, with no game state bound (every V# reads as null: with ISS_DYNAMIC_STRIDE the module of
// a buffers-only shader does not depend on any V# field the specialization does not check), emit
// its SPIR-V, create its module and compile its pipeline. The command processor adopts the
// finished ones between dispatches (DrainAot): program, permutation 0, pipeline and the cache
// entries, so the next sessions preload them like any other. A shader the game reached first is
// compiled by the normal path and the AOT result is dropped; a binding the AOT permutation does
// not fit (an image of another type, a formatted buffer...) gets its own permutation, as before.
// ---------------------------------------------------------------------------------------------
} // namespace Vulkan
#include <windows.h>
#include <psapi.h>
namespace Vulkan {
/// ISS_AOT_TRACE=1: names the module and offset of an access violation before the probe's
/// handler ends the process.
static LONG CALLBACK AotTraceFault(EXCEPTION_POINTERS* e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto rip = reinterpret_cast<uintptr_t>(e->ExceptionRecord->ExceptionAddress);
    if (rip >= 0x10000000 && rip < 0x20000000) {
        return EXCEPTION_CONTINUE_SEARCH; // Guest code: TLS reads the probe emulates.
    }
    HMODULE module{};
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(rip), &module)) {
        GetModuleFileNameA(module, name, MAX_PATH);
    }
    PWSTR description{};
    char thread_name[128] = "?";
    if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description) {
        WideCharToMultiByte(CP_UTF8, 0, description, -1, thread_name, sizeof(thread_name), nullptr,
                            nullptr);
        LocalFree(description);
    }
    std::fprintf(stderr, "AOT_TRACE_FAULT thread=%s\n", thread_name);
    std::fprintf(stderr, "AOT_TRACE_FAULT rip=%p module=%s offset=%#llx tid=%lu accessed=%#llx\n",
                 (void*)rip, name, (unsigned long long)(rip - reinterpret_cast<uintptr_t>(module)),
                 GetCurrentThreadId(),
                 (unsigned long long)(e->ExceptionRecord->NumberParameters >= 2
                                          ? e->ExceptionRecord->ExceptionInformation[1]
                                          : 0));
    // The return addresses on the stack that belong to modules: a rough backtrace.
    const auto* sp = reinterpret_cast<const uintptr_t*>(e->ContextRecord->Rsp);
    ULONG_PTR stack_low{}, stack_high{};
    GetCurrentThreadStackLimits(&stack_low, &stack_high);
    const int words = int(std::min<uintptr_t>(512, (stack_high - e->ContextRecord->Rsp) / 8));
    for (int i = 0, n = 0; i < words && n < 16; ++i) {
        HMODULE m{};
        if (sp[i] > 0x10000 && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                  reinterpret_cast<LPCSTR>(sp[i]), &m)) {
            char mn[MAX_PATH] = "?";
            GetModuleFileNameA(m, mn, MAX_PATH);
            const char* base = std::strrchr(mn, '\\');
            std::fprintf(stderr, "  frame %s+%#llx\n", base ? base + 1 : mn,
                         (unsigned long long)(sp[i] - reinterpret_cast<uintptr_t>(m)));
            ++n;
        }
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

/// The shader index of the player's own game files (scripts/aot_index.py does the same):
/// every .xpps pack ('KCAP', data base at 0x28) holds PS4 shader binaries, each one's code
/// starting with s_mov_b32 vcc_hi, N and followed (N + 1) * 8 bytes later by its OrbShdr footer
/// (type, length, hash); for compute shaders the Gnmx CsShader structure has the program offset
/// from the data base, then rsrc1, rsrc2 and the thread counts.
static bool BuildShaderIndex(const std::filesystem::path& game_dir, const std::filesystem::path& out) {
    struct Row {
        u32 r1, r2, x, y, z, dwords, type;
        std::string rel;
        u64 offset;
    };
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (auto it = std::filesystem::recursive_directory_iterator(game_dir, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".xpps") files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    std::mutex mutex;
    std::map<u64, Row> rows;
    std::atomic<size_t> next{0};
    constexpr u32 Mark = 0xBEEB03FF;
    const auto scan = [&](const std::filesystem::path& path) {
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (file == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER size{};
        GetFileSizeEx(file, &size);
        HANDLE mapping = size.QuadPart >= 0x100
                             ? CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
        const u8* data = mapping ? static_cast<const u8*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0))
                                 : nullptr;
        if (data && std::memcmp(data, "KCAP", 4) == 0) {
            const size_t n = size_t(size.QuadPart);
            const auto u32_at = [&](size_t at) {
                u32 v;
                std::memcpy(&v, data + at, 4);
                return v;
            };
            const u32 base = u32_at(0x28);
            std::unordered_map<u32, std::pair<u64, u32>> compute; // code start -> hash, dwords
            std::vector<std::pair<u64, Row>> found;
            const std::string_view view{reinterpret_cast<const char*>(data), n};
            const auto rel = std::filesystem::relative(path, game_dir, ec).generic_u8string();
            for (size_t i = view.find("OrbShdr"); i != std::string_view::npos && i + 32 <= n;
                 i = view.find("OrbShdr", i + 7)) {
                const u32 word = u32_at(i + 8);
                const u32 type = (word >> 2) & 0xF;
                if (type != 0 && type != 1 && type != 4) continue;
                const u32 length = word >> 8;
                u64 hash;
                std::memcpy(&hash, data + i + 16, 8);
                for (u64 k = 1; k * 8 <= i && k * 8 <= u64(length) + 0x4000; ++k) {
                    const size_t j = i - k * 8;
                    if (u32_at(j) != Mark || u32_at(j + 4) != k - 1) continue;
                    if (type == 4) {
                        if (j >= base) compute.emplace(u32(j - base), std::pair{hash, length / 4});
                    } else {
                        found.push_back({hash, Row{0, 0, 0, 0, 0, length / 4, type,
                                                   std::string(rel.begin(), rel.end()), j}});
                    }
                    break;
                }
            }
            if (!compute.empty()) {
                for (size_t k = 0; k + 28 <= n; k += 4) {
                    const u32 pgm = u32_at(k);
                    const auto it = compute.find(pgm);
                    if (it == compute.end() || u32_at(k + 4) != 0) continue;
                    const u32 r1 = u32_at(k + 8), r2 = u32_at(k + 12);
                    const u32 x = u32_at(k + 16), y = u32_at(k + 20), z = u32_at(k + 24);
                    if (!(x && x <= 1024 && y && y <= 1024 && z && z <= 64 && x * y * z <= 1024) ||
                        ((r2 >> 1) & 0x1F) > 16) {
                        continue;
                    }
                    found.push_back({it->second.first,
                                     Row{r1, r2, x, y, z, it->second.second, 4,
                                         std::string(rel.begin(), rel.end()), u64(pgm) + base}});
                }
            }
            std::scoped_lock lock{mutex};
            for (auto& [hash, row] : found) rows.try_emplace(hash, std::move(row));
        }
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        CloseHandle(file);
    };
    {
        std::vector<std::jthread> workers;
        for (u32 t = 0; t < std::max(2U, std::thread::hardware_concurrency() / 2); ++t) {
            workers.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < files.size();) {
                    scan(files[i]);
                    iss::g_preload_entries.store(unsigned(next.load()), std::memory_order_relaxed);
                }
            });
        }
    }
    std::vector<u8> bytes(12);
    std::memcpy(bytes.data(), "ISSAOT01", 8);
    const u32 count = u32(rows.size());
    std::memcpy(bytes.data() + 8, &count, 4);
    for (const auto& [hash, row] : rows) {
        const u32 regs[7] = {row.r1, row.r2, row.x, row.y, row.z, row.dwords, row.type};
        const u16 len = u16(row.rel.size());
        const size_t at = bytes.size();
        bytes.resize(at + 8 + 28 + 2 + len + 8);
        std::memcpy(bytes.data() + at, &hash, 8);
        std::memcpy(bytes.data() + at + 8, regs, 28);
        std::memcpy(bytes.data() + at + 36, &len, 2);
        std::memcpy(bytes.data() + at + 38, row.rel.data(), len);
        std::memcpy(bytes.data() + at + 38 + len, &row.offset, 8);
    }
    std::filesystem::create_directories(out.parent_path(), ec);
    Common::FS::IOFile file{out, Common::FS::FileAccessMode::Create};
    return file.IsOpen() && file.WriteSpan(std::span<const u8>{bytes}) == bytes.size() && count > 0;
}

/// ISS_AOT_COMPUTE / ISS_AOT_INDEX = "auto": the index in the shader cache folder, made from the
/// game files the first time (the loading screen says so); any other value is the index path.
std::string ResolveShaderIndex(const char* value) {
    if (!value || std::string_view{value} != "auto") return value ? value : "";
    static std::once_flag once;
    static std::string resolved;
    std::call_once(once, [] {
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "shaders.idx";
        resolved = path.string();
        if (std::filesystem::exists(path)) return;
        const char* game_dir = std::getenv("ISS_GAME_DIR");
        if (!game_dir) return;
        const u64 start = Common::Perf::NowNs();
        iss::g_preload_total.store(1658, std::memory_order_relaxed); // the game's pack count
        iss::g_preload_entries.store(0, std::memory_order_relaxed);
        iss::g_preload_phase.store(5, std::memory_order_relaxed);
        const bool ok = BuildShaderIndex(std::filesystem::path{game_dir}, path);
        iss::g_preload_phase.store(3, std::memory_order_relaxed);
        std::fprintf(stderr, "AOT_INDEX built=%d %s in %.1f s\n", ok ? 1 : 0, resolved.c_str(),
                     double(Common::Perf::NowNs() - start) / 1e9);
    });
    return resolved;
}

void PipelineCache::StartAotCompute(const char* index_value) {
    const std::string index_string = ResolveShaderIndex(index_value);
    const char* index_path = index_string.c_str();
    if (const char* value = std::getenv("ISS_AOT_TRACE"); value && value[0] == '1') {
        AddVectoredExceptionHandler(1, AotTraceFault);
    }
    if (const char* value = std::getenv("ISS_AOT_FORMAT_STATS"); value && value[0] == '1') {
        // The formats of the formatted buffers of the compute permutations preloaded: what the
        // game binds to them (per shader and over all).
        std::map<std::tuple<u32, u32, u32, u32>, u32> formats;
        u32 programs = 0, formatted_programs = 0, varying = 0;
        std::shared_lock lock{program_cache_mutex};
        for (const auto& [hash, program] : program_cache) {
            if (program->info.hw_stage != HwStage::Compute) continue;
            ++programs;
            std::set<std::vector<u32>> per_program;
            for (const auto& module : program->modules) {
                std::vector<u32> key;
                for (size_t i = 0; i < module.spec.buffers.size(); ++i) {
                    const auto& b = module.spec.buffers[i];
                    if (!b.is_formatted) continue;
                    u32 swizzle{};
                    std::memcpy(&swizzle, &b.dst_select, std::min(sizeof(swizzle), sizeof(b.dst_select)));
                    ++formats[{b.data_format, b.num_format, swizzle, u32(b.num_conversion)}];
                    key.insert(key.end(), {u32(i), b.data_format, b.num_format});
                }
                if (!key.empty()) per_program.insert(key);
            }
            formatted_programs += !per_program.empty();
            varying += per_program.size() > 1;
        }
        std::fprintf(stderr, "AOT_FORMAT_STATS compute programs=%u with formatted=%u varying=%u\n",
                     programs, formatted_programs, varying);
        for (const auto& [f, n] : formats) {
            std::fprintf(stderr, "AOT_FORMAT dfmt=%u nfmt=%u swizzle=%#x conv=%u count=%u\n",
                         std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f), n);
        }
    }
    std::vector<u8> bytes;
    {
        Common::FS::IOFile file{std::filesystem::path{index_path}, Common::FS::FileAccessMode::Read};
        if (!file.IsOpen()) {
            std::fprintf(stderr, "AOT_COMPUTE index not found: %s\n", index_path);
            return;
        }
        bytes.resize(file.GetSize());
        file.Read(bytes);
    }
    const char* game_dir = std::getenv("ISS_GAME_DIR");
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "ISSAOT01", 8) != 0 || !game_dir) {
        std::fprintf(stderr, "AOT_COMPUTE index unusable (or ISS_GAME_DIR unset)\n");
        return;
    }
    auto state = std::make_unique<AotCompute>();
    state->device = instance.GetDevice();
    u32 count;
    std::memcpy(&count, bytes.data() + 8, 4);
    size_t at = 12;
    const std::filesystem::path root{game_dir};
    for (u32 i = 0; i < count; ++i) {
        AotCompute::Entry e{};
        u32 regs[7];
        u16 path_len;
        if (at + 8 + 28 + 2 > bytes.size()) break;
        std::memcpy(&e.hash, bytes.data() + at, 8);
        std::memcpy(regs, bytes.data() + at + 8, 28);
        std::memcpy(&path_len, bytes.data() + at + 36, 2);
        at += 38;
        if (at + path_len + 8 > bytes.size()) break;
        const std::string rel(reinterpret_cast<const char*>(bytes.data() + at), path_len);
        at += path_len;
        std::memcpy(&e.offset, bytes.data() + at, 8);
        at += 8;
        e.rsrc1 = regs[0];
        e.rsrc2 = regs[1];
        e.threads_x = regs[2];
        e.threads_y = regs[3];
        e.threads_z = regs[4];
        e.dwords = regs[5];
        if (e.threads_x == 0) continue; // A pixel or vertex shader (the index has them too).
        e.path = (root / std::filesystem::path{std::u8string(rel.begin(), rel.end())}).string();
        e.pack = AotCompute::PackKey(rel);
        if (FindProgram(e.hash) != nullptr) {
            state->known.fetch_add(1, std::memory_order_relaxed);
            continue; // Preloaded from the cache.
        }
        state->entries.push_back(std::move(e));
    }
    // One pack open for its run of shaders.
    std::ranges::stable_sort(state->entries, {}, &AotCompute::Entry::path);
    state->taken = std::vector<std::atomic<u8>>(state->entries.size());
    state->state = std::vector<std::atomic<u8>>(state->entries.size());
    for (u32 i = 0; i < state->entries.size(); ++i) state->entry_of.emplace(state->entries[i].hash, i);
    for (u32 i = 0; i < state->entries.size();) {
        u32 end = i;
        while (end < state->entries.size() && state->entries[end].pack == state->entries[i].pack) ++end;
        state->packs.emplace(state->entries[i].pack, std::pair{i, end});
        i = end;
    }
    // ISS_AOT_WAIT=1: the game starts once every compute shader is compiled (the first run
    // after installing or after a driver update; later runs preload them from the cache), with
    // every core; otherwise a few low priority threads compile them while the game runs.
    const bool wait = [] {
        const char* value = std::getenv("ISS_AOT_WAIT");
        return value && value[0] == '1';
    }();
    u32 threads = wait ? std::max(1U, std::thread::hardware_concurrency() - 1)
                       : std::clamp(std::thread::hardware_concurrency() / 4, 2U, 6U);
    if (const char* value = std::getenv("ISS_AOT_THREADS"); value && *value) {
        threads = std::clamp<u32>(std::atoi(value), 1U, 16U);
    }
    state->start_ns = Common::Perf::NowNs();
    std::fprintf(stderr, "AOT_COMPUTE start: %zu shaders to compile, %llu already cached, %u threads\n",
                 state->entries.size(), (unsigned long long)state->known.load(), threads);
    AotCompute* const s = state.get();
    for (u32 t = 0; t < threads; ++t) {
        state->running.fetch_add(1, std::memory_order_relaxed);
        state->workers.emplace_back([this, s, t, wait](std::stop_token stop) {
            Common::SetCurrentThreadName(fmt::format("iss:AotCompute{}", t).c_str());
            if (!wait) Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
            Shader::Info::aot_translation = true;
            iss::assert_throws = true;
            static const bool trace = [] {
                const char* value = std::getenv("ISS_AOT_TRACE");
                return value && value[0] == '1';
            }();
            if (trace) {
                std::fprintf(stderr, "AOT_COMPUTE thread %u tid=%lu dll_fn=%p\n", t,
                             GetCurrentThreadId(), (void*)&BuildComputeRuntimeInfoFrom);
            }
            static const std::array<u32, Shader::NUM_USER_DATA_REGS> no_user_data{};
            std::FILE* pack = nullptr;
            std::string pack_path;
            std::vector<u32> code;
            Shader::Pools pools;
            for (size_t i; !stop.stop_requested() && (i = s->Take()) != SIZE_MAX;) {
                const auto& e = s->entries[i];
                s->state[i].store(1, std::memory_order_relaxed);
                if (FindProgram(e.hash) != nullptr) {
                    s->known.fetch_add(1, std::memory_order_relaxed);
                    s->state[i].store(5, std::memory_order_relaxed);
                    continue; // The game reached it first.
                }
                if (pack_path != e.path) {
                    if (pack) std::fclose(pack);
                    pack = std::fopen(e.path.c_str(), "rb");
                    pack_path = e.path;
                }
                code.resize(e.dwords);
                if (!pack || _fseeki64(pack, s64(e.offset), SEEK_SET) != 0 ||
                    std::fread(code.data(), 4, e.dwords, pack) != e.dwords) {
                    s->failed.fetch_add(1, std::memory_order_relaxed);
                    s->state[i].store(4, std::memory_order_relaxed);
                    continue;
                }
                AmdGpu::ComputeProgram cs{};
                const u32 settings[2] = {e.rsrc1, e.rsrc2};
                std::memcpy(&cs.settings, settings, sizeof(settings));
                cs.num_thread_x.full = u16(e.threads_x);
                cs.num_thread_y.full = u16(e.threads_y);
                cs.num_thread_z.full = u16(e.threads_z);
                Shader::RuntimeInfo runtime_info{};
                BuildComputeRuntimeInfoFrom(runtime_info, cs);
                const Shader::ShaderParams params{
                    .user_data = no_user_data, .code = code, .hash = e.hash};
                AotCompute::Result r{};
                if (trace) {
                    std::fprintf(stderr, "AOT_BEGIN %llx tid=%lu\n", (unsigned long long)e.hash,
                                 GetCurrentThreadId());
                }
                try {
                    const u64 t0 = Common::Perf::NowNs();
                    r.program = std::make_unique<Program>(HwStage::Compute, SwStage::Compute, params);
                    r.program->info.user_data = r.program->replay_user_data;
                    auto compile_runtime_info = runtime_info;
                    const auto ir = Shader::TranslateProgram(code, pools, r.program->info,
                                                             compile_runtime_info, profile);
                    // Only buffers-only shaders (formatted ones read the typical V#,
                    // resource_patching_pass.cpp). The program's info is shared by its later
                    // permutations (GetProgram builds their pipelines from it), so it must be
                    // the info a translation with the game's V#s gives: images and samplers
                    // are left to the game.
                    const auto& info = r.program->info;
                    const bool state_free =
                        info.images.empty() && info.samplers.empty() && info.fmasks.empty() &&
                        std::ranges::none_of(info.buffers, [](const auto& b) {
                            return b.buffer_type == Shader::BufferType::Guest &&
                                   b.sharp_fetch.summary ==
                                       decltype(b.sharp_fetch.summary)::Invalid;
                        });
                    if (!state_free) {
                        pools.ReleaseContents();
                        s->stateful.fetch_add(1, std::memory_order_relaxed);
                        s->state[i].store(2, std::memory_order_relaxed);
                        if (!info.images.empty()) s->why[0].fetch_add(1);
                        if (!info.samplers.empty()) s->why[1].fetch_add(1);
                        if (!info.fmasks.empty()) s->why[2].fetch_add(1);
                        if (std::ranges::any_of(info.buffers, [](const auto& b) { return b.is_formatted; }))
                            s->why[3].fetch_add(1);
                        if (std::ranges::any_of(info.buffers, [](const auto& b) {
                                return b.buffer_type == Shader::BufferType::Guest &&
                                       b.sharp_fetch.summary ==
                                           decltype(b.sharp_fetch.summary)::Invalid;
                            }))
                            s->why[4].fetch_add(1);
                        continue;
                    }
                    Shader::Backend::Bindings binding{};
                    r.spv = Shader::Backend::SPIRV::EmitSPIRV(profile, compile_runtime_info, ir,
                                                              binding);
                    pools.ReleaseContents();
                    r.spec = Shader::StageSpecialization(r.program->info, compile_runtime_info,
                                                         profile, Shader::Backend::Bindings{});
                    {
                        std::vector<u32> salted;
                        r.module = CompileSPV(iss::shader_salt::Apply(r.spv, salted), s->device);
                    }
                    const u64 t1 = Common::Perf::NowNs();
                    r.key.value = HashCombine(e.hash, u64{0});
                    ComputePipeline::SerializationSupport sdata{};
                    r.pipeline = std::make_unique<ComputePipeline>(
                        instance, scheduler, desc_heap, profile, *pipeline_cache, r.key,
                        r.program->info, r.module, sdata, true, true);
                    r.pipeline->Compile();
                    const u64 t2 = Common::Perf::NowNs();
                    s->translate_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
                    s->pipeline_ns.fetch_add(t2 - t1, std::memory_order_relaxed);
                } catch (const std::exception&) {
                    pools.ReleaseContents();
                    r.pipeline.reset();
                    if (r.module) s->device.destroyShaderModule(r.module);
                    s->failed.fetch_add(1, std::memory_order_relaxed);
                    s->state[i].store(4, std::memory_order_relaxed);
                    continue;
                }
                s->compiled.fetch_add(1, std::memory_order_relaxed);
                s->state[i].store(3, std::memory_order_relaxed);
                {
                    std::scoped_lock lock{s->mutex};
                    s->ready.push_back(std::move(r));
                }
                s->num_ready.fetch_add(1, std::memory_order_release);
            }
            if (pack) std::fclose(pack);
            s->running.fetch_sub(1, std::memory_order_acq_rel);
        });
    }
    {
        std::scoped_lock lock{g_aot_mutex};
        g_aot = state.get();
        for (const auto& pack : g_early_packs) state->Prioritize(pack);
        g_early_packs.clear();
    }
    aot = std::move(state);
    if (wait && !s->entries.empty()) {
        // The loading screen shows the progress (bridge.cpp, phase 4).
        iss::g_preload_total.store(unsigned(s->entries.size()), std::memory_order_relaxed);
        iss::g_preload_entries.store(0, std::memory_order_relaxed);
        iss::g_preload_phase.store(4, std::memory_order_relaxed);
        while (s->running.load(std::memory_order_acquire) != 0 ||
               s->num_ready.load(std::memory_order_acquire) != 0) {
            DrainAot();
            iss::g_preload_entries.store(
                unsigned(s->compiled.load() + s->stateful.load() + s->failed.load() +
                         s->known.load()),
                std::memory_order_relaxed);
            if (s->num_ready.load(std::memory_order_acquire) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        DrainAot(); // The report.
        iss::g_preload_phase.store(3, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------------------------
// ISS_AOT_RECIPES: a pipeline recipe (its .key and the .meta of its stages: specialization,
// runtime info and the shader's resource layout, no code) turned into SPIR-V from the player's
// own game files. The V#/T#/S# words the specialization implies are written over the null
// flattened buffer, the vertex attribute V#s into tables the user data points at, and the fetch
// shader data comes from the recipe (the game builds fetch shaders at run time). With
// ISS_AOT_RECIPE_VERIFY=1 every stage that has its SPIR-V is regenerated and compared with it.
// ---------------------------------------------------------------------------------------------
namespace {
struct ShaderLoc {
    std::string path;
    u64 offset;
    u32 dwords;
};
std::once_flag g_locs_once;
std::unordered_map<u64, ShaderLoc> g_locs;

const std::unordered_map<u64, ShaderLoc>& ShaderLocations() {
    std::call_once(g_locs_once, [] {
        const char* index = std::getenv("ISS_AOT_INDEX");
        if (!index || !*index) index = std::getenv("ISS_AOT_COMPUTE");
        static std::string resolved; // "auto": the cache folder's (ResolveShaderIndex)
        resolved = ResolveShaderIndex(index);
        index = resolved.c_str();
        const char* game_dir = std::getenv("ISS_GAME_DIR");
        if (!index || !*index || !game_dir) return;
        Common::FS::IOFile file{std::filesystem::path{index}, Common::FS::FileAccessMode::Read};
        if (!file.IsOpen()) return;
        std::vector<u8> bytes(file.GetSize());
        file.Read(bytes);
        if (bytes.size() < 12 || std::memcmp(bytes.data(), "ISSAOT01", 8) != 0) return;
        u32 count;
        std::memcpy(&count, bytes.data() + 8, 4);
        const std::filesystem::path root{game_dir};
        size_t at = 12;
        for (u32 i = 0; i < count && at + 38 <= bytes.size(); ++i) {
            u64 hash, offset;
            u32 regs[7];
            u16 len;
            std::memcpy(&hash, bytes.data() + at, 8);
            std::memcpy(regs, bytes.data() + at + 8, 28);
            std::memcpy(&len, bytes.data() + at + 36, 2);
            at += 38;
            if (at + len + 8 > bytes.size()) break;
            const std::string rel(reinterpret_cast<const char*>(bytes.data() + at), len);
            at += len;
            std::memcpy(&offset, bytes.data() + at, 8);
            at += 8;
            g_locs.emplace(hash,
                           ShaderLoc{(root / std::filesystem::path{std::u8string(rel.begin(), rel.end())})
                                         .string(),
                                     offset, regs[5]});
        }
    });
    return g_locs;
}

void PutSharp(std::vector<std::pair<u32, u32>>& writes, const auto& fetch, const void* sharp,
              u32 num_dwords) {
    const u32* words = static_cast<const u32*>(sharp);
    for (u32 i = 0; i < num_dwords && i < fetch.offsets.size(); ++i) {
        if ((fetch.load_mask >> i) & 1) writes.emplace_back(fetch.offsets[i], words[i]);
    }
}

void SetDstSel(auto& sharp, const AmdGpu::CompMapping& m) {
    sharp.dst_sel_x = u32(m.r);
    sharp.dst_sel_y = u32(m.g);
    sharp.dst_sel_z = u32(m.b);
    sharp.dst_sel_w = u32(m.a);
}
} // Anonymous namespace

/// Vertex attribute formats a float-class attribute can have (the specialization keeps only the
/// number class, the conversion changes the code): {number format, data format}.
/// Every float-class number format with the data formats vertex buffers use (variant index =
/// nfmt slot * 7 + dfmt slot); index 0 = Float, 32x4.
static constexpr std::array<AmdGpu::NumberFormat, 9> AttribNfmts{
    AmdGpu::NumberFormat::Float,   AmdGpu::NumberFormat::Unorm,   AmdGpu::NumberFormat::Snorm,
    AmdGpu::NumberFormat::Uscaled, AmdGpu::NumberFormat::Sscaled, AmdGpu::NumberFormat::SnormNz,
    AmdGpu::NumberFormat::Srgb,    AmdGpu::NumberFormat::Ubnorm,  AmdGpu::NumberFormat::UbnormNz};
static constexpr std::array<AmdGpu::DataFormat, 7> AttribDfmts{
    AmdGpu::DataFormat::Format32_32_32_32, AmdGpu::DataFormat::Format32_32_32,
    AmdGpu::DataFormat::Format32_32,       AmdGpu::DataFormat::Format32,
    AmdGpu::DataFormat::Format16_16_16_16, AmdGpu::DataFormat::Format16_16,
    AmdGpu::DataFormat::Format8_8_8_8};
static constexpr size_t NumAttribVariants = AttribNfmts.size() * AttribDfmts.size();
static std::pair<AmdGpu::NumberFormat, AmdGpu::DataFormat> AttribVariant(u8 v) {
    v %= NumAttribVariants;
    return {AttribNfmts[v / AttribDfmts.size()], AttribDfmts[v % AttribDfmts.size()]};
}

std::optional<std::vector<u32>> RegenerateRecipeSpirv(const Shader::Profile& profile,
                                                      const Shader::Info& meta,
                                                      const Shader::StageSpecialization& spec,
                                                      std::span<const u8> attrib_variants) {
    using namespace AmdGpu;
    const auto& locs = ShaderLocations();
    const auto loc = locs.find(meta.pgm_hash);
    if (loc == locs.end()) return std::nullopt;
    std::vector<u32> code(loc->second.dwords);
    {
        std::FILE* pack = std::fopen(loc->second.path.c_str(), "rb");
        if (!pack) return std::nullopt;
        const bool ok = _fseeki64(pack, s64(loc->second.offset), SEEK_SET) == 0 &&
                        std::fread(code.data(), 4, code.size(), pack) == code.size();
        std::fclose(pack);
        if (!ok) return std::nullopt;
    }
    std::vector<std::pair<u32, u32>> writes;
    const size_t num_buffers = meta.buffers.size();
    for (size_t i = 0; i < num_buffers && i < spec.buffers.size(); ++i) {
        const auto& desc = meta.buffers[i];
        if (desc.buffer_type != Shader::BufferType::Guest) continue;
        const auto& b = spec.buffers[i];
        Buffer v{};
        if (spec.bitset[i]) {
            v.base_address = 0x1000;
            v.num_records = 0x7FFFFFFF;
            v.stride = b.stride;
            v.swizzle_enable = b.swizzle_enable;
            v.dst_sel_x = 4; v.dst_sel_y = 5; v.dst_sel_z = 6; v.dst_sel_w = 7;
            if (b.is_formatted) {
                v.data_format = b.data_format;
                v.num_format = b.num_format;
                SetDstSel(v, b.dst_select);
            }
            if (b.swizzle_enable) {
                v.index_stride = b.index_stride;
                v.element_size = b.element_size;
            }
        }
        PutSharp(writes, desc.sharp_fetch, &v, 4);
    }
    for (size_t j = 0; j < meta.images.size() && j < spec.images.size(); ++j) {
        const auto& desc = meta.images[j];
        const auto& im = spec.images[j];
        Image t{};
        if (spec.bitset[num_buffers + j]) {
            t.base_address = 0x100;
            t.width = 1023;
            t.height = 1023;
            t.pitch = 1023;
            t.type = u64(im.type);
            auto nfmt = im.is_integer ? NumberFormat::Uint
                                      : (im.is_srgb ? NumberFormat::Srgb : NumberFormat::Unorm);
            auto dfmt = DataFormat::Format8_8_8_8;
            switch (im.num_conversion) {
            case NumberConversion::Uint32ToUnorm: nfmt = NumberFormat::Unorm; dfmt = DataFormat::Format32_32_32_32; break;
            case NumberConversion::Sint8ToSnormNz: nfmt = NumberFormat::SnormNz; break;
            case NumberConversion::Sint16ToSnormNz: nfmt = NumberFormat::SnormNz; dfmt = DataFormat::Format16_16_16_16; break;
            case NumberConversion::UintToUscaled: nfmt = NumberFormat::Uscaled; break;
            case NumberConversion::SintToSscaled: nfmt = NumberFormat::Sscaled; break;
            case NumberConversion::UnormToUbnorm: nfmt = NumberFormat::Ubnorm; break;
            case NumberConversion::SrgbToNorm: nfmt = NumberFormat::Srgb; dfmt = DataFormat::FormatBc6; break;
            default: break;
            }
            if (desc.is_depth) {
                nfmt = NumberFormat::Float;
                dfmt = DataFormat::Format32;
            }
            t.num_format = u64(nfmt);
            t.data_format = u64(dfmt);
            t.dst_sel_x = 4; t.dst_sel_y = 5; t.dst_sel_z = 6; t.dst_sel_w = 7;
            if (im.is_storage) SetDstSel(t, im.dst_select);
            t.last_level = im.num_bindings > 1 ? im.num_bindings - 1 : 0;
        }
        PutSharp(writes, desc.sharp_fetch, &t, desc.is_r128 ? 4 : 8);
    }
    for (size_t k = 0; k < meta.samplers.size() && k < spec.samplers.size(); ++k) {
        Sampler s{};
        s.raw0 = (u64(spec.samplers[k].force_unnormalized) << 15) |
                 (u64(spec.samplers[k].force_degamma) << 20);
        PutSharp(writes, meta.samplers[k].sharp_fetch, &s, 4);
    }
    // Vertex attribute V#s: tables the user data registers of the fetch shader point at.
    std::array<u32, Shader::NUM_USER_DATA_REGS> user_data{};
    std::map<u32, std::vector<u32>> tables;
    const auto& attributes = spec.fetch_shader_data.attributes;
    for (size_t a = 0; a < attributes.size() && a < spec.vs_attribs.size(); ++a) {
        const auto& attr = attributes[a];
        auto& table = tables[attr.sgpr_base];
        if (table.size() < attr.dword_offset + 4u) table.resize(attr.dword_offset + 4u);
        Buffer v{};
        v.base_address = 0x1000;
        v.num_records = 0x7FFFFFFF;
        v.stride = 16;
        v.data_format = u32(DataFormat::Format32_32_32_32);
        const auto cls = spec.vs_attribs[a].num_class;
        v.num_format = u32(cls == NumberClass::Sint ? NumberFormat::Sint
                         : cls == NumberClass::Uint ? NumberFormat::Uint : NumberFormat::Float);
        if (a < attrib_variants.size() && attrib_variants[a] != 0) {
            // 0: the number class's own format; v: AttribVariant(v - 1).
            const auto [nfmt, dfmt] = AttribVariant(attrib_variants[a] - 1);
            v.num_format = u32(nfmt);
            v.data_format = u32(dfmt);
        }
        SetDstSel(v, spec.vs_attribs[a].dst_select);
        std::memcpy(table.data() + attr.dword_offset, &v, 16);
    }
    for (auto& [sgpr, table] : tables) {
        if (sgpr + 1 >= user_data.size()) return std::nullopt;
        const u64 address = reinterpret_cast<u64>(table.data());
        user_data[sgpr] = u32(address);
        user_data[sgpr + 1] = u32(address >> 32);
    }

    const Shader::ShaderParams params{.user_data = user_data, .code = code, .hash = meta.pgm_hash};
    Shader::Info info(meta.hw_stage, meta.sw_stage, params);
    auto runtime_info = spec.runtime_info;
    const bool prev_aot = Shader::Info::aot_translation;
    const bool prev_throws = iss::assert_throws;
    const auto prev_recipe = std::exchange(iss::recipe_hash, meta.pgm_hash);
    Shader::Info::aot_translation = true;
    Shader::Info::aot_flat_writes = &writes;
    Shader::Info::aot_fetch = spec.fetch_shader_data.Empty() ? nullptr : &spec.fetch_shader_data;
    iss::assert_throws = true;
    SCOPE_EXIT {
        Shader::Info::aot_translation = prev_aot;
        Shader::Info::aot_flat_writes = nullptr;
        Shader::Info::aot_fetch = nullptr;
        iss::assert_throws = prev_throws;
        iss::recipe_hash = prev_recipe;
    };
    try {
        Shader::Pools pools;
        const auto ir = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
        auto binding = spec.start;
        return Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/// The attribute formats of a vertex stage whose recipe lacks them, found from the stage's SPIR-V
/// (recipe building, ISS_AOT_RECIPE_VERIFY): per float-class attribute the variant whose code
/// gets closest (opcode histogram), until the SPIR-V is identical.
std::optional<std::vector<u8>> RecoverAttribVariants(const Shader::Profile& profile,
                                                     const Shader::Info& meta,
                                                     const Shader::StageSpecialization& spec,
                                                     const std::vector<u32>& reference) {
    const auto histogram = [](const std::vector<u32>& spv) {
        std::unordered_map<u32, s32> h;
        for (size_t i = 5; i < spv.size();) {
            const u32 count = spv[i] >> 16;
            ++h[spv[i] & 0xFFFF];
            if (count == 0) break;
            i += count;
        }
        return h;
    };
    const auto ref = histogram(reference);
    const auto distance = [&](const std::vector<u32>& spv) {
        auto h = histogram(spv);
        u64 d = 0;
        for (const auto& [op, n] : ref) d += std::abs(n - h[op]);
        for (const auto& [op, n] : h) {
            if (!ref.contains(op)) d += std::abs(n);
        }
        return d;
    };
    static std::atomic<u32> attempts{0};
    static const u32 limit = [] {
        const char* value = std::getenv("ISS_AOT_RECIPE_RECOVER_LIMIT");
        return value ? u32(std::atoi(value)) : ~0u;
    }();
    if (attempts.fetch_add(1) >= limit) return std::nullopt;
    std::vector<u8> choice(spec.vs_attribs.size(), 0);
    auto made = RegenerateRecipeSpirv(profile, meta, spec, choice);
    {
        std::string classes;
        for (const auto& va : spec.vs_attribs) classes += fmt::format("{}", u32(va.num_class));
        if (std::getenv("ISS_AOT_TRACE")) std::fprintf(stderr, "AOT_RECOVER begin %llx attribs=%zu fetch_attribs=%zu classes=%s made=%d dist=%llu\n",
                     (unsigned long long)meta.pgm_hash, spec.vs_attribs.size(),
                     spec.fetch_shader_data.attributes.size(), classes.c_str(), made ? 1 : 0,
                     made ? (unsigned long long)distance(*made) : 0ull);
    }
    if (!made) return std::nullopt;
    if (*made == reference) return choice;
    u64 best = distance(*made);
    {
        // The usual case: Sint class attributes are SnormNz 16 bit (positions).
        auto trial = choice;
        constexpr u8 SnormNz16 = 5 * AttribDfmts.size() + 4 + 1;
        for (size_t a = 0; a < trial.size(); ++a) {
            if (spec.vs_attribs[a].num_class == AmdGpu::NumberClass::Sint) trial[a] = SnormNz16;
        }
        if (trial != choice) {
            if (auto spv = RegenerateRecipeSpirv(profile, meta, spec, trial)) {
                if (*spv == reference) return trial;
                if (const u64 d = distance(*spv); d < best) {
                    best = d;
                    choice = trial;
                }
            }
        }
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t a = 0; a < choice.size(); ++a) {
            for (u8 v = 0; v <= NumAttribVariants; ++v) {
                if (v == choice[a]) continue;
                // SnormNz exists for 8 and 16 bit data only (MapNumberConversion).
                if (const auto [nf, df] = AttribVariant(v == 0 ? 0 : v - 1);
                    v != 0 &&
                    nf == AmdGpu::NumberFormat::SnormNz && df != AmdGpu::DataFormat::Format8_8_8_8 &&
                    df != AmdGpu::DataFormat::Format16_16 &&
                    df != AmdGpu::DataFormat::Format16_16_16_16) {
                    continue;
                }
                auto trial = choice;
                trial[a] = v;
                auto spv = RegenerateRecipeSpirv(profile, meta, spec, trial);
                if (!spv) continue;
                if (*spv == reference) {
                    std::string text;
                    for (size_t k = 0; k < trial.size(); ++k) {
                        const auto [nf, df] = AttribVariant(trial[k] ? trial[k] - 1 : 0);
                        text += fmt::format(" a{}=n{}d{}c{}", k, u32(nf), u32(df),
                                            u32(spec.vs_attribs[k].num_class));
                    }
                    if (std::getenv("ISS_AOT_TRACE")) std::fprintf(stderr, "AOT_RECIPE_ATTRIBS %llx%s\n",
                                 (unsigned long long)meta.pgm_hash, text.c_str());
                    return trial;
                }
                if (const u64 d = distance(*spv); d < best) {
                    best = d;
                    choice = trial;
                }
            }
        }
    }
    static std::atomic<u32> dumped{0};
    if (const char* dir = std::getenv("ISS_AOT_RECIPE_DUMP"); dir && dumped.fetch_add(1) < 6) {
        std::string tag;
        for (const u8 v : choice) tag += fmt::format("-{}", v);
        if (auto spv = RegenerateRecipeSpirv(profile, meta, spec, choice)) {
            Common::FS::IOFile{(std::filesystem::path{dir} /
                                fmt::format("best_{:#018x}{}.spv", meta.pgm_hash, tag)),
                               Common::FS::FileAccessMode::Create}
                .WriteSpan(std::span<const u32>{*spv});
            Common::FS::IOFile{(std::filesystem::path{dir} /
                                fmt::format("best_{:#018x}_lib.spv", meta.pgm_hash)),
                               Common::FS::FileAccessMode::Create}
                .WriteSpan(std::span<const u32>{reference});
        }
    }
    return std::nullopt;
}

/// Recipe side table: attribute variants per vertex stage permutation (ISS_AOT_RECIPE_ATTRS).
namespace {
std::mutex g_attrib_mutex;
std::map<u64, std::vector<u8>> g_attrib_variants;
bool g_attrib_loaded{};
void LoadAttribVariants() {
    if (g_attrib_loaded) return;
    g_attrib_loaded = true;
    const char* path = std::getenv("ISS_AOT_RECIPE_ATTRS");
    std::filesystem::path file_path;
    if (path && *path) {
        file_path = path;
    } else if (const char* library = std::getenv("ISS_PIPELINE_LIBRARY"); library && *library) {
        file_path = std::filesystem::path{library} / "attribs.bin"; // A recipe pack's (make_recipes.py).
    } else {
        return;
    }
    Common::FS::IOFile file{file_path, Common::FS::FileAccessMode::Read};
    if (!file.IsOpen()) return;
    std::vector<u8> bytes(file.GetSize());
    file.Read(bytes);
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "ISSATR01", 8) != 0) return;
    u32 count;
    std::memcpy(&count, bytes.data() + 8, 4);
    size_t at = 12;
    for (u32 i = 0; i < count && at + 9 <= bytes.size(); ++i) {
        u64 hash;
        std::memcpy(&hash, bytes.data() + at, 8);
        const u8 n = bytes[at + 8];
        at += 9;
        if (at + n > bytes.size()) break;
        g_attrib_variants[hash].assign(bytes.data() + at, bytes.data() + at + n);
        at += n;
    }
}
} // Anonymous namespace

std::vector<u8> RecipeAttribVariants(u64 perm_hash) {
    std::scoped_lock lock{g_attrib_mutex};
    LoadAttribVariants();
    const auto it = g_attrib_variants.find(perm_hash);
    return it != g_attrib_variants.end() ? it->second : std::vector<u8>{};
}

void RememberAttribVariants(u64 perm_hash, std::vector<u8> variants) {
    std::scoped_lock lock{g_attrib_mutex};
    LoadAttribVariants();
    g_attrib_variants[perm_hash] = std::move(variants);
}

void SaveAttribVariants() {
    std::scoped_lock lock{g_attrib_mutex};
    const char* path = std::getenv("ISS_AOT_RECIPE_ATTRS");
    if (!path || !*path || g_attrib_variants.empty()) return;
    std::vector<u8> out(12);
    std::memcpy(out.data(), "ISSATR01", 8);
    const u32 count = u32(g_attrib_variants.size());
    std::memcpy(out.data() + 8, &count, 4);
    for (const auto& [hash, v] : g_attrib_variants) {
        const size_t at = out.size();
        out.resize(at + 9 + v.size());
        std::memcpy(out.data() + at, &hash, 8);
        out[at + 8] = u8(v.size());
        std::memcpy(out.data() + at + 9, v.data(), v.size());
    }
    Common::FS::IOFile file{std::filesystem::path{path}, Common::FS::FileAccessMode::Create};
    file.WriteSpan(std::span<const u8>{out});
}

void PipelineCache::DrainAot() {
    AotCompute* const s = aot.get();
    if (s == nullptr) return;
    if (s->num_ready.load(std::memory_order_acquire) == 0) {
        if (!s->reported && s->running.load(std::memory_order_acquire) == 0) {
            s->reported = true;
            std::fprintf(stderr,
                         "AOT_COMPUTE done in %.1f s: compiled=%llu adopted=%llu dropped=%llu "
                         "failed=%llu stateful=%llu cached_or_reached=%llu translate=%.1f ms "
                         "pipeline=%.1f ms avg\n",
                         double(Common::Perf::NowNs() - s->start_ns) / 1e9,
                         (unsigned long long)s->compiled.load(), (unsigned long long)s->adopted.load(),
                         (unsigned long long)s->dropped.load(), (unsigned long long)s->failed.load(),
                         (unsigned long long)s->stateful.load(), (unsigned long long)s->known.load(),
                         s->translate_ns.load() / 1e6 / std::max<u64>(1, s->compiled.load()),
                         s->pipeline_ns.load() / 1e6 / std::max<u64>(1, s->compiled.load()));
            std::fprintf(stderr,
                         "AOT_COMPUTE stateful: images=%llu samplers=%llu fmasks=%llu "
                         "formatted_buffers=%llu unknown_sharps=%llu; from opened packs first=%llu\n",
                         (unsigned long long)s->why[0].load(), (unsigned long long)s->why[1].load(),
                         (unsigned long long)s->why[2].load(), (unsigned long long)s->why[3].load(),
                         (unsigned long long)s->why[4].load(),
                         (unsigned long long)s->urgent_taken.load());
        }
        return;
    }
    // A few per call: each one is a few map insertions and cache writes.
    std::vector<AotCompute::Result> batch;
    {
        std::scoped_lock lock{s->mutex};
        const size_t n = std::min<size_t>(s->ready.size(), 16);
        batch.reserve(n);
        for (size_t i = s->ready.size() - n; i < s->ready.size(); ++i) {
            batch.push_back(std::move(s->ready[i]));
        }
        s->ready.resize(s->ready.size() - n);
        s->num_ready.fetch_sub(u32(n), std::memory_order_relaxed);
    }
    for (auto& r : batch) {
        const u64 hash = r.program->info.pgm_hash;
        if (program_cache.contains(hash) || compute_pipelines.contains(r.key)) {
            r.pipeline.reset();
            instance.GetDevice().destroyShaderModule(r.module);
            s->dropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        RegisterShaderBinary(std::move(r.spv), hash, 0);
        RegisterShaderMeta(r.program->info, r.spec.fetch_shader_data, r.spec, r.key.value, 0);
        ComputePipeline::SerializationSupport sdata{};
        RegisterPipelineData(r.key, sdata);
        {
            // The scout looks programs up under this lock.
            std::unique_lock lock{program_cache_mutex};
            r.program->AddPermut(r.module, std::move(r.spec));
            program_cache.emplace(hash, std::move(r.program));
        }
        compute_pipelines.emplace(r.key, std::move(r.pipeline));
        const u64 n = s->adopted.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & 1023) == 0) {
            std::fprintf(stderr,
                         "AOT_COMPUTE adopted=%llu compiled=%llu failed=%llu %.1f s; game compiled "
                         "first: waiting=%llu compiling=%llu stateful=%llu not_indexed=%llu\n",
                         (unsigned long long)n, (unsigned long long)s->compiled.load(),
                         (unsigned long long)s->failed.load(),
                         double(Common::Perf::NowNs() - s->start_ns) / 1e9,
                         (unsigned long long)s->misses[0].load(), (unsigned long long)s->misses[1].load(),
                         (unsigned long long)s->misses[2].load(), (unsigned long long)s->misses[6].load());
        }
    }
}

void PipelineCache::SaveDriverCache() {
    if (aot) aot->Stop(); // Not while its threads create pipelines in the cache.
    if (driver_cache_saved || !EmulatorSettings.IsPipelineCacheEnabled() || !pipeline_cache) return;
    if (compiler) compiler->WaitIdle();
    const auto result = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if ((result.result != vk::Result::eSuccess && result.result != vk::Result::eIncomplete) ||
        result.value.size() < 32 || result.value.size() > 256ULL * 1024 * 1024 - 24) {
        LOG_WARNING(Render_Vulkan, "Vulkan driver cache snapshot unavailable");
        return;
    }
    const auto envelope = PackDriverCache(result.value);
    Common::FS::IOFile file{driver_cache_path, Common::FS::FileAccessMode::Create};
    if (file.IsOpen() && file.Write(envelope) == envelope.size()) {
        file.Flush();
        driver_cache_saved = true;
        LOG_INFO(Render_Vulkan, "Vulkan driver cache saved: {} bytes", result.value.size());
    } else {
        LOG_WARNING(Render_Vulkan, "Vulkan driver cache could not be saved");
    }
}

void PipelineCache::CompileAsync(Pipeline* pipeline) {
    compiler->Submit(pipeline);
    // A pipeline found in the driver cache is ready within a few milliseconds, which is worth
    // waiting for instead of skipping the draw.
    const u32 wait_us =
        LazyPipelines() ? 0
                        : Common::Perf::GetTuning().async_pipeline_wait_us.load(std::memory_order_relaxed);
    if (wait_us != 0) {
        compiler->WaitFor(pipeline, std::chrono::microseconds{wait_us});
    }
}

void PipelineCache::QueuePreload(Pipeline* pipeline) { compiler->Submit(pipeline); }
void PipelineCache::PrioritizeCompile(const Pipeline* pipeline) { compiler->Prioritize(pipeline); }
bool PipelineCache::CompileNowIfQueued(const Pipeline* pipeline) {
    return compiler->RunIfQueued(pipeline);
}
void PipelineCache::WaitPreload() { compiler->WaitIdle(); }

iss::pipeline_reuse::Stamp PipelineCache::GraphicsReuseStamp() const {
    return {liverpool->ContextRegChanges(), VideoCore::CurrentUploadEpoch(),
            VideoCore::CurrentHostWriteGen(), scheduler.CurrentTick(),
            static_cast<u32>(liverpool->regs.stage_enable.raw),
            draw_indirect_params.vertex_sgpr_offset, draw_indirect_params.instance_sgpr_offset};
}

struct PipelineCache::PreparedGraphicsState {
    struct Stage {
        iss::pipeline_reuse::StageIdentity identity{};
        Shader::Backend::Bindings start{};
        std::array<std::byte, offsetof(AmdGpu::ShaderProgram, user_data)> registers{};
    };
    const PipelineCache* owner{};
    const GraphicsPipeline* pipeline{};
    iss::pipeline_reuse::Stamp stamp{};
    GraphicsPipelineKey key{};
    std::array<Stage, 2> stages{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime{};
};

std::shared_ptr<PipelineCache::PreparedGraphicsState>
PipelineCache::CapturePreparedGraphics(const GraphicsPipeline* pipeline) {
    const auto* draw = liverpool->CurrentScoutDraw();
    const auto& regs = liverpool->regs;
    if (!pipeline || !pipeline->IsReady() || !instance.IsVertexInputDynamicState() ||
        regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs ||
        draw_indirect_params.vertex_sgpr_offset || draw_indirect_params.instance_sgpr_offset ||
        !draw || draw->num_stages != 2 || draw->vertex_sgpr_offset || draw->instance_sgpr_offset ||
        !Common::Perf::GetTuning().scout_fast.load(std::memory_order_relaxed) ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed) ||
        EmulatorSettings.IsPatchShaders() || graphics_key != pipeline->GetGraphicsKey()) return {};
    const auto pipeline_stages = pipeline->GetStages();
    for (size_t slot = 0; slot < pipeline_stages.size(); ++slot) {
        if (pipeline_stages[slot] != infos[slot] ||
            (infos[slot] && slot != u32(SwStage::Fragment) && slot != u32(SwStage::Vertex))) return {};
    }
    auto saved = std::make_shared<PreparedGraphicsState>();
    saved->owner = this;
    saved->pipeline = pipeline;
    saved->stamp = GraphicsReuseStamp();
    saved->key = graphics_key;
    saved->infos = infos;
    saved->modules = modules;
    saved->runtime = runtime_infos;
    constexpr std::array stages{SwStage::Fragment, SwStage::Vertex};
    const std::array programs{&regs.ps_program, &regs.vs_program};
    Shader::Backend::Bindings binding{};
    Shader::Gcn::FetchShaderData* expected_fetch = nullptr;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        auto* program = static_cast<Program*>(const_cast<void*>(scouted.program));
        if (!scouted.valid || !scouted.resolved || !program || scouted.sw_stage != slot ||
            scouted.hw_stage != i || scouted.start != binding ||
            taken_scout_infos[slot] != infos[slot] || taken_scout_stages[slot] != &scouted ||
            infos[slot] != &program->info || scouted.perm_idx >= program->modules.size()) return {};
        const auto& perm = program->modules[scouted.perm_idx];
        if (!modules[slot] || modules[slot] != perm.module ||
            graphics_key.stage_hashes[slot] != HashCombine(scouted.params_hash, u64{scouted.perm_idx})) return {};
        if (!perm.spec.fetch_shader_data.Empty())
            expected_fetch = &program->modules[scouted.perm_idx].spec.fetch_shader_data;
        auto& stage = saved->stages[i];
        stage.identity = {program, std::bit_cast<u64>(static_cast<VkShaderModule>(perm.module)),
                          scouted.params_hash, scouted.perm_idx, scouted.code_dwords};
        stage.start = binding;
        std::memcpy(stage.registers.data(), programs[i], stage.registers.size());
        program->info.AddBindings(binding);
    }
    if (fetch_shader != expected_fetch) return {};
    // Reuse the same final live-input checks; publishing refreshes only the current draw's data.
    return TryPreparedGraphics(*saved) == pipeline ? saved : nullptr;
}

const GraphicsPipeline* PipelineCache::TryPreparedGraphics(const PreparedGraphicsState& saved) {
    const auto* draw = liverpool->CurrentScoutDraw();
    const auto& regs = liverpool->regs;
    if (saved.owner != this || !saved.pipeline || !saved.pipeline->IsReady() ||
        !instance.IsVertexInputDynamicState() ||
        regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs ||
        draw_indirect_params.vertex_sgpr_offset || draw_indirect_params.instance_sgpr_offset ||
        !draw || draw->num_stages != 2 || draw->vertex_sgpr_offset || draw->instance_sgpr_offset ||
        GraphicsReuseStamp() != saved.stamp ||
        !Common::Perf::GetTuning().scout_fast.load(std::memory_order_relaxed) ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed) ||
        EmulatorSettings.IsPatchShaders()) return nullptr;
    constexpr std::array stages{SwStage::Fragment, SwStage::Vertex};
    const std::array programs{&regs.ps_program, &regs.vs_program};
    Shader::Backend::Bindings binding{};
    Shader::Gcn::FetchShaderData* live_fetch = nullptr;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        const auto& stage = saved.stages[i];
        auto* program = static_cast<Program*>(const_cast<void*>(stage.identity.program));
        if (!program || !scouted.valid || !scouted.resolved || scouted.hw_stage != i ||
            scouted.sw_stage != slot || scouted.program != program ||
            scouted.start != binding || scouted.start != stage.start ||
            scouted.pgm_base != programs[i]->Address<VAddr>() ||
            scouted.perm_idx >= program->modules.size() || !scouted.flat ||
            scouted.flat_size != program->info.srt_info.flattened_bufsize_dw ||
            saved.infos[slot] != &program->info) return nullptr;
        auto& perm = program->modules[scouted.perm_idx];
        const iss::pipeline_reuse::StageIdentity identity{
            program, std::bit_cast<u64>(static_cast<VkShaderModule>(perm.module)),
            scouted.params_hash, scouted.perm_idx, scouted.code_dwords};
        if (saved.modules[slot] != perm.module ||
            !iss::pipeline_reuse::StageMatches(stage.identity, identity, stage.registers,
                std::as_bytes(std::span{programs[i], 1}).first(stage.registers.size()),
                std::as_bytes(std::span{&saved.runtime[slot], 1}), scouted.runtime_info,
                std::as_bytes(std::span{programs[i]->user_data}),
                std::as_bytes(std::span{scouted.user_data}))) return nullptr;
        // Module vectors may grow between uses. Reacquire instead of saving a dangling pointer.
        if (!perm.spec.fetch_shader_data.Empty()) live_fetch = &perm.spec.fetch_shader_data;
        program->info.AddBindings(binding);
    }
    if (GraphicsReuseStamp() != saved.stamp) return nullptr;
    // Every check completed before modifying lookup scratch or any shared stage Info.
    graphics_key = saved.key;
    infos = saved.infos;
    modules = saved.modules;
    runtime_infos = saved.runtime;
    fetch_shader = live_fetch;
    taken_scout_infos.fill(nullptr);
    taken_scout_stages.fill(nullptr);
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        auto& info = static_cast<Program*>(const_cast<void*>(scouted.program))->info;
        info.pgm_base = scouted.pgm_base;
        info.user_data = programs[i]->user_data;
        info.flattened_ud_buf.assign(scouted.flat, scouted.flat + scouted.flat_size);
        taken_scout_infos[slot] = &info;
        taken_scout_stages[slot] = &scouted;
    }
    graphics_reuse.valid = false;
    return saved.pipeline;
}

void PipelineCache::RememberGraphicsState(const GraphicsPipeline* pipeline) {
    graphics_reuse.valid = false;
    const auto* draw = liverpool->CurrentScoutDraw();
    const auto& regs = liverpool->regs;
    if (!pipeline || !pipeline->IsReady() || !instance.IsVertexInputDynamicState() ||
        regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs ||
        !draw || draw->num_stages != 2 ||
        !Common::Perf::GetTuning().scout_fast.load(std::memory_order_relaxed) ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed) ||
        EmulatorSettings.IsPatchShaders()) return;
    constexpr std::array stages{SwStage::Fragment, SwStage::Vertex};
    const std::array programs{&regs.ps_program, &regs.vs_program};
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        if (!scouted.valid || !scouted.resolved ||
            scouted.sw_stage != slot || scouted.hw_stage != i ||
            taken_scout_infos[slot] != infos[slot] || taken_scout_stages[slot] != &scouted ||
            !infos[slot] || !modules[slot]) return;
        auto& saved = graphics_reuse.stages[i];
        saved.identity = {scouted.program, std::bit_cast<u64>(static_cast<VkShaderModule>(modules[slot])),
                          scouted.params_hash, scouted.perm_idx, scouted.code_dwords};
        std::memcpy(saved.registers.data(), programs[i], saved.registers.size());
    }
    graphics_reuse.pipeline = pipeline;
    graphics_reuse.stamp = GraphicsReuseStamp();
    graphics_reuse.valid = true;
}

const GraphicsPipeline* PipelineCache::TryReuseGraphicsState() {
    const auto* draw = liverpool->CurrentScoutDraw();
    if (!graphics_reuse.valid || !draw || draw->num_stages != 2 ||
        GraphicsReuseStamp() != graphics_reuse.stamp ||
        !Common::Perf::GetTuning().scout_fast.load(std::memory_order_relaxed) ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed) ||
        EmulatorSettings.IsPatchShaders()) return nullptr;
    const auto& regs = liverpool->regs;
    constexpr std::array stages{SwStage::Fragment, SwStage::Vertex};
    const std::array programs{&regs.ps_program, &regs.vs_program};
    Shader::Backend::Bindings binding{};
    // Validate both stages before changing any Info or scout pointer. User data and
    // resources can differ from the seed draw; their current scout words are copied below.
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        const auto& saved = graphics_reuse.stages[i];
        auto* program = static_cast<Program*>(const_cast<void*>(saved.identity.program));
        if (!scouted.valid || !scouted.resolved || scouted.hw_stage != i ||
            scouted.sw_stage != slot || scouted.program != program ||
            scouted.start != binding || scouted.pgm_base != programs[i]->Address<VAddr>() ||
            scouted.perm_idx >= program->modules.size() || !scouted.flat ||
            scouted.flat_size != program->info.srt_info.flattened_bufsize_dw ||
            infos[slot] != &program->info) return nullptr;
        const auto module = program->modules[scouted.perm_idx].module;
        const iss::pipeline_reuse::StageIdentity identity{
            program, std::bit_cast<u64>(static_cast<VkShaderModule>(module)),
            scouted.params_hash, scouted.perm_idx, scouted.code_dwords};
        if (!iss::pipeline_reuse::StageMatches(
                saved.identity, identity, saved.registers,
                std::as_bytes(std::span{programs[i], 1}).first(saved.registers.size()),
                std::as_bytes(std::span{&runtime_infos[slot], 1}), scouted.runtime_info,
                std::as_bytes(std::span{programs[i]->user_data}),
                std::as_bytes(std::span{scouted.user_data}))) return nullptr;
        program->info.AddBindings(binding);
    }
    // All guards are host reads. A concurrent guest-memory write notification is a miss.
    if (GraphicsReuseStamp() != graphics_reuse.stamp) return nullptr;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto slot = static_cast<u32>(stages[i]);
        const auto& scouted = draw->stages[i];
        auto& info = static_cast<Program*>(const_cast<void*>(scouted.program))->info;
        info.pgm_base = scouted.pgm_base;
        info.user_data = programs[i]->user_data;
        info.flattened_ud_buf.assign(scouted.flat, scouted.flat + scouted.flat_size);
        taken_scout_infos[slot] = &info;
        taken_scout_stages[slot] = &scouted;
    }
    return graphics_reuse.pipeline;
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    iss::diagnostics::Scope phase_scope(iss::diagnostics::GraphicsPipeline);
    draw_indirect_params = params;
    static const bool state_reuse = [] {
        const char* value = std::getenv("ISS_PIPELINE_STATE_REUSE");
        return value && value[0] == '1';
    }();
    static const bool verify_reuse = [] {
        const char* value = std::getenv("ISS_PIPELINE_STATE_REUSE_VERIFY");
        return value && value[0] == '1';
    }();
    static const bool reuse_stats = [] {
        const char* value = std::getenv("ISS_PIPELINE_STATE_REUSE_STATS");
        return value && value[0] == '1';
    }();
    if (state_reuse) {
        ++graphics_reuse_checks;
        if (reuse_stats && (graphics_reuse_checks & ((1u << 18) - 1)) == 0) {
            std::fprintf(stderr, "PIPELINE_STATE_REUSE checks=%llu hits=%llu verified=%llu\n",
                         static_cast<unsigned long long>(graphics_reuse_checks),
                         static_cast<unsigned long long>(graphics_reuse_hits),
                         static_cast<unsigned long long>(graphics_reuse_verified));
        }
        if (const auto* reused = TryReuseGraphicsState()) {
            if (verify_reuse) {
                const auto expected_key = graphics_key;
                const auto expected_infos = infos;
                const auto expected_modules = modules;
                const auto* expected_fetch = fetch_shader;
                ASSERT_MSG(RefreshGraphicsKey() && graphics_key == expected_key &&
                               infos == expected_infos && modules == expected_modules &&
                               fetch_shader == expected_fetch,
                           "PIPELINE_STATE_REUSE disagrees with the full graphics lookup");
                ++graphics_reuse_verified;
            }
            ++graphics_reuse_hits;
            return reused;
        }
        graphics_reuse.valid = false;
    }
    const auto remember = [&](const GraphicsPipeline* pipeline) {
        if (state_reuse) RememberGraphicsState(pipeline);
        return pipeline;
    };
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    // Pipeline key memo: draws in a row mostly use the same pipeline, found again without
    // hashing the key and probing the map.
    const bool key_memo =
        Common::Perf::GetTuning().pipeline_key_memo.load(std::memory_order_relaxed);
    if (key_memo && last_graphics_pipeline != nullptr && last_graphics_key == graphics_key) {
        Common::Perf::Add(Common::Perf::PipelineKeyMemoHits);
        return remember(last_graphics_pipeline);
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (!is_new) {
        last_graphics_key = graphics_key;
        last_graphics_pipeline = it->second.get();
    }
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
        iss::blackbox::Event(iss::blackbox::Compile, "compila pipeline grafico %llx",
                             (unsigned long long)pipeline_hash);

        const u64 start_ns = Common::Perf::NowNs();
        static const bool lazy = [] {
            const char* value = std::getenv("ISS_LAZY_PIPELINES");
            return value && value[0] == '1';
        }();
        const bool defer =
            lazy || Common::Perf::GetTuning().async_pipelines.load(std::memory_order_relaxed) >= 1;
        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false, defer);
        GraphicsPipeline* const pipeline = it.value().get();
        last_graphics_key = graphics_key;
        last_graphics_pipeline = pipeline;

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            std::scoped_lock lock{module_related_pipelines_mutex};
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        if (defer) {
            CompileAsync(pipeline);
        }
        if (pipeline->NeedsBackgroundOptimization()) {
            // Drawn with its fast variant at once; the optimized one replaces it later.
            compiler->SubmitLow(pipeline);
        }
        Common::Perf::Add(Common::Perf::PipelineStallNs, Common::Perf::NowNs() - start_ns);
        iss::pipeline_library::LogMiss("graphics", pipeline_hash, 0, Common::Perf::NowNs() - start_ns);
        return remember(pipeline);
    }
    return remember(it->second.get());
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    return GetComputePipeline(iss::g4::CpOwner::Graphics);
}

const ComputePipeline* PipelineCache::GetComputePipelineFromSnapshot(
    const iss::g4::FallbackPacket& packet) {
    if (packet.Gate() != iss::g4::FallbackGate::Ready ||
        packet.compute_state.size() != sizeof(AmdGpu::ComputeProgram) ||
        packet.shader.code.empty() || packet.flattened_srt.empty() ||
        active_compute_snapshot || active_snapshot_cs || Shader::Info::srt_replay)
        return nullptr;
    AmdGpu::ComputeProgram cs{};
    std::memcpy(&cs, packet.compute_state.data(), sizeof(cs));
    if (reinterpret_cast<VAddr>(cs.Address<u32*>()) != packet.shader.guest_base ||
        cs.user_data != packet.user_data) return nullptr;
    const Shader::Info::SrtReplay replay{packet.user_data.data(), packet.flattened_srt};
    active_compute_snapshot = &packet;
    active_snapshot_cs = &cs;
    Shader::Info::srt_replay = &replay;
    const ComputePipeline* result{};
    try {
        result = GetComputePipeline(iss::g4::CpOwner::Graphics);
    } catch (const std::exception&) {
        result = nullptr;
    }
    Shader::Info::srt_replay = nullptr;
    active_snapshot_cs = nullptr;
    active_compute_snapshot = nullptr;
    return result;
}

const ComputePipeline* PipelineCache::GetComputePipeline(iss::g4::CpOwner owner) {
    iss::diagnostics::Scope phase_scope(iss::diagnostics::ComputePipeline);
    if (aot && owner == iss::g4::CpOwner::Graphics && !active_compute_snapshot) {
        DrainAot();
    }
    if (!RefreshComputeKey(owner)) {
        return nullptr;
    }
    auto& owner_pipelines = owner == iss::g4::CpOwner::Graphics ? compute_pipelines
                                                                : asc_compute_pipelines;
    auto& owner_key = owner == iss::g4::CpOwner::Graphics ? compute_key : asc_lookup.compute_key;
    auto& owner_infos = owner == iss::g4::CpOwner::Graphics ? infos : asc_lookup.infos;
    auto& owner_modules = owner == iss::g4::CpOwner::Graphics ? modules : asc_lookup.modules;
    const auto [it, is_new] = owner_pipelines.try_emplace(owner_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(owner_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);
        if (aot && owner == iss::g4::CpOwner::Graphics) aot->NoteMiss(owner_infos[0]->pgm_hash);
        iss::blackbox::Event(iss::blackbox::Compile, "compila pipeline compute %llx",
                             (unsigned long long)pipeline_hash);

        const u64 start_ns = Common::Perf::NowNs();
        const bool defer =
            !Common::Perf::GetTuning().tiered_compute_pipelines.load(std::memory_order_relaxed) &&
            (LazyPipelines() ||
             Common::Perf::GetTuning().async_pipelines.load(std::memory_order_relaxed) >= 2);
        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, owner_key, *owner_infos[0],
                                                       owner_modules[0], sdata, false, defer);
        ComputePipeline* const pipeline = it.value().get();
        // Persistent permutation indexes belong to the graphics/preload partition. ASC may
        // discover the same permutations in a different order, so it must not overwrite them.
        if (owner == iss::g4::CpOwner::Graphics) RegisterPipelineData(owner_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            std::scoped_lock lock{module_related_pipelines_mutex};
            auto& m = owner_modules[0];
            module_related_pipelines[m].emplace_back(owner_key);
        }
        if (defer) {
            CompileAsync(pipeline);
        }
        if (pipeline->NeedsBackgroundOptimization()) {
            // Its first handle is already complete and usable; no dispatch is skipped.
            compiler->Submit(pipeline);
        }
        Common::Perf::Add(Common::Perf::PipelineStallNs, Common::Perf::NowNs() - start_ns);
        iss::pipeline_library::LogMiss("compute", owner_key.value, owner_infos[0]->pgm_hash,
                                       Common::Perf::NowNs() - start_ns);
        return pipeline;
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    FillColorBuffers(regs, key.color_buffers);

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // D2 (ISS_DLSS_UPSCALE): the HUD layer's alpha is its transmittance (what is left of the
        // scene below; cleared to 1, the composite does hud.rgb + hud.a * scene). Every draw into
        // it keeps its colour blend and multiplies alpha by its colour dst factor (evaluated on
        // alpha: SRC_COLOR / SRC_ALPHA = As, ONE_MINUS_SRC_ALPHA = 1 - As, ONE = 1, ZERO = 0), so
        // over panels, additive glows and the darkening panels (ZERO / SRC_ALPHA, ZERO /
        // SRC_COLOR: test M, 2026-11-04) all keep their look. Blending off = opaque (alpha 0).
        // ISS_DLSS_HUD_LOG=1 (diagnostic): each distinct blend / mask of the draws into the HUD
        // layer, once (which draws lose their look in the layer).
        if (cb == 0 && iss::dlss::UpscaleWanted() && col_buf.Address() == iss::dlss::HudLayerAddr()) {
            static const bool hud_log = [] {
                const char* v = std::getenv("ISS_DLSS_HUD_LOG");
                return v && v[0] == '1';
            }();
            if (hud_log) {
                static std::vector<u64> seen;
                const auto& bc = regs.blend_control[0];
                const u64 combo = (u64(key.blend_controls[0].enable) << 40) |
                                  (u64(static_cast<u32>(bc.color_src_factor)) << 32) |
                                  (u64(static_cast<u32>(bc.color_dst_factor)) << 24) |
                                  (u64(static_cast<u32>(bc.color_func)) << 20) |
                                  (u64(bc.separate_alpha_blend) << 19) |
                                  (u64(static_cast<u32>(bc.alpha_src_factor)) << 12) |
                                  (u64(static_cast<u32>(bc.alpha_dst_factor)) << 4) |
                                  u64(static_cast<u32>(key.write_masks[0]));
                if (std::find(seen.begin(), seen.end(), combo) == seen.end() && seen.size() < 64) {
                    seen.push_back(combo);
                    std::fprintf(stderr,
                                 "DLSS HUD blend: enable %u src %u dst %u func %u sep %u asrc %u adst %u "
                                 "afunc %u mask %#x\n",
                                 u32(key.blend_controls[0].enable), u32(bc.color_src_factor),
                                 u32(bc.color_dst_factor), u32(bc.color_func),
                                 u32(bc.separate_alpha_blend), u32(bc.alpha_src_factor),
                                 u32(bc.alpha_dst_factor), u32(bc.alpha_func),
                                 u32(key.write_masks[0]));
                }
            }
        }
        if (cb == 0 && iss::dlss::UpscaleWanted() && col_buf.Address() == iss::dlss::HudLayerAddr() &&
            !(key.write_masks[0] & vk::ColorComponentFlagBits::eA)) {
            using Factor = AmdGpu::BlendControl::BlendFactor;
            auto& bc = key.blend_controls[0];
            if (!bc.enable) {
                bc = {};
                bc.enable = 1;
                bc.color_src_factor = Factor::One;
                bc.color_dst_factor = Factor::Zero;
                bc.color_func = AmdGpu::BlendControl::BlendFunc::Add;
            }
            const Factor src = bc.color_src_factor;
            const Factor dst = bc.color_dst_factor;
            // A multiply onto the empty layer (DST_COLOR / ZERO) cannot be shown: left alone.
            const bool keep = dst == Factor::Zero && (src == Factor::DstColor || src == Factor::OneMinusDstColor);
            iss::dlss::g_key_seen.fetch_add(1, std::memory_order_relaxed);
            if (!keep) {
                iss::dlss::g_key_fixed.fetch_add(1, std::memory_order_relaxed);
                bc.separate_alpha_blend = 1;
                bc.alpha_src_factor = Factor::Zero;
                bc.alpha_dst_factor = dst;
                bc.alpha_func = AmdGpu::BlendControl::BlendFunc::Add;
            }
        }

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

template <typename ProgramRegs>
Shader::ShaderParams PipelineCache::GetStageParams(const ProgramRegs& pgm,
                                                   iss::g4::CpOwner owner) {
    if constexpr (std::is_same_v<ProgramRegs, AmdGpu::ComputeProgram>) {
        if (active_compute_snapshot) {
            const auto& packet = *active_compute_snapshot;
            return {.user_data = packet.user_data,
                    .code = packet.shader.code,
                    .hash = packet.shader.hash};
        }
    }
    if (!Common::Perf::GetTuning().shader_params_memo.load(std::memory_order_relaxed)) {
        return AmdGpu::GetParams(pgm);
    }
    const auto* code = pgm.template Address<u32*>();
    const VAddr address = reinterpret_cast<VAddr>(code);
    // Read before the code: a submission or guest wait while reading makes the entry unusable.
    const u64 epoch = VideoCore::CurrentUploadEpoch();
    const u64 host_write_gen = VideoCore::CurrentHostWriteGen();
    auto& owner_memos = owner == iss::g4::CpOwner::Graphics ? params_memos
                                                           : asc_lookup.params_memos;
    auto& memo = owner_memos[(address * 0x9E3779B97F4A7C15ULL) >> (64 - NumParamsMemoBits)];
    if (memo.address == address && memo.epoch == epoch && memo.host_write_gen == host_write_gen) {
        Common::Perf::Add(Common::Perf::ParamsMemoHits);
        return {
            .user_data = pgm.user_data,
            .code = std::span<const u32>{code, memo.num_dwords},
            .hash = memo.hash,
        };
    }
    Common::Perf::Add(Common::Perf::ParamsMemoMisses);
    const auto params = AmdGpu::GetParams(pgm);
    memo = ParamsMemo{
        .address = address,
        .epoch = epoch,
        .host_write_gen = host_write_gen,
        .hash = params.hash,
        .num_dwords = static_cast<u32>(params.code.size()),
    };
    return params;
}

u64 PipelineCache::GraphicsProgramHash(const AmdGpu::ShaderProgram& pgm) {
    return GetStageParams(pgm, iss::g4::CpOwner::Graphics).hash;
}

template <typename ProgramRegs>
bool PipelineCache::TakeScoutedStage(Shader::HwStage stage_in, Shader::SwStage stage_out,
                                     const ProgramRegs& pgm, Shader::Backend::Bindings& binding) {
    if (!Common::Perf::GetTuning().scout_fast.load(std::memory_order_relaxed)) {
        return false;
    }
    const auto* draw = liverpool->CurrentScoutDraw();
    if (draw == nullptr) {
        return false;
    }
    const AmdGpu::ScoutStage* scouted = nullptr;
    for (u32 i = 0; i < draw->num_stages; ++i) {
        if (draw->stages[i].sw_stage == static_cast<u8>(stage_out)) {
            scouted = &draw->stages[i];
            break;
        }
    }
    if (scouted == nullptr || !scouted->valid || !scouted->resolved ||
        scouted->hw_stage != static_cast<u8>(stage_in)) {
        return false;
    }
    // The scout's inputs must be the command processor's: program, bindings so far, user data.
    const auto* code = pgm.template Address<u32*>();
    if (scouted->pgm_base != reinterpret_cast<VAddr>(code) || scouted->start != binding ||
        std::memcmp(scouted->user_data.data(), pgm.user_data.data(),
                    sizeof(scouted->user_data)) != 0) {
        Common::Perf::Add(Common::Perf::ScoutStale);
        return false;
    }
    auto* program = static_cast<Program*>(const_cast<void*>(scouted->program));
    auto& info = program->info;
    if (scouted->flat_size != info.srt_info.flattened_bufsize_dw ||
        scouted->perm_idx >= program->modules.size() ||
        Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) {
        return false;
    }
    // What GetProgram does for a permutation found in the memo, with the scout's lookups.
    const auto stage_out_idx = static_cast<u32>(stage_out);
    runtime_infos[stage_out_idx] = scouted->GetRuntimeInfo();
    info.pgm_base = reinterpret_cast<VAddr>(code);
    info.user_data = pgm.user_data;
    info.flattened_ud_buf.assign(scouted->flat, scouted->flat + scouted->flat_size);
    info.AddBindings(binding);
    auto& perm = program->modules[scouted->perm_idx];
    if (auto& fetch = perm.spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    infos[stage_out_idx] = &info;
    modules[stage_out_idx] = perm.module;
    taken_scout_infos[stage_out_idx] = &info;
    taken_scout_stages[stage_out_idx] = scouted;
    graphics_key.stage_hashes[stage_out_idx] =
        HashCombine(scouted->params_hash, u64{scouted->perm_idx});
    Common::Perf::Add(Common::Perf::SpecCalls);
    Common::Perf::Add(Common::Perf::ScoutFast);
    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = nullptr;
    async_translation_pending = false;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        if (TakeScoutedStage(stage_in, stage_out, *pgm, binding)) {
            return true;
        }
        const auto params = GetStageParams(*pgm, iss::g4::CpOwner::Graphics);
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        return !async_translation_pending;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    taken_scout_infos.fill(nullptr);

    // ISS_PARALLEL_TRANSLATE: a new material usually brings a new pixel and a new vertex
    // program; the vertex one is translated by a helper thread meanwhile (its runtime info only
    // depends on the registers), the command processor emits it in binding order after.
    std::unique_ptr<PreTranslation> pre;
    struct PreGuard {
        PipelineCache& cache;
        std::unique_ptr<PreTranslation>& pre;
        ~PreGuard() {
            cache.pre_translation = nullptr;
            if (pre && pre->done.valid()) {
                pre->done.wait();
                g_pre_unused.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } pre_guard{*this, pre};
    if (ParallelTranslateEnabled() && !active_compute_snapshot &&
        regs.stage_enable.raw == AmdGpu::ShaderStageEnable::VgtStages::Vs &&
        regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Fragment)) &&
        regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Vertex))) {
        const auto* ps = regs.ProgramForStage(static_cast<u32>(HwStage::Fragment));
        const auto* vs = regs.ProgramForStage(static_cast<u32>(HwStage::Vertex));
        if (ps && vs && ps->Address<u32*>() && vs->Address<u32*>()) {
            const auto ps_params = GetStageParams(*ps, iss::g4::CpOwner::Graphics);
            const auto vs_params = GetStageParams(*vs, iss::g4::CpOwner::Graphics);
            if (ps_params.hash != vs_params.hash && !program_cache.contains(ps_params.hash) &&
                !program_cache.contains(vs_params.hash)) {
                pre = std::make_unique<PreTranslation>();
                pre->hash = vs_params.hash;
                pre->program = std::make_unique<Program>(HwStage::Vertex, SwStage::Vertex,
                                                         vs_params);
                if (BuildRuntimeInfoFrom(pre->initial_runtime_info, regs, HwStage::Vertex,
                                         SwStage::Vertex, graphics_key.color_buffers,
                                         draw_indirect_params)) {
                    pre->runtime_info = pre->initial_runtime_info;
                    auto* p = pre.get();
                    const auto code = vs_params.code;
                    p->done = std::async(std::launch::async, [this, p, code] {
                        p->ir.emplace(Shader::TranslateProgram(code, p->pools, p->program->info,
                                                               p->runtime_info, profile));
                    });
                    pre_translation = p;
                    g_pre_started.fetch_add(1, std::memory_order_relaxed);
                    NotePreTranslation();
                } else {
                    pre.reset();
                }
            }
        }
    }

    bind_stage(HwStage::Fragment, SwStage::Fragment);
    if (async_translation_pending) {
        return false; // Its translation is running: the draw is skipped.
    }

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return !async_translation_pending;
}

bool PipelineCache::RefreshComputeKey(iss::g4::CpOwner owner) {
    if (owner == iss::g4::CpOwner::Graphics) graphics_reuse.valid = false;
    auto& owner_infos = owner == iss::g4::CpOwner::Graphics ? infos : asc_lookup.infos;
    auto& owner_modules = owner == iss::g4::CpOwner::Graphics ? modules : asc_lookup.modules;
    auto& owner_key = owner == iss::g4::CpOwner::Graphics ? compute_key : asc_lookup.compute_key;
    if (owner == iss::g4::CpOwner::Graphics) taken_scout_infos.fill(nullptr);
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = active_snapshot_cs ? *active_snapshot_cs : liverpool->GetCsRegs();
    const auto cs_params = GetStageParams(cs_pgm, owner);
    std::tie(owner_infos[0], owner_modules[0], owner_key.value) =
        GetProgram(owner, HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

// Where shader compilation spends its time (ISS_SHADER_SPLIT=1): translate, emit, module, register.
static std::array<std::atomic<u64>, 5> g_shader_split_ns{};
static std::atomic<u64> g_new_programs{}, g_new_permutations{}, g_scouted_new_perm{};
static void NoteShaderSplit(const std::array<u64, 5>& t) {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SHADER_SPLIT");
        return value && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    for (u32 i = 0; i < 4; ++i) {
        g_shader_split_ns[i].fetch_add(t[i + 1] - t[i], std::memory_order_relaxed);
    }
    const u64 n = g_shader_split_ns[4].fetch_add(1) + 1;
    if ((n & 63) == 0) {
        std::fprintf(stderr, "SHADER_SPLIT n=%llu translate=%.2f emit=%.2f module=%.2f register=%.2f ms/shader new_programs=%llu new_permutations=%llu scouted_new_perm=%llu\n",
                     (unsigned long long)n, g_shader_split_ns[0] / 1e6 / n,
                     g_shader_split_ns[1] / 1e6 / n, g_shader_split_ns[2] / 1e6 / n,
                     g_shader_split_ns[3] / 1e6 / n, (unsigned long long)g_new_programs.load(),
                     (unsigned long long)g_new_permutations.load(),
                     (unsigned long long)g_scouted_new_perm.load());
    }
}

vk::ShaderModule PipelineCache::CompileModuleIR(Shader::Info& info,
                                                Shader::RuntimeInfo& runtime_info,
                                                const Shader::IR::Program& ir_program,
                                                const std::span<const u32>& code, size_t perm_idx,
                                                Shader::Backend::Bindings& binding,
                                                iss::g4::CpOwner owner) {
    Common::Perf::ScopedTimer compile_timer{Common::Perf::ShaderCompiles,
                                            Common::Perf::ShaderCompileNs};
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {} (translated in parallel)",
             info.hw_stage, info.pgm_hash, perm_idx != 0 ? "(permutation)" : "");
    iss::blackbox::Event(iss::blackbox::Compile, "compila shader estagio %d %llx%s",
                         int(info.hw_stage), (unsigned long long)info.pgm_hash,
                         perm_idx != 0 ? " (permutacao)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");
    vk::ShaderModule module;
    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        std::vector<u32> salted;
        module = CompileSPV(iss::shader_salt::Apply(spv, salted), instance.GetDevice());
    }
    if (owner == iss::g4::CpOwner::Graphics)
        RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    return module;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding,
                                              iss::g4::CpOwner owner) {
    Common::Perf::ScopedTimer compile_timer{Common::Perf::ShaderCompiles,
                                            Common::Perf::ShaderCompileNs};
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    iss::blackbox::Event(iss::blackbox::Compile, "compila shader estagio %d %llx%s",
                         int(info.hw_stage), (unsigned long long)info.pgm_hash,
                         perm_idx != 0 ? " (permutacao)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    std::array<u64, 5> split{};
    split[0] = Common::Perf::NowNs();
    auto& owner_pools = owner == iss::g4::CpOwner::Graphics ? pools : asc_lookup.pools;
    const auto ir_program = Shader::TranslateProgram(code, owner_pools, info, runtime_info, profile);
    split[1] = Common::Perf::NowNs();
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    split[2] = Common::Perf::NowNs();
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        std::vector<u32> salted;
        module = CompileSPV(iss::shader_salt::Apply(spv, salted), instance.GetDevice());
    }

    split[3] = Common::Perf::NowNs();
    if (owner == iss::g4::CpOwner::Graphics)
        RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    split[4] = Common::Perf::NowNs();
    NoteShaderSplit(split);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

namespace {

/// Permutation memo key: the fields of the resource descriptors the specialization reads
/// (Shader::StageSpecialization), the fetch shader code, the runtime info and the start
/// bindings. Addresses and sizes are left out, so that a shader bound with other buffers and
/// textures of the same layout finds its permutation.
template <typename InfoT>
u64 SpecializationKey(const InfoT& info, const Shader::RuntimeInfo& runtime_info,
                      const Shader::Backend::Bindings& binding) {
    boost::container::small_vector<u32, 160> words;
    words.push_back(binding.unified);
    words.push_back(binding.buffer);
    if (info.has_fetch_shader) {
        // The fetch shader tells the vertex attributes (their descriptors are checked against
        // the permutation found, see VertexAttributesMatch): its code, up to the end of its page.
        const u32* code;
        std::memcpy(&code, &info.user_data[info.fetch_shader_sgpr_base], sizeof(code));
        const auto address = reinterpret_cast<uintptr_t>(code);
        const u64 room = (0x1000 - (address & 0xFFF)) / sizeof(u32);
        const u64 count = std::min<u64>(room, 48);
        words.push_back(static_cast<u32>(address));
        words.push_back(static_cast<u32>(address >> 32));
        words.insert(words.end(), code, code + count);
    }
    // ISS_DYNAMIC_STRIDE: the stride is no part of the permutation either.
    const u32 dw1_mask =
        Shader::dynamic_buffer_stride && info.sw_stage == Shader::SwStage::Compute
            ? ~(0xFFu | (0x3FFFu << 16))
            : ~0xFFu;
    for (const auto& desc : info.buffers) {
        const auto sharp = desc.GetSharp(info);
        const auto* raw = reinterpret_cast<const u32*>(&sharp);
        words.push_back(static_cast<bool>(sharp));
        words.push_back(raw[1] & dw1_mask); // Stride, swizzle enable (not the address).
        words.push_back(raw[3]);          // Formats, swizzles, index stride, element size.
    }
    for (const auto& desc : info.images) {
        const auto sharp = desc.GetSharp(info);
        const auto* raw = reinterpret_cast<const u32*>(&sharp);
        words.push_back(static_cast<bool>(sharp));
        words.push_back(raw[1] & ~0x3Fu); // Formats (not the address).
        words.push_back(raw[3]);          // Swizzles, levels, type.
        words.push_back(raw[4]);
        words.push_back(raw[5]);
    }
    for (const auto& desc : info.fmasks) {
        const auto sharp = desc.GetSharp(info);
        const auto* raw = reinterpret_cast<const u32*>(&sharp);
        words.push_back(static_cast<bool>(sharp));
        words.push_back(raw[2]); // Width and height.
    }
    for (const auto& desc : info.samplers) {
        const auto sharp = desc.GetSharp(info);
        words.push_back(static_cast<u32>(sharp.raw0 & ((1u << 15) | (1u << 20))));
    }
    u64 key = XXH3_64bits(words.data(), words.size() * sizeof(u32));
    key = HashCombine(key, XXH3_64bits(&runtime_info, sizeof(runtime_info)));
    return key;
}

/// Whether the vertex attributes of the permutation (from its fetch shader) are specialized
/// as the descriptors bound now require (the key above leaves them out: their descriptors are
/// found by parsing the fetch shader).
template <typename InfoT>
bool VertexAttributesMatch(const InfoT& info, const Shader::RuntimeInfo& runtime_info,
                           const Shader::StageSpecialization& spec) {
    if (info.sw_stage != Shader::SwStage::Vertex) {
        return true;
    }
    const auto& attributes = spec.fetch_shader_data.attributes;
    if (spec.vs_attribs.size() != attributes.size()) {
        return false;
    }
    for (size_t i = 0; i < attributes.size(); ++i) {
        const auto& attribute = attributes[i];
        const auto sharp = attribute.GetSharp(info);
        Shader::VsAttribSpecialization expected{};
        if (sharp) {
            using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
            if (const auto step_rate = attribute.GetStepRate(); step_rate != InstanceIdType::None) {
                expected.divisor = step_rate == InstanceIdType::OverStepRate0
                                       ? runtime_info.sw.vs.step_rate_0
                                       : (step_rate == InstanceIdType::OverStepRate1
                                              ? runtime_info.sw.vs.step_rate_1
                                              : 1);
            }
            expected.num_class = AmdGpu::GetNumberClass(sharp.GetNumberFmt());
            expected.dst_select = sharp.DstSelect();
        }
        if (!(spec.vs_attribs[i] == expected)) {
            return false;
        }
    }
    return true;
}

} // Anonymous namespace

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    return GetProgram(iss::g4::CpOwner::Graphics, hw_stage, sw_stage, params, binding);
}

PipelineCache::Result PipelineCache::GetProgram(iss::g4::CpOwner owner, HwStage hw_stage,
                                                SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    if (owner == iss::g4::CpOwner::Graphics) graphics_reuse.valid = false;
    ASSERT_MSG(owner == iss::g4::CpOwner::Graphics ||
                   (hw_stage == HwStage::Compute && sw_stage == SwStage::Compute),
               "ASC pipeline partition accepts compute stages only");
    static const bool fast_lookup = [] {
        const char* value = std::getenv("SHADPS4_FAST_STAGE_LOOKUP");
        const bool enabled = value && value[0] == '1';
        LOG_INFO(Render_Vulkan, "Fast stage lookup: {}", enabled);
        return enabled;
    }();
    Common::Perf::Add(Common::Perf::SpecCalls);
    // Scout: the stage lookup precomputed for this draw, if its inputs are those of the draw
    // the command processor is at (the scout decides from its own copy of the registers).
    const AmdGpu::ScoutStage* scouted = nullptr;
    if (const auto* draw = owner == iss::g4::CpOwner::Graphics && !active_compute_snapshot
                               ? liverpool->CurrentScoutDraw() : nullptr; draw != nullptr) {
        for (u32 i = 0; i < draw->num_stages; ++i) {
            if (draw->stages[i].sw_stage == static_cast<u8>(sw_stage)) {
                scouted = &draw->stages[i];
                break;
            }
        }
        if (scouted == nullptr || !scouted->valid) {
            scouted = nullptr;
            Common::Perf::Add(Common::Perf::ScoutMisses);
        } else if (scouted->pgm_base != params.Base() || scouted->start != binding ||
                   scouted->ud_hash !=
                       XXH3_64bits(params.user_data.data(), params.user_data.size_bytes())) {
            scouted = nullptr;
            Common::Perf::Add(Common::Perf::ScoutStale);
        } else {
            Common::Perf::Add(Common::Perf::ScoutHits);
        }
    }
    auto& owner_runtime_infos = owner == iss::g4::CpOwner::Graphics ? runtime_infos
                                                                   : asc_lookup.runtime_infos;
    auto& owner_fetch_shader = owner == iss::g4::CpOwner::Graphics ? fetch_shader
                                                                   : asc_lookup.fetch_shader;
    auto& owner_programs = owner == iss::g4::CpOwner::Graphics ? program_cache
                                                               : asc_program_cache;
    const Shader::RuntimeInfo& current_runtime_info = [&]() -> const Shader::RuntimeInfo& {
        if (scouted != nullptr) {
            owner_runtime_infos[u32(sw_stage)] = scouted->GetRuntimeInfo();
            return owner_runtime_infos[u32(sw_stage)];
        }
        return BuildRuntimeInfo(hw_stage, sw_stage, owner);
    }();
    // BuildRuntimeInfo owns this storage; lookup-only paths do not mutate it.
    // Retain the old copy path for A/B. Compilation always gets a private copy.
    std::optional<Shader::RuntimeInfo> runtime_copy;
    if (!fast_lookup) runtime_copy.emplace(current_runtime_info);
    const auto& runtime_info = fast_lookup ? current_runtime_info : *runtime_copy;
    auto it_pgm = owner_programs.find(params.hash);
    const bool new_program = it_pgm == owner_programs.end();
    if (new_program) {
        // The vertex stage translated meanwhile by a helper thread, for the same inputs.
        PreTranslation* pre = nullptr;
        if (pre_translation != nullptr && owner == iss::g4::CpOwner::Graphics &&
            !active_compute_snapshot && sw_stage == SwStage::Vertex &&
            pre_translation->hash == params.hash && pre_translation->program &&
            std::memcmp(&pre_translation->initial_runtime_info, &runtime_info,
                        sizeof(Shader::RuntimeInfo)) == 0) {
            pre = pre_translation;
            pre->done.get(); // An exception of the translation is the serial one's.
            if (!pre->ir) {
                pre = nullptr;
            }
        }
        if (pre != nullptr && ParallelTranslateVerify()) {
            // The serial translation of the same inputs must emit the same SPIR-V.
            Shader::Pools verify_pools;
            auto check = std::make_unique<Program>(hw_stage, sw_stage, params);
            auto verify_runtime_info = runtime_info;
            const auto verify_ir = Shader::TranslateProgram(params.code, verify_pools, check->info,
                                                            verify_runtime_info, profile);
            auto serial_binding = binding;
            auto parallel_binding = binding;
            const auto serial = Shader::Backend::SPIRV::EmitSPIRV(profile, verify_runtime_info,
                                                                  verify_ir, serial_binding);
            const auto parallel = Shader::Backend::SPIRV::EmitSPIRV(profile, pre->runtime_info,
                                                                    *pre->ir, parallel_binding);
            g_pre_verified.fetch_add(1, std::memory_order_relaxed);
            if (serial != parallel || !(serial_binding == parallel_binding)) {
                g_pre_mismatches.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr, "PARALLEL_TRANSLATE_MISMATCH shader=%#llx\n",
                             (unsigned long long)params.hash);
                pre = nullptr; // The serial translation is used.
            }
        }
        // ISS_ASYNC_SKIP_DRAWS: not translated here; the speculative translator does it while
        // the draws using it are skipped.
        if (pre == nullptr && spec_translator && AsyncSkipDraws() &&
            owner == iss::g4::CpOwner::Graphics && !active_compute_snapshot &&
            (hw_stage == HwStage::Fragment || hw_stage == HwStage::Vertex)) {
            Shader::RuntimeInfo probe{};
            if (BuildRuntimeInfoFrom(probe, liverpool->regs, hw_stage, sw_stage,
                                     graphics_key.color_buffers, draw_indirect_params)) {
                const u64 spec_key = SpecKey(params.hash, probe);
                const int state = spec_translator->Peek(spec_key);
                if (state < 0) {
                    RequestSpecTranslation(hw_stage, sw_stage, params, probe);
                    g_async_requested.fetch_add(1, std::memory_order_relaxed);
                }
                if (state == int(SpecTranslation::Queued)) {
                    spec_translator->Prioritize(spec_key); // On screen now: first.
                }
                if (state < 0 || state == int(SpecTranslation::Queued) ||
                    state == int(SpecTranslation::Running)) {
                    async_translation_pending = true;
                    const u64 n = g_async_pending.fetch_add(1, std::memory_order_relaxed) + 1;
                    if ((n & 1023) == 1)
                        std::fprintf(stderr, "ASYNC_TRANSLATE skipped_lookups=%llu requested=%llu\n",
                                     (unsigned long long)n,
                                     (unsigned long long)g_async_requested.load());
                    return Result{nullptr, vk::ShaderModule{}, u64{0}};
                }
            }
        }
        // ISS_SPEC_TRANSLATE: the translation the scout started for this program and the
        // same runtime info (built the same way from these registers).
        std::shared_ptr<SpecTranslation> spec_entry;
        if (pre == nullptr && spec_translator && owner == iss::g4::CpOwner::Graphics &&
            !active_compute_snapshot &&
            (hw_stage == HwStage::Fragment || hw_stage == HwStage::Vertex ||
             (hw_stage == HwStage::Compute && SpecComputeEnabled()))) {
            Shader::RuntimeInfo probe{};
            const bool built =
                hw_stage == HwStage::Compute
                    ? (BuildComputeRuntimeInfoFrom(probe, liverpool->GetCsRegs()), true)
                    : BuildRuntimeInfoFrom(probe, liverpool->regs, hw_stage, sw_stage,
                                           graphics_key.color_buffers, draw_indirect_params);
            if (built) {
                spec_entry = spec_translator->Take(SpecKey(params.hash, probe));
                if (!spec_entry) {
                    g_spec_missed.fetch_add(1, std::memory_order_relaxed);
                } else if (!spec_translator->WaitDone(*spec_entry) || !spec_entry->ir ||
                           std::memcmp(&spec_entry->initial_runtime_info, &probe,
                                       sizeof(Shader::RuntimeInfo)) != 0) {
                    spec_entry.reset();
                }
            }
            NoteSpec();
        }
        if (spec_entry && SpecTranslateVerify()) {
            // The serial translation with the command processor's registers must emit the
            // same SPIR-V.
            Shader::Pools verify_pools;
            auto check = std::make_unique<Program>(hw_stage, sw_stage, params);
            auto verify_runtime_info = runtime_info;
            const auto verify_ir = Shader::TranslateProgram(params.code, verify_pools, check->info,
                                                            verify_runtime_info, profile);
            auto serial_binding = binding;
            auto spec_binding = binding;
            const auto serial = Shader::Backend::SPIRV::EmitSPIRV(profile, verify_runtime_info,
                                                                  verify_ir, serial_binding);
            const auto speculative = Shader::Backend::SPIRV::EmitSPIRV(
                profile, spec_entry->runtime_info, *spec_entry->ir, spec_binding);
            g_spec_verified.fetch_add(1, std::memory_order_relaxed);
            if (serial != speculative || !(serial_binding == spec_binding)) {
                g_spec_mismatches.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr, "SPEC_TRANSLATE_MISMATCH shader=%#llx stage=%u\n",
                             (unsigned long long)params.hash, static_cast<u32>(hw_stage));
                spec_entry.reset();
            }
        }
        auto new_program_ptr = pre != nullptr    ? std::move(pre->program)
                               : spec_entry != nullptr ? std::move(spec_entry->program)
                                                 : std::make_unique<Program>(hw_stage, sw_stage, params);
        if (pre != nullptr) {
            pre_translation = nullptr;
            g_pre_used.fetch_add(1, std::memory_order_relaxed);
        }
        if (spec_entry != nullptr) {
            // From now on the program reads the command processor's registers.
            new_program_ptr->info.user_data = params.user_data;
            g_spec_adopted.fetch_add(1, std::memory_order_relaxed);
        }
        Program* const program = new_program_ptr.get();
        if (active_compute_snapshot) {
            program->replay_user_data = active_compute_snapshot->user_data;
            program->info.user_data = program->replay_user_data;
            program->info.pgm_base = active_compute_snapshot->shader.guest_base;
        }
        auto start = binding;
        auto compile_runtime_info = pre != nullptr    ? pre->runtime_info
                                    : spec_entry != nullptr ? spec_entry->runtime_info
                                                      : runtime_info;
        g_new_programs.fetch_add(1, std::memory_order_relaxed);
        const Shader::IR::Program* translated = pre != nullptr    ? &*pre->ir
                                                : spec_entry != nullptr ? &*spec_entry->ir
                                                                  : nullptr;
        const auto module =
            translated != nullptr
                ? CompileModuleIR(program->info, compile_runtime_info, *translated, params.code, 0,
                                  binding, owner)
                : CompileModule(program->info, compile_runtime_info, params.code, 0, binding,
                                owner);
        if (spec_entry != nullptr) {
            // The flattened user data of this draw, from the command processor's registers.
            program->info.RefreshFlatBuf();
        }
        auto spec = Shader::StageSpecialization(program->info, compile_runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        if (owner == iss::g4::CpOwner::Graphics)
            RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        if (owner == iss::g4::CpOwner::Graphics) {
            // The scout only reads the graphics partition.
            std::unique_lock lock{program_cache_mutex};
            owner_programs.emplace(params.hash, std::move(new_program_ptr));
        } else {
            owner_programs.emplace(params.hash, std::move(new_program_ptr));
        }
        if (auto& fetch = program->modules[0].spec.fetch_shader_data; !fetch.Empty()) {
            owner_fetch_shader = &fetch;
        }
        if (owner == iss::g4::CpOwner::Graphics && !active_compute_snapshot)
            iss::dlss::NoteProgram(sw_stage == SwStage::Fragment, params.hash, params.Base(),
                                   program->info);
        return std::make_tuple(&program->info, module, perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = active_compute_snapshot ? active_compute_snapshot->shader.guest_base
                                             : params.Base(); // inline cbuffer address fixup
    if (owner == iss::g4::CpOwner::Graphics && !active_compute_snapshot)
        iss::dlss::NoteProgram(sw_stage == SwStage::Fragment, params.hash, params.Base(), info);
    if (active_compute_snapshot) {
        program->replay_user_data = active_compute_snapshot->user_data;
        info.user_data = program->replay_user_data;
    } else {
        info.user_data = params.user_data;
    }
    if (scouted != nullptr && scouted->flat_size == info.srt_info.flattened_bufsize_dw) {
        info.flattened_ud_buf.assign(scouted->flat, scouted->flat + scouted->flat_size);
    } else {
        scouted = nullptr;
        info.RefreshFlatBuf();
    }
    Common::Perf::Add(Common::Perf::SpecFlatDwords, info.flattened_ud_buf.size());
    if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
        // Frame x-ray: the GCN code of the shaders used (once per shader).
        if (xray_dumped.insert(params.hash).second) {
            using namespace Common::FS;
            const auto dir = GetUserPath(PathType::ShaderDir) / "xray";
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            const auto file = IOFile{dir / fmt::format("{}_{:#018x}.bin", hw_stage, params.hash),
                                     FileAccessMode::Create};
            file.WriteSpan(params.code);
        }
    }

    // Permutation memo: the specialization is a function of a few fields of the resource
    // descriptors (formats, strides, swizzles: not their addresses), of the fetch shader, the
    // runtime info and the start bindings. Bound again with the same ones, the stage reuses the
    // permutation found then instead of walking its resources again (and parsing its fetch
    // shader). Tessellation stages also read constants from memory: not memoized.
    const bool memoize = Common::Perf::GetTuning().spec_memo.load(std::memory_order_relaxed) &&
                         sw_stage != SwStage::TessellationControl &&
                         sw_stage != SwStage::TessellationEval;
    u64 memo_key = 0;
    if (memoize) {
        memo_key = scouted != nullptr ? scouted->memo_key
                                      : SpecializationKey(info, runtime_info, binding);
        if (fast_lookup) {
            Common::Perf::Add(Common::Perf::StageLastMemoChecks);
            const auto recent = program->last_stage_lookup.Find(
                memo_key, program->modules.size(), [&](u32 index) {
                    return VertexAttributesMatch(info, runtime_info, program->modules[index].spec);
                });
            if (recent) {
                Common::Perf::Add(Common::Perf::StageLastMemoHits);
                Common::Perf::Add(Common::Perf::SpecMemoHits);
                info.AddBindings(binding);
                auto& perm = program->modules[*recent];
                if (auto& fetch = perm.spec.fetch_shader_data; !fetch.Empty())
                    owner_fetch_shader = &fetch;
                return std::make_tuple(&program->info, perm.module,
                                       HashCombine(params.hash, u64{*recent}));
            }
        }
        for (const auto& entry : program->memo) {
            // The command processor is the only writer: plain (relaxed) reads.
            const u32 entry_perm = entry.perm_idx.load(std::memory_order_relaxed);
            if (entry.valid.load(std::memory_order_relaxed) &&
                entry.key.load(std::memory_order_relaxed) == memo_key &&
                entry_perm < program->modules.size() &&
                VertexAttributesMatch(info, runtime_info, program->modules[entry_perm].spec)) {
                Common::Perf::Add(Common::Perf::SpecMemoHits);
                if (fast_lookup) program->last_stage_lookup.Remember(memo_key, entry_perm);
                info.AddBindings(binding);
                auto& perm = program->modules[entry_perm];
                if (auto& fetch = perm.spec.fetch_shader_data; !fetch.Empty()) {
                    owner_fetch_shader = &fetch;
                }
                return std::make_tuple(&program->info, perm.module,
                                       HashCombine(params.hash, u64{entry_perm}));
            }
        }
        Common::Perf::Add(Common::Perf::SpecMemoMisses);
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        auto compile_runtime_info = runtime_info;
        g_new_permutations.fetch_add(1, std::memory_order_relaxed);
        if (scouted != nullptr) {
            g_scouted_new_perm.fetch_add(1, std::memory_order_relaxed);
        }
        module = CompileModule(new_info, compile_runtime_info, params.code, perm_idx, binding,
                               owner);

        if (owner == iss::g4::CpOwner::Graphics)
            RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        if (owner == iss::g4::CpOwner::Graphics) {
            // The scout reads graphics permutations under the same lock.
            std::unique_lock lock{program_cache_mutex};
            program->AddPermut(module, std::move(spec));
        } else {
            program->AddPermut(module, std::move(spec));
        }
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    if (memoize) {
        program->WriteMemo(memo_key, static_cast<u32>(perm_idx));
    }
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        owner_fetch_shader = &fetch;
    }
    return std::make_tuple(&program->info, module, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    graphics_reuse.valid = false;
    // Pipelines being compiled use the modules and can be erased below.
    compiler->WaitIdle();
    std::optional<vk::ShaderModule> new_module{};
    for (auto* programs : {&program_cache, &asc_program_cache}) {
        for (const auto& [_, program] : *programs) {
            for (auto& m : program->modules) {
                if (m.module == module) {
                    const auto& d = instance.GetDevice();
                    d.destroyShaderModule(m.module);
                    m.module = CompileSPV(spv_code, d);
                    new_module = m.module;
                }
            }
        }
    }
    std::scoped_lock lock{module_related_pipelines_mutex};
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                last_graphics_pipeline = nullptr;
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
                asc_compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
