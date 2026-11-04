// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_gbuffer_rasterizer.inl.
//
// ISS_NATIVE_PASSES=1 (with ISS_NATIVE_MATERIALS=1): the Vulkan side of native_passes.h.
//  - Engine threads: each draw of a lane is recorded into its segment's secondary command
//    buffer from the lane's registers and the material template of the draw's render state
//    (PassKey: every graphics context and vertex/pixel shader register but the user data, the
//    per draw words and the target addresses). Buffers resolve without the buffer cache: the
//    imported heap by address, the others in their published arena (as native_recorder.inl).
//  - Command processor: a segment is executed at its first draw once its assumptions hold (the
//    same render state and user data at the first draw, templates still current, arena reads
//    still in place, the same attachment formats); a learn request draws through the full path
//    and keeps the template under the command processor's PassKey of the draw.

namespace NativePasses {
namespace np = iss::native_passes;
using Template = Rasterizer::NativeMaterialTemplate;

constexpr u64 RingSize = 16ull << 20;

struct Globals {
    const Instance* instance{};
    Scheduler* scheduler{};
    VideoCore::BufferCache* buffer_cache{};
    PipelineCache* pipeline_cache{}; ///< speculative compute translations (ISS_SPEC_COMPUTE)
    const VideoCore::TextureCache* texture_cache{}; ///< NativeMaterial::HandlesAlive
    std::atomic<const VideoCore::Buffer*> unified{};
    std::atomic<VAddr> unified_base{};
    std::atomic<bool> ready{};
};
static Globals g;

// Keys the command processor gave up learning.
struct Unlearnable {
    std::shared_mutex mutex;
    std::unordered_set<u64> keys;
    std::unordered_map<u64, u32> attempts; ///< command processor only
};
static Unlearnable& Rejected() {
    static Unlearnable u;
    return u;
}

// The render state key (both sides compute it from their registers).
static u64 PassKey(const AmdGpu::Regs& regs) {
    constexpr u32 CtxFirst = 0xA000, CtxCount = 0x400, ShFirst = 0x2C00, ShCount = 0x80;
    static thread_local std::array<u32, CtxCount + ShCount + 1> words;
    std::memcpy(words.data(), &regs.reg_array[CtxFirst], CtxCount * 4);
    std::memcpy(words.data() + CtxCount, &regs.reg_array[ShFirst], ShCount * 4);
    NativeMaterial::ClearDrawWords(regs, words.data(), CtxFirst, CtxCount);
    for (u32 k = 0; k < np::UserWords; ++k) {
        words[CtxCount + (np::PsUser0 - ShFirst) + k] = 0;
        words[CtxCount + (np::VsUser0 - ShFirst) + k] = 0;
    }
    // The target and depth addresses: the command processor attaches the real targets and
    // compares the formats; alternating targets would otherwise change the key every frame.
    for (const u32 reg : {0x005u, 0x012u, 0x013u, 0x014u, 0x015u}) words[reg] = 0;
    for (u32 n = 0; n < 8; ++n)
        for (const u32 field : {0u, 7u, 9u}) words[0x318 + n * 15 + field] = 0;
    words[CtxCount + ShCount] = u32(regs.primitive_type);
    return XXH3_64bits(words.data(), sizeof(words)) | 1;
}

// ISS_NATIVE_PASSES_DYNAMIC=1: dynamic state (viewports, scissors, depth bounds/bias, stencil
// reference, blend constants) recorded from each draw's registers and left out of the key.
// Default (as the 16/10 run "h"): it is part of the key and comes from the template.
static bool DynamicMode() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_DYNAMIC");
        return p && p[0] == '1';
    }();
    return enabled;
}

// The words of the render state key, as PassKey: graphics context and vertex/pixel shader,
// without user data, per draw words and target addresses; the primitive type last.
// What is left out never depends on the values, so it is one AND mask, built once (the command
// processor computes this per executed segment: the per word checks cost ~1 ms of it a frame).
static const std::array<u32, 0x400 + 0x80>& StateKeep(const AmdGpu::Regs& regs) {
    static const auto keep = [&regs] {
        constexpr u32 CtxFirst = 0xA000, CtxCount = 0x400, ShFirst = 0x2C00;
        std::array<u32, CtxCount + 0x80> k;
        k.fill(~0u);
        NativeMaterial::ClearDrawWords(regs, k.data(), CtxFirst, CtxCount);
        for (u32 n = 0; n < np::UserWords; ++n) {
            k[CtxCount + (np::PsUser0 - ShFirst) + n] = 0;
            k[CtxCount + (np::VsUser0 - ShFirst) + n] = 0;
        }
        // What only the command processor's target setup reads: the depth view, HTile, clear
        // values, bases and extents; per color target all but its format (INFO) and samples
        // (ATTRIB). Alternating targets and clear values would otherwise change the key.
        for (const u32 reg : {0x002u, 0x005u, 0x00Au, 0x00Bu, 0x012u, 0x013u, 0x014u, 0x015u,
                              0x016u, 0x017u})
            k[reg] = 0;
        for (u32 n = 0; n < 8; ++n)
            for (u32 field = 0; field < 15; ++field)
                if (field != 4 && field != 5) k[0x318 + n * 15 + field] = 0;
        // Of CB_COLORn_ATTRIB, the sample and fragment counts (not the tiling).
        for (u32 n = 0; n < 8; ++n) k[0x318 + n * 15 + 5] &= 0x3F000u;
        // Dynamic state is recorded from each draw's registers; the unread ones are not state.
        for (u32 w = 0; w < CtxCount; ++w)
            if ((DynamicMode() && np::IsDynamicReg(CtxFirst + w)) || np::IsUnreadReg(CtxFirst + w))
                k[w] = 0;
        return k;
    }();
    return keep;
}
static void StateWords(const AmdGpu::Regs& regs, std::array<u32, 0x400 + 0x80 + 1>& words) {
    constexpr u32 CtxFirst = 0xA000, CtxCount = 0x400, ShFirst = 0x2C00, ShCount = 0x80;
    const auto& keep = StateKeep(regs);
    const u32* ctx = &regs.reg_array[CtxFirst];
    const u32* sh = &regs.reg_array[ShFirst];
    for (u32 w = 0; w < CtxCount; ++w) words[w] = ctx[w] & keep[w];
    for (u32 w = 0; w < ShCount; ++w) words[CtxCount + w] = sh[w] & keep[CtxCount + w];
    words[CtxCount + ShCount] = u32(regs.primitive_type);
}
// Mask bit of state word w (its tracked index: context 0..0x3FF, shader 0x400.., the primitive
// type at VGT_PRIMITIVE_TYPE 0xC242).
static bool MaskHas(const u32* mask, u32 w) {
    const u32 t = w < 0x480 ? w : 0x480 + (0xC242 - 0xC240);
    return (mask[t >> 5] >> (t & 31)) & 1;
}
// The key of the words the lane wrote (and of which ones they are).
static u64 MaskedKey(const AmdGpu::Regs& regs, const u32* mask) {
    static thread_local std::array<u32, 0x400 + 0x80 + 1> words;
    StateWords(regs, words);
    for (u32 w = 0; w < words.size(); ++w) words[w] &= 0u - u32(MaskHas(mask, w));
    NativeMaterial::Hasher h;
    h.Put(words.data(), sizeof(words));
    h.Put(mask, np::MaskWords * 4);
    return h.Digest() | 1;
}
// The words the lane did not write: what a template learned there also depended on.
static u64 InheritedHash(const AmdGpu::Regs& regs, const u32* mask) {
    static thread_local std::array<u32, 0x400 + 0x80 + 1> words;
    StateWords(regs, words);
    for (u32 w = 0; w < words.size(); ++w) words[w] &= u32(MaskHas(mask, w)) - 1u;
    return XXH3_64bits(words.data(), sizeof(words));
}

// The same keys from state words taken once (StateWords) for a whole segment check: the command
// processor checks ~16 000 segments a second in the city, each with a MaskedKey and one
// InheritedHash per mask, and every call took the 1 153 words again and tested the mask bit by
// bit (~10% of the command processor). A mask expanded to one AND word per state word is kept
// (its content is the key): masking is then a plain vectorized AND. Results are bit identical.
using StateWordArray = std::array<u32, 0x400 + 0x80 + 1>;
static const StateWordArray& ExpandedMask(const u32* mask) {
    struct Entry {
        std::array<u32, np::MaskWords> mask;
        StateWordArray expanded;
    };
    static thread_local std::unordered_map<u64, std::unique_ptr<Entry>> cache;
    const u64 h = XXH3_64bits(mask, np::MaskWords * 4);
    auto& slot = cache[h];
    if (!slot || std::memcmp(slot->mask.data(), mask, np::MaskWords * 4) != 0) {
        if (cache.size() > 65536) {
            cache.clear();
            return ExpandedMask(mask);
        }
        slot = std::make_unique<Entry>();
        std::memcpy(slot->mask.data(), mask, np::MaskWords * 4);
        for (u32 w = 0; w < slot->expanded.size(); ++w)
            slot->expanded[w] = 0u - u32(MaskHas(mask, w));
    }
    return slot->expanded;
}
static u64 MaskedKeyWords(const StateWordArray& state, const u32* mask) {
    static thread_local StateWordArray words;
    const auto& m = ExpandedMask(mask);
    for (u32 w = 0; w < words.size(); ++w) words[w] = state[w] & m[w];
    NativeMaterial::Hasher h;
    h.Put(words.data(), sizeof(words));
    h.Put(mask, np::MaskWords * 4);
    return h.Digest() | 1;
}
static u64 InheritedHashWords(const StateWordArray& state, const u32* mask) {
    static thread_local StateWordArray words;
    const auto& m = ExpandedMask(mask);
    for (u32 w = 0; w < words.size(); ++w) words[w] = state[w] & ~m[w];
    return XXH3_64bits(words.data(), sizeof(words));
}

// The registers the recording reads from the lane's own values (not from the template): the
// index offset (base vertex), primitive restart and its index, the instance step rates.
static u64 AssumedHash(const AmdGpu::Regs& regs) {
    const std::array<u32, 5> words{regs.reg_array[0xA102], regs.reg_array[0xA103],
                                   regs.reg_array[0xA2A5], regs.reg_array[0xA2A8],
                                   regs.reg_array[0xA2A9]};
    return XXH3_64bits(words.data(), sizeof(words));
}

// Keys the command processor learned: the inherited words they were learned with (command
// processor only), and the keys whose inherited words changed (the engine threads ask for them
// to be learned again).
struct Learned {
    /// Per key, the inherited words it is valid with: those of its learning, and those of later
    /// learnings that made the very same template (pipeline and dynamic state).
    std::unordered_map<u64, std::vector<u64>> inherited;
    std::shared_mutex mutex;
    std::unordered_set<u64> relearn;
};
static Learned& LearnedKeys() {
    static Learned l;
    return l;
}

// Diagnosis (command processor only): the inherited words of sampled keys at their learning, and
// which words differ when a segment is rejected for them.
struct InheritedDiag {
    std::unordered_map<u64, std::vector<u32>> words;
    std::array<u64, 0x400 + 0x80 + 1> differ{};
    u64 samples{};
};
static InheritedDiag& InhDiag() {
    static InheritedDiag d;
    return d;
}
static std::vector<u32> InheritedWords(const AmdGpu::Regs& regs, const u32* mask) {
    std::array<u32, 0x400 + 0x80 + 1> words;
    StateWords(regs, words);
    for (u32 w = 0; w < words.size(); ++w) words[w] &= u32(MaskHas(mask, w)) - 1u;
    return {words.begin(), words.end()};
}

// A lane's command pool: its segments' secondaries are recorded and ended under the lane's lock
// (native_passes.cpp), on whichever thread holds it.
struct LanePool {
    vk::CommandPool pool{};
};

struct Rec {
    vk::CommandBuffer cmd{};
    u32 generation{}; ///< of the segment it records for
    bool begun{};
    bool needs_reset{};
    const Template* current{};
    vk::Pipeline bound_pipeline{};
    std::vector<std::shared_ptr<const Template>> templates; ///< unique, the first sets formats
    std::vector<NativeRecorder::ArenaRead> arena_reads;      ///< unique
    /// BufferCache::ArenaBindingGeneration when its arena reads were resolved (arena_gen_mixed:
    /// not all at the same one). Unchanged at execution: the same arenas back the same pages
    /// (residency only grows), so only their CPU writes are checked.
    u64 arena_gen{};
    bool arena_gen_mixed{};
    /// The buffer ranges its draws read, merged (computed when it closes, on the engine thread:
    /// the command processor orders them after their writers).
    struct Read {
        const VideoCore::Buffer* buffer;
        u64 offset, size;
    };
    std::vector<Read> reads;
    /// The textures of its templates, each (image, layout, range) once (computed when it closes:
    /// the command processor's checks and transitions walk them, not every template's).
    std::vector<const Rasterizer::NativeImageCapture*> images;
    bool reads_unified{};
    VAddr unified_lo{}, unified_hi{};
    std::vector<const void*> workers; ///< the workers that hold ring space for it
    /// The render state keys its draws used, each with the lane's written mask then (the first
    /// is the first draw's).
    std::vector<std::pair<u64, std::array<u32, np::MaskWords>>> keys;
    /// Dynamic state groups its draws read (DynGroup bits).
    u32 dyn_used{};
    /// AssumedHash of the lane's registers at the first draw (its draws were recorded with them;
    /// later draws of the segment only write them explicitly).
    u64 lane_assumed{};
    /// GPU black box (gpu_blackbox.h): this recording's segment number (0: off) and its draws.
    u32 bb_segment{};
    u32 bb_draws{};
    /// The depth/stencil attachment class of the first draw (DepthStencilClass, ~0u: none yet).
    u32 first_ds{~0u};
    /// DrawClass bits of its draws.
    u32 classes{};
    /// ISS_NATIVE_PASSES_FLAT_CHECK: hash of the first draw's flattened user data (the descriptors
    /// and constants its shaders load through the user data pointers) as it was recorded.
    u64 first_flat{};
};

// ISS_NATIVE_PASSES_FLAT_CHECK (diagnosis 1, 2 also rejects): the command processor walks the
// first draw's user data pointers again when it executes a segment and compares with what the
// segment was recorded with: guest memory the engine filled after the recording reached the draw
// would be drawn stale (the late map structures since native draws, user 2026-10-18).
static int FlatCheck() {
    static const int mode = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_FLAT_CHECK");
        return p ? std::atoi(p) : 0;
    }();
    return mode;
}
static u64 HashWords(u64 h, std::span<const u32> words) {
    for (const u32 w : words) h = (h ^ w) * 0x100000001b3ull;
    return h;
}
// On a recording thread: the views InstallViews snapshotted.
static u64 RecordedFlatHash(const GraphicsPipeline& pipeline) {
    u64 h = 0xcbf29ce484222325ull;
    for (const auto* stage : pipeline.GetStages())
        if (stage) h = HashWords(h, stage->FlatUserData());
    return h;
}

// ISS_NATIVE_PASSES_GUARD_DEPTH (default 1): the command processor attaches the depth/stencil
// target of a whole segment in the layout its first draw selects (Rasterizer::BeginRendering:
// depth write -> writable, else stencil -> stencil writable, else read only). A later draw that
// writes depth or stencil would write into a read only layout (undefined; on AMD the depth
// compression metadata goes stale and later reads of that depth can stall the GPU). Such a draw
// starts a new segment.
static bool GuardDepth() {
    static const bool on = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_GUARD_DEPTH");
        return !p || p[0] != '0';
    }();
    return on;
}
static u32 DepthStencilClass(const AmdGpu::Regs& regs) {
    const bool attached = (regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
                          (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid());
    if (!attached) return 0;
    return 1u | (regs.depth_control.depth_write_enable ? 2u : 0u) |
           (regs.depth_control.stencil_enable ? 4u : 0u);
}
static std::atomic<u64> g_depth_mix{};
static std::atomic<u64> g_auto_draws{}; ///< DRAW_INDEX_AUTO draws recorded

// ISS_NATIVE_PASSES_CLASSES (bits, default 7): which draws the native passes record. 1 color
// without blending (opaque), 2 depth only (shadows), 4 color with blending. The others are left to
// the command processor. Blended segments hung the GPU a few frames later in a command processor
// draw (lab bisection 2026-10-17): only those into the pause UI layer, see BlendTargetAllowed.
static u32 AllowedClasses() {
    static const u32 bits = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_CLASSES");
        return p ? u32(std::strtoul(p, nullptr, 0)) : 7u;
    }();
    return bits;
}
static u32 DrawClass(const AmdGpu::Regs& regs) {
    bool color = false, blend = false;
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Disable)
        for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i)
            if (regs.color_buffers[i] && regs.color_target_mask.GetMask(i)) {
                color = true;
                blend |= regs.blend_control[i].enable != 0;
            }
    return !color ? 2u : blend ? 4u : 1u;
}
static std::array<std::atomic<u64>, 8> g_class_left{};

static std::string p_list(const std::vector<u64>& list) {
    std::string text;
    for (const u64 a : list) {
        char item[24];
        std::snprintf(item, sizeof(item), "%s%llx", text.empty() ? "" : ",", (unsigned long long)a);
        text += item;
    }
    return text;
}
// ISS_NATIVE_PASSES_BLEND_TARGETS (hex addresses, comma separated; unset: all): blended draws are
// recorded only when one of their blended color targets is listed (lab bisection by target).
static bool BlendTargetAllowed(const AmdGpu::Regs& regs) {
    static const std::vector<u64> list = [] {
        std::vector<u64> out;
        if (const char* p = std::getenv("ISS_NATIVE_PASSES_BLEND_TARGETS")) {
            std::string text = p;
            size_t at = 0;
            while (at < text.size()) {
                const size_t end = text.find(',', at);
                const std::string item = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
                if (!item.empty()) out.push_back(std::strtoull(item.c_str(), nullptr, 16));
                if (end == std::string::npos) break;
                at = end + 1;
            }
        }
        std::fprintf(stderr, "NATIVE_PASSES blend targets: %s\n", out.empty() ? "all" : p_list(out).c_str());
        return out;
    }();
    // ISS_NATIVE_PASSES_BLEND_DENY (hex addresses; default 203d70000; "0" none): blended draws into
    // these color targets are left to the command processor. 0x203d70000 is a G-buffer target
    // that the pause menu reuses as its UI layer (1920x1080, blended, interleaved with command
    // processor draws): native blended segments there hung the GPU (lab bisection 16:19); native
    // blended decals (0x291e10000) and transparents (0x294df0000) did not.
    static const std::vector<u64> deny = [] {
        std::vector<u64> out;
        const char* p = std::getenv("ISS_NATIVE_PASSES_BLEND_DENY");
        std::string text = p ? p : "203d70000";
        size_t at = 0;
        while (at < text.size()) {
            const size_t end = text.find(',', at);
            const std::string item = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
            if (const u64 a = std::strtoull(item.c_str(), nullptr, 16)) out.push_back(a);
            if (end == std::string::npos) break;
            at = end + 1;
        }
        std::fprintf(stderr, "NATIVE_PASSES blend deny: %s\n", out.empty() ? "none" : p_list(out).c_str());
        return out;
    }();
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i)
        if (regs.color_buffers[i] && regs.color_target_mask.GetMask(i) &&
            std::find(deny.begin(), deny.end(), regs.color_buffers[i].Address()) != deny.end())
            return false;
    if (list.empty()) return true;
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i)
        if (regs.color_buffers[i] && regs.color_target_mask.GetMask(i) && regs.blend_control[i].enable &&
            std::find(list.begin(), list.end(), regs.color_buffers[i].Address()) != list.end())
            return true;
    return false;
}

// ISS_NATIVE_PASSES_EXEC_CLASSES (bits, default 7): segments with a class outside it are recorded
// but left to the command processor at execution (survey of a class without running it).
static u32 ExecClasses() {
    static const u32 bits = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_EXEC_CLASSES");
        return p ? u32(std::strtoul(p, nullptr, 0)) : 7u;
    }();
    return bits;
}
// Survey by the segment's classes (index = class bits): segments checked, segments with a texture
// that is another image over the memory of one of its attachments (aliasing), the same image.
struct AliasSurvey {
    std::array<std::atomic<u64>, 8> segments{}, alias_color{}, alias_depth{}, same_image{};
    std::atomic<u64> exec_left{}, feedback_left{}, same_vkimage{}, layout_drift{};
    std::array<std::atomic<u64>, 8> depth_textures{}; ///< segments sampling a depth image, by class
    std::atomic<u64> depth_left{};
    std::atomic<u32> logged{};
};
static AliasSurvey g_alias;
// ISS_NATIVE_PASSES_NO_DEPTH_TEXTURES=1: segments that sample a depth image are left to the command
// processor (lab test of the depth layout mismatch the validation layer reports).
static bool NoDepthTextures() {
    static const bool on = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_NO_DEPTH_TEXTURES");
        return p && p[0] == '1';
    }();
    return on;
}
static bool GuardFeedback() {
    static const bool on = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_GUARD_FEEDBACK");
        return !p || p[0] != '0';
    }();
    return on;
}

// Dynamic state groups: always read (viewports, scissors, line width, blend constants), and
// read only when enabled (depth bounds, stencil reference and masks, polygon offset).
enum DynGroup : u32 { DynAlways = 1, DynBounds = 2, DynStencil = 4, DynBias = 8 };
static u32 DynGroupOf(u32 reg) {
    const u32 r = reg - 0xA000;
    if (r == 0x008 || r == 0x009) return DynBounds;
    if (r == 0x10C || r == 0x10D) return DynStencil;
    if (r >= 0x2DE && r <= 0x2E3) return DynBias;
    return DynAlways;
}

// An engine thread's upload ring and descriptor pools. What a segment took is released once the
// segment is retired, in the order the thread took it.
constexpr u32 PoolSets = 256;
struct Use {
    np::Segment* segment;
    u32 generation;
    u64 ring_end;
    u64 seq;
};
struct Worker {
    vk::Device device{};
    vk::Buffer ring{};
    vk::DeviceMemory memory{};
    u8* mapped{};
    u64 head{}, tail{};
    std::deque<Use> uses;
    u64 next_seq{};
    vk::DescriptorPool current_pool{};
    u64 current_last_user{};
    struct Full {
        vk::DescriptorPool pool;
        u64 last_user;
    };
    std::deque<Full> full_pools;
    std::vector<vk::DescriptorPool> free_pools;

    static bool Retired(const Use& u) {
        if (u.segment->generation != u.generation) return true; // recycled: it was retired
        const u32 state = u.segment->state.load(std::memory_order_acquire);
        if (state == np::Abandoned || state == np::Free) return true;
        if (state != np::Done) return false;
        return u.segment->executed_tick.load(std::memory_order_relaxed) <=
               g.scheduler->GetWorkSemaphore()->KnownGpuTick();
    }

    void Reclaim() {
        while (!uses.empty() && Retired(uses.front())) {
            const Use u = uses.front();
            uses.pop_front();
            tail = u.ring_end;
            while (!full_pools.empty() && full_pools.front().last_user <= u.seq) {
                device.resetDescriptorPool(full_pools.front().pool);
                free_pools.push_back(full_pools.front().pool);
                full_pools.pop_front();
            }
        }
    }

    // The use of the segment by this thread (one per segment and thread).
    template <typename R> // Rec, or NativeCompute::Rec
    u64 Note(np::Segment& s, R& r) {
        const bool listed = std::find(r.workers.begin(), r.workers.end(), this) != r.workers.end();
        if (!listed) {
            r.workers.push_back(this);
            uses.push_back({&s, s.generation, head, next_seq++});
        }
        return uses.back().seq;
    }

    u8* Upload(u64 bytes, u64 alignment, u64& offset) {
        u64 at = (head + alignment - 1) & ~(alignment - 1);
        if ((at % RingSize) + bytes > RingSize) at += RingSize - at % RingSize;
        if (at + bytes - tail > RingSize) return nullptr;
        head = at + bytes;
        if (!uses.empty()) uses.back().ring_end = head;
        offset = at % RingSize;
        return mapped + offset;
    }

    vk::DescriptorSet AllocateSet(vk::DescriptorSetLayout layout, u64 seq) {
        for (u32 attempt = 0; attempt < 2; ++attempt) {
            if (!current_pool) {
                if (!free_pools.empty()) {
                    current_pool = free_pools.back();
                    free_pools.pop_back();
                } else {
                    const std::array<vk::DescriptorPoolSize, 3> sizes{{
                        {vk::DescriptorType::eStorageBuffer, PoolSets * 16},
                        {vk::DescriptorType::eSampledImage, PoolSets * 48},
                        {vk::DescriptorType::eSampler, PoolSets * 16},
                    }};
                    vk::DescriptorPool pool{};
                    const vk::DescriptorPoolCreateInfo info{
                        .maxSets = PoolSets,
                        .poolSizeCount = u32(sizes.size()),
                        .pPoolSizes = sizes.data(),
                    };
                    if (device.createDescriptorPool(&info, nullptr, &pool) != vk::Result::eSuccess)
                        return {};
                    current_pool = pool;
                }
            }
            const vk::DescriptorSetAllocateInfo info{
                .descriptorPool = current_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &layout,
            };
            vk::DescriptorSet set{};
            if (device.allocateDescriptorSets(&info, &set) == vk::Result::eSuccess) {
                current_last_user = seq;
                return set;
            }
            full_pools.push_back({current_pool, current_last_user});
            current_pool = vk::DescriptorPool{};
        }
        return {};
    }

    bool Init() {
        const auto& instance = *g.instance;
        device = instance.GetDevice();
        ring = Check(device.createBuffer(vk::BufferCreateInfo{
            .size = RingSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer,
        }));
        const auto requirements = device.getBufferMemoryRequirements(ring);
        const auto& properties = instance.GetMemoryProperties();
        const auto want = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;
        // Device local first (host visible VRAM: ~256 MB without resizable BAR, shared with the
        // driver), then host memory when it is full: 10 pool threads ran out of it at startup
        // (a1-pool10-job48 2026-10-18, ErrorOutOfDeviceMemory).
        std::array<u32, 2> types{~0u, ~0u};
        for (u32 pass = 0; pass < 2; ++pass)
            for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
                const auto flags = properties.memoryTypes[i].propertyFlags;
                if (!(requirements.memoryTypeBits & (1u << i)) || (flags & want) != want) continue;
                if (pass == 0 && !(flags & vk::MemoryPropertyFlagBits::eDeviceLocal)) continue;
                types[pass] = i;
                break;
            }
        for (const u32 type : types) {
            if (type == ~0u || memory) continue;
            const auto [result, allocated] = device.allocateMemory(vk::MemoryAllocateInfo{
                .allocationSize = requirements.size,
                .memoryTypeIndex = type,
            });
            if (result == vk::Result::eSuccess) memory = allocated;
        }
        if (!memory) return false;
        Check(device.bindBufferMemory(ring, memory, 0));
        mapped = static_cast<u8*>(Check(device.mapMemory(memory, 0, RingSize)));
        return mapped != nullptr;
    }
};

static Worker* ThisWorker() {
    static thread_local std::unique_ptr<Worker> worker;
    static thread_local bool failed = false;
    if (worker || failed) return worker.get();
    auto w = std::make_unique<Worker>();
    bool ok = false;
    try {
        ok = w->Init();
    } catch (...) {
        ok = false;
    }
    if (!ok) {
        failed = true;
        std::fprintf(stderr, "NATIVE_PASSES worker setup failed\n");
        return nullptr;
    }
    worker = std::move(w);
    return worker.get();
}

enum Miss : u32 { MissState, MissShape, MissTemplate, MissViews, MissPipeline, MissFormat,
                  MissBuffer, MissRing, MissPool, MissCommand, MissUnified, MissCount };
constexpr const char* miss_names[MissCount] = {"state", "no_shape", "no_template", "views",
                                               "pipeline", "format", "buffer", "ring", "pool",
                                               "command", "not_ready"};
struct EngineStats {
    std::array<std::atomic<u64>, MissCount> miss{};
    std::atomic<u64> draws{}, empty{};
};
static EngineStats engine_stats;

// The dynamic state of a draw from its registers, as the command processor's
// Rasterizer::UpdateDynamicState (viewport/scissor, depth/stencil, primitive, rasterization,
// color blending; no attachment feedback loop: such draws are not recorded).
static void DrawDynamicState(const AmdGpu::Regs& regs, const Instance& instance,
                             const GraphicsPipeline& pipeline, DynamicState& dynamic_state) {
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
        s16(regs.generic_scissor.top_left_x), enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y), enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x, enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y, enable_offset ? regs.window_offset.window_y_offset : 0);
    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;
    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) continue;
        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;
        vk::Viewport viewport{};
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }
        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }
        if (regs.IsClipDisabled()) {
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
            vp_scsr.top_left_x = std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y = std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
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
        viewports.push_back(vk::Viewport{.x = -1.0f, .y = -1.0f, .width = 1.0f, .height = 1.0f,
                                         .minDepth = 0.0f, .maxDepth = 1.0f});
        scissors.push_back(vk::Rect2D{.offset = {0, 0}, .extent = {1, 1}});
    }
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);

    const auto depth_test_enabled = regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }
    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
                                   regs.poly_offset.depth_bias,
                                   (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }
    const auto stencil_test_enabled = regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
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
        const auto back = regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp || zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op = uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op = regs.depth_control.backface_enable
                                 ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                                 : front_op;
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList || topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        return type == AmdGpu::PrimitiveType::PatchPrimitive || type == AmdGpu::PrimitiveType::QuadList ||
               type == AmdGpu::PrimitiveType::RectList;
    };
    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(LiverpoolToVK::FrontFace(regs.polygon_control.front_face));
    dynamic_state.SetLineWidth(regs.line_control.Width());
    dynamic_state.SetBlendConstants(regs.blend_constants);
    if (iss::dlss::UpscaleWanted()) [[unlikely]] {
        // D2: HUD draws recorded by the engine threads also write the layer's alpha.
        auto masks = pipeline.GetGraphicsKey().write_masks;
        iss::dlss::ApplyHudLayerMask(masks, regs.color_buffers[0].Address(), 1);
        dynamic_state.SetColorWriteMasks(masks);
    } else {
        dynamic_state.SetColorWriteMasks(pipeline.GetGraphicsKey().write_masks);
    }
    dynamic_state.SetAttachmentFeedbackLoopEnabled(false);
}

// ISS_NATIVE_PASSES_CLIP (default on in submit mode, where the lane's registers are exact): draws
// with user clip planes recorded, the planes uploaded from each draw's registers (they were ~1/3
// of the draws the recorder left to the command processor).
static bool ClipMode() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_CLIP");
        return (!p || p[0] != '0') && np::SubmitMode();
    }();
    return enabled;
}

// ISS_NATIVE_PASSES_TEMPLATE_GATE (default on in submit mode): one learn request per template
// (render state and material) in flight, instead of one per render state key. A street of new
// materials in one state waited 250 ms per material (~570 draws a frame left to the command
// processor while moving, the drops in new areas); now each is learned at its first draw.
static bool TemplateGate() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_TEMPLATE_GATE");
        return (!p || p[0] != '0') && np::SubmitMode();
    }();
    return enabled;
}
static std::atomic<u64> g_learn_gated{};
static np::DrawResult LearnGate(u64 id) {
    if (!TemplateGate()) return np::NeedLearn; // the lanes dedup by render state key
    static std::mutex mutex;
    static std::unordered_map<u64, u64> posted_at;
    const u64 now = u64(std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count());
    std::scoped_lock lock{mutex};
    if (posted_at.size() > 262144) posted_at.clear();
    auto& at = posted_at[id];
    if (at && now - at < 250) {
        g_learn_gated.fetch_add(1, std::memory_order_relaxed);
        return np::NoRecord;
    }
    at = now;
    return np::NeedLearn;
}

static Rec& RecOf(np::Segment& s) {
    if (!s.recording) s.recording = new Rec;
    return *static_cast<Rec*>(s.recording);
}

// Begins the segment's secondary for the template's attachment formats (under the lane lock).
static bool Begin(np::Segment& s, Rec& r, const Template& t) {
    auto* lane = static_cast<LanePool*>(*s.lane_user);
    if (!lane) {
        lane = new LanePool;
        lane->pool = Check(g.instance->GetDevice().createCommandPool(vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = g.instance->GetGraphicsQueueFamilyIndex(),
        }));
        *s.lane_user = lane;
    }
    if (!r.cmd) {
        const vk::CommandBufferAllocateInfo info{
            .commandPool = lane->pool,
            .level = vk::CommandBufferLevel::eSecondary,
            .commandBufferCount = 1,
        };
        if (g.instance->GetDevice().allocateCommandBuffers(&info, &r.cmd) != vk::Result::eSuccess)
            return false;
    } else if (r.needs_reset) {
        r.cmd.reset();
    }
    const vk::CommandBufferInheritanceRenderingInfo rendering{
        .colorAttachmentCount = t.num_colors,
        .pColorAttachmentFormats = t.color_formats.data(),
        .depthAttachmentFormat = t.depth_format,
        .stencilAttachmentFormat = t.stencil_format,
        .rasterizationSamples = t.samples,
    };
    const vk::CommandBufferInheritanceInfo inheritance{.pNext = &rendering};
    r.cmd.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit |
                 vk::CommandBufferUsageFlagBits::eRenderPassContinue,
        .pInheritanceInfo = &inheritance,
    });
    r.begun = true;
    r.needs_reset = true;
    r.bb_segment = GpuCrashDiag::Blackbox() ? GpuCrashDiag::NewNativeSegment() : 0;
    r.bb_draws = 0;
    return true;
}

// A segment's recording made fresh for its first draw.
static void Reset(Rec& r) {
    // A secondary left recording (a segment given up at its first draw) is reset by Begin.
    if (r.begun) r.needs_reset = true;
    r.begun = false;
    r.current = nullptr;
    r.bound_pipeline = vk::Pipeline{};
    r.templates.clear();
    r.arena_reads.clear();
    r.arena_gen = 0;
    r.arena_gen_mixed = false;
    r.images.clear();
    r.reads_unified = false;
    r.workers.clear();
    r.keys.clear();
    r.dyn_used = 0;
    r.first_ds = ~0u;
    r.classes = 0;
}

static np::DrawResult Record(np::Segment& s, void* regs_ptr, u64 key, const u32* written,
                             const np::DrawArgs& d) {
    auto& stats = engine_stats;
    const auto miss = [&](Miss m, np::DrawResult result = np::NoRecord) {
        stats.miss[m].fetch_add(1, std::memory_order_relaxed);
        Shader::Info::num_ud_snapshots = 0;
        return result;
    };
    if (!g.ready.load(std::memory_order_acquire)) return miss(MissUnified);
    Worker* w = ThisWorker();
    if (!w) return miss(MissCommand);
    w->Reclaim();
    auto& regs = *static_cast<AmdGpu::Regs*>(regs_ptr);
    Rec& r = RecOf(s);
    if (r.generation != s.generation) {
        Reset(r);
        r.generation = s.generation;
    }
    if (!d.autod && !d.indirect) {
        regs.index_base_address.base_addr_lo = u32(d.index_address);
        regs.index_base_address.base_addr_hi = u32(d.index_address >> 32);
        regs.max_index_size = d.index_count;
    }
    // DRAW_INDEX_AUTO sets the count and the initiator only (as the command processor does); an
    // indirect draw keeps the lane's index buffer size (INDEX_BUFFER_SIZE).
    if (!d.indirect) regs.num_indices = d.index_count;
    regs.draw_initiator = d.initiator;
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Normal ||
        regs.primitive_type == AmdGpu::PrimitiveType::None ||
        (regs.reg_array[0xA003] & 0x78000000u) || regs.enable_primitive_restart ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable ||
        (regs.clipper_control.user_clip_plane_enable != 0 && !ClipMode()))
        return miss(MissState);
    stats.draws.fetch_add(1, std::memory_order_relaxed);

    {
        auto& learned = LearnedKeys();
        std::shared_lock lock{learned.mutex};
        if (learned.relearn.contains(key)) return miss(MissShape, np::NeedLearn);
    }
    // The template of the draw's render state and material.
    std::shared_ptr<const Template> t;
    static thread_local std::array<NativeRecorder::StageView, 4> views;
    {
        auto& tables = NativeMaterial::State();
        const GraphicsPipeline* shape = nullptr;
        {
            std::shared_lock lock{tables.mutex};
            const auto it = tables.shapes.find(key);
            if (it != tables.shapes.end()) shape = it->second;
        }
        if (!shape) return miss(MissShape, LearnGate(key));
        u32 num_views = 0;
        Shader::Info::num_ud_snapshots = 0;
        if (!NativeRecorder::InstallViews(*shape, regs, views, num_views)) return miss(MissViews);
        const u64 material = NativeMaterial::MaterialHash(*shape);
        {
            std::shared_lock lock{tables.mutex};
            const auto it = tables.templates.find(NativeMaterial::Mix(key, material));
            if (it != tables.templates.end()) t = it->second;
        }
        if (!t) return miss(MissTemplate, LearnGate(NativeMaterial::Mix(key, material)));
        NativeMaterial::Touch(*t);
        if (t->pipeline != shape && !NativeRecorder::InstallViews(*t->pipeline, regs, views, num_views))
            return miss(MissViews);
        // Its textures and samplers before their handles are pushed (the command processor
        // draws it and drops the stale template).
        if (!NativeMaterial::HandlesAlive(*g.texture_cache, *t)) return miss(MissViews);
    }
    const auto* pipeline = t->pipeline;
    if (!pipeline->IsReady() || (!pipeline->UsesPushDescriptors() && !NativeMaterial::SetsEnabled()))
        return miss(MissPipeline);
    // ISS_NATIVE_INDIRECT_CLASSES (default 3): the indexed indirect draws of these classes only
    // (1 opaque, 2 depth only, 4 blended). With the blended ones too the city lost its sunlight
    // and shadows in some frames (a3-indirect4, a3-cls7b 2026-10-18); 1 + 2 or 4 alone were right.
    // The blended ones are ~4% of them (0.19M of 4.36M).
    static const u32 indirect_classes = [] {
        const char* p = std::getenv("ISS_NATIVE_INDIRECT_CLASSES");
        return p ? u32(std::strtoul(p, nullptr, 0)) : 3u;
    }();
    if (d.indirect && (indirect_classes & DrawClass(regs)) == 0) return miss(MissState);
    // ISS_NATIVE_PASSES_CLASSES: a class left to the command processor.
    if (const u32 cls = DrawClass(regs);
        (r.classes |= cls, (AllowedClasses() & cls) == 0 || (cls == 4 && !BlendTargetAllowed(regs)))) {
        g_class_left[cls].fetch_add(1, std::memory_order_relaxed);
        return miss(MissState);
    }
    // ISS_NATIVE_PASSES_GUARD_DEPTH: every draw of a segment (empty ones too) in the depth/stencil
    // attachment state of its first.
    const u32 ds = DepthStencilClass(regs);
    if (GuardDepth() && r.begun && r.first_ds != ~0u && ds != r.first_ds) {
        if (g_depth_mix.fetch_add(1, std::memory_order_relaxed) < 20)
            std::fprintf(stderr, "NATIVE_PASSES_DEPTH_MIX db=%llx first=%x now=%x\n",
                         (unsigned long long)regs.depth_buffer.DepthAddress(), r.first_ds, ds);
        return miss(MissFormat, np::NewSegment);
    }
    if (d.indirect && regs.num_indices == 0) return miss(MissCommand);
    if (regs.num_indices == 0) {
        // Nothing the GPU would draw (the command processor skips it too).
        stats.empty.fetch_add(1, std::memory_order_relaxed);
        Shader::Info::num_ud_snapshots = 0;
        if (!r.begun) return miss(MissCommand);
        return np::Recorded;
    }

    if (!r.begun) {
        if (!Begin(s, r, *t)) return miss(MissCommand);
        r.first_ds = ds;
        r.first_flat = FlatCheck() ? RecordedFlatHash(*pipeline) : 0;
    } else {
        const auto& first = *r.templates.front();
        if (t->num_colors != first.num_colors || t->color_formats != first.color_formats ||
            t->depth_format != first.depth_format || t->stencil_format != first.stencil_format ||
            t->samples != first.samples)
            return miss(MissFormat, np::NewSegment);
    }
    const u64 seq = w->Note(s, r);
    const auto cmd = r.cmd;
    const VideoCore::Buffer* unified = g.unified.load(std::memory_order_acquire);
    const VAddr unified_base = g.unified_base.load(std::memory_order_relaxed);
    auto& cache = *g.buffer_cache;
    const u64 alignment = g.instance->StorageMinAlignment();
    // Resolves a guest range the draw reads: the imported heap or a published arena.
    const auto resolve = [&](VAddr address, u64 size, vk::Buffer& handle, u64& offset) {
        if (unified && cache.IsUnified(address, size)) {
            handle = unified->Handle();
            offset = address - unified_base;
            if (!r.reads_unified) {
                r.unified_lo = address;
                r.unified_hi = address + size;
            }
            r.unified_lo = std::min(r.unified_lo, address);
            r.unified_hi = std::max(r.unified_hi, VAddr(address + size));
            r.reads_unified = true;
            return true;
        }
        // ISS_NATIVE_PASSES_NO_ARENA=1 (diagnosis): only draws whose buffers are all in the
        // imported heap are recorded.
        static const bool no_arena = [] {
            const char* p = std::getenv("ISS_NATIVE_PASSES_NO_ARENA");
            return p && p[0] == '1';
        }();
        if (no_arena) return false;
        // Read first: an arena created or migrated after it changes the generation.
        const u64 arena_gen = cache.ArenaBindingGeneration();
        const auto arena = cache.PeekPublishedArena(address, u32(size));
        if (!arena) return false;
        handle = arena.arena->Handle();
        offset = arena.offset;
        for (const auto& a : r.arena_reads)
            if (a.address == address && a.size == size) return true;
        if (r.arena_reads.empty()) r.arena_gen = arena_gen;
        else if (r.arena_gen != arena_gen) r.arena_gen_mixed = true;
        r.arena_reads.push_back({address, size, handle, offset, arena.arena});
        return true;
    };
    // Anything failing below leaves the bindings unknown for the next draw.
    const auto fail = [&](Miss m) {
        r.current = nullptr;
        r.bound_pipeline = vk::Pipeline{};
        return miss(m);
    };

    if (!DynamicMode()) {
        // The template's dynamic state (the render state key includes its registers).
        if (t.get() != r.current) {
            auto dynamic = *t->dynamic;
            dynamic.Invalidate();
            dynamic.Commit(*g.instance, CommandRecorder{cmd});
            r.current = t.get();
        }
    } else {
    // The dynamic state from this draw's registers. A thread's scratch state follows one
    // segment at a time: another segment's secondary starts from nothing.
        static thread_local std::unique_ptr<DynamicState> scratch;
        static thread_local const np::Segment* scratch_segment = nullptr;
        static thread_local u32 scratch_generation = 0;
        if (!scratch) scratch = std::make_unique<DynamicState>(*t->dynamic);
        if (scratch_segment != &s || scratch_generation != s.generation || !r.current) {
            scratch->Invalidate();
            scratch_segment = &s;
            scratch_generation = s.generation;
        }
        DrawDynamicState(regs, *g.instance, *pipeline, *scratch);
        r.dyn_used |= DynAlways;
        if (regs.depth_control.depth_bounds_enable) r.dyn_used |= DynBounds;
        if (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())
            r.dyn_used |= DynStencil;
        if (regs.polygon_control.NeedsBias()) r.dyn_used |= DynBias;
        scratch->Commit(*g.instance, CommandRecorder{cmd});
        r.current = t.get();
    }
    if (pipeline->Handle() != r.bound_pipeline) {
        r.bound_pipeline = pipeline->Handle();
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, r.bound_pipeline);
    }
    bool listed = false;
    for (const auto& u : r.templates) listed |= u.get() == t.get();
    if (!listed) r.templates.push_back(t);
    if (r.keys.empty()) r.lane_assumed = AssumedHash(regs);
    bool key_listed = false;
    for (const auto& k : r.keys) key_listed |= k.first == key;
    if (!key_listed) {
        auto& entry = r.keys.emplace_back();
        entry.first = key;
        std::memcpy(entry.second.data(), written, np::MaskWords * 4);
    }

    std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> infos;
    Pipeline::DescriptorWrites writes;
    Shader::PushData push = t->push;
    // The viewport transform the shaders read (as the command processor's push data): this
    // draw's, not the one the template was learned at.
    push.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    const auto stages = pipeline->GetStages();
    for (u32 k = 0; k < t->buffers.size(); ++k) {
        const auto& slot = t->buffers[k];
        const auto& stage = *stages[slot.stage];
        auto& info = infos[k];
        if (slot.special == NativeMaterial::Flat) {
            const auto flat = stage.FlatUserData();
            const u64 bytes = flat.size() * sizeof(u32);
            u64 offset;
            u8* dst = w->Upload(bytes, alignment, offset);
            if (!dst) return fail(MissRing);
            std::memcpy(dst, flat.data(), bytes);
            info = vk::DescriptorBufferInfo{w->ring, offset, bytes};
            continue;
        }
        if (slot.special == NativeMaterial::ClipPlanes) {
            if (regs.clipper_control.user_clip_plane_enable == 0) {
                info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                continue;
            }
            // The planes as the command processor uploads them (vk_rasterizer BindBuffers).
            std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
            for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                const auto& plane = regs.clip_user_data[i];
                planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
            }
            u64 offset;
            u8* dst = w->Upload(sizeof(planes), alignment, offset);
            if (!dst) return fail(MissRing);
            std::memcpy(dst, planes.data(), sizeof(planes));
            info = vk::DescriptorBufferInfo{w->ring, offset, sizeof(planes)};
            continue;
        }
        const auto sharp = stage.buffers[slot.desc].GetSharp(stage);
        if (!sharp.base_address || !sharp.GetSize()) {
            info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
            continue;
        }
        vk::Buffer handle;
        u64 offset;
        if (!resolve(sharp.base_address, sharp.GetSize(), handle, offset)) return fail(MissBuffer);
        const u64 aligned = Common::AlignDown(offset, alignment);
        push.buf_offsets[slot.binding] = u8(offset - aligned);
        info = vk::DescriptorBufferInfo{handle, aligned, sharp.GetSize() + (offset - aligned)};
    }
    NativeMaterial::BuildWrites(*t, infos.data(), writes);
    if (pipeline->UsesPushDescriptors()) {
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, pipeline->GetLayout(), 0, writes);
    } else if (!writes.empty()) {
        const auto set = w->AllocateSet(pipeline->GetDescriptorSetLayout(), seq);
        if (!set) return fail(MissPool);
        for (auto& write : writes) write.dstSet = set;
        w->device.updateDescriptorSets(u32(writes.size()), writes.data(), 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->GetLayout(), 0, 1, &set,
                               0, nullptr);
    }
    cmd.pushConstants(pipeline->GetLayout(), AllGraphicsStageBits, 0u, sizeof(push), &push);

    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    cmd.setVertexInputEXT(bindings, attributes);
    if (!guest_buffers.empty()) {
        VertexInputs<vk::Buffer> handles;
        VertexInputs<vk::DeviceSize> offsets;
        for (const auto& buffer : guest_buffers) {
            vk::Buffer handle{};
            u64 offset = 0;
            if (buffer.base_address && buffer.GetSize() &&
                !resolve(buffer.base_address, buffer.GetSize(), handle, offset))
                return fail(MissBuffer);
            handles.push_back(handle);
            offsets.push_back(offset);
        }
        cmd.bindVertexBuffers(0, u32(handles.size()), handles.data(), offsets.data());
    }
    if (!d.autod) {
        const bool index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = index16 ? 2 : 4;
        vk::Buffer index_handle;
        u64 index_offset;
        if (!resolve(regs.index_base_address.Address<VAddr>(), u64(regs.num_indices) * index_size,
                     index_handle, index_offset))
            return fail(MissBuffer);
        cmd.bindIndexBuffer(index_handle, index_offset,
                            index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto [vertex_offset, instance_offset] =
        GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
    if (r.bb_segment) {
        // GPU black box: where the GPU is inside the segment if it hangs.
        GpuCrashDiag::Entry entry{
            .kind = GpuCrashDiag::Kind::NativeDraw,
            .count = regs.num_indices,
            .instances = regs.num_instances.NumInstances(),
            .address = regs.index_base_address.Address<VAddr>(),
        };
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) continue;
            if (stage->sw_stage == Shader::SwStage::Fragment) entry.hashes[1] = stage->pgm_hash;
            else if (stage->sw_stage == Shader::SwStage::Vertex) entry.hashes[0] = stage->pgm_hash;
        }
        GpuCrashDiag::MarkNativeDraw(static_cast<VkCommandBuffer>(cmd), r.bb_segment, r.bb_draws++,
                                     entry, BlackboxContext(regs, 2));
    }
    if (d.indirect) {
        // ISS_NATIVE_INDIRECT: the arguments the GPU wrote (the segment reads them: the command
        // processor orders it after their writer, NativePassExecute).
        vk::Buffer args_handle;
        u64 args_offset;
        if (!resolve(d.args_address, sizeof(VkDrawIndexedIndirectCommand), args_handle, args_offset))
            return fail(MissBuffer);
        cmd.drawIndexedIndirect(args_handle, args_offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    } else if (d.autod) {
        // As the command processor's Draw(false): no index buffer, vertices from vertex_offset.
        cmd.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                 instance_offset);
        g_auto_draws.fetch_add(1, std::memory_order_relaxed);
    } else {
        cmd.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0, s32(vertex_offset),
                        instance_offset);
    }
    Shader::Info::num_ud_snapshots = 0;
    return np::Recorded;
}

static bool Close(np::Segment& s) {
    auto* r = static_cast<Rec*>(s.recording);
    if (!r || !r->begun || r->templates.empty()) return false;
    r->cmd.end();
    r->begun = false;
    // Its buffer reads, sorted and merged (ranges of one buffer up to 64 KB apart).
    auto& reads = r->reads;
    reads.clear();
    for (const auto& a : r->arena_reads) reads.push_back({a.arena, a.offset, a.size});
    if (r->reads_unified)
        if (const auto* unified = g.unified.load(std::memory_order_acquire))
            reads.push_back({unified, r->unified_lo - g.unified_base.load(std::memory_order_relaxed),
                             r->unified_hi - r->unified_lo});
    std::sort(reads.begin(), reads.end(), [](const Rec::Read& a, const Rec::Read& b) {
        return a.buffer != b.buffer ? a.buffer < b.buffer : a.offset < b.offset;
    });
    size_t merged = 0;
    for (size_t i = 0; i < reads.size(); ++i) {
        if (merged && reads[merged - 1].buffer == reads[i].buffer &&
            reads[i].offset <= reads[merged - 1].offset + reads[merged - 1].size + (64u << 10)) {
            auto& last = reads[merged - 1];
            last.size = std::max(last.offset + last.size, reads[i].offset + reads[i].size) - last.offset;
        } else {
            reads[merged++] = reads[i];
        }
    }
    reads.resize(merged);
    auto& images = r->images;
    images.clear();
    for (const auto& t : r->templates)
        for (const auto& cap : t->images) {
            if (!cap.id) continue;
            bool listed = false;
            for (const auto* other : images)
                if (other->id == cap.id && other->layout == cap.layout &&
                    other->range == cap.range) {
                    listed = true;
                    break;
                }
            if (!listed) images.push_back(&cap);
        }
    return true;
}

static void Abandon(np::Segment&) {
    // Nothing to release now: the ring space and sets go back as the segment counts retired,
    // the secondary is reset when the segment is reused.
}

static bool Retired(const np::Segment& s) {
    const u32 state = s.state.load(std::memory_order_acquire);
    if (state == np::Abandoned || state == np::Free) return true;
    return state == np::Done && s.executed_tick.load(std::memory_order_relaxed) <=
                                    g.scheduler->GetWorkSemaphore()->KnownGpuTick();
}

static u64 Key(void* regs, const u32* written) {
    return MaskedKey(*static_cast<const AmdGpu::Regs*>(regs), written);
}

static void* NewRegs(u32*& words) {
    auto* regs = new AmdGpu::Regs{};
    words = regs->reg_array.data();
    return regs;
}

static void ClearState(void* regs) {
    static_cast<AmdGpu::Regs*>(regs)->SetDefaults();
}

static bool IsUnlearnable(u64 key) {
    auto& u = Rejected();
    std::shared_lock lock{u.mutex};
    return u.keys.contains(key);
}

struct CpStats {
    u64 executed{}, draws{}, report{}, learned{}, learn_mismatch{}, learn_failed{}, relearn{},
        inherited_merged{};
    std::array<u64, 13> reject{};
    // ISS_NATIVE_INDIRECT: segments starting at an indexed indirect draw.
    u64 indirect_executed{}, indirect_draws{}, synced_reads{};
    u64 flat_checked{}, flat_mismatch{};
    std::array<u64, 13> indirect_reject{};
};
static CpStats cp_stats;
static std::array<u64, 0x400> dynamic_rejects{}; // by context register
enum Reject : u32 { RejectContext, RejectUser, RejectInherited, RejectPipeline, RejectSamplers,
                    RejectArena, RejectTargets, RejectTextures, RejectFormats, RejectEnv,
                    RejectInheritedState, RejectDynamic, RejectAssumed };
constexpr const char* reject_names[13] = {"context", "user_data", "inherited", "pipeline",
                                          "samplers", "arena", "targets", "textures", "formats",
                                          "env", "inherited_state", "dynamic", "assumed"};

static void Report() {
    std::fprintf(stderr,
                 "NATIVE_PASSES_GUARDS template_gate=%d learn_gated=%llu auto_draws=%llu depth_mix=%llu "
                 "(guard %s) classes=%u left[opaque=%llu depth=%llu blend=%llu]\n",
                 int(TemplateGate()), (unsigned long long)g_learn_gated.load(std::memory_order_relaxed),
                 (unsigned long long)g_auto_draws.load(std::memory_order_relaxed),
                 (unsigned long long)g_depth_mix.load(std::memory_order_relaxed),
                 GuardDepth() ? "on" : "off", AllowedClasses(),
                 (unsigned long long)g_class_left[1].load(), (unsigned long long)g_class_left[2].load(),
                 (unsigned long long)g_class_left[4].load());
    {
        std::string text;
        for (u32 c = 1; c < 8; ++c) {
            if (!g_alias.segments[c].load()) continue;
            char item[160];
            std::snprintf(item, sizeof(item), " c%u[segs=%llu alias_color=%llu alias_depth=%llu same=%llu]", c,
                          (unsigned long long)g_alias.segments[c].load(),
                          (unsigned long long)g_alias.alias_color[c].load(),
                          (unsigned long long)g_alias.alias_depth[c].load(),
                          (unsigned long long)g_alias.same_image[c].load());
            text += item;
        }
        {
            char item[160];
            std::snprintf(item, sizeof(item), " depth_tex[c1=%llu c2=%llu c4=%llu c5=%llu] depth_left=%llu",
                          (unsigned long long)g_alias.depth_textures[1].load(),
                          (unsigned long long)g_alias.depth_textures[2].load(),
                          (unsigned long long)g_alias.depth_textures[4].load(),
                          (unsigned long long)g_alias.depth_textures[5].load(),
                          (unsigned long long)g_alias.depth_left.load());
            text += item;
        }
        std::fprintf(stderr,
                     "NATIVE_PASSES_ALIAS exec_classes=%u exec_left=%llu feedback_left=%llu same_vkimage=%llu layout_drift=%llu (guard %s)%s\n",
                     ExecClasses(), (unsigned long long)g_alias.exec_left.load(),
                     (unsigned long long)g_alias.feedback_left.load(),
                     (unsigned long long)g_alias.same_vkimage.load(),
                     (unsigned long long)g_alias.layout_drift.load(), GuardFeedback() ? "on" : "off",
                     text.c_str());
    }
    {
        std::vector<std::pair<u64, u32>> top;
        for (u32 w = 0; w < 0x400; ++w)
            if (dynamic_rejects[w]) top.emplace_back(dynamic_rejects[w], 0xA000 + w);
        std::sort(top.rbegin(), top.rend());
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 16); ++i) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", top[i].second, (unsigned long long)top[i].first);
            text += item;
        }
        if (!top.empty()) std::fprintf(stderr, "NATIVE_PASSES_DYNAMIC regs[%s ]\n", text.c_str());
    }
    if (const auto& diag = InhDiag(); diag.samples) {
        std::vector<std::pair<u64, u32>> top;
        for (u32 w = 0; w < diag.differ.size(); ++w)
            if (diag.differ[w]) top.emplace_back(diag.differ[w], w < 0x400 ? 0xA000 + w : w < 0x480 ? 0x2C00 + w - 0x400 : 0xC242);
        std::sort(top.rbegin(), top.rend());
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 32); ++i) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", top[i].second, (unsigned long long)top[i].first);
            text += item;
        }
        std::fprintf(stderr, "NATIVE_PASSES_INHERITED samples=%llu regs[%s ]\n",
                     (unsigned long long)diag.samples, text.c_str());
    }
    auto& e = engine_stats;
    std::string misses;
    for (u32 m = 0; m < MissCount; ++m) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", miss_names[m],
                      (unsigned long long)e.miss[m].load(std::memory_order_relaxed));
        misses += item;
    }
    std::string rejects, indirect_rejects;
    for (u32 r = 0; r < 13; ++r) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", reject_names[r],
                      (unsigned long long)cp_stats.reject[r]);
        rejects += item;
        std::snprintf(item, sizeof(item), " %s=%llu", reject_names[r],
                      (unsigned long long)cp_stats.indirect_reject[r]);
        indirect_rejects += item;
    }
    if (FlatCheck())
        std::fprintf(stderr, "NATIVE_FLAT_CHECK checked=%llu mismatch=%llu mode=%d\n",
                     (unsigned long long)cp_stats.flat_checked,
                     (unsigned long long)cp_stats.flat_mismatch, FlatCheck());
    if (np::IndirectEnabled())
        std::fprintf(stderr, "NATIVE_INDIRECT_CP executed=%llu synced_reads=%llu draws=%llu reject[%s ]\n",
                     (unsigned long long)cp_stats.indirect_executed,
                     (unsigned long long)cp_stats.synced_reads,
                     (unsigned long long)cp_stats.indirect_draws, indirect_rejects.c_str());
    std::fprintf(stderr,
                 "NATIVE_PASSES_VK engine draws=%llu empty=%llu miss[%s ] | cp executed=%llu "
                 "draws=%llu learned=%llu learn_mismatch=%llu learn_failed=%llu relearn=%llu merged=%llu "
                 "reject[%s ]\n",
                 (unsigned long long)e.draws.load(), (unsigned long long)e.empty.load(),
                 misses.c_str(), (unsigned long long)cp_stats.executed,
                 (unsigned long long)cp_stats.draws, (unsigned long long)cp_stats.learned,
                 (unsigned long long)cp_stats.learn_mismatch,
                 (unsigned long long)cp_stats.learn_failed,
                 (unsigned long long)cp_stats.relearn,
                 (unsigned long long)cp_stats.inherited_merged, rejects.c_str());
    np::Report();
    NativeMaterialReport();
}
} // namespace NativePasses

namespace NativeCompute {
static void Install(iss::native_passes::Callbacks& callbacks);
}

void Rasterizer::NativePassesSetup() {
    using namespace NativePasses;
    if (!g.unified.load(std::memory_order_relaxed) && buffer_cache.UnifiedBuffer()) {
        g.unified_base.store(buffer_cache.UnifiedBase(), std::memory_order_relaxed);
        g.unified.store(buffer_cache.UnifiedBuffer(), std::memory_order_release);
    }
    static bool done = false;
    if (done) return;
    done = true;
    if (!np::Enabled() || !NativeMaterial::Enabled()) return;
    g.instance = &instance;
    g.scheduler = &scheduler;
    g.buffer_cache = &buffer_cache;
    g.pipeline_cache = &pipeline_cache;
    g.texture_cache = &texture_cache;
    Shader::Info::ud_snapshots_used.store(true, std::memory_order_relaxed);
    g.ready.store(true, std::memory_order_release);
    const auto& regs = DrawRegs();
    const auto* base = regs.reg_array.data();
    np::Callbacks callbacks{};
    callbacks.index_type_word = u32(reinterpret_cast<const u32*>(&regs.index_buffer_type) - base);
    callbacks.instances_word = u32(reinterpret_cast<const u32*>(&regs.num_instances) - base);
    callbacks.index_base_word = u32(reinterpret_cast<const u32*>(&regs.index_base_address) - base);
    callbacks.max_index_word = u32(reinterpret_cast<const u32*>(&regs.max_index_size) - base);
    callbacks.num_indices_word = u32(reinterpret_cast<const u32*>(&regs.num_indices) - base);
    callbacks.new_regs = &NewRegs;
    callbacks.clear_state = &ClearState;
    callbacks.record = &Record;
    callbacks.key = &Key;
    callbacks.close = &Close;
    callbacks.abandon = &Abandon;
    callbacks.retired = &Retired;
    callbacks.unlearnable = &IsUnlearnable;
    callbacks.report = &Report;
    NativeCompute::Install(callbacks);
    np::SetCallbacks(callbacks);
    std::fprintf(stderr, "NATIVE_PASSES on (index_type word 0x%x, instances word 0x%x)\n",
                 callbacks.index_type_word, callbacks.instances_word);
}

bool Rasterizer::NativePassExecute(iss::native_passes::Segment& s,
                                   const std::function<void(const u32*)>& begin_draw,
                                   const std::function<void()>& end_draw, u64& tick) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::RecorderExecute};
    using namespace NativePasses;
    auto& c = cp_stats;
    if ((++c.report & 1023) == 1) Report();
    const bool indirect_first = ((s.first_draw[0] >> 8) & 0xFF) == 0x25;
    const auto reject = [&](Reject why) {
        ++c.reject[why];
        if (indirect_first) ++c.indirect_reject[why];
        return false;
    };
    if (cp_split_enabled || draw_pipe_state || OnDrawWorker() || DrawQueue() != 0 ||
        iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
        guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        occlusion.HasOpen() || buffer_cache.HasComputeQueue() ||
        !instance.IsVertexInputDynamicState() || iss::hazard_probe::Enabled())
        return reject(RejectEnv);
    auto* r = static_cast<Rec*>(s.recording);
    if (!r || r->templates.empty()) return reject(RejectPipeline);
    iss::cp_path::Switch(iss::cp_path::RecChain);
    auto& regs = DrawRegs();
    // The first draw's registers the lane wrote are these, and every key the segment used was
    // learned with the words it inherits here.
    if (r->keys.empty()) return reject(RejectContext);
    // ISS_NATIVE_PASSES_FAST_KEYS (default on, =0 off): the state words once for every key below.
    // Validated with ISS_NATIVE_PASSES_FAST_KEYS_CHECK=1: 19.9 M checks, 0 differences (fk-check,
    // city, 2026-10-19).
    static const bool fast_keys = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_FAST_KEYS");
        return !(p && p[0] == '0');
    }();
    static thread_local StateWordArray state_words;
    if (fast_keys) StateWords(regs, state_words);
    // ISS_NATIVE_PASSES_FAST_KEYS_CHECK=1 (diagnosis): both ways, the differences counted.
    static const bool fast_check = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_FAST_KEYS_CHECK");
        return p && p[0] == '1';
    }();
    if (fast_keys && fast_check) [[unlikely]] {
        static u64 checks = 0, diffs = 0;
        for (const auto& [key, mask] : r->keys) {
            checks += 2;
            diffs += MaskedKeyWords(state_words, mask.data()) != MaskedKey(regs, mask.data());
            diffs += InheritedHashWords(state_words, mask.data()) != InheritedHash(regs, mask.data());
        }
        if ((checks & 0xFFFFF) < 2 * r->keys.size())
            std::fprintf(stderr, "FAST_KEYS_CHECK checks=%llu diffs=%llu\n",
                         (unsigned long long)checks, (unsigned long long)diffs);
    }
    {
        u64 first_key = fast_keys ? MaskedKeyWords(state_words, r->keys.front().second.data())
                                  : MaskedKey(regs, r->keys.front().second.data());
        if (((s.first_draw[0] >> 8) & 0xFF) == 0x25)
            first_key = np::IndirectKey(first_key, s.first_draw[2], s.first_draw[3]);
        if (first_key != s.first_context) return reject(RejectContext);
    }
    // The words the lane assumed are this command buffer's here (its draws used them).
    if (AssumedHash(regs) != r->lane_assumed) return reject(RejectAssumed);
    {
        auto& learned = LearnedKeys();
        const std::array<u32, np::MaskWords>* last_mask = nullptr;
        u64 inherited = 0;
        for (const auto& [key, mask] : r->keys) {
            if (!last_mask || *last_mask != mask) {
                inherited = fast_keys ? InheritedHashWords(state_words, mask.data())
                                      : InheritedHash(regs, mask.data());
                last_mask = &mask;
            }
            const auto it = learned.inherited.find(key);
            if (it == learned.inherited.end() ||
                std::find(it->second.begin(), it->second.end(), inherited) == it->second.end()) {
                auto& diag = InhDiag();
                if (const auto d = diag.words.find(key); d != diag.words.end()) {
                    const auto now = InheritedWords(regs, mask.data());
                    for (u32 w = 0; w < now.size(); ++w) diag.differ[w] += now[w] != d->second[w];
                    ++diag.samples;
                }
                std::unique_lock lock{learned.mutex};
                learned.relearn.insert(key);
                ++c.relearn;
                return reject(RejectInheritedState);
            }
        }
    }
    // The user data the first draw read: the words the lane wrote (the engine writes every table
    // pointer a draw reads, eboot 0x14AD0; the others are leftovers of other programs), all of
    // them with ISS_NATIVE_PASSES_STRICT_UD=1.
    static const bool strict_ud = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_STRICT_UD");
        return p && p[0] == '1';
    }();
    const auto& first_mask = r->keys.front().second;
    for (u32 k = 0; k < np::UserWords; ++k) {
        const int vs = np::TrackedIndex(np::VsUser0 + k), ps = np::TrackedIndex(np::PsUser0 + k);
        const bool vs_written = (first_mask[vs >> 5] >> (vs & 31)) & 1;
        const bool ps_written = (first_mask[ps >> 5] >> (ps & 31)) & 1;
        if (((strict_ud || vs_written) && regs.reg_array[np::VsUser0 + k] != s.first_user[k]) ||
            ((strict_ud || ps_written) &&
             regs.reg_array[np::PsUser0 + k] != s.first_user[np::UserWords + k]))
            return reject(RejectUser);
    }
    for (const auto& [reg, value] : s.dyn_check)
        if (DynamicMode() && (r->dyn_used & DynGroupOf(reg)) && regs.reg_array[reg] != value) {
            ++dynamic_rejects[reg - 0xA000];
            return reject(RejectDynamic);
        }
    if (regs.index_buffer_type.raw != s.first_index_type ||
        regs.num_instances.num_instances != s.first_instances)
        return reject(RejectInherited);
    for (const auto& t : r->templates) {
        if (!t->pipeline->IsReady()) return reject(RejectPipeline);
        if (t->has_samplers && texture_cache.SamplersGeneration() != t->samplers_generation)
            return reject(RejectSamplers);
    }
    if (FlatCheck()) {
        // The first draw's flattened user data as the full path reads it now (this thread has no
        // recording snapshots: the stages' own buffers, from the command processor's registers).
        const auto& first_pipeline = *r->templates.front()->pipeline;
        u64 h = 0xcbf29ce484222325ull;
        for (const auto* stage : first_pipeline.GetStages()) {
            if (!stage) continue;
            auto& info = const_cast<Shader::Info&>(*stage);
            info.RefreshFlatBuf();
            h = HashWords(h, info.flattened_ud_buf);
        }
        ++c.flat_checked;
        if (h != r->first_flat) {
            ++c.flat_mismatch;
            if (FlatCheck() >= 2) return reject(RejectUser);
        }
    }
    // The arenas are looked up again (under the residency lock the recording threads contend
    // for: ~8% of the command processor) only when one was created or migrated since.
    const bool arenas_same =
        !r->arena_gen_mixed && buffer_cache.ArenaBindingGeneration() == r->arena_gen;
    // ISS_NATIVE_PASSES_SYNC_READS (default on, =0 off): a read the CPU wrote since is uploaded
    // where the segment reads it (the indirect draws' arguments the engine writes every frame:
    // 51 682 of their segments rejected in the city, a3-indirect3 2026-10-18).
    static const bool sync_reads = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_SYNC_READS");
        return !p || p[0] != '0';
    }();
    for (const auto& a : r->arena_reads) {
        if (!buffer_cache.IsRegionCpuClean(a.address, a.size)) {
            if (!sync_reads) return reject(RejectArena);
            const auto [buffer, offset] = buffer_cache.SyncReadInPlace(a.address, u32(a.size));
            if (buffer != a.arena || offset != a.offset) return reject(RejectArena);
            ++c.synced_reads;
            continue;
        }
        if (arenas_same) continue;
        const auto arena = buffer_cache.PeekPublishedArena(a.address, u32(a.size));
        if (!arena || arena.arena->Handle() != a.handle || arena.offset != a.offset ||
            arena.arena != a.arena)
            return reject(RejectArena);
    }

    // As Draw does before each draw.
    iss::cp_path::Switch(iss::cp_path::RecTargets);
    DrainDrawPipe(5);
    SubmitFeedbackBatch(scheduler);
    PeriodicSubmit(scheduler);
    scheduler.PopPendingOperations();

    // The render targets, as the first template's pipeline sets them up for the first draw.
    const auto& first = *r->templates.front();
    const auto* pipeline = first.pipeline;
    ResetBindings(false);
    PrepareRenderState(pipeline);
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb)
        if (cb_descs[cb].first &&
            texture_cache.IsMetaCleared(regs.color_buffers[cb].CmaskAddress(),
                                        cb_descs[cb].second.view_info.range.base.layer)) {
            ResetBindings(false);
            return reject(RejectTargets);
        }
    if (db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                                     db_desc.second.view_info.range.base.layer)) {
        ResetBindings(false);
        return reject(RejectTargets);
    }
    // Textures checked after the targets are marked (a texture that is a target is feedback).
    for (const auto& t : r->templates) {
        bool stale = false;
        if (!NativeMaterial::TexturesValid(texture_cache, *t, stale)) {
            if (stale) NativeMaterialDrop(t->key);
            ResetBindings(false);
            return reject(RejectTextures);
        }
    }
    {
        // Survey: a texture of the segment that is another image over an attachment's memory
        // (aliasing), or the attachment itself.
        const u32 cls = r->classes & 7;
        g_alias.segments[cls].fetch_add(1, std::memory_order_relaxed);
        const auto overlaps = [&](VideoCore::ImageId target, VideoCore::ImageId texture) {
            const auto& a = texture_cache.GetImage(target).info;
            const auto& b = texture_cache.GetImage(texture).info;
            return a.guest_address < b.guest_address + b.guest_size &&
                   b.guest_address < a.guest_address + a.guest_size;
        };
        bool alias_color = false, alias_depth = false, same = false;
        // The Vulkan images the attachments are (two cache images can share one backing).
        boost::container::static_vector<vk::Image, 9> attached;
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb)
            if (cb_descs[cb].first) attached.push_back(texture_cache.GetImage(cb_descs[cb].first).GetImage());
        if (db_desc.first) attached.push_back(texture_cache.GetImage(db_desc.first).GetImage());
        for (const auto* cap : r->images) {
            if (std::find(attached.begin(), attached.end(),
                          texture_cache.GetImage(cap->id).GetImage()) != attached.end()) {
                if (!same) g_alias.same_vkimage.fetch_add(1, std::memory_order_relaxed);
                same = true;
            }
            for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
                const auto target = cb_descs[cb].first;
                if (!target) continue;
                if (target == cap->id) same = true;
                else if (overlaps(target, cap->id)) alias_color = true;
            }
            if (const auto target = db_desc.first) {
                if (target == cap->id) same = true;
                else if (overlaps(target, cap->id)) alias_depth = true;
            }
        }
        if (alias_color) g_alias.alias_color[cls].fetch_add(1, std::memory_order_relaxed);
        if (alias_depth) g_alias.alias_depth[cls].fetch_add(1, std::memory_order_relaxed);
        if (same) g_alias.same_image[cls].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u32> indirect_logged{0};
        if ((alias_color || alias_depth || same) &&
            (indirect_first ? indirect_logged.fetch_add(1) < 20 : g_alias.logged.fetch_add(1) < 40)) {
            if (indirect_first) std::fprintf(stderr, "NATIVE_INDIRECT_FEEDBACK ");
            std::string text;
            for (const auto& t : r->templates)
                for (const auto& cap : t->images)
                    if (cap.id) {
                        const auto& info = texture_cache.GetImage(cap.id).info;
                        char item[96];
                        std::snprintf(item, sizeof(item), " tex%u@%llx(%ux%u fmt %d)", cap.id.index,
                                      (unsigned long long)info.guest_address, info.size.width,
                                      info.size.height, int(info.pixel_format));
                        text += item;
                    }
            const auto db = db_desc.first;
            std::fprintf(stderr,
                         "NATIVE_PASSES_ALIAS_SEG classes=%u color=%d depth=%d same=%d db%u@%llx(%ux%u)%s\n",
                         cls, int(alias_color), int(alias_depth), int(same), db ? db.index : 0u,
                         db ? (unsigned long long)texture_cache.GetImage(db).info.guest_address : 0ull,
                         db ? texture_cache.GetImage(db).info.size.width : 0u,
                         db ? texture_cache.GetImage(db).info.size.height : 0u, text.c_str());
        }
        // ISS_NATIVE_PASSES_GUARD_FEEDBACK (default 1): a texture that is one of the segment's
        // attachments, or another image over an attachment's memory, is a feedback loop. The
        // command processor draws it with the feedback layout (attachment_feedback_loop); the
        // segment would transition the texture to its read layout after the attachments were set
        // up, leaving the attachment in the wrong layout while it is drawn into (on AMD the
        // compression metadata goes stale and a later draw stalls the GPU: the blended-segment
        // hangs of 2026-10-16/17). TexturesValid's generation fast path skips its is_target check,
        // so this runs on every execution. Only blended segments had them (survey 16:03).
        bool depth_texture = false;
        for (const auto* cap : r->images)
            if (texture_cache.GetImage(cap->id).info.props.is_depth) depth_texture = true;
        if (depth_texture) {
            g_alias.depth_textures[cls].fetch_add(1, std::memory_order_relaxed);
            if (NoDepthTextures()) {
                g_alias.depth_left.fetch_add(1, std::memory_order_relaxed);
                ResetBindings(false);
                return reject(RejectTargets);
            }
        }
        if ((alias_color || alias_depth || same) && GuardFeedback()) {
            g_alias.feedback_left.fetch_add(1, std::memory_order_relaxed);
            ResetBindings(false);
            return reject(RejectTargets);
        }
        // ISS_NATIVE_PASSES_EXEC_CLASSES: recorded for the survey, drawn by the command processor.
        if (r->classes & ~ExecClasses()) {
            g_alias.exec_left.fetch_add(1, std::memory_order_relaxed);
            ResetBindings(false);
            return reject(RejectTargets);
        }
    }
    const RenderState render = BeginRendering(pipeline);
    if (attachment_feedback_loop) {
        ResetBindings(false);
        return reject(RejectTargets);
    }
    bool formats_equal = render.num_color_attachments == first.num_colors;
    for (u32 cb = 0; formats_equal && cb < first.num_colors; ++cb) {
        const auto& [image_id, desc] = cb_descs[cb];
        const auto format = render.color_attachments[cb].image_view && image_id
                                ? texture_cache.FindRenderTarget(image_id, desc).info.format
                                : vk::Format::eUndefined;
        formats_equal = format == first.color_formats[cb];
    }
    if (formats_equal) {
        const auto& depth = render.depth_stencil_attachment;
        vk::Format depth_format{}, stencil_format{};
        if ((depth.has_depth || depth.has_stencil) && db_desc.first) {
            const auto format = texture_cache.FindDepthTarget(db_desc.first, db_desc.second).info.format;
            depth_format = depth.has_depth ? format : vk::Format::eUndefined;
            stencil_format = depth.has_stencil ? format : vk::Format::eUndefined;
        }
        formats_equal = depth_format == first.depth_format && stencil_format == first.stencil_format;
    }
    if (!formats_equal) {
        ResetBindings(false);
        return reject(RejectFormats);
    }

    // Accepted: textures to their read layouts, buffer reads ordered after their writers.
    iss::cp_path::Switch(iss::cp_path::RecBarriers);
    const u64 now = scheduler.CurrentTick();
    for (const auto* cap : r->images) {
        auto& image = texture_cache.GetImage(cap->id);
        needs_barrier |= runtime.Transit(&image, cap->layout, vk::PipelineStageFlagBits2::eAllCommands,
                                         vk::AccessFlagBits2::eShaderRead, cap->range);
        image.usage.texture = 1u;
        image.tick_accessed_last = now;
    }
    // The layouts the secondary was recorded with must be the images' layouts now: each texture
    // in the layout of its descriptor, each attachment in the layout the rendering begins with
    // (a texture transition may have moved one). Otherwise the command processor draws it.
    {
        bool drift = false;
        for (const auto* cap : r->images) {
            const auto& image = texture_cache.GetImage(cap->id);
            if (image.backing->subresource_states.empty() && image.backing->state.layout != cap->layout)
                drift = true;
        }
        for (u32 cb = 0; cb < render.num_color_attachments; ++cb)
            if (cb_descs[cb].first && render.color_attachments[cb].image_view &&
                texture_cache.GetImage(cb_descs[cb].first).backing->state.layout !=
                    render.color_attachments[cb].image_layout)
                drift = true;
        if (db_desc.first && render.depth_stencil_attachment.image_view &&
            texture_cache.GetImage(db_desc.first).backing->state.layout !=
                render.depth_stencil_attachment.image_layout)
            drift = true;
        if (drift) {
            if (g_alias.layout_drift.fetch_add(1, std::memory_order_relaxed) < 20)
                std::fprintf(stderr, "NATIVE_PASSES_LAYOUT_DRIFT segment with %zu draws\n",
                             s.draw_packets.size());
            if (needs_barrier) runtime.FlushBarriers();
            ResetBindings(false);
            return reject(RejectTargets);
        }
    }
    // Its buffer reads, merged when it closed (Close).
    const auto& reads = r->reads;
    for (const auto& rd : reads)
        needs_barrier |= Rt().IsBufferAccessed(rd.buffer, rd.offset, rd.size, false);
    if (needs_barrier) runtime.FlushBarriers();
    iss::cp_path::Switch(iss::cp_path::RecDraws);
    if (r->bb_segment) {
        // GPU black box: the segment as one command of the command processor's queue.
        static thread_local std::vector<VideoCore::ImageId> bb_images;
        bb_images.clear();
        for (const auto* cap : r->images) bb_images.push_back(cap->id);
        auto bb_ctx = BlackboxContext(regs, 1);
        BlackboxImages(bb_ctx, bb_images, texture_cache, cb_descs, db_desc);
        GpuCrashDiag::MarkBox(&scheduler, scheduler.Recorder(), scheduler.CurrentTick(),
                              GpuCrashDiag::Entry{
                                  .kind = GpuCrashDiag::Kind::NativeSegment,
                                  .count = u32(s.draw_packets.size()),
                                  .dims = {render.width, render.height, render.num_layers},
                                  .address = r->bb_segment,
                              },
                              bb_ctx);
    }
    scheduler.BeginSecondaryRendering(render);
    scheduler.Recorder().Record([cmd = r->cmd](vk::CommandBuffer primary) {
        primary.executeCommands(cmd);
    });
    scheduler.EndRendering();
    scheduler.GetDynamicState().Invalidate();
    fast_draw.valid = false;
    for (const u32* packet : s.draw_packets) {
        begin_draw(packet);
        DebugState.IncDrawCall();
        end_draw();
    }
    iss::cp_path::Switch(iss::cp_path::RecAccess);
    ResetBindings(false);
    for (const auto& rd : reads)
        Rt().AccessBuffer(rd.buffer, rd.offset, rd.size, vk::PipelineStageFlagBits2::eAllGraphics,
                          vk::AccessFlagBits2::eShaderRead);
    tick = scheduler.CurrentTick();
    ++c.executed;
    c.draws += s.draw_packets.size();
    if (indirect_first) {
        ++c.indirect_executed;
        c.indirect_draws += s.draw_packets.size();
    }
    return true;
}

// A learn request: the draw takes the full path and its template is kept under this key.
u64 Rasterizer::NativePassLearnKey(u64 engine_key) {
    using namespace NativePasses;
    if (native_pass_learn_mask.size() != np::MaskWords) return 0;
    u64 key = MaskedKey(DrawRegs(), native_pass_learn_mask.data());
    // ISS_NATIVE_INDIRECT: a learn request at an indexed indirect draw (its key mixes the packet's
    // base vertex and instance user data locations).
    if (native_pass_learn_indirect)
        key = np::IndirectKey(key, u32(native_pass_learn_indirect >> 16) & 0xFFFF,
                              u32(native_pass_learn_indirect) & 0xFFFF);
    if (key != engine_key) {
        ++cp_stats.learn_mismatch;
        return 0;
    }
    return key;
}

void Rasterizer::NativePassLearn(u64 key, const GraphicsPipeline* pipeline, const RenderState& render) {
    using namespace NativePasses;
    auto& u = Rejected();
    const auto give_up = [&] {
        ++cp_stats.learn_failed;
        if (++u.attempts[key] >= 4) {
            std::unique_lock lock{u.mutex};
            u.keys.insert(key);
        }
    };
    if (!NativeGbuffer::ShapeOk(*pipeline)) {
        give_up();
        return;
    }
    // The template this key and material had (another inherited state), to compare.
    std::shared_ptr<const Template> previous;
    auto& tables = NativeMaterial::State();
    u64 template_key = 0;
    if (const auto shape = tables.shapes.find(key); shape != tables.shapes.end()) {
        NativeMaterial::Refresh(*shape->second);
        template_key = NativeMaterial::Mix(key, NativeMaterial::MaterialHash(*shape->second));
        if (const auto it = tables.templates.find(template_key); it != tables.templates.end())
            previous = it->second;
    }
    const auto before = NativeMaterial::Counters().learned;
    // The recorder uploads the user clip planes of each draw (ClipMode).
    native_material_allow_clip = ClipMode();
    NativeMaterialLearn(key, pipeline, render);
    native_material_allow_clip = false;
    if (NativeMaterial::Counters().learned == before) {
        give_up();
        return;
    }
    ++cp_stats.learned;
    auto& learned = LearnedKeys();
    const u64 inherited = InheritedHash(DrawRegs(), native_pass_learn_mask.data());
    if ((key & 0x3F0) == 0x100 && InhDiag().words.size() < 16384)
        InhDiag().words.try_emplace(key, InheritedWords(DrawRegs(), native_pass_learn_mask.data()));
    auto& accepted = learned.inherited[key];
    std::shared_ptr<const Template> now;
    if (template_key)
        if (const auto it = tables.templates.find(template_key); it != tables.templates.end())
            now = it->second;
    // The dynamic state is compared unless segments record it from each draw's registers.
    const bool same = previous && now && previous->pipeline == now->pipeline &&
                      (DynamicMode() || previous->dynamic == now->dynamic) &&
                      previous->num_colors == now->num_colors &&
                      previous->color_formats == now->color_formats &&
                      previous->depth_format == now->depth_format;
    if (same) {
        ++cp_stats.inherited_merged;
        if (std::find(accepted.begin(), accepted.end(), inherited) == accepted.end()) {
            if (accepted.size() >= 32) accepted.erase(accepted.begin());
            accepted.push_back(inherited);
        }
    } else {
        accepted.assign(1, inherited);
    }
    std::unique_lock lock{learned.mutex};
    learned.relearn.erase(key);
}
