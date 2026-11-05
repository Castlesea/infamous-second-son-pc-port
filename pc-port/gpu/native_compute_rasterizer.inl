// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_passes_rasterizer.inl.
//
// ISS_NATIVE_COMPUTE (default on with ISS_NATIVE_PASSES_SUBMIT=1; =0 turns it off): the engine's
// graphics queue dispatches recorded into Vulkan at the submission, as native_passes.h does for
// draws.
//  - Pool threads (native_passes.cpp RecordDispatch): consecutive DISPATCH_DIRECT packets with
//    only compute register writes between them form a compute segment (at most 64), prepared
//    from the exact compute registers (ComputeProgram) and the dispatch's compute template: the
//    pipeline, the descriptors with resolved buffers, the push data (Replay). The command
//    processor records them into its own command buffer (not from a secondary: see Replay). Buffers resolve without the buffer cache (the imported heap by address,
//    the others in their published arena); a dispatch that reads or writes what an earlier one
//    of the segment wrote (or writes what it read) gets a compute memory barrier first.
//  - Command processor: a segment is executed at its first dispatch packet when its templates
//    are current (pipeline, samplers, textures), its arena reads clean and still in place and
//    its writes resolve where they were recorded (BufferCache::ObtainBuffer, as the dispatch's
//    own binding would: it marks them GPU written); the texture reads are transitioned, every
//    range is ordered after earlier work, and recorded for later work, as the full path does.
//  - Templates: a dispatch whose template is not known posts a learn request at its packet;
//    the command processor dispatches it through the full path (DispatchRecord) and keeps what
//    BindResources resolved: pipeline, sampled textures, samplers, buffer slots (written ones
//    too). Storage images, special buffers (but the flattened user data), texel buffers,
//    feedback records and DMA reads are not taken.

namespace NativeCompute {
namespace np = iss::native_passes;
using NativePasses::g;

static bool Enabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE");
        return !p || p[0] != '0';
    }();
    return enabled;
}

constexpr u32 ProgramWords = sizeof(AmdGpu::ComputeProgram) / 4;
static_assert(ProgramWords <= np::CsWords);
static_assert(offsetof(AmdGpu::ComputeProgram, user_data) == 0x40 * 4);
static_assert(Shader::NUM_USER_DATA_REGS == np::UserWords);

// The key of the compute registers: every word but the initiator, the dimensions and the user
// data (as both sides have them: the lane's words at CsBase, the command processor's state).
static u64 ComputeKey(const AmdGpu::ComputeProgram& cs) {
    std::array<u32, ProgramWords> w;
    std::memcpy(w.data(), &cs, sizeof(cs));
    w[0] = w[1] = w[2] = w[3] = 0;
    for (u32 k = 0; k < np::UserWords; ++k) w[0x40 + k] = 0;
    return XXH3_64bits(w.data(), sizeof(w)) | 1;
}

static AmdGpu::ComputeProgram LaneProgram(const AmdGpu::Regs& regs) {
    AmdGpu::ComputeProgram cs;
    std::memcpy(&cs, &regs.reg_array[np::CsBase], sizeof(cs));
    return cs;
}

enum Special : u8 { Guest, Flat, Gds };
// ISS_NATIVE_COMPUTE_GDS (default on, =0 off): dispatches reading or writing the GDS (the particle
// rings' counters) recorded too, the GDS buffer as one range of its own (no guest address).
static bool GdsEnabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_GDS");
        return !p || p[0] != '0';
    }();
    return enabled;
}
constexpr VAddr GdsAddress = 0xFFFF000000000000ull;
struct Slot {
    u8 desc{};    ///< index in the stage's buffers
    u8 binding{}; ///< push data buffer offset slot
    u8 special{};
    bool written{}, formatted{};
    u32 write{};
};

struct Template {
    const ComputePipeline* pipeline{};
    const ComputePipeline* shape{};
    Pipeline::DescriptorWrites writes;
    std::vector<vk::DescriptorImageInfo> image_infos;
    std::vector<Rasterizer::NativeImageCapture> images;
    /// The texture descriptions of the images (one per image resource): the command processor
    /// looks them up again at every execution (TextureCache::FindImage/FindTexture), as the full
    /// path does whenever the texture cache changed. A template keeping the image it learned
    /// read a stale copy where the cache now resolves the memory to another image (the longhouse
    /// ambient lighting drifted, ab-aftermath-rotate 2026-10-17).
    std::vector<VideoCore::TextureCache::ImageDesc> image_descs;
    std::vector<Slot> buffers;
    Shader::PushData push{};
    u64 samplers_generation{};
    mutable std::atomic<u64> cache_generation{};
    u64 key{};
    bool has_samplers{};
    u32 order{}; ///< learn order (ISS_NATIVE_COMPUTE_ROTATE)
};

struct Tables {
    /// Written by the command processor only (exclusive lock); the pool threads read shared.
    std::shared_mutex mutex;
    std::unordered_map<u64, const ComputePipeline*> shapes; ///< compute key -> first pipeline
    std::unordered_map<u64, std::shared_ptr<const Template>> templates;
};
static Tables& State() {
    static Tables tables;
    return tables;
}

// What the pipeline permutation and the template's textures depend on, read through the user
// data (a pool thread's view, or the command processor's registers).
static u64 MaterialHash(const Shader::Info& stage) {
    NativeMaterial::Hasher h;
    for (const auto& desc : stage.buffers) {
        if (desc.IsSpecial()) {
            const u32 type = u32(desc.buffer_type);
            h.Put(&type, sizeof(type));
            continue;
        }
        const auto sharp = desc.GetSharp(stage);
        std::array<u32, 4> w;
        std::memcpy(w.data(), &sharp, sizeof(w));
        const std::array<u32, 3> k{w[1] >> 12, w[3], u32(sharp.base_address != 0)};
        h.Put(k.data(), sizeof(k));
    }
    for (const auto& desc : stage.images) {
        const auto sharp = desc.GetSharp(stage);
        h.Put(&sharp, sizeof(sharp));
    }
    for (const auto& desc : stage.samplers) {
        const auto sharp = desc.GetSharp(stage);
        h.Put(&sharp, sizeof(sharp));
    }
    return h.Digest();
}

// A pool thread's view of the compute stage's user data (Shader::Info::ud_snapshots).
static bool InstallView(const Shader::Info& stage, const AmdGpu::Regs& regs,
                        std::array<u32, 512>& flat, u32 slot) {
    const u32 size = stage.srt_info.flattened_bufsize_dw;
    if (size > flat.size() || size < Shader::NUM_USER_DATA_REGS || slot >= 2) return false;
    const u32* ud = &regs.reg_array[np::CsBase + 0x40];
    std::memcpy(flat.data(), ud, Shader::NUM_USER_DATA_REGS * 4);
    if (stage.srt_info.walker_func) stage.srt_info.walker_func(ud, flat.data());
    const auto cs = LaneProgram(regs);
    Shader::Info::ud_snapshots[slot] = Shader::Info::UdSnapshot{
        .info = &stage,
        .ud = ud,
        .ud_size = Shader::NUM_USER_DATA_REGS,
        .flat = flat.data(),
        .flat_size = size,
        .base = cs.Address<VAddr>(),
        .reads = nullptr,
        .num_reads = 0,
    };
    Shader::Info::num_ud_snapshots = slot + 1;
    return true;
}

// A buffer range a segment reads or writes.
struct Range {
    const VideoCore::Buffer* buffer{}; ///< the arena or the imported heap
    VAddr address{};
    u64 offset{}, size{};
    bool written{}, arena{}, formatted{};
    bool gds{}; ///< the GDS buffer (address GdsAddress)
};
// By guest address: the same memory can be in two arenas (ranges of other blocks).
static bool Overlap(const Range& a, const Range& b) {
    return a.address < b.address + b.size && b.address < a.address + a.size;
}

// A dispatch prepared by a pool thread: what the command processor records for it (its
// template's pipeline, the descriptors with the resolved buffers, the push data, a compute
// barrier before it when it depends on an earlier one of the segment). Recorded into the
// command processor's own command buffer: the same commands run from a secondary command
// buffer gave the longhouse ambient lighting pass wrong results (ab-aftermath-probe
// 2026-10-17: secondary wrong even with full barriers, the same commands replayed right).
struct Replay {
    const Template* t{};
    std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> infos{};
    Shader::PushData push{};
    std::array<u32, 3> dims{};
    bool barrier{};
};

struct Rec {
    std::vector<Replay> replays;
    vk::CommandBuffer cmd{};
    u32 generation{};
    bool begun{};
    bool needs_reset{};
    vk::Pipeline bound_pipeline{};
    std::vector<std::shared_ptr<const Template>> templates; ///< unique
    std::vector<Range> ranges;  ///< unique (address, size, written)
    std::vector<Range> pending; ///< since the last barrier of the secondary
    u64 arena_gen{};
    bool arena_gen_mixed{};
    std::vector<const void*> workers;
    u32 bb_segment{}, bb_dispatches{}, barriers{};
    /// The dispatches whose stage reads user data tables from memory (an SRT walker): the user
    /// data and the hash of the flattened words recorded. The pool thread read that memory at the
    /// submission, before the GPU work of the same frame that may write it (the command processor
    /// reads it when it gets there, through the readback of GPU written pages): the command
    /// processor walks it again and takes the segment only when every one is the same.
    struct Check {
        const Shader::Info* stage;
        std::array<u32, Shader::NUM_USER_DATA_REGS> ud;
        u64 hash;
        VAddr program;
        std::vector<u32> flat; ///< diagnosis: the words recorded (first mismatches print them)
    };
    std::vector<Check> checks;
};

enum Miss : u32 { MissNotReady, MissWorker, MissShape, MissTemplate, MissViews, MissPipeline,
                  MissBuffer, MissRing, MissCommand, MissEmpty, MissCount };
constexpr const char* miss_names[MissCount] = {"not_ready", "worker", "no_shape", "no_template",
                                               "views", "pipeline", "buffer", "ring",
                                               "command", "empty_first"};
enum Reject : u32 { RejectEnv, RejectRegs, RejectPipeline, RejectSamplers, RejectTextures,
                    RejectArena, RejectWrite, RejectAlias, RejectFlat, RejectRotate, RejectImage, RejectRead, RejectCount };
constexpr const char* reject_names[RejectCount] = {"env", "regs", "pipeline", "samplers",
                                                   "textures", "arena", "write", "alias", "flat",
                                                   "rotate", "image", "read"};

// ISS_NATIVE_COMPUTE_ROTATE=1 (diagnosis): which template's segments make a scene wrong. From 40 s
// of process time, windows of 6 s: 0 every segment native, 1..8 the segments with template k-1
// (learn order) left to the command processor, 9 all of them; from 100 s native again.
static double ProcessSeconds() {
    const auto ticks = [](const FILETIME& f) { return (u64(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    static const u64 created = [&] {
        FILETIME creation{}, exit_time{}, kernel{}, user{};
        GetProcessTimes(GetCurrentProcess(), &creation, &exit_time, &kernel, &user);
        return ticks(creation);
    }();
    FILETIME now{};
    GetSystemTimeAsFileTime(&now);
    return double(ticks(now) - created) / 1e7;
}
static int RotateWindow(double& seconds) {
    static const bool on = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_ROTATE");
        return p && p[0] == '1';
    }();
    if (!on) return -1;
    seconds = ProcessSeconds();
    if (seconds < 40.0 || seconds >= 100.0) return -1;
    return int((seconds - 40.0) / 6.0);
}
struct Stats {
    std::array<std::atomic<u64>, MissCount> miss{};
    std::atomic<u64> recorded{}, barriers{}, learn_requests{};
    // Command processor.
    u64 executed{}, dispatches{}, learned{}, learn_failed{}, learn_mismatch{}, report{},
        flat_checks{}, flat_mismatch{}, image_lookups{}, image_moved{}, read_moved{},
        read_stream{}, reads_uploaded{}, upload_moved{}, arena_gone{}, arena_moved{},
        asc_executed{}, asc_dispatches{}, asc_calls{};
    std::array<u64, RejectCount> reject{};
    std::array<u64, 8> learn_rejects{};
};
static Stats stats;

static Rec& RecOf(np::Segment& s) {
    if (!s.compute_recording) s.compute_recording = new Rec;
    return *static_cast<Rec*>(s.compute_recording);
}

static void Reset(Rec& r) {
    if (r.begun) r.needs_reset = true;
    r.begun = false;
    r.bound_pipeline = vk::Pipeline{};
    r.templates.clear();
    r.ranges.clear();
    r.pending.clear();
    r.arena_gen = 0;
    r.arena_gen_mixed = false;
    r.workers.clear();
    r.bb_dispatches = 0;
    r.barriers = 0;
    r.checks.clear();
    r.replays.clear();
}

static u64 Key(void* regs) {
    return ComputeKey(LaneProgram(*static_cast<const AmdGpu::Regs*>(regs)));
}

// One DISPATCH_DIRECT packet into the segment (a pool thread, under its lane's lock).
static np::DrawResult Record(np::Segment& s, void* regs_ptr, u64 key, const u32*) {
    const auto miss = [&](Miss m, np::DrawResult result = np::NoRecord) {
        stats.miss[m].fetch_add(1, std::memory_order_relaxed);
        Shader::Info::num_ud_snapshots = 0;
        return result;
    };
    if (!g.ready.load(std::memory_order_acquire)) return miss(MissNotReady);
    NativePasses::Worker* w = NativePasses::ThisWorker();
    if (!w) return miss(MissWorker);
    w->Reclaim();
    auto& regs = *static_cast<AmdGpu::Regs*>(regs_ptr);
    Rec& r = RecOf(s);
    if (r.generation != s.generation) {
        Reset(r);
        r.generation = s.generation;
    }
    const auto cs = LaneProgram(regs);
    if ((cs.dispatch_initiator & 1) == 0 || cs.dim_x == 0 || cs.dim_y == 0 || cs.dim_z == 0) {
        // Nothing the GPU would run (no command for it); never a segment's first.
        if (!r.begun) return miss(MissEmpty);
        return np::Recorded;
    }

    // The template of the compute registers and the material.
    std::shared_ptr<const Template> t;
    static thread_local std::array<u32, 512> flat0, flat1;
    {
        auto& tables = State();
        const ComputePipeline* shape = nullptr;
        {
            std::shared_lock lock{tables.mutex};
            const auto it = tables.shapes.find(key);
            if (it != tables.shapes.end()) shape = it->second;
        }
        if (!shape) {
            // A program the command processor may not have translated yet: the speculative
            // translator starts on it now (ISS_SPEC_COMPUTE), before the command processor
            // reaches the dispatch (937 new compute programs translated on it in a cold run,
            // cold2 2026-10-18).
            if (g.pipeline_cache) g.pipeline_cache->RequestComputeSpecTranslation(cs);
            return miss(MissShape, NativePasses::LearnGate(key));
        }
        Shader::Info::num_ud_snapshots = 0;
        const auto& shape_stage = shape->GetStage(Shader::SwStage::Compute);
        if (!InstallView(shape_stage, regs, flat0, 0)) return miss(MissViews);
        const u64 material = MaterialHash(shape_stage);
        const u64 template_key = NativeMaterial::Mix(key, material);
        {
            std::shared_lock lock{tables.mutex};
            const auto it = tables.templates.find(template_key);
            if (it != tables.templates.end()) t = it->second;
        }
        if (!t) return miss(MissTemplate, NativePasses::LearnGate(template_key));
        if (t->pipeline != shape &&
            !InstallView(t->pipeline->GetStage(Shader::SwStage::Compute), regs, flat1, 1))
            return miss(MissViews);
    }
    const auto* pipeline = t->pipeline;
    if (!pipeline->IsReady() || !pipeline->UsesPushDescriptors()) return miss(MissPipeline);
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (stage.srt_info.walker_func) {
        const auto flat = stage.FlatUserData();
        auto& check = r.checks.emplace_back();
        check.stage = &stage;
        std::memcpy(check.ud.data(), &regs.reg_array[np::CsBase + 0x40], sizeof(check.ud));
        check.hash = XXH3_64bits(flat.data(), flat.size() * sizeof(u32));
        check.program = cs.Address<VAddr>();
        static std::atomic<u32> kept{0};
        if (kept.load(std::memory_order_relaxed) < 4096) {
            kept.fetch_add(1, std::memory_order_relaxed);
            check.flat.assign(flat.begin(), flat.end());
        } else {
            check.flat.clear();
        }
    }

    r.begun = true;
    w->Note(s, r);
    const VideoCore::Buffer* unified = g.unified.load(std::memory_order_acquire);
    const VAddr unified_base = g.unified_base.load(std::memory_order_relaxed);
    auto& cache = *g.buffer_cache;
    const u64 alignment = g.instance->StorageMinAlignment();

    // The dispatch's buffer ranges and descriptors.
    static thread_local std::vector<Range> mine;
    mine.clear();
    std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> infos;
    Shader::PushData push = t->push;
    for (u32 k = 0; k < t->buffers.size(); ++k) {
        const auto& slot = t->buffers[k];
        auto& info = infos[k];
        if (slot.special == Flat) {
            const auto flat = stage.FlatUserData();
            const u64 bytes = flat.size() * sizeof(u32);
            u64 offset;
            u8* dst = w->Upload(bytes, alignment, offset);
            if (!dst) return miss(MissRing);
            std::memcpy(dst, flat.data(), bytes);
            info = vk::DescriptorBufferInfo{w->ring, offset, bytes};
            continue;
        }
        if (slot.special == Gds) {
            const auto* gds = cache.GetGdsBuffer();
            info = vk::DescriptorBufferInfo{gds->Handle(), 0, gds->SizeBytes()};
            mine.push_back(Range{.buffer = gds, .address = GdsAddress, .offset = 0,
                                 .size = gds->SizeBytes(), .written = slot.written, .gds = true});
            continue;
        }
        const auto sharp = stage.buffers[slot.desc].GetSharp(stage);
        if (!sharp.base_address || !sharp.GetSize()) {
            info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
            continue;
        }
        const VAddr address = sharp.base_address;
        const u64 size = sharp.GetSize();
        Range range{.address = address, .size = size, .written = slot.written,
                    .formatted = slot.formatted};
        vk::Buffer handle;
        if (unified && cache.IsUnified(address, size)) {
            range.buffer = unified;
            range.offset = address - unified_base;
            handle = unified->Handle();
        } else {
            // Read first: an arena created or migrated after it changes the generation.
            const u64 arena_gen = cache.ArenaBindingGeneration();
            const auto arena = cache.PeekPublishedArena(address, u32(size));
            if (!arena) return miss(MissBuffer);
            range.buffer = arena.arena;
            range.offset = arena.offset;
            range.arena = true;
            handle = arena.arena->Handle();
            bool first_arena = true;
            for (const auto& a : r.ranges) first_arena &= !a.arena;
            if (first_arena) r.arena_gen = arena_gen;
            else if (r.arena_gen != arena_gen) r.arena_gen_mixed = true;
        }
        const u64 aligned = Common::AlignDown(range.offset, alignment);
        push.buf_offsets[slot.binding] = u8(range.offset - aligned);
        info = vk::DescriptorBufferInfo{handle, aligned, size + (range.offset - aligned)};
        mine.push_back(range);
    }

    // Ordered after the segment's earlier dispatches it depends on.
    // ISS_NATIVE_COMPUTE_GDS_ATOMIC=1: GDS against GDS is no hazard. The GDS is one range for the
    // whole buffer, so every pair of particle dispatches (append counters, atomic) got a barrier:
    // ~60 of a segment's 64 dispatches, each a GPU drain (~5 us), ~2/3 of the forest's GPU frame
    // (gprof-visitors 2026-10-18). The game puts no wait between them (the PS4 overlaps them).
    static const bool gds_atomic = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_GDS_ATOMIC");
        return p && p[0] == '1';
    }();
    bool hazard = false;
    for (const auto& a : mine)
        for (const auto& b : r.pending) {
            const bool h = (a.written || b.written) && !(gds_atomic && a.gds && b.gds) && Overlap(a, b);
            // ISS_NATIVE_COMPUTE_HAZARD_LOG=1 (diagnosis): which ranges order the dispatches.
            static const bool hazard_log = [] {
                const char* p = std::getenv("ISS_NATIVE_COMPUTE_HAZARD_LOG");
                return p && p[0] == '1';
            }();
            if (h && !hazard && hazard_log) {
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 300)
                    std::fprintf(stderr,
                                 "NATIVE_COMPUTE_HAZARD pgm=%llx dispatch=%zu new=%llx+%llx%s%s%s "
                                 "earlier=%llx+%llx%s%s%s\n",
                                 (unsigned long long)stage.pgm_hash, r.replays.size(),
                                 (unsigned long long)a.address, (unsigned long long)a.size,
                                 a.written ? "W" : "R", a.arena ? "a" : "u", a.gds ? "g" : "",
                                 (unsigned long long)b.address, (unsigned long long)b.size,
                                 b.written ? "W" : "R", b.arena ? "a" : "u", b.gds ? "g" : "");
            }
            hazard |= h;
        }
    // ISS_NATIVE_COMPUTE_PS4_ORDER (default on, =0: a barrier wherever two dispatches' buffer
    // ranges overlap): the dispatches of a segment run as on the PS4, with no barrier between them.
    // A segment only spans dispatches with register writes between them: the game asks for no
    // wait there (a wait, an event, a fence, a DMA ends the segment, and the next segment is ordered
    // after it). The particle stages all bind one 12 MB pool, so the overlap rule put a barrier
    // before ~3 of 4 dispatches: The Visitors' particle compute 436 -> 80 ms of GPU a second,
    // 58.2 -> 82.5 FPS uncapped, fire and smoke the same (nb-off/nb-on, nb2 renderer captures,
    // 2026-10-18 night).
    static const bool ps4_order = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_PS4_ORDER");
        const char* old = std::getenv("ISS_NATIVE_COMPUTE_NO_BARRIERS");
        return !(p && p[0] == '0') || (old && old[0] == '1');
    }();
    if (ps4_order) hazard = false;
    if (hazard) {
        r.pending.clear();
        ++r.barriers;
        stats.barriers.fetch_add(1, std::memory_order_relaxed);
    }
    for (const auto& a : mine) {
        r.pending.push_back(a);
        bool listed = false;
        for (auto& b : r.ranges)
            if (b.address == a.address && b.size == a.size && b.buffer == a.buffer) {
                b.written |= a.written;
                listed = true;
                break;
            }
        if (!listed) r.ranges.push_back(a);
    }
    bool listed = false;
    for (const auto& u : r.templates) listed |= u.get() == t.get();
    if (!listed) r.templates.push_back(t);

    auto& d = r.replays.emplace_back();
    d.t = t.get();
    d.infos = infos;
    d.push = push;
    d.dims = {cs.dim_x, cs.dim_y, cs.dim_z};
    d.barrier = hazard;
    stats.recorded.fetch_add(1, std::memory_order_relaxed);
    Shader::Info::num_ud_snapshots = 0;
    return np::Recorded;
}

static bool Close(np::Segment& s) {
    auto* r = static_cast<Rec*>(s.compute_recording);
    if (!r || !r->begun) return false;
    r->begun = false;
    return !r->templates.empty() && !r->replays.empty();
}

static bool (*close_draws)(np::Segment&) = nullptr;
static bool CloseAny(np::Segment& s) {
    return s.kind == np::Kind::Compute ? Close(s) : close_draws(s);
}

static void Install(np::Callbacks& callbacks) {
    if (!Enabled() || !np::SubmitMode()) return;
    close_draws = callbacks.close;
    callbacks.close = &CloseAny;
    callbacks.compute_key = &Key;
    callbacks.record_dispatch = &Record;
    std::fprintf(stderr, "NATIVE_COMPUTE on (graphics queue dispatches recorded at submit)\n");
}

static void Report() {
    std::string misses;
    for (u32 m = 0; m < MissCount; ++m) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", miss_names[m],
                      (unsigned long long)stats.miss[m].load(std::memory_order_relaxed));
        misses += item;
    }
    std::string rejects;
    for (u32 m = 0; m < RejectCount; ++m) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", reject_names[m],
                      (unsigned long long)stats.reject[m]);
        rejects += item;
    }
    std::string learn_rejects;
    for (u32 m = 0; m < stats.learn_rejects.size(); ++m) {
        char item[32];
        std::snprintf(item, sizeof(item), " %u=%llu", m, (unsigned long long)stats.learn_rejects[m]);
        learn_rejects += item;
    }
    std::fprintf(stderr,
                 "NATIVE_COMPUTE engine recorded=%llu barriers=%llu miss[%s ] | cp executed=%llu "
                 "dispatches=%llu flat_checks=%llu flat_mismatch=%llu image_lookups=%llu "
                 "image_moved=%llu read_moved=%llu read_stream=%llu reads_uploaded=%llu upload_moved=%llu arena_gone=%llu arena_moved=%llu learned=%llu "
                 "learn_failed=%llu learn_mismatch=%llu "
                 "templates=%zu shapes=%zu reject[%s ] learn_reject[%s ] "
                 "asc[calls=%llu executed=%llu dispatches=%llu ]\n",
                 (unsigned long long)stats.recorded.load(), (unsigned long long)stats.barriers.load(),
                 misses.c_str(), (unsigned long long)stats.executed,
                 (unsigned long long)stats.dispatches, (unsigned long long)stats.flat_checks,
                 (unsigned long long)stats.flat_mismatch, (unsigned long long)stats.image_lookups,
                 (unsigned long long)stats.image_moved, (unsigned long long)stats.read_moved,
                 (unsigned long long)stats.read_stream, (unsigned long long)stats.reads_uploaded, (unsigned long long)stats.upload_moved, (unsigned long long)stats.arena_gone, (unsigned long long)stats.arena_moved,
                 (unsigned long long)stats.learned,
                 (unsigned long long)stats.learn_failed, (unsigned long long)stats.learn_mismatch,
                 State().templates.size(), State().shapes.size(), rejects.c_str(),
                 learn_rejects.c_str(), (unsigned long long)stats.asc_calls,
                 (unsigned long long)stats.asc_executed, (unsigned long long)stats.asc_dispatches);
}
} // namespace NativeCompute

bool Rasterizer::NativeComputeExecute(iss::native_passes::Segment& s, u64& tick, bool asc) {
    // ISS_CP_PATH_CYCLES: reported as rec_submit (the native recorder's phase, unused here).
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::RecSubmit};
    using namespace NativeCompute;
    auto& c = stats;
    // Every 10 s at most (was every 1024 segments: ~200 lines a second on this thread).
    static u64 last_report_ns = 0;
    ++c.report;
    if (const u64 now = Common::Perf::NowNs(); now - last_report_ns >= 10'000'000'000ull) {
        last_report_ns = now;
        Report();
    }
    c.asc_calls += asc;
    const auto reject = [&](Reject why) {
        ++c.reject[why];
        return false;
    };
    // An asynchronous compute queue's segment runs where its full path dispatches do: in the
    // graphics scheduler's command buffer (asc_queue routing off, no compute record context).
    if (cp_split_enabled || draw_pipe_state || OnDrawWorker() || (DrawQueue() != 0 && !asc) ||
        (asc && OnComputeRecordQueue()) || ComputeBindingsActive() || iss::command_trace::enabled || IsDumping() ||
        host_markers_enabled || guest_markers_enabled || GpuCrashDiag::Enabled() ||
        EmulatorSettings.IsPatchShaders() || buffer_cache.HasComputeQueue() ||
        iss::hazard_probe::Enabled())
        return reject(RejectEnv);
    auto* r = static_cast<Rec*>(s.compute_recording);
    if (!r || r->templates.empty() || r->generation != s.generation) return reject(RejectEnv);
    // The registers the pool thread recorded the first dispatch with are these (exact in submit
    // mode: a check against a scan gone wrong).
    const auto& cs = DrawCsRegs();
    if (ComputeKey(cs) != s.first_context || s.first_user.size() != np::UserWords ||
        std::memcmp(cs.user_data.data(), s.first_user.data(), np::UserWords * 4) != 0)
        return reject(RejectRegs);
    {
        double seconds = 0;
        const int w = RotateWindow(seconds);
        static int last = -2;
        if (w != last) {
            last = w;
            std::fprintf(stderr, "NATIVE_COMPUTE_ROTATE window=%d at=%.1fs\n", w, seconds);
        }
        if (w > 0) {
            bool deny = w == 9;
            for (const auto& t : r->templates) deny |= t->order == u32(w - 1);
            if (deny) return reject(RejectRotate);
        }
    }
    // The user data tables as the command processor reads them now (GPU written pages through
    // the readback): the same words the pool thread recorded each dispatch with.
    for (const auto& k : r->checks) {
        static std::array<u32, 512> flat;
        const u32 size = k.stage->srt_info.flattened_bufsize_dw;
        std::memcpy(flat.data(), k.ud.data(), sizeof(k.ud));
        k.stage->srt_info.walker_func(k.ud.data(), flat.data());
        ++c.flat_checks;
        if (XXH3_64bits(flat.data(), size * sizeof(u32)) != k.hash) {
            if (++c.flat_mismatch <= 24) {
                // Which words of the user data tables changed between the submission and now.
                std::string words;
                for (u32 w = 0; w < size && w < k.flat.size() && words.size() < 400; ++w)
                    if (k.flat[w] != flat[w]) {
                        char item[48];
                        std::snprintf(item, sizeof(item), " [%u]%08x->%08x", w, k.flat[w], flat[w]);
                        words += item;
                    }
                std::string ud;
                for (u32 w = 0; w < k.ud.size(); ++w) {
                    char item[16];
                    std::snprintf(item, sizeof(item), " %08x", k.ud[w]);
                    ud += item;
                }
                std::fprintf(stderr,
                             "NATIVE_COMPUTE_FLAT_MISMATCH program=%llx (eboot-relative unknown) "
                             "flat_words=%u ud[%s ] changed[%s ]\n",
                             (unsigned long long)k.program, size, ud.c_str(), words.c_str());
            }
            return reject(RejectFlat);
        }
    }
    iss::cp_path::Switch(iss::cp_path::CsTextures);
    for (const auto& t : r->templates) {
        if (!t->pipeline->IsReady()) return reject(RejectPipeline);
        if (t->has_samplers && texture_cache.SamplersGeneration() != t->samplers_generation)
            return reject(RejectSamplers);
    }
    for (const auto& t : r->templates) {
        bool stale = false;
        if (!NativeMaterial::TexturesValid(texture_cache, *t, stale)) {
            if (stale) {
                std::unique_lock lock{State().mutex};
                State().templates.erase(t->key);
            }
            return reject(RejectTextures);
        }
    }
    // The textures as the full path finds them now (the same image and view, contents refreshed).
    for (const auto& t : r->templates)
        for (u32 k = 0; k < t->images.size() && k < t->image_descs.size(); ++k) {
            const auto& cap = t->images[k];
            if (!cap.id) continue;
            auto desc = t->image_descs[k];
            ++c.image_lookups;
            auto id = texture_cache.FindImage(desc);
            if (const auto depth = texture_cache.GetAssociatedDepth(texture_cache.GetImage(id)))
                id = depth;
            const bool same_image = id == cap.id;
            const bool same_view =
                same_image && *texture_cache.FindTexture(id, desc).image_view == cap.view;
            if (!same_view) {
                if (++c.image_moved <= 16) {
                    const auto& was = texture_cache.GetImage(cap.id).info;
                    const auto& now = texture_cache.GetImage(id).info;
                    std::fprintf(stderr,
                                 "NATIVE_COMPUTE_IMAGE_MOVED program=%llx image %u@%llx(%ux%u fmt %d) "
                                 "-> %u@%llx(%ux%u fmt %d) same_image=%d\n",
                                 (unsigned long long)t->pipeline->GetStage(Shader::SwStage::Compute)
                                     .pgm_hash,
                                 cap.id.index, (unsigned long long)was.guest_address,
                                 was.size.width, was.size.height, int(was.pixel_format), id.index,
                                 (unsigned long long)now.guest_address, now.size.width,
                                 now.size.height, int(now.pixel_format), int(same_image));
                }
                return reject(RejectImage);
            }
        }
    iss::cp_path::Switch(iss::cp_path::CsReads);
    // Reads in an arena: still there and not written by the CPU since (the full path would
    // upload them first).
    const bool arenas_same =
        !r->arena_gen_mixed && buffer_cache.ArenaBindingGeneration() == r->arena_gen;
    // ISS_NATIVE_COMPUTE_UPLOAD (default on, =0 off): a read the CPU wrote since is uploaded below
    // as the full path's binding would (ObtainBuffer), the segment staying native when the data
    // lands where it was recorded. Rejecting them sent 63% of Parting's compute segments to the
    // full path (scenes-porte-n 2026-10-17: arena=15473 against 9103 run).
    static const bool upload_reads = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_UPLOAD");
        return !p || p[0] != '0';
    }();
    boost::container::small_vector<const Range*, 8> dirty_reads;
    for (const auto& a : r->ranges) {
        if (!a.arena || a.written) continue;
        if (!buffer_cache.IsRegionCpuClean(a.address, a.size)) {
            if (!upload_reads) return reject(RejectArena);
            dirty_reads.push_back(&a);
            continue;
        }
        if (arenas_same) continue;
        const auto arena = buffer_cache.PeekPublishedArena(a.address, u32(a.size));
        if (!arena) {
            ++c.arena_gone;
            return reject(RejectArena);
        }
        if (arena.arena != a.buffer || arena.offset != a.offset) {
            ++c.arena_moved;
            return reject(RejectArena);
        }
    }
    // A texture of the segment over memory one of its dispatches writes: the full path.
    for (const auto& t : r->templates)
        for (const auto& cap : t->images) {
            if (!cap.id) continue;
            const auto& info = texture_cache.GetImage(cap.id).info;
            for (const auto& a : r->ranges)
                if (a.written && a.address < info.guest_address + info.guest_size &&
                    info.guest_address < a.address + a.size)
                    return reject(RejectAlias);
        }

    // ISS_ASC_NATIVE_QUEUE: a compute ring's segment of buffers only (no image, no CPU data to
    // upload for its reads, no GDS) runs on the compute queue, concurrently with the rendering.
    bool on_compute = asc && AscQueueEnabled();
    if (on_compute) {
        bool images = false, gds = false;
        for (const auto& t : r->templates)
            for (const auto& cap : t->images) images |= bool(cap.id);
        for (const auto& a : r->ranges) gds |= a.gds;
        on_compute = !images && dirty_reads.empty();
        if (!on_compute) AscSegmentOnGraphics(images ? 0 : !dirty_reads.empty() ? 1 : 2);
    }
    if (asc && !on_compute) AscBeforeGraphicsWork(asc_vqid);

    // As Dispatch does before each dispatch.
    DrainDrawPipe(5);
    SubmitFeedbackBatch(scheduler);
    PeriodicSubmit(scheduler);
    scheduler.PopPendingOperations();

    // The reads too: where the full path binds them now (a stream copy of CPU data, another arena:
    // the full path).
    static const bool read_check = [] {
        const char* p = std::getenv("ISS_NATIVE_COMPUTE_READ_CHECK");
        return p && p[0] == '1';
    }();
    for (const auto& a : r->ranges) {
        if (a.written || !read_check || a.gds) continue;
        const u64 size = memory->ClampRangeSize(a.address, a.size);
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(a.address, u32(size), false,
                                                                a.formatted);
        if (buffer != a.buffer || offset != a.offset) {
            const bool stream = buffer == &buffer_cache.GetStreamBuffer();
            c.read_stream += stream;
            if (++c.read_moved <= 16)
                std::fprintf(stderr,
                             "NATIVE_COMPUTE_READ_MOVED program=%llx addr=%llx size=%llx %s "
                             "recorded=%s\n",
                             (unsigned long long)r->templates.front()->pipeline->GetStage(
                                 Shader::SwStage::Compute).pgm_hash,
                             (unsigned long long)a.address, (unsigned long long)a.size,
                             stream ? "stream" : "other_buffer", a.arena ? "arena" : "unified");
            return reject(RejectRead);
        }
    }
    for (const Range* dirty : dirty_reads) {
        auto& a = const_cast<Range&>(*dirty);
        // Memory a dispatch of the segment also writes: the full path (it interleaves them).
        for (const auto& b : r->ranges)
            if (b.written && Overlap(a, b)) return reject(RejectRead);
        const u64 size = memory->ClampRangeSize(a.address, a.size);
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(a.address, u32(size), false,
                                                                a.formatted);
        if (buffer != a.buffer || offset != a.offset) {
            // Bound elsewhere now (the buffer cache gives CPU written data as a stream copy): the
            // dispatches reading the range bind that copy, as the full path's binding would.
            const u64 align = instance.StorageMinAlignment();
            const vk::Buffer old_handle = a.buffer->Handle();
            const u64 old_aligned = Common::AlignDown(a.offset, align);
            const u64 new_aligned = Common::AlignDown(offset, align);
            for (auto& d : r->replays)
                for (u32 k = 0; k < d.t->buffers.size(); ++k) {
                    auto& info = d.infos[k];
                    if (info.buffer != old_handle || info.offset != old_aligned ||
                        info.range != a.size + (a.offset - old_aligned))
                        continue;
                    info = vk::DescriptorBufferInfo{buffer->Handle(), new_aligned,
                                                    a.size + (offset - new_aligned)};
                    d.push.buf_offsets[d.t->buffers[k].binding] = u8(offset - new_aligned);
                }
            a.buffer = buffer;
            a.offset = offset;
            a.arena = false;
            ++c.upload_moved;
        }
        ++c.reads_uploaded;
    }
    iss::cp_path::Switch(iss::cp_path::CsWrites);
    // A write resolving to CPU data uploaded first (on the graphics queue): the compute queue
    // waits for that upload.
    const u64 uploads_before = on_compute ? Common::Perf::Total(Common::Perf::BufferUploads) : 0;
    // The writes resolve where they were recorded, through the buffer cache as the dispatch's
    // own binding would (GPU written from now on; CPU data uploaded first).
    for (const auto& a : r->ranges) {
        if (a.gds) {
            buffer_cache.NoteGdsAccess(a.written);
            continue;
        }
        if (!a.written) continue;
        const u64 size = memory->ClampRangeSize(a.address, a.size);
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(a.address, u32(size), true,
                                                                a.formatted);
        if (buffer != a.buffer || offset != a.offset) return reject(RejectWrite);
    }
    if (on_compute && Common::Perf::Total(Common::Perf::BufferUploads) != uploads_before)
        AscBeforeGraphicsWork(asc_vqid);

    {
        static u32 logged = 0;
        bool first = false;
        for (const auto& t : r->templates) first |= t->order == 0;
        if (first && logged < 6) {
            ++logged;
            std::string text;
            for (const auto& a : r->ranges) {
                char item[96];
                std::snprintf(item, sizeof(item), " %llx+%llx%s%s", (unsigned long long)a.address,
                              (unsigned long long)a.size, a.written ? "W" : "R",
                              a.arena ? "a" : "u");
                if (text.size() < 1500) text += item;
            }
            std::fprintf(stderr, "NATIVE_COMPUTE_SEG0 dispatches=%zu templates=%zu barriers=%u ranges[%s ]\n",
                         s.draw_packets.size(), r->templates.size(), r->barriers, text.c_str());
        }
    }
    iss::cp_path::Switch(iss::cp_path::CsBarriers);
    if (!on_compute) {
        const u64 now = scheduler.CurrentTick();
        for (const auto& t : r->templates)
            for (const auto& cap : t->images) {
                if (!cap.id) continue;
                auto& image = texture_cache.GetImage(cap.id);
                needs_barrier |= runtime.Transit(&image, cap.layout,
                                                 vk::PipelineStageFlagBits2::eAllCommands,
                                                 vk::AccessFlagBits2::eShaderRead, cap.range);
                image.usage.texture = 1u;
                image.tick_accessed_last = now;
            }
        for (const auto& a : r->ranges)
            needs_barrier |= Rt().IsBufferAccessed(a.buffer, a.offset, a.size, a.written);
        scheduler.EndRendering();
        if (needs_barrier) runtime.FlushBarriers();
        needs_barrier = false;
    }
    iss::cp_path::Switch(iss::cp_path::CsRecord);
    const auto record = [&](auto& rec, bool mark) {
        vk::Pipeline bound{};
        for (const auto& d : r->replays) {
            const auto* t = d.t;
            if (d.barrier) {
                const vk::MemoryBarrier2 barrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask =
                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask =
                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
                };
                rec.pipelineBarrier2(
                    vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
            }
            if (t->pipeline->Handle() != bound) {
                bound = t->pipeline->Handle();
                rec.bindPipeline(vk::PipelineBindPoint::eCompute, bound);
            }
            Pipeline::DescriptorWrites writes = t->writes;
            for (u32 k = 0; k < t->buffers.size(); ++k)
                writes[t->buffers[k].write].pBufferInfo = &d.infos[k];
            if (!writes.empty())
                rec.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, t->pipeline->GetLayout(), 0,
                                         writes);
            rec.pushConstants(t->pipeline->GetLayout(), vk::ShaderStageFlagBits::eCompute, 0u,
                              sizeof(d.push), &d.push);
            if constexpr (!std::is_same_v<std::decay_t<decltype(rec)>, vk::CommandBuffer>) {
                if (mark)
                    MarkDispatch(Sched(), rec,
                                 t->pipeline->GetStage(Shader::SwStage::Compute).pgm_hash,
                                 d.dims[0], d.dims[1], d.dims[2], 0);
            }
            rec.dispatch(d.dims[0], d.dims[1], d.dims[2]);
        }
    };
    if (on_compute) {
        // The compute queue: after the earlier compute work (the previous segments may write
        // what these dispatches read) and the graphics work this ring must follow.
        EnsureComputeQueueInitialized();
        AscBeforeComputeWork(asc_vqid);
        vk::CommandBuffer cmd = compute_scheduler->CommandBuffer();
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        record(cmd, false);
    } else {
        // The prepared dispatches, into the command processor's command buffer.
        // ISS_GPU_PROFILE=2: timed as one "shader" per kind of segment (a5c... compute rings,
        // 6fc... graphics queue), in the Perf GPU compute report.
        Sched().ProfileDispatchBegin(asc ? 0xA5C0000000000000ull | r->barriers
                                         : 0x6FC0000000000000ull | r->barriers);
        auto rec = scheduler.Recorder();
        record(rec, GpuCrashDiag::Enabled() || GpuCrashDiag::Blackbox());
        Sched().ProfileDispatchEnd();
    }
    iss::cp_path::Switch(iss::cp_path::CsAccess);
    if (!on_compute) {
        // The compute pipeline and push descriptors the command processor's bind cache assumed.
        scheduler.GetDynamicState().Invalidate();
        fast_draw.valid = false;
    }
    bool wrote = false;
    for (const auto& a : r->ranges) {
        if (!on_compute)
            Rt().AccessBuffer(a.buffer, a.offset, a.size, vk::PipelineStageFlagBits2::eComputeShader,
                              vk::AccessFlagBits2::eShaderRead |
                                  (a.written ? vk::AccessFlagBits2::eShaderWrite
                                             : vk::AccessFlagBits2::eNone));
        if (!a.written || a.gds) continue;
        wrote = true;
        texture_cache.InvalidateMemoryFromGPU(a.address, a.size);
        if (!skipped_writes.empty()) ClearSkippedWrites(a.address, a.size);
    }
    if (wrote) BumpBindEpoch();
    // The graphics session these writes were stamped with waits for them when it is submitted
    // (Scheduler before-submit callback: AscFlush).
    for (size_t i = 0; i < s.draw_packets.size(); ++i) DebugState.IncDispatch();
    tick = scheduler.CurrentTick();
    ++c.executed;
    c.dispatches += s.draw_packets.size();
    if (asc) {
        ++c.asc_executed;
        c.asc_dispatches += s.draw_packets.size();
    }
    return true;
}

void Rasterizer::NativeComputeLearn(u64 key, const ComputePipeline* pipeline,
                                    const std::vector<NativeImageCapture>& caps) {
    using namespace NativeCompute;
    auto& c = stats;
    auto& u = NativePasses::Rejected();
    const auto give_up = [&](u32 why) {
        ++c.learn_failed;
        ++c.learn_rejects[std::min<u32>(why, 7)];
        if (++u.attempts[key] >= 4) {
            std::unique_lock lock{u.mutex};
            u.keys.insert(key);
        }
    };
    if ((c.learned + c.learn_failed) % 256 == 255) Report();
    if (ComputeKey(DrawCsRegs()) != key) {
        ++c.learn_mismatch;
        return;
    }
    const auto& stage = pipeline->GetStage(Shader::SwStage::Compute);
    if (WritesFeedbackRecord(stage) || stage.uses_dma || !pipeline->UsesPushDescriptors())
        return give_up(1);
    for (const auto& cap : caps)
        if (cap.id && !cap.plain) return give_up(2);

    auto t = std::make_shared<Template>();
    t->pipeline = pipeline;
    for (const auto& res : stage.images) {
        if (res.NumBindings(stage) != 1) return give_up(2);
        const auto tsharp = res.GetSharp(stage);
        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            t->image_descs.emplace_back();
            continue;
        }
        auto& desc = t->image_descs.emplace_back(tsharp, res);
        if (res.mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
            desc.view_info.range.base.level += res.constant_mip_index;
            desc.view_info.range.extent.levels = 1;
        }
    }
    u32 write = 0, binding = 0, num_images = 0, num_image_infos = 0;
    for (u32 k = 0; k < stage.buffers.size(); ++k) {
        const auto& desc = stage.buffers[k];
        if (write >= set_writes.size() || !set_writes[write].pBufferInfo ||
            set_writes[write].descriptorCount != 1)
            return give_up(3);
        u8 special = Guest;
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf) special = Flat;
            else if (desc.buffer_type == Shader::BufferType::GdsBuffer && GdsEnabled()) special = Gds;
            else return give_up(4);
        }
        t->buffers.push_back(Slot{u8(k), u8(binding++), special, desc.is_written,
                                  desc.is_formatted, write++});
    }
    for (u32 k = 0; k < stage.images.size(); ++k) {
        if (write >= set_writes.size() || !set_writes[write].pImageInfo ||
            set_writes[write].descriptorType != vk::DescriptorType::eSampledImage)
            return give_up(5);
        num_images += set_writes[write].descriptorCount;
        num_image_infos += set_writes[write].descriptorCount;
        ++write;
    }
    for (u32 k = 0; k < stage.samplers.size(); ++k) {
        if (write >= set_writes.size() || !set_writes[write].pImageInfo ||
            set_writes[write].descriptorType != vk::DescriptorType::eSampler)
            return give_up(5);
        num_image_infos += set_writes[write].descriptorCount;
        t->has_samplers = true;
        ++write;
    }
    if (write != set_writes.size() || num_images != caps.size()) return give_up(6);

    t->writes = set_writes;
    t->image_infos.reserve(num_image_infos);
    for (auto& w : t->writes) {
        if (w.pBufferInfo) {
            w.pBufferInfo = nullptr;
        } else {
            const u32 first = u32(t->image_infos.size());
            for (u32 i = 0; i < w.descriptorCount; ++i) t->image_infos.push_back(w.pImageInfo[i]);
            w.pImageInfo = &t->image_infos[first];
        }
    }
    t->images = caps;
    t->push = push_data;
    t->push.buf_offsets.fill(0);
    t->samplers_generation = texture_cache.SamplersGeneration();
    t->cache_generation = texture_cache.Generation();

    u32 written_slots = 0;
    for (const auto& slot : t->buffers) written_slots += slot.written;
    auto& tables = State();
    const auto shape_it = tables.shapes.find(key);
    const auto* shape = shape_it != tables.shapes.end() ? shape_it->second : pipeline;
    const auto& shape_stage = shape->GetStage(Shader::SwStage::Compute);
    if (shape_stage.buffers.size() != stage.buffers.size() ||
        shape_stage.images.size() != stage.images.size() ||
        shape_stage.samplers.size() != stage.samplers.size())
        return give_up(7);
    if (shape != pipeline) const_cast<Shader::Info&>(shape_stage).RefreshFlatBuf();
    t->shape = shape;
    t->key = NativeMaterial::Mix(key, MaterialHash(shape_stage));
    const u32 order = u32(c.learned);
    t->order = order;
    {
        std::unique_lock lock{tables.mutex};
        if (tables.templates.size() >= 65536) iss::DestroyInBackground(tables.templates);
        tables.shapes.try_emplace(key, pipeline);
        tables.templates[t->key] = std::move(t);
    }
    ++c.learned;
    // The first templates one by one, then every 256th (4453 lines in an 8 minute session).
    if (order < 64 || order % 256 == 0) {
        std::fprintf(stderr,
                     "NATIVE_COMPUTE_TEMPLATE order=%u key=%llx program=%llx pgm_hash=%llx walker=%d "
                     "flat_words=%u buffers=%zu written=%u images=%zu samplers=%zu\n",
                     order, (unsigned long long)key,
                     (unsigned long long)DrawCsRegs().Address<VAddr>(),
                     (unsigned long long)stage.pgm_hash, int(stage.srt_info.walker_func != nullptr),
                     stage.srt_info.flattened_bufsize_dw, stage.buffers.size(), written_slots,
                     stage.images.size(), stage.samplers.size());
    }
}

void Rasterizer::NativeComputeLearnFailed(u64 key) {
    using namespace NativeCompute;
    ++stats.learn_failed;
    auto& u = NativePasses::Rejected();
    if (++u.attempts[key] >= 4) {
        std::unique_lock lock{u.mutex};
        u.keys.insert(key);
    }
}
