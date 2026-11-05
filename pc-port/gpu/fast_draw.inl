// SPDX-License-Identifier: GPL-2.0-or-later
// Fast draw, included by vk_rasterizer.cpp (Rasterizer members).
//
// Most draws of the city repeat the whole state of the draw before them and only bind other
// geometry: the same material is drawn for many meshes, each with its own fetch V# list
// (vertex shader user data) and index range. The full path looks the pipeline up again,
// resolves every descriptor, rebuilds the render state and pushes the same descriptors and
// constants again. A fast draw checks that nothing those results depend on changed since the
// previous draw and records only what differs: vertex buffers, index buffer and the draw.
//
// The previous draw leaves its pipeline, push constants, descriptors, vertex input, dynamic
// state and render pass in the command buffer. BindCache::epoch proves nothing else was bound
// or forgotten since; StageMemoStamp proves no upload, write, transition, invalidation,
// dispatch, copy or readback happened; the context register change counter, the programs and
// the pixel stage user data prove the same pipeline and pixel descriptors; the vertex stage
// descriptors and fetch formats are compared after refreshing its flattened user data.


// ISS_DRAW_CENSUS=1: draws counted by render pass shape (size, color attachments, depth, pixel
// stage) and by the path they took (0 plain fast, 1 lean, 2 rebind, 3 shader switch, 4 full).
static bool DrawCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_DRAW_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
struct DrawCensusEntry {
    std::array<u64, 5> paths{};
    u64 vs_hash{}, ps_hash{};
};
static std::unordered_map<u64, DrawCensusEntry> g_draw_census;
static u64 g_draw_census_total;

// ISS_ENGINE_MARKERS (native/engine_draw.cpp): the engine objects of the next draw, from the
// NOP markers its draw routines write. Statistics: how often a draw's (technique, material)
// and (technique, material, geometry) already appeared in this frame, by draw path.
struct EngineDrawIdentity {
    u64 technique{}, geometry{}, material{}, item{};
    u32 kind{};
    bool pending{};
};
static EngineDrawIdentity g_engine_draw;
static u64 g_engine_material{}, g_engine_material_technique{};
static u64 g_engine_frame_seen{};
std::atomic<u64> g_iss_frame{};
struct EngineDrawStats {
    std::array<u64, 5> draws{}, marked{}, material_seen{}, full_seen{};
    std::array<u64, 5> markers{};
    std::unordered_set<u64> material_keys, full_keys, techniques, materials, geometries;
    u64 frames{}, unique_material_keys{}, unique_full_keys{}, unique_techniques{},
        unique_materials{}, unique_geometries{};
    u64 total{};
};
static EngineDrawStats g_engine_stats;
void EngineMarker(const u32* payload) {
    const u32 kind = payload[0] & 0xFF;
    u64 v[4];
    std::memcpy(v, payload + 1, sizeof(v));
    static const bool technique_census = [] {
        const char* value = std::getenv("ISS_ENGINE_TECHNIQUE_CENSUS");
        return value && value[0] == '1';
    }();
    if (technique_census && v[0]) {
        static std::unordered_map<u64, u64> last_vtable;
        const u64 vtable = *reinterpret_cast<const u64*>(v[0]) - IssImageBase();
        auto [it, inserted] = last_vtable.try_emplace(v[0], vtable);
        if (inserted || it->second != vtable) {
            it->second = vtable;
            std::fprintf(stderr, "TECHNIQUE_MAP object=0x%llx vtable=0x%llx kind=%u\n",
                         (unsigned long long)v[0], (unsigned long long)vtable, kind);
        }
    }
    auto& s = g_engine_stats;
    ++s.markers[std::min<u32>(kind, 4)];
    if (kind == 4) {
        g_engine_material_technique = v[0];
        g_engine_material = v[1];
        return;
    }
    auto& d = g_engine_draw;
    d.kind = kind;
    d.technique = v[0];
    d.geometry = v[1];
    d.material = kind == 1 ? v[2] : g_engine_material;
    d.item = kind == 1 ? v[3] : 0;
    d.pending = true;
}
static u64 EngineKey(u64 a, u64 b, u64 c) {
    return (a * 0x9E3779B97F4A7C15ull) ^ (b * 0xC2B2AE3D27D4EB4Full) ^ (c * 0x165667B19E3779F9ull);
}
static bool EngineStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_ENGINE_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
static void NoteEngineDraw(u32 path) {
    auto& s = g_engine_stats;
    auto& d = g_engine_draw;
    if (!EngineStatsEnabled()) {
        d.pending = false;
        return;
    }
    if (const u64 frame = g_iss_frame.load(std::memory_order_relaxed);
        frame != g_engine_frame_seen) {
        g_engine_frame_seen = frame;
        ++s.frames;
        s.unique_material_keys += s.material_keys.size();
        s.unique_full_keys += s.full_keys.size();
        s.unique_techniques += s.techniques.size();
        s.unique_materials += s.materials.size();
        s.unique_geometries += s.geometries.size();
        s.material_keys.clear();
        s.full_keys.clear();
        s.techniques.clear();
        s.materials.clear();
        s.geometries.clear();
    }
    ++s.draws[path];
    if (d.pending) {
        d.pending = false;
        ++s.marked[path];
        const u64 mk = EngineKey(d.technique, d.material, d.kind);
        const u64 fk = EngineKey(mk, d.geometry, 1);
        s.material_seen[path] += s.material_keys.insert(mk).second ? 0 : 1;
        s.full_seen[path] += s.full_keys.insert(fk).second ? 0 : 1;
        s.techniques.insert(d.technique);
        s.materials.insert(d.material);
        s.geometries.insert(d.geometry);
    }
    if ((++s.total & ((1u << 20) - 1)) != 0) {
        return;
    }
    const double f = s.frames ? double(s.frames) : 1.0;
    std::fprintf(stderr,
                 "ENGINE_DRAWS markers[item=%llu inst=%llu depth=%llu material=%llu] per_frame "
                 "techniques=%.0f materials=%.0f geometries=%.0f material_keys=%.0f full_keys=%.0f\n",
                 (unsigned long long)s.markers[1], (unsigned long long)s.markers[2],
                 (unsigned long long)s.markers[3], (unsigned long long)s.markers[4],
                 s.unique_techniques / f, s.unique_materials / f, s.unique_geometries / f,
                 s.unique_material_keys / f, s.unique_full_keys / f);
    static const char* const names[] = {"plain", "lean", "rebind", "switch", "full"};
    for (u32 p = 0; p < 5; ++p) {
        std::fprintf(stderr,
                     "ENGINE_DRAWS path=%s draws=%llu marked=%llu material_seen=%llu "
                     "full_seen=%llu\n",
                     names[p], (unsigned long long)s.draws[p], (unsigned long long)s.marked[p],
                     (unsigned long long)s.material_seen[p], (unsigned long long)s.full_seen[p]);
    }
}
// ISS_ENGINE_FAST=1: fast draws follow the engine's draws (see FastDrawState::engine_key).
static bool EngineFastEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_ENGINE_FAST");
        return value && value[0] == '1';
    }();
    return enabled;
}
static u64 CurrentEngineKey() {
    const auto& d = g_engine_draw;
    if (!EngineFastEnabled() || !d.pending || d.material == 0) {
        return 0;
    }
    return EngineKey(d.technique, d.material, d.kind) | 1;
}

// ISS_PROGRAM_CENSUS=1: draws of the depth passes (depth attachment, at most one color) by vertex
// and pixel program, with their resource shape and what the previous draw had (another program,
// or the same programs with another pipeline): the families a native depth path could take.
static bool ProgramCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_PROGRAM_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
struct ProgramCensusEntry {
    std::array<u64, 5> paths{};
    u64 vs_change{}, same_programs_new_pipeline{}, same_pipeline{};
    u32 width{}, height{}, colors{}, attributes{}, vs_buffers{}, vs_images{}, ps_buffers{},
        ps_images{}, fetch_size{};
    u64 vs_hash{}, ps_hash{};
    u64 instance212_draws{}, instance212_mask{};
};
static std::unordered_map<u64, ProgramCensusEntry> g_program_census;
static u64 g_program_census_total;
static void NoteProgramCensus(const RenderState& state, u32 path, const GraphicsPipeline* pipeline) {
    static const GraphicsPipeline* previous{};
    static u64 previous_programs{};
    static const bool gbuffer = [] {
        const char* p = std::getenv("ISS_PROGRAM_CENSUS_GBUFFER");
        return p && p[0] == '1';
    }();
    if (!state.depth_stencil_attachment.image_view ||
        (gbuffer ? state.num_color_attachments != 5 : state.num_color_attachments > 1)) {
        previous = pipeline;
        previous_programs = 0;
        return;
    }
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    const auto* vs = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Vertex)];
    const u64 vs_hash = vs ? vs->pgm_hash : 0, ps_hash = ps ? ps->pgm_hash : 0;
    const u64 programs = vs_hash << 32 | ps_hash;
    const u64 key = programs * 0x9E3779B97F4A7C15ull ^ (u64{state.width} | u64{state.height} << 16 |
                                                        u64{state.num_color_attachments} << 32);
    auto& e = g_program_census[key];
    if (e.vs_hash == 0 && e.ps_hash == 0) {
        e.vs_hash = vs_hash;
        e.ps_hash = ps_hash;
        e.width = state.width;
        e.height = state.height;
        e.colors = state.num_color_attachments;
        e.attributes = u32(pipeline->GetFetchShader().attributes.size());
        e.fetch_size = pipeline->GetFetchShader().size;
        e.vs_buffers = vs ? u32(vs->buffers.size()) : 0;
        e.vs_images = vs ? u32(vs->images.size()) : 0;
        e.ps_buffers = ps ? u32(ps->buffers.size()) : 0;
        e.ps_images = ps ? u32(ps->images.size()) : 0;
    }
    ++e.paths[path];
    if (gbuffer && vs) {
        u64 mask = 0;
        for (u32 i = 0; i < std::min<std::size_t>(vs->buffers.size(), 64); ++i)
            if (!vs->buffers[i].IsSpecial() && vs->buffers[i].GetSharp(*vs).GetStride() == 212)
                mask |= u64{1} << i;
        e.instance212_mask |= mask;
        e.instance212_draws += mask != 0;
    }
    if (pipeline == previous) ++e.same_pipeline;
    else if (programs == previous_programs) ++e.same_programs_new_pipeline;
    else ++e.vs_change;
    previous = pipeline;
    previous_programs = programs;
    if ((++g_program_census_total & ((1u << 18) - 1)) != 0) return;
    std::vector<std::pair<u64, u64>> top;
    for (const auto& [k, v] : g_program_census)
        top.emplace_back(v.paths[0] + v.paths[1] + v.paths[2] + v.paths[3] + v.paths[4], k);
    std::sort(top.rbegin(), top.rend());
    std::fprintf(stderr, "PROGRAM_CENSUS total=%llu entries=%zu\n",
                 (unsigned long long)g_program_census_total, top.size());
    for (size_t i = 0; i < std::min<size_t>(top.size(), 40); ++i) {
        const auto& v = g_program_census[top[i].second];
        std::fprintf(stderr,
                     "PROGRAM_CENSUS pass=%ux%u colors=%u vs=0x%08llx ps=0x%08llx share=%.2f%% "
                     "plain=%llu lean=%llu rebind=%llu switch=%llu full=%llu from_other=%llu "
                     "same_prog_new_pipe=%llu same_pipe=%llu attrs=%u fetch=%u vs_buf=%u vs_img=%u "
                     "ps_buf=%u ps_img=%u inst212_draws=%llu inst212_mask=0x%llx\n",
                     v.width, v.height, v.colors, (unsigned long long)v.vs_hash,
                     (unsigned long long)v.ps_hash,
                     100.0 * double(top[i].first) / double(g_program_census_total),
                     (unsigned long long)v.paths[0], (unsigned long long)v.paths[1],
                     (unsigned long long)v.paths[2], (unsigned long long)v.paths[3],
                     (unsigned long long)v.paths[4], (unsigned long long)v.vs_change,
                     (unsigned long long)v.same_programs_new_pipeline,
                     (unsigned long long)v.same_pipeline, v.attributes, v.fetch_size, v.vs_buffers,
                     v.vs_images, v.ps_buffers, v.ps_images, v.instance212_draws, v.instance212_mask);
    }
}

// ISS_PASS_COST=1: command processor time by render pass shape (the ISS_DRAW_CENSUS key). The
// command processor reads the time stamp counter before each graphics packet (PassCostTick);
// the time since the previous draw (state packets, waits, the draw itself) goes to the pass of
// the draw that ends it: "entry" when that draw starts another pass than the draw before it,
// "inside" when it continues the same pass. Packets that write or wait on memory are counted
// the same way. A gap longer than PassCostStallCycles (a suspended coroutine, the idle loop) is
// set apart as a stall. Draws are also counted by the engine marker before them
// (ISS_ENGINE_MARKERS) and the vtable of its technique: which engine routines fill each pass.
static bool PassCostEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_PASS_COST");
        return value && value[0] == '1';
    }();
    return enabled;
}
// 11: the idle time before a command buffer, 12: its setup (native passes wait for their
// recording there), 13: compute queue packets (the same thread processes them).
constexpr u32 PassCostEvents = 14;
constexpr u64 PassCostCompute = 1ull << 62;
constexpr u64 PassCostStallCycles = 400'000; // ~120 us
constexpr u64 PassCostWindowFrames = 1200;
struct PassCostEntry {
    u64 inside{}, entry{}, inside_flushes{}, entries{}, draws{};
    std::array<u64, PassCostEvents> inside_events{}, entry_events{};
};
struct PassMarkEntry {
    u64 pass{}, vtable{}, draws{};
    u32 kind{}, fresh{};
};
struct PassCostState {
    u64 last_tsc{}, pending{}, stall{}, stalls{};
    std::array<u64, PassCostEvents> stall_by_class{}, stalls_by_class{};
    std::array<u64, PassCostEvents> pending_events{};
    u64 draw_key{}, flushed_key{~0ull};
    std::unordered_map<u64, PassCostEntry> passes;
    std::unordered_map<u64, PassMarkEntry> marks;
    std::unordered_map<u64, u64> vtables; // technique -> vtable, eboot relative
    u64 window_frame{}, window_ns{}, window_tsc{};
};
static PassCostState g_pass_cost;

static u64 PassCostKey(const RenderState& state, const GraphicsPipeline* pipeline) {
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    return u64{state.width} | (u64{state.height} << 16) |
           (u64{state.num_color_attachments} << 32) |
           (u64{state.depth_stencil_attachment.image_view ? 1u : 0u} << 40) |
           (u64{ps ? 1u : 0u} << 41) | (u64{std::min<u32>(state.num_layers, 255)} << 48);
}

static void NotePassCostDraw(const RenderState& state, const GraphicsPipeline* pipeline) {
    auto& s = g_pass_cost;
    const u64 key = PassCostKey(state, pipeline);
    s.draw_key = key;
    ++s.passes[key].draws;
    const auto& d = g_engine_draw;
    u64 vtable = 0;
    if (d.technique) {
        auto [it, inserted] = s.vtables.try_emplace(d.technique, 0);
        if (inserted) {
            it->second = *reinterpret_cast<const u64*>(d.technique) - IssImageBase();
        }
        vtable = it->second;
    }
    const u32 kind = d.technique ? d.kind : 0;
    const u32 fresh = d.pending ? 1 : 0;
    const u64 mark = key * 0x9E3779B97F4A7C15ull ^ vtable * 0xC2B2AE3D27D4EB4Full ^
                     (u64{kind} << 1 | fresh);
    auto& m = s.marks[mark];
    m.pass = key;
    m.vtable = vtable;
    m.kind = kind;
    m.fresh = fresh;
    ++m.draws;
}

static void PassCostReport(u64 frame) {
    auto& s = g_pass_cost;
    const u64 now_ns = Common::Perf::NowNs(), now_tsc = __rdtsc();
    const u64 frames = frame - s.window_frame;
    const double ghz = double(now_tsc - s.window_tsc) / double(std::max<u64>(now_ns - s.window_ns, 1));
    const double ms = 1.0 / (ghz * 1e6 * double(frames)); // cycles -> ms per frame
    u64 total = 0;
    std::vector<std::pair<u64, u64>> top;
    for (const auto& [k, e] : s.passes) {
        total += e.inside + e.entry;
        top.emplace_back(e.inside + e.entry, k);
    }
    std::sort(top.rbegin(), top.rend());
    std::fprintf(stderr,
                 "PASS_COST window frames=%llu tsc_ghz=%.3f attributed_ms=%.3f stall_ms=%.3f "
                 "stalls=%.1f pending_ms=%.3f\n",
                 (unsigned long long)frames, ghz, double(total) * ms, double(s.stall) * ms,
                 double(s.stalls) / double(frames), double(s.pending) * ms);
    {
        // Stalls by the packet class they happened in (PassCostClass; 11 idle, 12 setup).
        static const char* const names[PassCostEvents] = {
            "other", "draw", "dispatch", "ew", "eos", "eop", "wrm", "wdata", "dma", "copy_sem",
            "acq", "idle", "setup", "compute"};
        std::string text;
        for (u32 i = 0; i < PassCostEvents; ++i) {
            if (!s.stalls_by_class[i]) continue;
            char item[64];
            std::snprintf(item, sizeof(item), " %s=%.3fms/%.1f", names[i],
                          double(s.stall_by_class[i]) * ms,
                          double(s.stalls_by_class[i]) / double(frames));
            text += item;
        }
        std::fprintf(stderr, "PASS_COST stalls_by_class%s\n", text.c_str());
        s.stall_by_class = {};
        s.stalls_by_class = {};
    }
    const auto events = [&](const std::array<u64, PassCostEvents>& e) {
        static thread_local char text[256];
        std::snprintf(text, sizeof(text), "ew=%.1f eos=%.1f eop=%.1f wrm=%.1f wdata=%.1f dma=%.1f "
                      "copy_sem=%.1f acq=%.1f", e[3] / double(frames), e[4] / double(frames),
                      e[5] / double(frames), e[6] / double(frames), e[7] / double(frames),
                      e[8] / double(frames), e[9] / double(frames), e[10] / double(frames));
        return text;
    };
    for (size_t i = 0; i < std::min<size_t>(top.size(), 24); ++i) {
        const u64 k = top[i].second;
        const auto& e = s.passes[k];
        if (k == PassCostCompute) {
            std::fprintf(stderr, "PASS_COST pass=dispatch ms=%.3f share=%.1f%% dispatches=%.1f "
                         "inside_ms=%.3f entry_ms=%.3f\n", double(e.inside + e.entry) * ms,
                         100.0 * double(e.inside + e.entry) / double(std::max<u64>(total, 1)),
                         double(e.inside_flushes + e.entries) / double(frames),
                         double(e.inside) * ms, double(e.entry) * ms);
            continue;
        }
        std::fprintf(stderr,
                     "PASS_COST pass=%ux%u layers=%u colors=%u depth=%u ps=%u ms=%.3f share=%.1f%% "
                     "inside_ms=%.3f entry_ms=%.3f entries=%.1f draws=%.1f draw_packets=%.1f\n",
                     unsigned(k & 0xFFFF), unsigned((k >> 16) & 0xFFFF), unsigned(k >> 48),
                     unsigned((k >> 32) & 0xFF), unsigned((k >> 40) & 1), unsigned((k >> 41) & 1),
                     double(e.inside + e.entry) * ms,
                     100.0 * double(e.inside + e.entry) / double(std::max<u64>(total, 1)),
                     double(e.inside) * ms, double(e.entry) * ms, double(e.entries) / double(frames),
                     double(e.draws) / double(frames),
                     double(e.inside_flushes + e.entries) / double(frames));
        std::fprintf(stderr, "PASS_COST   inside %s\n", events(e.inside_events));
        std::fprintf(stderr, "PASS_COST   entry  %s\n", events(e.entry_events));
    }
    std::vector<std::pair<u64, u64>> marks;
    for (const auto& [k, m] : s.marks) marks.emplace_back(m.draws, k);
    std::sort(marks.rbegin(), marks.rend());
    static const char* const kinds[] = {"none", "item", "instanced", "depth", "material", "item_end"};
    for (size_t i = 0; i < std::min<size_t>(marks.size(), 48); ++i) {
        const auto& m = s.marks[marks[i].second];
        std::fprintf(stderr, "PASS_MARK pass=%ux%u colors=%u depth=%u kind=%s fresh=%u "
                     "technique_vtable=0x%llx draws=%.1f\n",
                     unsigned(m.pass & 0xFFFF), unsigned((m.pass >> 16) & 0xFFFF),
                     unsigned((m.pass >> 32) & 0xFF), unsigned((m.pass >> 40) & 1),
                     kinds[std::min<u32>(m.kind, 5)], m.fresh, (unsigned long long)m.vtable,
                     double(m.draws) / double(frames));
    }
    s.passes.clear();
    s.marks.clear();
    s.stall = s.stalls = 0;
    s.window_frame = frame;
    s.window_ns = now_ns;
    s.window_tsc = now_tsc;
}

static void PassCostTick(u32 packet_class) {
    auto& s = g_pass_cost;
    const u64 now = __rdtsc();
    const u64 delta = now - s.last_tsc;
    s.last_tsc = now;
    if (packet_class == 13) {
        // Compute queue work: its own bucket (with its waits), never a pass's.
        s.stall_by_class[13] += delta;
        ++s.stalls_by_class[13];
    } else if (delta > PassCostStallCycles) {
        s.stall += delta;
        ++s.stalls;
        s.stall_by_class[std::min(packet_class, PassCostEvents - 1)] += delta;
        ++s.stalls_by_class[std::min(packet_class, PassCostEvents - 1)];
    } else {
        s.pending += delta;
    }
    if (packet_class >= 3 && packet_class <= 10) {
        ++s.pending_events[packet_class];
    }
    if (packet_class == 1 || packet_class == 2) {
        // A native depth run records many draws inside one packet without the census: they
        // stay in the pass of the last counted draw.
        const u64 key = packet_class == 2 ? PassCostCompute : s.draw_key;
        auto& e = s.passes[key];
        const bool entry = key != s.flushed_key;
        (entry ? e.entry : e.inside) += s.pending;
        ++(entry ? e.entries : e.inside_flushes);
        auto& events = entry ? e.entry_events : e.inside_events;
        for (u32 i = 3; i < PassCostEvents; ++i) {
            events[i] += s.pending_events[i];
            s.pending_events[i] = 0;
        }
        s.pending = 0;
        s.flushed_key = key;
    }
    const u64 frame = g_iss_frame.load(std::memory_order_relaxed);
    if (s.window_ns == 0) {
        s.window_frame = frame;
        s.window_ns = Common::Perf::NowNs();
        s.window_tsc = now;
    } else if (frame - s.window_frame >= PassCostWindowFrames) {
        PassCostReport(frame);
    }
}

// ISS_NATIVE_MAP_CENSUS=1: one bounded, ordered CP-side inventory of graphics target spans.
// Frames 1200..1202 are sampled after warmup. Addresses are guest addresses, not Vulkan handles.
struct NativeMapSpan {
    u64 frame{}, first{}, draws{}, technique_vtable{}, vs_hash{}, ps_hash{};
    std::array<u64, 8> color_addr{}, cmask_addr{}, fmask_addr{};
    std::array<u32, 8> color_format{}, color_samples{}, color_clear{};
    u64 depth_addr{}, depth_write_addr{}, htile_addr{};
    u32 depth_format{}, depth_samples{}, depth_clear{}, stencil_clear{};
    u32 width{}, height{}, layers{}, colors{};
};
static bool NativeMapCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_MAP_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
static void NoteNativeMapDraw(const RenderState& state, const GraphicsPipeline* pipeline,
                              const AmdGpu::Regs& regs) {
    if (!NativeMapCensusEnabled()) return;
    const u64 frame = g_iss_frame.load(std::memory_order_relaxed);
    if (frame < 1200 || frame > 1203) return;
    static std::vector<NativeMapSpan> spans;
    static u64 current_frame = ~0ull;
    static u64 ordinal{};
    if (current_frame != frame) {
        for (const auto& s : spans) {
            std::fprintf(stderr,
                         "NATIVE_MAP frame=%llu order=%llu draws=%llu size=%ux%u layers=%u "
                         "colors=%u depth=0x%llx depth_write=0x%llx htile=0x%llx "
                         "dfmt=%u dsamples=%u dclear=%u sclear=%u vtable=0x%llx "
                         "vs=0x%llx ps=0x%llx\n",
                         (unsigned long long)s.frame, (unsigned long long)s.first,
                         (unsigned long long)s.draws, s.width, s.height, s.layers, s.colors,
                         (unsigned long long)s.depth_addr, (unsigned long long)s.depth_write_addr,
                         (unsigned long long)s.htile_addr, s.depth_format, s.depth_samples,
                         s.depth_clear, s.stencil_clear, (unsigned long long)s.technique_vtable,
                         (unsigned long long)s.vs_hash, (unsigned long long)s.ps_hash);
            for (u32 i = 0; i < s.colors; ++i)
                std::fprintf(stderr,
                             "NATIVE_MAP_COLOR frame=%llu order=%llu slot=%u addr=0x%llx "
                             "fmt=%u samples=%u clear=%u cmask=0x%llx fmask=0x%llx\n",
                             (unsigned long long)s.frame, (unsigned long long)s.first, i,
                             (unsigned long long)s.color_addr[i], s.color_format[i],
                             s.color_samples[i], s.color_clear[i],
                             (unsigned long long)s.cmask_addr[i],
                             (unsigned long long)s.fmask_addr[i]);
        }
        spans.clear();
        current_frame = frame;
        ordinal = 0;
    }
    if (frame == 1203) return;
    NativeMapSpan next{};
    next.frame = frame;
    next.first = ordinal++;
    next.width = state.width;
    next.height = state.height;
    next.layers = state.num_layers;
    next.colors = state.num_color_attachments;
    for (u32 i = 0; i < next.colors; ++i) {
        const auto& cb = regs.color_buffers[i];
        next.color_addr[i] = cb.Address();
        next.cmask_addr[i] = cb.CmaskAddress();
        next.fmask_addr[i] = cb.FmaskAddress();
        next.color_format[i] = cb.info.raw;
        next.color_samples[i] = cb.NumSamples();
        next.color_clear[i] = state.color_attachments[i].is_clear;
    }
    next.depth_addr = regs.depth_buffer.DepthAddress();
    next.depth_write_addr = regs.depth_buffer.DepthWriteAddress();
    next.htile_addr = regs.depth_htile_data_base.GetAddress();
    next.depth_format = u32(regs.depth_buffer.z_info.format);
    next.depth_samples = regs.depth_buffer.NumSamples();
    next.depth_clear = state.depth_stencil_attachment.depth_clear;
    next.stencil_clear = state.depth_stencil_attachment.stencil_clear;
    const auto* vs = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Vertex)];
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    next.vs_hash = vs ? vs->pgm_hash : 0;
    next.ps_hash = ps ? ps->pgm_hash : 0;
    if (g_engine_draw.pending && g_engine_draw.technique)
        next.technique_vtable = *reinterpret_cast<const u64*>(g_engine_draw.technique) - IssImageBase();
    if (!spans.empty()) {
        auto& prev = spans.back();
        if (prev.width == next.width && prev.height == next.height &&
            prev.layers == next.layers && prev.colors == next.colors &&
            prev.color_addr == next.color_addr && prev.color_format == next.color_format &&
            prev.depth_addr == next.depth_addr && prev.depth_write_addr == next.depth_write_addr &&
            prev.depth_format == next.depth_format && prev.color_clear == next.color_clear &&
            prev.depth_clear == next.depth_clear && prev.stencil_clear == next.stencil_clear) {
            ++prev.draws;
            return;
        }
    }
    next.draws = 1;
    spans.push_back(next);
}

static void NoteDrawCensus(const RenderState& state, u32 path, const GraphicsPipeline* pipeline,
                           const AmdGpu::Regs& regs) {
    NoteNativeMapDraw(state, pipeline, regs);
    if (PassCostEnabled()) {
        NotePassCostDraw(state, pipeline);
    }
    NoteEngineDraw(path);
    if (ProgramCensusEnabled()) {
        NoteProgramCensus(state, path, pipeline);
    }
    if (!DrawCensusEnabled()) {
        return;
    }
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    const auto* vs = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Vertex)];
    const u64 key = u64{state.width} | (u64{state.height} << 16) |
                    (u64{state.num_color_attachments} << 32) |
                    (u64{state.depth_stencil_attachment.image_view ? 1u : 0u} << 40) |
                    (u64{ps ? 1u : 0u} << 41) | (u64{std::min<u32>(state.num_layers, 255)} << 48);
    auto& entry = g_draw_census[key];
    ++entry.paths[path];
    entry.vs_hash = vs ? vs->pgm_hash : 0;
    entry.ps_hash = ps ? ps->pgm_hash : 0;
    if ((++g_draw_census_total & ((1u << 21) - 1)) != 0) {
        return;
    }
    std::vector<std::pair<u64, u64>> top;
    for (const auto& [k, e] : g_draw_census) {
        top.emplace_back(e.paths[0] + e.paths[1] + e.paths[2] + e.paths[3] + e.paths[4], k);
    }
    std::sort(top.rbegin(), top.rend());
    std::fprintf(stderr, "DRAW_CENSUS total=%llu passes=%zu\n",
                 (unsigned long long)g_draw_census_total, top.size());
    for (size_t i = 0; i < std::min<size_t>(top.size(), 16); ++i) {
        const u64 k = top[i].second;
        const auto& e = g_draw_census[k];
        std::fprintf(stderr,
                     "DRAW_CENSUS pass=%ux%u layers=%u colors=%u depth=%u ps=%u share=%.1f%% "
                     "plain=%llu lean=%llu rebind=%llu switch=%llu full=%llu last_vs=0x%llx last_ps=0x%llx\n",
                     unsigned(k & 0xFFFF), unsigned((k >> 16) & 0xFFFF), unsigned(k >> 48),
                     unsigned((k >> 32) & 0xFF), unsigned((k >> 40) & 1), unsigned((k >> 41) & 1),
                     100.0 * double(top[i].first) / double(g_draw_census_total),
                     (unsigned long long)e.paths[0], (unsigned long long)e.paths[1],
                     (unsigned long long)e.paths[2], (unsigned long long)e.paths[3],
                     (unsigned long long)e.paths[4], (unsigned long long)e.vs_hash,
                     (unsigned long long)e.ps_hash);
    }
}

static bool FastDrawEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_DRAW");
        return value && value[0] == '1';
    }();
    return enabled;
}

// Rebind (ISS_FAST_REBIND=1): the engine's mesh loop (EBOOT 0x1A2030 -> 0x1A1960) changes,
// between meshes of one material, only the fetch V# list, the per object constant buffer of the
// vertex stage and the index range. A fast draw whose vertex stage buffers only moved (same
// kind, other address/size) resolves the descriptors again (stage memo: the pixel stage hits)
// and pushes them, keeping pipeline, render targets, render state and dynamic state.
static std::array<u64, 3> FastDrawInfo(const vk::WriteDescriptorSet& write, u32 i) {
    if (write.pBufferInfo) {
        const auto& info = write.pBufferInfo[i];
        return {reinterpret_cast<u64>(static_cast<VkBuffer>(info.buffer)), info.offset,
                info.range};
    }
    const auto& info = write.pImageInfo[i];
    return {reinterpret_cast<u64>(static_cast<VkSampler>(info.sampler)),
            reinterpret_cast<u64>(static_cast<VkImageView>(info.imageView)),
            static_cast<u64>(info.imageLayout)};
}

void Rasterizer::RememberPushedWrites() {
    auto& f = fast_draw;
    f.writes_known = false;
    if (set_writes.size() > f.writes.size()) {
        return;
    }
    u32 info_index = 0;
    for (u32 w = 0; w < set_writes.size(); ++w) {
        const auto& write = set_writes[w];
        if (info_index + write.descriptorCount > f.infos.size() ||
            (!write.pBufferInfo && !write.pImageInfo)) {
            return;
        }
        f.writes[w] = {write.dstBinding, write.descriptorCount, u32(write.descriptorType),
                       info_index};
        for (u32 i = 0; i < write.descriptorCount; ++i) {
            f.infos[info_index++] = FastDrawInfo(write, i);
        }
    }
    f.num_writes = static_cast<u32>(set_writes.size());
    f.num_infos = info_index;
    f.writes_known = true;
}

int Rasterizer::ComparePushedWrites() const {
    const auto& f = fast_draw;
    if (!f.writes_known || f.num_writes != set_writes.size()) {
        return 2;
    }
    const u64 stream = reinterpret_cast<u64>(
        static_cast<VkBuffer>(const_cast<Rasterizer*>(this)->buffer_cache.GetStreamBuffer().Handle()));
    int result = 0;
    for (u32 w = 0; w < set_writes.size(); ++w) {
        const auto& write = set_writes[w];
        const auto& kept = f.writes[w];
        if (kept.binding != write.dstBinding || kept.count != write.descriptorCount ||
            kept.type != u32(write.descriptorType)) {
            return 2;
        }
        for (u32 i = 0; i < write.descriptorCount; ++i) {
            const auto a = f.infos[kept.first + i];
            const auto b = FastDrawInfo(write, i);
            if (a == b) {
                continue;
            }
            if (write.pBufferInfo && a[0] == stream && b[0] == stream && a[2] == b[2]) {
                result = 1;
                continue;
            }
            return 2;
        }
    }
    return result;
}

// Context register dwords that only feed Vulkan dynamic state (UpdateDepthStencilState,
// UpdatePrimitiveState), with the bits allowed to differ. PA_SU_SC_MODE_CNTL: cull front/back,
// front face, polygon offset enables (polygon mode and provoking vertex are pipeline key).
// DB_DEPTH_CONTROL: depth write, bounds, functions, backface (stencil/depth enables decide the
// depth attachment and its layout). DB_STENCILREFMASK front/back: all.
struct FastDynReg {
    u32 offset;
    u32 mask;
};
static constexpr std::array<FastDynReg, 4> kFastDynRegs{{
    {static_cast<u32>(offsetof(AmdGpu::Regs, polygon_control) / 4), 0x3807u},
    {static_cast<u32>(offsetof(AmdGpu::Regs, depth_control) / 4), 0x7007FCu},
    {static_cast<u32>(offsetof(AmdGpu::Regs, stencil_ref_front) / 4), 0xFFFFFFFFu},
    {static_cast<u32>(offsetof(AmdGpu::Regs, stencil_ref_back) / 4), 0xFFFFFFFFu},
}};

// Changed registers outside kFastDynRegs, by absolute dword offset. Free: only feed Vulkan
// dynamic state (scissors, viewport depth ranges, blend constants, stencil operations, polygon
// offset, depth bounds, line width), kept on the plain paths with ISS_FAST_DYN_EXT=1. Target:
// decide the attachments, their clears or sample counts (PrepareRenderState, BeginRendering),
// only the full path derives them. Other: the pipeline key, push constants or descriptors,
// all derived again by the switch path (ISS_FAST_STATE_SWITCH=1).
enum class FastRegClass : u8 { Other = 0, Free = 1, Target = 2 };
struct FastRegTable {
    static constexpr u32 First = AmdGpu::Regs::ContextRegWordOffset;
    std::array<u8, 0x400> context{};
    FastRegTable() {
        const auto mark = [&](size_t byte_offset, size_t bytes, FastRegClass c) {
            const u32 first = static_cast<u32>(byte_offset / 4) - First;
            for (u32 i = 0; i < (bytes + 3) / 4; ++i) {
                context[first + i] = static_cast<u8>(c);
            }
        };
#define ISS_FAST_REG(field, c)                                                                    \
    mark(offsetof(AmdGpu::Regs, field), sizeof(static_cast<AmdGpu::Regs*>(nullptr)->field),    \
         FastRegClass::c)
        // The depth block (and its padding: counters, sizes, result compare) up to the border
        // color base, then the exceptions.
        mark(size_t{First} * 4, offsetof(AmdGpu::Regs, ta_bc_base) - size_t{First} * 4,
             FastRegClass::Target);
        ISS_FAST_REG(depth_render_override, Other);
        ISS_FAST_REG(depth_bounds_min, Free);
        ISS_FAST_REG(depth_bounds_max, Free);
        ISS_FAST_REG(screen_scissor, Free);
        ISS_FAST_REG(window_offset, Free);
        ISS_FAST_REG(window_scissor, Free);
        ISS_FAST_REG(generic_scissor, Free);
        ISS_FAST_REG(viewport_scissors, Free);
        ISS_FAST_REG(viewport_depths, Free);
        ISS_FAST_REG(blend_constants, Free);
        ISS_FAST_REG(stencil_control, Free);
        ISS_FAST_REG(poly_offset, Free);
        ISS_FAST_REG(line_control, Free);
        ISS_FAST_REG(color_target_mask, Target);
        ISS_FAST_REG(color_control, Target);
        // Multisampling (configuration and sample locations) and the color buffers.
        mark(offsetof(AmdGpu::Regs, aa_config),
             offsetof(AmdGpu::Regs, color_buffers) - offsetof(AmdGpu::Regs, aa_config) +
                 sizeof(static_cast<AmdGpu::Regs*>(nullptr)->color_buffers),
             FastRegClass::Target);
#undef ISS_FAST_REG
    }
};

static FastRegClass ClassifyFastReg(u32 reg) {
    static const FastRegTable table;
    if (reg >= FastRegTable::First && reg < FastRegTable::First + table.context.size()) {
        return static_cast<FastRegClass>(table.context[reg - FastRegTable::First]);
    }
    // Draw parameters written as user config registers, taken from every draw.
    constexpr u32 primitive_type = offsetof(AmdGpu::Regs, primitive_type) / 4;
    constexpr u32 num_indices = offsetof(AmdGpu::Regs, num_indices) / 4;
    constexpr u32 num_instances = offsetof(AmdGpu::Regs, num_instances) / 4;
    if (reg == primitive_type || reg == num_indices || reg == num_instances) {
        return FastRegClass::Other;
    }
    return FastRegClass::Target;
}

static bool FastDynExtEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_DYN_EXT");
        return value && value[0] == '1';
    }();
    return enabled;
}

static bool FastStateSwitchEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_STATE_SWITCH");
        return value && value[0] == '1';
    }();
    return enabled;
}

static u64 g_fast_dyn_ext{};
// Changed register dwords kept per draw (Liverpool::changed_regs, a draw pipe packet).
static constexpr u32 FastChangedMax = 32;
static u64 g_fast_state_switch{};

// A change the switch path can take: no target register, the depth/stencil enables (which
// decide the depth attachment and its layout) unchanged, and no depth clear (which depth
// writes decide).
static bool FastStateSwitchAllowed(const AmdGpu::Regs& regs, const u32* changed, u32 n,
                                   const std::array<u32, 4>& snapshot) {
    if (n == 0 || n > FastChangedMax) {
        return false;
    }
    for (u32 i = 0; i < n; ++i) {
        if (ClassifyFastReg(changed[i]) == FastRegClass::Target) {
            return false;
        }
    }
    const auto& depth = kFastDynRegs[1];
    if (((regs.reg_array[depth.offset] ^ snapshot[1]) & ~depth.mask) != 0) {
        return false;
    }
    return !regs.depth_render_control.depth_clear_enable;
}

static bool FastDynamicOnlyChange(const AmdGpu::Regs& regs, const u32* changed, u32 n,
                                  const std::array<u32, 4>& snapshot) {
    if (n == 0 || n > FastChangedMax) {
        return false;
    }
    bool ext = false;
    for (u32 i = 0; i < n; ++i) {
        const u32 reg = changed[i];
        bool listed = false;
        for (const auto& dyn : kFastDynRegs) {
            listed |= dyn.offset == reg;
        }
        if (!listed && FastDynExtEnabled() && ClassifyFastReg(reg) == FastRegClass::Free) {
            listed = ext = true;
        }
        if (!listed) {
            return false;
        }
    }
    for (u32 i = 0; i < kFastDynRegs.size(); ++i) {
        if (((regs.reg_array[kFastDynRegs[i].offset] ^ snapshot[i]) & ~kFastDynRegs[i].mask) != 0) {
            return false;
        }
    }
    // With a depth clear enabled, depth writes also decide the render pass clear.
    if (regs.depth_render_control.depth_clear_enable) {
        return false;
    }
    g_fast_dyn_ext += ext ? 1 : 0;
    return true;
}

static void FastSnapshotDynRegs(const AmdGpu::Regs& regs, std::array<u32, 4>& snapshot) {
    for (u32 i = 0; i < kFastDynRegs.size(); ++i) {
        snapshot[i] = regs.reg_array[kFastDynRegs[i].offset];
    }
}

static bool FastRebindEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_REBIND");
        return value && value[0] == '1';
    }();
    return enabled;
}

// Lean rebind (ISS_FAST_LEAN=1): what the engine changes between meshes of one material is the
// per object constant buffer of the vertex stage (and its flattened user data copy). Instead of
// resolving every descriptor of both stages again, the descriptors of the previous draw stay in
// set_writes/buffer_infos (no BindResources since), only the vertex stage buffers that moved are
// resolved, patched in place and pushed alone (push descriptors update incrementally while the
// layout stays bound), with their push constant offsets.
static bool FastLeanEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_LEAN");
        return value && value[0] == '1';
    }();
    return enabled;
}

void Rasterizer::NoteLeanLayout(const GraphicsPipeline* pipeline) {
    auto& f = fast_draw;
    f.lean_ok = false;
    f.lean_ps_ok = false;
    f.bind_seq = bind_resources_seq;
    if (!FastLeanEnabled() || !f.writes_known || !pipeline->UsesPushDescriptors()) {
        return;
    }
    // BindResources writes the stages in order (fragment before vertex), each its buffers,
    // then its images and samplers.
    const auto stages = pipeline->GetStages();
    const auto* ps = stages[static_cast<u32>(Shader::SwStage::Fragment)];
    const auto* vs = stages[static_cast<u32>(Shader::SwStage::Vertex)];
    const u32 base = ps ? u32(ps->buffers.size() + ps->images.size() + ps->samplers.size()) : 0;
    const u32 binding_base = ps ? u32(ps->buffers.size()) : 0;
    if (!vs || base + vs->buffers.size() > set_writes.size() ||
        binding_base + vs->buffers.size() > push_data.buf_offsets.size()) {
        return;
    }
    for (u32 i = 0; i < vs->buffers.size(); ++i) {
        const auto& write = set_writes[base + i];
        if (!write.pBufferInfo || write.descriptorCount != 1 ||
            (write.descriptorType != vk::DescriptorType::eUniformBuffer &&
             write.descriptorType != vk::DescriptorType::eStorageBuffer)) {
            return;
        }
    }
    f.vs_write_base = base;
    f.vs_binding_base = binding_base;
    f.lean_ok = true;
    // The pixel stage buffers are the first writes (bindings 0..).
    f.lean_ps_ok = false;
    if (ps && ps->buffers.size() <= set_writes.size()) {
        bool ok = true;
        for (u32 i = 0; i < ps->buffers.size(); ++i) {
            const auto& write = set_writes[i];
            ok &= write.pBufferInfo && write.descriptorCount == 1 &&
                  (write.descriptorType == vk::DescriptorType::eUniformBuffer ||
                   write.descriptorType == vk::DescriptorType::eStorageBuffer);
        }
        f.lean_ps_ok = ok;
    }
}

void Rasterizer::CapturePsSharps(const GraphicsPipeline* pipeline) {
    auto& f = fast_draw;
    f.ps_sharps_ok = false;
    f.engine_key = CurrentEngineKey();
    const auto* ps = pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
    if (!f.engine_key || !ps || ps->buffers.size() > FastDrawState::MaxSharps ||
        ps->images.size() > FastDrawState::MaxSharps ||
        ps->samplers.size() > FastDrawState::MaxSharps) {
        return;
    }
    f.num_ps_buffers = static_cast<u8>(ps->buffers.size());
    f.num_ps_images = static_cast<u8>(ps->images.size());
    f.num_ps_samplers = static_cast<u8>(ps->samplers.size());
    for (u32 i = 0; i < f.num_ps_buffers; ++i) {
        f.ps_vsharps[i] =
            ps->buffers[i].IsSpecial() ? AmdGpu::Buffer{} : ps->buffers[i].GetSharp(*ps);
    }
    for (u32 i = 0; i < f.num_ps_images; ++i) {
        f.ps_tsharps[i] = ps->images[i].GetSharp(*ps);
    }
    for (u32 i = 0; i < f.num_ps_samplers; ++i) {
        f.ps_ssharps[i] = ps->samplers[i].GetSharp(*ps);
    }
    f.ps_sharps_ok = true;
}

int Rasterizer::TryLeanRebind(const GraphicsPipeline* pipeline, Shader::Info& vs, u32 changed_vs,
                              const Shader::PushData* verify_push, u32* written,
                              u32& num_written, Shader::Info* ps, u32 changed_ps) {
    auto& f = fast_draw;
    num_written = 0;
    if (!f.lean_ok || !f.writes_known || vs.uses_dma || g_num_traced_ranges != 0 ||
        iss::hazard_probe::Enabled() || (!verify_push && f.bind_seq != bind_resources_seq)) {
        return -1;
    }
    if (ps && (verify_push || !f.lean_ps_ok || ps->uses_dma)) {
        return -1;
    }
    struct Patch {
        u32 write;
        u32 buffer;
        bool special;
        vk::DescriptorBufferInfo info;
        u32 adjust;
        const VideoCore::Buffer* bound;
        u64 offset;
        u32 size;
        VAddr address;
        VAddr old_address;
        u64 old_size;
    };
    boost::container::static_vector<Patch, FastDrawState::MaxSharps> patches;
    const u64 alignment = instance.StorageMinAlignment();
    auto& stream = buffer_cache.GetStreamBuffer();
    // The pixel stage of another draw of the same engine material: its flattened user data
    // again, and the buffers whose sharps moved (per object constants).
    for (u32 i = 0; ps && i < f.num_ps_buffers; ++i) {
        const auto& desc = ps->buffers[i];
        Patch p{.write = i, .buffer = i};
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                if (DrawRegs().clipper_control.user_clip_plane_enable != 0) {
                    return -1;
                }
                continue;
            }
            if (desc.buffer_type != Shader::BufferType::Flatbuf) {
                return -1;
            }
            const u32 ubo_size = ps->FlatUserData().size() * sizeof(u32);
            const u64 offset = stream.Copy(ps->FlatUserData().data(), ubo_size, alignment);
            p.special = true;
            p.info = vk::DescriptorBufferInfo{stream.Handle(), offset, ubo_size};
            patches.push_back(p);
            continue;
        }
        if (((changed_ps >> i) & 1) == 0) {
            continue;
        }
        const auto& old = f.ps_vsharps[i];
        p.old_address = old.base_address;
        p.old_size = old.base_address && old.GetSize()
                         ? memory->ClampRangeSize(old.base_address, old.GetSize())
                         : 0;
        const auto sharp = desc.GetSharp(*ps);
        if (sharp.base_address == 0 || sharp.GetSize() == 0) {
            p.info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        } else {
            const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                sharp.base_address, size, false, desc.is_formatted, nullptr);
            const u64 offset_aligned = Common::AlignDown(offset, alignment);
            p.adjust = static_cast<u32>(offset - offset_aligned);
            p.info = vk::DescriptorBufferInfo{buffer->Handle(), offset_aligned, size + p.adjust};
            if (buffer != &stream) {
                p.bound = buffer;
                p.offset = offset;
                p.size = static_cast<u32>(size);
                p.address = sharp.base_address;
            }
        }
        patches.push_back(p);
    }
    for (u32 i = 0; i < f.num_vs_buffers; ++i) {
        const auto& desc = vs.buffers[i];
        Patch p{.write = f.vs_write_base + i, .buffer = f.vs_binding_base + i};
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                if (DrawRegs().clipper_control.user_clip_plane_enable != 0) {
                    return -1;
                }
                continue;
            }
            if (desc.buffer_type != Shader::BufferType::Flatbuf) {
                return -1;
            }
            // This draw's flattened user data: another stream copy.
            const u32 ubo_size = vs.FlatUserData().size() * sizeof(u32);
            const u64 offset = verify_push ? 0 : stream.Copy(vs.FlatUserData().data(), ubo_size,
                                                             alignment);
            p.special = true;
            p.info = vk::DescriptorBufferInfo{stream.Handle(), offset, ubo_size};
            patches.push_back(p);
            continue;
        }
        if (((changed_vs >> i) & 1) == 0) {
            continue;
        }
        const auto& old = f.vs_vsharps[i];
        p.old_address = old.base_address;
        p.old_size = old.base_address && old.GetSize()
                         ? memory->ClampRangeSize(old.base_address, old.GetSize())
                         : 0;
        const auto sharp = desc.GetSharp(vs);
        if (sharp.base_address == 0 || sharp.GetSize() == 0) {
            p.info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        } else {
            const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                sharp.base_address, size, false, desc.is_formatted, nullptr);
            const u64 offset_aligned = Common::AlignDown(offset, alignment);
            p.adjust = static_cast<u32>(offset - offset_aligned);
            p.info = vk::DescriptorBufferInfo{buffer->Handle(), offset_aligned, size + p.adjust};
            if (buffer != &stream) {
                p.bound = buffer;
                p.offset = offset;
                p.size = static_cast<u32>(size);
                p.address = sharp.base_address;
            }
        }
        patches.push_back(p);
    }

    if (verify_push) {
        // set_writes/push_data hold the full BindResources result of this draw: the previous
        // draw's pushed writes with the patches must equal it (another stream copy of the
        // same size counts as equal), and so must the push constants.
        if (!f.writes_known || f.num_writes != set_writes.size()) {
            return 0;
        }
        const u64 stream_handle = reinterpret_cast<u64>(static_cast<VkBuffer>(stream.Handle()));
        Shader::PushData expect = *verify_push;
        for (u32 w = 0; w < set_writes.size(); ++w) {
            const auto& write = set_writes[w];
            const auto& kept = f.writes[w];
            if (kept.binding != write.dstBinding || kept.count != write.descriptorCount ||
                kept.type != u32(write.descriptorType)) {
                return 0;
            }
            for (u32 i = 0; i < write.descriptorCount; ++i) {
                auto a = f.infos[kept.first + i];
                for (const auto& p : patches) {
                    if (p.write == w) {
                        a = {reinterpret_cast<u64>(static_cast<VkBuffer>(p.info.buffer)),
                             p.info.offset, p.info.range};
                    }
                }
                const auto b = FastDrawInfo(write, i);
                if (a != b && !(write.pBufferInfo && a[0] == stream_handle &&
                                b[0] == stream_handle && a[2] == b[2])) {
                    return 0;
                }
            }
        }
        for (const auto& p : patches) {
            if (!p.special) {
                expect.buf_offsets[p.buffer] = p.adjust;
            }
        }
        return std::memcmp(&expect, &push_data, sizeof(expect)) == 0 ? 1 : 0;
    }

    // Apply: descriptors in place, push offsets, the buffers this draw reads.
    bound_buffers.assign(f.bound_buffers.begin(), f.bound_buffers.end());
    for (const auto& p : patches) {
        *const_cast<vk::DescriptorBufferInfo*>(set_writes[p.write].pBufferInfo) = p.info;
        written[num_written++] = p.write;
        if (p.special) {
            continue;
        }
        push_data.buf_offsets[p.buffer] = p.adjust;
        if (p.old_address != 0) {
            for (auto it = bound_buffers.begin(); it != bound_buffers.end(); ++it) {
                if (it->address == p.old_address && it->size == p.old_size) {
                    bound_buffers.erase(it);
                    break;
                }
            }
        }
        if (p.bound) {
            bound_buffers.push_back(BoundBuffer{p.bound, p.offset, p.size, false, p.address});
            needs_barrier |= runtime.IsBufferAccessed(p.bound, p.offset, p.size, false);
        }
    }
    return 1;
}

void Rasterizer::CaptureFastDraw(const GraphicsPipeline* pipeline, bool is_indexed,
                                 const RenderState& state) {
    auto& f = fast_draw;
    f.valid = false;
    f.captured = false;
    if (!FastDrawEnabled()) {
        return;
    }
    const auto reject = [&](u32 why) {
        ++fast_draw_stats.capture_rejects;
        ++fast_draw_stats.reject_why[why];
    };
    if (attachment_feedback_loop || occlusion.HasOpen() || GpuCrashDiag::Enabled() ||
        !StageMemoContextEligible()) {
        return reject(0);
    }
    const auto stages = pipeline->GetStages();
    const auto* vs = stages[static_cast<u32>(Shader::SwStage::Vertex)];
    const auto* ps = stages[static_cast<u32>(Shader::SwStage::Fragment)];
    for (u32 i = 0; i < stages.size(); ++i) {
        if (stages[i] && stages[i] != vs && stages[i] != ps) {
            return reject(1);
        }
    }
    if (!vs) {
        return reject(2);
    }
    const auto& regs = DrawRegs();
    // Descriptors fully defined by the user data and memory: no writes. Copies of the
    // flattened user data or clip planes into the stream buffer change every draw.
    bool force_rebind = false;
    for (const auto* stage : {vs, ps}) {
        if (!stage) {
            continue;
        }
        // Formatted (texel) buffers read only are allowed: their synchronization from GPU
        // modified images could only change through the targets drawn since (checked below).
        bool eligible = stage->buffers.size() <= StageMemo::MaxBuffers &&
                        stage->images.size() <= StageMemo::MaxImages &&
                        stage->samplers.size() <= StageMemo::MaxSamplers;
        for (const auto& desc : stage->buffers) {
            eligible &= !desc.is_written &&
                        (!desc.IsSpecial() || desc.buffer_type == Shader::BufferType::Flatbuf ||
                         desc.buffer_type == Shader::BufferType::ClipPlanes);
        }
        for (const auto& desc : stage->images) {
            eligible &= !desc.is_written && !desc.is_atomic;
        }
        if (!eligible) {
            // Detail (reasons 6/7/9/10 reuse unused slots): formatted, written, special.
            for (const auto& desc : stage->buffers) {
                if (desc.is_written) return reject(10);
                if (desc.is_formatted) return reject(6);
                if (desc.IsSpecial() && desc.buffer_type != Shader::BufferType::Flatbuf &&
                    desc.buffer_type != Shader::BufferType::ClipPlanes) return reject(7);
            }
            for (const auto& desc : stage->images) {
                if (desc.is_written || desc.is_atomic) return reject(9);
            }
            return reject(3);
        }
        for (const auto& desc : stage->buffers) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf ||
                (desc.buffer_type == Shader::BufferType::ClipPlanes &&
                 regs.clipper_control.user_clip_plane_enable != 0)) {
                // Copied into the stream buffer by every draw: only fast draws that resolve the
                // descriptors again (rebind) can follow.
                if (!FastRebindEnabled()) {
                    return reject(4);
                }
                force_rebind = true;
            }
        }
    }
    if (vs->buffers.size() > FastDrawState::MaxSharps ||
        vs->images.size() > FastDrawState::MaxSharps ||
        vs->samplers.size() > FastDrawState::MaxSharps) {
        return reject(5);
    }
    const auto& fetch = pipeline->GetFetchShader();
    if (fetch.attributes.size() > FastDrawState::MaxFetch) {
        return reject(6);
    }
    // A render pass that clears only clears at its first draw.
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        if (state.color_attachments[cb].is_clear) {
            return reject(7);
        }
    }
    // Depth and stencil clears that follow the registers are found again by every draw (and
    // only applied when the render pass begins); a consumed metadata clear is not.
    const auto& ds = state.depth_stencil_attachment;
    const bool regs_depth_clear = regs.depth_render_control.depth_clear_enable &&
                                  regs.depth_control.depth_enable &&
                                  regs.depth_control.depth_write_enable;
    // (is_clear shares its storage with has_depth/depth_clear/... for this attachment.)
    if (ds.depth_clear && !regs_depth_clear) {
        return reject(8);
    }
    // The targets PrepareRenderState bound (bound_images starts with them).
    f.targets.clear();
    const auto& key = pipeline->GetGraphicsKey();
    for (u32 cb = 0; cb < static_cast<u32>(std::bit_width(key.mrt_mask)); ++cb) {
        if (const auto image_id = cb_descs[cb].first) {
            f.targets.push_back(image_id);
        }
    }
    if (db_desc.first) {
        f.targets.push_back(db_desc.first);
    }
    if (f.targets.size() > bound_images.size() ||
        !std::equal(f.targets.begin(), f.targets.end(), bound_images.begin())) {
        // BeginRendering found an image again (rebind): not the simple case.
        return reject(9);
    }
    // A texel buffer read in the memory of a target this draw renders to: not replayed.
    for (const auto* stage : {vs, ps}) {
        if (!stage) {
            continue;
        }
        for (const auto& desc : stage->buffers) {
            if (!desc.is_formatted || desc.IsSpecial()) {
                continue;
            }
            const auto sharp = desc.GetSharp(*stage);
            const VAddr start = sharp.base_address;
            const VAddr end = start + sharp.GetSize();
            for (const auto image_id : f.targets) {
                const auto& info = texture_cache.GetImage(image_id).info;
                // SynchronizeMemoryFromImage only copies an image that starts at the texel
                // buffer's base (BufferCache asserts it): heap wide views are not affected.
                if (info.guest_address == start && start < end) {
                    static u32 printed = 0;
                    if (printed < 16) {
                        ++printed;
                        std::fprintf(stderr,
                                     "FAST_DRAW_TEXEL stage=0x%llx base=0x%llx size=0x%llx "
                                     "stride=%u records=%u rt=0x%llx rt_size=0x%llx\n",
                                     static_cast<unsigned long long>(stage->pgm_hash),
                                     static_cast<unsigned long long>(start),
                                     static_cast<unsigned long long>(end - start),
                                     unsigned(sharp.GetStride()), unsigned(sharp.num_records),
                                     static_cast<unsigned long long>(info.guest_address),
                                     static_cast<unsigned long long>(info.guest_size));
                    }
                    return reject(11);
                }
            }
        }
    }
    f.bound_buffers.assign(bound_buffers.begin(), bound_buffers.end());
    RememberPushedWrites();
    for (const auto& bound : f.bound_buffers) {
        if (bound.is_written) {
            return reject(10);
        }
    }
    f.num_vs_buffers = static_cast<u8>(vs->buffers.size());
    f.num_vs_images = static_cast<u8>(vs->images.size());
    f.num_vs_samplers = static_cast<u8>(vs->samplers.size());
    for (u32 i = 0; i < f.num_vs_buffers; ++i) {
        f.vs_vsharps[i] = vs->buffers[i].IsSpecial() ? AmdGpu::Buffer{}
                                                      : vs->buffers[i].GetSharp(*vs);
    }
    for (u32 i = 0; i < f.num_vs_images; ++i) {
        f.vs_tsharps[i] = vs->images[i].GetSharp(*vs);
    }
    for (u32 i = 0; i < f.num_vs_samplers; ++i) {
        f.vs_ssharps[i] = vs->samplers[i].GetSharp(*vs);
    }
    f.num_fetch = static_cast<u8>(fetch.attributes.size());
    for (u32 i = 0; i < f.num_fetch; ++i) {
        const auto sharp = fetch.attributes[i].GetSharp(*vs);
        std::array<u32, 4> words;
        std::memcpy(words.data(), &sharp, sizeof(words));
        f.fetch_formats[i] = {words[1] >> 12, words[3]};
    }
    std::memcpy(f.ps_program.data(), &regs.ps_program, f.ps_program.size());
    std::memcpy(f.vs_program.data(), &regs.vs_program, f.vs_program.size());
    f.stage_enable = regs.stage_enable.raw;
    FastSnapshotDynRegs(regs, f.dyn_regs);
    f.ps_scout_program = nullptr;
    if (ps && !OnDrawWorker()) {
        if (const auto* taken = pipeline_cache.TakenScoutStage(*ps);
            taken && taken->valid && taken->resolved) {
            f.ps_scout_program = taken->program;
            f.ps_perm = taken->perm_idx;
        }
    }
    f.index_type = regs.index_buffer_type.raw;
    f.num_instances = regs.num_instances.NumInstances();
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        f.cb_extents[cb] = DrawCbExtents()[cb].raw;
    }
    f.db_extent = DrawDbExtent().raw;
    f.pipeline = pipeline;
    f.force_rebind = force_rebind;
    f.indexed = is_indexed;
    f.state = state;
    f.push = push_data;
    NoteLeanLayout(pipeline);
    CapturePsSharps(pipeline);
    f.captured = true;
}

void Rasterizer::FinishFastDraw() {
    auto& f = fast_draw;
    if (!f.captured) {
        f.valid = false;
        return;
    }
    // After the draw and ResetBindings: what the next draw must find unchanged.
    f.context_changes = DrawContextChanges();
    f.tick = scheduler.CurrentTick();
    f.bind_epoch = scheduler.GetDynamicState().binds.epoch;
    f.flush_seq = runtime.FlushSeq();
    f.stamp = CaptureStageMemoStamp();
    f.valid = true;
}

// ISS_FAST_DRAW_REGS=1: which context/config/uconfig register dwords changed before the draws
// that were not fast because of a context change.
static std::unordered_map<u32, u64> g_fast_draw_changed_regs;
static u64 g_fast_draw_changed_overflow;

// Shader switch (ISS_FAST_SWITCH=1): the engine's mesh loop changes material (programs) between
// runs of meshes while the render targets and the context registers stay. The new pipeline is
// looked up and its descriptors resolved as DrawRecord does, but the render pass, the targets
// (PrepareRenderState/BeginRendering) of the previous draw are kept when the new pipeline
// renders to the same attachments (mrt mask and sample counts, the only key fields
// BeginRendering reads).
static bool FastSwitchEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_SWITCH");
        return value && value[0] == '1';
    }();
    return enabled;
}

// ISS_DRAW_CENSUS: on a shader switch, whether the new programs (vertex program without user
// data, pixel program address) were one of the last 16 pipelines drawn in this render pass
// (no context change since): what a per pipeline state cache would hit.
struct SwitchRevisit {
    std::array<std::array<u8, 64>, 16> keys{};
    std::array<u64, 16> context{};
    std::array<u64, 16> tick{};
    u32 next{};
    u64 checks{}, revisits{}, same_ps_ud{};
};
static SwitchRevisit g_switch_revisit;
static void NoteSwitchRevisit(const AmdGpu::Regs& regs, u64 context_changes, u64 tick,
                              const u8* from_vs, const u8* from_ps) {
    if (!DrawCensusEnabled()) {
        return;
    }
    auto& r = g_switch_revisit;
    std::array<u8, 64> key{};
    constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
    static_assert(program_regs * 2 <= 64);
    std::memcpy(key.data(), &regs.vs_program, program_regs);
    std::memcpy(key.data() + program_regs, &regs.ps_program, program_regs);
    // The pipeline being left is one of the recent ones too.
    std::array<u8, 64> from{};
    std::memcpy(from.data(), from_vs, program_regs);
    std::memcpy(from.data() + program_regs, from_ps, program_regs);
    bool known = false;
    for (u32 i = 0; i < 16; ++i) {
        known |= r.tick[i] == tick && r.context[i] == context_changes && r.keys[i] == from;
    }
    if (!known) {
        r.keys[r.next] = from;
        r.context[r.next] = context_changes;
        r.tick[r.next] = tick;
        r.next = (r.next + 1) & 15;
    }
    ++r.checks;
    for (u32 i = 0; i < 16; ++i) {
        if (r.tick[i] == tick && r.context[i] == context_changes && r.keys[i] == key) {
            ++r.revisits;
            break;
        }
    }
    r.keys[r.next] = key;
    r.context[r.next] = context_changes;
    r.tick[r.next] = tick;
    r.next = (r.next + 1) & 15;
    if ((r.checks & ((1u << 20) - 1)) == 0) {
        std::fprintf(stderr, "SWITCH_REVISIT checks=%llu revisits=%llu (%.1f%%)\n",
                     (unsigned long long)r.checks, (unsigned long long)r.revisits,
                     100.0 * double(r.revisits) / double(r.checks));
    }
}

// ISS_SWITCH_CYCLES=1: where a shader switch spends its cycles.
struct SwitchCycles {
    bool enabled = [] {
        const char* value = std::getenv("ISS_SWITCH_CYCLES");
        return value && value[0] == '1';
    }();
    std::array<u64, 8> cycles{};
    u64 count{}, last{};
    void Start() { last = __rdtsc(); }
    void Mark(u32 phase) {
        const u64 now = __rdtsc();
        cycles[phase] += now - last;
        last = now;
    }
    void Done() {
        if ((++count & 0xFFFF) != 0) {
            return;
        }
        static constexpr std::array names{"lookup", "bind_res", "vertex", "push_dyn",
                                          "record", "capture", "reset", "other"};
        std::string line = "SWITCH_CYCLES";
        for (u32 i = 0; i < cycles.size(); ++i) {
            line += fmt::format(" {}={:.0f}", names[i], double(cycles[i]) / double(count));
        }
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
    }
};
static SwitchCycles g_switch_cycles;

bool Rasterizer::FastSwitchDraw(bool is_indexed, u32 index_offset) {
    auto& f = fast_draw;
    auto& stats = fast_draw_stats;
    auto& sc = g_switch_cycles;
    if (sc.enabled) sc.Start();
    NoteSwitchRevisit(DrawRegs(), DrawContextChanges(), scheduler.CurrentTick(),
                      f.vs_program.data(), f.ps_program.data());
    const bool worker = OnDrawWorker();
    const u32 created_before = worker ? 0 : pipeline_cache.NewPipelineCount();
    const GraphicsPipeline* pipeline =
        worker ? pipe_pipeline : LookupNativeGraphicsPipeline();
    if (sc.enabled) sc.Mark(0);
    // ISS_SWITCH_DIFF=1: a switch that lands on the previous draw's pipeline, by what changed in
    // the program registers (address, settings, user data of each stage) and the pixel shader.
    static const bool switch_diff = [] {
        const char* value = std::getenv("ISS_SWITCH_DIFF");
        return value && value[0] == '1';
    }();
    if (switch_diff && !worker && pipeline) {
        static std::array<u64, 10> diff_counts{};
        static u64 diff_total{};
        const auto& r = DrawRegs();
        constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
        const auto* vs_now = reinterpret_cast<const u8*>(&r.vs_program);
        const auto* ps_now = reinterpret_cast<const u8*>(&r.ps_program);
        const bool same = pipeline == f.pipeline;
        const bool vs_addr = std::memcmp(vs_now, f.vs_program.data(), 8) != 0;
        const bool vs_set = std::memcmp(vs_now + 8, f.vs_program.data() + 8, program_regs - 8) != 0;
        const bool vs_ud = std::memcmp(vs_now + program_regs, f.vs_program.data() + program_regs,
                                       f.vs_program.size() - program_regs) != 0;
        const bool ps_addr = std::memcmp(ps_now, f.ps_program.data(), 8) != 0;
        const bool ps_set = std::memcmp(ps_now + 8, f.ps_program.data() + 8, program_regs - 8) != 0;
        const bool ps_ud = std::memcmp(ps_now + program_regs, f.ps_program.data() + program_regs,
                                       f.ps_program.size() - program_regs) != 0;
        ++diff_total;
        if (same) {
            ++diff_counts[0];
            diff_counts[1] += vs_addr;
            diff_counts[2] += vs_set;
            diff_counts[3] += vs_ud;
            diff_counts[4] += ps_addr;
            diff_counts[5] += ps_set;
            diff_counts[6] += ps_ud;
            diff_counts[7] += !vs_addr && !vs_set && !ps_addr && !ps_set;
            diff_counts[8] += !vs_addr && !vs_set && !ps_addr && !ps_set && !ps_ud;
        }
        if ((diff_total & 0x3FFFF) == 0) {
            std::fprintf(stderr,
                         "SWITCH_DIFF switches=%llu same_pipeline=%llu vs_addr=%llu vs_settings=%llu "
                         "vs_user=%llu ps_addr=%llu ps_settings=%llu ps_user=%llu only_user=%llu "
                         "only_vs_user=%llu\n",
                         (unsigned long long)diff_total, (unsigned long long)diff_counts[0],
                         (unsigned long long)diff_counts[1], (unsigned long long)diff_counts[2],
                         (unsigned long long)diff_counts[3], (unsigned long long)diff_counts[4],
                         (unsigned long long)diff_counts[5], (unsigned long long)diff_counts[6],
                         (unsigned long long)diff_counts[7], (unsigned long long)diff_counts[8]);
        }
    }
    if (!worker && pipeline && pipeline_cache.NewPipelineCount() != created_before) {
        // Diagnostic: which key fields differ from the previous draw's pipeline.
        static u32 printed = 0;
        if (printed < 48) {
            ++printed;
            const auto& a = f.pipeline->GetGraphicsKey();
            const auto& b = pipeline->GetGraphicsKey();
            std::string diff;
            const auto field = [&](const char* name, const void* x, const void* y, size_t n) {
                if (std::memcmp(x, y, n) != 0) {
                    diff += name;
                    diff += ' ';
                }
            };
            field("stage_hashes", &a.stage_hashes, &b.stage_hashes, sizeof(a.stage_hashes));
            field("vertex_formats", &a.vertex_buffer_formats, &b.vertex_buffer_formats,
                  sizeof(a.vertex_buffer_formats));
            field("num_color", &a.num_color_attachments, &b.num_color_attachments, 4);
            field("color_buffers", &a.color_buffers, &b.color_buffers, sizeof(a.color_buffers));
            field("blend", &a.blend_controls, &b.blend_controls, sizeof(a.blend_controls));
            field("write_masks", &a.write_masks, &b.write_masks, sizeof(a.write_masks));
            field("cb_shader_mask", &a.cb_shader_mask, &b.cb_shader_mask,
                  sizeof(a.cb_shader_mask));
            field("mrt", &a.mrt_mask, &b.mrt_mask, sizeof(a.mrt_mask));
            field("samples", &a.num_samples, &b.num_samples, 2);
            std::string vs_fmt;
            for (u32 i = 0; i < 4; ++i) {
                vs_fmt += fmt::format("{}/{} ", u32(a.vertex_buffer_formats[i]),
                                      u32(b.vertex_buffer_formats[i]));
            }
            std::string hashes;
            for (u32 i = 0; i < a.stage_hashes.size(); ++i) {
                if (a.stage_hashes[i] || b.stage_hashes[i]) {
                    hashes += fmt::format("{:x}/{:x} ", a.stage_hashes[i], b.stage_hashes[i]);
                }
            }
            const auto line = fmt::format("FAST_SWITCH_NEW diff=[{}] vfmt(old/new)={} hashes={}",
                                          diff, vs_fmt, hashes);
            std::fputs(line.c_str(), stderr);
            std::fputc(10, stderr);
        }
    }
    // A failure leaves the lookup to Draw (one lookup per draw).
    if (!worker) {
        fast_lookup = pipeline;
        fast_lookup_set = true;
    }
    const auto fail = [&] {
        ++stats.switch_fails;
        return false;
    };
    if (!pipeline || (!pipeline->IsReady() && SkipUnreadyDraws()) ||
        g_dump.frames_left != 0 ||
        iss::command_trace::enabled) {
        return fail();
    }
    const auto& key = pipeline->GetGraphicsKey();
    const auto& old_key = f.pipeline->GetGraphicsKey();
    if (key.mrt_mask != old_key.mrt_mask) {
        return fail();
    }
    for (u32 cb = 0; cb < static_cast<u32>(std::bit_width(key.mrt_mask)); ++cb) {
        if (key.color_samples[cb] != old_key.color_samples[cb]) {
            return fail();
        }
    }
    const auto stages = pipeline->GetStages();
    const auto* vs = stages[static_cast<u32>(Shader::SwStage::Vertex)];
    const auto* ps = stages[static_cast<u32>(Shader::SwStage::Fragment)];
    for (u32 i = 0; i < stages.size(); ++i) {
        if (stages[i] && stages[i] != vs && stages[i] != ps) {
            return fail();
        }
    }
    if (!vs) {
        return fail();
    }
    if (!worker) {
        fast_lookup_set = false;
    }

    // As DrawRecord, with the targets PrepareRenderState bound for the previous draw.
    f.valid = false;
    last_draw_pipeline = pipeline;
    for (const auto image_id : f.targets) {
        bound_images.emplace_back(image_id);
        texture_cache.GetImage(image_id).binding.is_target = 1u;
    }
    if (sc.enabled) sc.Mark(7);
    if (!BindResources(pipeline)) {
        return true;
    }
    if (is_indexed && native_visual_latch.material_verified) CompleteNativeVisualPlans(pipeline);
    if (sc.enabled) sc.Mark(1);
    RenderState state = f.state;
    bool feedback = false;
    for (const auto image_id : f.targets) {
        feedback |= texture_cache.GetImage(image_id).binding.is_bound != 0;
    }
    if (feedback) {
        // A target is also read: the full render state and its layouts.
        state = BeginRendering(pipeline);
    }
    const auto& regs = DrawRegs();
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }
    if (sc.enabled) sc.Mark(2);
    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    if (sc.enabled) sc.Mark(3);
    scheduler.BeginRendering(state);
    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch);
    const auto cmdbuf = scheduler.Recorder();
    BindGraphicsPipeline(pipeline);
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
    if (sc.enabled) sc.Mark(4);
    // The following draws compare with this one, as after DrawRecord.
    CaptureFastDraw(pipeline, is_indexed, state);
    if (sc.enabled) sc.Mark(5);
    ResetBindings(false);
    FinishFastDraw();
    if (sc.enabled) {
        sc.Mark(6);
        sc.Done();
    }
    ++stats.hits;
    ++stats.switches;
    NoteDrawCensus(state, 3, pipeline, DrawRegs());
    return true;
}

static bool ProgramAliasEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_PROGRAM_ALIAS");
        const bool on = value && value[0] == '1';
        std::fprintf(stderr, "PORT_TUNING fast_program_alias=%d\n", on ? 1 : 0);
        return on;
    }();
    return enabled;
}
static u64 g_program_alias{}, g_program_alias_rejects{}, g_program_alias_mismatches{};

static bool FastCacheEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_FAST_CACHE");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool Rasterizer::TryReenterPipeline(const AmdGpu::Regs& regs, u64 engine_key) {
    auto& f = fast_draw;
    constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
    if (!f.writes_known) {
        return false;
    }
    // The departing state is kept (one entry per programs, and engine material with
    // ISS_ENGINE_FAST), the target is looked for first.
    u32 slot = static_cast<u32>(fast_cache.size());
    int target = -1;
    for (u32 i = 0; i < fast_cache.size(); ++i) {
        const auto& e = fast_cache[i];
        if (!e.valid) {
            continue;
        }
        if (e.engine_key == f.engine_key &&
            std::memcmp(e.vs_program.data(), f.vs_program.data(), f.vs_program.size()) == 0 &&
            std::memcmp(e.ps_program.data(), f.ps_program.data(), program_regs) == 0) {
            slot = i;
            continue;
        }
        if (target < 0 && e.writes_known && e.pipeline && e.tick == f.tick &&
            e.context_changes == f.context_changes && e.stage_enable == f.stage_enable &&
            std::memcmp(e.vs_program.data(), &regs.vs_program, e.vs_program.size()) == 0 &&
            std::memcmp(e.ps_program.data(), &regs.ps_program,
                        engine_key ? program_regs : e.ps_program.size()) == 0 &&
            (!engine_key || (e.engine_key == engine_key && e.ps_sharps_ok)) &&
            e.state == f.state) {
            target = static_cast<int>(i);
        }
    }
    if (slot == fast_cache.size()) {
        slot = fast_cache_next;
        if (static_cast<int>(slot) == target) {
            slot = (slot + 1) % fast_cache.size();
        }
        fast_cache_next = (slot + 1) % fast_cache.size();
    }
    fast_cache[slot] = f;
    if (target < 0) {
        ++fast_draw_stats.reentry_misses;
        return false;
    }
    auto& e = fast_cache[target];
    // Nothing that could change a resolved descriptor happened since its last draw.
    u32 num_buffers = 0;
    u32 num_images = 0;
    for (u32 w = 0; w < e.num_writes; ++w) {
        const auto type = static_cast<vk::DescriptorType>(e.writes[w].type);
        const bool buffer = type == vk::DescriptorType::eStorageBuffer ||
                            type == vk::DescriptorType::eUniformBuffer;
        (buffer ? num_buffers : num_images) += e.writes[w].count;
    }
    if (CaptureStageMemoStamp() != e.stamp || num_buffers > buffer_infos.capacity() ||
        num_images > image_infos.capacity()) {
        e.valid = false;
        ++fast_draw_stats.reentry_misses;
        return false;
    }
    f = e;
    e.valid = false;
    // Its descriptor set again, from the values pushed by its last draw.
    PrepareDescriptorWrites(set_writes, f.num_writes, false);
    buffer_infos.clear();
    image_infos.clear();
    for (u32 w = 0; w < f.num_writes; ++w) {
        const auto& pw = f.writes[w];
        const auto type = static_cast<vk::DescriptorType>(pw.type);
        if (type == vk::DescriptorType::eStorageBuffer ||
            type == vk::DescriptorType::eUniformBuffer) {
            const auto& info = f.infos[pw.first];
            buffer_infos.emplace_back(vk::Buffer{reinterpret_cast<VkBuffer>(info[0])}, info[1],
                                      info[2]);
            WriteBufferDescriptor(set_writes[w], pw.binding, &buffer_infos.back());
            set_writes[w].descriptorType = type;
        } else {
            const u32 first = static_cast<u32>(image_infos.size());
            for (u32 i = 0; i < pw.count; ++i) {
                const auto& info = f.infos[pw.first + i];
                image_infos.emplace_back(vk::Sampler{reinterpret_cast<VkSampler>(info[0])},
                                         vk::ImageView{reinterpret_cast<VkImageView>(info[1])},
                                         static_cast<vk::ImageLayout>(info[2]));
            }
            WriteImageDescriptor(set_writes[w], pw.binding, pw.count, type, &image_infos[first]);
        }
    }
    push_data = f.push;
    f.bind_epoch = scheduler.GetDynamicState().binds.epoch;
    f.bind_seq = bind_resources_seq;
    ++fast_draw_stats.reentries;
    return true;
}

bool Rasterizer::TryFastDraw(bool is_indexed, u32 index_offset) {
    auto& f = fast_draw;
    auto& stats = fast_draw_stats;
    // A draw pipe worker (ISS_DRAW_PIPE_FAST) takes the draw's changed registers and pipeline
    // from its packet; the command processor's own lookup state is never written there.
    const bool worker = OnDrawWorker();
    if (!worker) {
        fast_lookup_set = false;
    }
    ++stats.checks;
    static const bool note_regs = [] {
        const char* value = std::getenv("ISS_FAST_DRAW_REGS");
        return value && value[0] == '1';
    }();
    // The register dwords changed since the previous draw: taken, then cleared.
    const u32* changed = worker ? pipe_changed_regs.data() : liverpool->changed_regs.data();
    const u32 num_changed = worker ? pipe_num_changed_regs : liverpool->num_changed_regs;
    if (!worker) {
        liverpool->note_changed_regs = true;
    }
    const bool context_changed = f.valid && DrawContextChanges() != f.context_changes;
    bool dyn_update = false;
    bool state_switch = false;
    if (context_changed) {
        dyn_update = FastDynamicOnlyChange(DrawRegs(), changed, num_changed, f.dyn_regs);
        if (!dyn_update && f.valid && FastStateSwitchEnabled() && FastSwitchEnabled()) {
            state_switch = FastStateSwitchAllowed(DrawRegs(), changed, num_changed, f.dyn_regs);
        }
        if (note_regs && !dyn_update && !state_switch) {
            const u32 n = std::min<u32>(num_changed, FastChangedMax);
            for (u32 i = 0; i < n; ++i) {
                ++g_fast_draw_changed_regs[changed[i]];
            }
            g_fast_draw_changed_overflow += num_changed - n;
        }
    }
    if (!worker) {
        liverpool->num_changed_regs = 0;
    }
    if (++stats.report >= 262144) {
        stats.report = 0;
        ReportFastDraw();
    }
    if (!f.valid) {
        ++stats.invalid;
        return false;
    }
    const auto& regs = DrawRegs();
    {
        u32 why = 0;
        if (context_changed && !dyn_update && !state_switch) {
            why = 1;
        } else if (scheduler.CurrentTick() != f.tick) {
            why = 2;
        } else if (scheduler.GetDynamicState().binds.epoch != f.bind_epoch) {
            why = 3;
        } else if (regs.stage_enable.raw != f.stage_enable) {
            // Index type, instance count and indexing only change the index binding and the
            // draw parameters, both taken from this draw.
            why = 4;
        }
        if (why != 0) {
            ++stats.state;
            ++stats.reject_why[11 + why];
            return false;
        }
    }
    bool reentry = false;
    const u64 engine_key = worker ? 0 : CurrentEngineKey();
    constexpr size_t engine_program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
    if (!state_switch && engine_key != 0) {
        if (engine_key != f.engine_key ||
            std::memcmp(&regs.ps_program, f.ps_program.data(), engine_program_regs) != 0 ||
            std::memcmp(&regs.vs_program, f.vs_program.data(), f.vs_program.size()) != 0) {
            reentry = TryReenterPipeline(regs, engine_key);
        }
    } else if (!state_switch && !worker && FastCacheEnabled() &&
               (std::memcmp(&regs.ps_program, f.ps_program.data(), f.ps_program.size()) != 0 ||
                std::memcmp(&regs.vs_program, f.vs_program.data(), f.vs_program.size()) != 0)) {
        reentry = TryReenterPipeline(regs, 0);
    }
    // The same engine material and programs: the pixel stage descriptors are compared (its
    // user data points to a table the engine copies again for every draw).
    bool engine_ps = false;
    u32 changed_ps = 0;
    if (!state_switch && engine_key != 0 && engine_key == f.engine_key && f.ps_sharps_ok &&
        f.lean_ps_ok &&
        FastLeanEnabled() &&
        std::memcmp(&regs.ps_program, f.ps_program.data(), f.ps_program.size()) != 0 &&
        std::memcmp(&regs.ps_program, f.ps_program.data(), engine_program_regs) == 0 &&
        std::memcmp(&regs.vs_program, f.vs_program.data(), f.vs_program.size()) == 0) {
        auto& ps = const_cast<Shader::Info&>(f.pipeline->GetStage(Shader::SwStage::Fragment));
        ps.RefreshFlatBuf();
        engine_ps = true;
        for (u32 i = 0; i < f.num_ps_buffers && engine_ps; ++i) {
            const auto& desc = ps.buffers[i];
            if (desc.IsSpecial()) {
                continue;
            }
            const auto sharp = desc.GetSharp(ps);
            if (std::memcmp(&sharp, &f.ps_vsharps[i], sizeof(sharp)) != 0) {
                std::array<u32, 4> now, kept;
                std::memcpy(now.data(), &sharp, sizeof(now));
                std::memcpy(kept.data(), &f.ps_vsharps[i], sizeof(kept));
                if ((now[1] >> 12) != (kept[1] >> 12) || now[3] != kept[3] || desc.is_formatted) {
                    engine_ps = false;
                }
                changed_ps |= 1u << i;
            }
        }
        for (u32 i = 0; i < f.num_ps_images && engine_ps; ++i) {
            const auto sharp = ps.images[i].GetSharp(ps);
            engine_ps = std::memcmp(&sharp, &f.ps_tsharps[i], sizeof(sharp)) == 0;
        }
        for (u32 i = 0; i < f.num_ps_samplers && engine_ps; ++i) {
            const auto sharp = ps.samplers[i].GetSharp(ps);
            engine_ps = std::memcmp(&sharp, &f.ps_ssharps[i], sizeof(sharp)) == 0;
        }
        if (engine_ps) {
            ++stats.engine_ps;
        } else {
            ++stats.engine_ps_fails;
            changed_ps = 0;
        }
    }
    // Another material of the same shaders: the same programs, other pixel stage user data, and
    // the scout entry of this draw resolved the pixel stage to the same permutation (so the
    // same pipeline). Its flattened user data is taken and its descriptors resolved again.
    const AmdGpu::ScoutStage* ps_scouted = nullptr;
    bool rebind_ps = false;
    {
        constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
        const auto* draw = worker ? nullptr : liverpool->CurrentScoutDraw();
        const auto* ps_stage = f.pipeline->GetStages()[static_cast<u32>(Shader::SwStage::Fragment)];
        if (!state_switch && !engine_ps && FastRebindEnabled() && f.ps_scout_program && draw &&
            ps_stage &&
            std::memcmp(&regs.ps_program, f.ps_program.data(), f.ps_program.size()) != 0 &&
            std::memcmp(&regs.ps_program, f.ps_program.data(), program_regs) == 0 &&
            std::memcmp(&regs.vs_program, f.vs_program.data(), f.vs_program.size()) == 0) {
            for (u32 i = 0; i < draw->num_stages; ++i) {
                const auto& scouted = draw->stages[i];
                if (scouted.sw_stage != static_cast<u8>(Shader::SwStage::Fragment)) {
                    continue;
                }
                if (scouted.valid && scouted.resolved && scouted.program == f.ps_scout_program &&
                    scouted.perm_idx == f.ps_perm && scouted.flat &&
                    scouted.flat_size == ps_stage->srt_info.flattened_bufsize_dw &&
                    scouted.pgm_base == regs.ps_program.Address<VAddr>() &&
                    std::memcmp(scouted.user_data.data(), regs.ps_program.user_data.data(),
                                sizeof(scouted.user_data)) == 0) {
                    ps_scouted = &scouted;
                    rebind_ps = true;
                }
                break;
            }
        }
    }
    // ISS_FAST_PROGRAM_ALIAS=1: the engine keeps copies of the same shader at several addresses
    // (measured: half of the shader switches landed on the previous draw's pipeline, most with
    // only the program addresses changed). Programs with the hash, settings and pixel stage user
    // data of the previous draw's resolve to its pipeline in the full lookup (keyed by the hash,
    // the rest of the key and the specialization are compared below as for any plain draw). The
    // stages take the new program bases, as that lookup sets them. Stages with buffers addressed
    // relative to the program base are left to the switch.
    bool aliased = false;
    if (!rebind_ps && !engine_ps && !state_switch && !worker && ProgramAliasEnabled()) {
        constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
        constexpr size_t address_bytes = 8;
        const auto* vs_now = reinterpret_cast<const u8*>(&regs.vs_program);
        const auto* ps_now = reinterpret_cast<const u8*>(&regs.ps_program);
        const bool vs_moved = std::memcmp(vs_now, f.vs_program.data(), address_bytes) != 0;
        const bool ps_moved = std::memcmp(ps_now, f.ps_program.data(), address_bytes) != 0;
        if ((vs_moved || ps_moved) &&
            std::memcmp(vs_now + address_bytes, f.vs_program.data() + address_bytes,
                        program_regs - address_bytes) == 0 &&
            std::memcmp(ps_now + address_bytes, f.ps_program.data() + address_bytes,
                        f.ps_program.size() - address_bytes) == 0) {
            const auto stages = f.pipeline->GetStages();
            auto* vs_info = const_cast<Shader::Info*>(stages[static_cast<u32>(Shader::SwStage::Vertex)]);
            auto* ps_info = const_cast<Shader::Info*>(stages[static_cast<u32>(Shader::SwStage::Fragment)]);
            const auto relative = [](const Shader::Info* info) {
                for (const auto& buffer : info->buffers) {
                    if (buffer.post_op == Shader::SharpFetchPostOp::OffsetByProgramBase) return true;
                }
                return false;
            };
            if (vs_info && !relative(vs_info) && (!ps_info || !relative(ps_info)) &&
                (!ps_moved || (ps_info && regs.ps_program.address != 0)) &&
                (!vs_moved || pipeline_cache.GraphicsProgramHash(regs.vs_program) == vs_info->pgm_hash) &&
                (!ps_moved || pipeline_cache.GraphicsProgramHash(regs.ps_program) == ps_info->pgm_hash)) {
                vs_info->pgm_base = regs.vs_program.Address<VAddr>();
                std::memcpy(f.vs_program.data(), vs_now, program_regs);
                if (ps_info) {
                    ps_info->pgm_base = regs.ps_program.Address<VAddr>();
                    std::memcpy(f.ps_program.data(), ps_now, address_bytes);
                }
                ++g_program_alias;
                aliased = true;
            } else {
                ++g_program_alias_rejects;
            }
        }
    }
    bool switch_pipeline = false;
    if (!rebind_ps && !engine_ps &&
        (std::memcmp(&regs.ps_program, f.ps_program.data(), f.ps_program.size()) != 0 ||
         std::memcmp(&regs.vs_program, f.vs_program.data(), f.vs_program.size()) != 0)) {
        ++stats.program;
        switch_pipeline = FastSwitchEnabled();
        // Statistics: the same programs with other pixel stage user data (another material
        // of the same shaders), or other programs.
        constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
        if (std::memcmp(&regs.ps_program, f.ps_program.data(), program_regs) == 0 &&
            std::memcmp(&regs.vs_program, f.vs_program.data(), program_regs) == 0) {
            ++stats.reject_why[11];
        } else {
            ++stats.reject_why[10];
        }
        if (!switch_pipeline) {
            return false;
        }
    }
    if (state_switch) {
        // Other render state on the same targets: the pipeline looked up again.
        switch_pipeline = true;
        ++g_fast_state_switch;
    }
    if (worker && !switch_pipeline && pipe_pipeline != f.pipeline) {
        // The worker reads only the stages its packet captured: those of the packet's
        // pipeline (looked up by the command processor for this draw).
        if (!FastSwitchEnabled()) {
            return false;
        }
        switch_pipeline = true;
    }
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (DrawCbExtents()[cb].raw != f.cb_extents[cb]) {
            ++stats.context;
            return false;
        }
    }
    if (DrawDbExtent().raw != f.db_extent) {
        ++stats.context;
        return false;
    }
    if (occlusion.HasOpen() || GpuCrashDiag::Enabled() || !StageMemoContextEligible()) {
        ++stats.occlusion;
        return false;
    }
    if (CaptureStageMemoStamp() != f.stamp) {
        ++stats.stamp;
        return false;
    }
    if (switch_pipeline) {
        return FastSwitchDraw(is_indexed, index_offset);
    }
    const auto* pipeline = f.pipeline;
    // The vertex stage reads this draw's user data: its flattened copy is refreshed (from the
    // scout entry of the draw when it has one, as the full lookup does), then its descriptors
    // and fetch formats must be those of the previous draw.
    auto& vs = const_cast<Shader::Info&>(pipeline->GetStage(Shader::SwStage::Vertex));
    bool flat_done = false;
    const AmdGpu::ScoutStage* vs_scouted = nullptr;
    if (worker) {
        // The packet's snapshot holds this draw's flattened user data (the stage itself is
        // the command processor's).
        flat_done = true;
    } else if (const auto* draw = liverpool->CurrentScoutDraw(); draw != nullptr) {
        for (u32 i = 0; i < draw->num_stages; ++i) {
            const auto& scouted = draw->stages[i];
            if (scouted.sw_stage != static_cast<u8>(Shader::SwStage::Vertex)) {
                continue;
            }
            if (scouted.valid && scouted.flat &&
                scouted.flat_size == vs.srt_info.flattened_bufsize_dw &&
                scouted.pgm_base == regs.vs_program.Address<VAddr>() &&
                std::memcmp(scouted.user_data.data(), regs.vs_program.user_data.data(),
                            sizeof(scouted.user_data)) == 0) {
                vs.flattened_ud_buf.assign(scouted.flat, scouted.flat + scouted.flat_size);
                flat_done = true;
                vs_scouted = &scouted;
            }
            break;
        }
    }
    if (!flat_done) {
        vs.RefreshFlatBuf();
    }
    bool rebind = rebind_ps || f.force_rebind || reentry || engine_ps;
    u32 changed_vs = 0;
    if (rebind_ps) {
        auto& ps = const_cast<Shader::Info&>(pipeline->GetStage(Shader::SwStage::Fragment));
        ps.flattened_ud_buf.assign(ps_scouted->flat, ps_scouted->flat + ps_scouted->flat_size);
    }
    for (u32 i = 0; i < f.num_vs_buffers; ++i) {
        const auto& desc = vs.buffers[i];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto sharp = desc.GetSharp(vs);
        if (std::memcmp(&sharp, &f.vs_vsharps[i], sizeof(sharp)) != 0) {
            // Another buffer of the same kind (per object constants): only its address and
            // size may differ for the same permutation; its descriptors are resolved again.
            std::array<u32, 4> now, kept;
            std::memcpy(now.data(), &sharp, sizeof(now));
            std::memcpy(kept.data(), &f.vs_vsharps[i], sizeof(kept));
            if ((now[1] >> 12) != (kept[1] >> 12) || now[3] != kept[3] || desc.is_formatted ||
                !FastRebindEnabled()) {
                ++stats.vertex;
                return false;
            }
            rebind = true;
            changed_vs |= 1u << i;
        }
    }
    for (u32 i = 0; i < f.num_vs_images; ++i) {
        const auto sharp = vs.images[i].GetSharp(vs);
        if (std::memcmp(&sharp, &f.vs_tsharps[i], sizeof(sharp)) != 0) {
            ++stats.vertex;
            return false;
        }
    }
    for (u32 i = 0; i < f.num_vs_samplers; ++i) {
        const auto sharp = vs.samplers[i].GetSharp(vs);
        if (std::memcmp(&sharp, &f.vs_ssharps[i], sizeof(sharp)) != 0) {
            ++stats.vertex;
            return false;
        }
    }
    const auto& fetch = pipeline->GetFetchShader();
    for (u32 i = 0; i < f.num_fetch; ++i) {
        const auto sharp = fetch.attributes[i].GetSharp(vs);
        std::array<u32, 4> words;
        std::memcpy(words.data(), &sharp, sizeof(words));
        if ((words[1] >> 12) != f.fetch_formats[i][0] || words[3] != f.fetch_formats[i][1]) {
            ++stats.fetch;
            return false;
        }
    }

    static const bool verify = [] {
        const char* value = std::getenv("ISS_FAST_DRAW_VERIFY");
        return value && value[0] == '1';
    }();
    if (verify) {
        // Run the full lookups this draw would take and compare their results with the state
        // the previous draw left: pipeline, descriptors, constants, render pass, no barrier.
        const u64 hazard_before = runtime.HazardSeq();
        const Shader::PushData verify_old_push = push_data;
        const char* what = nullptr;
        const auto* full = pipeline_cache.GetGraphicsPipeline();
        if (full != pipeline) {
            what = "pipeline";
        } else {
            PrepareRenderState(full);
            if (!BindResources(full)) {
                what = "bind";
            } else {
                const auto state = BeginRendering(full);
                const auto& binds = scheduler.GetDynamicState().binds;
                const auto stage_flags = AllGraphicsStageBits;
                if (!rebind && (needs_barrier || runtime.HazardSeq() != hazard_before)) {
                    what = "barrier";
                } else if (!(state == f.state) || attachment_feedback_loop) {
                    what = "render_state";
                } else if (rebind) {
                    // New vertex stage descriptors are pushed by the rebind; a lean rebind
                    // must find what the full resolution found.
                    u32 n = 0;
                    if (!rebind_ps && FastLeanEnabled() &&
                        TryLeanRebind(pipeline, vs, changed_vs, &verify_old_push, nullptr, n) == 0) {
                        what = "lean";
                    }
                } else if (!binds.MatchesPush(full->GetLayout(), stage_flags, 0,
                                              sizeof(push_data), &push_data)) {
                    what = "push";
                } else if (!rebind && ComparePushedWrites() != 0) {
                    if (ComparePushedWrites() == 1) {
                        ++stats.stream_equiv;
                    } else {
                        what = "pushed_writes";
                    }
                } else if (false) {
                    what = "descriptors";
                    // Detail: the first write whose kept descriptor differs.
                    const auto& kept = binds.descriptors[0];
                    const u64 stream = reinterpret_cast<u64>(
                        static_cast<VkBuffer>(buffer_cache.GetStreamBuffer().Handle()));
                    if (kept.num_writes != set_writes.size() || kept.layout != full->GetLayout()) {
                        std::fprintf(stderr, "FAST_DRAW_DESC shape kept=%u now=%zu layout=%d\n",
                                     kept.num_writes, set_writes.size(),
                                     int(kept.layout == full->GetLayout()));
                    } else {
                        u32 info_index = 0;
                        bool printed = false;
                        bool real = false;
                        for (u32 w = 0; w < set_writes.size(); ++w) {
                            const auto& write = set_writes[w];
                            for (u32 i = 0; i < write.descriptorCount; ++i, ++info_index) {
                                u64 a, b, c;
                                if (write.pBufferInfo) {
                                    a = reinterpret_cast<u64>(
                                        static_cast<VkBuffer>(write.pBufferInfo[i].buffer));
                                    b = write.pBufferInfo[i].offset;
                                    c = write.pBufferInfo[i].range;
                                } else {
                                    a = reinterpret_cast<u64>(
                                        static_cast<VkSampler>(write.pImageInfo[i].sampler));
                                    b = reinterpret_cast<u64>(
                                        static_cast<VkImageView>(write.pImageInfo[i].imageView));
                                    c = static_cast<u64>(write.pImageInfo[i].imageLayout);
                                }
                                const auto& k = kept.infos[info_index];
                                // Another stream copy of the same range (same command buffer
                                // and upload epoch, unchanged descriptor): the same bytes.
                                const bool stream_copy = k.a == stream && a == stream && k.c == c;
                                if ((k.a != a || k.b != b || k.c != c) && !stream_copy) {
                                    real = true;
                                }
                                if ((k.a != a || k.b != b || k.c != c) && !stream_copy &&
                                    !printed) {
                                    std::fprintf(stderr,
                                                 "FAST_DRAW_DESC write=%u binding=%u type=%d "
                                                 "kept=[%llx %llx %llx] now=[%llx %llx %llx] "
                                                 "kept_stream=%d now_stream=%d\n",
                                                 w, write.dstBinding, int(write.descriptorType),
                                                 (unsigned long long)k.a, (unsigned long long)k.b,
                                                 (unsigned long long)k.c, (unsigned long long)a,
                                                 (unsigned long long)b, (unsigned long long)c,
                                                 int(k.a == stream), int(a == stream));
                                    printed = true;
                                }
                            }
                        }
                        if (!real) {
                            what = nullptr;
                            ++stats.stream_equiv;
                        }
                    }
                } else if (!rebind && bound_buffers.size() != f.bound_buffers.size()) {
                    what = "bound_buffers";
                }
            }
            ResetBindings(false);
        }
        ++stats.verified;
        if (what) {
            ++stats.mismatches;
            g_program_alias_mismatches += aliased ? 1 : 0;
            if (stats.mismatches <= 64) {
                std::fprintf(stderr, "FAST_DRAW_MISMATCH what=%s vs=0x%llx rebind=%d\n", what,
                             static_cast<unsigned long long>(vs.pgm_hash), int(rebind));
            }
            f.valid = false;
            return false;
        }
    }

    // Same state: the render targets are bound as PrepareRenderState binds them, the geometry
    // of this draw is bound, and the draw is recorded in the render pass of the previous one.
    for (const auto image_id : f.targets) {
        bound_images.emplace_back(image_id);
        texture_cache.GetImage(image_id).binding.is_target = 1u;
    }
    RenderState state = f.state;
    bool feedback = false;
    if (rebind && !rebind_ps && !verify && FastLeanEnabled()) {
        std::array<u32, 64> written;
        u32 num_written = 0;
        auto* lean_ps = engine_ps ? &const_cast<Shader::Info&>(
                                        pipeline->GetStage(Shader::SwStage::Fragment))
                                  : nullptr;
        if (TryLeanRebind(pipeline, vs, changed_vs, nullptr, written.data(), num_written, lean_ps,
                          changed_ps) == 1) {
            if (reentry) {
                // Another pipeline was bound since: its whole set again.
                num_written = std::min<u32>(static_cast<u32>(set_writes.size()), 64);
                for (u32 i = 0; i < num_written; ++i) {
                    written[i] = i;
                }
            }
            keep_vertex_input = !reentry;
            BindVertexBuffers(pipeline);
            keep_vertex_input = false;
            if (is_indexed) {
                BindIndexBuffer(index_offset);
            }
            if (needs_barrier) {
                runtime.FlushBarriers();
            }
            const auto cmdbuf = scheduler.Recorder();
            cmdbuf.pushConstants(pipeline->GetLayout(), AllGraphicsStageBits, 0u,
                                 sizeof(push_data), &push_data);
            if (num_written != 0) {
                boost::container::static_vector<vk::WriteDescriptorSet, FastDrawState::MaxSharps>
                    writes;
                for (u32 i = 0; i < num_written; ++i) {
                    writes.push_back(set_writes[written[i]]);
                }
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics,
                                            pipeline->GetLayout(), 0,
                                            vk::ArrayProxy<const vk::WriteDescriptorSet>(
                                                static_cast<u32>(writes.size()), writes.data()));
            }
            if (dyn_update || reentry) {
                UpdateDynamicState(pipeline, is_indexed);
                ++stats.dyn_updates;
            } else if (iss::dlss::UpscaleWanted()) [[unlikely]] {
                UpdateColorBlendingState(pipeline);
            }
            scheduler.BeginRendering(state);
            const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
            BindGraphicsPipeline(pipeline);
            if (is_indexed) {
                cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                                   s32(vertex_offset), instance_offset);
            } else {
                cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                            instance_offset);
            }
            iss::feedback_join::Recorded();
            DebugState.IncDrawCall();
            // As the plain fast draw: targets released, the reads of the kept buffers recorded
            // again only after a barrier flush; the moved buffers' reads always.
            for (const auto image_id : bound_images) {
                texture_cache.GetImage(image_id).binding = {};
            }
            bound_images.clear();
            const bool flushed = runtime.FlushSeq() != f.flush_seq;
            for (u32 i = 0; i < f.num_vs_buffers; ++i) {
                if (((changed_vs >> i) & 1) != 0) {
                    f.vs_vsharps[i] = vs.buffers[i].GetSharp(vs);
                }
            }
            for (u32 i = 0; lean_ps && i < f.num_ps_buffers; ++i) {
                if (((changed_ps >> i) & 1) != 0) {
                    f.ps_vsharps[i] = lean_ps->buffers[i].GetSharp(*lean_ps);
                }
            }
            if (lean_ps) {
                std::memcpy(f.ps_program.data(), &regs.ps_program, f.ps_program.size());
            }
            for (const auto& bound : bound_buffers) {
                bool moved = false;
                for (u32 i = 0; i < f.num_vs_buffers && !flushed; ++i) {
                    moved |= ((changed_vs >> i) & 1) != 0 &&
                             bound.address == f.vs_vsharps[i].base_address;
                }
                for (u32 i = 0; lean_ps && i < f.num_ps_buffers && !flushed; ++i) {
                    moved |= ((changed_ps >> i) & 1) != 0 &&
                             bound.address == f.ps_vsharps[i].base_address;
                }
                if (flushed || moved) {
                    runtime.AccessBuffer(bound.buffer, bound.offset, bound.size,
                                         vk::PipelineStageFlagBits2::eAllGraphics,
                                         vk::AccessFlagBits2::eShaderRead);
                }
            }
            f.bound_buffers.assign(bound_buffers.begin(), bound_buffers.end());
            bound_buffers.clear();
            needs_barrier = false;
            for (u32 i = 0; i < num_written; ++i) {
                f.infos[f.writes[written[i]].first] = FastDrawInfo(set_writes[written[i]], 0);
            }
            f.context_changes = DrawContextChanges();
            FastSnapshotDynRegs(regs, f.dyn_regs);
            f.tick = scheduler.CurrentTick();
            f.bind_epoch = scheduler.GetDynamicState().binds.epoch;
            f.flush_seq = runtime.FlushSeq();
            f.stamp = CaptureStageMemoStamp();
            f.push = push_data;
            f.engine_key = engine_key;
            ++stats.hits;
            ++stats.rebinds;
            ++stats.leans;
            NoteDrawCensus(state, 1, pipeline, DrawRegs());
            return true;
        }
        ++stats.lean_fallbacks;
    }
    if (rebind) {
        // The scout entry of this draw (texture memo keys) or none: never the previous draw's.
        if (!worker) {
            pipeline_cache.SetTakenScout(static_cast<u32>(Shader::SwStage::Vertex), &vs,
                                         vs_scouted);
        }
        if (rebind_ps && !worker) {
            pipeline_cache.AddTakenScout(static_cast<u32>(Shader::SwStage::Fragment),
                                         &pipeline->GetStage(Shader::SwStage::Fragment),
                                         ps_scouted);
        }
        if (!BindResources(pipeline)) {
            // As DrawRecord: nothing drawn.
            f.valid = false;
            return true;
        }
        for (const auto image_id : f.targets) {
            feedback |= texture_cache.GetImage(image_id).binding.is_bound != 0;
        }
        if (feedback) {
            // A target is also read now: the full render state and its layouts.
            state = BeginRendering(pipeline);
        }
    }
    keep_vertex_input = !reentry;
    BindVertexBuffers(pipeline);
    keep_vertex_input = false;
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }
    if (needs_barrier) {
        runtime.FlushBarriers();
    }
    if (rebind) {
        pipeline->BindResources(set_writes, push_data);
    }
    if (feedback || dyn_update || reentry) {
        // Dynamic state registers changed (or a feedback loop): derived again from them.
        UpdateDynamicState(pipeline, is_indexed);
        ++stats.dyn_updates;
    } else if (iss::dlss::UpscaleWanted()) [[unlikely]] {
        UpdateColorBlendingState(pipeline);
    }
    scheduler.BeginRendering(state);
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    const auto cmdbuf = scheduler.Recorder();
    BindGraphicsPipeline(pipeline);
    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    iss::feedback_join::Recorded();
    DebugState.IncDrawCall();

    if (rebind) {
        // What later fast draws compare and replay: this draw's vertex descriptors.
        for (u32 i = 0; i < f.num_vs_buffers; ++i) {
            if (!vs.buffers[i].IsSpecial()) {
                f.vs_vsharps[i] = vs.buffers[i].GetSharp(vs);
            }
        }
        bool written = false;
        for (const auto& bound : bound_buffers) {
            written |= bound.is_written;
        }
        f.bound_buffers.assign(bound_buffers.begin(), bound_buffers.end());
        RememberPushedWrites();
        NoteLeanLayout(pipeline);
        CapturePsSharps(pipeline);
        f.push = push_data;
        ResetBindings(false);
        f.valid = !feedback && !written;
        f.context_changes = DrawContextChanges();
        FastSnapshotDynRegs(regs, f.dyn_regs);
        std::memcpy(f.ps_program.data(), &regs.ps_program, f.ps_program.size());
        f.tick = scheduler.CurrentTick();
        f.bind_epoch = scheduler.GetDynamicState().binds.epoch;
        f.flush_seq = runtime.FlushSeq();
        f.stamp = CaptureStageMemoStamp();
        ++stats.hits;
        ++stats.rebinds;
        stats.ps_rebinds += rebind_ps ? 1 : 0;
        NoteDrawCensus(state, 2, pipeline, DrawRegs());
        return true;
    }
    // ResetBindings: the target flags, and the reads of the descriptors' buffers again only
    // when a barrier flush forgot them since the previous draw recorded them.
    for (const auto image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    bound_images.clear();
    if (runtime.FlushSeq() != f.flush_seq) {
        for (const auto& bound : f.bound_buffers) {
            runtime.AccessBuffer(bound.buffer, bound.offset, bound.size,
                                 vk::PipelineStageFlagBits2::eAllGraphics,
                                 vk::AccessFlagBits2::eShaderRead);
        }
    }
    bound_buffers.clear();
    needs_barrier = false;
    f.context_changes = DrawContextChanges();
    FastSnapshotDynRegs(regs, f.dyn_regs);
    f.tick = scheduler.CurrentTick();
    f.bind_epoch = scheduler.GetDynamicState().binds.epoch;
    f.flush_seq = runtime.FlushSeq();
    f.stamp = CaptureStageMemoStamp();
    ++stats.hits;
    NoteDrawCensus(state, 0, pipeline, DrawRegs());
    return true;
}

void Rasterizer::ReportFastDraw() {
    const auto& s = fast_draw_stats;
    if (s.cyc_plain + s.cyc_rebind + s.cyc_fail != 0) {
        const u64 plain = s.hits - s.rebinds - s.switches;
        const u64 fails = s.checks - s.hits;
        std::string line = fmt::format(
            "FAST_DRAW_CYCLES plain={:.0f} rebind={:.0f} lean={:.0f} failed_check={:.0f} "
            "leans={} lean_fallbacks={} switch={:.0f} switches={} switch_fails={} "
            "reentries={} reentry_misses={} engine_ps={} engine_ps_fails={}",
            plain ? double(s.cyc_plain) / double(plain) : 0.0,
            s.rebinds > s.leans ? double(s.cyc_rebind) / double(s.rebinds - s.leans) : 0.0,
            s.leans ? double(s.cyc_lean) / double(s.leans) : 0.0,
            fails ? double(s.cyc_fail) / double(fails) : 0.0, s.leans, s.lean_fallbacks,
            s.switches ? double(s.cyc_switch) / double(s.switches) : 0.0, s.switches,
            s.switch_fails, s.reentries, s.reentry_misses, s.engine_ps, s.engine_ps_fails);
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
    }
    if (!g_fast_draw_changed_regs.empty()) {
        std::vector<std::pair<u64, u32>> top;
        for (const auto& [reg, count] : g_fast_draw_changed_regs) {
            top.emplace_back(count, reg);
        }
        std::sort(top.rbegin(), top.rend());
        std::string line = "FAST_DRAW_REGS";
        for (size_t i = 0; i < std::min<size_t>(top.size(), 32); ++i) {
            line += fmt::format(" {:#x}:{}", top[i].second, top[i].first);
        }
        line += fmt::format(" overflow:{}", g_fast_draw_changed_overflow);
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
    }
    std::fprintf(stderr,
                 "FAST_DRAW checks=%llu hits=%llu rebinds=%llu ps_rebinds=%llu dyn=%llu miss[invalid=%llu state=%llu program=%llu "
                 "context=%llu occlusion=%llu stamp=%llu vertex=%llu fetch=%llu] "
                 "capture_rejects=%llu verified=%llu mismatches=%llu\n",
                 (unsigned long long)s.checks, (unsigned long long)s.hits,
                 (unsigned long long)s.rebinds, (unsigned long long)s.ps_rebinds,
                 (unsigned long long)s.dyn_updates,
                 (unsigned long long)s.invalid, (unsigned long long)s.state,
                 (unsigned long long)s.program, (unsigned long long)s.context,
                 (unsigned long long)s.occlusion, (unsigned long long)s.stamp,
                 (unsigned long long)s.vertex, (unsigned long long)s.fetch,
                 (unsigned long long)s.capture_rejects, (unsigned long long)s.verified,
                 (unsigned long long)s.mismatches);
    if (FastStateSwitchEnabled() || FastDynExtEnabled()) {
        std::fprintf(stderr, "FAST_STATE_SWITCH taken=%llu dyn_ext=%llu\n",
                     (unsigned long long)g_fast_state_switch, (unsigned long long)g_fast_dyn_ext);
    }
    if (ProgramAliasEnabled()) {
        std::fprintf(stderr, "FAST_PROGRAM_ALIAS taken=%llu rejects=%llu mismatches=%llu\n",
                     (unsigned long long)g_program_alias, (unsigned long long)g_program_alias_rejects,
                     (unsigned long long)g_program_alias_mismatches);
    }
    std::fprintf(stderr, "FAST_DRAW_STATE stream_equiv=%llu context=%llu tick=%llu binds=%llu draw_regs=%llu\n",
                 (unsigned long long)s.stream_equiv, (unsigned long long)s.reject_why[12], (unsigned long long)s.reject_why[13],
                 (unsigned long long)s.reject_why[14], (unsigned long long)s.reject_why[15]);
    std::fprintf(stderr, "FAST_DRAW_REJECTS %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
                 (unsigned long long)s.reject_why[0], (unsigned long long)s.reject_why[1],
                 (unsigned long long)s.reject_why[2], (unsigned long long)s.reject_why[3],
                 (unsigned long long)s.reject_why[4], (unsigned long long)s.reject_why[5],
                 (unsigned long long)s.reject_why[6], (unsigned long long)s.reject_why[7],
                 (unsigned long long)s.reject_why[8], (unsigned long long)s.reject_why[9],
                 (unsigned long long)s.reject_why[10], (unsigned long long)s.reject_why[11]);
}
