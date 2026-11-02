// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, before native_gbuffer_rasterizer.inl.
//
// ISS_NATIVE_MATERIALS=1 (with ISS_NATIVE_GBUFFER=1): engine materials as prebuilt Vulkan
// templates (native renderer, step 1).
//
// The first draw of a material in a render state takes the full path (pipeline lookup, texture
// and sampler resolution, descriptor writes); what it resolved is kept as the material's
// template: the pipeline, the texture views with their layouts, the samplers, and the shape of
// the descriptor set. Later draws of the same material in the same state, in this frame or the
// following ones, only resolve their buffers (per draw instance records and constants) and push
// the template's descriptors: no pipeline lookup, no texture search.
//
// A template is found by
//  - the state key: the range's context registers (except those the range writes and the per
//    draw index/instance registers) and the effective values of the registers the range writes,
//    so the registers that select the pipeline and its fixed state are the same;
//  - the material hash: the bytes of every image and sampler descriptor the stages read, and
//    the kind of every buffer descriptor and vertex fetch (stride and format word, not address
//    or size), read through the stage user data of the draw.
// Templates are revalidated at every use: each texture by its generation (a guest write),
// identity (uid, backing, view) when the texture cache changed, and not being a target of the
// draw; the samplers by the sampler cache generation. A stale template is dropped and the draw
// takes the full path, which learns it again.

namespace NativeMaterial {
namespace ng = iss::native_gbuffer;

static bool Enabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_MATERIALS");
        return p && p[0] == '1';
    }();
    return enabled;
}

// ISS_NATIVE_GBUFFER_SETS=1: pipelines with more bindings than the device pushes (the long one
// color item path range, eboot technique 0xCD84F0) take the native paths too, with descriptor
// sets: the command processor's from the descriptor heap (Pipeline::BindResources), the engine
// threads' from their own pools (NativeRecorder::Worker).
static bool SetsEnabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_GBUFFER_SETS");
        return p && p[0] == '1';
    }();
    return enabled;
}

enum Special : u8 { Guest, Flat, ClipPlanes };
struct BufferSlot {
    u8 stage{};   ///< index in GetStages()
    u8 desc{};    ///< index in the stage's buffers
    u8 binding{}; ///< push data buffer offset slot
    u8 special{};
    u32 write{};
};

struct Stats {
    u64 hits{}, no_shape{}, no_template{}, stale{}, learned{}, rejected{}, report{};
};
static Stats& Counters() {
    static Stats stats; // command processor thread only
    return stats;
}

static u64 Mix(u64 a, u64 b) {
    const std::array<u64, 2> words{a, b};
    return XXH3_64bits(words.data(), sizeof(words));
}

// Words of the register file that change with every draw (index buffer, count, instances): not
// part of the state.
static void ClearDrawWords(const AmdGpu::Regs& regs, u32* copy, u32 first, u32 count) {
    const auto* base = reinterpret_cast<const u32*>(&regs);
    const auto clear = [&](const void* field, size_t bytes) {
        const u32 index = u32(reinterpret_cast<const u32*>(field) - base);
        for (u32 w = 0; w < (bytes + 3) / 4; ++w)
            if (index + w >= first && index + w < first + count) copy[index + w - first] = 0;
    };
    clear(&regs.index_buffer_type, sizeof(regs.index_buffer_type));
    clear(&regs.index_base_address, sizeof(regs.index_base_address));
    clear(&regs.max_index_size, sizeof(regs.max_index_size));
    clear(&regs.draw_initiator, sizeof(regs.draw_initiator));
    clear(&regs.num_indices, sizeof(regs.num_indices));
    clear(&regs.num_instances, sizeof(regs.num_instances));
}

// The context of a range: the graphics context registers and the vertex/pixel shader registers
// (without their user data), with the registers the range writes cleared (their values per draw
// enter the state key), and the primitive type.
static u64 ContextHash(const AmdGpu::Regs& regs, const std::vector<u32>& state_regs) {
    constexpr u32 CtxFirst = 0xA000, CtxCount = 0x400, ShFirst = 0x2C00, ShCount = 0x80;
    static thread_local std::array<u32, CtxCount + ShCount + 1> words;
    std::memcpy(words.data(), &regs.reg_array[CtxFirst], CtxCount * 4);
    std::memcpy(words.data() + CtxCount, &regs.reg_array[ShFirst], ShCount * 4);
    ClearDrawWords(regs, words.data(), CtxFirst, CtxCount);
    for (u32 k = 0; k < ng::UserWords; ++k) {
        words[CtxCount + (ng::PsUser0 - ShFirst) + k] = 0;
        words[CtxCount + (ng::VsUser0 - ShFirst) + k] = 0;
    }
    for (const u32 reg : state_regs) {
        if (reg >= CtxFirst && reg < CtxFirst + CtxCount) words[reg - CtxFirst] = 0;
        else if (reg >= ShFirst && reg < ShFirst + ShCount) words[CtxCount + reg - ShFirst] = 0;
    }
    // ISS_NATIVE_MATERIALS_LOOSE_TARGETS=1: the render target and depth addresses leave the
    // context (formats stay): a recording does not depend on them (the command processor
    // attaches the real targets and compares the formats), and alternating targets otherwise
    // make every engine thread's key one frame stale.
    static const bool loose_targets = [] {
        const char* p = std::getenv("ISS_NATIVE_MATERIALS_LOOSE_TARGETS");
        return p && p[0] == '1';
    }();
    if (loose_targets) {
        for (const u32 reg : {0x005u, 0x012u, 0x013u, 0x014u, 0x015u}) words[reg] = 0;
        for (u32 n = 0; n < 8; ++n)
            for (const u32 field : {0u, 7u, 9u}) words[0x318 + n * 15 + field] = 0;
    }
    words[CtxCount + ShCount] = u32(regs.primitive_type);
    return XXH3_64bits(words.data(), sizeof(words));
}

// A streaming hash without XXH3's state type (not exposed here): words gathered in a block,
// each full block hashed with the running hash as seed.
struct Hasher {
    std::array<u32, 512> block{};
    u32 size{};
    u64 seed{};
    void Put(const void* data, size_t bytes) {
        const auto* p = static_cast<const u8*>(data);
        while (bytes) {
            const size_t room = (block.size() - size) * 4;
            const size_t n = std::min(room, (bytes + 3) & ~size_t(3));
            const size_t copy = std::min(n, bytes);
            std::memset(reinterpret_cast<u8*>(&block[size]) + copy, 0, n - copy);
            std::memcpy(&block[size], p, copy);
            size += u32(n / 4);
            p += copy;
            bytes -= copy;
            if (size == block.size()) Flush();
        }
    }
    void Flush() {
        seed = XXH3_64bits_withSeed(block.data(), size * 4, seed);
        size = 0;
    }
    u64 Digest() {
        Flush();
        return seed;
    }
};

// The render state version of a draw: the range context and the values the range wrote.
static u64 StateKey(u64 context, const ng::Batch& b, u32 state) {
    const auto& st = b.states[state];
    Hasher h;
    h.Put(&context, sizeof(context));
    for (u32 k = 0; k < b.state_regs.size(); ++k) {
        if ((st.inherit >> k) & 1) continue;
        const std::array<u32, 2> pair{b.state_regs[k], b.state_values[st.first_value + k]};
        h.Put(pair.data(), sizeof(pair));
    }
    return h.Digest();
}

// What the pipeline permutation and the template's textures depend on, read through the stage
// user data as refreshed for the draw.
static u64 MaterialHash(const GraphicsPipeline& pipeline) {
    Hasher h;
    const auto kind = [&](const AmdGpu::Buffer& sharp) {
        std::array<u32, 4> w;
        std::memcpy(w.data(), &sharp, sizeof(w));
        const std::array<u32, 3> k{w[1] >> 12, w[3], u32(sharp.base_address != 0)};
        h.Put(k.data(), sizeof(k));
    };
    const auto stages = pipeline.GetStages();
    for (u32 s = 0; s < stages.size(); ++s) {
        const auto* stage = stages[s];
        if (!stage) continue;
        h.Put(&s, sizeof(s));
        for (const auto& desc : stage->buffers) {
            if (desc.IsSpecial()) {
                const u32 type = u32(desc.buffer_type);
                h.Put(&type, sizeof(type));
            } else {
                kind(desc.GetSharp(*stage));
            }
        }
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(*stage);
            h.Put(&sharp, sizeof(sharp));
        }
        for (const auto& desc : stage->samplers) {
            const auto sharp = desc.GetSharp(*stage);
            h.Put(&sharp, sizeof(sharp));
        }
    }
    const auto& fetch = pipeline.GetFetchShader();
    if (!fetch.attributes.empty()) {
        const auto& vs = pipeline.GetStage(Shader::SwStage::Vertex);
        for (const auto& attribute : fetch.attributes) kind(attribute.GetSharp(vs));
    }
    return h.Digest();
}

static void Refresh(const GraphicsPipeline& pipeline) {
    for (const auto* stage : pipeline.GetStages())
        if (stage) const_cast<Shader::Info*>(stage)->RefreshFlatBuf();
}
} // namespace NativeMaterial


// A learned material. Immutable once published (the engine threads of ISS_NATIVE_RECORDER read
// it while the command processor learns others), except the texture cache generation it was
// last validated at.
struct Rasterizer::NativeMaterialTemplate {
    const GraphicsPipeline* pipeline{};
    const GraphicsPipeline* shape{}; ///< the pipeline whose stages the material hash reads
    /// The descriptor writes: textures and samplers point into image_infos; the buffer writes
    /// (buffers[k].write) get the draw's buffer infos.
    Pipeline::DescriptorWrites writes;
    std::vector<vk::DescriptorImageInfo> image_infos;
    std::vector<NativeImageCapture> images; ///< one per texture descriptor
    std::vector<NativeMaterial::BufferSlot> buffers;
    Shader::PushData push{};                ///< without buffer offsets
    /// As the learning draw committed it, shared by the templates of the same state: a copy
    /// per template (with the scheduler's bind cache, ~15 KB) made learning and dropping
    /// templates slow and the table large.
    std::shared_ptr<const DynamicState> dynamic;
    /// The attachments of the rendering the learning draw was recorded in.
    std::array<vk::Format, AmdGpu::NUM_COLOR_BUFFERS> color_formats{};
    u32 num_colors{};
    vk::Format depth_format{}, stencil_format{};
    vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
    u64 samplers_generation{};
    mutable std::atomic<u64> cache_generation{};
    /// The learn clock (Tables::clock) at its last use: the least recently used are evicted.
    mutable std::atomic<u32> last_used{};
    u64 key{};
    bool has_samplers{};
};

namespace NativeMaterial {
using Template = Rasterizer::NativeMaterialTemplate;
struct Tables {
    /// Written by the command processor only (under the exclusive lock); the engine threads read
    /// under the shared lock, the command processor reads without it.
    std::shared_mutex mutex;
    std::unordered_map<u64, const GraphicsPipeline*> shapes; ///< state key -> first pipeline
    std::unordered_map<u64, std::shared_ptr<const Template>> templates;
    std::unordered_map<u64, std::shared_ptr<const DynamicState>> dynamics; ///< by state bytes
    std::atomic<u32> clock{}; ///< advanced by every learn
};
// The dynamic state values (the bind cache, last, left out).
constexpr size_t DynamicBytes = offsetof(DynamicState, binds);
static Tables& State() {
    static Tables tables;
    return tables;
}

// Marks a template used (any thread). The clock moves only when one is learned, so most uses
// find it current and write nothing.
static void Touch(const Template& t) {
    const u32 now = State().clock.load(std::memory_order_relaxed);
    if (t.last_used.load(std::memory_order_relaxed) != now)
        t.last_used.store(now, std::memory_order_relaxed);
}

// ISS_NATIVE_MATERIALS_MAX (default 1048576, ~2 KB each). At the old cap of 131072, reached after
// 8-10 minutes of play, every drop (of all the templates, or of the least recently used half)
// left the command processor 40-100 times slower for good: 1-3 FPS with the same draws
// (2026-10-31, RX 9060 XT). Why is not known; a session should not reach the cap, and past it
// the least recently used half is dropped (command processor, under the exclusive lock).
static size_t MaxTemplates() {
    static const size_t max = [] {
        const char* p = std::getenv("ISS_NATIVE_MATERIALS_MAX");
        const size_t value = p ? std::strtoull(p, nullptr, 10) : 0;
        return value >= 1024 ? value : size_t(1) << 20;
    }();
    return max;
}
static void EvictTemplates(Tables& tables) {
    const auto start = std::chrono::steady_clock::now();
    const u32 now = tables.clock.load(std::memory_order_relaxed);
    const auto age = [now](const Template& t) {
        return now - t.last_used.load(std::memory_order_relaxed);
    };
    std::vector<u32> ages;
    ages.reserve(tables.templates.size());
    u32 max_age = 0;
    for (const auto& [key, t] : tables.templates) {
        ages.push_back(age(*t));
        max_age = std::max(max_age, ages.back());
    }
    const size_t keep = MaxTemplates() / 2;
    auto nth = ages.begin() + (ages.size() - keep);
    std::nth_element(ages.begin(), nth, ages.end(), std::greater<u32>());
    const u32 cutoff = *nth; // the age of the oldest kept
    std::vector<std::shared_ptr<const Template>> dropped;
    dropped.reserve(ages.size() - keep);
    for (int pass = 0; pass < 2 && tables.templates.size() > keep; ++pass) {
        // First the older than the cutoff, then as many of its age as are still over.
        for (auto it = tables.templates.begin(); it != tables.templates.end();) {
            const u32 a = age(*it->second);
            if (a > cutoff || (pass == 1 && a == cutoff && tables.templates.size() > keep)) {
                dropped.push_back(std::move(it->second));
                it = tables.templates.erase(it);
            } else {
                ++it;
            }
        }
    }
    const size_t count = dropped.size();
    iss::DestroyInBackground(dropped);
    std::fprintf(stderr, "NATIVE_MATERIALS_EVICT dropped=%zu kept=%zu max_age=%u cutoff=%u ms=%.2f\n",
                 count, tables.templates.size(), max_age, cutoff,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count());
}

// The template's writes with the draw's buffer infos (one per buffer slot).
static void BuildWrites(const Template& t, const vk::DescriptorBufferInfo* buffer_infos,
                        Pipeline::DescriptorWrites& out) {
    out = t.writes;
    for (u32 k = 0; k < t.buffers.size(); ++k) out[t.buffers[k].write].pBufferInfo = &buffer_infos[k];
}

// Whether every texture of the template is still the one it was learned with (command
// processor). Nothing is changed.
template <typename T> // a material or compute template (images, cache_generation)
static bool TexturesValid(VideoCore::TextureCache& cache, const T& t, bool& stale) {
    stale = false;
    const u64 generation = cache.Generation();
    const bool same_cache = generation == t.cache_generation.load(std::memory_order_relaxed);
    for (const auto& cap : t.images) {
        if (!cap.id) continue;
        if (cache.ImageGeneration(cap.id) != cap.generation ||
            (!same_cache && !cache.IsImageAllocated(cap.id))) {
            stale = true;
            return false;
        }
        auto& image = cache.GetImage(cap.id);
        if (!same_cache) {
            bool view_alive = false;
            if (image.image_uid == cap.uid && image.backing == cap.backing)
                for (const auto view_id : image.backing->image_view_ids)
                    view_alive |= *cache.GetImageView(view_id).image_view == cap.view;
            if (!view_alive) {
                stale = true;
                return false;
            }
        }
        // A target of this draw read as a texture (feedback) takes the full path.
        if (image.binding.is_target || image.binding.needs_rebind) return false;
        // Contents the guest (or the GPU through the buffer cache) changed since the last upload,
        // or a range no longer tracked (guest writes there would go unseen): the full path
        // (FindTexture -> UpdateImage) refreshes it first. The pause menu streams its tiles in.
        if (True(image.flags & VideoCore::ImageFlagBits::Dirty) ||
            image.track_addr != image.info.guest_address ||
            image.track_addr_end != image.info.guest_address + image.info.guest_size)
            return false;
    }
    t.cache_generation.store(generation, std::memory_order_relaxed);
    return true;
}
} // namespace NativeMaterial

const Rasterizer::NativeMaterialTemplate* Rasterizer::NativeMaterialFind(u64 state_key) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::MaterialFind};
    using namespace NativeMaterial;
    auto& tables = State();
    auto& c = Counters();
    const auto shape_it = tables.shapes.find(state_key);
    if (shape_it == tables.shapes.end()) {
        ++c.no_shape;
        return nullptr;
    }
    const auto* shape = shape_it->second;
    Refresh(*shape);
    const auto it = tables.templates.find(Mix(state_key, MaterialHash(*shape)));
    if (it == tables.templates.end()) {
        ++c.no_template;
        return nullptr;
    }
    const auto& t = *it->second;
    Touch(t);
    if (t.pipeline != shape) Refresh(*t.pipeline);
    return &t;
}

const GraphicsPipeline* Rasterizer::NativeMaterialPipeline(const NativeMaterialTemplate& t) const {
    return t.pipeline;
}

void Rasterizer::NativeMaterialDrop(u64 key) {
    auto& tables = NativeMaterial::State();
    std::unique_lock lock{tables.mutex};
    tables.templates.erase(key);
}

bool Rasterizer::NativeMaterialBind(const NativeMaterialTemplate& t) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::MaterialBind};
    using namespace NativeMaterial;
    auto& c = Counters();
    if (!t.pipeline->IsReady() && SkipUnreadyDraws()) return false;
    const auto stale = [&] {
        ++c.stale;
        // Dropped: the full path learns the material again.
        NativeMaterialDrop(t.key);
        return false;
    };
    if (t.has_samplers && texture_cache.SamplersGeneration() != t.samplers_generation)
        return stale();
    for (const auto& slot : t.buffers)
        if (slot.special == ClipPlanes && DrawRegs().clipper_control.user_clip_plane_enable != 0)
            return false;
    // Every texture is checked before any is transitioned.
    bool is_stale = false;
    if (!TexturesValid(texture_cache, t, is_stale)) return is_stale ? stale() : false;
    const u64 tick = scheduler.CurrentTick();
    for (const auto& cap : t.images) {
        if (!cap.id) continue;
        auto& image = texture_cache.GetImage(cap.id);
        needs_barrier |= runtime.Transit(&image, cap.layout, vk::PipelineStageFlagBits2::eAllCommands,
                                         vk::AccessFlagBits2::eShaderRead, cap.range);
        image.usage.texture = 1u;
        image.binding.is_bound = 1u;
        image.tick_accessed_last = tick;
        bound_images.emplace_back(cap.id);
    }

    push_data = t.push;
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 alignment = instance.StorageMinAlignment();
    const bool reuse_check =
        Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    const auto stages = t.pipeline->GetStages();
    // The writes point at these until the caller pushes them (command processor thread only).
    static std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> infos;
    for (u32 k = 0; k < t.buffers.size(); ++k) {
        const auto& slot = t.buffers[k];
        auto& info = infos[k];
        const auto& stage = *stages[slot.stage];
        if (slot.special == Flat) {
            const u32 size = u32(stage.FlatUserData().size() * sizeof(u32));
            info = vk::DescriptorBufferInfo{stream.Handle(),
                                            stream.Copy(stage.FlatUserData().data(), size, alignment),
                                            size};
            continue;
        }
        if (slot.special == ClipPlanes) {
            info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
            continue;
        }
        const auto& desc = stage.buffers[slot.desc];
        const auto sharp = desc.GetSharp(stage);
        if (!sharp.base_address || !sharp.GetSize()) {
            info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
            continue;
        }
        const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
        bool known_unwritten = false;
        const auto [buffer, offset] =
            buffer_cache.ObtainBuffer(sharp.base_address, u32(size), false, desc.is_formatted,
                                      reuse_check ? &known_unwritten : nullptr);
        const u64 offset_aligned = Common::AlignDown(offset, alignment);
        const u64 adjust = offset - offset_aligned;
        push_data.AddOffset(slot.binding, u32(adjust));
        info = vk::DescriptorBufferInfo{buffer->Handle(), offset_aligned, size + adjust};
        if (buffer != &stream) {
            bound_buffers.emplace_back(buffer, offset, u32(size), false, sharp.base_address);
            if (known_unwritten) Common::Perf::Add(Common::Perf::BufferChecksReused);
            else needs_barrier |= Rt().IsBufferAccessed(buffer, offset, size, false);
        }
    }
    BuildWrites(t, infos.data(), set_writes);
    ++c.hits;
    return true;
}

void Rasterizer::NativeMaterialLearn(u64 state_key, const GraphicsPipeline* pipeline,
                                     const RenderState& render) {
    using namespace NativeMaterial;
    auto& c = Counters();
    auto* caps = native_image_capture;
    if (!caps || !pipeline) return;
    const auto reject = [&] { ++c.rejected; };
    if (DrawRegs().clipper_control.user_clip_plane_enable != 0 && !native_material_allow_clip)
        return reject();
    for (const auto& cap : *caps)
        if (cap.id && !cap.plain) return reject();

    auto t = std::make_shared<Template>();
    t->pipeline = pipeline;
    // The writes BindResources made, stage by stage: buffers, textures, samplers.
    const auto stages = pipeline->GetStages();
    u32 write = 0, binding = 0, num_images = 0, num_image_infos = 0;
    for (u32 s = 0; s < stages.size(); ++s) {
        const auto* stage = stages[s];
        if (!stage) continue;
        for (u32 k = 0; k < stage->buffers.size(); ++k) {
            const auto& desc = stage->buffers[k];
            if (write >= set_writes.size() || !set_writes[write].pBufferInfo ||
                set_writes[write].descriptorCount != 1)
                return reject();
            u8 special = Guest;
            if (desc.IsSpecial()) {
                if (desc.buffer_type == Shader::BufferType::Flatbuf) special = Flat;
                else if (desc.buffer_type == Shader::BufferType::ClipPlanes) special = ClipPlanes;
                else return reject();
            } else if (desc.is_written) {
                return reject();
            }
            t->buffers.push_back(BufferSlot{u8(s), u8(k), u8(binding++), special, write++});
        }
        for (u32 k = 0; k < stage->images.size(); ++k) {
            if (write >= set_writes.size() || !set_writes[write].pImageInfo ||
                set_writes[write].descriptorType != vk::DescriptorType::eSampledImage)
                return reject();
            num_images += set_writes[write].descriptorCount;
            num_image_infos += set_writes[write].descriptorCount;
            ++write;
        }
        for (u32 k = 0; k < stage->samplers.size(); ++k) {
            if (write >= set_writes.size() || !set_writes[write].pImageInfo ||
                set_writes[write].descriptorType != vk::DescriptorType::eSampler)
                return reject();
            num_image_infos += set_writes[write].descriptorCount;
            t->has_samplers = true;
            ++write;
        }
    }
    if (write != set_writes.size() || num_images != caps->size()) return reject();

    // Own copies of the writes and their texture and sampler infos.
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
    t->images = *caps;
    t->push = push_data;
    t->push.buf_offsets.fill(0);
    t->samplers_generation = texture_cache.SamplersGeneration();
    t->cache_generation = texture_cache.Generation();

    // The formats of the rendering, as Scheduler::BeginRendering attaches the views.
    t->num_colors = render.num_color_attachments;
    for (u32 cb = 0; cb < t->num_colors; ++cb) {
        const auto& [image_id, desc] = cb_descs[cb];
        t->color_formats[cb] = render.color_attachments[cb].image_view && image_id
                                   ? texture_cache.FindRenderTarget(image_id, desc).info.format
                                   : vk::Format::eUndefined;
    }
    const auto& depth = render.depth_stencil_attachment;
    if ((depth.has_depth || depth.has_stencil) && db_desc.first) {
        const auto format = texture_cache.FindDepthTarget(db_desc.first, db_desc.second).info.format;
        t->depth_format = depth.has_depth ? format : vk::Format::eUndefined;
        t->stencil_format = depth.has_stencil ? format : vk::Format::eUndefined;
    }
    t->samples = vk::SampleCountFlagBits(std::max<u32>(pipeline->GetGraphicsKey().num_samples, 1));

    auto& tables = State();
    const auto shape_it = tables.shapes.find(state_key);
    const auto* shape = shape_it != tables.shapes.end() ? shape_it->second : pipeline;
    // The shape's stages must locate the same descriptors as this pipeline's.
    const auto shape_stages = shape->GetStages();
    for (u32 s = 0; s < stages.size(); ++s) {
        const auto* a = stages[s];
        const auto* b = shape_stages[s];
        if (!a != !b) return reject();
        if (a && (a->buffers.size() != b->buffers.size() || a->images.size() != b->images.size() ||
                  a->samplers.size() != b->samplers.size()))
            return reject();
    }
    if (shape != pipeline) Refresh(*shape);
    t->shape = shape;
    t->key = Mix(state_key, MaterialHash(*shape));
    {
        std::unique_lock lock{tables.mutex};
        if (tables.templates.size() >= MaxTemplates()) EvictTemplates(tables);
        t->last_used.store(tables.clock.fetch_add(1, std::memory_order_relaxed) + 1,
                           std::memory_order_relaxed);
        const auto& live = scheduler.GetDynamicState();
        const u64 dynamic_hash = XXH3_64bits(&live, DynamicBytes);
        auto& shared = tables.dynamics[dynamic_hash];
        if (!shared || std::memcmp(shared.get(), &live, DynamicBytes) != 0) {
            if (tables.dynamics.size() > 8192) {
                iss::DestroyInBackground(tables.dynamics);
                tables.dynamics[dynamic_hash] = nullptr;
            }
            auto copy = std::make_shared<DynamicState>(live);
            copy->binds.Forget();
            tables.dynamics[dynamic_hash] = std::move(copy);
        }
        t->dynamic = tables.dynamics[dynamic_hash];
        tables.shapes.try_emplace(state_key, pipeline);
        tables.templates[t->key] = std::move(t);
    }
    ++c.learned;
}

static void NativeMaterialReport() {
    using namespace NativeMaterial;
    auto& c = Counters();
    std::fprintf(stderr,
                 "NATIVE_MATERIALS hits=%llu learned=%llu no_shape=%llu no_template=%llu "
                 "stale=%llu rejected=%llu templates=%zu shapes=%zu\n",
                 (unsigned long long)c.hits, (unsigned long long)c.learned,
                 (unsigned long long)c.no_shape, (unsigned long long)c.no_template,
                 (unsigned long long)c.stale, (unsigned long long)c.rejected,
                 State().templates.size(), State().shapes.size());
}
