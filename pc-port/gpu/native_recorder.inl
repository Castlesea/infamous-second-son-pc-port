// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_material_rasterizer.inl and before
// native_gbuffer_rasterizer.inl.
//
// ISS_NATIVE_RECORDER=1 (with ISS_NATIVE_GBUFFER=1 and ISS_NATIVE_MATERIALS=1): native renderer
// step 2. The engine thread that wrote a material range records its draws into a Vulkan
// secondary command buffer right away, in parallel with the other engine threads, from:
//  - the registers at the range start that its command buffer wrote, over the baseline the
//    command processor publishes (native_gbuffer.h, RangeStart);
//  - the material templates (native_material_rasterizer.inl): pipeline, textures, samplers,
//    descriptor set shape, dynamic state, attachment formats;
//  - the draw's buffers, resolved without the buffer cache: the imported heap
//    (ISS_UNIFIED_MEMORY) by address, the others in their published arena
//    (BufferCache::PeekPublishedArena, read only, any thread);
//  - the stage user data flattened by the stage's SRT walker into a thread local view
//    (Shader::Info::UdSnapshot), copied to a per thread upload ring.
// A range with a draw the recorder cannot take (no template, a buffer neither imported nor
// published, a state the tracking does not follow) is not recorded.
//
// The command processor, at the range marker, uses the recording only if everything it assumed
// still holds: the same range context, the same inherited user data, index type and instance
// count, every template's textures and samplers current, every arena range uploaded and still
// at the same place. Then it prepares the render targets, transitions the textures, orders the
// buffer reads after their writers, and executes the secondary in the rendering
// (VK_KHR_maintenance7 lets it run among commands recorded inline). Otherwise the recording is
// discarded and the range takes the command processor's paths as before.


namespace NativeRecorder {
namespace ng = iss::native_gbuffer;
using Template = Rasterizer::NativeMaterialTemplate;

constexpr u64 RingSize = 8ull << 20;

// What the engine threads use, set by the command processor before the first recording.
struct Globals {
    const Instance* instance{};
    Scheduler* scheduler{};
    VideoCore::BufferCache* buffer_cache{};
    const VideoCore::TextureCache* texture_cache{}; ///< NativeMaterial::HandlesAlive
    std::atomic<const VideoCore::Buffer*> unified{};
    std::atomic<VAddr> unified_base{};
    std::atomic<bool> ready{};
};
static Globals g;

struct ArenaRead {
    VAddr address;
    u64 size;
    vk::Buffer handle;
    u64 offset;
    const VideoCore::Buffer* arena; ///< as published when recorded (checked by handle)
};

// The arena reads one execution validated (the chained ranges repeat their buffers).
struct ValidatedReads {
    struct Hash {
        size_t operator()(const std::pair<VAddr, u64>& k) const noexcept {
            return size_t((k.first ^ (k.second << 40)) * 0x9E3779B97F4A7C15ull);
        }
    };
    std::unordered_set<std::pair<VAddr, u64>, Hash> set;
    bool Contains(VAddr address, u64 size) const { return set.contains({address, size}); }
};

struct Worker;
struct Recording {
    vk::CommandBuffer cmd{};
    /// 0: recorded, waiting; 1: not used (discarded); 2: being executed by the command
    /// processor; 3 + tick: executed in that submission.
    std::atomic<u64> state{};
    Worker* owner{};
    u32 generation{};
    u64 ring_end{};
    u32 draws{};
    u64 context_hash{};
    /// The inherited words the draws used, checked against the command processor's registers.
    std::array<u32, ng::UserWords> vs_user{}, ps_user{};
    u32 vs_mask{}, ps_mask{};
    u32 index_type{}, instances{};
    bool uses_index_type{}, uses_instances{};
    std::vector<std::shared_ptr<const Template>> templates; ///< unique, the first sets up targets
    std::vector<ArenaRead> arena_reads;                      ///< unique
    bool reads_unified{};
    VAddr unified_lo{}, unified_hi{}; ///< span of the imported heap read
    // A chain (ng::LinkFn): the next range of the command buffer, recorded from this one's end.
    Recording* next{};
    const ng::Batch* batch{};
    u32 batch_generation{};
    u64 seq{}; ///< order of the recordings of its worker (descriptor pool reuse)
};

// ISS_NATIVE_GBUFFER_SETS: descriptor sets of an engine thread, from pools used in recording
// order. A full pool is reset once the recordings that took sets from it are released (they are
// released in recording order, Worker::Reclaim).
constexpr u32 PoolSets = 256;
struct PoolBlock {
    vk::DescriptorPool pool;
    u64 last_user; ///< seq of the last recording that took a set from it
};

struct Worker {
    vk::Device device{};
    vk::CommandPool pool{};
    vk::Buffer ring{};
    vk::DeviceMemory memory{};
    u8* mapped{};
    u64 head{}, tail{}; ///< monotonic byte positions in the ring
    std::deque<Recording*> live;
    std::vector<Recording*> spare;
    bool ok{};
    u64 next_seq{};
    vk::DescriptorPool current_pool{};
    u64 current_last_user{};
    std::deque<PoolBlock> full_pools;
    std::vector<vk::DescriptorPool> free_pools;

    // A descriptor set of `layout` for recording `seq`; null when no pool can take it.
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
            // Full (or fragmented): retired until its users are released.
            full_pools.push_back({current_pool, current_last_user});
            current_pool = vk::DescriptorPool{};
        }
        return {};
    }

    // Releases what the GPU is done with, in recording order.
    void Reclaim() {
        const u64 gpu_tick = g.scheduler->GetWorkSemaphore()->KnownGpuTick();
        const u32 latest = ng::LatestGeneration();
        while (!live.empty()) {
            Recording* r = live.front();
            const u64 state = r->state.load(std::memory_order_acquire);
            const bool done =
                state == 1 || (state >= 3 && state - 3 <= gpu_tick) ||
                // Its range slot was reused: the command processor can no longer find it.
                (state == 0 && latest - r->generation > 8192 + 1024);
            if (!done) break;
            live.pop_front();
            tail = r->ring_end;
            r->cmd.reset();
            r->templates.clear();
            r->arena_reads.clear();
            spare.push_back(r);
            while (!full_pools.empty() && full_pools.front().last_user <= r->seq) {
                device.resetDescriptorPool(full_pools.front().pool);
                free_pools.push_back(full_pools.front().pool);
                full_pools.pop_front();
            }
        }
    }

    // Upload ring space, aligned; nullptr when the ring is full.
    u8* Upload(u64 bytes, u64 alignment, u64& offset) {
        u64 at = (head + alignment - 1) & ~(alignment - 1);
        if ((at % RingSize) + bytes > RingSize) at += RingSize - at % RingSize; // no wrap inside
        if (at + bytes - tail > RingSize) return nullptr;
        head = at + bytes;
        offset = at % RingSize;
        return mapped + offset;
    }

    bool Init() {
        const auto& instance = *g.instance;
        device = instance.GetDevice();
        pool = Check(device.createCommandPool(vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
        }));
        ring = Check(device.createBuffer(vk::BufferCreateInfo{
            .size = RingSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eUniformBuffer,
        }));
        const auto requirements = device.getBufferMemoryRequirements(ring);
        const auto& properties = instance.GetMemoryProperties();
        const auto want = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;
        u32 type = ~0u;
        for (u32 pass = 0; pass < 2 && type == ~0u; ++pass)
            for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
                const auto flags = properties.memoryTypes[i].propertyFlags;
                if (!(requirements.memoryTypeBits & (1u << i)) || (flags & want) != want) continue;
                if (pass == 0 && !(flags & vk::MemoryPropertyFlagBits::eDeviceLocal)) continue;
                type = i;
                break;
            }
        if (type == ~0u) return false;
        memory = Check(device.allocateMemory(vk::MemoryAllocateInfo{
            .allocationSize = requirements.size,
            .memoryTypeIndex = type,
        }));
        Check(device.bindBufferMemory(ring, memory, 0));
        mapped = static_cast<u8*>(Check(device.mapMemory(memory, 0, RingSize)));
        return mapped != nullptr;
    }

    Recording* Take() {
        Recording* r;
        if (!spare.empty()) {
            r = spare.back();
            spare.pop_back();
        } else {
            r = new Recording;
            r->owner = this;
            const vk::CommandBufferAllocateInfo info{
                .commandPool = pool,
                .level = vk::CommandBufferLevel::eSecondary,
                .commandBufferCount = 1,
            };
            Check(device.allocateCommandBuffers(&info, &r->cmd));
        }
        r->state.store(0, std::memory_order_relaxed);
        r->next = nullptr;
        r->batch = nullptr;
        r->seq = next_seq++;
        return r;
    }

    // A recording that will not be used goes back at once.
    void Abandon(Recording* r) {
        r->cmd.reset();
        r->templates.clear();
        r->arena_reads.clear();
        spare.push_back(r);
    }
};

static Worker* ThisWorker() {
    static thread_local std::unique_ptr<Worker> worker;
    static thread_local bool failed = false;
    if (worker || failed) return worker.get();
    auto w = std::make_unique<Worker>();
    try {
        w->ok = w->Init();
    } catch (...) {
        w->ok = false;
    }
    if (!w->ok) {
        failed = true;
        std::fprintf(stderr, "NATIVE_RECORDER worker setup failed\n");
        return nullptr;
    }
    worker = std::move(w);
    return worker.get();
}

enum Miss : u32 { MissShape, MissTemplate, MissBuffer, MissState, MissRing, MissFormat,
                  MissPipeline, MissPool, MissCount };
struct EngineStats {
    std::atomic<u64> recorded{}, draws{};
    std::array<std::atomic<u64>, MissCount> miss{};
};
static EngineStats engine_stats;
constexpr const char* miss_names[MissCount] = {"no_shape", "no_template", "buffer", "state",
                                               "ring", "format", "pipeline", "pool"};

// Diagnosis of shapes the engine thread could not find: a sampled range start as the engine
// thread knew it, compared by the command processor with its registers at that range.
struct DivergenceSample {
    std::atomic<u32> generation{}; ///< 0: free; written last
    std::array<u32, ng::TrackedWords> words{};
};
static std::array<DivergenceSample, 64> divergence_samples;
static std::atomic<u32> divergence_counter{};
static std::array<u64, ng::TrackedWords> divergence_regs{}; // command processor only
static u64 divergence_compared{};

static u32 TrackedReg(u32 index) {
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        if (index < base + ng::TrackedCount[r]) return ng::TrackedFirst[r] + index - base;
        base += ng::TrackedCount[r];
    }
    return 0;
}

// The flattened user data of a stage, in a thread local view (Shader::Info::UdSnapshot).
struct StageView {
    const Shader::Info* info{};
    std::array<u32, 512> flat{};
    u32 flat_size{};
};

static bool InstallViews(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                         std::array<StageView, 4>& views, u32& num_views) {
    for (const auto* stage : pipeline.GetStages()) {
        if (!stage) continue;
        bool present = false;
        for (u32 i = 0; i < num_views; ++i) present |= views[i].info == stage;
        if (present) continue;
        if (num_views == views.size()) return false;
        const AmdGpu::ShaderProgram* program =
            stage->hw_stage == Shader::HwStage::Fragment ? &regs.ps_program
            : stage->hw_stage == Shader::HwStage::Vertex ? &regs.vs_program : nullptr;
        if (!program) return false;
        auto& view = views[num_views];
        const u32 size = stage->srt_info.flattened_bufsize_dw;
        if (size > view.flat.size() || size < Shader::NUM_USER_DATA_REGS) return false;
        const auto* ud = reinterpret_cast<const u32*>(&program->user_data);
        std::memcpy(view.flat.data(), ud, Shader::NUM_USER_DATA_REGS * 4);
        if (stage->srt_info.walker_func) stage->srt_info.walker_func(ud, view.flat.data());
        view.info = stage;
        view.flat_size = size;
        Shader::Info::ud_snapshots[num_views] = Shader::Info::UdSnapshot{
            .info = stage,
            .ud = ud,
            .ud_size = Shader::NUM_USER_DATA_REGS,
            .flat = view.flat.data(),
            .flat_size = size,
            .base = program->Address<VAddr>(),
            .reads = nullptr,
            .num_reads = 0,
        };
        Shader::Info::num_ud_snapshots = ++num_views;
    }
    return true;
}

static void SetDrawRegisters(AmdGpu::Regs& regs, const ng::Batch& b, u32 index,
                             const std::array<u32, ng::MaxStateRegs>& saved_state,
                             const std::array<u32, ng::UserWords>& saved_vs,
                             const std::array<u32, ng::UserWords>& saved_ps, u32 saved_instances,
                             u32 saved_index_type) {
    const auto& d = b.draws[index];
    const auto& state = b.states[d.state];
    for (u32 k = 0; k < b.state_regs.size(); ++k)
        regs.reg_array[b.state_regs[k]] =
            (state.inherit >> k) & 1 ? saved_state[k] : b.state_values[state.first_value + k];
    for (u32 k = 0; k < ng::UserWords; ++k) {
        regs.reg_array[ng::VsUser0 + k] = (d.vs_inherit >> k) & 1 ? saved_vs[k] : d.vs_user[k];
        regs.reg_array[ng::PsUser0 + k] = (d.ps_inherit >> k) & 1 ? saved_ps[k] : d.ps_user[k];
    }
    regs.num_instances.num_instances = d.instances_inherit ? saved_instances : d.instances;
    regs.index_buffer_type.raw = d.index_type_inherit ? saved_index_type : d.index_type;
    regs.index_base_address.base_addr_lo = d.index_lo;
    regs.index_base_address.base_addr_hi = d.index_hi;
    regs.num_indices = d.index_count;
    regs.max_index_size = d.max_index;
    regs.draw_initiator = d.initiator;
}

// Records a range on the engine thread that wrote it (ng::RecordFn).
static std::atomic<u32> trace_left{3}; // the first recordings print their steps
#define NR_STEP(name)                                                                             \
    do {                                                                                          \
        if (tracing) {                                                                            \
            std::fprintf(stderr, "NATIVE_RECORDER_STEP %s\n", name);                             \
            std::fflush(stderr);                                                                  \
        }                                                                                         \
    } while (0)

static ng::Recording* RecordBatch(const ng::Batch& b, const ng::RangeStart& start) {
    if (!g.ready.load(std::memory_order_acquire)) return nullptr;
    u32 left = trace_left.load(std::memory_order_relaxed);
    const bool tracing = left && trace_left.compare_exchange_strong(left, left - 1);
    NR_STEP("begin");
    Worker* w = ThisWorker();
    if (!w) return nullptr;
    NR_STEP("worker");
    w->Reclaim();
    const u32 n = u32(b.draws.size());
    auto& stats = engine_stats;
    const auto miss = [&](Miss m) -> ng::Recording* {
        stats.miss[m].fetch_add(1, std::memory_order_relaxed);
        Shader::Info::num_ud_snapshots = 0;
        return nullptr;
    };

    static thread_local std::unique_ptr<AmdGpu::Regs> regs_storage;
    if (!regs_storage) regs_storage = std::make_unique<AmdGpu::Regs>();
    auto& regs = *regs_storage;
    u32 word = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(&regs.reg_array[ng::TrackedFirst[r]], &start.words[word], ng::TrackedCount[r] * 4);
        word += ng::TrackedCount[r];
    }
    regs.index_buffer_type.raw = start.index_type;
    regs.num_instances.num_instances = start.instances;
    const u64 context = NativeMaterial::ContextHash(regs, b.state_regs);
    NR_STEP("context");

    std::array<u32, ng::MaxStateRegs> saved_state{};
    for (u32 k = 0; k < b.state_regs.size(); ++k) saved_state[k] = regs.reg_array[b.state_regs[k]];
    std::array<u32, ng::UserWords> saved_vs{}, saved_ps{};
    for (u32 k = 0; k < ng::UserWords; ++k) {
        saved_vs[k] = regs.reg_array[ng::VsUser0 + k];
        saved_ps[k] = regs.reg_array[ng::PsUser0 + k];
    }

    const VideoCore::Buffer* unified = g.unified.load(std::memory_order_acquire);
    const VAddr unified_base = g.unified_base.load(std::memory_order_relaxed);
    auto& cache = *g.buffer_cache;
    const u64 alignment = g.instance->StorageMinAlignment();

    Recording* rec = w->Take();
    rec->generation = b.generation;
    rec->batch = &b;
    rec->batch_generation = b.generation;
    rec->context_hash = context;
    rec->vs_mask = rec->ps_mask = 0;
    rec->uses_index_type = rec->uses_instances = false;
    rec->reads_unified = false;
    rec->draws = n;
    const u64 head_before = w->head;
    const auto fail = [&](Miss m) -> ng::Recording* {
        w->head = head_before;
        w->Abandon(rec);
        return miss(m);
    };

    // Resolves a guest range the draw reads: the imported heap or a published arena.
    const auto resolve = [&](VAddr address, u64 size, vk::Buffer& handle, u64& offset) {
        if (unified && cache.IsUnified(address, size)) {
            handle = unified->Handle();
            offset = address - unified_base;
            if (!rec->reads_unified) {
                rec->unified_lo = address;
                rec->unified_hi = address + size;
            }
            rec->unified_lo = std::min(rec->unified_lo, address);
            rec->unified_hi = std::max(rec->unified_hi, VAddr(address + size));
            rec->reads_unified = true;
            return true;
        }
        const auto arena = cache.PeekPublishedArena(address, u32(size));
        if (!arena) return false;
        handle = arena.arena->Handle();
        offset = arena.offset;
        bool seen = false;
        for (const auto& r : rec->arena_reads)
            if (r.address == address && r.size == size) {
                seen = true;
                break;
            }
        if (!seen) rec->arena_reads.push_back({address, size, handle, offset, arena.arena});
        return true;
    };

    std::array<StageView, 4> views;
    const Template* current = nullptr;
    vk::Pipeline bound_pipeline{};
    bool begun = false;
    std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> infos;
    Pipeline::DescriptorWrites writes;
    for (u32 i = 0; i < n; ++i) {
        const auto& d = b.draws[i];
        SetDrawRegisters(regs, b, i, saved_state, saved_vs, saved_ps, start.instances,
                         start.index_type);
        rec->vs_mask |= d.vs_inherit;
        rec->ps_mask |= d.ps_inherit;
        rec->uses_instances |= d.instances_inherit;
        rec->uses_index_type |= d.index_type_inherit;
        if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Normal ||
            regs.primitive_type == AmdGpu::PrimitiveType::None ||
            (regs.reg_array[0xA003] & 0x78000000u) || regs.enable_primitive_restart ||
            regs.depth_render_control.depth_clear_enable ||
            regs.depth_render_control.stencil_clear_enable)
            return fail(MissState);
        if (regs.num_indices == 0) continue;

        const u64 key = NativeMaterial::StateKey(context, b, d.state) | 1;
        std::shared_ptr<const Template> t;
        {
            auto& tables = NativeMaterial::State();
            const GraphicsPipeline* shape = nullptr;
            {
                std::shared_lock lock{tables.mutex};
                const auto it = tables.shapes.find(key);
                if (it != tables.shapes.end()) shape = it->second;
            }
            if (!shape) {
                if ((divergence_counter.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
                    auto& sample = divergence_samples[b.generation & 63];
                    sample.generation.store(0, std::memory_order_relaxed);
                    sample.words = start.words;
                    sample.generation.store(b.generation, std::memory_order_release);
                }
                return fail(MissShape);
            }
            u32 num_views = 0;
            Shader::Info::num_ud_snapshots = 0;
            Shader::Info::ud_snapshots_used.store(true, std::memory_order_relaxed);
            NR_STEP("views");
            if (!InstallViews(*shape, regs, views, num_views)) return fail(MissState);
            NR_STEP("material");
            const u64 material = NativeMaterial::MaterialHash(*shape);
            {
                std::shared_lock lock{tables.mutex};
                const auto it = tables.templates.find(NativeMaterial::Mix(key, material));
                if (it != tables.templates.end()) t = it->second;
            }
            if (!t) return fail(MissTemplate);
            NativeMaterial::Touch(*t);
            if (t->pipeline != shape && !InstallViews(*t->pipeline, regs, views, num_views))
                return fail(MissState);
            if (!NativeMaterial::HandlesAlive(*g.texture_cache, *t)) return fail(MissState);
        }
        const auto* pipeline = t->pipeline;
        if (!pipeline->IsReady() ||
            (!pipeline->UsesPushDescriptors() && !NativeMaterial::SetsEnabled()))
            return fail(MissPipeline);

        NR_STEP("template");
        if (!begun) {
            const vk::CommandBufferInheritanceRenderingInfo rendering{
                .colorAttachmentCount = t->num_colors,
                .pColorAttachmentFormats = t->color_formats.data(),
                .depthAttachmentFormat = t->depth_format,
                .stencilAttachmentFormat = t->stencil_format,
                .rasterizationSamples = t->samples,
            };
            const vk::CommandBufferInheritanceInfo inheritance{.pNext = &rendering};
            rec->cmd.begin(vk::CommandBufferBeginInfo{
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit |
                         vk::CommandBufferUsageFlagBits::eRenderPassContinue,
                .pInheritanceInfo = &inheritance,
            });
            begun = true;
        } else {
            const auto& first = *rec->templates.front();
            if (t->num_colors != first.num_colors || t->color_formats != first.color_formats ||
                t->depth_format != first.depth_format || t->stencil_format != first.stencil_format ||
                t->samples != first.samples)
                return fail(MissFormat);
        }
        bool listed = false;
        for (const auto& u : rec->templates) listed |= u.get() == t.get();
        if (!listed) rec->templates.push_back(t);
        const auto cmd = rec->cmd;

        NR_STEP("dynamic");
        if (t.get() != current) {
            auto dynamic = *t->dynamic;
            dynamic.Invalidate();
            dynamic.Commit(*g.instance, CommandRecorder{cmd});
            current = t.get();
        }
        if (pipeline->Handle() != bound_pipeline) {
            bound_pipeline = pipeline->Handle();
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, bound_pipeline);
        }

        // Descriptors: the template's textures and samplers, the draw's buffers.
        Shader::PushData push = t->push;
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
                info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                continue;
            }
            const auto sharp = stage.buffers[slot.desc].GetSharp(stage);
            if (!sharp.base_address || !sharp.GetSize()) {
                info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                continue;
            }
            vk::Buffer handle;
            u64 offset;
            if (!resolve(sharp.base_address, sharp.GetSize(), handle, offset))
                return fail(MissBuffer);
            const u64 aligned = Common::AlignDown(offset, alignment);
            push.buf_offsets[slot.binding] = u8(offset - aligned);
            info = vk::DescriptorBufferInfo{handle, aligned, sharp.GetSize() + (offset - aligned)};
        }
        NR_STEP("descriptors");
        NativeMaterial::BuildWrites(*t, infos.data(), writes);
        if (pipeline->UsesPushDescriptors()) {
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, pipeline->GetLayout(), 0,
                                     writes);
        } else if (!writes.empty()) {
            const auto set = w->AllocateSet(pipeline->GetDescriptorSetLayout(), rec->seq);
            if (!set) return fail(MissPool);
            for (auto& write : writes) write.dstSet = set;
            w->device.updateDescriptorSets(u32(writes.size()), writes.data(), 0, nullptr);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline->GetLayout(), 0, 1,
                                   &set, 0, nullptr);
        }
        cmd.pushConstants(pipeline->GetLayout(), AllGraphicsStageBits, 0u, sizeof(push), &push);

        // Vertex input and buffers, index buffer.
        VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
        VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
        VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                                  regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
        NR_STEP("vertex");
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
        const bool index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = index16 ? 2 : 4;
        vk::Buffer index_handle;
        u64 index_offset;
        if (!resolve(regs.index_base_address.Address<VAddr>(), u64(regs.num_indices) * index_size,
                     index_handle, index_offset))
            return fail(MissBuffer);
        cmd.bindIndexBuffer(index_handle, index_offset,
                            index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

        const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
        const auto [vertex_offset, instance_offset] =
            GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
        cmd.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0, s32(vertex_offset),
                        instance_offset);
        NR_STEP("draw");
    }
    Shader::Info::num_ud_snapshots = 0;
    if (!begun) {
        w->head = head_before;
        w->Abandon(rec);
        return nullptr; // nothing drawn
    }
    rec->cmd.end();
    for (u32 k = 0; k < ng::UserWords; ++k) {
        rec->vs_user[k] = saved_vs[k];
        rec->ps_user[k] = saved_ps[k];
    }
    rec->index_type = start.index_type;
    rec->instances = start.instances;
    rec->ring_end = w->head;
    w->live.push_back(rec);
    stats.recorded.fetch_add(1, std::memory_order_relaxed);
    stats.draws.fetch_add(n, std::memory_order_relaxed);
    return reinterpret_cast<ng::Recording*>(rec);
}

enum Reject : u32 { RejectNone, RejectContext, RejectInherited, RejectPipeline, RejectSamplers,
                    RejectTextures, RejectArena, RejectTargets, RejectFormats, RejectCount };
struct CpStats {
    u64 executed{}, draws{}, missing{}, ranges{}, chained{};
    std::array<u64, RejectCount> reject{};
};
static CpStats cp_stats;
constexpr const char* reject_names[RejectCount] = {"none", "context", "inherited", "pipeline",
                                                   "samplers", "textures", "arena", "targets",
                                                   "formats"};

static void Report() {
    if (divergence_compared) {
        std::array<u32, ng::TrackedWords> order;
        for (u32 i = 0; i < order.size(); ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + 16, order.end(),
                          [](u32 a, u32 b) { return divergence_regs[a] > divergence_regs[b]; });
        std::string top;
        for (u32 i = 0; i < 16 && divergence_regs[order[i]]; ++i) {
            char item[40];
            std::snprintf(item, sizeof(item), " %x=%llu", TrackedReg(order[i]),
                          (unsigned long long)divergence_regs[order[i]]);
            top += item;
        }
        std::fprintf(stderr, "NATIVE_RECORDER_DIVERGENCE samples=%llu regs[%s ]\n",
                     (unsigned long long)divergence_compared, top.c_str());
    }
    auto& e = engine_stats;
    std::string misses;
    for (u32 m = 0; m < MissCount; ++m) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", miss_names[m],
                      (unsigned long long)e.miss[m].load(std::memory_order_relaxed));
        misses += item;
    }
    std::string rejects;
    for (u32 r = 1; r < RejectCount; ++r) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", reject_names[r],
                      (unsigned long long)cp_stats.reject[r]);
        rejects += item;
    }
    std::fprintf(stderr,
                 "NATIVE_RECORDER engine recorded=%llu draws=%llu miss[%s ] | cp executed=%llu "
                 "chained=%llu draws=%llu not_recorded=%llu reject[%s ]\n",
                 (unsigned long long)e.recorded.load(), (unsigned long long)e.draws.load(),
                 misses.c_str(), (unsigned long long)cp_stats.executed,
                 (unsigned long long)cp_stats.chained,
                 (unsigned long long)cp_stats.draws, (unsigned long long)cp_stats.missing,
                 rejects.c_str());
}
// Links two consecutive recorded ranges of a command buffer (ng::LinkFn, engine thread) when
// they render to the same attachment formats.
static void Link(const ng::Batch& previous, const ng::Batch& next) {
    auto* a = reinterpret_cast<Recording*>(previous.recording);
    auto* b = reinterpret_cast<Recording*>(next.recording);
    if (!a || !b || a->templates.empty() || b->templates.empty() || a->next) return;
    const auto& x = *a->templates.front();
    const auto& y = *b->templates.front();
    if (x.num_colors != y.num_colors || x.color_formats != y.color_formats ||
        x.depth_format != y.depth_format || x.stencil_format != y.stencil_format ||
        x.samples != y.samples)
        return;
    a->next = b;
}

// A range the command processor takes another path for: its recording is released.
static void Discard(const ng::Batch& b) {
    if (auto* rec = reinterpret_cast<Recording*>(b.recording)) {
        u64 expected = 0;
        rec->state.compare_exchange_strong(expected, 1);
    }
}
} // namespace NativeRecorder

void Rasterizer::NativeRecorderSetup() {
    using namespace NativeRecorder;
    static bool done = false;
    if (done) return;
    done = true;
    if (!ng::RecorderEnabled() || !NativeMaterial::Enabled()) return;
    g.instance = &instance;
    g.scheduler = &scheduler;
    g.buffer_cache = &buffer_cache;
    g.texture_cache = &texture_cache;
    Shader::Info::ud_snapshots_used.store(true, std::memory_order_relaxed);
    g.ready.store(true, std::memory_order_release);
    ng::SetRecorder(&RecordBatch, &Link);
    HMODULE module{};
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCSTR>(&RecordBatch), &module);
    std::fprintf(stderr, "NATIVE_RECORDER on (iss-gpu base %p)\n", static_cast<void*>(module));
}

u32 Rasterizer::NativeRecorderExecute(const iss::native_gbuffer::Batch& b, u64 context,
                                      const std::function<void(u32)>& set_registers,
                                      const std::function<void(const u32*)>& begin_draw,
                                      const std::function<void()>& end_draw) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::RecorderExecute};
    using namespace NativeRecorder;
    auto& c = cp_stats;
    if ((++c.ranges & 4095) == 1) Report();
    // The imported heap, once there, for the engine threads.
    if (!g.unified.load(std::memory_order_relaxed) && buffer_cache.UnifiedBuffer()) {
        g.unified_base.store(buffer_cache.UnifiedBase(), std::memory_order_relaxed);
        g.unified.store(buffer_cache.UnifiedBuffer(), std::memory_order_release);
    }
    auto& regs = DrawRegs();
    // The registers at this range start are the engine threads' next baseline.
    static u64 published_context = 0;
    auto* rec = reinterpret_cast<Recording*>(b.recording);
    if (!rec) {
        auto& sample = divergence_samples[b.generation & 63];
        if (sample.generation.load(std::memory_order_acquire) == b.generation) {
            ++divergence_compared;
            u32 base = 0;
            for (u32 r = 0; r < 3; ++r) {
                for (u32 k = 0; k < ng::TrackedCount[r]; ++k)
                    divergence_regs[base + k] +=
                        sample.words[base + k] != regs.reg_array[ng::TrackedFirst[r] + k];
                base += ng::TrackedCount[r];
            }
            sample.generation.store(0, std::memory_order_relaxed);
        }
        ++c.missing;
        if (context != published_context) {
            ng::PublishBaseline(regs.reg_array.data(), regs.index_buffer_type.raw,
                                regs.num_instances.num_instances);
            published_context = context;
        }
        return 0;
    }
    u64 expected = 0;
    if (!rec->state.compare_exchange_strong(expected, 2)) return 0; // already decided
    const auto reject = [&](Reject r) -> u32 {
        ++c.reject[r];
        ResetBindings(false);
        rec->state.store(1, std::memory_order_release);
        // Discarded: its engine thread reclaims it.
        if (r == RejectContext || r == RejectInherited) {
            ng::PublishBaseline(regs.reg_array.data(), regs.index_buffer_type.raw,
                                regs.num_instances.num_instances);
            published_context = context;
        }
        return 0;
    };
    if (context != rec->context_hash) return reject(RejectContext);
    for (u32 k = 0; k < ng::UserWords; ++k) {
        if (((rec->vs_mask >> k) & 1) && regs.reg_array[ng::VsUser0 + k] != rec->vs_user[k])
            return reject(RejectInherited);
        if (((rec->ps_mask >> k) & 1) && regs.reg_array[ng::PsUser0 + k] != rec->ps_user[k])
            return reject(RejectInherited);
    }
    if ((rec->uses_index_type && regs.index_buffer_type.raw != rec->index_type) ||
        (rec->uses_instances && regs.num_instances.num_instances != rec->instances))
        return reject(RejectInherited);
    for (const auto& t : rec->templates) {
        if (!t->pipeline->IsReady()) return reject(RejectPipeline);
        if (t->has_samplers && texture_cache.SamplersGeneration() != t->samplers_generation)
            return reject(RejectSamplers);
    }
    static thread_local ValidatedReads validated_reads;
    validated_reads.set.clear();
    for (const auto& r : rec->arena_reads) {
        if (!buffer_cache.IsRegionCpuClean(r.address, r.size)) return reject(RejectArena);
        const auto arena = buffer_cache.PeekPublishedArena(r.address, u32(r.size));
        if (!arena || arena.arena->Handle() != r.handle || arena.offset != r.offset ||
            arena.arena != r.arena)
            return reject(RejectArena);
        validated_reads.set.emplace(r.address, r.size);
    }

    // The render targets, as the first template's pipeline sets them up for the first draw.
    const auto& first = *rec->templates.front();
    const auto* pipeline = first.pipeline;
    ResetBindings(false);
    set_registers(0);
    PrepareRenderState(pipeline);
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb)
        if (cb_descs[cb].first &&
            texture_cache.IsMetaCleared(regs.color_buffers[cb].CmaskAddress(),
                                        cb_descs[cb].second.view_info.range.base.layer))
            return reject(RejectTargets);
    if (db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                                     db_desc.second.view_info.range.base.layer))
        return reject(RejectTargets);
    // Textures checked after the targets are marked (a texture that is a target is feedback).
    // Each template once per execution: the chained ranges mostly repeat the head's.
    static thread_local std::vector<const Template*> validated;
    validated.clear();
    for (const auto& t : rec->templates) {
        bool stale = false;
        if (!NativeMaterial::TexturesValid(texture_cache, *t, stale)) {
            if (stale) NativeMaterialDrop(t->key);
            ResetBindings(false);
            return reject(RejectTextures);
        }
        validated.push_back(t.get());
    }
    // The ranges chained to this one, as far as each is still valid (the others take their own
    // marker): the registers between them are plain writes the command processor applies.
    iss::cp_path::Switch(iss::cp_path::RecChain);
    std::vector<Recording*> chain{rec};
    for (Recording* m = rec->next; m; m = m->next) {
        if (m->batch_generation != m->batch->generation ||
            reinterpret_cast<Recording*>(m->batch->recording) != m ||
            !native_span_end || m->batch->end > native_span_end)
            break;
        u64 idle = 0;
        if (!m->state.compare_exchange_strong(idle, 2)) break;
        bool ok = true;
        for (const auto& t : m->templates) {
            if (!ok) break;
            if (std::find(validated.begin(), validated.end(), t.get()) != validated.end()) continue;
            bool stale = false;
            ok = t->pipeline->IsReady() &&
                 (!t->has_samplers || texture_cache.SamplersGeneration() == t->samplers_generation) &&
                 NativeMaterial::TexturesValid(texture_cache, *t, stale);
            if (stale) NativeMaterialDrop(t->key);
            if (ok) validated.push_back(t.get());
        }
        for (const auto& r : m->arena_reads) {
            if (!ok) break;
            if (validated_reads.Contains(r.address, r.size)) continue;
            const auto arena = buffer_cache.PeekPublishedArena(r.address, u32(r.size));
            ok = buffer_cache.IsRegionCpuClean(r.address, r.size) && arena &&
                 arena.arena->Handle() == r.handle && arena.offset == r.offset &&
                 arena.arena == r.arena;
            if (ok) validated_reads.set.emplace(r.address, r.size);
        }
        if (!ok) {
            m->state.store(0, std::memory_order_release);
            break;
        }
        chain.push_back(m);
    }
    const auto release_chain = [&] {
        for (size_t i = 1; i < chain.size(); ++i) chain[i]->state.store(0, std::memory_order_release);
    };
    iss::cp_path::Switch(iss::cp_path::RecTargets);
    const RenderState render = BeginRendering(pipeline);
    if (attachment_feedback_loop) {
        release_chain();
        ResetBindings(false);
        return reject(RejectTargets);
    }
    // The attachment formats the secondary was recorded for.
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
            const auto format =
                texture_cache.FindDepthTarget(db_desc.first, db_desc.second).info.format;
            depth_format = depth.has_depth ? format : vk::Format::eUndefined;
            stencil_format = depth.has_stencil ? format : vk::Format::eUndefined;
        }
        formats_equal = depth_format == first.depth_format && stencil_format == first.stencil_format;
    }
    if (!formats_equal) {
        release_chain();
        ResetBindings(false);
        return reject(RejectFormats);
    }

    // Accepted: textures to their read layouts, buffer reads ordered after their writers.
    iss::cp_path::Switch(iss::cp_path::RecBarriers);
    const u64 tick = scheduler.CurrentTick();
    for (const Template* t : validated) {
        for (const auto& cap : t->images) {
            if (!cap.id) continue;
            auto& image = texture_cache.GetImage(cap.id);
            needs_barrier |= runtime.Transit(&image, cap.layout,
                                             vk::PipelineStageFlagBits2::eAllCommands,
                                             vk::AccessFlagBits2::eShaderRead, cap.range);
            image.usage.texture = 1u;
            image.tick_accessed_last = tick;
        }
    }
    // The buffer reads: ordered after their writers now, recorded as reads after the
    // secondary (as ResetBindings does for bound buffers; a range reads more than they hold).
    struct Read {
        const VideoCore::Buffer* buffer;
        u64 offset, size;
    };
    iss::cp_path::Switch(iss::cp_path::RecReads);
    static thread_local std::vector<Read> reads;
    reads.clear();
    for (const Recording* m : chain) {
        for (const auto& r : m->arena_reads) reads.push_back({r.arena, r.offset, r.size});
        if (m->reads_unified) {
            if (const auto* unified = buffer_cache.UnifiedBuffer()) {
                const u64 offset = m->unified_lo - buffer_cache.UnifiedBase();
                const u64 size = m->unified_hi - m->unified_lo;
                reads.push_back({unified, offset, size});
            }
        }
    }
    // One check and one recorded read per buffer stretch: ranges of the same buffer that
    // overlap or lie within 64 KB of each other (the chained ranges read neighbouring records).
    std::sort(reads.begin(), reads.end(), [](const Read& a, const Read& b) {
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
    for (const auto& r : reads) needs_barrier |= Rt().IsBufferAccessed(r.buffer, r.offset, r.size, false);
    iss::cp_path::Switch(iss::cp_path::RecSubmit);
    if (needs_barrier) runtime.FlushBarriers();
    // A rendering scope of secondary contents only (no VK_KHR_maintenance7 mixing).
    scheduler.BeginSecondaryRendering(render);
    std::vector<vk::CommandBuffer> secondaries;
    secondaries.reserve(chain.size());
    for (const Recording* m : chain) secondaries.push_back(m->cmd);
    scheduler.Recorder().Record([secondaries = std::move(secondaries)](vk::CommandBuffer cmd) {
        cmd.executeCommands(secondaries);
    });
    scheduler.EndRendering();
    // Nothing bound by the secondary survives it in the primary.
    scheduler.GetDynamicState().Invalidate();
    fast_draw.valid = false;
    native_stretch.clear();
    iss::cp_path::Switch(iss::cp_path::RecDraws);
    for (const Recording* m : chain) {
        for (const auto& d : m->batch->draws) {
            begin_draw(d.packet);
            DebugState.IncDrawCall();
            end_draw();
        }
        if (m != rec) native_stretch.push_back(m->batch);
    }
    iss::cp_path::Switch(iss::cp_path::RecAccess);
    ResetBindings(false);
    for (const auto& r : reads)
        Rt().AccessBuffer(r.buffer, r.offset, r.size, vk::PipelineStageFlagBits2::eAllGraphics,
                          vk::AccessFlagBits2::eShaderRead);
    for (Recording* m : chain) {
        m->state.store(3 + scheduler.CurrentTick(), std::memory_order_release);
        ++c.executed;
        c.draws += m->batch->draws.size();
    }
    c.chained += chain.size() - 1;
    return u32(b.draws.size());
}
