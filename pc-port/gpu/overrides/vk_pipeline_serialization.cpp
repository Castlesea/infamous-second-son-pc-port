// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <set>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <filesystem>
#include "common/path_util.h"
#include "shader_salt.h"
#include "common/serdes.h"
#include "common/elf_info.h"
#include "pipeline_library.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/scope_exit.h"

namespace iss {
extern thread_local unsigned long long recipe_hash;
}
#include "common/perf_stats.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 8u; // 7: legacy multiply-add (v_mac_legacy_f32),
                                                // 8: upstream ADDC/ALIGNBIT/CVT_PK_U8/thread id fixes
static constexpr u32 ShaderMetaVersion = 7u; // capture walker + relocation metadata
static constexpr u32 LegacyShaderMetaVersion = 6u; // packed library made before SRT capture
static constexpr u32 PipelineKeyVersion = 3u;
static thread_local bool loading_legacy_srt{};

/// Compute shaders compiled with a different recompiler variant (see SetComputeShaderVariant)
/// are stored with another binary version, so changing the variant recompiles them once.
static u32 compute_shader_variant{};

static u32 BinaryVersionFor(const Shader::Info& info) {
    return info.hw_stage == Shader::HwStage::Compute
               ? ShaderBinaryVersion + compute_shader_variant * 100u
               : ShaderBinaryVersion;
}
} // namespace Serialization

// The shader preload's progress for the window title while the game starts (bridge.cpp).
namespace iss {
std::atomic<unsigned> g_preload_entries{0};
std::atomic<int> g_preload_phase{0};
std::atomic<unsigned> g_preload_total{0};
}

namespace Vulkan {

std::optional<std::vector<u32>> RegenerateRecipeSpirv(const Shader::Profile& profile,
                                                      const Shader::Info& meta,
                                                      const Shader::StageSpecialization& spec,
                                                      std::span<const u8> attrib_variants = {});
std::optional<std::vector<u8>> RecoverAttribVariants(const Shader::Profile& profile,
                                                     const Shader::Info& meta,
                                                     const Shader::StageSpecialization& spec,
                                                     const std::vector<u32>& reference);
std::vector<u8> RecipeAttribVariants(u64 perm_hash);
void RememberAttribVariants(u64 perm_hash, std::vector<u8> variants);
void SaveAttribVariants();
namespace {
struct RecipeStats {
    u32 made, matched, mismatched, failed, recovered;
};
std::array<RecipeStats, 6> recipe_stats{}; // by SwStage (WarmUp is serial)
/// ISS_AOT_RECIPES: SPIR-V of recipe stages made in parallel before the (serial) preload.
std::mutex recipe_mutex;
std::unordered_map<u64, std::vector<u32>> recipe_spirv; // permutation hash -> SPIR-V
/// ISS_AOT_RECIPE_VERIFY: the stages whose SPIR-V made from the game files is identical to the
/// cached one (written to ISS_AOT_RECIPE_VERIFIED for make_recipes.py: recipes keep only those).
std::vector<u64> recipe_verified;
/// Library and cache never mix inside one shader program: its permutations share the Info of the
/// first one loaded, and permutation indices are local to a lineage. A program the cache has is
/// the cache's (the library's stages of it are left out), one only the library has is the
/// library's. A cache program given a library permutation made pipelines that crashed the driver
/// (0xC0000409 in the preload, 2026-10-19). WarmUp is serial; program_cache_mutex guards these.
bool g_stage_from_library = false;
std::unordered_set<u64> g_library_programs;
u32 g_library_stages_left = 0, g_cache_stages_left = 0;
} // Anonymous namespace

void SetComputeShaderVariant(u32 variant) {
    Serialization::compute_shader_variant = variant;
}

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::BinaryVersionFor(info));

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion &&
        meta_version != Serialization::LegacyShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version % 100u != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    const bool previous_legacy = Serialization::loading_legacy_srt;
    Serialization::loading_legacy_srt =
        meta_version == Serialization::LegacyShaderMetaVersion;
    SCOPE_EXIT { Serialization::loading_legacy_srt = previous_legacy; };
    const bool loaded = info.Deserialize(ar);
    if (!loaded) return false;
    // Compiled with another recompiler variant: compiled again.
    return binary_version == Serialization::BinaryVersionFor(info);
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadCachedPipelineStage(u64 hash, size_t stage) {
    if (const auto it = preload_stages.find(hash); it != preload_stages.end()) {
        const auto [program, index] = it->second;
        infos[stage] = &program->info;
        modules[stage] = program->modules[index].module;
        if (auto& fetch = program->modules[index].spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
        }
        ++preload_stage_hits;
        return true;
    }
    // Storage is read-only during WarmUp. Repeated keys resolve to the same
    // validated metadata/SPIR-V; retain an index, not a vector element pointer.
    std::vector<u8> meta_blob;
    const bool from_library =
        iss::pipeline_library::Read(fmt::format("{:#018x}.meta", hash), meta_blob);
    if (!from_library)
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                          fmt::format("{:#018x}", hash), meta_blob);
    if (meta_blob.empty()) return false;
    Serialization::Archive meta_ar{std::move(meta_blob)};
    g_stage_from_library = from_library;
    SCOPE_EXIT { g_stage_from_library = false; };
    return LoadPipelineStage(meta_ar, stage, hash);
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;
    SCOPE_EXIT { infos.fill(nullptr); modules.fill(nullptr); fetch_shader = nullptr; };
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    if (!LoadCachedPipelineStage(compute_key.value, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (!is_new) {
        // A merged library can carry the same pipeline from two cache lineages (different file
        // names, same key): the first one is kept.
        return false;
    }

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true,
                                          parallel_preload);
    if (parallel_preload) QueuePreload(it.value().get());

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;
    SCOPE_EXIT { infos.fill(nullptr); modules.fill(nullptr); fetch_shader = nullptr; };
    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        if (!LoadCachedPipelineStage(hash, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (!is_new) {
        return false; // the same pipeline from another lineage of a merged library
    }

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true, parallel_preload);
    if (parallel_preload) QueuePreload(it.value().get());

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage, u64 expected_hash) {
    // A crash while this stage loads names it (host_hooks.cpp).
    const auto prev_recipe = std::exchange(::iss::recipe_hash, expected_hash);
    SCOPE_EXIT {
        ::iss::recipe_hash = prev_recipe;
    };
    std::vector<u8> metadata(ar.CurrPtr(), ar.CurrPtr() + ar.SizeBytes());
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, program->info, spec, perm_idx)) {
        return false;
    }
    if (HashCombine(program->info.pgm_hash, u64{perm_idx}) != expected_hash) {
        LOG_WARNING(Render_Vulkan, "Cached shader metadata does not match pipeline key {:#x}",
                    expected_hash);
        return false;
    }
    {
        // A program of the other source (see g_library_programs): before any SPIR-V of this
        // stage is made or stored.
        std::unique_lock lock{program_cache_mutex};
        if ((program_cache.count(program->info.pgm_hash) != 0) &&
            g_library_programs.contains(program->info.pgm_hash) != g_stage_from_library) {
            ++(g_stage_from_library ? g_library_stages_left : g_cache_stages_left);
            return false;
        }
    }

    std::vector<u32> spv{};
    std::vector<u8> bytes;
    if (iss::pipeline_library::Read(fmt::format("{:#018x}_{}.spv", program->info.pgm_hash, perm_idx), bytes)) {
        spv.resize(bytes.size() / sizeof(u32));
        std::memcpy(spv.data(), bytes.data(), bytes.size());
    } else {
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                           fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx), spv);
    }
    // ISS_AOT_RECIPES / ISS_AOT_RECIPE_VERIFY: the stage's SPIR-V made from the game files.
    static const bool recipes = [] {
        const char* value = std::getenv("ISS_AOT_RECIPES");
        return value && value[0] == '1';
    }();
    static const bool verify = [] {
        const char* value = std::getenv("ISS_AOT_RECIPE_VERIFY");
        return value && value[0] == '1';
    }();
    if (recipes && spv.empty()) {
        std::scoped_lock lock{recipe_mutex};
        if (const auto it = recipe_spirv.find(expected_hash); it != recipe_spirv.end()) {
            spv = std::move(it->second);
            recipe_spirv.erase(it);
            ++recipe_stats[std::min(int(program->info.sw_stage), 5)].made;
            RegisterShaderBinary(std::vector<u32>(spv), program->info.pgm_hash, perm_idx);
        }
    }
    if ((recipes && spv.empty()) || (verify && !spv.empty())) {
        const auto variants = RecipeAttribVariants(expected_hash);
        auto made = RegenerateRecipeSpirv(profile, program->info, spec, variants);
        const int stage = int(program->info.sw_stage);
        auto& st = recipe_stats[std::min(stage, 5)];
        if (!made) {
            ++st.failed;
        } else if (spv.empty()) {
            ++st.made;
            spv = *made;
            RegisterShaderBinary(std::vector<u32>(spv), program->info.pgm_hash, perm_idx);
        } else if (*made == spv) {
            ++st.matched;
            recipe_verified.push_back(expected_hash);
        } else if (auto found = program->info.sw_stage == Shader::SwStage::Vertex
                                    ? RecoverAttribVariants(profile, program->info, spec, spv)
                                    : std::nullopt) {
            ++st.recovered;
            RememberAttribVariants(expected_hash, std::move(*found));
            recipe_verified.push_back(expected_hash);
        } else {
            ++st.mismatched;
            if (const char* dir = std::getenv("ISS_AOT_RECIPE_DUMP"); dir && st.mismatched <= 6) {
                const auto base = std::filesystem::path{dir} /
                                  fmt::format("{}_{:#018x}_{}", stage, program->info.pgm_hash, perm_idx);
                std::filesystem::create_directories(dir);
                Common::FS::IOFile{base.string() + "_lib.spv", Common::FS::FileAccessMode::Create}
                    .WriteSpan(std::span<const u32>{spv});
                Common::FS::IOFile{base.string() + "_new.spv", Common::FS::FileAccessMode::Create}
                    .WriteSpan(std::span<const u32>{*made});
            }
        }
    }
    if (spv.empty()) {
        return false;
    }
    // ISS_COMPILE_TRACE=1 (diagnosis): each stage's module creation, flushed (see the pipeline
    // compile trace in vk_graphics_pipeline.cpp).
    static const bool stage_trace = [] {
        const char* p = std::getenv("ISS_COMPILE_TRACE");
        return p && p[0] == '1';
    }();
    if (stage_trace) {
        std::fprintf(stderr, "STAGE_BEGIN hash=%llx pgm=%llx perm=%zu library=%d words=%zu\n",
                     (unsigned long long)expected_hash, (unsigned long long)program->info.pgm_hash,
                     perm_idx, int(g_stage_from_library), spv.size());
        std::fflush(stderr);
    }
    SCOPE_EXIT {
        if (stage_trace) {
            std::fputs("STAGE_END\n", stderr);
            std::fflush(stderr);
        }
    };

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    // The scout thread looks programs and their permutations up meanwhile.
    std::unique_lock lock{program_cache_mutex};
    if (g_stage_from_library && !(program_cache.count(program->info.pgm_hash) != 0)) {
        g_library_programs.insert(program->info.pgm_hash);
    }
    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        { std::vector<u32> salted; module = CompileSPV(iss::shader_salt::Apply(spv, salted), instance.GetDevice()); }
        it_pgm.value() = std::move(program);
    } else {
        const auto& saved = it_pgm.value()->modules;
        if (perm_idx < saved.size() && saved[perm_idx].module) {
            if (!preload_identity.Matches(expected_hash, metadata)) {
                LOG_WARNING(Render_Vulkan, "Conflicting metadata at cached shader key {:#x}",
                            expected_hash);
                return false;
            }
            module = saved[perm_idx].module;
        } else {
            { std::vector<u32> salted; module = CompileSPV(iss::shader_salt::Apply(spv, salted), instance.GetDevice()); }
        }
    }
    // The temporary Program above is destroyed for an existing shader. The
    // specialization must point to the Program that actually owns this slot.
    spec.info = &it_pgm.value()->info;
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    preload_identity.Remember(expected_hash, std::move(metadata));
    preload_stages.emplace(expected_hash, std::pair{it_pgm.value().get(), perm_idx});

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;
    if (auto& fetch = it_pgm.value()->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }

    return true;
}

void PipelineCache::PrepareRecipeStages() {
    // The recipe stages whose SPIR-V neither the pack nor the cache has (the first run).
    std::vector<std::pair<u64, const std::vector<u8>*>> metas;
    std::set<std::string> names;
    iss::pipeline_library::ForEachFile([&](const std::string& name, const std::vector<u8>& bytes) {
        names.insert(name);
        if (name.ends_with(".meta") && bytes.size() >= 24) {
            u64 hash;
            std::memcpy(&hash, bytes.data() + 8, 8);
            metas.emplace_back(hash, &bytes);
        }
    });
    if (metas.empty()) return;
    const u64 start = Common::Perf::NowNs();
    iss::g_preload_total.store(unsigned(metas.size()), std::memory_order_relaxed);
    iss::g_preload_entries.store(0, std::memory_order_relaxed);
    iss::g_preload_phase.store(4, std::memory_order_relaxed);
    std::atomic<size_t> next{0}, made{0}, cached{0}, failed{0};
    std::mutex db_mutex;
    const u32 threads = std::max(1U, std::thread::hardware_concurrency() - 1);
    {
        std::vector<std::jthread> workers;
        for (u32 t = 0; t < threads; ++t) {
            workers.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < metas.size();) {
                    iss::g_preload_entries.fetch_add(1, std::memory_order_relaxed);
                    const auto [perm_hash, bytes] = metas[i];
                    Serialization::Archive ar{std::vector<u8>(*bytes)};
                    Shader::Info info{};
                    Shader::StageSpecialization spec{};
                    spec.info = &info;
                    size_t perm_idx{};
                    if (!LoadShaderMeta(ar, info, spec, perm_idx)) {
                        ++failed;
                        continue;
                    }
                    const auto spv_name = fmt::format("{:#018x}_{}", info.pgm_hash, perm_idx);
                    if (names.contains(spv_name + ".spv")) {
                        ++cached;
                        continue;
                    }
                    {
                        std::vector<u32> existing;
                        std::scoped_lock lock{db_mutex};
                        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary, spv_name,
                                                           existing);
                        if (!existing.empty()) {
                            ++cached;
                            continue;
                        }
                    }
                    const auto variants = RecipeAttribVariants(perm_hash);
                    auto spv = RegenerateRecipeSpirv(profile, info, spec, variants);
                    if (!spv) {
                        ++failed;
                        continue;
                    }
                    ++made;
                    std::scoped_lock lock{recipe_mutex};
                    recipe_spirv.emplace(perm_hash, std::move(*spv));
                }
            });
        }
    }
    iss::g_preload_phase.store(1, std::memory_order_relaxed);
    std::fprintf(stderr, "AOT_RECIPE prepared %zu stages from the game files in %.1f s (%zu cached, %zu failed, %u threads)\n",
                 made.load(), double(Common::Perf::NowNs() - start) / 1e9, cached.load(), failed.load(),
                 threads);
}

void PipelineCache::WarmUp() {
    preload_identity.Clear();
    preload_stages.clear();
    preload_stage_hits = 0;
    SCOPE_EXIT { preload_identity.Clear(); preload_stages.clear(); };
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();
    const auto primary = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                         Common::ElfInfo::Instance().GameSerial();
    // Archived caches cannot be checked for conflicting permutation indices by
    // the directory library. Keep their existing behavior unchanged.
    const bool library_ready = !EmulatorSettings.IsPipelineCacheArchived() &&
        iss::pipeline_library::Open({reinterpret_cast<const u8*>(&profile), sizeof(profile)}, primary);
    bool library_finished = false;
    SCOPE_EXIT {
        if (!library_finished) iss::pipeline_library::Discard();
    };

    // Starts a fresh cache for the current profile, so it is filled during this session.
    const auto start_new_cache = [this] {
        Storage::DataBase::Instance().FinishPreload();
        std::vector<u8> new_profile(sizeof(profile));
        std::memcpy(new_profile.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(new_profile));
    };

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    const bool seed_profile = profile_data.empty() && library_ready;
    if (seed_profile) iss::pipeline_library::Read("profile.bin", profile_data);
    if (profile_data.empty()) {
        start_new_cache();
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        // Usually happens after an emulator update. Previously the cache was only ignored,
        // which also disabled saving, so it was never rebuilt and every session recompiled
        // every shader.
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Rebuilding the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Reset();
        start_new_cache();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Rebuilding the cache");
        Storage::DataBase::Instance().Reset();
        start_new_cache();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};
    const u64 preload_start = Common::Perf::NowNs();
    u64 last_progress = preload_start;
    LOG_INFO(Render, "Pipeline preload started; parallel={}", parallel_preload);
    {
        // The loading screen's total: the cache's keys (files *.key under the cache folder, as
        // the launcher counts them) and the shader library's.
        u64 own_keys = 0;
        std::error_code ec;
        const auto root = Common::FS::GetUserPath(Common::FS::PathType::CacheDir);
        for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
             !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (it.depth() > 2) it.disable_recursion_pending();
            if (it->path().extension() == ".key") ++own_keys;
        }
        iss::g_preload_total.store(unsigned(own_keys + iss::pipeline_library::NumKeys()),
                                   std::memory_order_relaxed);
    }
    iss::g_preload_phase.store(1, std::memory_order_relaxed);

    const auto preload_key = [&](std::vector<u8>&& data) {
            ++num_total_pipelines;
            iss::g_preload_entries.fetch_add(1, std::memory_order_relaxed);
            if (data.size() < 16) return;
            const u64 now = Common::Perf::NowNs();
            if (now - last_progress >= 1'000'000'000ULL) {
                LOG_INFO(Render, "Pipeline preload: {} entries, {} accepted, {} reused stages, {:.1f} s",
                         num_total_pipelines, num_pipelines, preload_stage_hits,
                         double(now - preload_start) / 1e9);
                last_progress = now;
            }

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        };
    if (const char* value = std::getenv("ISS_AOT_RECIPES"); value && value[0] == '1' && library_ready) {
        PrepareRecipeStages();
    }
    Storage::DataBase::Instance().ForEachBlob(Storage::BlobType::PipelineKey, preload_key);
    const u32 normal_accepted = num_pipelines;
    const u32 normal_total = num_total_pipelines;
    iss::pipeline_library::ForEachKey(preload_key);

    if (parallel_preload) {
        // ISS_PRELOAD_BACKGROUND=1 (with ISS_ASYNC_SKIP_DRAWS): the game starts once every
        // pipeline is read; their compiles go on in the background, the draws of those not
        // compiled yet skipped and the dispatches waiting for theirs as for any new one.
        static const bool background = [] {
            const char* value = std::getenv("ISS_PRELOAD_BACKGROUND");
            return value && value[0] == '1';
        }();
        const double read_s = double(Common::Perf::NowNs() - preload_start) / 1e9;
        if (background) {
            LOG_INFO(Render, "Pipeline preload: read in {:.2f} s; compiles go on in the background",
                     read_s);
        } else {
            LOG_INFO(Render,
                     "Pipeline preload: read in {:.2f} s; finishing compiler work before starting "
                     "the game",
                     read_s);
            iss::g_preload_phase.store(2, std::memory_order_relaxed);
            WaitPreload();
        }
    }
    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (library_ready)
        std::fprintf(stderr, "PIPELINE_LIBRARY programs=%zu library_stages_left_to_cache=%u "
                     "cache_stages_left_to_library=%u\n", g_library_programs.size(),
                     g_library_stages_left, g_cache_stages_left);
    for (int stage = 0; stage < 6; ++stage) {
        const auto& st = recipe_stats[stage];
        if (st.made + st.matched + st.mismatched + st.failed + st.recovered == 0) continue;
        std::fprintf(stderr,
                     "AOT_RECIPE stage=%d made=%u matched=%u recovered_attribs=%u mismatched=%u "
                     "failed=%u\n",
                     stage, st.made, st.matched, st.recovered, st.mismatched, st.failed);
    }
    SaveAttribVariants();
    if (const char* path = std::getenv("ISS_AOT_RECIPE_VERIFIED"); path && *path && !recipe_verified.empty()) {
        Common::FS::IOFile file{std::filesystem::path{path}, Common::FS::FileAccessMode::Create};
        file.WriteSpan(std::span<const u64>{recipe_verified});
    }
    LOG_INFO(Render, "Pipeline preload finished in {:.2f} s; {} repeated stage loads avoided",
             double(Common::Perf::NowNs() - preload_start) / 1e9, preload_stage_hits);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
    if (seed_profile) {
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
    }
    iss::pipeline_library::Finish(num_pipelines - normal_accepted,
        (num_total_pipelines - normal_total) - (num_pipelines - normal_accepted));
    library_finished = true;
    iss::g_preload_phase.store(3, std::memory_order_relaxed);
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(walker_func_size);
    srt.Write(flattened_bufsize_dw);
    srt.Write(capture_walker_func_size);
    srt.Write(capture_schema);
    srt.Write(capture_relocations);
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
    if (capture_walker_func_size) {
        srt.Write(reinterpret_cast<void*>(capture_walker_func), capture_walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    if (Serialization::loading_legacy_srt) {
        // Version 6 serialized the entire 24-byte POD, including a process-local
        // function pointer and its trailing padding. Never reuse that pointer.
        struct LegacySrtLayout {
            PFN_SrtWalker process_local_walker;
            size_t code_size;
            u32 flattened_words;
            u32 padding;
        } legacy{};
        static_assert(sizeof(LegacySrtLayout) == 24);
        srt.Read(&legacy, sizeof(legacy));
        walker_func_size = legacy.code_size;
        flattened_bufsize_dw = legacy.flattened_words;
    } else {
        srt.Read(walker_func_size);
        srt.Read(flattened_bufsize_dw);
        srt.Read(capture_walker_func_size);
        srt.Read(capture_schema);
        srt.Read(capture_relocations);
    }

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    if (capture_walker_func_size) {
        capture_walker_func = RegisterCaptureWalkerCode(ar.CurrPtr(), capture_walker_func_size,
                                                        capture_relocations);
        ar.Advance(capture_walker_func_size);
        if (!capture_walker_func) return false;
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (!fetch_shader_data.Empty()) {
        spec.Write(sizeof(fetch_shader_data));
        fetch_shader_data.Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
