// SPDX-License-Identifier: GPL-2.0-or-later
#define NOMINMAX
#include <windows.h>
#include "runtime_timeline.h"
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "common/perf_stats.h"
#include "pipeline_library.h"

namespace iss::pipeline_library {
namespace {
std::map<std::string, std::vector<u8>> files;
std::vector<std::string> extra_keys;
bool enabled{};
constexpr size_t MaxFile = 32ULL * 1024 * 1024;
constexpr size_t MaxTotal = 1024ULL * 1024 * 1024; // a merged development library is ~550 MB
constexpr size_t GraphicsKeyMinBytes = 1592; // current MSVC x64 serialization ABI

u32 Word32(const std::vector<u8>& bytes, size_t offset) {
    u32 value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
u64 Word64(const std::vector<u8>& bytes, size_t offset) {
    u64 value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
u64 HashCombine(u64 shader, u64 permutation) {
    return shader ^ (permutation + 0x9e3779b9ULL + (shader << 12) + (shader >> 4));
}
/// cache_variants: the stages the library left to the cache (their permutation, from the cache's
/// metadata), which the library's pipeline keys may name.
void ValidateRecipes(const std::map<std::string, std::vector<u8>>& snapshot,
                     const std::map<u64, u64>& cache_variants) {
    std::map<u64, u64> variants = cache_variants;
    for (const auto& [name, bytes] : snapshot) {
        if (!name.ends_with(".spv")) continue;
        const auto underscore = name.find('_');
        const auto dot = name.rfind('.');
        const u64 shader = std::stoull(name.substr(2, 16), nullptr, 16);
        const auto decimal = name.substr(underscore + 1, dot - underscore - 1);
        const u64 permutation = std::stoull(decimal);
        if (std::to_string(permutation) != decimal)
            throw std::runtime_error("noncanonical SPIR-V permutation index");
        const u64 hash = HashCombine(shader, permutation);
        if (!variants.emplace(hash, permutation).second)
            throw std::runtime_error("colliding shader permutation hash");
        char meta_name[32];
        std::snprintf(meta_name, sizeof(meta_name), "0x%016llx.meta", static_cast<unsigned long long>(hash));
        if (!snapshot.contains(meta_name)) throw std::runtime_error("SPIR-V missing metadata");
    }
    // ISS_AOT_RECIPES: a recipe pack has no SPIR-V (made from the game files when loaded): its
    // metadata names the permutation itself.
    static const bool recipes = [] {
        const char* value = std::getenv("ISS_AOT_RECIPES");
        return value && value[0] == '1';
    }();
    if (recipes) {
        for (const auto& [name, bytes] : snapshot) {
            if (!name.ends_with(".meta") || bytes.size() < 24) continue;
            const u64 hash = std::stoull(name.substr(2, 16), nullptr, 16);
            if (Word64(bytes, 8) == hash) variants.emplace(hash, Word64(bytes, 16));
        }
    }
    for (const auto& [name, bytes] : snapshot) {
        if (!name.ends_with(".meta")) continue;
        const u64 hash = std::stoull(name.substr(2, 16), nullptr, 16);
        const auto variant = variants.find(hash);
        // Metadata v6 (older packs) or v7 (what the cache writes now, with the SRT capture
        // fields): LoadShaderMeta decodes both, the header words checked here are the same.
        if (variant == variants.end() || bytes.size() < 24 ||
            (Word32(bytes, 0) != 6 && Word32(bytes, 0) != 7) ||
            Word32(bytes, 4) % 100 != 8 || Word64(bytes, 8) != hash ||
            Word64(bytes, 16) != variant->second)
            throw std::runtime_error("invalid or unpaired shader metadata");
    }
    for (const auto& [name, bytes] : snapshot) {
        if (!name.ends_with(".key")) continue;
        if (bytes.size() < 16 || Word32(bytes, 0) != 3)
            throw std::runtime_error("invalid pipeline key");
        if (name.starts_with("c_")) {
            const u64 hash = Word64(bytes, 8);
            char expected[32];
            std::snprintf(expected, sizeof(expected), "c_0x%016llx.key", static_cast<unsigned long long>(hash));
            if (Word32(bytes, 4) != 1 || bytes.size() != 16 || name != expected ||
                !variants.contains(hash))
                throw std::runtime_error("invalid or incomplete compute recipe");
        } else {
            if (Word32(bytes, 4) != 0 || bytes.size() < GraphicsKeyMinBytes)
                throw std::runtime_error("invalid graphics recipe");
            for (size_t stage = 0; stage < 6; ++stage) {
                const u64 hash = Word64(bytes, 8 + stage * 8);
                if (hash != 0 && !variants.contains(hash))
                    throw std::runtime_error("incomplete graphics recipe");
            }
        }
    }
}

std::vector<u8> ReadFile(const std::filesystem::path& path, size_t limit = MaxFile) {
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
        throw std::runtime_error("missing/non-regular recipe file");
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > limit) throw std::runtime_error("recipe size out of bounds");
    std::vector<u8> bytes(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("short recipe read");
    return bytes;
}
std::string Sha256(std::span<const u8> bytes) {
    BCRYPT_ALG_HANDLE alg{};
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("SHA256 unavailable");
    u8 digest[32];
    const auto result = BCryptHash(alg, nullptr, 0, const_cast<u8*>(bytes.data()),
                                  static_cast<ULONG>(bytes.size()), digest, sizeof(digest));
    BCryptCloseAlgorithmProvider(alg, 0);
    if (result < 0) throw std::runtime_error("SHA256 failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string hash; hash.reserve(64);
    for (const auto b : digest) { hash += hex[b >> 4]; hash += hex[b & 15]; }
    return hash;
}
std::vector<u8> CheckedFile(const std::filesystem::path& root, const nlohmann::json& item,
                            bool profile = false, std::span<const u8> packed = {}) {
    const std::string name = item.at("filename").get<std::string>();
    static const std::regex asset(R"((?:[cg]_0x[0-9a-f]{16}\.key|0x[0-9a-f]{16}\.meta|0x[0-9a-f]{16}_[0-9]+\.spv))");
    if (profile ? name != "profile.bin" : !std::regex_match(name, asset))
        throw std::runtime_error("invalid recipe filename");
    std::vector<u8> bytes;
    if (!packed.empty() && !profile) {
        const auto offset = item.at("offset").get<size_t>();
        const auto size = item.at("size").get<size_t>();
        if (!size || size > MaxFile || offset > packed.size() || size > packed.size() - offset)
            throw std::runtime_error("packed recipe outside bounds");
        bytes.assign(packed.begin() + offset, packed.begin() + offset + size);
    } else {
        bytes = ReadFile(root / name);
    }
    if (bytes.size() != item.at("size").get<size_t>() || Sha256(bytes) != item.at("sha256").get<std::string>())
        throw std::runtime_error("recipe integrity mismatch");
    if (!profile) {
        u32 tag{}; if (bytes.size() >= 4) std::memcpy(&tag, bytes.data(), 4);
        if (name.ends_with(".key") && (bytes.size() < 16 || tag != 3))
            throw std::runtime_error("unsupported recipe key version");
        if (name.ends_with(".meta") && (bytes.size() < 24 || (tag != 6 && tag != 7)))
            throw std::runtime_error("unsupported recipe metadata version");
        if (name.ends_with(".spv") && (bytes.size() < 20 || bytes.size() % 4 || tag != 0x07230203))
            throw std::runtime_error("invalid recipe SPIR-V");
    }
    return bytes;
}
} // namespace

/// A recipe of a compute shader: its pipeline key, its metadata (runtime info stage) or its
/// SPIR-V (GLCompute entry point).
static bool IsComputeAsset(const std::string& name, std::span<const u8> bytes) {
    if (name.starts_with("c_")) return true;
    if (name.ends_with(".meta")) {
        u32 stage{};
        if (bytes.size() < 36) return false;
        std::memcpy(&stage, bytes.data() + 32, 4); // RuntimeInfo::sw_stage after the bindings
        return stage == 5;                         // SwStage::Compute
    }
    if (name.ends_with(".spv")) {
        const size_t words = bytes.size() / 4;
        for (size_t i = 5; i + 1 < words;) {
            u32 w, model;
            std::memcpy(&w, bytes.data() + i * 4, 4);
            const u32 count = w >> 16;
            if ((w & 0xFFFF) == 15) { // OpEntryPoint
                std::memcpy(&model, bytes.data() + (i + 1) * 4, 4);
                return model == 5; // GLCompute
            }
            if (count == 0) break;
            i += count;
        }
    }
    return false;
}

bool Open(std::span<const u8> profile, const std::filesystem::path& primary) {
    files.clear(); extra_keys.clear(); enabled = false;
    const char* option = std::getenv("ISS_PIPELINE_LIBRARY");
    if (!option || !*option) return false;
    const bool skip_compute = [] {
        const char* stride = std::getenv("ISS_DYNAMIC_STRIDE");
        const char* aot = std::getenv("ISS_AOT_COMPUTE");
        const char* stats = std::getenv("ISS_AOT_FORMAT_STATS");
        return (stride && stride[0] == '1') ||
               (aot && aot[0] && !(stats && stats[0] == '1'));
    }();
    const auto started = Common::Perf::NowNs();
    try {
        const auto root = std::filesystem::canonical(std::filesystem::u8path(option));
        const auto manifest_bytes = ReadFile(root / "manifest.json", 32 * 1024 * 1024);
        const auto manifest = nlohmann::json::parse(manifest_bytes);
        if (manifest.at("schema").get<int>() != 1) throw std::runtime_error("unsupported manifest schema");
        auto saved_profile = CheckedFile(root, manifest.at("profile"), true);
        if (!std::ranges::equal(saved_profile, profile)) throw std::runtime_error("incompatible shader profile");
        std::vector<u8> packed;
        if (manifest.contains("pack")) {
            const auto& pack = manifest.at("pack");
            if (pack.at("filename").get<std::string>() != "recipes.bin")
                throw std::runtime_error("invalid pack filename");
            packed = ReadFile(root / "recipes.bin", MaxTotal);
            if (packed.size() != pack.at("size").get<size_t>() || Sha256(packed) != pack.at("sha256").get<std::string>())
                throw std::runtime_error("packed library integrity mismatch");
        }
        std::set<std::string> primary_names;
        if (std::filesystem::is_directory(primary)) {
            for (const auto& item : std::filesystem::directory_iterator(primary))
                primary_names.insert(item.path().filename().string());
        }
        std::map<std::string, std::vector<u8>> snapshot;
        size_t total = saved_profile.size();
        size_t pack_offset = 0;
        size_t kept_from_cache = 0;
        std::map<u64, u64> cache_variants;
        const auto& assets = manifest.at("assets");
        if (!assets.is_array() || assets.size() > 50000) throw std::runtime_error("asset count out of bounds");
        for (const auto& item : assets) {
            if (!packed.empty() && item.at("offset").get<size_t>() != pack_offset)
                throw std::runtime_error("noncontiguous packed recipes");
            auto bytes = CheckedFile(root, item, false, packed);
            pack_offset += bytes.size();
            const auto name = item.at("filename").get<std::string>();
            total += bytes.size();
            if (total > MaxTotal) throw std::runtime_error("library exceeds memory budget");
            // ISS_DYNAMIC_STRIDE / ISS_AOT_COMPUTE: compute shaders are compiled with another
            // recompiler variant (and by the AOT compiler into the cache): the library's are
            // stale; its graphics recipes are kept.
            if (skip_compute && IsComputeAsset(name, bytes)) continue;
            // Never shadow a normal cache file: a name the cache has with other bytes is left to
            // the cache (its own metadata and SPIR-V load as without a library). The name is
            // HashCombine(shader, permutation), so it is the same stage of the same lineage; the
            // bytes differ when the runtime rewrote it in a newer format (recipes ship .meta v6,
            // the cache writes v7). Dropping the whole library for that turned every recipe off
            // from the second session on (2096 of 19054 names, 2026-10-19); taking the cache's
            // bytes into the library made those stages recipes, regenerated from v7 metadata
            // without their attribute formats: SPIR-V that crashed the driver (0xC0000409).
            if (primary_names.contains(name)) {
                const auto own = ReadFile(primary / name);
                if (own != bytes) {
                    ++kept_from_cache;
                    if (name.ends_with(".meta") && own.size() >= 24) {
                        const u64 hash = std::stoull(name.substr(2, 16), nullptr, 16);
                        if (Word64(own, 8) == hash) cache_variants.emplace(hash, Word64(own, 16));
                    }
                    continue;
                }
            }
            if (!snapshot.emplace(name, std::move(bytes)).second)
                throw std::runtime_error("duplicate recipe filename");
        }
        if (!packed.empty() && pack_offset != packed.size())
            throw std::runtime_error("trailing bytes in recipe pack");
        ValidateRecipes(snapshot, cache_variants);
        for (const auto& [name, bytes] : snapshot) {
            if (!name.ends_with(".key")) continue;
            if (!primary_names.contains(name)) extra_keys.push_back(name);
        }
        snapshot.emplace("profile.bin", std::move(saved_profile));
        files = std::move(snapshot); enabled = true;
        std::fprintf(stderr, "PIPELINE_LIBRARY ready=1 assets=%zu additional_keys=%zu bytes=%zu left_to_cache=%zu prepare_ms=%.3f\n",
                     assets.size(), extra_keys.size(), total, kept_from_cache,
                     double(Common::Perf::NowNs()-started)/1e6);
        return true;
    } catch (const std::exception& e) {
        files.clear(); extra_keys.clear();
        std::fprintf(stderr, "PIPELINE_LIBRARY ready=0 fallback=normal reason=%s\n", e.what());
        return false;
    }
}
bool Read(const std::string& filename, std::vector<u8>& out) {
    const auto found = files.find(filename);
    if (!enabled || found == files.end()) return false;
    out = found->second; return true;
}
void ForEachKey(const std::function<void(std::vector<u8>&&)>& callback) {
    for (const auto& name : extra_keys) { auto data = files.at(name); callback(std::move(data)); }
}
size_t NumKeys() { return extra_keys.size(); }
void ForEachFile(const std::function<void(const std::string&, const std::vector<u8>&)>& callback) {
    if (!enabled) return;
    for (const auto& [name, bytes] : files) callback(name, bytes);
}
void Finish(u32 accepted, u32 rejected) {
    if (enabled) std::fprintf(stderr, "PIPELINE_LIBRARY warmup_complete=1 accepted=%u rejected=%u pending=0\n",accepted,rejected);
    // All metadata/modules are now in PipelineCache. Runtime never reads this snapshot.
    files.clear(); extra_keys.clear(); enabled = false;
}
void Discard() {
    files.clear(); extra_keys.clear(); enabled = false;
}
void LogMiss(const char* kind, u64 key, u64 guest_hash, u64 duration_ns) {
    iss::timeline::Emit("pipeline_miss",key,guest_hash,static_cast<u64>(kind[0]),0,0,0,duration_ns);
    static const bool log = [] { const char* p=std::getenv("ISS_PIPELINE_EVENTS"); return p&&p[0]=='1'; }();
    if (!log) return;
    std::fprintf(stderr, "PIPELINE_MISS kind=%s key=0x%016llx guest_hash=0x%016llx frame=%llu ns=%llu elapsed_ms=%.3f\n",
                 kind, static_cast<unsigned long long>(key), static_cast<unsigned long long>(guest_hash),
                 static_cast<unsigned long long>(Common::Perf::GuestFrameCounter().load()),
                 static_cast<unsigned long long>(Common::Perf::NowNs()), double(duration_ns)/1e6);
}
} // namespace iss::pipeline_library
