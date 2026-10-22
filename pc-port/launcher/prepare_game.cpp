// SPDX-License-Identifier: GPL-2.0-or-later
// Game preparation for the launcher: the C++ port of scripts/ps4_format.py (parse_self),
// scripts/prepare.py (module) and scripts/link_modules.py (build_linked). It links the user's
// own eboot.bin, sce_module/libc.prx and sce_module/libSceFios2.prx by exact, versioned PS4
// identity into the ISSPRB02 image iss-probe.exe runs (boot-linked.bin). Byte identical to the
// Python output. The game files are only read; nothing in the game folder is written.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace iss_prepare {
namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s64 = std::int64_t;
using Bytes = std::vector<u8>;

constexpr const char* ExpectedEboot = "2d1ca79630d7bbe6fa29575f59aa7996d43071d74ac8e4f7d40137967d9039ab";

[[noreturn]] void Fail(const std::string& why) { throw std::runtime_error(why); }

void Check(const Bytes& data, u64 offset, u64 size) {
    if (offset > data.size() || size > data.size() - offset) Fail("out of bounds read");
}
template <typename T> T Read(const Bytes& data, u64 offset) {
    Check(data, offset, sizeof(T));
    T v;
    std::memcpy(&v, data.data() + offset, sizeof(T));
    return v;
}

std::string Sha256Hex(const Bytes& data) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    u8 digest[32] = {};
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) Fail("SHA-256 unavailable");
    BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0);
    // In 256 MB pieces (ULONG lengths).
    for (u64 at = 0; at < data.size(); at += 256u << 20) {
        const u64 n = std::min<u64>(256u << 20, data.size() - at);
        BCryptHashData(hash, const_cast<u8*>(data.data() + at), ULONG(n), 0);
    }
    BCryptFinishHash(hash, digest, sizeof(digest), 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (u8 b : digest) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
}

struct Ph {
    u32 type, flags;
    u64 offset, vaddr, paddr, filesz, memsz, align;
};
struct Identity {
    std::string name;
    u32 version_raw;
    bool operator==(const Identity& o) const { return name == o.name && version_raw == o.version_raw; }
};
struct Symbol {
    std::string name;
    u8 type, binding;
    u16 section;
    u64 value, size;
    bool has_identity = false;
    std::string nid;
    Identity library, module;
};
struct Reloc {
    u64 target;
    u32 kind;
    u64 sym;
    s64 addend;
};
struct Module {
    std::string path;
    std::string sha256;
    Bytes elf;
    u64 entry = 0;
    std::vector<Ph> ph;
    std::vector<Symbol> symbols;
    std::vector<Reloc> relocs;
    std::map<u64, u64> tags; // dynamic tags (first value of each, as Python dict(dynamic))
};

// The identity key of a symbol: (nid, library, library version, module, module version).
using Key = std::tuple<std::string, std::string, u32, std::string, u32>;
std::optional<Key> IdentityKey(const Symbol& s) {
    if (!s.has_identity) return std::nullopt;
    return Key{s.nid, s.library.name, s.library.version_raw, s.module.name, s.module.version_raw};
}
std::string KeyText(const Key& k) {
    return std::get<0>(k) + "#" + std::get<1>(k) + "#" + std::to_string(std::get<2>(k)) + "#" +
           std::get<3>(k) + "#" + std::to_string(std::get<4>(k));
}
struct KeyHash {
    size_t operator()(const Key& k) const { return std::hash<std::string>()(KeyText(k)); }
};

std::string EncodeId(u64 value) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string encoded(1, alphabet[value & 63]);
    while (value >> 6) {
        value >>= 6;
        encoded.insert(encoded.begin(), alphabet[value & 63]);
    }
    return encoded;
}

Bytes ReadFile(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) Fail("cannot read " + path.filename().string());
    f.seekg(0, std::ios::end);
    const auto size = u64(f.tellg());
    f.seekg(0);
    Bytes data(size);
    f.read(reinterpret_cast<char*>(data.data()), std::streamsize(size));
    return data;
}

// ps4_format.parse_self: the ELF rebuilt from a plaintext PS4 SELF.
void ParseSelf(const Bytes& data, Module& m) {
    if (Read<u32>(data, 0) != 0x1D3D154Fu) Fail("expected PS4 SELF");
    const u16 count = Read<u16>(data, 24);
    const u64 base = 32 + u64(count) * 32;
    Check(data, base, 64);
    const u8* ident = data.data() + base;
    if (std::memcmp(ident, "\x7f" "ELF\x02\x01\x01", 7) != 0 || Read<u16>(data, base + 18) != 62)
        Fail("expected little-endian x86-64 ELF");
    m.entry = Read<u64>(data, base + 24);
    const u64 phoff = Read<u64>(data, base + 32);
    const u16 phentsize = Read<u16>(data, base + 54), phnum = Read<u16>(data, base + 56);
    if (phentsize != 56 || !(0 < phnum && phnum < 256)) Fail("unsupported program headers");
    for (u32 i = 0; i < phnum; ++i) {
        const u64 at = base + phoff + u64(i) * 56;
        Ph p{Read<u32>(data, at), Read<u32>(data, at + 4), Read<u64>(data, at + 8), Read<u64>(data, at + 16),
             Read<u64>(data, at + 24), Read<u64>(data, at + 32), Read<u64>(data, at + 40), Read<u64>(data, at + 48)};
        m.ph.push_back(p);
    }
    u64 end = 0;
    for (const auto& p : m.ph) end = std::max(end, p.offset + p.filesz);
    if (end > 512ull * 1024 * 1024) Fail("image exceeds 512 MiB");
    m.elf.assign(end, 0);
    const u64 header_end = phoff + u64(phnum) * 56;
    Check(data, base, header_end);
    if (header_end > end) Fail("program headers outside the image");
    std::memcpy(m.elf.data(), data.data() + base, header_end);
    std::vector<std::pair<u64, u64>> covered{{0, header_end}};
    for (u32 i = 0; i < count; ++i) {
        const u64 at = 32 + u64(i) * 32;
        const u64 flags = Read<u64>(data, at), off = Read<u64>(data, at + 8), size = Read<u64>(data, at + 16),
                  mem = Read<u64>(data, at + 24);
        if (!(flags & 0x800)) continue;
        if (flags & 10) Fail("encrypted/compressed SELF segment is unsupported (decrypted dump needed)");
        const u64 index = (flags >> 20) & 4095;
        if (index >= m.ph.size()) Fail("invalid segment index");
        const Ph& p = m.ph[index];
        if (size != p.filesz || size != mem) Fail("unsupported blocked segment layout");
        Check(data, off, size);
        if (p.offset + size > m.elf.size()) Fail("segment outside the image");
        std::memcpy(m.elf.data() + p.offset, data.data() + off, size);
        covered.push_back({p.offset, p.offset + size});
    }
    for (size_t i = 0; i < m.ph.size(); ++i) {
        const Ph& p = m.ph[i];
        if (!p.filesz) continue;
        const bool in = std::any_of(covered.begin(), covered.end(), [&](auto& c) {
            return c.first <= p.offset && p.offset + p.filesz <= c.second;
        });
        if (!in && (p.type == 1 || p.type == 2 || p.type == 0x61000000 || p.type == 0x61000010))
            Fail("required segment " + std::to_string(i) + " is unavailable");
    }
}

// prepare.module: dynamic tags, libraries, symbols and relocations.
Module LoadModule(const fs::path& path, bool hash) {
    Module m;
    m.path = path.string();
    const Bytes source = ReadFile(path);
    if (hash) m.sha256 = Sha256Hex(source);
    ParseSelf(source, m);
    const Ph* dyn = nullptr;
    const Ph* lib = nullptr;
    for (const auto& p : m.ph) {
        if (p.type == 2 && !dyn) dyn = &p;
        if (p.type == 0x61000000 && !lib) lib = &p;
    }
    if (!dyn || !lib) Fail("missing dynamic segments");
    if (dyn->filesz % 16) Fail("invalid dynamic table length");
    std::vector<std::pair<u64, u64>> dynamic;
    for (u64 pos = dyn->offset; pos < dyn->offset + dyn->filesz; pos += 16) {
        const u64 tag = Read<u64>(m.elf, pos), value = Read<u64>(m.elf, pos + 8);
        if (!tag) break;
        dynamic.push_back({tag, value});
    }
    for (auto& [t, v] : dynamic) m.tags[t] = v; // the last value wins, as dict(dynamic)
    auto tag = [&](u64 t, u64 fallback) {
        const auto it = m.tags.find(t);
        return it == m.tags.end() ? fallback : it->second;
    };
    if (tag(0x6100003b, 24) != 24 || tag(0x61000033, 24) != 24) Fail("unexpected symbol/relocation entry size");
    Check(m.elf, lib->offset, lib->filesz);
    const Bytes blob(m.elf.begin() + lib->offset, m.elf.begin() + lib->offset + lib->filesz);
    const u64 str_off = m.tags.at(0x61000035), str_size = m.tags.at(0x61000037);
    Check(blob, str_off, str_size);
    auto string = [&](u64 off) {
        if (off >= str_size) Fail("string outside the table");
        const u8* s = blob.data() + str_off + off;
        const u8* e = static_cast<const u8*>(std::memchr(s, 0, str_size - off));
        if (!e) Fail("unterminated string");
        return std::string(reinterpret_cast<const char*>(s), e - s);
    };
    std::unordered_map<std::string, Identity> libraries, modules;
    for (auto& [t, v] : dynamic) {
        if (t == 0x61000013 || t == 0x61000015 || t == 0x6100000d || t == 0x6100000f) {
            auto& table = (t == 0x61000013 || t == 0x61000015) ? libraries : modules;
            const std::string key = EncodeId(v >> 48);
            const Identity id{string(v & 0xffffffff), u32((v >> 32) & 65535)};
            const auto it = table.find(key);
            if (it != table.end() && !(it->second == id)) Fail("conflicting library/module identity");
            table[key] = id;
        }
    }
    const u64 sym_off = m.tags.at(0x61000039), sym_size = m.tags.at(0x6100003f);
    Check(blob, sym_off, sym_size);
    if (sym_size % 24) Fail("invalid symbol table length");
    for (u64 pos = 0; pos < sym_size; pos += 24) {
        const u64 at = sym_off + pos;
        Symbol s;
        const u32 name = Read<u32>(blob, at);
        const u8 info = Read<u8>(blob, at + 4);
        s.section = Read<u16>(blob, at + 6);
        s.value = Read<u64>(blob, at + 8);
        s.size = Read<u64>(blob, at + 16);
        s.type = info & 15;
        s.binding = info >> 4;
        s.name = string(name);
        std::vector<std::string> parts;
        size_t start = 0;
        while (true) {
            const size_t hash_at = s.name.find('#', start);
            parts.push_back(s.name.substr(start, hash_at - start));
            if (hash_at == std::string::npos) break;
            start = hash_at + 1;
        }
        if (!s.name.empty() && parts.size() != 1) {
            if (parts.size() != 3) Fail("unsupported symbol encoding: " + s.name);
            const auto l = libraries.find(parts[1]);
            const auto mo = modules.find(parts[2]);
            if (l == libraries.end() || mo == modules.end()) Fail("unknown library/module of " + s.name);
            s.has_identity = true;
            s.nid = parts[0];
            s.library = l->second;
            s.module = mo->second;
        }
        m.symbols.push_back(std::move(s));
    }
    const std::pair<u64, u64> tables[2] = {{0x61000029, 0x6100002d}, {0x6100002f, 0x61000031}};
    for (auto [ot, st] : tables) {
        const u64 off = tag(ot, 0), size = tag(st, 0);
        Check(blob, off, size);
        if (size % 24) Fail("invalid relocation table length");
        for (u64 pos = 0; pos < size; pos += 24) {
            const u64 target = Read<u64>(blob, off + pos), info = Read<u64>(blob, off + pos + 8);
            const s64 addend = Read<s64>(blob, off + pos + 16);
            const u64 sym = info >> 32;
            if (sym >= m.symbols.size()) Fail("relocation references nonexistent symbol");
            m.relocs.push_back({target, u32(info & 0xffffffff), sym, addend});
        }
    }
    return m;
}

void PutQ(Bytes& b, u64 v) {
    const u8* p = reinterpret_cast<const u8*>(&v);
    b.insert(b.end(), p, p + 8);
}
void StoreQ(Bytes& image, u64 at, u64 v) {
    if (at + 8 > image.size()) Fail("store outside the image");
    std::memcpy(image.data() + at, &v, 8);
}

// link_modules.build_linked.
Bytes BuildLinked(std::vector<Module>& mods) {
    Bytes image;
    std::vector<std::array<u64, 3>> segments;
    std::vector<std::array<u64, 8>> metadata;
    std::unordered_map<Key, std::tuple<u64, u32, size_t>, KeyHash> exports;
    std::vector<std::array<u64, 4>> relocs;
    std::unordered_map<Key, size_t, KeyHash> import_index;
    std::vector<Key> import_order;
    std::unordered_map<Key, u32, KeyHash> import_kinds;
    for (size_t index = 0; index < mods.size(); ++index) {
        Module& m = mods[index];
        const u64 base = (image.size() + 65535) & ~u64(65535);
        std::vector<const Ph*> loads;
        for (const auto& p : m.ph)
            if (p.type == 1 || p.type == 0x61000010) loads.push_back(&p);
        u64 size = 0;
        for (auto* p : loads) size = std::max(size, p->vaddr + p->memsz);
        if (base + size > 512ull * 1024 * 1024) Fail("linked image exceeds 512 MiB");
        image.resize(base + size, 0);
        auto mapped = [&](u64 address, u64 length) {
            return std::any_of(loads.begin(), loads.end(), [&](const Ph* p) {
                return p->vaddr <= address && address + length <= p->vaddr + p->memsz;
            });
        };
        for (auto* p : loads) {
            if (p->filesz > p->memsz || p->vaddr % 4096 || p->flags > 7) Fail("invalid load segment");
            Check(m.elf, p->offset, p->filesz);
            std::memcpy(image.data() + base + p->vaddr, m.elf.data() + p->offset, p->filesz);
            segments.push_back({base + p->vaddr, p->memsz, p->flags});
        }
        const Ph* tls = nullptr;
        for (const auto& p : m.ph)
            if (p.type == 7) {
                tls = &p;
                break;
            }
        std::array<u64, 4> tls_values = tls ? std::array<u64, 4>{base + tls->vaddr, tls->filesz, tls->memsz, tls->align}
                                            : std::array<u64, 4>{0, 0, 0, 1};
        if (tls && (tls->filesz > tls->memsz || tls->memsz > 1024 * 1024 || !mapped(tls->vaddr, tls->memsz) ||
                    tls->align < 1 || tls->align > 4096 || (tls->align & (tls->align - 1))))
            Fail("invalid TLS layout");
        const auto init_tag = m.tags.find(12);
        const u64 init = init_tag != m.tags.end() ? base + init_tag->second : ~u64(0);
        metadata.push_back({base, size, init, tls_values[0], tls_values[1], tls_values[2], tls_values[3], index + 1});
        for (const auto& s : m.symbols) {
            const auto key = IdentityKey(s);
            if (key && s.section && (s.binding == 1 || s.binding == 2) && (s.type == 1 || s.type == 2)) {
                if (!mapped(s.value, std::max<u64>(1, s.size)) || exports.count(*key))
                    Fail("invalid or ambiguous native export");
                exports[*key] = {base + s.value, s.type == 1 ? 2u : 1u, index};
            }
        }
        for (const auto& r : m.relocs) {
            if (!mapped(r.target, 8)) Fail("relocation outside module");
            const u64 target = r.target + base;
            const Symbol& sym = m.symbols[r.sym];
            if (r.kind == 8) {
                if (!(0 <= r.addend && u64(r.addend) < size)) Fail("relative relocation outside module");
                relocs.push_back({target, 0, base + u64(r.addend), 0});
            } else if (r.kind == 16) {
                if (!sym.name.empty() || sym.type != 3 || r.addend || !tls)
                    Fail("unsupported external TLS module relocation");
                StoreQ(image, target, index + 1);
            } else if (r.kind == 1 || r.kind == 6 || r.kind == 7) {
                if (sym.section) {
                    const u64 value = sym.value + u64(r.addend);
                    if (sym.section == 0xfff1) {
                        StoreQ(image, target, value);
                    } else {
                        if (!(s64(value) >= 0 && value < size)) Fail("defined relocation outside module");
                        relocs.push_back({target, 0, base + value, 0});
                    }
                } else if (sym.binding == 2 && !sym.has_identity) {
                    StoreQ(image, target, 0);
                } else {
                    const auto key = IdentityKey(sym);
                    if (!key || (sym.type != 1 && sym.type != 2)) Fail("unsupported undefined symbol");
                    if (!import_index.count(*key)) {
                        import_index[*key] = import_order.size();
                        import_order.push_back(*key);
                    }
                    const u32 rk = sym.type == 1 ? 2 : 1;
                    const auto known = import_kinds.find(*key);
                    if (known != import_kinds.end() && known->second != rk)
                        Fail("conflicting import types for the same identity");
                    import_kinds[*key] = rk;
                    if ((rk == 1 && r.addend) || !(0 <= r.addend && r.addend < 4096)) Fail("unsupported import addend");
                    relocs.push_back({target, rk, import_index[*key], u64(r.addend)});
                }
            } else {
                Fail("unsupported relocation " + std::to_string(r.kind));
            }
        }
    }
    // Native bindings; explicit SysV C stdio compatibility aliases (vprintf, fflush, _Stdout) of
    // libSceLibcInternal to the bundled libc, never generalized by NID.
    static const char* const alias_nids[] = {"GMpvxPFW924", "MUjC4lbHrK4", "2sWzhYqFH4E"};
    std::vector<std::array<u64, 3>> bindings;
    for (size_t i = 0; i < import_order.size(); ++i) {
        const Key& key = import_order[i];
        auto it = exports.find(key);
        if (it == exports.end() && std::get<1>(key) == "libSceLibcInternal" && std::get<2>(key) == 1 &&
            std::get<3>(key) == "libSceLibcInternal" && std::get<4>(key) == 257 &&
            std::any_of(std::begin(alias_nids), std::end(alias_nids), [&](const char* n) { return std::get<0>(key) == n; }))
            it = exports.find(Key{std::get<0>(key), "libc", 1, "libc", 257});
        if (it != exports.end()) {
            const auto& [address, kind, owner] = it->second;
            if (kind != import_kinds[key]) Fail("native export/import type mismatch");
            bindings.push_back({i, address, kind});
        }
    }
    const Module& main = mods[0];
    u64 procparam = 0;
    bool found = false;
    for (const auto& p : main.ph)
        if (p.type == 0x61000001) {
            procparam = p.vaddr;
            found = true;
            break;
        }
    if (!found) Fail("missing process parameters");
    Bytes raw;
    raw.insert(raw.end(), {'I', 'S', 'S', 'P', 'R', 'B', '0', '2'});
    PutQ(raw, image.size());
    PutQ(raw, main.entry);
    PutQ(raw, segments.size());
    PutQ(raw, relocs.size());
    PutQ(raw, import_order.size());
    PutQ(raw, metadata.size());
    PutQ(raw, bindings.size());
    PutQ(raw, procparam);
    for (const auto& row : metadata)
        for (u64 v : row) PutQ(raw, v);
    for (const auto& row : bindings)
        for (u64 v : row) PutQ(raw, v);
    for (const auto& row : segments)
        for (u64 v : row) PutQ(raw, v);
    for (const Key& key : import_order) {
        const std::string name = KeyText(key);
        if (name.size() >= 128) Fail("canonical import identity too long");
        Bytes field(128, 0);
        std::memcpy(field.data(), name.data(), name.size());
        raw.insert(raw.end(), field.begin(), field.end());
    }
    for (const auto& row : relocs)
        for (u64 v : row) PutQ(raw, v);
    raw.insert(raw.end(), image.begin(), image.end());
    return raw;
}

} // namespace

// Prepares out_file from the game folder. progress(0..1, step) is called between the steps.
// Returns an empty string on success, else the reason.
std::wstring PrepareGame(const fs::path& game, const fs::path& out_file,
                         const std::function<void(float, const wchar_t*)>& progress) {
    try {
        progress(0.05f, L"Reading eboot.bin");
        std::vector<Module> mods;
        mods.push_back(LoadModule(game / "eboot.bin", true));
        if (mods[0].sha256 != ExpectedEboot)
            return L"Unsupported eboot.bin version: this port needs the decrypted retail build of "
                   L"inFAMOUS Second Son (CUSA00004 or CUSA00223, the same eboot.bin) it was made for.";
        progress(0.55f, L"Reading sce_module");
        mods.push_back(LoadModule(game / "sce_module" / "libc.prx", false));
        mods.push_back(LoadModule(game / "sce_module" / "libSceFios2.prx", false));
        progress(0.70f, L"Linking");
        const Bytes raw = BuildLinked(mods);
        progress(0.90f, L"Writing");
        std::error_code ec;
        fs::create_directories(out_file.parent_path(), ec);
        const fs::path tmp = out_file.wstring() + L".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(raw.data()), std::streamsize(raw.size()));
            if (!f) return L"Could not write the prepared game files.";
        }
        fs::rename(tmp, out_file, ec);
        if (ec) return L"Could not write the prepared game files.";
        // The source identity, so a different game folder prepares again.
        std::ofstream(out_file.wstring() + L".source", std::ios::trunc) << game.string() << "\n" << mods[0].sha256 << "\n";
        progress(1.0f, L"Done");
        return {};
    } catch (const std::exception& e) {
        const std::string what = e.what();
        return L"Preparation failed: " + std::wstring(what.begin(), what.end());
    }
}

} // namespace iss_prepare
