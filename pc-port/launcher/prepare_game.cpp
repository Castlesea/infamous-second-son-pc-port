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

// The eboot.bin builds the port runs: inFAMOUS Second Son 1.07 (CUSA00004 and CUSA00223 ship the
// same one) and inFAMOUS First Light 1.04 (CUSA00897 and CUSA00575 ship the same one).
constexpr const char* ExpectedEboots[] = {
    "2d1ca79630d7bbe6fa29575f59aa7996d43071d74ac8e4f7d40137967d9039ab",
    "693dc18504f30ae106bcd33cbb0c1d29a3dbbdb330754ee9dc2194ca6b56e240",
};

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

// Port of iss_intel_fix.py: rewrites 96 SSE4A EXTRQ instructions for Intel CPU compatibility.
// EXTRQ is an AMD-only instruction (STATUS_ILLEGAL_INSTRUCTION on Intel); each site is replaced by
// a call to a software equivalent in the reserved code cave at SSE4A_CAVE.
void PatchIntelCompat(Bytes& raw) {
    static constexpr uint64_t IMAGE_SIZE = 0x823854DUL;
    static constexpr uint64_t SSE4A_CAVE = 0xCBF000UL;
    if (raw.size() < 16 || memcmp(raw.data(), "ISSPRB02", 8) != 0) return;
    uint64_t declared; memcpy(&declared, raw.data() + 8, 8);
    if (declared != IMAGE_SIZE || raw.size() < IMAGE_SIZE) return;
    const size_t base = raw.size() - IMAGE_SIZE;
    u8* img = raw.data() + base;

    // Generate extrq_helper(dst, src) code: software EXTRQ xmmDst,xmmSrc using integer registers.
    auto make_helper = [](int dst, int src) {
        Bytes c;
        auto put = [&](std::initializer_list<u8> b) { c.insert(c.end(), b); };
        auto rex_b = [](bool r) -> u8 { return u8(0x48u | (r ? 4u : 0u)); };
        put({0x50, 0x51, 0x52}); // push rax,rcx,rdx
        put({0x66, rex_b(dst>=8), 0x0F, 0x7E, u8(0xC0|(dst&7)<<3)}); // movq rax,xmmDst
        put({0x66, rex_b(src>=8), 0x0F, 0x7E, u8(0xC1|(src&7)<<3)}); // movq rcx,xmmSrc
        put({0x0F, 0xB6, 0xD5}); // movzx edx,ch (index = bits[13:8] of src)
        put({0xC4, 0xE2, 0xEB, 0xF7, 0xC0}); // shrx rax,rax,rdx
        put({0xF7, 0xD1}); // not ecx
        put({0x8D, 0x49, 0x01}); // lea ecx,[rcx+1] (-length mod 64)
        put({0xC4, 0xE2, 0xF1, 0xF7, 0xC0}); // shlx rax,rax,rcx
        put({0xC4, 0xE2, 0xF3, 0xF7, 0xC0}); // shrx rax,rax,rcx
        put({0x66, rex_b(dst>=8), 0x0F, 0x6E, u8(0xC0|(dst&7)<<3)}); // movq xmmDst,rax
        put({0x5A, 0x59, 0x58, 0xC3}); // pop rdx,rcx,rax; ret
        return c;
    };
    auto put_i32 = [](Bytes& c, int32_t v) {
        c.insert(c.end(), reinterpret_cast<u8*>(&v), reinterpret_cast<u8*>(&v) + 4);
    };
    auto unhex = [](const char* s) {
        Bytes b; for (; *s && *(s+1); s += 2) { char t[3]={s[0],s[1],0}; b.push_back(u8(strtol(t,nullptr,16))); } return b;
    };

    // The 96 EXTRQ sites from iss_intel_fix.py: {offset_in_img, insn_len, dst, src, follower_len, follower_hex}
    static const struct { uint64_t ip; int len,dst,src,fl_len; const char* fl; } sites[] = {
        {0x890a77,4,1,7,5,"c4c1796ff8"},{0x890a84,4,5,4,4,"c55162c1"},{0x890a9f,4,1,4,4,"c5f96fee"},
        {0x890aa7,4,5,3,4,"c5d162e9"},{0x890c8e,5,5,15,0,""},{0x890c9c,4,7,0,4,"c5c162c5"},
        {0x890cbb,4,5,0,5,"c4c1796ff9"},{0x890ccd,4,7,0,4,"c5c162ed"},
        {0x890ce3,4,7,0,9,"c5f828842490020000"},{0x890cf0,4,0,3,4,"c57962ef"},
        {0x890d20,5,3,10,0,""},{0x890d2e,4,0,4,4,"c57962cb"},{0x890df9,5,0,15,0,""},
        {0x890e15,4,1,3,4,"c5f162c0"},{0x890e2b,5,1,10,0,""},{0x890e3d,4,3,4,4,"c5e162c9"},
        {0x8910b3,4,0,3,4,"c5f96fde"},{0x8910bb,4,3,7,4,"c5e162d8"},
        {0x8910d6,4,0,7,4,"c5f96ffc"},{0x8910de,4,4,5,4,"c5d962e0"},
        {0x89137d,4,0,1,9,"c5f8288c24d0020000"},{0x89138a,4,1,3,4,"c5f162c8"},
        {0x8913aa,4,0,3,9,"c5f8289c2490020000"},{0x8913b7,5,3,10,0,""},
        {0x8913ce,4,0,3,4,"c5f96fda"},{0x8913df,4,3,4,4,"c5e162d8"},
        {0x8913f5,4,0,4,5,"c4c1796fe5"},{0x89140c,4,4,6,4,"c5d962c0"},
        {0x8914d7,4,0,1,9,"c5796fa424f0000000"},{0x8914f2,4,1,3,4,"c5f162c0"},
        {0x891511,4,1,3,5,"c441796fce"},{0x89151f,5,3,10,0,""},
        {0x8917ea,4,0,3,5,"c4c1796fd2"},{0x8917f3,5,2,9,0,""},
        {0x891807,5,0,15,0,""},{0x891811,5,2,14,0,""},
        {0x891827,4,0,3,9,"c5f8289424c0020000"},{0x891834,5,2,9,0,""},
        {0x891846,5,2,15,0,""},{0x891854,5,3,14,0,""},
        {0x891958,4,6,0,4,"c5f828e5"},{0x891960,4,4,0,6,"c4e3210fc304"},
        {0x89196a,5,10,0,0,""},{0x891978,4,7,0,10,"c5b0c6942460020000cc"},
        {0x89199e,4,1,6,9,"c5f828ac2490020000"},{0x8919ab,4,5,3,4,"c5d162c9"},
        {0x891a10,4,7,6,9,"c5f828b424f0020000"},{0x891a1d,4,6,3,4,"c5c962df"},
        {0x8938f1,4,4,7,5,"c4c1796ff3"},{0x8938fa,4,6,0,4,"c5c962e4"},
        {0x89391f,4,6,0,9,"c5796f842410030000"},{0x893931,5,7,14,0,""},
        {0x893b2d,4,3,6,9,"c5f828b42450020000"},{0x893b3a,4,6,0,4,"c5c962c3"},
        {0x893b56,4,3,0,5,"c4c1796ff2"},{0x893b68,4,6,0,4,"c54962db"},
        {0x893b7e,4,3,0,9,"c578288c2410020000"},{0x893b90,5,6,14,0,""},
        {0x893bbb,5,4,14,0,""},{0x893bc9,4,0,5,4,"c57962d4"},
        {0x893c8b,4,0,1,9,"c5f8288c2430020000"},{0x893ca1,4,1,3,4,"c5f162c8"},
        {0x893cb2,5,0,14,0,""},{0x893cbc,4,3,5,4,"c5e162c0"},
        {0x893e60,4,0,5,4,"c5f96fe7"},{0x893e68,4,4,1,4,"c5d962e0"},
        {0x893e83,4,0,1,4,"c5f96ffa"},{0x893e8b,4,7,6,4,"c5c162c0"},
        {0x893fa7,4,3,0,9,"c5f828842400030000"},{0x893fbd,4,0,4,4,"c5f962db"},
        {0x893fd7,4,0,4,9,"c5f828a424e0020000"},{0x893fe4,4,4,6,4,"c5d962c0"},
        {0x89404b,4,6,3,6,"c4e3110fd104"},{0x89405e,4,0,2,4,"c5f962f6"},
        {0x894075,5,7,10,0,""},{0x894083,4,0,1,4,"c5f962c7"},
        {0x8940f0,4,5,3,9,"c57828842400030000"},{0x8940fd,5,8,2,0,""},
        {0x89412c,5,0,10,0,""},{0x89413a,4,5,1,4,"c5d162c0"},
        {0x89436a,4,0,1,4,"c5f96fcc"},{0x894372,4,1,3,4,"c5f162d8"},
        {0x8943d9,4,5,1,4,"c5f96ff2"},{0x8943e5,4,6,7,4,"c5c962ed"},
        {0x8944a9,4,3,5,4,"c5f96fec"},{0x8944b1,4,5,1,4,"c5d162db"},
        {0x89450f,4,5,1,4,"c5f96ff8"},{0x894517,5,7,14,0,""},
        {0x894618,4,0,5,4,"c5f96fec"},{0x894629,4,5,7,4,"c5d162e8"},
        {0x89463f,4,0,7,5,"c4c1796ffc"},{0x894651,4,7,6,4,"c5c162c0"},
        {0x8946bb,4,1,0,9,"c5f8288424e0010000"},{0x8946c8,4,4,0,9,"c5e1ebac2440020000"},
        {0x894757,5,11,0,0,""},{0x89475c,5,12,14,0,""},
    };

    // Write helper functions into the code cave, one per unique (dst,src) pair.
    std::map<std::pair<int,int>, uint64_t> helpers;
    uint64_t pos = SSE4A_CAVE;
    for (const auto& s : sites) {
        auto key = std::make_pair(s.dst, s.src);
        if (!helpers.count(key)) {
            Bytes code = make_helper(s.dst, s.src);
            memcpy(img + pos, code.data(), code.size());
            helpers[key] = pos;
            pos = (pos + code.size() + 15) & ~uint64_t(15);
        }
    }

    // Write a stub per site: red-zone protection + call helper + follower + jmp back.
    // Replace original EXTRQ (+ follower if needed) with jmp to stub.
    for (const auto& s : sites) {
        uint64_t helper = helpers[{s.dst, s.src}];
        uint64_t stub = pos;
        uint64_t back = s.ip + s.len + s.fl_len;
        Bytes code;
        auto put = [&](std::initializer_list<u8> b) { code.insert(code.end(), b); };
        // lea rsp,[rsp-0x80]
        put({0x48, 0x8D, 0x64, 0x24, 0x80});
        // call helper
        put({0xE8}); put_i32(code, int32_t(helper - (stub + code.size() + 4)));
        // lea rsp,[rsp+0x80]
        put({0x48, 0x8D, 0xA4, 0x24, 0x80, 0x00, 0x00, 0x00});
        // follower instruction bytes
        Bytes fl = unhex(s.fl); code.insert(code.end(), fl.begin(), fl.end());
        // jmp back
        put({0xE9}); put_i32(code, int32_t(back - (stub + code.size() + 4)));
        memcpy(img + stub, code.data(), code.size());
        pos = (pos + code.size() + 15) & ~uint64_t(15);
        // Patch original site: jmp stub + int3 fill
        int32_t jmp_rel = int32_t(stub - (s.ip + 5));
        img[s.ip] = 0xE9;
        memcpy(img + s.ip + 1, &jmp_rel, 4);
        int covered = s.len + s.fl_len;
        for (int i = 5; i < covered; ++i) img[s.ip + i] = 0xCC;
    }
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
        if (std::none_of(std::begin(ExpectedEboots), std::end(ExpectedEboots),
                         [&](const char* sha) { return mods[0].sha256 == sha; }))
            return L"Unsupported eboot.bin version: this port needs the decrypted retail build it was made "
                   L"for (inFAMOUS Second Son 1.07: CUSA00004 or CUSA00223; inFAMOUS First Light 1.04: CUSA00897 or CUSA00575).";
        progress(0.55f, L"Reading sce_module");
        mods.push_back(LoadModule(game / "sce_module" / "libc.prx", false));
        mods.push_back(LoadModule(game / "sce_module" / "libSceFios2.prx", false));
        progress(0.70f, L"Linking");
        Bytes raw = BuildLinked(mods);
        PatchIntelCompat(raw);
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
