// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_PASSES: the engine side (native_passes.h). Runs on the engine's threads, except
// Claim/Finish/PublishSpanStart (command processor) and BeforeSubmit (the submitting thread).
#include "native_passes.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <thread>
#include <intrin.h>

#ifndef ISS_SYSV
#define ISS_SYSV __attribute__((sysv_abi))
#endif

namespace iss::native_passes {
namespace {
using u8 = std::uint8_t;

constexpr u64 DrawIndexOffset = 0x623C60; // sce::Gnm::DrawCommandBuffer::drawIndex (eboot)
// push rbp; mov rbp,rsp; push r15; push r14; push r12; push rbx; mov rbx,rdi
constexpr unsigned char DrawIndexPrologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                                 0x56, 0x41, 0x54, 0x53, 0x48, 0x89, 0xfb};
constexpr u32 ContextBase = 0xA000, ShBase = 0x2C00, UconfigBase = 0xC000, ConfigBase = 0x2000;
constexpr u32 DrawDwords = 10; // DRAW_INDEX_2 and its trailing NOP (gnmdriver sceGnmDrawIndex)
constexpr u32 MaxSegmentDraws = 4096;
constexpr u32 MaxComputeDispatches = 64;
// The first draw's trailing NOP (gnmdriver sceGnmDrawIndex writes it, its payload is free):
// 'NPS', the segment's generation, the lane's tag. The command processor takes a segment only
// with its token there, so a command buffer address reused later never finds an older one.
constexpr u32 TokenMagic = 0x4E505300u;
constexpr u32 TrackedWords_ = 0x400 + 0x80 + 0x20;

struct CommandBuffer {
    u32* begin; // the current chunk (sce::Gnm::CommandBuffer m_beginptr)
    u32* end;
    u32* cursor;
    bool(ISS_SYSV* reserve)(CommandBuffer*, u32, void*);
    void* user;
};
using Reserve = bool(ISS_SYSV*)(CommandBuffer*, u32, void*);
using DrawIndex = u64(ISS_SYSV*)(CommandBuffer*, u32 count, u64 index_address, u32 flags);

DrawIndex original{};
Callbacks cb{};
std::atomic<bool> callbacks_set{};

// Statistics (relaxed).
enum Close : u32 { CloseBoundary, CloseTarget, CloseChunk, CloseNoRecord, CloseLearn,
                   CloseFormat, CloseLane, CloseSubmit, CloseFull, CloseReset, CloseCount };
constexpr const char* close_names[CloseCount] = {"packet", "target", "chunk", "no_record",
                                                 "learn", "format", "lane", "submit", "full",
                                                 "reset"};
struct Stats {
    std::atomic<u64> draws{}, recorded{}, segments{}, learn_posted{}, unknown_state{}, restarts{},
        unknown_draws{},
        not_ready{}, abandoned{}, no_slot{}, no_segment{}, learn_deduped{}, closed_inside{};
    std::array<std::atomic<u64>, CloseCount> close{};
    std::array<std::atomic<u64>, 257> boundary_opcodes{};
    std::array<std::atomic<u64>, 258> unknown_opcodes{}; // 256: not type 3, 257: past the cursor
    // Command processor.
    std::atomic<u64> claimed{}, executed{}, executed_draws{}, rejected{}, learned{};
    // Compute (ISS_NATIVE_COMPUTE): dispatches seen and recorded, segments, learn requests,
    // dispatches without gnmdriver's trailing NOP, and what closed compute segments (opcode).
    std::atomic<u64> dispatches{}, dispatches_recorded{}, cs_segments{}, cs_learn_posted{},
        cs_no_nop{};
    // ISS_NATIVE_INDIRECT: indexed indirect draws seen and recorded.
    std::atomic<u64> indirect_seen{}, indirect_recorded{};
    std::array<std::atomic<u64>, 257> cs_close_opcodes{};
};
Stats stats;
// Learn request divergence by tracked word (command processor only).
std::array<u64, TrackedWords_> divergence{};
u64 divergence_samples{};

// ---- Registry: segments by first draw packet (lock free lookups) --------------------------
constexpr u32 RegistrySize = 1u << 16;
struct Slot {
    std::atomic<const u32*> key{};
    std::atomic<Segment*> segment{};
};
std::unique_ptr<Slot[]> registry;

u32 Hash(const u32* key) {
    const u64 v = reinterpret_cast<u64>(key) >> 2;
    return u32((v * 0x9E3779B97F4A7C15ull) >> 48) & (RegistrySize - 1);
}

void Publish(Segment* s) {
    const u32 h = Hash(s->first_draw);
    for (u32 probe = 0; probe < 16; ++probe) {
        Slot& slot = registry[(h + probe) & (RegistrySize - 1)];
        const u32* key = slot.key.load(std::memory_order_acquire);
        Segment* old = slot.segment.load(std::memory_order_acquire);
        // Reuse the slot of the same key, or a slot whose segment is no longer pending.
        const bool stale = !old || old->first_draw != key ||
                           old->state.load(std::memory_order_acquire) != Ready;
        if (key == s->first_draw || !key || stale) {
            slot.segment.store(s, std::memory_order_release);
            slot.key.store(s->first_draw, std::memory_order_release);
            return;
        }
    }
    stats.no_slot.fetch_add(1, std::memory_order_relaxed);
    // No room: the segment is not found; it is released like an unused one.
    u32 expected = Ready;
    if (s->state.compare_exchange_strong(expected, Abandoned)) cb.abandon(*s);
}

// ---- Span starts: the command processor's registers at command buffer starts ---------------
constexpr u32 TrackedWords = 0x400 + 0x80 + 0x20;
struct SpanStart {
    std::array<u32, TrackedWords> words{};
};
std::mutex span_mutex;
std::unordered_map<const u32*, std::unique_ptr<SpanStart>> span_starts;
std::unique_ptr<SpanStart> latest_start;

bool LoadStart(const u32* span, u32* words) {
    std::scoped_lock lock{span_mutex};
    const SpanStart* start = nullptr;
    if (const auto it = span_starts.find(span); it != span_starts.end()) start = it->second.get();
    else start = latest_start.get();
    if (!start) return false;
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(words + TrackedFirst[r], &start->words[base], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    return true;
}

// ---- Lanes ---------------------------------------------------------------------------------
struct Lane {
    // Recording and closing (the submitting thread closes too). Held across the engine's draw
    // routine, so no segment is closed between recording a draw and its packet (the draw would
    // be both in the secondary and interpreted); recursive for the chunk callback inside it.
    std::recursive_mutex mutex;
    CommandBuffer* dcb{};
    void* regs{};
    u32* words{};
    bool known{};
    u32* begin{};          ///< chunk the lane follows
    const u32* scanned{};  ///< packets before this are applied
    Reserve original_reserve{};
    u64 key{};
    bool key_dirty{true};
    std::array<u32, MaskWords> written{}; ///< tracked words written since the start
    Segment* open{};
    /// The open compute segment (submit mode; never open together with `open`).
    Segment* open_cs{};
    /// Register writes since the open segment's last recorded draw (not in it yet).
    std::vector<std::pair<u32, u32>> pending;
    bool pending_index_type{}, pending_instances{};
    u32 index_type{}, instances{};
    std::unordered_map<u32, u32> final_index; ///< reg -> index in open->final_regs
    /// ISS_NATIVE_INDIRECT: the indirect argument table (SET_BASE) and which of SET_BASE (1),
    /// INDEX_BASE (2), INDEX_BUFFER_SIZE (4) this job saw (its indirect draws need all three).
    u64 indirect_base{};
    u32 indirect_seen{};
    bool pending_indirect_base{};
    /// The open segment records indexed indirect draws (they never share one with other draws:
    /// longer segments sample more textures and the feedback guard rejected them whole).
    bool open_indirect{};
    /// Segments of this lane, oldest first (recycled once retired).
    std::deque<Segment*> segments;
    std::vector<Segment*> spare;
    u32 next_generation{1};
    void* user{}; ///< the rasterizer's (Segment::lane_user)
    /// The segment holding a recorded draw whose packet the engine is writing now.
    Segment* in_flight{};
    /// The first chunk of the lane's previous start: the command processor's registers there are
    /// what this command buffer started from last time.
    const u32* first_chunk{};
    u64 restarts{};
    u32 tag{};
};
// The latest value any engine thread wrote to each tracked register (engine time order: a view's
// setup is written before the jobs that draw it): what a lane assumes for the dynamic state it
// did not write itself.
std::array<std::atomic<u32>, TrackedTotal> latest{};
std::array<std::atomic<bool>, TrackedTotal> latest_set{};
const bool predict_latest = [] {
    const char* p = std::getenv("ISS_NATIVE_PASSES_PREDICT");
    return p && p[0] == '1';
}();

std::mutex lanes_mutex;
std::unordered_map<CommandBuffer*, std::unique_ptr<Lane>> lanes;
std::vector<Lane*> all_lanes;
// The engine's chunk callback (one function for all its command buffers).
std::atomic<Reserve> engine_reserve{};

Lane* FindLane(CommandBuffer* dcb) {
    thread_local CommandBuffer* last_dcb = nullptr;
    thread_local Lane* last_lane = nullptr;
    if (dcb == last_dcb) return last_lane;
    std::scoped_lock lock{lanes_mutex};
    auto& slot = lanes[dcb];
    if (!slot) {
        slot = std::make_unique<Lane>();
        slot->dcb = dcb;
        slot->tag = u32(all_lanes.size() + 1) * 0x9E3779B1u;
        slot->regs = cb.new_regs(slot->words);
        all_lanes.push_back(slot.get());
    }
    last_dcb = dcb;
    last_lane = slot.get();
    return last_lane;
}

bool IsTargetReg(u32 reg) {
    return (reg >= ContextBase && reg < ContextBase + 0x20) ||
           (reg >= ContextBase + 0x318 && reg < ContextBase + 0x3A0);
}
// Target range registers that do not change the attachments (render control and override,
// occlusion count, depth bounds, clear values, screen scissor).
bool AttachmentNeutral(u32 reg) {
    switch (reg - ContextBase) {
    case 0x00: case 0x01: case 0x03: case 0x04: case 0x08: case 0x09: case 0x0A: case 0x0B:
    case 0x0C: case 0x0D:
        return true;
    default:
        return false;
    }
}

Segment* TakeSegment(Lane& l) {
    // Recycle retired segments, oldest first.
    while (!l.segments.empty()) {
        Segment* s = l.segments.front();
        u32 state = s->state.load(std::memory_order_acquire);
        // A ready segment far behind was passed over by the command processor (a draw it took
        // in a run of its own, a command buffer it skipped): never taken.
        if (state == Ready && l.next_generation - s->generation > 8192) {
            u32 expected = Ready;
            if (s->state.compare_exchange_strong(expected, Abandoned)) {
                stats.abandoned.fetch_add(1, std::memory_order_relaxed);
                cb.abandon(*s);
            }
            state = s->state.load(std::memory_order_acquire);
        }
        if (state == Open || state == Ready || state == Claimed) break;
        if (state == Done && !cb.retired(*s)) break;
        l.segments.pop_front();
        s->state.store(Free, std::memory_order_relaxed);
        l.spare.push_back(s);
    }
    Segment* s;
    if (!l.spare.empty()) {
        s = l.spare.back();
        l.spare.pop_back();
    } else {
        if (l.segments.size() > 16384) {
            stats.no_segment.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        s = new Segment;
    }
    s->kind = Kind::Segment;
    s->token = l.tag;
    s->lane_user = &l.user;
    s->generation = l.next_generation++;
    s->draw_packets.clear();
    s->final_regs.clear();
    s->has_index_type = s->has_instances = s->has_last_direct = false;
    s->has_indirect_base = false;
    s->first_user.clear();
    s->executed_tick.store(0, std::memory_order_relaxed);
    s->state.store(Open, std::memory_order_release);
    l.segments.push_back(s);
    return s;
}

// The pending writes become the open segment's (it records the draw they precede).
void MergePending(Lane& l, Segment& s) {
    if (l.pending_indirect_base) {
        s.has_indirect_base = true;
        s.final_indirect_base = l.indirect_base;
        l.pending_indirect_base = false;
    }
    for (const auto& [reg, value] : l.pending) {
        const auto [it, inserted] = l.final_index.try_emplace(reg, u32(s.final_regs.size()));
        if (inserted) s.final_regs.emplace_back(reg, value);
        else s.final_regs[it->second].second = value;
    }
    l.pending.clear();
    if (l.pending_index_type) {
        s.has_index_type = true;
        s.final_index_type = l.index_type;
    }
    if (l.pending_instances) {
        s.has_instances = true;
        s.final_instances = l.instances;
    }
    l.pending_index_type = l.pending_instances = false;
}

void CloseOne(Lane& l, Segment*& slot, Close why) {
    if (!slot) return;
    Segment* s = slot;
    slot = nullptr;
    l.final_index.clear();
    // Its last recorded draw has no packet yet (closed inside the draw routine): never used.
    if (s == l.in_flight) {
        stats.closed_inside.fetch_add(1, std::memory_order_relaxed);
        s->draw_packets.clear();
    }
    stats.close[why].fetch_add(1, std::memory_order_relaxed);
    u32 expected = Open;
    if (s->draw_packets.empty() || !cb.close(*s)) {
        if (s->state.compare_exchange_strong(expected, Abandoned)) cb.abandon(*s);
        return;
    }
    if (!s->state.compare_exchange_strong(expected, Ready)) return; // abandoned meanwhile
    stats.segments.fetch_add(1, std::memory_order_relaxed);
    if (s->kind == Kind::Compute) stats.cs_segments.fetch_add(1, std::memory_order_relaxed);
    Publish(s);
}

// Closes the open draw and compute segments.
void CloseSegment(Lane& l, Close why) {
    CloseOne(l, l.open, why);
    CloseOne(l, l.open_cs, why);
}
void CloseDraws(Lane& l, Close why) {
    CloseOne(l, l.open, why);
}
// The open compute segment ends before this packet (opcode, 256: not a type 3 packet).
void CloseCompute(Lane& l, Close why, u32 opcode) {
    if (!l.open_cs) return;
    stats.cs_close_opcodes[std::min<u32>(opcode, 256)].fetch_add(1, std::memory_order_relaxed);
    CloseOne(l, l.open_cs, why);
}

// Pending learn requests are tiny segments of their own kind.
// Keys with a learn request in flight, by when it was posted (ms): the others of the same key
// are left to the command processor's paths until it learns the first.
std::mutex learn_mutex;
std::unordered_map<u64, u64> learn_posted_at;
// Submit mode with ISS_NATIVE_PASSES_TEMPLATE_GATE (default on): the recorder dedups by template
// (native_passes_rasterizer.inl LearnGate), not by key here.
bool dedup_by_key = true;

Segment* PostLearnAt(Lane& l, const u32* packet, u64 key, bool compute, u32 draw_dwords = 10) {
    {
        const u64 now = GetTickCount64();
        std::scoped_lock lock{learn_mutex};
        auto& at = learn_posted_at[key];
        if (dedup_by_key && at && now - at < 250) {
            stats.learn_deduped.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        at = now;
        if (learn_posted_at.size() > 65536) learn_posted_at.clear();
    }
    Segment* s = TakeSegment(l);
    if (!s) return nullptr;
    s->kind = Kind::Learn;
    s->first_context = key;
    s->snapshot.clear();
    if (compute) {
        s->mask.clear();
        s->first_draw = packet;
        s->end = packet + 9; // DISPATCH_DIRECT and its trailing NOP
        s->state.store(Ready, std::memory_order_release);
        stats.cs_learn_posted.fetch_add(1, std::memory_order_relaxed);
        Publish(s);
        return s;
    }
    s->mask.assign(l.written.begin(), l.written.end());
    if ((stats.learn_posted.load(std::memory_order_relaxed) & 31) == 0) {
        s->snapshot.reserve(TrackedWords);
        for (u32 r = 0; r < 3; ++r)
            s->snapshot.insert(s->snapshot.end(), l.words + TrackedFirst[r],
                               l.words + TrackedFirst[r] + TrackedCount[r]);
    }
    s->first_draw = packet;
    s->end = packet + draw_dwords;
    s->state.store(Ready, std::memory_order_release);
    stats.learn_posted.fetch_add(1, std::memory_order_relaxed);
    Publish(s);
    return s;
}
Segment* PostLearn(Lane& l, const u32* packet) {
    return PostLearnAt(l, packet, l.key, false);
}

// Applies the packets of [p, end) to the lane; a packet a segment cannot span closes the open
// one first.
void Apply(Lane& l, const u32* p, const u32* end) {
    const auto base_of = [](u32 opcode) {
        return opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
             : opcode == 0x76 ? ShBase : UconfigBase;
    };
    const auto boundary = [&](u32 why) {
        stats.boundary_opcodes[std::min<u32>(why, 256)].fetch_add(1, std::memory_order_relaxed);
        CloseCompute(l, CloseBoundary, why);
        CloseSegment(l, why == 0x69 ? CloseTarget : CloseBoundary);
    };
    while (p < end) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        if (type != 3) {
            l.known = false;
            stats.unknown_opcodes[256].fetch_add(1, std::memory_order_relaxed);
            boundary(256);
            return;
        }
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        if (p + 1 + count > end) {
            l.known = false;
            stats.unknown_opcodes[257].fetch_add(1, std::memory_order_relaxed);
            boundary(256);
            return;
        }
        switch (opcode) {
        case 0x68: case 0x69: case 0x76: case 0x79: {
            if (count < 2) break;
            const u32 base = base_of(opcode);
            const u32 first = base + p[1];
            // Compute shader registers go to the queue's compute state: not a draw's. They end a
            // draw segment and stay in an open compute segment (its dispatches read them).
            if (opcode == 0x76 && p[1] >= 0x200) {
                stats.boundary_opcodes[opcode].fetch_add(1, std::memory_order_relaxed);
                CloseDraws(l, CloseBoundary);
                for (u32 i = 0; i + 1 < count && p[1] - 0x200 + i < CsWords; ++i) {
                    const u32 reg = CsBase + p[1] - 0x200 + i, value = p[2 + i];
                    l.words[reg] = value;
                    if (l.open_cs) l.pending.emplace_back(reg, value);
                }
                break;
            }
            CloseCompute(l, CloseBoundary, opcode);
            bool target = false;
            if (opcode == 0x69)
                for (u32 i = 0; i + 1 < count; ++i)
                    target |= IsTargetReg(first + i) && !AttachmentNeutral(first + i);
            if (target) boundary(0x69);
            for (u32 i = 0; i + 1 < count; ++i) {
                const u32 reg = first + i, value = p[2 + i];
                // The stage user data are not part of the render state key.
                if (l.words[reg] != value && !(reg >= PsUser0 && reg < PsUser0 + UserWords) &&
                    !(reg >= VsUser0 && reg < VsUser0 + UserWords))
                    l.key_dirty = true;
                if (const int t = TrackedIndex(reg); t >= 0) {
                    latest[t].store(value, std::memory_order_relaxed);
                    latest_set[t].store(true, std::memory_order_relaxed);
                    if (!(l.written[t >> 5] >> (t & 31) & 1)) {
                        l.written[t >> 5] |= 1u << (t & 31);
                        l.key_dirty = true;
                    }
                }
                l.words[reg] = value;
                if (l.open) l.pending.emplace_back(reg, value);
            }
            break;
        }
        case 0x2A:
            CloseCompute(l, CloseBoundary, opcode);
            l.index_type = p[1];
            l.words[cb.index_type_word] = p[1];
            l.pending_index_type = l.open != nullptr;
            break;
        case 0x2F:
            CloseCompute(l, CloseBoundary, opcode);
            l.instances = p[1];
            l.words[cb.instances_word] = p[1];
            l.pending_instances = l.open != nullptr;
            break;
        case 0x10: {
            // The NOPs the command processor acts on: engine/native markers and the flip.
            const u32 word = count > 0 ? p[1] : 0;
            if ((word & 0xFFFFFF00u) == 0x49535300u || (word & 0xFFFFFF00u) == 0x49534700u ||
                (word & 0xFFFFFF00u) == 0x49534800u || word == 0x68750776u)
                boundary(0x10);
            break;
        }
        case 0x12: // CLEAR_STATE: the defaults, as the command processor sets them
            boundary(opcode);
            cb.clear_state(l.regs);
            l.written.fill(~0u); // the defaults, as the command processor's
            l.index_type = l.words[cb.index_type_word];
            l.instances = l.words[cb.instances_word];
            l.key_dirty = true;
            break;
        case 0x28: // CONTEXT_CONTROL: nothing for the command processor
            break;
        // ISS_NATIVE_INDIRECT: the state of indexed indirect draws, kept by the lane (and as the
        // segment's final state) instead of closing it.
        case 0x11: // SET_BASE
            if (IndirectEnabled() && count >= 3 && (p[1] & 0xF) == 1) {
                CloseCompute(l, CloseBoundary, opcode);
                if (l.open && !l.open_indirect) boundary(opcode);
                l.indirect_base = u64(p[2]) | (u64(p[3] & 0xFFFF) << 32);
                l.indirect_seen |= 1;
                if (l.open) l.pending_indirect_base = true;
                break;
            }
            boundary(opcode);
            break;
        case 0x26: // INDEX_BASE
            if (IndirectEnabled() && count >= 2) {
                CloseCompute(l, CloseBoundary, opcode);
                if (l.open && !l.open_indirect) boundary(opcode);
                const u32 w = cb.index_base_word;
                const u32 hi = (l.words[w] & ~0xFFu) | (p[2] & 0xFFu);
                l.words[w] = hi;
                l.words[w + 1] = p[1];
                if (l.open) {
                    l.pending.emplace_back(w, hi);
                    l.pending.emplace_back(w + 1, p[1]);
                }
                l.indirect_seen |= 2;
                break;
            }
            boundary(opcode);
            break;
        case 0x13: // INDEX_BUFFER_SIZE
            if (IndirectEnabled() && count >= 1) {
                CloseCompute(l, CloseBoundary, opcode);
                if (l.open && !l.open_indirect) boundary(opcode);
                l.words[cb.num_indices_word] = p[1];
                if (l.open) l.pending.emplace_back(cb.num_indices_word, p[1]);
                l.indirect_seen |= 4;
                break;
            }
            boundary(opcode);
            break;
        case 0x50: // DMA_DATA to 0x3022C: an explicit no-op for the command processor
            if (count < 6 || p[4] != 0x3022C) boundary(opcode);
            break;
        // State the lane cannot follow: loaded from memory, indexed.
        case 0x5F: case 0x60: case 0x61: case 0x62: case 0x63:
        case 0x64: case 0x65: case 0x66: case 0x6A: case 0x9B: case 0x80:
            l.known = false;
            stats.unknown_opcodes[opcode].fetch_add(1, std::memory_order_relaxed);
            boundary(opcode);
            break;
        default:
            // Draws of other kinds, dispatches, events, waits, DMA, writes: the command
            // processor executes them between segments.
            boundary(opcode);
            break;
        }
        p += 1 + count;
    }
}

// A new frame's (or a reset) command buffer: from the published registers of its start.
void Restart(Lane& l) {
    // Ready segments stay: the command processor reaches them frames later (their token keeps
    // them apart from the new frame's).
    CloseSegment(l, CloseReset);
    // The command processor's registers where this command buffer started last time (its chunk
    // addresses change from frame to frame), else where it starts now, else the latest.
    l.known = LoadStart(l.first_chunk ? l.first_chunk : l.dcb->begin, l.words);
    l.first_chunk = l.dcb->begin;
    ++l.restarts;
    stats.restarts.fetch_add(1, std::memory_order_relaxed);
    if (!l.known) stats.unknown_state.fetch_add(1, std::memory_order_relaxed);
    l.index_type = l.words[cb.index_type_word];
    l.instances = l.words[cb.instances_word];
    l.key_dirty = true;
    l.written.fill(0);
    l.begin = l.dcb->begin;
    l.scanned = l.dcb->begin;
    l.pending.clear();
}

// The engine's chunk callback for a followed command buffer: the end of the chunk is applied
// before it is left; a segment never spans chunks.
bool ISS_SYSV ReserveWrapper(CommandBuffer* dcb, u32 dwords, void* user) {
    Lane* l = FindLane(dcb);
    std::unique_lock lock{l->mutex};
    // Not the chunk the lane followed: the command buffer was started over (a new frame).
    if (!(l->begin == dcb->begin && l->scanned >= dcb->begin && l->scanned <= dcb->cursor))
        Restart(*l);
    Apply(*l, l->scanned, dcb->cursor);
    CloseSegment(*l, CloseChunk);
    const Reserve next = l->original_reserve ? l->original_reserve
                                             : engine_reserve.load(std::memory_order_acquire);
    lock.unlock();
    const bool result = next(dcb, dwords, user);
    lock.lock();
    l->begin = dcb->begin;
    l->scanned = dcb->cursor;
    return result;
}

thread_local Lane* thread_lane = nullptr;

u64 ISS_SYSV DrawIndexHook(CommandBuffer* dcb, u32 count, u64 index_address, u32 flags) {
    if ((stats.draws.fetch_add(1, std::memory_order_relaxed) & 0x3FFFF) == 0x3FFFF &&
        callbacks_set.load(std::memory_order_acquire) && cb.report)
        cb.report();
    if (!callbacks_set.load(std::memory_order_acquire) ||
        !(dcb->begin <= dcb->cursor && dcb->cursor <= dcb->end)) {
        stats.not_ready.fetch_add(1, std::memory_order_relaxed);
        return original(dcb, count, index_address, flags);
    }
    // Room for the draw first, as the routine reserves it: its packet lands at the cursor.
    if (static_cast<u64>(dcb->end - dcb->cursor) < DrawDwords &&
        !dcb->reserve(dcb, DrawDwords, dcb->user))
        return original(dcb, count, index_address, flags);
    Lane* l = FindLane(dcb);
    // A thread moving to another command buffer closes the one it left (it may not come back
    // before the submission).
    if (thread_lane && thread_lane != l) {
        std::scoped_lock other{thread_lane->mutex};
        CloseSegment(*thread_lane, CloseLane);
    }
    thread_lane = l;
    std::unique_lock lock{l->mutex};
    // Follow the command buffer: its chunk callback wrapped, its packets up to the cursor.
    if (dcb->reserve != &ReserveWrapper) {
        Reserve expected = nullptr;
        engine_reserve.compare_exchange_strong(expected, dcb->reserve);
        l->original_reserve = dcb->reserve;
        dcb->reserve = &ReserveWrapper;
        Restart(*l);
    } else if (l->begin != dcb->begin || l->scanned < dcb->begin || l->scanned > dcb->cursor) {
        Restart(*l);
    }
    Apply(*l, l->scanned, dcb->cursor);
    u32* const packet = dcb->cursor;
    bool recorded = false;
    Segment* token_for = nullptr; // a segment or learn request starting at this draw
    if (l->known && l->key_dirty) {
        l->key = cb.key(l->regs, l->written.data()) | 1;
        l->key_dirty = false;
    }
    if (l->known && !cb.unlearnable(l->key)) {
        // The draw initiator is 0 outside the PS4 Pro mode (gnmdriver sceGnmDrawIndex).
        const DrawArgs draw{count, 0, index_address};
        for (u32 attempt = 0; attempt < 2 && !recorded; ++attempt) {
            const bool fresh = !l->open;
            if (fresh) {
                l->open = TakeSegment(*l);
                if (!l->open) break;
                l->open->first_draw = packet;
                l->final_index.clear();
                l->pending.clear(); // in the command processor's registers at the first draw
                l->pending_index_type = l->pending_instances = false;
                Segment& s = *l->open;
                s.first_context = l->key;
                s.first_index_type = l->index_type;
                s.first_instances = l->instances;
                s.first_user.assign(l->words + VsUser0, l->words + VsUser0 + UserWords);
                s.first_user.insert(s.first_user.end(), l->words + PsUser0,
                                    l->words + PsUser0 + UserWords);
                // The dynamic state it did not write, as assumed (checked by the command processor).
                s.dyn_check.clear();
                for (u32 reg = 0xA000; reg < 0xA400; ++reg) {
                    if (!IsDynamicReg(reg)) continue;
                    const u32 t = reg - 0xA000;
                    if ((l->written[t >> 5] >> (t & 31)) & 1) continue;
                    // ISS_NATIVE_PASSES_PREDICT=1: the latest the engine wrote; otherwise what the
                    // command processor had where this command buffer started last time.
                    if (predict_latest && latest_set[t].load(std::memory_order_relaxed))
                        l->words[reg] = latest[t].load(std::memory_order_relaxed);
                    s.dyn_check.emplace_back(reg, l->words[reg]);
                }
            }
            const DrawResult r = cb.record(*l->open, l->regs, l->key, l->written.data(), draw);
            if (r == Recorded) {
                recorded = true;
                break;
            }
            if (r == NewSegment && !fresh) {
                CloseSegment(*l, CloseFormat);
                continue;
            }
            CloseSegment(*l, r == NeedLearn ? CloseLearn : CloseNoRecord);
            if (r == NeedLearn) token_for = PostLearn(*l, packet);
            break;
        }
    } else if (l->known) {
        CloseSegment(*l, CloseNoRecord);
    } else {
        stats.unknown_draws.fetch_add(1, std::memory_order_relaxed);
    }
    Segment* const into = recorded ? l->open : nullptr;
    l->in_flight = into;
    const u64 result = original(dcb, count, index_address, flags);
    l->in_flight = nullptr;
    // The draw's own packets: at `packet` (room was reserved before), or the segment, whose
    // recording already holds the draw, cannot be used.
    if (into && l->open == into) {
        Segment& s = *into;
        if (dcb->cursor == packet + DrawDwords && packet[0] >> 30 == 3 &&
            ((packet[0] >> 8) & 0xFF) == 0x27) {
            MergePending(*l, *l->open);
            s.draw_packets.push_back(packet);
            if (s.draw_packets.size() == 1) token_for = &s;
            stats.recorded.fetch_add(1, std::memory_order_relaxed);
            s.end = dcb->cursor;
            s.last_max_index = packet[1];
            s.last_index_lo = packet[2];
            s.last_index_hi = packet[3];
            s.last_index_count = packet[4];
            s.last_initiator = packet[5];
            if (s.draw_packets.size() >= MaxSegmentDraws) CloseSegment(*l, CloseFull);
        } else {
            s.draw_packets.clear();
            CloseSegment(*l, CloseBoundary);
        }
    }
    if (token_for && dcb->cursor == packet + DrawDwords && ((packet[6] >> 8) & 0xFF) == 0x10 &&
        ((packet[6] >> 16) & 0x3FFF) == 2) {
        packet[7] = TokenMagic;
        packet[8] = token_for->generation;
        packet[9] = token_for->token;
    }
    if (l->begin == dcb->begin && dcb->cursor >= packet) l->scanned = dcb->cursor;
    return result;
}

// ---- ISS_NATIVE_PASSES_SUBMIT: recording at the submission, from exact registers ----------
//
// The registers at each draw are the command processor's own: the scan applies the register
// writes of every command buffer in submission order (from where the previous submission left
// them, seeded once from the command processor), so no inherited word is assumed. A lane per
// pool thread records the command buffers it is given, start to end.
const bool submit_mode = [] {
    const char* p = std::getenv("ISS_NATIVE_PASSES_SUBMIT");
    return p && p[0] == '1';
}();

// The recording of one command buffer: its jobs left (the command processor waits for 0).
struct DcbWork {
    std::atomic<u32> jobs_left{};
    std::atomic<bool> scanned{}; ///< its jobs are queued (jobs_left is final)
    /// Where each job starts (submission order) and whether it is recorded: set before `scanned`.
    std::vector<const u32*> job_begins;
    std::unique_ptr<std::atomic<u8>[]> job_done;
};
// A stretch of a command buffer recorded from the registers at its start.
struct Job {
    const u32* begin{};
    const u32* end{};
    std::vector<u32> start; // tracked words
    std::shared_ptr<DcbWork> work;
    u32 index{}; ///< in its command buffer's jobs
};

struct SubmitState {
    std::mutex mutex; // one submission scan at a time
    std::unique_ptr<Lane> shadow; // the registers as the command processor will have them
    bool seeded{};
    // The pool and its queue of jobs.
    std::vector<std::thread> threads;
    std::mutex pool_mutex;
    std::condition_variable work_cv, done_cv;
    std::deque<Job> jobs;
    std::unordered_map<const u32*, std::shared_ptr<DcbWork>> pending; // by command buffer
    // ISS_NATIVE_ASC_ASYNC: the compute submissions the pool is recording, by ring position, with
    // when the command processor first found one unfinished (0: not yet).
    std::unordered_map<const u32*, std::pair<std::shared_ptr<DcbWork>, u64>> asc_pending;
    // ISS_NATIVE_PASSES_ASYNC_SCAN=1: the submissions the scan thread has yet to scan, in order.
    struct ScanItem {
        const u32* begin{};
        const u32* end{};
        std::shared_ptr<DcbWork> work;
    };
    std::thread scan_thread;
    std::mutex scan_mutex;
    std::condition_variable scan_cv;
    std::deque<ScanItem> scans;
    // Statistics.
    std::atomic<u64> submits{}, scan_us{}, dcbs_recorded{}, jobs_done{}, cp_waits{}, cp_wait_us{};
    // ISS_NATIVE_PASSES_CP_RECORDS: jobs the command processor recorded itself, and its time.
    std::atomic<u64> cp_records{}, cp_record_us{};
    // ISS_NATIVE_PASSES_SPIN_US: waits that ended while watching the job (no sleep).
    std::atomic<u64> cp_spins{};
};
SubmitState submit;

std::unique_ptr<Lane> NewJobLane() {
    auto l = std::make_unique<Lane>();
    l->regs = cb.new_regs(l->words);
    l->tag = u32(reinterpret_cast<std::uintptr_t>(l.get()) >> 4) * 0x9E3779B1u | 1;
    l->known = true;
    return l;
}

void LoadTracked(Lane& l, const std::vector<u32>& start) {
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(l.words + TrackedFirst[r], &start[base], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    if (start.size() >= TrackedWords + CsWords)
        std::memcpy(l.words + CsBase, &start[TrackedWords], CsWords * 4);
    l.index_type = l.words[cb.index_type_word];
    l.instances = l.words[cb.instances_word];
    l.written.fill(~0u); // every word is exact: nothing is assumed
    l.key_dirty = true;
    l.known = true;
    l.pending.clear();
    l.pending_index_type = l.pending_instances = false;
    l.indirect_seen = 0;
    l.pending_indirect_base = false;
    // A scan job's start also carries the indexed indirect state (SET_BASE and which of the three
    // packets were seen): a job can now start at an indirect draw.
    if (start.size() >= TrackedWords + CsWords + 3) {
        const u32* x = &start[TrackedWords + CsWords];
        l.indirect_base = u64(x[0]) | (u64(x[1]) << 32);
        l.indirect_seen = x[2];
    }
}

// A DISPATCH_DIRECT packet (graphics command buffer, submit mode) into the lane's open compute
// segment, or a learn request at it. Returns where the packets after it start.
const u32* RecordDispatch(Lane& l, const u32* packet, const u32* next, const u32* end) {
    stats.dispatches.fetch_add(1, std::memory_order_relaxed);
    const bool has_nop = next + 4 <= end && (next[0] >> 30) == 3 &&
                         ((next[0] >> 8) & 0xFF) == 0x10 && ((next[0] >> 16) & 0x3FFF) == 2;
    const u32* const after = has_nop ? next + 4 : next;
    // The registers the command processor sets for the dispatch (ComputeProgram words 0..3).
    l.words[CsBase] = packet[4];
    l.words[CsBase + 1] = packet[1];
    l.words[CsBase + 2] = packet[2];
    l.words[CsBase + 3] = packet[3];
    if (!has_nop) stats.cs_no_nop.fetch_add(1, std::memory_order_relaxed);
    const u64 key = cb.compute_key(l.regs) | 1;
    bool recorded = false;
    Segment* token_for = nullptr;
    if (!cb.unlearnable(key)) {
        for (u32 attempt = 0; attempt < 2 && !recorded; ++attempt) {
            const bool fresh = !l.open_cs;
            if (fresh) {
                if (!has_nop) break; // no room for the token: the command processor dispatches
                l.open_cs = TakeSegment(l);
                if (!l.open_cs) break;
                Segment& seg = *l.open_cs;
                seg.kind = Kind::Compute;
                seg.first_draw = packet;
                l.final_index.clear();
                l.pending.clear();
                l.pending_index_type = l.pending_instances = false;
                seg.first_context = key;
                seg.first_user.assign(l.words + CsBase + 0x40, l.words + CsBase + 0x40 + UserWords);
                seg.dyn_check.clear();
            }
            const DrawResult r = cb.record_dispatch(*l.open_cs, l.regs, key, packet);
            if (r == Recorded) {
                recorded = true;
                break;
            }
            if (r == NewSegment && !fresh) {
                CloseCompute(l, CloseFormat, 0x15);
                continue;
            }
            CloseCompute(l, r == NeedLearn ? CloseLearn : CloseNoRecord, 0x15);
            if (r == NeedLearn && has_nop) token_for = PostLearnAt(l, packet, key, true);
            break;
        }
    }
    // A dispatch left to the command processor ends the open segment: a later recorded dispatch
    // would stretch the segment over it and the command processor would jump over it (a key
    // given up for learning, e.g. the particle dispatches using GDS, was never run: fire and
    // smoke missing in The Visitors, vis-forest-on 2026-10-18).
    if (!recorded) CloseCompute(l, CloseNoRecord, 0x15);
    if (recorded) {
        Segment& seg = *l.open_cs;
        MergePending(l, seg);
        seg.draw_packets.push_back(packet);
        if (seg.draw_packets.size() == 1) token_for = &seg;
        stats.dispatches_recorded.fetch_add(1, std::memory_order_relaxed);
        seg.end = after;
        seg.last_dims[0] = packet[1];
        seg.last_dims[1] = packet[2];
        seg.last_dims[2] = packet[3];
        seg.last_dispatch_initiator = packet[4];
        // Long compute runs (the longhouse: ~150 dispatches back to back) are cut: the pool records
        // them in parallel (the scan starts a job every job_draws draws or dispatches) and the
        // hazard checks of a segment stay short.
        if (seg.draw_packets.size() >= MaxComputeDispatches) CloseCompute(l, CloseFull, 0x15);
    }
    if (token_for && has_nop) {
        auto* w = const_cast<u32*>(next);
        w[1] = TokenMagic;
        w[2] = token_for->generation;
        w[3] = token_for->token;
    }
    return after;
}

// Records the segments of one command buffer from its exact start registers.
void RecordCommandBuffer(Lane& l, const u32* begin, const u32* end, const std::vector<u32>& start) {
    std::scoped_lock lock{l.mutex};
    LoadTracked(l, start);
    const u32* p = begin;
    const u32* plain_from = begin;
    while (p < end) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        if (type != 3) break;
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        const u32* next = p + 1 + count;
        if (next > end) break;
        // DRAW_INDEX_2, and DRAW_INDEX_AUTO (non-indexed; the pause menu UI and many full screen
        // passes): both carry gnmdriver's trailing NOP for the token.
        const bool direct = opcode == 0x27 && count == 5;
        const bool autod = opcode == 0x2D && count == 2;
        const bool dispatch = opcode == 0x15 && count == 4 && cb.record_dispatch;
        const bool indirect = opcode == 0x25 && count == 4 && IndirectEnabled();
        if (!direct && !autod && !dispatch && !indirect) {
            p = next;
            continue;
        }
        // A draw: the packets since the previous draw first (they may close the segment).
        Apply(l, plain_from, p);
        if (dispatch) {
            CloseDraws(l, CloseBoundary);
            plain_from = p = RecordDispatch(l, p, next, end);
            continue;
        }
        CloseCompute(l, CloseBoundary, opcode);
        const u32* const packet = p;
        // The trailing NOP gnmdriver writes after it carries the token.
        const bool has_nop = next + 4 <= end && (next[0] >> 30) == 3 &&
                             ((next[0] >> 8) & 0xFF) == 0x10 && ((next[0] >> 16) & 0x3FFF) == 2;
        const u32* const after = has_nop ? next + 4 : next;
        stats.draws.fetch_add(1, std::memory_order_relaxed);
        if (l.key_dirty) {
            l.key = cb.key(l.regs, l.written.data()) | 1;
            l.key_dirty = false;
        }
        // An indexed indirect draw records under its own key, once the job saw its state.
        const u64 key = indirect ? IndirectKey(l.key, packet[2], packet[3]) : l.key;
        const bool indirect_known = !indirect || (l.indirect_seen & 7) == 7;
        bool recorded = false;
        Segment* token_for = nullptr;
        if (l.open && l.open_indirect != indirect) CloseSegment(l, CloseBoundary);
        if (indirect_known && !cb.unlearnable(key)) {
            DrawArgs draw = direct
                ? DrawArgs{packet[4], packet[5], u64(packet[2]) | (u64(packet[3]) << 32)}
                : DrawArgs{packet[1], packet[2], 0, true};
            if (indirect) {
                draw = DrawArgs{0, packet[4], 0, false};
                draw.indirect = true;
                draw.args_address = l.indirect_base + packet[1];
            }
            for (u32 attempt = 0; attempt < 2 && !recorded; ++attempt) {
                const bool fresh = !l.open;
                if (fresh) {
                    if (!has_nop) break; // no room for the token: the command processor draws it
                    l.open = TakeSegment(l);
                    if (!l.open) break;
                    l.open->first_draw = packet;
                    l.final_index.clear();
                    l.pending.clear();
                    l.pending_index_type = l.pending_instances = false;
                    Segment& seg = *l.open;
                    l.open_indirect = indirect;
                    seg.first_context = key;
                    seg.first_index_type = l.index_type;
                    seg.first_instances = l.instances;
                    seg.first_user.assign(l.words + VsUser0, l.words + VsUser0 + UserWords);
                    seg.first_user.insert(seg.first_user.end(), l.words + PsUser0,
                                          l.words + PsUser0 + UserWords);
                    seg.dyn_check.clear();
                }
                const DrawResult r = cb.record(*l.open, l.regs, key, l.written.data(), draw);
                if (r == Recorded) {
                    recorded = true;
                    break;
                }
                if (r == NewSegment && !fresh) {
                    CloseSegment(l, CloseFormat);
                    continue;
                }
                CloseSegment(l, r == NeedLearn ? CloseLearn : CloseNoRecord);
                if (r == NeedLearn && has_nop)
                    token_for = PostLearnAt(l, packet, key, false, indirect ? 9u : u32(DrawDwords));
                break;
            }
        } else {
            CloseSegment(l, CloseNoRecord);
        }
        if (indirect) stats.indirect_seen.fetch_add(1, std::memory_order_relaxed);
        if (recorded) {
            Segment& seg = *l.open;
            if (indirect && seg.has_last_direct) {
                // The index buffer of an earlier direct draw becomes final state: an INDEX_BASE
                // since then (merged below) replaces it.
                const auto put = [&](u32 reg, u32 value) {
                    const auto [it, inserted] = l.final_index.try_emplace(reg, u32(seg.final_regs.size()));
                    if (inserted) seg.final_regs.emplace_back(reg, value);
                    else seg.final_regs[it->second].second = value;
                };
                const u32 w = cb.index_base_word;
                put(w, (l.words[w] & ~0xFFu) | (seg.last_index_hi & 0xFFu));
                put(w + 1, seg.last_index_lo);
                put(cb.max_index_word, seg.last_max_index);
                seg.has_last_direct = false;
            }
            MergePending(l, seg);
            seg.draw_packets.push_back(packet);
            if (seg.draw_packets.size() == 1) token_for = &seg;
            stats.recorded.fetch_add(1, std::memory_order_relaxed);
            seg.end = after;
            if (indirect) {
                seg.last_index_count = l.words[cb.num_indices_word];
                seg.last_initiator = packet[4];
                stats.indirect_recorded.fetch_add(1, std::memory_order_relaxed);
            } else if (direct) {
                seg.last_max_index = packet[1];
                seg.last_index_lo = packet[2];
                seg.last_index_hi = packet[3];
                seg.last_index_count = packet[4];
                seg.last_initiator = packet[5];
                seg.has_last_direct = true;
            } else {
                seg.last_index_count = packet[1];
                seg.last_initiator = packet[2];
            }
            if (seg.draw_packets.size() >= MaxSegmentDraws) CloseSegment(l, CloseFull);
        }
        if (token_for && has_nop) {
            auto* w = const_cast<u32*>(next);
            w[1] = TokenMagic;
            w[2] = token_for->generation;
            w[3] = token_for->token;
        }
        plain_from = after;
        p = after;
    }
    CloseSegment(l, CloseChunk);
}

// A recorded job: its command buffer learns it (the caller does not hold pool_mutex).
void FinishJob(const Job& job) {
    submit.jobs_done.fetch_add(1, std::memory_order_relaxed);
    if (job.work->job_done) job.work->job_done[job.index].store(1, std::memory_order_release);
    const bool last = job.work->jobs_left.fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (last || job.work->job_done) {
        std::scoped_lock lock{submit.pool_mutex};
        submit.done_cv.notify_all();
    }
}

void PoolThread() {
    SetThreadDescription(GetCurrentThread(), L"ISS:NativePassesPool");
    auto lane = NewJobLane();
    for (;;) {
        Job job;
        {
            std::unique_lock lock{submit.pool_mutex};
            submit.work_cv.wait(lock, [] { return !submit.jobs.empty(); });
            job = std::move(submit.jobs.front());
            submit.jobs.pop_front();
        }
        RecordCommandBuffer(*lane, job.begin, job.end, job.start);
        FinishJob(job);
    }
}

u64 NowUs() {
    // steady_clock: the counter * 1000000 before dividing overflowed after ~10 days of uptime
    // (a 10 MHz counter), which broke the 50 ms timeout of AscRecorded.
    return u64(std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

// The scissor setters of the eboot's DrawCommandBuffer (setScreenScissor 0x626890, called by the
// frame setup; setGenericScissor 0x6269D0, called by the shadow view bind 0x439AB0 and restore
// 0x439C90): their values become the latest at once, in engine time, for the lanes that draw the
// view in other command buffers and inherit them.
constexpr unsigned char SetterPrologue[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                              0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
using Setter = u64(ISS_SYSV*)(CommandBuffer*, u64, u64, u64, u64, u64);
Setter screen_scissor_original{}, generic_scissor_original{};

void PublishWrites(const u32* p, const u32* end) {
    while (p < end) {
        const u32 header = *p;
        if ((header >> 30) == 2) { ++p; continue; }
        if ((header >> 30) != 3) return;
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        if (p + 1 + count > end) return;
        if (((header >> 8) & 0xFF) == 0x69)
            for (u32 i = 0; i + 1 < count; ++i)
                if (const int t = TrackedIndex(ContextBase + p[1] + i); t >= 0) {
                    latest[t].store(p[2 + i], std::memory_order_relaxed);
                    latest_set[t].store(true, std::memory_order_relaxed);
                }
        p += 1 + count;
    }
}

template <Setter* original_fn>
u64 ISS_SYSV SetterHook(CommandBuffer* dcb, u64 a, u64 b, u64 c, u64 d, u64 e) {
    u32* const before = dcb->cursor;
    u32* const begin = dcb->begin;
    const u64 result = (*original_fn)(dcb, a, b, c, d, e);
    if (dcb->begin == begin && dcb->cursor >= before) PublishWrites(before, dcb->cursor);
    else if (dcb->cursor >= dcb->begin) PublishWrites(dcb->begin, dcb->cursor);
    return result;
}

// Jumps from `offset` (starting with `prologue`) to `hook`; the trampoline runs the original.
void* Patch(u64 base, u64 offset, const unsigned char* prologue, size_t size, const void* hook) {
    auto* code = reinterpret_cast<unsigned char*>(base + offset);
    if (size < 13 || std::memcmp(code, prologue, size) != 0) return nullptr;
    auto* trampoline = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return nullptr;
    std::memcpy(trampoline, prologue, size);
    const unsigned char jump_back[6] = {0xff, 0x25, 0, 0, 0, 0};
    const auto back = reinterpret_cast<std::uintptr_t>(code + size);
    std::memcpy(trampoline + size, jump_back, 6);
    std::memcpy(trampoline + size + 6, &back, 8);
    unsigned char patch[14] = {0x48, 0xb8};
    const auto target = reinterpret_cast<std::uintptr_t>(hook);
    std::memcpy(patch + 2, &target, 8);
    patch[10] = 0xff;
    patch[11] = 0xe0;
    patch[12] = 0x90;
    patch[13] = 0x90;
    DWORD old, ignored;
    if (!VirtualProtect(code, size, PAGE_EXECUTE_READWRITE, &old)) return nullptr;
    std::memcpy(code, patch, std::min<size_t>(size, sizeof(patch)));
    for (size_t i = sizeof(patch); i < size; ++i) code[i] = 0x90;
    VirtualProtect(code, size, old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), code, size);
    return trampoline;
}
} // namespace

bool IndirectEnabled() {
    // Default on (=0 off): city 4.17 M indirect draws native, scenes visually checked (a3v-*).
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_INDIRECT");
        return !value || value[0] != '0';
    }();
    return enabled;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_PASSES");
        return value && value[0] == '1';
    }();
    return enabled;
}

int Install(u64 image_base) {
    if (!Enabled()) return 0;
    if (const char* g = std::getenv("ISS_NATIVE_GBUFFER"); g && g[0] == '1') {
        std::fprintf(stderr, "NATIVE_PASSES refused: ISS_NATIVE_GBUFFER also wraps the chunk callback\n");
        return -1;
    }
    registry = std::make_unique<Slot[]>(RegistrySize);
    if (submit_mode) {
        if (const char* p = std::getenv("ISS_NATIVE_PASSES_TEMPLATE_GATE"); !p || p[0] != '0')
            dedup_by_key = false;
        std::fprintf(stderr, "NATIVE_PASSES submit mode (no draw time hooks)" "\n");
        return 1;
    }
    original = reinterpret_cast<DrawIndex>(Patch(image_base, DrawIndexOffset, DrawIndexPrologue,
                                                 sizeof(DrawIndexPrologue),
                                                 reinterpret_cast<const void*>(&DrawIndexHook)));
    std::fprintf(stderr, "NATIVE_PASSES %s at eboot+0x%llx\n", original ? "installed" : "refused",
                 (unsigned long long)DrawIndexOffset);
    screen_scissor_original = reinterpret_cast<Setter>(
        Patch(image_base, 0x626890, SetterPrologue, sizeof(SetterPrologue),
              reinterpret_cast<const void*>(&SetterHook<&screen_scissor_original>)));
    generic_scissor_original = reinterpret_cast<Setter>(
        Patch(image_base, 0x6269D0, SetterPrologue, sizeof(SetterPrologue),
              reinterpret_cast<const void*>(&SetterHook<&generic_scissor_original>)));
    std::fprintf(stderr, "NATIVE_PASSES scissor setters screen=%d generic=%d\n",
                 screen_scissor_original != nullptr, generic_scissor_original != nullptr);
    return original ? 1 : -1;
}

void SetCallbacks(const Callbacks& callbacks) {
    cb = callbacks;
    callbacks_set.store(true, std::memory_order_release);
}

Segment* Claim(const u32* packet) {
    if (!registry) return nullptr;
    const u32 h = Hash(packet);
    for (u32 probe = 0; probe < 16; ++probe) {
        Slot& slot = registry[(h + probe) & (RegistrySize - 1)];
        const u32* key = slot.key.load(std::memory_order_acquire);
        if (!key) return nullptr;
        if (key != packet) continue;
        Segment* s = slot.segment.load(std::memory_order_acquire);
        if (!s) return nullptr;
        const u32* const t = TokenWords(packet);
        if (t[0] != TokenMagic || t[1] != s->generation || t[2] != s->token) return nullptr;
        u32 expected = Ready;
        if (!s->state.compare_exchange_strong(expected, Claimed)) return nullptr;
        if (s->first_draw != packet || t[1] != s->generation) {
            s->state.store(Ready, std::memory_order_release);
            return nullptr;
        }
        stats.claimed.fetch_add(1, std::memory_order_relaxed);
        return s;
    }
    return nullptr;
}

void Finish(Segment& s, bool executed, u64 tick) {
    if (s.kind == Kind::Learn) {
        stats.learned.fetch_add(1, std::memory_order_relaxed);
        s.state.store(Abandoned, std::memory_order_release);
        return;
    }
    if (executed) {
        stats.executed.fetch_add(1, std::memory_order_relaxed);
        stats.executed_draws.fetch_add(s.draw_packets.size(), std::memory_order_relaxed);
        s.executed_tick.store(tick, std::memory_order_relaxed);
        s.state.store(Done, std::memory_order_release);
    } else {
        stats.rejected.fetch_add(1, std::memory_order_relaxed);
        cb.abandon(s);
        s.state.store(Abandoned, std::memory_order_release);
    }
}

void PublishSpanStart(const u32* span, const u32* reg_array) {
    auto start = std::make_unique<SpanStart>();
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(&start->words[base], reg_array + TrackedFirst[r], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    std::scoped_lock lock{span_mutex};
    latest_start = std::make_unique<SpanStart>(*start);
    if (span_starts.size() >= 4096) span_starts.clear();
    span_starts[span] = std::move(start);
}

void NoteDivergence(const Segment& s, const u32* reg_array) {
    if (s.snapshot.size() != TrackedWords_) return;
    ++divergence_samples;
    u32 base = 0;
    // Only the words the lane wrote: they must be the command processor's.
    for (u32 r = 0; r < 3; ++r) {
        for (u32 k = 0; k < TrackedCount[r]; ++k) {
            const u32 t = base + k;
            if (s.mask.size() == MaskWords && (s.mask[t >> 5] >> (t & 31) & 1))
                divergence[t] += s.snapshot[t] != reg_array[TrackedFirst[r] + k];
        }
        base += TrackedCount[r];
    }
}


// The command processor's command buffer whose jobs it waits for one by one (JOB_WAIT).
thread_local const u32* next_job_boundary = reinterpret_cast<const u32*>(~std::uintptr_t{0});
namespace {
thread_local std::shared_ptr<DcbWork> cp_work;
thread_local u32 cp_next_job = 0;
bool JobWait() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_JOB_WAIT");
        return !p || p[0] != '0';
    }();
    return enabled;
}
// ISS_NATIVE_PASSES_CP_RECORDS=1 (opt-in: no measurable gain, sessions 2026-10-20): a job no pool thread has taken yet is
// recorded by the command processor itself instead of sleeping until a pool thread is free, takes
// it and wakes it (the same recording, on the thread that needs it; the pool was behind on the
// heavy frames: cp_wait p99 12.7 ms, session 2026-10-19).
bool CpRecords() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_CP_RECORDS");
        return p && p[0] == '1';
    }();
    return enabled;
}

// Waits (pool lock held) until job k of the command processor's command buffer is recorded.
void WaitJob(std::unique_lock<std::mutex>& lock, DcbWork& work, u32 k) {
    const auto done = [&] { return work.job_done[k].load(std::memory_order_acquire) != 0; };
    if (done()) return;
    const u64 t0 = NowUs();
    if (CpRecords()) {
        const auto it = std::find_if(submit.jobs.begin(), submit.jobs.end(), [&](const Job& j) {
            return j.work.get() == &work && j.index == k;
        });
        if (it != submit.jobs.end()) {
            Job job = std::move(*it);
            submit.jobs.erase(it);
            lock.unlock();
            static thread_local std::unique_ptr<Lane> lane;
            if (!lane) lane = NewJobLane();
            RecordCommandBuffer(*lane, job.begin, job.end, job.start);
            FinishJob(job);
            lock.lock();
            submit.cp_records.fetch_add(1, std::memory_order_relaxed);
            submit.cp_record_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
            return;
        }
    }
    // ISS_NATIVE_PASSES_SPIN_US=N (opt-in, default 0: 88% of the waits ended spinning but the
    // command processor's wait time and the FPS stayed the same, clean-s2b 2026-10-20). A pool thread is recording the job; most end
    // within ~100 us, and a condition variable wake on Windows costs tens of microseconds more
    // (~9 waits a submission, ~86 us each, session 2026-10-20). Watch the job briefly first.
    static const u64 spin_us = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_SPIN_US");
        return p ? u64(std::max(0, std::atoi(p))) : u64(0);
    }();
    if (spin_us) {
        lock.unlock();
        while (!done() && NowUs() - t0 < spin_us) _mm_pause();
        lock.lock();
        if (done()) {
            submit.cp_spins.fetch_add(1, std::memory_order_relaxed);
            submit.cp_wait_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
            return;
        }
    }
    submit.done_cv.wait(lock, done);
    submit.cp_waits.fetch_add(1, std::memory_order_relaxed);
    submit.cp_wait_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
}
void SetBoundary() {
    next_job_boundary = cp_work && cp_next_job < cp_work->job_begins.size()
                            ? cp_work->job_begins[cp_next_job]
                            : reinterpret_cast<const u32*>(~std::uintptr_t{0});
}
} // namespace

void WaitRecorded(const u32* dcb) {
    if (!SubmitMode()) return;
    std::unique_lock lock{submit.pool_mutex};
    const auto it = submit.pending.find(dcb);
    if (it == submit.pending.end()) return;
    const auto work = it->second;
    submit.pending.erase(it);
    const auto scanned = [&] { return work->scanned.load(std::memory_order_acquire); };
    const auto done = [&] {
        return scanned() && work->jobs_left.load(std::memory_order_acquire) == 0;
    };
    if (JobWait()) {
        // The scan's jobs are known once it is scanned: wait for the first one only; the
        // others as the command processor reaches them (WaitRecordedAt).
        if (!scanned()) {
            const u64 t0 = NowUs();
            submit.done_cv.wait(lock, scanned);
            submit.cp_waits.fetch_add(1, std::memory_order_relaxed);
            submit.cp_wait_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
        }
        cp_work.reset();
        cp_next_job = 0;
        if (!work->job_done || work->job_begins.empty() || done()) {
            SetBoundary();
            return;
        }
        WaitJob(lock, *work, 0);
        cp_work = work;
        cp_next_job = 1;
        SetBoundary();
        return;
    }
    if (done()) return;
    const u64 t0 = NowUs();
    submit.done_cv.wait(lock, done);
    submit.cp_waits.fetch_add(1, std::memory_order_relaxed);
    submit.cp_wait_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
}

void WaitRecordedSlow(const u32* packet) {
    if (!cp_work) {
        SetBoundary();
        return;
    }
    std::unique_lock lock{submit.pool_mutex};
    while (cp_next_job < cp_work->job_begins.size() && packet >= cp_work->job_begins[cp_next_job]) {
        WaitJob(lock, *cp_work, cp_next_job);
        ++cp_next_job;
    }
    if (cp_next_job >= cp_work->job_begins.size()) cp_work.reset();
    SetBoundary();
}

bool SubmitMode() {
    return Enabled() && submit_mode;
}

bool NeedsSeed() {
    return SubmitMode() && callbacks_set.load(std::memory_order_acquire) && !submit.seeded;
}

void Seed(const u32* reg_array, const u32* cs_words, u32 cs_count) {
    std::scoped_lock lock{submit.mutex};
    if (!submit.shadow) submit.shadow = NewJobLane();
    std::vector<u32> words;
    for (u32 r = 0; r < 3; ++r)
        words.insert(words.end(), reg_array + TrackedFirst[r], reg_array + TrackedFirst[r] + TrackedCount[r]);
    words.resize(TrackedWords + CsWords);
    std::memcpy(&words[TrackedWords], cs_words, std::min(cs_count, CsWords) * 4);
    LoadTracked(*submit.shadow, words);
    submit.seeded = true;
    std::fprintf(stderr, "NATIVE_PASSES_SUBMIT seeded from the command processor's registers" "\n");
}

void ScanOnly(const u32* dcb, u32 dwords) {
    if (!SubmitMode() || !submit.seeded) return;
    std::scoped_lock lock{submit.mutex};
    std::scoped_lock shadow_lock{submit.shadow->mutex};
    Apply(*submit.shadow, dcb, dcb + dwords);
}

// Scans one command buffer in submission order from the shadow registers (the caller holds
// submit.mutex): its jobs, the registers at its start and every job_draws draws.
void ScanCommandBuffer(const u32* begin, const u32* end, const std::shared_ptr<DcbWork>& work,
                       std::vector<Job>& jobs) {
    static const u32 job_draws = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_JOB_DRAWS");
        // 48 (was 96): the command processor's wait for the pool 1.47 -> 0.89 ms per frame in the
        // city, the first job of each command buffer ready sooner (jd48 2026-10-18).
        return p ? u32(std::max(8, std::atoi(p))) : 48u;
    }();
    const auto snapshot = [](const Lane& shadow) {
        std::vector<u32> start;
        start.reserve(TrackedWords + CsWords + 3);
        for (u32 r = 0; r < 3; ++r)
            start.insert(start.end(), shadow.words + TrackedFirst[r],
                         shadow.words + TrackedFirst[r] + TrackedCount[r]);
        start.insert(start.end(), shadow.words + CsBase, shadow.words + CsBase + CsWords);
        // The indexed indirect state (LoadTracked).
        start.push_back(u32(shadow.indirect_base));
        start.push_back(u32(shadow.indirect_base >> 32));
        start.push_back(shadow.indirect_seen);
        return start;
    };
    Lane& shadow = *submit.shadow;
    std::scoped_lock shadow_lock{shadow.mutex};
    const size_t first_job = jobs.size();
    jobs.push_back({begin, end, snapshot(shadow), work});
    const u32* p = begin;
    const u32* applied = begin;
    u32 draws = 0;
    while (p < end) {
        const u32 header = *p;
        if ((header >> 30) == 2) { ++p; continue; }
        if ((header >> 30) != 3) break;
        const u32* next = p + 1 + ((header >> 16) & 0x3FFF) + 1;
        if (next > end) break;
        const u32 op = (header >> 8) & 0xFF;
        // Every recorded kind of draw counts: indexed indirect draws (0x25, ISS_NATIVE_INDIRECT)
        // and auto-index draws (0x2D) left out made the passes made of them one long job, which a
        // single pool thread recorded while the command processor waited (a third of its time
        // inside the city's hitches, cpprof3 2026-10-18).
        if ((op == 0x27 || op == 0x25 || op == 0x2D || (op == 0x15 && cb.record_dispatch)) &&
            ++draws % job_draws == 0) {
            // A new job from this draw on, at the registers it draws with.
            Apply(shadow, applied, p);
            applied = p;
            jobs.back().end = p;
            jobs.push_back({p, end, snapshot(shadow), work});
        }
        p = next;
    }
    Apply(shadow, applied, end);
    const u32 n = u32(jobs.size() - first_job);
    work->job_begins.resize(n);
    work->job_done = std::make_unique<std::atomic<u8>[]>(n);
    for (u32 k = 0; k < n; ++k) {
        jobs[first_job + k].index = k;
        work->job_begins[k] = jobs[first_job + k].begin;
        work->job_done[k].store(0, std::memory_order_relaxed);
    }
    work->jobs_left.store(n, std::memory_order_relaxed);
}

void StartPool() {
    if (!submit.threads.empty()) return;
    static const u32 n = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_THREADS");
        return p ? u32(std::max(1, std::atoi(p))) : 6u;
    }();
    for (u32 i = 0; i < n; ++i) submit.threads.emplace_back(PoolThread);
}

// Hands the jobs of scanned command buffers to the pool. first: ahead of the queued ones (a
// compute queue's submission: its submitting thread waits for them).
void QueueJobs(std::vector<Job>& jobs, const std::vector<std::shared_ptr<DcbWork>>& works,
               bool first = false) {
    {
        std::scoped_lock pool_lock{submit.pool_mutex};
        if (first)
            for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) submit.jobs.push_front(std::move(*it));
        else
            for (auto& job : jobs) submit.jobs.push_back(std::move(job));
        for (const auto& work : works) work->scanned.store(true, std::memory_order_release);
    }
    jobs.clear();
    submit.work_cv.notify_all();
    submit.done_cv.notify_all();
}

// ISS_NATIVE_PASSES_ASYNC_SCAN=1: scans the submissions off the submitting thread, one command
// buffer at a time. The frame is a chain (the game waits for the GPU, submits, the command
// processor waits for that submit): the game's submit returns at once and the command processor
// starts on the first command buffer sooner.
void ScanThread() {
    SetThreadDescription(GetCurrentThread(), L"ISS:NativePassesScan");
    std::vector<Job> jobs;
    for (;;) {
        SubmitState::ScanItem item;
        {
            std::unique_lock lock{submit.scan_mutex};
            submit.scan_cv.wait(lock, [] { return !submit.scans.empty(); });
            item = std::move(submit.scans.front());
            submit.scans.pop_front();
        }
        const u64 t0 = NowUs();
        {
            std::scoped_lock lock{submit.mutex};
            ScanCommandBuffer(item.begin, item.end, item.work, jobs);
        }
        QueueJobs(jobs, {item.work});
        submit.scan_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
    }
}

bool AsyncScan() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_PASSES_ASYNC_SCAN");
        return p && p[0] == '1';
    }();
    return enabled;
}

void BeforeSubmit(u32 count, const u32* const* dcbs, const u32* sizes_in_bytes) {
    if (SubmitMode()) {
        if (!submit.seeded || count == 0) return;
        StartPool();
        std::vector<std::shared_ptr<DcbWork>> works;
        works.reserve(count);
        for (u32 i = 0; i < count; ++i) works.push_back(std::make_shared<DcbWork>());
        if (AsyncScan()) {
            // The command processor waits for each command buffer's scan and recording.
            {
                std::scoped_lock pool_lock{submit.pool_mutex};
                for (u32 i = 0; i < count; ++i) submit.pending[dcbs[i]] = works[i];
            }
            {
                std::scoped_lock lock{submit.scan_mutex};
                if (!submit.scan_thread.joinable()) submit.scan_thread = std::thread(ScanThread);
                for (u32 i = 0; i < count; ++i)
                    submit.scans.push_back({dcbs[i], dcbs[i] + sizes_in_bytes[i] / 4, works[i]});
            }
            submit.scan_cv.notify_one();
            submit.submits.fetch_add(1, std::memory_order_relaxed);
            submit.dcbs_recorded.fetch_add(count, std::memory_order_relaxed);
            return;
        }
        const u64 t0 = NowUs();
        // The registers at each command buffer's start and every job_draws draws, in submission
        // order: the jobs, recorded by the pool after this returns.
        std::vector<Job> jobs;
        {
            std::scoped_lock lock{submit.mutex};
            for (u32 i = 0; i < count; ++i)
                ScanCommandBuffer(dcbs[i], dcbs[i] + sizes_in_bytes[i] / 4, works[i], jobs);
        }
        {
            std::scoped_lock pool_lock{submit.pool_mutex};
            for (u32 i = 0; i < count; ++i) submit.pending[dcbs[i]] = works[i];
        }
        QueueJobs(jobs, works);
        submit.submits.fetch_add(1, std::memory_order_relaxed);
        submit.scan_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
        submit.dcbs_recorded.fetch_add(count, std::memory_order_relaxed);
        return;
    }

    if (!Enabled()) return;
    std::vector<Lane*> copy;
    {
        std::scoped_lock lock{lanes_mutex};
        copy = all_lanes;
    }
    for (Lane* l : copy) {
        std::scoped_lock lock{l->mutex};
        CloseSegment(*l, CloseSubmit);
    }
}

// ---- ISS_NATIVE_ASC: the asynchronous compute queues ---------------------------------------
//
// The engine's particle simulation runs on the compute rings (DingDong): ~300 tiny dispatches a
// frame in the city, ~1300 in the forest, ~5 us of the command processor each through the full
// path (cpdiag-p-cidade 2026-10-18: the rings were ~1.9 ms of its ~10 ms a frame). At each
// DingDong the ring's indirect buffers are scanned on the submitting thread from the queue's own
// compute registers (a shadow per queue, kept in submission order; the words the queue never
// wrote are the command processor's) and their dispatches recorded as the graphics queue's are
// (RecordCommandBuffer), on the submitting thread itself (or by the pool, the submitting thread
// waiting for it, bounded), before the command processor gets the submission. The token of every scanned dispatch is cleared
// first, so a segment recorded for an earlier submission of the same memory is never claimed.
namespace {
constexpr u32 MaxAscQueues = 64;
struct AscShadow {
    std::mutex mutex;
    std::array<u32, CsWords> words{};
    std::array<bool, CsWords> seen{};
};
std::array<AscShadow, MaxAscQueues> asc_shadows;
struct AscStats {
    std::atomic<u64> submits{}, ibs{}, dispatches{}, jobs{}, waits{}, wait_us{}, timeouts{},
        scan_us{};
};
AscStats asc_stats;

u32 AscEnv(const char* name, u32 fallback) {
    const char* p = std::getenv(name);
    return p && *p ? u32(std::max(1, std::atoi(p))) : fallback;
}

// Scans one indirect buffer of a compute queue into jobs (the caller holds the shadow's lock).
void ScanAscBuffer(AscShadow& sh, const u32* begin, const u32* end,
                   const std::shared_ptr<DcbWork>& work, std::vector<Job>& jobs, u32 depth) {
    // A segment holds at most MaxComputeDispatches anyway; smaller jobs only help the pool.
    static const u32 job_dispatches = AscEnv("ISS_NATIVE_ASC_JOB", 64);
    const auto snapshot = [&] {
        std::vector<u32> start(TrackedWords + CsWords, 0u);
        std::memcpy(&start[TrackedWords], sh.words.data(), CsWords * 4);
        return start;
    };
    asc_stats.ibs.fetch_add(1, std::memory_order_relaxed);
    u32 in_job = 0;  // dispatches of the open job
    bool open = false;
    const u32* p = begin;
    while (p < end) {
        const u32 header = *p;
        if ((header >> 30) == 2) { ++p; continue; }
        if ((header >> 30) != 3) break;
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        const u32* next = p + 1 + count;
        if (next > end) break;
        if (opcode == 0x76 && count >= 2 && p[1] >= 0x200) {
            for (u32 i = 0; i + 1 < count && p[1] - 0x200 + i < CsWords; ++i) {
                sh.words[p[1] - 0x200 + i] = p[2 + i];
                sh.seen[p[1] - 0x200 + i] = true;
            }
        } else if (opcode == 0x15 && count == 4) {
            asc_stats.dispatches.fetch_add(1, std::memory_order_relaxed);
            // gnmdriver's trailing NOP: its token is cleared (only the new recording sets it).
            if (next + 4 <= end && (next[0] >> 30) == 3 && ((next[0] >> 8) & 0xFF) == 0x10 &&
                ((next[0] >> 16) & 0x3FFF) == 2)
                const_cast<u32*>(next)[1] = 0;
            if (!open || in_job >= job_dispatches) {
                // A job from this dispatch on, at the registers it dispatches with.
                if (open) jobs.back().end = p;
                jobs.push_back({p, end, snapshot(), work});
                open = true;
                in_job = 0;
            }
            ++in_job;
        } else if (opcode == 0x3F && count >= 3) {
            // A nested indirect buffer: the open job ends here (its registers change inside).
            if (open) jobs.back().end = p;
            open = false;
            const auto* ib = reinterpret_cast<const u32*>(u64(p[1]) | (u64(p[2] & 0xFFFF) << 32));
            const u32 ib_dwords = p[3] & 0xFFFFF;
            if (depth < 2 && ib && ib_dwords) ScanAscBuffer(sh, ib, ib + ib_dwords, work, jobs, depth + 1);
        }
        p = next;
    }
}
} // namespace

bool AscEnabled() {
    // Default on (=0 off) since GDS dispatches are recorded too (ISS_NATIVE_COMPUTE_GDS): the forest
    // 99.7% of the rings native, CP 12.6 -> 9.7 ms, GPU 12.4 -> 11.6 ms, fire and smoke right
    // (a4-visitors-*, 2026-10-18).
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_ASC");
        const char* c = std::getenv("ISS_NATIVE_COMPUTE");
        return !(p && p[0] == '0') && !(c && c[0] == '0');
    }();
    return enabled && SubmitMode() && cb.record_dispatch;
}

void BeforeAscSubmit(u32 qid, const u32* ring, u32 dwords, const u32* cp_cs, u32 cp_words) {
    if (!AscEnabled() || qid >= MaxAscQueues || !ring || dwords == 0) return;
    const u64 t0 = NowUs();
    StartPool();
    auto work = std::make_shared<DcbWork>();
    std::vector<Job> jobs;
    {
        AscShadow& sh = asc_shadows[qid];
        std::scoped_lock lock{sh.mutex};
        for (u32 i = 0; i < CsWords && i < cp_words; ++i)
            if (!sh.seen[i]) sh.words[i] = cp_cs[i];
        // The ring: register writes and indirect buffers (the dispatches are in these).
        const u32* p = ring;
        const u32* const end = ring + dwords;
        while (p < end) {
            const u32 header = *p;
            if ((header >> 30) == 2) { ++p; continue; }
            if ((header >> 30) != 3) break;
            const u32 count = ((header >> 16) & 0x3FFF) + 1;
            const u32 opcode = (header >> 8) & 0xFF;
            const u32* next = p + 1 + count;
            if (next > end) break; // split at the ring's end: the rest comes with the next one
            if (opcode == 0x76 && count >= 2 && p[1] >= 0x200) {
                for (u32 i = 0; i + 1 < count && p[1] - 0x200 + i < CsWords; ++i) {
                    sh.words[p[1] - 0x200 + i] = p[2 + i];
                    sh.seen[p[1] - 0x200 + i] = true;
                }
            } else if (opcode == 0x3F && count >= 3) {
                const auto* ib =
                    reinterpret_cast<const u32*>(u64(p[1]) | (u64(p[2] & 0xFFFF) << 32));
                const u32 ib_dwords = p[3] & 0xFFFFF;
                if (ib && ib_dwords) ScanAscBuffer(sh, ib, ib + ib_dwords, work, jobs, 0);
            }
            p = next;
        }
    }
    asc_stats.submits.fetch_add(1, std::memory_order_relaxed);
    if (jobs.empty()) {
        asc_stats.scan_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
        return;
    }
    asc_stats.jobs.fetch_add(jobs.size(), std::memory_order_relaxed);
    // ISS_NATIVE_ASC_INLINE (default on, =0: the pool): recorded here, on the submitting thread
    // (a submission is ~30 dispatches; the pool made it wait ~0.8 ms behind the graphics jobs,
    // cpdiag-q-cidade 2026-10-18).
    static const bool record_inline = [] {
        const char* p = std::getenv("ISS_NATIVE_ASC_INLINE");
        return !p || p[0] != '0';
    }();
    static const bool async_mode = [] {
        const char* p = std::getenv("ISS_NATIVE_ASC_ASYNC");
        return p && p[0] == '1';
    }();
    if (record_inline && !async_mode) {
        static thread_local std::unique_ptr<Lane> lane;
        if (!lane) lane = NewJobLane();
        for (const Job& job : jobs) RecordCommandBuffer(*lane, job.begin, job.end, job.start);
        asc_stats.scan_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
        return;
    }
    work->jobs_left.store(u32(jobs.size()), std::memory_order_relaxed);
    // ISS_NATIVE_ASC_ASYNC=1: the game's thread returns at once (recording ~1300 dispatches a
    // frame took ~13% of it in the forest, profv 2026-10-18); the command processor waits for
    // the recording when it reaches this submission (AscRecorded).
    static const bool record_async = [] {
        const char* p = std::getenv("ISS_NATIVE_ASC_ASYNC");
        return p && p[0] == '1';
    }();
    if (record_async) {
        {
            std::scoped_lock lock{submit.pool_mutex};
            submit.asc_pending[ring] = {work, 0};
        }
        QueueJobs(jobs, {work}, true);
        asc_stats.scan_us.fetch_add(NowUs() - t0, std::memory_order_relaxed);
        return;
    }
    QueueJobs(jobs, {work}, true);
    const u64 t1 = NowUs();
    asc_stats.scan_us.fetch_add(t1 - t0, std::memory_order_relaxed);
    static const u32 wait_limit_us = AscEnv("ISS_NATIVE_ASC_WAIT_US", 4000);
    std::unique_lock lock{submit.pool_mutex};
    const bool done = submit.done_cv.wait_for(lock, std::chrono::microseconds(wait_limit_us), [&] {
        return work->jobs_left.load(std::memory_order_acquire) == 0;
    });
    asc_stats.waits.fetch_add(1, std::memory_order_relaxed);
    asc_stats.wait_us.fetch_add(NowUs() - t1, std::memory_order_relaxed);
    if (!done) asc_stats.timeouts.fetch_add(1, std::memory_order_relaxed);
}

bool AscRecorded(const u32* ring) {
    std::unique_lock lock{submit.pool_mutex};
    if (submit.asc_pending.empty()) return true;
    const auto it = submit.asc_pending.find(ring);
    if (it == submit.asc_pending.end()) return true;
    auto& [work, since] = it->second;
    if (work->jobs_left.load(std::memory_order_acquire) == 0) {
        if (since) {
            asc_stats.waits.fetch_add(1, std::memory_order_relaxed);
            asc_stats.wait_us.fetch_add(NowUs() - since, std::memory_order_relaxed);
        }
        submit.asc_pending.erase(it);
        return true;
    }
    const u64 now = NowUs();
    if (!since) since = now;
    if (now - since > 50'000) {
        // Never stall the command processor on it: the dispatches not recorded yet take the
        // full path (their tokens are not set).
        asc_stats.timeouts.fetch_add(1, std::memory_order_relaxed);
        submit.asc_pending.erase(it);
        return true;
    }
    return false;
}

void Report() {
    auto& s = stats;
    if (const u64 n = submit.submits.load(std::memory_order_relaxed))
        std::fprintf(stderr,
                     "NATIVE_PASSES_SUBMIT submits=%llu dcbs=%llu jobs=%llu scan_us/submit=%llu "
                     "cp_waits=%llu cp_wait_us/submit=%llu cp_records=%llu cp_record_us/submit=%llu cp_spins=%llu\n",
                     (unsigned long long)n, (unsigned long long)submit.dcbs_recorded.load(),
                     (unsigned long long)submit.jobs_done.load(),
                     (unsigned long long)(submit.scan_us.load() / n),
                     (unsigned long long)submit.cp_waits.load(),
                     (unsigned long long)(submit.cp_wait_us.load() / n),
                     (unsigned long long)submit.cp_records.load(),
                     (unsigned long long)(submit.cp_record_us.load() / n),
                     (unsigned long long)submit.cp_spins.load());
    if (const u64 n = asc_stats.submits.load(std::memory_order_relaxed))
        std::fprintf(stderr,
                     "NATIVE_ASC submits=%llu ibs=%llu dispatches=%llu jobs=%llu scan_us/submit=%.1f "
                     "waits=%llu wait_us/wait=%.1f timeouts=%llu\n",
                     (unsigned long long)n, (unsigned long long)asc_stats.ibs.load(),
                     (unsigned long long)asc_stats.dispatches.load(),
                     (unsigned long long)asc_stats.jobs.load(),
                     double(asc_stats.scan_us.load()) / double(n),
                     (unsigned long long)asc_stats.waits.load(),
                     asc_stats.waits.load() ? double(asc_stats.wait_us.load()) /
                                                  double(asc_stats.waits.load())
                                            : 0.0,
                     (unsigned long long)asc_stats.timeouts.load());
    if (divergence_samples) {
        std::vector<std::pair<u64, u32>> regs;
        u32 base = 0;
        for (u32 r = 0; r < 3; ++r) {
            for (u32 k = 0; k < TrackedCount[r]; ++k)
                if (divergence[base + k]) regs.emplace_back(divergence[base + k], TrackedFirst[r] + k);
            base += TrackedCount[r];
        }
        std::sort(regs.rbegin(), regs.rend());
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(regs.size(), 24); ++i) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", regs[i].second, (unsigned long long)regs[i].first);
            text += item;
        }
        std::fprintf(stderr, "NATIVE_PASSES_DIVERGENCE samples=%llu regs[%s ]\n",
                     (unsigned long long)divergence_samples, text.c_str());
    }
    std::string closes;
    for (u32 i = 0; i < CloseCount; ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", close_names[i],
                      (unsigned long long)s.close[i].load(std::memory_order_relaxed));
        closes += item;
    }
    std::vector<std::pair<u64, u32>> top;
    for (u32 i = 0; i < 257; ++i)
        if (const u64 n = s.boundary_opcodes[i].load(std::memory_order_relaxed)) top.emplace_back(n, i);
    std::sort(top.rbegin(), top.rend());
    std::string opcodes;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
        char item[32];
        std::snprintf(item, sizeof(item), " %x=%llu", top[i].second, (unsigned long long)top[i].first);
        opcodes += item;
    }
    std::string unknown;
    for (u32 i = 0; i < 258; ++i)
        if (const u64 n = s.unknown_opcodes[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", i, (unsigned long long)n);
            unknown += item;
        }
    if (IndirectEnabled())
        std::fprintf(stderr, "NATIVE_INDIRECT seen=%llu recorded=%llu\n",
                     (unsigned long long)s.indirect_seen.load(),
                     (unsigned long long)s.indirect_recorded.load());
    std::fprintf(stderr, "NATIVE_PASSES_LANES restarts=%llu unknown_draws=%llu no_segment=%llu learn_deduped=%llu closed_inside=%llu unknown[%s ]\n",
                 (unsigned long long)s.restarts.load(), (unsigned long long)s.unknown_draws.load(),
                 (unsigned long long)s.no_segment.load(),
                 (unsigned long long)s.learn_deduped.load(),
                 (unsigned long long)s.closed_inside.load(), unknown.c_str());
    std::fprintf(stderr,
                 "NATIVE_PASSES draws=%llu recorded=%llu segments=%llu learn_posted=%llu "
                 "unknown_state=%llu not_ready=%llu abandoned=%llu no_slot=%llu | cp claimed=%llu "
                 "executed=%llu executed_draws=%llu rejected=%llu learned=%llu close[%s ] "
                 "boundaries[%s ]\n",
                 (unsigned long long)s.draws.load(), (unsigned long long)s.recorded.load(),
                 (unsigned long long)s.segments.load(), (unsigned long long)s.learn_posted.load(),
                 (unsigned long long)s.unknown_state.load(), (unsigned long long)s.not_ready.load(),
                 (unsigned long long)s.abandoned.load(), (unsigned long long)s.no_slot.load(),
                 (unsigned long long)s.claimed.load(), (unsigned long long)s.executed.load(),
                 (unsigned long long)s.executed_draws.load(), (unsigned long long)s.rejected.load(),
                 (unsigned long long)s.learned.load(), closes.c_str(), opcodes.c_str());
    if (const u64 n = s.dispatches.load(std::memory_order_relaxed)) {
        std::vector<std::pair<u64, u32>> cs_top;
        for (u32 i = 0; i < 257; ++i)
            if (const u64 k = s.cs_close_opcodes[i].load(std::memory_order_relaxed))
                cs_top.emplace_back(k, i);
        std::sort(cs_top.rbegin(), cs_top.rend());
        std::string cs_closes;
        for (size_t i = 0; i < std::min<size_t>(cs_top.size(), 10); ++i) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", cs_top[i].second,
                          (unsigned long long)cs_top[i].first);
            cs_closes += item;
        }
        std::fprintf(stderr,
                     "NATIVE_COMPUTE_SCAN dispatches=%llu recorded=%llu segments=%llu "
                     "learn_posted=%llu no_nop=%llu closed_by[%s ]\n",
                     (unsigned long long)n, (unsigned long long)s.dispatches_recorded.load(),
                     (unsigned long long)s.cs_segments.load(),
                     (unsigned long long)s.cs_learn_posted.load(),
                     (unsigned long long)s.cs_no_nop.load(), cs_closes.c_str());
    }
}

} // namespace iss::native_passes
