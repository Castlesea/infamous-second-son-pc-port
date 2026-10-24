// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Pause guard. While the game is paused (eboot counter at +0x1D1F7FC: sub_27DB10 increments it,
// sub_27E210 decrements it; it is also 1 in the start menus) and for ISS_PAUSE_GUARD_HOLD_MS
// (3000) after it, the native paths of ISS_PAUSE_GUARD_PATHS are off and the command processor
// draws as the original engine does. Native work executed in the pause hung the GPU (TDR), see
// out/experimental/20261017-native-passes-v2/CORRECAO-HANG-PAUSE.md: native blended segments into
// the pause UI layer, which ISS_NATIVE_PASSES_BLEND_DENY now keeps on the command processor.
// ISS_PAUSE_GUARD_PATHS (bits, default 0): 63 turns every path off in the pause, as before (the
// pause menu at 37-48 FPS against 60 with them on, 2026-10-24, no hang in play testing).
// ISS_PAUSE_GUARD=0 turns the guard off altogether.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "gpu_blackbox.h"

namespace iss::pause_guard {

inline constexpr std::uint64_t CounterOffset = 0x1D1F7FC;
inline const volatile std::uint32_t* counter{};
// What Blocked(path) asks about; ISS_PAUSE_GUARD_PATHS is a mask of them.
enum Path : std::uint32_t {
    Passes = 1,    ///< native pass segments (DRAW_INDEX_2, DRAW_INDEX_AUTO, indirect draws)
    Compute = 2,   ///< native compute segments (graphics and ASC queues)
    Shadows = 4,   ///< native shadow batches and suffixes
    GBuffer = 8,   ///< native G-buffer ranges
    Materials = 16, ///< native material bindings
    FastDraw = 32, ///< the per-draw fast paths of Rasterizer::Draw and the engine draw items
    AllPaths = 63,
};

inline std::int64_t hold_ms = 3000;
inline std::uint32_t paths = 0;
inline std::atomic<std::int64_t> last_paused_ms{};
inline std::atomic<bool> active{};
inline std::atomic<std::uint64_t> engaged{};

inline std::int64_t NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline void Install(std::uint64_t image_base) {
    if (const char* v = std::getenv("ISS_PAUSE_GUARD"); v && v[0] == '0') {
        std::fprintf(stderr, "PAUSE_GUARD off (ISS_PAUSE_GUARD=0): native paths run in the pause\n");
        return;
    }
    if (const char* v = std::getenv("ISS_PAUSE_GUARD_HOLD_MS")) hold_ms = std::strtoll(v, nullptr, 10);
    if (const char* v = std::getenv("ISS_PAUSE_GUARD_PATHS"); v && *v)
        paths = std::uint32_t(std::strtoul(v, nullptr, 0)) & AllPaths;
    counter = reinterpret_cast<const volatile std::uint32_t*>(image_base + CounterOffset);
    std::fprintf(stderr, "PAUSE_GUARD on: counter eboot+0x%llx = %u, hold %lld ms, paths 0x%x\n",
                 (unsigned long long)CounterOffset, *counter, (long long)hold_ms, unsigned(paths));
}

// True while every native path must stay off. One volatile load when the game is running.
inline bool Blocked() {
    if (!counter) return false;
    if (*counter == 0 && !active.load(std::memory_order_relaxed)) return false;
    const std::int64_t now = NowMs();
    if (*counter != 0) {
        last_paused_ms.store(now, std::memory_order_relaxed);
        if (!active.exchange(true, std::memory_order_relaxed)) {
            engaged.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "PAUSE_GUARD engaged: native paths off\n");
            iss::blackbox::Event(iss::blackbox::General, "PAUSE: entrou (nativo desligado)");
        }
        return true;
    }
    if (now - last_paused_ms.load(std::memory_order_relaxed) < hold_ms) return true;
    if (active.exchange(false, std::memory_order_relaxed)) {
        std::fprintf(stderr, "PAUSE_GUARD released: native paths on\n");
        iss::blackbox::Event(iss::blackbox::General, "PAUSE: saiu ha 3 s (nativo religado)");
    }
    return false;
}

// True while the given native path must stay off: the pause and its hold, and the path is one
// ISS_PAUSE_GUARD_PATHS turns off.
inline bool Blocked(Path path) { return Blocked() && (paths & path) != 0; }

} // namespace iss::pause_guard
