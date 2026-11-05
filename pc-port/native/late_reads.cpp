// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_PARTICLE_LATE_READ=1: the particle control blocks (survivor counts, light counts, CPU
// collision) are read at the start of the next frame, just before eboot+0x3ee7d0 consumes them,
// instead of right after this frame's particle chain is kicked. Game:Main then no longer waits for
// that chain's GPU work in the middle of the frame (readback site eboot+0x3f3e00: ~58% of the
// guest's readback wait in a D.U.P. fight, 2026-11-04). The data is complete, one frame later.
// Approach and bytes from axylpup/shadPS4 (second-son-v1.1.1, SHADPS4_PARTICLE_LATE_READ V1,
// GPL-2.0-or-later). CUSA00004 01.07 only: every byte is compared first, all or nothing.
//   - a cave in the unreferenced function at 0x3ee160 calls 0x3edef0 (readback) and 0x3ee470
//     (collision join + label wait), sets up 0x3ee7d0's arguments and jumps back;
//   - the main loop jumps to the cave at 0x29cc89;
//   - the old 0x3edef0 call at 0x29cd1e becomes a 5-byte NOP (the 0x3ee470 call at 0x29cd56 stays).
// ISS_PARTICLE_LIGHT_ONE_SUBMIT=1 (with the late read; AxylPup SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT,
// same bytes): 0x3f3ed0 records two DMA_DATAs per light-emitting particle effect and then submits
// the ring (call 0x1691e0 at 0x3f4036): ~25 submits per fight frame, ~74 at worst. With the late
// read the copies stay in the command buffer and go out with 0x3ee7d0's first submit of the same
// frame, so that call becomes a 5-byte NOP.
#include "late_reads.h"
#include "../src/runtime.h"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

namespace {
struct Bytes {
    std::uint64_t at;
    std::size_t size;
    const unsigned char* bytes;
};

constexpr unsigned char kCaveOriginal[] = {
    0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x8b, 0x5f, 0x58, 0xeb, 0x13,
    0x0f, 0x1f, 0x40, 0x00, 0x48, 0x89, 0xdf, 0xe8, 0x58, 0x5d, 0x00, 0x00,
    0x48, 0x8b, 0x9b, 0xb0, 0x00, 0x00, 0x00, 0x48, 0x85, 0xdb, 0x75, 0xec,
    0x48, 0x83, 0xc4, 0x08, 0x5b, 0x5d, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90};
// call 0x41b8d0; lea rax,[image+0xf1bcd0]; lea rdi,[image+0x7d3eb10]; mov rsi,[rax+0x1dca0];
// call 0x3ee7d0
constexpr unsigned char kLoopOriginal[] = {
    0xe8, 0x47, 0xec, 0x17, 0x00, 0x48, 0x8d, 0x05, 0x40, 0xf0, 0xc7, 0x00, 0x48, 0x8d, 0x3d, 0x79,
    0x1e, 0xaa, 0x07, 0x48, 0x8b, 0xb0, 0xa0, 0xdc, 0x01, 0x00, 0xe8, 0x2d, 0x1b, 0x15, 0x00};
// lea rbx,[image+0x7d3eb10]; mov rdi,rbx; call 0x3edef0
constexpr unsigned char kReadOriginal[] = {0x48, 0x8d, 0x1d, 0xf5, 0x1d, 0xaa, 0x07, 0x48,
                                           0x89, 0xdf, 0xe8, 0xcd, 0x11, 0x15, 0x00};
// mov rdi,rbx; call 0x3ee470 (kept)
constexpr unsigned char kJoinOriginal[] = {0x48, 0x89, 0xdf, 0xe8, 0x15, 0x17, 0x15, 0x00};

// lea rbx,[image+0x7d3eb10]; mov rdi,rbx; call 0x3edef0; mov rdi,rbx; call 0x3ee470;
// lea rax,[image+0xf1bcd0]; mov rdi,rbx; mov rsi,[rax+0x1dca0]; jmp 0x29cc9e
constexpr unsigned char kCave[] = {
    0x48, 0x8d, 0x1d, 0xa9, 0x09, 0x95, 0x07, 0x48, 0x89, 0xdf, 0xe8, 0x81, 0xfd, 0xff, 0xff,
    0x48, 0x89, 0xdf, 0xe8, 0xf9, 0x02, 0x00, 0x00, 0x48, 0x8d, 0x05, 0x52, 0xdb, 0xb2, 0x00,
    0x48, 0x89, 0xdf, 0x48, 0x8b, 0xb0, 0xa0, 0xdc, 0x01, 0x00, 0xe9, 0x11, 0xeb, 0xea, 0xff};
// jmp 0x3ee160, then NOPs up to the call 0x3ee7d0 at 0x29cc9e
constexpr unsigned char kLoopJump[] = {0xe9, 0xd2, 0x14, 0x15, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                                       0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
constexpr unsigned char kNop5[] = {0x0f, 0x1f, 0x44, 0x00, 0x00};
// call 0x1691e0 (the per-light ring submit at 0x3f4036)
constexpr unsigned char kLightSubmitOriginal[] = {0xe8, 0xa5, 0x51, 0xd7, 0xff};

bool Write(unsigned char* image, std::uint64_t at, const void* bytes, std::size_t size) {
    DWORD old, ignored;
    if (!VirtualProtect(image + at, size, PAGE_EXECUTE_READWRITE, &old)) return false;
    std::memcpy(image + at, bytes, size);
    return VirtualProtect(image + at, size, old, &ignored) &&
           FlushInstructionCache(GetCurrentProcess(), image + at, size);
}
} // namespace

namespace iss::native {
int InstallParticleLateRead(unsigned char* image, std::uint64_t size) {
    const char* option = std::getenv("ISS_PARTICLE_LATE_READ");
    if (!option || option[0] != '1') return 0;
    const Bytes original[] = {{0x3ee160, sizeof(kCaveOriginal), kCaveOriginal},
                              {0x29cc84, sizeof(kLoopOriginal), kLoopOriginal},
                              {0x29cd14, sizeof(kReadOriginal), kReadOriginal},
                              {0x29cd53, sizeof(kJoinOriginal), kJoinOriginal}};
    for (const auto& o : original) {
        if (size <= o.at + o.size || std::memcmp(image + o.at, o.bytes, o.size) != 0) {
            std::fprintf(stderr, "PARTICLE_LATE_READ rejected: eboot+0x%llx differs (CUSA00004 01.07 only)\n",
                         static_cast<unsigned long long>(o.at));
            return 0;
        }
    }
    const bool ok = Write(image, 0x3ee160, kCave, sizeof(kCave)) &&
                    Write(image, 0x29cc89, kLoopJump, sizeof(kLoopJump)) &&
                    Write(image, 0x29cd1e, kNop5, sizeof(kNop5));
    if (!ok) {
        std::fputs("PARTICLE_LATE_READ failed: could not write the game code\n", stderr);
        return -1;
    }
    std::fputs("PARTICLE_LATE_READ applied: particle control blocks read one frame later "
               "(cave eboot+0x3ee160, jump 0x29cc89, nop 0x29cd1e)\n", stderr);
    if (const char* one = std::getenv("ISS_PARTICLE_LIGHT_ONE_SUBMIT"); one && one[0] == '1') {
        constexpr std::uint64_t at = 0x3f4036;
        if (size <= at + sizeof(kLightSubmitOriginal) ||
            std::memcmp(image + at, kLightSubmitOriginal, sizeof(kLightSubmitOriginal)) != 0) {
            std::fputs("PARTICLE_LIGHT_ONE_SUBMIT rejected: eboot+0x3f4036 differs\n", stderr);
        } else if (!Write(image, at, kNop5, sizeof(kNop5))) {
            std::fputs("PARTICLE_LIGHT_ONE_SUBMIT failed: could not write the game code\n", stderr);
        } else {
            std::fputs("PARTICLE_LIGHT_ONE_SUBMIT applied: per-light ring submit at eboot+0x3f4036 "
                       "is a NOP\n", stderr);
        }
    }
    return 1;
}
} // namespace iss::native
