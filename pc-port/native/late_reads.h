// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>

namespace iss::native {
// ISS_PARTICLE_LATE_READ=1 (late_reads.cpp): the particle readback moves to the start of the next
// frame. Before guest entry; 0 when off or on a different game version, <0 on a write failure.
int InstallParticleLateRead(unsigned char* image, std::uint64_t image_size);
} // namespace iss::native
