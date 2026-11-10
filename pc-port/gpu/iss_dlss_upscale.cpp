// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_DLSS_UPSCALE (D2): see iss_dlss_upscale.h. Port of axylpup/shadPS4 second-son-v1.1.1
// vk_dlss_upscale.cpp (GPL-2.0-or-later). The composite shader is iss_dlss_composite.comp
// (SPIR-V below, glslangValidator -V --target-env vulkan1.2).
#include "iss_dlss_upscale.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"

namespace iss::dlss {

namespace {
constexpr u32 kCompositeSpv[] = {
    0x07230203, 0x00010500, 0x0008000b, 0x00000091, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000032, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e,
    0x00000000, 0x00000001, 0x000a000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000002f,
    0x00000037, 0x0000005e, 0x00000062, 0x0000006a, 0x00060010, 0x00000004, 0x00000011, 0x00000008,
    0x00000008, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00070005, 0x0000000b, 0x62677253, 0x694c6f54, 0x7261656e, 0x33667628, 0x0000003b,
    0x00030005, 0x0000000a, 0x00000063, 0x00030005, 0x0000000d, 0x00006f6c, 0x00030005, 0x00000012,
    0x00006968, 0x00030005, 0x0000002b, 0x00736f70, 0x00080005, 0x0000002f, 0x475f6c67, 0x61626f6c,
    0x766e496c, 0x7461636f, 0x496e6f69, 0x00000044, 0x00040005, 0x00000034, 0x657a6973, 0x00000000,
    0x00050005, 0x00000037, 0x6d617266, 0x756f5f65, 0x00000074, 0x00030005, 0x00000050, 0x00007675,
    0x00030005, 0x0000005b, 0x00647568, 0x00050005, 0x0000005e, 0x5f647568, 0x6579616c, 0x00000072,
    0x00060005, 0x00000062, 0x656e696c, 0x735f7261, 0x6c706d61, 0x00007265, 0x00040005, 0x00000069,
    0x65736162, 0x00000000, 0x00040005, 0x0000006a, 0x6e656373, 0x00000065, 0x00040005, 0x00000073,
    0x6f636e65, 0x00646564, 0x00040005, 0x00000081, 0x70617773, 0x0062725f, 0x00040005, 0x00000088,
    0x61726170, 0x0000006d, 0x00040047, 0x0000002f, 0x0000000b, 0x0000001c, 0x00030047, 0x00000037,
    0x00000019, 0x00040047, 0x00000037, 0x00000021, 0x00000002, 0x00040047, 0x00000037, 0x00000022,
    0x00000000, 0x00040047, 0x0000005e, 0x00000021, 0x00000000, 0x00040047, 0x0000005e, 0x00000022,
    0x00000000, 0x00040047, 0x00000062, 0x00000021, 0x00000003, 0x00040047, 0x00000062, 0x00000022,
    0x00000000, 0x00040047, 0x0000006a, 0x00000021, 0x00000001, 0x00040047, 0x0000006a, 0x00000022,
    0x00000000, 0x00040047, 0x00000081, 0x00000001, 0x00000000, 0x00040047, 0x00000090, 0x0000000b,
    0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000003, 0x00040020, 0x00000008, 0x00000007,
    0x00000007, 0x00040021, 0x00000009, 0x00000007, 0x00000008, 0x0004002b, 0x00000006, 0x0000000f,
    0x414eb852, 0x0004002b, 0x00000006, 0x00000014, 0x3d6147ae, 0x0004002b, 0x00000006, 0x00000017,
    0x3f870a3d, 0x0004002b, 0x00000006, 0x0000001a, 0x4019999a, 0x0006002c, 0x00000007, 0x0000001b,
    0x0000001a, 0x0000001a, 0x0000001a, 0x0004002b, 0x00000006, 0x00000020, 0x3d25aee6, 0x0006002c,
    0x00000007, 0x00000021, 0x00000020, 0x00000020, 0x00000020, 0x00020014, 0x00000022, 0x00040017,
    0x00000023, 0x00000022, 0x00000003, 0x00040015, 0x00000028, 0x00000020, 0x00000001, 0x00040017,
    0x00000029, 0x00000028, 0x00000002, 0x00040020, 0x0000002a, 0x00000007, 0x00000029, 0x00040015,
    0x0000002c, 0x00000020, 0x00000000, 0x00040017, 0x0000002d, 0x0000002c, 0x00000003, 0x00040020,
    0x0000002e, 0x00000001, 0x0000002d, 0x0004003b, 0x0000002e, 0x0000002f, 0x00000001, 0x00040017,
    0x00000030, 0x0000002c, 0x00000002, 0x00090019, 0x00000035, 0x00000006, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000002, 0x00000002, 0x00040020, 0x00000036, 0x00000000, 0x00000035,
    0x0004003b, 0x00000036, 0x00000037, 0x00000000, 0x0004002b, 0x0000002c, 0x0000003a, 0x00000000,
    0x00040020, 0x0000003b, 0x00000007, 0x00000028, 0x0004002b, 0x0000002c, 0x00000044, 0x00000001,
    0x00040017, 0x0000004e, 0x00000006, 0x00000002, 0x00040020, 0x0000004f, 0x00000007, 0x0000004e,
    0x0004002b, 0x00000006, 0x00000053, 0x3f000000, 0x00040017, 0x00000059, 0x00000006, 0x00000004,
    0x00040020, 0x0000005a, 0x00000007, 0x00000059, 0x00090019, 0x0000005c, 0x00000006, 0x00000001,
    0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00040020, 0x0000005d, 0x00000000,
    0x0000005c, 0x0004003b, 0x0000005d, 0x0000005e, 0x00000000, 0x0002001a, 0x00000060, 0x00040020,
    0x00000061, 0x00000000, 0x00000060, 0x0004003b, 0x00000061, 0x00000062, 0x00000000, 0x0003001b,
    0x00000064, 0x0000005c, 0x0004002b, 0x00000006, 0x00000067, 0x00000000, 0x0004003b, 0x0000005d,
    0x0000006a, 0x00000000, 0x0004002b, 0x00000028, 0x0000006f, 0x00000000, 0x0004002b, 0x0000002c,
    0x00000076, 0x00000003, 0x00040020, 0x00000077, 0x00000007, 0x00000006, 0x0004002b, 0x00000006,
    0x0000007d, 0x3f800000, 0x00030030, 0x00000022, 0x00000081, 0x0004002b, 0x0000002c, 0x0000008f,
    0x00000008, 0x0006002c, 0x0000002d, 0x00000090, 0x0000008f, 0x0000008f, 0x00000044, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x0000002a,
    0x0000002b, 0x00000007, 0x0004003b, 0x0000002a, 0x00000034, 0x00000007, 0x0004003b, 0x0000004f,
    0x00000050, 0x00000007, 0x0004003b, 0x0000005a, 0x0000005b, 0x00000007, 0x0004003b, 0x00000008,
    0x00000069, 0x00000007, 0x0004003b, 0x00000008, 0x00000073, 0x00000007, 0x0004003b, 0x00000008,
    0x00000088, 0x00000007, 0x0004003d, 0x0000002d, 0x00000031, 0x0000002f, 0x0007004f, 0x00000030,
    0x00000032, 0x00000031, 0x00000031, 0x00000000, 0x00000001, 0x0004007c, 0x00000029, 0x00000033,
    0x00000032, 0x0003003e, 0x0000002b, 0x00000033, 0x0004003d, 0x00000035, 0x00000038, 0x00000037,
    0x00040068, 0x00000029, 0x00000039, 0x00000038, 0x0003003e, 0x00000034, 0x00000039, 0x00050041,
    0x0000003b, 0x0000003c, 0x0000002b, 0x0000003a, 0x0004003d, 0x00000028, 0x0000003d, 0x0000003c,
    0x00050041, 0x0000003b, 0x0000003e, 0x00000034, 0x0000003a, 0x0004003d, 0x00000028, 0x0000003f,
    0x0000003e, 0x000500af, 0x00000022, 0x00000040, 0x0000003d, 0x0000003f, 0x000400a8, 0x00000022,
    0x00000041, 0x00000040, 0x000300f7, 0x00000043, 0x00000000, 0x000400fa, 0x00000041, 0x00000042,
    0x00000043, 0x000200f8, 0x00000042, 0x00050041, 0x0000003b, 0x00000045, 0x0000002b, 0x00000044,
    0x0004003d, 0x00000028, 0x00000046, 0x00000045, 0x00050041, 0x0000003b, 0x00000047, 0x00000034,
    0x00000044, 0x0004003d, 0x00000028, 0x00000048, 0x00000047, 0x000500af, 0x00000022, 0x00000049,
    0x00000046, 0x00000048, 0x000200f9, 0x00000043, 0x000200f8, 0x00000043, 0x000700f5, 0x00000022,
    0x0000004a, 0x00000040, 0x00000005, 0x00000049, 0x00000042, 0x000300f7, 0x0000004c, 0x00000000,
    0x000400fa, 0x0000004a, 0x0000004b, 0x0000004c, 0x000200f8, 0x0000004b, 0x000100fd, 0x000200f8,
    0x0000004c, 0x0004003d, 0x00000029, 0x00000051, 0x0000002b, 0x0004006f, 0x0000004e, 0x00000052,
    0x00000051, 0x00050050, 0x0000004e, 0x00000054, 0x00000053, 0x00000053, 0x00050081, 0x0000004e,
    0x00000055, 0x00000052, 0x00000054, 0x0004003d, 0x00000029, 0x00000056, 0x00000034, 0x0004006f,
    0x0000004e, 0x00000057, 0x00000056, 0x00050088, 0x0000004e, 0x00000058, 0x00000055, 0x00000057,
    0x0003003e, 0x00000050, 0x00000058, 0x0004003d, 0x0000005c, 0x0000005f, 0x0000005e, 0x0004003d,
    0x00000060, 0x00000063, 0x00000062, 0x00050056, 0x00000064, 0x00000065, 0x0000005f, 0x00000063,
    0x0004003d, 0x0000004e, 0x00000066, 0x00000050, 0x00070058, 0x00000059, 0x00000068, 0x00000065,
    0x00000066, 0x00000002, 0x00000067, 0x0003003e, 0x0000005b, 0x00000068, 0x0004003d, 0x0000005c,
    0x0000006b, 0x0000006a, 0x0004003d, 0x00000060, 0x0000006c, 0x00000062, 0x00050056, 0x00000064,
    0x0000006d, 0x0000006b, 0x0000006c, 0x0004003d, 0x00000029, 0x0000006e, 0x0000002b, 0x00040064,
    0x0000005c, 0x00000070, 0x0000006d, 0x0007005f, 0x00000059, 0x00000071, 0x00000070, 0x0000006e,
    0x00000002, 0x0000006f, 0x0008004f, 0x00000007, 0x00000072, 0x00000071, 0x00000071, 0x00000000,
    0x00000001, 0x00000002, 0x0003003e, 0x00000069, 0x00000072, 0x0004003d, 0x00000059, 0x00000074,
    0x0000005b, 0x0008004f, 0x00000007, 0x00000075, 0x00000074, 0x00000074, 0x00000000, 0x00000001,
    0x00000002, 0x00050041, 0x00000077, 0x00000078, 0x0000005b, 0x00000076, 0x0004003d, 0x00000006,
    0x00000079, 0x00000078, 0x0004003d, 0x00000007, 0x0000007a, 0x00000069, 0x0005008e, 0x00000007,
    0x0000007b, 0x0000007a, 0x00000079, 0x00050081, 0x00000007, 0x0000007c, 0x00000075, 0x0000007b,
    0x00060050, 0x00000007, 0x0000007e, 0x00000067, 0x00000067, 0x00000067, 0x00060050, 0x00000007,
    0x0000007f, 0x0000007d, 0x0000007d, 0x0000007d, 0x0008000c, 0x00000007, 0x00000080, 0x00000001,
    0x0000002b, 0x0000007c, 0x0000007e, 0x0000007f, 0x0003003e, 0x00000073, 0x00000080, 0x000300f7,
    0x00000083, 0x00000000, 0x000400fa, 0x00000081, 0x00000082, 0x00000083, 0x000200f8, 0x00000082,
    0x0004003d, 0x00000007, 0x00000084, 0x00000073, 0x0008004f, 0x00000007, 0x00000085, 0x00000084,
    0x00000084, 0x00000002, 0x00000001, 0x00000000, 0x0003003e, 0x00000073, 0x00000085, 0x000200f9,
    0x00000083, 0x000200f8, 0x00000083, 0x0004003d, 0x00000035, 0x00000086, 0x00000037, 0x0004003d,
    0x00000029, 0x00000087, 0x0000002b, 0x0004003d, 0x00000007, 0x00000089, 0x00000073, 0x0003003e,
    0x00000088, 0x00000089, 0x00050039, 0x00000007, 0x0000008a, 0x0000000b, 0x00000088, 0x00050051,
    0x00000006, 0x0000008b, 0x0000008a, 0x00000000, 0x00050051, 0x00000006, 0x0000008c, 0x0000008a,
    0x00000001, 0x00050051, 0x00000006, 0x0000008d, 0x0000008a, 0x00000002, 0x00070050, 0x00000059,
    0x0000008e, 0x0000008b, 0x0000008c, 0x0000008d, 0x0000007d, 0x00040063, 0x00000086, 0x00000087,
    0x0000008e, 0x000100fd, 0x00010038, 0x00050036, 0x00000007, 0x0000000b, 0x00000000, 0x00000009,
    0x00030037, 0x00000008, 0x0000000a, 0x000200f8, 0x0000000c, 0x0004003b, 0x00000008, 0x0000000d,
    0x00000007, 0x0004003b, 0x00000008, 0x00000012, 0x00000007, 0x0004003d, 0x00000007, 0x0000000e,
    0x0000000a, 0x00060050, 0x00000007, 0x00000010, 0x0000000f, 0x0000000f, 0x0000000f, 0x00050088,
    0x00000007, 0x00000011, 0x0000000e, 0x00000010, 0x0003003e, 0x0000000d, 0x00000011, 0x0004003d,
    0x00000007, 0x00000013, 0x0000000a, 0x00060050, 0x00000007, 0x00000015, 0x00000014, 0x00000014,
    0x00000014, 0x00050081, 0x00000007, 0x00000016, 0x00000013, 0x00000015, 0x00060050, 0x00000007,
    0x00000018, 0x00000017, 0x00000017, 0x00000017, 0x00050088, 0x00000007, 0x00000019, 0x00000016,
    0x00000018, 0x0007000c, 0x00000007, 0x0000001c, 0x00000001, 0x0000001a, 0x00000019, 0x0000001b,
    0x0003003e, 0x00000012, 0x0000001c, 0x0004003d, 0x00000007, 0x0000001d, 0x0000000d, 0x0004003d,
    0x00000007, 0x0000001e, 0x00000012, 0x0004003d, 0x00000007, 0x0000001f, 0x0000000a, 0x000500ba,
    0x00000023, 0x00000024, 0x0000001f, 0x00000021, 0x000600a9, 0x00000007, 0x00000025, 0x00000024,
    0x0000001e, 0x0000001d, 0x000200fe, 0x00000025, 0x00010038
};
constexpr vk::ImageUsageFlags SceneUsage =
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};
} // namespace

Upscaler::Upscaler(const Vulkan::Instance& instance_)
    : instance{instance_}, device{instance_.GetDevice()} {
    sampler = Vulkan::Check<"create dlss composite sampler">(device.createSamplerUnique(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }));
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings{{
        {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 3, .descriptorType = vk::DescriptorType::eSampler, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute, .pImmutableSamplers = &sampler.get()},
    }};
    set_layout = Vulkan::Check<"create dlss composite set layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    pipeline_layout = Vulkan::Check<"create dlss composite layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout.get(),
    }));
    auto module = Vulkan::Check<"create dlss composite module">(device.createShaderModuleUnique({
        .codeSize = sizeof(kCompositeSpv),
        .pCode = kCompositeSpv,
    }));
    // ISS_DLSS_SWAP_RB=0: keep R / B as they are (AxylPup swaps: the game's images hold BGR).
    static const VkBool32 swap_rb = [] {
        const char* v = std::getenv("ISS_DLSS_SWAP_RB");
        return VkBool32(!(v && v[0] == '0'));
    }();
    const vk::SpecializationMapEntry entry{.constantID = 0, .offset = 0, .size = sizeof(VkBool32)};
    const vk::SpecializationInfo spec{.mapEntryCount = 1, .pMapEntries = &entry,
                                      .dataSize = sizeof(VkBool32), .pData = &swap_rb};
    pipeline = Vulkan::Check<"create dlss composite pipeline">(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = *module,
                    .pName = "main",
                    .pSpecializationInfo = &spec,
                },
                .layout = pipeline_layout.get(),
            }));
    LOG_INFO(Render_Vulkan, "DLSS upscale: composite ready (swap R/B {})", swap_rb != 0);
}

Upscaler::~Upscaler() = default;

void Upscaler::SetOutputSize(u32 width, u32 height) {
    ++flips;
    if (flips % 600 == 0)
        std::fprintf(stderr,
                     "DLSS HUD draws: blend path %llu (alpha %llu), native passes %llu (alpha %llu); "
                     "pipeline keys %llu (alpha fixed %llu); layer %#llx active %d\n",
                     (unsigned long long)g_hud_draws[0].load(), (unsigned long long)g_hud_alpha[0].load(),
                     (unsigned long long)g_hud_draws[1].load(), (unsigned long long)g_hud_alpha[1].load(),
                     (unsigned long long)g_key_seen.load(), (unsigned long long)g_key_fixed.load(),
                     (unsigned long long)HudLayerAddr(), HudLayerActive() ? 1 : 0);
    while (!retired.empty() && retired.front().first + 16 <= flips) {
        retired.pop_front();
    }
    const u32 game_w = g_inputs.game_w, game_h = g_inputs.game_h;
    // At most 2x the render size (DLSS Performance's floor); the post pass scales the rest.
    double fit = 1.0;
    if (game_w > 0 && game_h > 0 && (width > 2 * game_w || height > 2 * game_h)) {
        fit = std::min(2.0 * game_w / width, 2.0 * game_h / height);
    }
    output_size = vk::Extent2D{std::max<u32>(game_w, static_cast<u32>(width * fit)),
                               std::max<u32>(game_h, static_cast<u32>(height * fit))};
    // DLSS needs an output larger than its input; smaller windows keep DLAA in place.
    const bool on = game_w > 0 && width > game_w && height > game_h &&
                    g_ready.load(std::memory_order_relaxed) &&
                    instance.IsDynamicColorWriteMaskSupported();
    if (on != g_hud_layer.load(std::memory_order_relaxed)) {
        LOG_INFO(Render_Vulkan, "DLSS upscale: {} (render {}x{}, window {}x{}, output {}x{})",
                 on ? "on" : "off", game_w, game_h, width, height, output_size.width,
                 output_size.height);
        std::fprintf(stderr, "DLSS upscale: %s (render %ux%u, window %ux%u, output %ux%u)\n",
                     on ? "on" : "off", game_w, game_h, width, height, output_size.width,
                     output_size.height);
        g_hud_layer.store(on, std::memory_order_relaxed);
    }
    if (!on) {
        ready.store(false, std::memory_order_release);
    }
}

void Upscaler::Resize(vk::Extent2D size) {
    if (scene) retired.emplace_back(flips, std::move(scene));
    if (out) retired.emplace_back(flips, std::move(out));
    const auto make = [&](vk::Format format, const char* name) {
        auto t = std::make_unique<Target>();
        t->image = VideoCore::UniqueImage(device, instance.GetAllocator());
        t->image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{size.width, size.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = SceneUsage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        Vulkan::SetObjectName(device, static_cast<vk::Image>(t->image), "{}", name);
        t->view = Vulkan::Check<"create dlss upscale view">(device.createImageViewUnique({
            .image = t->image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = ColorRange,
        }));
        return t;
    };
    // Transfer usage too: NGX clears its output with vkCmdClearColorImage.
    scene = make(vk::Format::eR8G8B8A8Unorm, "DLSS output");
    out = make(vk::Format::eR16G16B16A16Sfloat, "DLSS composite");
    image_size = size;
    LOG_INFO(Render_Vulkan, "DLSS upscale: output images {}x{}", size.width, size.height);
}

EvalImage Upscaler::SceneTarget() {
    if (!scene || image_size != output_size) {
        Resize(output_size);
    }
    ready.store(false, std::memory_order_release);
    return EvalImage{
        .image = scene->image,
        .view = scene->view.get(),
        .format = vk::Format::eR8G8B8A8Unorm,
        .layout = vk::ImageLayout::eGeneral,
        .usage = SceneUsage,
        .width = image_size.width,
        .height = image_size.height,
    };
}

void Upscaler::ToGeneral(vk::CommandBuffer cmdbuf, vk::Image image) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
        .newLayout = vk::ImageLayout::eGeneral,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &barrier});
}

void Upscaler::ToRead(vk::CommandBuffer cmdbuf, vk::Image image) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &barrier});
}

CompositeJob Upscaler::TakeJob(vk::ImageView hud_view) {
    ready.store(false, std::memory_order_release);
    return CompositeJob{
        .pipeline = pipeline.get(),
        .layout = pipeline_layout.get(),
        .hud_view = hud_view,
        .scene_view = scene->view.get(),
        .out_image = out->image,
        .out_view = out->view.get(),
        .size = image_size,
    };
}

void RecordComposite(vk::CommandBuffer cmdbuf, const CompositeJob& job) {
    const vk::ImageMemoryBarrier2 to_write{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
        .newLayout = vk::ImageLayout::eGeneral,
        .image = job.out_image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &to_write});
    const vk::DescriptorImageInfo hud_info{.imageView = job.hud_view,
                                           .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo scene_info{.imageView = job.scene_view,
                                             .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    const vk::DescriptorImageInfo out_info{.imageView = job.out_view,
                                           .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 3> writes{{
        {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &hud_info},
        {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &scene_info},
        {.dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &out_info},
    }};
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, job.pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, job.layout, 0, writes);
    cmdbuf.dispatch((job.size.width + 7) / 8, (job.size.height + 7) / 8, 1);
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = job.out_image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                               .pImageMemoryBarriers = &to_read});
}

Upscaler* GetUpscaler(const Vulkan::Instance& instance) {
    if (!UpscaleWanted() || !g_ready.load(std::memory_order_relaxed)) {
        return nullptr;
    }
    // Never destroyed: a static destructor would run after the device is gone.
    static Upscaler* const upscaler = new Upscaler(instance);
    return upscaler;
}

} // namespace iss::dlss
