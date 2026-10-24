// SPDX-License-Identifier: GPL-2.0-or-later
// Prepared PS material resources selected by the engine's validated Item identity.
// Pipeline selection, VS/object data, geometry and recording retain the common renderer.
static bool NativeVerifyEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_VERIFY");
    return enabled;
}
void Rasterizer::NativeVisualPlan::Rebind() {
    for (size_t i = 0; i < writes.size(); ++i) {
        writes[i].pBufferInfo = buffer_starts[i] == UINT32_MAX ? nullptr : buffers.data() + buffer_starts[i];
        writes[i].pImageInfo = image_starts[i] == UINT32_MAX ? nullptr : images.data() + image_starts[i];
    }
}
bool Rasterizer::NativeVisualPlan::Capture(std::span<const vk::WriteDescriptorSet> source,
                                         const Shader::PushData& data) {
    *this = {};
    push = data;
    for (const auto& w : source) {
        if (w.pNext || w.pTexelBufferView || !w.descriptorCount ||
            (bool(w.pBufferInfo) == bool(w.pImageInfo))) { *this = {}; return false; }
        writes.push_back(w);
        buffer_starts.push_back(UINT32_MAX); image_starts.push_back(UINT32_MAX);
        if (w.pBufferInfo) {
            buffer_starts.back() = static_cast<u32>(buffers.size());
            buffers.insert(buffers.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
        } else {
            image_starts.back() = static_cast<u32>(images.size());
            images.insert(images.end(), w.pImageInfo, w.pImageInfo + w.descriptorCount);
        }
    }
    Rebind();
    return true;
}
bool Rasterizer::NativeVisualCaptureEnabled() const {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_VISUAL_PROBE") &&
        iss::native_item::Flag("ISS_HYBRID_BENCH_HIDDEN");
    return enabled && NativeVerifyEnabled() && NativeContext() &&
        Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed) >= 1200;
}
void Rasterizer::CompleteNativeVisualPlans(const GraphicsPipeline* pipeline) {
    auto& latch = native_visual_latch;
    latch.eligible = false;
    if (!latch.material_verified || !NativeVisualCaptureEnabled() || !pipeline ||
        !pipeline->IsReady()) return;
    const auto* id = iss::native_item::current;
    if (!id || latch.identity != std::array<u64,4>{id->technique,id->geometry,id->material,id->item}) return;
    std::vector<std::vector<u32>> flat, user;
    std::vector<u64> bases;
    u32 stages_found = 0;
    for (u32 slot = 0; const auto* stage : pipeline->GetStages()) {
        if (stage) {
            ++stages_found;
            if ((slot != u32(Shader::SwStage::Vertex) && slot != u32(Shader::SwStage::Fragment)) ||
                stage->uses_dma || stage->uses_shader_clock || stage->has_storage_images ||
                stage->uses_buffer_atomic_float_min_max || stage->uses_image_atomic_float_min_max ||
                stage->uses_buffer_int64_atomics || stage->uses_shared_int64_atomics) return;
            for (const auto& b : stage->buffers)
                if (b.is_written || (b.IsSpecial() && b.buffer_type != Shader::BufferType::Flatbuf &&
                    b.buffer_type != Shader::BufferType::ClipPlanes)) return;
            for (const auto& image : stage->images) if (image.is_written || image.is_atomic) return;
        }
        const auto f = stage ? stage->FlatUserData() : std::span<const u32>{};
        const auto u = stage ? stage->user_data : std::span<const u32>{};
        flat.emplace_back(f.begin(), f.end()); user.emplace_back(u.begin(), u.end());
        bases.push_back(stage ? stage->ProgramBase() : 0);
        ++slot;
    }
    if (stages_found != 2) return;
    // A visual sample must validate this exact draw, independently of the normal 1/64 pipeline cadence.
    const auto* canonical = pipeline_cache.GetGraphicsPipeline();
    bool same = canonical == pipeline;
    if (same) {
        const auto stages = canonical->GetStages();
        for (u32 slot = 0; same && slot < stages.size(); ++slot) {
            const auto* stage = stages[slot];
            const auto f = stage ? stage->FlatUserData() : std::span<const u32>{};
            const auto u = stage ? stage->user_data : std::span<const u32>{};
            same = (stage ? stage->ProgramBase() : 0) == bases[slot] &&
                std::ranges::equal(f, flat[slot]) && std::ranges::equal(u, user[slot]);
        }
    }
    if (!same) {
        if (auto* record = NativeRecord(false)) { record->disabled = true; record->valid = false; }
        ++native_item_stats.mismatches;
        std::fprintf(stderr,"NATIVE_VERIFY_MISMATCH field=visual_exact_draw_pipeline_or_stage_data\n");
        return;
    }
    if (latch.ps_first > set_write_index || latch.ps_count > set_write_index - latch.ps_first ||
        latch.candidate_ps.writes.size() != latch.ps_count ||
        latch.reference_ps.writes.size() != latch.ps_count ||
        std::memcmp(&latch.candidate_ps.push, &latch.reference_ps.push, sizeof(Shader::PushData))) return;
    // All remaining descriptors and push effects are from the one common VS/object preparation.
    latch.candidate_ps.Rebind(); latch.reference_ps.Rebind();
    std::vector<vk::WriteDescriptorSet> candidate(set_writes.begin(), set_writes.begin() + set_write_index);
    auto reference = candidate;
    for (u32 i = 0; i < latch.ps_count; ++i) {
        candidate[latch.ps_first+i] = latch.candidate_ps.writes[i];
        reference[latch.ps_first+i] = latch.reference_ps.writes[i];
    }
    latch.eligible = latch.candidate.Capture(candidate, push_data) &&
        latch.reference.Capture(reference, push_data);
}
bool Rasterizer::NativeContext() const {
    static const bool legacy_verify = iss::native_item::Flag("ISS_FAST_DRAW_VERIFY");
    static const bool item_enabled = iss::native_item::Flag("ISS_NATIVE_ITEMS");
    return item_enabled && iss::native_item::Enabled() && iss::native_item::current &&
           StageMemoContextEligible() && !UseDrawPipe() && !occlusion.HasOpen() &&
           !GpuCrashDiag::Enabled() && !legacy_verify && !StageMemoVerify() && !NativeVisualFailed();
}
Rasterizer::NativeDrawRecord* Rasterizer::NativeRecord(bool create) {
    const auto* id = iss::native_item::current;
    if (!id || !id->technique || !id->material || !id->geometry) return nullptr;
    const auto& regs = DrawRegs();
    const std::array<u64, 4> key{id->technique, id->material,
        regs.vs_program.Address<VAddr>(), regs.ps_program.Address<VAddr>()};
    if (auto it = native_items.find(key); it != native_items.end()) return it->second.get();
    if (!create || native_items.size() >= 8192) return nullptr;
    return native_items.emplace(key, std::make_unique<NativeDrawRecord>()).first->second.get();
}
void Rasterizer::ReportNativeItems() {
    if (!iss::native_item::Enabled()) return;
    const auto& s = native_item_stats;
    std::fprintf(stderr, "NATIVE_ITEMS candidates=%llu captures=%llu replays=%llu verified=%llu mismatches=%llu draws=%llu pipeline_hits=%llu pipeline_verified=%llu records=%zu fallback=",
        s.candidates, s.captures, s.replays, s.verified, s.mismatches, s.draws,
        s.pipeline_hits, s.pipeline_verified, native_items.size());
    for (auto n : s.fallback) std::fprintf(stderr, "%llu,", n);
    std::fputc('\n', stderr);
}
const GraphicsPipeline* Rasterizer::LookupNativeGraphicsPipeline() {
    if (!NativeContext()) return pipeline_cache.GetGraphicsPipeline();
    auto* rec = NativeRecord(false);
    if (!rec || !rec->valid || rec->disabled) return pipeline_cache.GetGraphicsPipeline();
    const auto* pipeline = rec->prepared_pipeline
        ? pipeline_cache.TryPreparedGraphics(*rec->prepared_pipeline) : nullptr;
    if (pipeline) {
        ++native_item_stats.pipeline_hits;
        if (NativeVerifyEnabled() && (native_item_stats.pipeline_hits & 63) == 0) {
            std::vector<std::vector<u32>> expected_flat;
            std::vector<u64> expected_base;
            for (const auto* stage : pipeline->GetStages()) {
                const auto data = stage ? stage->FlatUserData() : std::span<const u32>{};
                expected_flat.emplace_back(data.begin(), data.end());
                expected_base.push_back(stage ? stage->ProgramBase() : 0);
            }
            const auto* original = pipeline_cache.GetGraphicsPipeline();
            ++native_item_stats.pipeline_verified;
            bool same = original == pipeline;
            if (same) {
                const auto stages = original->GetStages();
                for (u32 slot = 0; same && slot < stages.size(); ++slot) {
                    const auto* stage = stages[slot];
                    const auto data = stage ? stage->FlatUserData() : std::span<const u32>{};
                    same = (stage ? stage->ProgramBase() : 0) == expected_base[slot] &&
                           std::ranges::equal(data, expected_flat[slot]);
                }
            }
            if (!same) {
                rec->disabled = true; rec->valid = false;
                ++native_item_stats.mismatches;
                const auto& id = *iss::native_item::current;
                std::fprintf(stderr, "NATIVE_VERIFY_MISMATCH field=pipeline_or_stage_data technique=%llx material=%llx\n",
                    id.technique, id.material);
            }
            pipeline = original;
        }
    } else {
        pipeline = pipeline_cache.GetGraphicsPipeline();
        if (pipeline == rec->pipeline)
            rec->prepared_pipeline = pipeline_cache.CapturePreparedGraphics(pipeline);
    }
    return pipeline;
}
bool Rasterizer::TryNativeItemDraw(bool indexed, u32 index_offset) {
    if (!indexed || !NativeContext() || native_recording) return false;
    auto* rec = NativeRecord(false);
    if (!rec || !rec->valid || rec->disabled) return false;
    const auto* pipeline = fast_lookup_set ? fast_lookup : LookupNativeGraphicsPipeline();
    fast_lookup = pipeline;
    fast_lookup_set = true;
    if (!pipeline || !pipeline->IsReady() || rec->pipeline != pipeline || rec->disabled) {
        ++native_item_stats.fallback[0];
        return false;
    }
    native_recording = rec;
    fast_lookup_set = false;
    DrawRecord(pipeline, indexed, index_offset);
    native_recording = nullptr;
    ++native_item_stats.draws;
    if ((native_item_stats.draws & 65535) == 0) ReportNativeItems();
    return true;
}

vk::DescriptorBufferInfo Rasterizer::PrepareNativeFlatBuffer(const Shader::Info& stage) {
    auto& stream = buffer_cache.GetStreamBuffer();
    const u32 size = stage.FlatUserData().size() * sizeof(u32);
    const u64 offset = stream.Copy(stage.FlatUserData().data(), size, instance.StorageMinAlignment());
    return vk::DescriptorBufferInfo{stream.Handle(), offset, size};
}

bool Rasterizer::PrepareNativeReadOnlyBuffer(const AmdGpu::Buffer& sharp, u64 shader_hash,
                                            u32 binding, Shader::PushData& push) {
    if (!sharp.base_address || !sharp.GetSize()) {
        buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
        return false;
    }
    const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
    if (size != sharp.GetSize())
        LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}", sharp.GetSize(), size, shader_hash);
    bool known_unwritten = false;
    const bool reuse = Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(sharp.base_address, size, false, false,
                                                          reuse ? &known_unwritten : nullptr);
    const u64 aligned = Common::AlignDown(offset, instance.StorageMinAlignment());
    const u64 adjust = offset - aligned;
    if (adjust % 4 != 0)
        LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned", shader_hash);
    push.AddOffset(binding, adjust);
    buffer_infos.emplace_back(buffer->Handle(), aligned, size + adjust);
    bool hazard = false;
    if (buffer != &buffer_cache.GetStreamBuffer()) {
        bound_buffers.emplace_back(buffer, offset, size, false, sharp.base_address);
        if (known_unwritten) Common::Perf::Add(Common::Perf::BufferChecksReused);
        else hazard = runtime.IsBufferAccessed(buffer, offset, size, false);
        needs_barrier |= hazard;
    }
    return hazard;
}

bool Rasterizer::BindNativeMaterial(const Pipeline* pipeline, const Shader::Info& stage,
                                    Shader::Backend::Bindings& binding, Shader::PushData& push) {
    if (iss::pause_guard::Blocked(iss::pause_guard::Materials)) return false; // pause guard: original binding path
    if (!NativeContext() || stage.uses_dma || !pipeline->IsReady()) return false;
    auto* rec = NativeRecord(true);
    if (!rec || rec->disabled) return false;
    if (stage.buffers.size() > 16 || stage.images.size() > 16 || stage.samplers.size() > 16)
        return false;
    for (const auto& d : stage.buffers) {
        if (d.is_written || d.is_formatted || (d.IsSpecial() && d.buffer_type != Shader::BufferType::Flatbuf &&
                            d.buffer_type != Shader::BufferType::ClipPlanes)) return false;
        if (!d.IsSpecial()) {
            const auto sharp = d.GetSharp(stage);
            if (sharp.base_address && sharp.GetSize() &&
                memory->ClampRangeSize(sharp.base_address, sharp.GetSize()) > UINT32_MAX)
                return false;
        }
    }
    const auto stages = pipeline->GetStages();
    for (u32 slot = 0; slot < stages.size(); ++slot)
        if (stages[slot] && slot != u32(Shader::SwStage::Vertex) &&
            slot != u32(Shader::SwStage::Fragment)) return false;
    if (DrawRegs().clipper_control.user_clip_plane_enable != 0) return false;
    for (const auto& d : stage.images)
        if (d.is_written || d.is_atomic || d.NumBindings(stage) != 1 ||
            d.mip_fallback_mode != Shader::MipStorageFallbackMode::None)
            return false;
    for (const auto& d : stage.samplers) {
        const auto sharp = d.GetSharp(stage);
        if (!sharp.Valid() || sharp.border_color_type.Value() == AmdGpu::BorderColor::Custom)
            return false;
    }
    auto& stats = native_item_stats;
    ++stats.candidates;
    if ((stats.candidates & 4095) == 1) ReportNativeItems();
    const auto first = binding;
    const u32 bstart = buffer_infos.size(), istart = image_infos.size();
    bool hit = rec->valid && rec->pipeline == pipeline && rec->stage == &stage &&
               rec->first == binding && rec->buffers.size() == stage.buffers.size() &&
               rec->images.size() == stage.images.size() && rec->samplers.size() == stage.samplers.size();
    if (!hit) ++stats.fallback[1];
    // Every cached resource has a local lifetime/dirty/selection proof. No global stamp.
    for (u32 i = 0; hit && i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        if (desc.IsSpecial()) continue;
        const auto sharp = desc.GetSharp(stage);
        if (rec->buffer_tokens[i] &&
            (std::memcmp(&sharp, &rec->buffers[i], sizeof(sharp)) != 0 ||
             !buffer_cache.ValidatePreparedRead(*rec->buffer_tokens[i]))) {
            hit = false; ++stats.fallback[2];
        }
    }
    for (u32 i = 0; hit && i < stage.images.size(); ++i) {
        const auto sharp = stage.images[i].GetSharp(stage);
        if (std::memcmp(&sharp, &rec->images[i], sizeof(sharp)) != 0 ||
            (rec->image_tokens[i] &&
             (!memory->IsValidGpuMapping(sharp.Address(), 0) ||
              !texture_cache.ValidatePreparedTexture(*rec->image_tokens[i])))) {
            hit = false; ++stats.fallback[3];
        }
    }
    for (u32 i = 0; hit && i < stage.samplers.size(); ++i) {
        const auto sharp = stage.samplers[i].GetSharp(stage);
        if (std::memcmp(&sharp, &rec->samplers[i], sizeof(sharp)) != 0) {
            hit = false; ++stats.fallback[4];
        }
    }
    const bool verify = hit && NativeVerifyEnabled() && ((++stats.verify_candidates & 63) == 0);
    const bool visual = verify && NativeVisualCaptureEnabled();
    struct Reference {
        Shader::Backend::Bindings binding;
        Shader::PushData push;
        std::vector<vk::DescriptorBufferInfo> buffers;
        std::vector<vk::DescriptorImageInfo> images;
        std::vector<vk::WriteDescriptorSet> writes;
        std::vector<std::array<u64, 3>> descriptors;
        std::vector<BoundBuffer> bound_buffers;
        std::vector<VideoCore::ImageId> bound_images;
    };
    std::optional<Reference> reference;
    NativeAccessAudit audit;
    std::vector<BoundBuffer> planned_buffers;
    std::vector<VideoCore::ImageId> planned_images;
    const auto input_push = push;
    const u32 wstart = set_write_index;
    const auto restore_reference = [&] {
        if (!reference) return;
        buffer_infos.resize(bstart); image_infos.resize(istart);
        for (const auto& v : reference->buffers) buffer_infos.push_back(v);
        for (const auto& v : reference->images) image_infos.push_back(v);
        set_write_index = wstart;
        for (const auto& w : reference->writes) set_writes[set_write_index++] = w;
        binding = reference->binding; push = reference->push;
    };
    if (verify) {
        // Reference preparation runs once, without recording a draw. Object allocations
        // are shared with the candidate: duplicating them would compare different stream
        // offsets for identical bytes and change the rendering work being verified.
        const auto buffer_start = bound_buffers.size(), image_start = bound_images.size();
        audit.record = rec;
        audit.buffers.resize(stage.buffers.size());
        audit.images.resize(stage.images.size());
        {
            native_access_audit = &audit;
            SCOPE_EXIT { native_access_audit = nullptr; };
            BindBuffers(stage, binding, push);
            BindTextures(stage, binding);
        }
        reference.emplace();
        reference->binding = binding; reference->push = push;
        reference->buffers.assign(buffer_infos.begin() + bstart, buffer_infos.end());
        reference->images.assign(image_infos.begin() + istart, image_infos.end());
        reference->writes.assign(set_writes.begin() + wstart, set_writes.begin() + set_write_index);
        if (visual) {
            native_visual_latch.reference_ps.Capture(
                std::span<const vk::WriteDescriptorSet>{set_writes.data() + wstart, set_write_index - wstart}, push);
        }
        reference->bound_buffers.assign(bound_buffers.begin() + buffer_start, bound_buffers.end());
        reference->bound_images.assign(bound_images.begin() + image_start, bound_images.end());
        for (const auto& w : reference->writes)
            for (u32 i = 0; i < w.descriptorCount; ++i)
                reference->descriptors.push_back(FastDrawInfo(w, i));
        buffer_infos.resize(bstart); image_infos.resize(istart);
        binding = first; push = input_push; set_write_index = wstart;
    }
    if (hit) {
        for (u32 i = 0; i < stage.buffers.size(); ++i) {
            const auto& desc = stage.buffers[i];
            vk::DescriptorBufferInfo info{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
            bool info_appended = false;
            const auto& token = rec->buffer_tokens[i];
            // A sampled candidate consumes the decision observed before the canonical
            // descriptor, never a post-reference revalidation that could hide a barrier.
            const bool token_usable = verify ? audit.buffers[i].cached
                                            : token && buffer_cache.CommitPreparedRead(*token);
            if (verify && (desc.IsSpecial() || !token_usable)) {
                info = reference->buffers[i];
                push.buf_offsets[binding.buffer] = reference->push.buf_offsets[binding.buffer];
                planned_buffers.insert(planned_buffers.end(), audit.buffers[i].bound.begin(),
                                       audit.buffers[i].bound.end());
            } else if (desc.IsSpecial()) {
                if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                    info = PrepareNativeFlatBuffer(stage);
                }
            } else if (token_usable) {
                info = rec->buffer_infos[i];
                push.buf_offsets[binding.buffer] = rec->offsets[i];
                if (verify) {
                    planned_buffers.push_back({token->buffer, token->offset, token->size, false, token->address});
                } else {
                    bound_buffers.emplace_back(token->buffer, token->offset, token->size, false, token->address);
                    needs_barrier |= runtime.IsBufferAccessed(token->buffer, token->offset, token->size, false);
                }
            } else {
                // Exactly the same helper as the original admitted read-only PS branch.
                // Verification consumes that helper's canonical allocation above, once.
                PrepareNativeReadOnlyBuffer(desc.GetSharp(stage), stage.pgm_hash, binding.buffer, push);
                info_appended = true;
            }
            if (!info_appended) buffer_infos.push_back(info);
            WriteBufferDescriptor(set_writes[set_write_index++], binding.unified++, &buffer_infos.back());
            ++binding.buffer;
        }
        // A preceding object upload can change backing selection. Revalidate after all
        // buffers; a miss uses the normal texture path without repeating the buffers.
        for (const auto& token : rec->image_tokens) {
            if (verify) break; // use the exact pre-reference whole-section decision below
            if (token && !texture_cache.ValidatePreparedTexture(*token)) {
                rec->valid = false;
                ++stats.fallback[9];
                if (verify) restore_reference();
                else BindTextures(stage, binding);
                return true;
            }
        }
        for (u32 i = 0; i < stage.images.size(); ++i) {
            if (verify && audit.images_cached && rec->image_tokens[i])
                planned_images.push_back(rec->image_tokens[i]->image_id);
            if (const auto& token = rec->image_tokens[i]; token && !verify) {
                auto& image = texture_cache.GetImage(token->image_id);
                bound_images.push_back(token->image_id);
                image.binding.is_bound = 1u;
                image.tick_accessed_last = scheduler.CurrentTick();
                needs_barrier |= runtime.Transit(&image, token->layout,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead,
                    token->resolved_view.range);
                image.usage.texture = true;
            }
            image_infos.push_back(verify && !audit.images_cached ? reference->images[i]
                                                               : rec->image_infos[i]);
            WriteImageDescriptor(set_writes[set_write_index++], binding.unified++, 1,
                vk::DescriptorType::eSampledImage, &image_infos.back());
        }
        // Sampler cache ownership/LRU remain authoritative; no stale sampler handle retained.
        for (const auto& desc : stage.samplers) {
            const auto handle = texture_cache.GetSampler(desc.GetSharp(stage), DrawRegs().ta_bc_base, desc.is_depth);
            image_infos.emplace_back(handle, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            WriteImageDescriptor(set_writes[set_write_index++], binding.unified++, 1,
                vk::DescriptorType::eSampler, &image_infos.back());
        }
        if (verify) {
            if (!audit.images_cached) planned_images = reference->bound_images;
            bool same = binding == reference->binding &&
                set_write_index - wstart == reference->writes.size();
            const char* what = same ? "" : "descriptor_shape";
            u32 descriptor = 0;
            for (u32 i = 0; same && i < reference->writes.size(); ++i) {
                const auto& got = set_writes[wstart + i];
                const auto& expected = reference->writes[i];
                same = got.dstBinding == expected.dstBinding &&
                    got.dstArrayElement == expected.dstArrayElement &&
                    got.descriptorType == expected.descriptorType &&
                    got.descriptorCount == expected.descriptorCount;
                if (!same) { what = "descriptor_shape"; break; }
                for (u32 j = 0; same && j < got.descriptorCount; ++j)
                    same = FastDrawInfo(got, j) == reference->descriptors[descriptor++];
                if (!same) what = "descriptor_resource_offset_range_view_sampler_layout";
            }
            if (same && std::memcmp(&push, &reference->push, sizeof(push)) != 0) {
                same = false; what = "push_constants";
            }
            if (same && !std::ranges::equal(planned_buffers, reference->bound_buffers,
                [](const BoundBuffer& a, const BoundBuffer& b) {
                    return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size &&
                           a.is_written == b.is_written && a.address == b.address;
                })) {
                same = false; what = "ordered_buffer_access_plan";
            }
            if (same && planned_images != reference->bound_images) {
                same = false; what = "ordered_image_access_plan";
            }
            for (const auto& access : audit.buffers) {
                if (same && access.cached && (access.hazard || access.hazard_before != access.hazard_after)) {
                    same = false; what = "cached_buffer_barrier_or_side_effect";
                }
            }
            for (const auto& access : audit.images) {
                if (same && audit.images_cached && access.cached &&
                    (access.transition || access.hazard_before != access.hazard_after)) {
                    same = false; what = "cached_image_barrier_or_side_effect";
                }
            }
            ++stats.verified;
            if (!same) {
                ++stats.mismatches; rec->disabled = true; rec->valid = false;
                const auto& id = *iss::native_item::current;
                std::fprintf(stderr, "NATIVE_VERIFY_MISMATCH field=%s technique=%llx material=%llx\n",
                    what, id.technique, id.material);
            }
            // Reference binding/hazard state is retained; the draw itself is recorded once.
            if (visual && same && native_visual_latch.reference_ps.writes.size() == reference->writes.size()) {
                auto& latch = native_visual_latch;
                latch.prepared_resources = 0;
                for (const auto& access : audit.buffers) latch.prepared_resources += access.cached;
                if (audit.images_cached)
                    for (const auto& access : audit.images) latch.prepared_resources += access.cached;
                latch.material_verified = latch.candidate_ps.Capture(
                    std::span<const vk::WriteDescriptorSet>{set_writes.data() + wstart, set_write_index - wstart}, push);
                latch.ps_first = wstart;
                latch.ps_count = set_write_index - wstart;
                const auto& id = *iss::native_item::current;
                latch.identity = {id.technique, id.geometry, id.material, id.item};
            }
            restore_reference();
        } else ++stats.replays;
        return true;
    }
    BindBuffers(stage, binding, push);
    BindTextures(stage, binding);
    rec->valid = false;
    rec->pipeline = pipeline; rec->stage = &stage; rec->first = first; rec->last = binding;
    rec->captured_push = push;
    rec->prepared_pipeline = pipeline_cache.CapturePreparedGraphics(
        static_cast<const GraphicsPipeline*>(pipeline));
    rec->buffers.resize(stage.buffers.size()); rec->buffer_tokens.assign(stage.buffers.size(), std::nullopt);
    rec->buffer_infos.resize(stage.buffers.size()); rec->offsets.resize(stage.buffers.size());
    for (u32 i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        rec->buffer_infos[i] = buffer_infos[bstart + i];
        rec->offsets[i] = push.buf_offsets[first.buffer + i];
        if (desc.IsSpecial()) continue;
        const auto sharp = desc.GetSharp(stage); rec->buffers[i] = sharp;
        if (desc.is_formatted || !sharp.base_address || !sharp.GetSize()) continue;
        const u32 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
        for (const auto& bound : bound_buffers) {
            const auto& info = rec->buffer_infos[i];
            if (!bound.is_written && bound.address == sharp.base_address && bound.size == size &&
                info.buffer == bound.buffer->Handle() &&
                info.offset + rec->offsets[i] == bound.offset) {
                rec->buffer_tokens[i] = buffer_cache.CapturePreparedRead(bound.address, size, bound.buffer, bound.offset);
                break;
            }
        }
    }
    rec->images.resize(stage.images.size()); rec->image_tokens.assign(stage.images.size(), std::nullopt);
    rec->image_infos.resize(stage.images.size());
    if (image_bindings.size() != stage.images.size()) { ++stats.fallback[6]; return true; }
    for (u32 i = 0; i < stage.images.size(); ++i) {
        const auto sharp = stage.images[i].GetSharp(stage); rec->images[i] = sharp;
        rec->image_infos[i] = image_infos[istart + i];
        if (!sharp.Address() || sharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) continue;
        const auto& [id, desc] = image_bindings[i];
        if (!id) { ++stats.fallback[7]; return true; }
        const VideoCore::TextureCache::ImageDesc original{sharp, stage.images[i]};
        rec->image_tokens[i] = texture_cache.CapturePreparedTexture(original, id, desc.view_info,
            rec->image_infos[i].imageView);
        if (!rec->image_tokens[i]) { ++stats.fallback[8]; return true; }
    }
    rec->samplers.resize(stage.samplers.size());
    for (u32 i = 0; i < stage.samplers.size(); ++i) rec->samplers[i] = stage.samplers[i].GetSharp(stage);
    rec->valid = true;
    ++stats.captures;
    return true;
}
