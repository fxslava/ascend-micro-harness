/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The golden NZ-tiled cache image, for the TURBOQUANT_INGRESS_NZ instance of the prefill
// reshape operator.
//
// The quantiser is NOT re-implemented here. tests/common/turboquant_oracle.hpp already
// produces, per (strip, plane, head) panel, the rotated values, the power-of-two scale and
// the INT4 code of every (token, channel) -- all of it layout-independent, in ChunkRef::q
// (indexed [token][channel]) and ChunkRef::scale. The strip and NZ instances of the kernel
// compute exactly the same numbers; what differs is only WHERE each one puts them. So this
// header does placement and nothing else, and it does it by calling the production address
// functions in common/turboquant_layout.h rather than restating them:
//
//   NzTiledChannelByte / NzTiledChannelNibble   the packed planes
//   NzTiledScaleIndex                            the per-row scale slots
//
// That is deliberate and is what makes the comparison meaningful. If this header carried its
// own copy of the arithmetic, a wrong shared contract would be invisible: the oracle and the
// kernel would be wrong together. Calling the shared functions means this oracle agrees with
// the kernel by construction, and the independent check that the CONTRACT itself is what the
// decode reads lives in tests/cpu/test_turboquant_kv_layout_bridge.cpp, which re-derives the
// reader's side from the decode's own steps.

#ifndef ASCEND_MICRO_TURBOQUANT_NZ_INGRESS_ORACLE_HPP
#define ASCEND_MICRO_TURBOQUANT_NZ_INGRESS_ORACLE_HPP

#include "turboquant_oracle.hpp"
#include "../../kernels/ops/turboquant/baseline/common/turboquant_layout.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace tq_nz {

namespace tq = vllm_ascend::turboquant;

// The physical cache geometry the NZ instance writes into. `page_tokens` is the fused
// decode's block_size, which its valid_launch admits at 64 or 128 -- NOT the 16-token strip
// the prefill operator quantises in one pass. Several strips share a page, each in its own
// rows of the kCubeTileRows-row tile.
struct Geometry {
    int num_kv_heads = 4;
    int head_size = 256;
    int page_tokens = 64;
    int num_pages = 2;
    // What one reshape launch quantises in a pass: the prefill operator's kTokens, which is
    // a compile-time constant of the kernel and not the cache's page size.
    int strip_tokens = 16;

    int packed_row_bytes() const { return head_size / 2; }
    std::size_t cache_bytes() const
    {
        return static_cast<std::size_t>(num_pages) * page_tokens * num_kv_heads *
               packed_row_bytes();
    }
    int scale_slot() const { return static_cast<int>(tq::NzScaleSlotFloats(
        static_cast<std::uint32_t>(num_kv_heads))); }
    std::size_t scale_floats() const
    {
        return static_cast<std::size_t>(num_pages) * page_tokens * scale_slot();
    }
    bool valid() const
    {
        return num_kv_heads >= 1 && num_kv_heads <= 4 && head_size == 256 &&
               (page_tokens == 64 || page_tokens == 128) && num_pages > 0 &&
               page_tokens % static_cast<int>(tq::kCubeTileRows) == 0 &&
               strip_tokens > 0 && static_cast<int>(tq::kCubeTileRows) % strip_tokens == 0;
    }
};

// A strip's placement: which physical page, and which token of it the strip's token 0 is.
struct StripSlot {
    int page = 0;
    int token_in_page = 0;
};

// The golden image a run of `strips` strips should leave, flooded with `fill` everywhere no
// strip wrote so an over-wide burst shows up as a changed canary rather than as nothing.
struct Image {
    std::vector<std::uint8_t> key;
    std::vector<std::uint8_t> value;
    std::vector<float> scale;
};

// Place one (plane, head) panel of one strip. `q` is the panel's INT4 codes indexed
// [token][channel] over strip_tokens * head_size, `scale` its per-token scales.
inline void place_panel(const Geometry& g, const StripSlot& slot, int plane, int head,
                        const int* q, const float* scale, std::vector<std::uint8_t>& packed,
                        std::vector<float>& scales)
{
    for (int t = 0; t < g.strip_tokens; ++t) {
        const std::uint64_t global_slot =
            static_cast<std::uint64_t>(slot.page) * g.page_tokens + slot.token_in_page + t;
        for (int c = 0; c < g.head_size; ++c) {
            const std::uint64_t byte = tq::NzTiledChannelByte(
                global_slot, static_cast<std::uint64_t>(head), static_cast<std::uint64_t>(c),
                static_cast<std::uint64_t>(g.num_kv_heads),
                static_cast<std::uint64_t>(g.head_size));
            const std::uint32_t half = tq::NzTiledChannelNibble(
                static_cast<std::uint64_t>(c), static_cast<std::uint64_t>(g.head_size));
            const int code = q[static_cast<std::size_t>(t) * g.head_size + c];
            const auto nibble = static_cast<std::uint8_t>(code & 0x0F);
            std::uint8_t& dst = packed[static_cast<std::size_t>(byte)];
            if (half == 0u) {
                dst = static_cast<std::uint8_t>((dst & 0xF0) | nibble);
            } else {
                dst = static_cast<std::uint8_t>((dst & 0x0F) | (nibble << 4));
            }
        }
        const std::uint64_t si = tq::NzTiledScaleIndex(
            global_slot, static_cast<std::uint64_t>(head), static_cast<std::uint64_t>(plane),
            static_cast<std::uint64_t>(g.num_kv_heads));
        scales[static_cast<std::size_t>(si)] = scale[t];
    }
}

// The whole golden image for a reference the strip oracle already produced. `ref` must have
// been built at block_size == g.strip_tokens, since that is the panel one reshape launch
// quantises; `slots` says where each of its strips lands in the NZ cache.
inline Image build_image(const Geometry& g, const tq_oracle::Reference& ref,
                         const std::vector<StripSlot>& slots, std::uint8_t fill,
                         float scale_fill)
{
    Image out;
    out.key.assign(g.cache_bytes(), fill);
    out.value.assign(g.cache_bytes(), fill);
    out.scale.assign(g.scale_floats(), scale_fill);
    const int strips = static_cast<int>(slots.size());
    for (int s = 0; s < strips; ++s) {
        for (int head = 0; head < g.num_kv_heads; ++head) {
            const tq_oracle::ChunkRef& key_chunk = ref.chunk(s, 0, head);
            const tq_oracle::ChunkRef& value_chunk = ref.chunk(s, 1, head);
            place_panel(g, slots[s], 0, head, key_chunk.q.data(), key_chunk.scale.data(),
                        out.key, out.scale);
            place_panel(g, slots[s], 1, head, value_chunk.q.data(), value_chunk.scale.data(),
                        out.value, out.scale);
        }
    }
    return out;
}

// Read one logical element back out of a packed NZ plane, which is what a round-trip check
// needs in order to dequantise what the DEVICE actually wrote rather than what it should
// have written.
inline int read_code(const Geometry& g, const std::uint8_t* packed, int page, int head,
                     int token_in_page, int channel)
{
    const std::uint64_t global_slot =
        static_cast<std::uint64_t>(page) * g.page_tokens + token_in_page;
    const std::uint64_t byte = tq::NzTiledChannelByte(
        global_slot, static_cast<std::uint64_t>(head), static_cast<std::uint64_t>(channel),
        static_cast<std::uint64_t>(g.num_kv_heads), static_cast<std::uint64_t>(g.head_size));
    const std::uint32_t half = tq::NzTiledChannelNibble(
        static_cast<std::uint64_t>(channel), static_cast<std::uint64_t>(g.head_size));
    const std::uint8_t raw = half == 0u ? static_cast<std::uint8_t>(packed[byte] & 0x0F)
                                       : static_cast<std::uint8_t>(packed[byte] >> 4);
    // Signed INT4 in [-8, 7].
    return raw >= 8u ? static_cast<int>(raw) - 16 : static_cast<int>(raw);
}

inline float read_scale(const Geometry& g, const float* scales, int plane, int page, int head,
                        int token_in_page)
{
    const std::uint64_t global_slot =
        static_cast<std::uint64_t>(page) * g.page_tokens + token_in_page;
    return scales[tq::NzTiledScaleIndex(global_slot, static_cast<std::uint64_t>(head),
                                        static_cast<std::uint64_t>(plane),
                                        static_cast<std::uint64_t>(g.num_kv_heads))];
}

}  // namespace tq_nz

#endif
