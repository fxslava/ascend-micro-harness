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

// Is an ingress-write -> fused-decode-read round trip EXPRESSIBLE?
// =================================================================
// There are two 4-bit KV cache layouts in this repo and they are not the same bytes.
//
//   (A) STRIP, written by the ingress operators (turboquant_reshape_scatter.cpp,
//       turboquant_reshape_batch.cpp, turboquant_reshape_and_cache_bf16.cpp). Contract
//       RFC 3.1, normative at turboquant_reshape_scatter.cpp:55, host oracle
//       tests/common/turboquant_scatter_oracle.hpp. R = 16 the strip height:
//
//         key   nibble = ((block*H + head)*S + token)*d + channel
//         value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
//         scale index  = ((kv*P + block)*H + head)*S + token
//
//   (B) NZ-TILED, read by the kv4fp8 fused decode. Address function
//       vllm_ascend::turboquant::NzTiledPackedByte (common/turboquant_layout.h): 64-row
//       tiles, 32-byte column groups. Its nibble pairing is NOT the contract's: the
//       byte-wise affine expand (UnpackAffine in common/turboquant_codec_mx.h) DeInterleaves
//       a chunk into a low plane and a high plane, and StageNzTileToL1
//       (vector/turboquant_vector_service.h:1140) stages the low plane at L1 offset 0 and
//       the high plane at + tilePlaneBytes_. The L1 operand is the 64 x d NZ image, so the
//       low plane lands on NZ groups [0, d/2/kOperandC0) = channels [0, d/2) and the high
//       plane on channels [d/2, d). So the byte holding packed column `col` carries channel
//       `col` in its low nibble and channel `col + d/2` in its high nibble, of ONE token.
//       The scale plane is per-row slotted:
//       (physical*S + token)*ScaleSlotFloats(H) + kv*H + head.
//
// This probe maps every (block, kv_head, token, channel) of a 1-2 page / block_size 64
// geometry through BOTH address functions and reports whether they name the same byte and
// the same nibble. It calls the production arithmetic -- the oracle's key_nibble and
// value_nibble, and the kernels' NzTiledPackedByte -- rather than re-typing the formulas.
//
// It also answers the question a raw disagreement count cannot: could ANY relabelling of
// bytes bridge the two? No. A byte holds two logical elements, and WHICH two differs:
//
//   (A) key   pairs (token, 2k) with (token, 2k+1)        -- channel axis, distance 1
//   (A) value pairs (2j, channel) with (2j+1, channel)    -- TOKEN axis, distance 1
//   (B) both  pair (token, c) with (token, c + d/2)       -- channel axis, distance d/2
//
// An injective map of bytes carries each co-byte pair onto a co-byte pair, so if the two
// matchings share no pair then no relabelling of addresses can bridge the layouts: a bridge
// is necessarily a nibble REPACK. Check 3 proves that by building both matchings and
// intersecting them. Note the value plane is not merely permuted but pairs along a different
// axis of the (token, channel) grid than the fused decode expects.
//
// Model: scripts/kv_plane_layout_check.py, which does this comparison for the strip
// writer/reader pair (and finds 0 disagreements there, because both of those are layout A).

#include "../common/turboquant_scatter_oracle.hpp"
#include "../../kernels/ops/turboquant/fused_decode/common/turboquant_layout.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <utility>
#include <vector>

namespace {

namespace tq = vllm_ascend::turboquant;
using tq_scatter::Shape;

int g_failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        std::printf("  FAIL %s\n", what);
        ++g_failures;
    }
}

// ScaleSlotFloats(numKvHeads) from common/turboquant_common.h, which is __aicore__ and so
// cannot be included on the host. Re-derived from the same layout constant it uses.
std::size_t scale_slot_floats(int num_kv_heads)
{
    const std::size_t lanes = 2u * static_cast<std::size_t>(num_kv_heads);
    const std::size_t block = tq::kFp32PerBlock;
    return ((lanes + block - 1) / block) * block;
}

// ---------------------------------------------------------------- the two address functions

struct Addr {
    std::size_t byte;
    int nibble;  // 0 = low, 1 = high
};

bool same(const Addr &a, const Addr &b) { return a.byte == b.byte && a.nibble == b.nibble; }

// (A) STRIP, as the ingress writers write it: the production host oracle's nibble index,
// split into (byte, nibble) by the contract's "two per byte, low nibble first".
Addr strip_addr(const Shape &s, int plane, int block, int head, int token, int channel)
{
    const std::size_t nibble = plane == 0
                                   ? tq_scatter::key_nibble(s, block, head, token, channel)
                                   : tq_scatter::value_nibble(s, block, head, token, channel);
    return Addr{nibble / 2, static_cast<int>(nibble % 2)};
}

// (B) NZ-TILED, as the fused decode reads it. K and V are symmetric here (ReadPlane branches
// only on which GM tensor), so one function serves both planes.
Addr nz_addr(const Shape &s, int block, int head, int token, int channel)
{
    const std::size_t packed_bytes = static_cast<std::size_t>(s.head_size) / tq::kPackFactor;
    const std::size_t column = static_cast<std::size_t>(channel) % packed_bytes;
    const int nibble = static_cast<std::size_t>(channel) < packed_bytes ? 0 : 1;
    const std::size_t page_base = static_cast<std::size_t>(block) * s.block_size *
                                  static_cast<std::size_t>(s.num_kv_heads) * packed_bytes;
    const std::size_t in_page = tq::NzTiledPackedByte(static_cast<std::uint64_t>(token),
                                                      static_cast<std::uint64_t>(head), column,
                                                      static_cast<std::uint64_t>(s.num_kv_heads),
                                                      packed_bytes);
    return Addr{page_base + in_page, nibble};
}

// -------------------------------------------------------------------------------- the checks

struct Tally {
    std::size_t total = 0;
    std::size_t same_byte_same_nibble = 0;
    std::size_t same_byte_other_nibble = 0;
    std::size_t other_byte = 0;
    Addr first_strip{0, 0};
    Addr first_nz{0, 0};
    int fb = -1, fh = -1, ft = -1, fc = -1;
};

// Check 1. Do the ingress writer and the fused decode name the same byte and nibble for the
// same logical (block, kv_head, token, channel)? Every element of the whole cache, both planes.
Tally compare_plane(const Shape &s, int plane)
{
    Tally t;
    for (int block = 0; block < s.num_blocks; ++block) {
        for (int head = 0; head < s.num_kv_heads; ++head) {
            for (int token = 0; token < s.block_size; ++token) {
                for (int channel = 0; channel < s.head_size; ++channel) {
                    const Addr a = strip_addr(s, plane, block, head, token, channel);
                    const Addr b = nz_addr(s, block, head, token, channel);
                    ++t.total;
                    if (same(a, b)) {
                        ++t.same_byte_same_nibble;
                        continue;
                    }
                    if (a.byte == b.byte) {
                        ++t.same_byte_other_nibble;
                    } else {
                        ++t.other_byte;
                    }
                    if (t.fb < 0) {
                        t.fb = block;
                        t.fh = head;
                        t.ft = token;
                        t.fc = channel;
                        t.first_strip = a;
                        t.first_nz = b;
                    }
                }
            }
        }
    }
    return t;
}

// Check 2. Both maps are bijections onto the same byte range, so the disagreement is a
// permutation-and-repack question and not a capacity mismatch. (A map that landed outside the
// page, or collided, would be a different and much simpler finding.)
void check_both_bijective(const Shape &s, int plane)
{
    const std::size_t bytes = s.cache_bytes();
    std::vector<std::uint8_t> hits_strip(bytes * 2, 0);
    std::vector<std::uint8_t> hits_nz(bytes * 2, 0);
    std::size_t strip_out = 0, nz_out = 0, strip_dup = 0, nz_dup = 0;
    for (int block = 0; block < s.num_blocks; ++block) {
        for (int head = 0; head < s.num_kv_heads; ++head) {
            for (int token = 0; token < s.block_size; ++token) {
                for (int channel = 0; channel < s.head_size; ++channel) {
                    const Addr a = strip_addr(s, plane, block, head, token, channel);
                    const Addr b = nz_addr(s, block, head, token, channel);
                    if (a.byte >= bytes) {
                        ++strip_out;
                    } else if (hits_strip[a.byte * 2 + static_cast<std::size_t>(a.nibble)]++) {
                        ++strip_dup;
                    }
                    if (b.byte >= bytes) {
                        ++nz_out;
                    } else if (hits_nz[b.byte * 2 + static_cast<std::size_t>(b.nibble)]++) {
                        ++nz_dup;
                    }
                }
            }
        }
    }
    std::size_t strip_miss = 0, nz_miss = 0;
    for (std::size_t i = 0; i < bytes * 2; ++i) {
        if (!hits_strip[i]) ++strip_miss;
        if (!hits_nz[i]) ++nz_miss;
    }
    std::printf("   plane %d: strip out-of-cache %zu, collisions %zu, unreached nibbles %zu\n",
                plane, strip_out, strip_dup, strip_miss);
    std::printf("   plane %d: NZ    out-of-cache %zu, collisions %zu, unreached nibbles %zu\n",
                plane, nz_out, nz_dup, nz_miss);
    check(strip_out == 0 && strip_dup == 0 && strip_miss == 0,
          "the strip map is a bijection onto the cache");
    check(nz_out == 0 && nz_dup == 0 && nz_miss == 0,
          "the NZ-tiled map is a bijection onto the cache");
}

// Check 3. The structural obstruction, and the reason a layout bridge cannot be an address
// permutation. A byte holds two logical elements. Build, for one (block, kv_head) page and
// each layout, the "shares a byte with" pairing of the (token, channel) grid -- a perfect
// matching, since both maps are bijections (check 2). Any injective map of bytes carries one
// layout's byte onto another byte, hence carries each co-byte PAIR onto a co-byte pair; so if
// the two matchings share no pair, no relabelling of addresses can bridge the layouts and a
// bridge must re-pack nibbles.
//
// The two matchings run along different axes, which is the sharpest form of the finding:
//   strip key   pairs (t, 2k) with (t, 2k+1)       -- channel axis, distance 1
//   strip value pairs (2j, c) with (2j+1, c)       -- TOKEN axis, distance 1
//   NZ-tiled    pairs (t, c)  with (t, c + d/2)    -- channel axis, distance d/2
// so the strip value plane does not even pair along the same axis as the plane the fused
// decode expects.
struct Matching {
    // element index = token * head_size + channel, partner or -1.
    std::vector<int> partner;
    std::size_t pairs = 0;
    std::size_t channel_axis = 0;  // pairs sharing a token
    std::size_t token_axis = 0;    // pairs sharing a channel
    int ea = -1, eb = -1;          // one example pair, as element indices
};

Matching build_matching(const Shape &s, bool nz, int plane, int block, int head)
{
    const std::size_t bytes = s.cache_bytes();
    Matching m;
    m.partner.assign(static_cast<std::size_t>(s.block_size) * s.head_size, -1);
    // byte -> the one element already seen there, per nibble.
    std::vector<int> seen(bytes * 2, -1);
    for (int token = 0; token < s.block_size; ++token) {
        for (int channel = 0; channel < s.head_size; ++channel) {
            const Addr a = nz ? nz_addr(s, block, head, token, channel)
                              : strip_addr(s, plane, block, head, token, channel);
            const int element = token * s.head_size + channel;
            // The two nibbles of byte a.byte: this element's partner is whatever sits in the
            // other nibble of the same byte.
            const std::size_t other_slot = a.byte * 2 + static_cast<std::size_t>(1 - a.nibble);
            seen[a.byte * 2 + static_cast<std::size_t>(a.nibble)] = element;
            if (seen[other_slot] >= 0) {
                const int mate = seen[other_slot];
                m.partner[static_cast<std::size_t>(element)] = mate;
                m.partner[static_cast<std::size_t>(mate)] = element;
            }
        }
    }
    for (std::size_t e = 0; e < m.partner.size(); ++e) {
        const int mate = m.partner[e];
        if (mate < 0 || static_cast<std::size_t>(mate) < e) continue;
        ++m.pairs;
        const int ta = static_cast<int>(e) / s.head_size, ca = static_cast<int>(e) % s.head_size;
        const int tb = mate / s.head_size, cb = mate % s.head_size;
        if (ta == tb) ++m.channel_axis;
        if (ca == cb) ++m.token_axis;
        if (m.ea < 0) {
            m.ea = static_cast<int>(e);
            m.eb = mate;
        }
    }
    return m;
}

void describe(const Shape &s, const char *name, const Matching &m)
{
    std::printf("     %-12s pairs %5zu  (same token, two channels %5zu | same channel, two "
                "tokens %5zu)",
                name, m.pairs, m.channel_axis, m.token_axis);
    if (m.ea >= 0) {
        std::printf("  e.g. (t %d, c %d)+(t %d, c %d)", m.ea / s.head_size, m.ea % s.head_size,
                    m.eb / s.head_size, m.eb % s.head_size);
    }
    std::printf("\n");
}

void check_byte_partition(const Shape &s, int plane)
{
    const int block = 0, head = 0;
    const Matching strip = build_matching(s, false, plane, block, head);
    const Matching nz = build_matching(s, true, plane, block, head);
    const std::size_t expect = static_cast<std::size_t>(s.block_size) * s.head_size / 2;
    std::printf("   plane %d (%s), page (block 0, head 0):\n", plane,
                plane == 0 ? "key" : "value");
    describe(s, "ingress", strip);
    describe(s, "fused decode", nz);
    std::size_t common = 0;
    for (std::size_t e = 0; e < strip.partner.size(); ++e) {
        if (strip.partner[e] >= 0 && strip.partner[e] == nz.partner[e]) ++common;
    }
    common /= 2;
    std::printf("     co-byte pairs the two layouts have in common: %zu of %zu\n", common, expect);
    check(strip.pairs == expect, "the strip layout's co-byte relation is a perfect matching");
    check(nz.pairs == expect, "the NZ layout's co-byte relation is a perfect matching");
    check(nz.channel_axis == expect && nz.token_axis == 0,
          "the fused decode pairs two channels of one token in a byte");
    if (plane == 0) {
        check(strip.channel_axis == expect && strip.token_axis == 0,
              "the ingress key plane pairs two channels of one token in a byte");
    } else {
        // Not a weaker statement than the key plane's -- a stronger one. The strip value plane
        // is channel-major inside a 16-token strip, so a byte holds ONE channel of TWO tokens.
        check(strip.token_axis == expect && strip.channel_axis == 0,
              "the ingress value plane pairs two tokens of one channel in a byte");
    }
    check(common == 0,
          "the two co-byte matchings are disjoint, so no byte permutation bridges the layouts");
}

// Check 4. The scale plane diverges too, and separately: the ingress writer's is planar over
// the whole cache, the fused decode's is a per-row slot of ScaleSlotFloats(H) lanes.
void check_scales(const Shape &s)
{
    const std::size_t slot = scale_slot_floats(s.num_kv_heads);
    std::size_t total = 0, agree = 0, fi_strip = 0, fi_nz = 0;
    bool first = true;
    int fkv = -1, fb = -1, fh = -1, ft = -1;
    for (int kv = 0; kv < tq_scatter::kPlanes; ++kv) {
        for (int block = 0; block < s.num_blocks; ++block) {
            for (int head = 0; head < s.num_kv_heads; ++head) {
                for (int token = 0; token < s.block_size; ++token) {
                    const std::size_t a = tq_scatter::scale_index(s, kv, block, head, token);
                    const std::size_t b =
                        (static_cast<std::size_t>(block) * s.block_size + token) * slot +
                        static_cast<std::size_t>(kv) * s.num_kv_heads + head;
                    ++total;
                    if (a == b) {
                        ++agree;
                    } else if (first) {
                        first = false;
                        fkv = kv;
                        fb = block;
                        fh = head;
                        ft = token;
                        fi_strip = a;
                        fi_nz = b;
                    }
                }
            }
        }
    }
    const std::size_t ingress_floats = static_cast<std::size_t>(tq_scatter::kPlanes) *
                                       s.num_blocks * s.num_kv_heads * s.block_size;
    const std::size_t decode_floats = static_cast<std::size_t>(s.num_blocks) * s.block_size * slot;
    std::printf("   scale plane: %zu/%zu indices agree; slot lanes %zu\n", agree, total, slot);
    std::printf("     ingress floats %zu (planar [kv][P][H][S]); fused decode floats %zu "
                "([P*S][slot])\n",
                ingress_floats, decode_floats);
    if (!first) {
        std::printf("     first divergence kv %d block %d head %d token %d: ingress %zu, "
                    "decode %zu\n",
                    fkv, fb, fh, ft, fi_strip, fi_nz);
    }
}

void run(int num_blocks, int num_kv_heads)
{
    Shape s;
    s.num_requests = 1;
    s.num_kv_heads = num_kv_heads;
    s.head_size = 256;
    s.block_size = 64;  // what the task asks for, and what the fused decode's valid_launch admits
    s.num_blocks = num_blocks;
    check(s.valid(), "the geometry is a valid ingress shape");
    // The fused decode is built on 64-row Cube tiles, so block_size must be a whole number of them.
    check(s.block_size % static_cast<int>(tq::kCubeTileRows) == 0,
          "block_size is a whole number of Cube tiles");

    std::printf("\n== geometry: %d page(s), %d kv head(s), block_size %d, head_size %d, "
                "%zu cache bytes\n",
                s.num_blocks, s.num_kv_heads, s.block_size, s.head_size, s.cache_bytes());

    std::printf("\n1. ingress writer (STRIP) vs fused decode (NZ-TILED), same logical element\n");
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane) {
        const Tally t = compare_plane(s, plane);
        std::printf("   %s plane: %zu elements; same byte+nibble %zu, same byte other nibble %zu, "
                    "different byte %zu\n",
                    plane == 0 ? "key  " : "value", t.total, t.same_byte_same_nibble,
                    t.same_byte_other_nibble, t.other_byte);
        if (t.fb >= 0) {
            std::printf("     first divergence (block %d, head %d, token %d, channel %d): "
                        "ingress byte %zu nibble %d, decode byte %zu nibble %d\n",
                        t.fb, t.fh, t.ft, t.fc, t.first_strip.byte, t.first_strip.nibble,
                        t.first_nz.byte, t.first_nz.nibble);
        }
        // The finding, asserted rather than printed: they do NOT agree. If this ever stops
        // holding, an ingress -> fused-decode round trip became expressible and the branch
        // this probe governs has to be re-taken.
        check(t.same_byte_same_nibble < t.total,
              "the two layouts do not agree everywhere (the round trip is not expressible)");
    }

    std::printf("\n2. both maps are bijections onto the same byte range\n");
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane) {
        check_both_bijective(s, plane);
    }

    std::printf("\n3. co-byte matchings -- can any byte permutation bridge them?\n");
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane) {
        check_byte_partition(s, plane);
    }

    std::printf("\n4. the scale plane\n");
    check_scales(s);
}

}  // namespace

int main()
{
    std::printf("turboquant KV layout bridge probe: ingress STRIP vs fused-decode NZ-TILED\n");
    std::printf("  strip height R %d, Cube tile rows %u, operand C0 %u, pack factor %u\n",
                tq_scatter::kStripHeight, tq::kCubeTileRows, tq::kOperandC0, tq::kPackFactor);
    run(1, 1);
    run(2, 1);
    run(2, 4);
    if (g_failures) {
        std::printf("\nturboquant KV layout bridge probe: FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("\nturboquant KV layout bridge probe: PASS\n");
    std::printf("  CONCLUSION: the ingress writers and the fused decode do NOT address the same\n");
    std::printf("  bytes, and the obstruction is structural (check 3): a bridge must repack\n");
    std::printf("  nibbles, not permute addresses. An ingress-write -> fused-decode-read round\n");
    std::printf("  trip is NOT expressible with the operators as they stand.\n");
    return 0;
}
