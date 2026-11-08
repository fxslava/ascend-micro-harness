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

// Can an ingress store be read back by the kv4fp8 fused decode?
// =============================================================
// There are two 4-bit KV cache layouts in this repo, and this is the gate that says which
// of them an ingress writer is speaking.
//
//   (A) STRIP -- RFC 3.1, the shipping contract of the paged decode's readers. Normative at
//       turboquant_reshape_scatter.cpp:55, host oracle tests/common/turboquant_scatter_oracle.hpp,
//       R = 16 the strip height:
//         key   nibble = ((block*H + head)*S + token)*d + channel
//         value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token % R)
//         scale index  = ((kv*P + block)*H + head)*S + token
//
//   (B) NZ-TILED -- what the kv4fp8 fused decode reads. kCubeTileRows-row tiles, kOperandC0-byte
//       column groups, and a nibble pairing that puts channels (c, c + d/2) of ONE token in a
//       byte. The ingress side of it is NzTiledChannelByte / NzTiledChannelNibble /
//       NzTiledScaleIndex in common/turboquant_layout.h.
//
// WHAT THIS TEST ASSERTS, AND WHY IT CHANGED
// ------------------------------------------
// It used to assert that (A) and (B) DISAGREE, because at the time no ingress writer produced
// (B) and the point was to prove no address permutation could bridge them. Both halves of that
// are still here and still true -- check 3 keeps the proof, and check 4 keeps the legacy
// contrast -- but they are no longer the headline. The headline is check 1: that the layout an
// ingress writer targets under TURBOQUANT_INGRESS_NZ names THE SAME BYTE AND THE SAME NIBBLE as
// the fused decode reads, for every logical element. That is the condition under which a real
// write -> read round trip exists, so this test is the gate on it.
//
// The comparison is not vacuous. The writer's address function is the production
// NzTiledChannelByte; the reader's is re-derived here, independently, by walking what the
// decode actually does to get from a GM byte to a logical (token, channel):
//
//   1. ReadPlane issues ONE tilePlaneBytes_ burst from
//      (row * numKvHeads + kvHead * kCubeTileRows) * packedBytes, so packed[i] = GM[tileBase + i].
//   2. UnpackAffine DeInterleaves it: nzLow[i] = low nibble of packed[i], nzHigh[i] = high nibble.
//   3. StageNzTileToL1 stages nzLow at L1 offset 0 and nzHigh at + tilePlaneBytes_, in ONE
//      contiguous burst (nzTileToL1Params_ is {1, kCubeTileRows * operandElems / kOperandC0, 0, 0}),
//      so L1[i] = nzLow[i] for i < tilePlaneBytes and nzHigh[i - tilePlaneBytes] above it.
//   4. L1 is the kCubeTileRows x d NZ image, where element (row n, col e) sits at
//      (e / kOperandC0) * kCubeTileRows * kOperandC0 + n * kOperandC0 + e % kOperandC0.
//
// Inverting 1..4 gives DecodeReadByte below. If the writer's arithmetic and that walk ever
// diverge, this test fails and the round trip is off.

#include "../common/turboquant_scatter_oracle.hpp"
#include "../../kernels/ops/turboquant/fused_decode/common/turboquant_layout.h"

#include <cstdint>
#include <cstdio>
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

struct Addr {
    std::size_t byte;
    int nibble;  // 0 = low, 1 = high
};

bool same(const Addr &a, const Addr &b) { return a.byte == b.byte && a.nibble == b.nibble; }

// ------------------------------------------------- (B) the ingress writer's address function
// The production contract, as the kernel issues it. K and V are symmetric, so one function
// serves both planes -- which is the whole reason the NZ ingress is simpler than the strip one.
Addr IngressNzByte(const Shape &s, int block, int head, int token, int channel)
{
    const std::uint64_t slot = static_cast<std::uint64_t>(block) * s.block_size + token;
    return Addr{tq::NzTiledChannelByte(slot, static_cast<std::uint64_t>(head),
                                       static_cast<std::uint64_t>(channel),
                                       static_cast<std::uint64_t>(s.num_kv_heads),
                                       static_cast<std::uint64_t>(s.head_size)),
                static_cast<int>(tq::NzTiledChannelNibble(static_cast<std::uint64_t>(channel),
                                                          static_cast<std::uint64_t>(s.head_size)))};
}

// ------------------------------------------------------- (B) the fused decode's reader, walked
// Re-derived from ReadPlane -> UnpackAffine -> StageNzTileToL1 -> the L1 NZ image, per the
// header comment. Deliberately written in the decode's own terms and NOT by calling the
// writer's helper, so that check 1 compares two independent derivations.
Addr DecodeReadByte(const Shape &s, int block, int head, int token, int channel)
{
    const std::uint64_t packed_bytes = static_cast<std::uint64_t>(s.head_size) / tq::kPackFactor;
    const std::uint64_t slot = static_cast<std::uint64_t>(block) * s.block_size + token;
    const std::uint64_t tile_row = slot % tq::kCubeTileRows;

    // Step 1: the tile's GM base, exactly as ReadPlane computes it.
    const std::uint64_t tile_base =
        ((slot - tile_row) * s.num_kv_heads + static_cast<std::uint64_t>(head) * tq::kCubeTileRows) *
        packed_bytes;

    // Step 4: where (tile_row, channel) sits in the L1 NZ image of the staged tile.
    const std::uint64_t group = static_cast<std::uint64_t>(channel) / tq::kOperandC0;
    const std::uint64_t l1 = group * tq::kCubeTileRows * tq::kOperandC0 +
                             tile_row * tq::kOperandC0 +
                             static_cast<std::uint64_t>(channel) % tq::kOperandC0;

    // Steps 3 and 2: below tilePlaneBytes the L1 element came from a LOW nibble, above it from
    // a HIGH one, and the packed index is the L1 index modulo the plane.
    const std::uint64_t tile_plane_bytes = tq::kCubeTileRows * packed_bytes;
    const int nibble = l1 < tile_plane_bytes ? 0 : 1;
    const std::uint64_t packed_index = l1 < tile_plane_bytes ? l1 : l1 - tile_plane_bytes;

    return Addr{tile_base + packed_index, nibble};
}

std::size_t DecodeReadScale(const Shape &s, int plane, int block, int head, int token)
{
    // scaleCacheGm_[row * scaleSlot_] with scaleSlot_ = ScaleSlotFloats(numKvHeads), and lane
    // kv * numKvHeads + head within the row's slot.
    const std::size_t slot_lanes = ((2u * static_cast<unsigned>(s.num_kv_heads) + tq::kFp32PerBlock - 1u) /
                                    tq::kFp32PerBlock) * tq::kFp32PerBlock;
    const std::size_t row = static_cast<std::size_t>(block) * s.block_size + token;
    return row * slot_lanes + static_cast<std::size_t>(plane) * s.num_kv_heads + head;
}

// ------------------------------------------------------------- (A) the legacy strip contract
Addr StripByte(const Shape &s, int plane, int block, int head, int token, int channel)
{
    const std::size_t nibble = plane == 0
                                   ? tq_scatter::key_nibble(s, block, head, token, channel)
                                   : tq_scatter::value_nibble(s, block, head, token, channel);
    return Addr{nibble / 2, static_cast<int>(nibble % 2)};
}

// --------------------------------------------------------------------------------- the checks

// Check 1. THE GATE. The ingress writer's address function and the fused decode's reader must
// name the same byte and the same nibble for every logical element, on both planes, and the
// same scale float. Zero tolerance: one disagreement is a silent-corruption path.
void check_agreement(const Shape &s)
{
    std::size_t total = 0, agree = 0, bad_byte = 0, bad_nibble = 0;
    int fb = -1, fh = -1, ft = -1, fc = -1;
    Addr fw{0, 0}, fr{0, 0};
    for (int block = 0; block < s.num_blocks; ++block) {
        for (int head = 0; head < s.num_kv_heads; ++head) {
            for (int token = 0; token < s.block_size; ++token) {
                for (int channel = 0; channel < s.head_size; ++channel) {
                    const Addr w = IngressNzByte(s, block, head, token, channel);
                    const Addr r = DecodeReadByte(s, block, head, token, channel);
                    ++total;
                    if (same(w, r)) {
                        ++agree;
                        continue;
                    }
                    if (w.byte != r.byte) ++bad_byte;
                    if (w.nibble != r.nibble) ++bad_nibble;
                    if (fb < 0) {
                        fb = block; fh = head; ft = token; fc = channel;
                        fw = w; fr = r;
                    }
                }
            }
        }
    }
    std::printf("   packed planes: %zu of %zu logical elements agree "
                "(wrong byte %zu, wrong nibble %zu)\n",
                agree, total, bad_byte, bad_nibble);
    if (fb >= 0) {
        std::printf("     first disagreement (block %d, head %d, token %d, channel %d): "
                    "writer byte %zu nibble %d, decode byte %zu nibble %d\n",
                    fb, fh, ft, fc, fw.byte, fw.nibble, fr.byte, fr.nibble);
    }
    check(agree == total, "the ingress writer and the fused decode name the same byte and nibble");

    std::size_t s_total = 0, s_agree = 0;
    int sp = -1, sb = -1, sh = -1, st = -1;
    std::size_t sw = 0, sr = 0;
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane) {
        for (int block = 0; block < s.num_blocks; ++block) {
            for (int head = 0; head < s.num_kv_heads; ++head) {
                for (int token = 0; token < s.block_size; ++token) {
                    const std::size_t w = tq::NzTiledScaleIndex(
                        static_cast<std::uint64_t>(block) * s.block_size + token,
                        static_cast<std::uint64_t>(head), static_cast<std::uint64_t>(plane),
                        static_cast<std::uint64_t>(s.num_kv_heads));
                    const std::size_t r = DecodeReadScale(s, plane, block, head, token);
                    ++s_total;
                    if (w == r) {
                        ++s_agree;
                    } else if (sp < 0) {
                        sp = plane; sb = block; sh = head; st = token; sw = w; sr = r;
                    }
                }
            }
        }
    }
    std::printf("   scale plane:   %zu of %zu indices agree\n", s_agree, s_total);
    if (sp >= 0) {
        std::printf("     first disagreement (kv %d, block %d, head %d, token %d): "
                    "writer %zu, decode %zu\n", sp, sb, sh, st, sw, sr);
    }
    check(s_agree == s_total, "the ingress writer and the fused decode name the same scale float");
}

// Check 2. The writer's map is a bijection onto the cache: every nibble written exactly once,
// nothing outside the allocation. Without this, check 1 could pass on a map that collapses
// several logical elements onto one nibble.
void check_bijective(const Shape &s)
{
    const std::size_t bytes = s.cache_bytes();
    std::vector<std::uint8_t> hits(bytes * 2, 0);
    std::size_t out = 0, dup = 0;
    for (int block = 0; block < s.num_blocks; ++block) {
        for (int head = 0; head < s.num_kv_heads; ++head) {
            for (int token = 0; token < s.block_size; ++token) {
                for (int channel = 0; channel < s.head_size; ++channel) {
                    const Addr a = IngressNzByte(s, block, head, token, channel);
                    if (a.byte >= bytes) {
                        ++out;
                    } else if (hits[a.byte * 2 + static_cast<std::size_t>(a.nibble)]++) {
                        ++dup;
                    }
                }
            }
        }
    }
    std::size_t miss = 0;
    for (std::size_t i = 0; i < bytes * 2; ++i) {
        if (!hits[i]) ++miss;
    }
    std::printf("   out-of-cache %zu, collisions %zu, unreached nibbles %zu\n", out, dup, miss);
    check(out == 0 && dup == 0 && miss == 0,
          "the NZ ingress map is a bijection onto the whole cache");
}

// Check 3. RETAINED PROOF, now a regression guard rather than the headline: why the writer had
// to change layout at all, instead of the cache being re-addressed. A byte holds two logical
// elements, and since both maps are bijections "shares a byte with" is a perfect matching on
// the (token, channel) grid. Any injective map of bytes carries a co-byte pair onto a co-byte
// pair, so two matchings with no pair in common cannot be related by ANY relabelling of
// addresses -- the bridge has to re-pack nibbles. The legacy strip layout and the decode's are
// exactly such a pair, and the value plane does not even pair along the same axis:
//
//   strip key   pairs (t, 2k)  with (t, 2k+1)      channel axis, distance 1
//   strip value pairs (2j, c)  with (2j+1, c)      TOKEN axis,   distance 1
//   NZ-tiled    pairs (t, c)   with (t, c + d/2)   channel axis, distance d/2
struct Matching {
    std::vector<int> partner;
    std::size_t pairs = 0, channel_axis = 0, token_axis = 0;
    int ea = -1, eb = -1;
};

Matching build_matching(const Shape &s, bool nz, int plane, int block, int head)
{
    const std::size_t bytes = s.cache_bytes();
    Matching m;
    m.partner.assign(static_cast<std::size_t>(s.block_size) * s.head_size, -1);
    std::vector<int> seen(bytes * 2, -1);
    for (int token = 0; token < s.block_size; ++token) {
        for (int channel = 0; channel < s.head_size; ++channel) {
            const Addr a = nz ? IngressNzByte(s, block, head, token, channel)
                              : StripByte(s, plane, block, head, token, channel);
            const int element = token * s.head_size + channel;
            const std::size_t other = a.byte * 2 + static_cast<std::size_t>(1 - a.nibble);
            seen[a.byte * 2 + static_cast<std::size_t>(a.nibble)] = element;
            if (seen[other] >= 0) {
                const int mate = seen[other];
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
        if (m.ea < 0) { m.ea = static_cast<int>(e); m.eb = mate; }
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

void check_legacy_contrast(const Shape &s, int plane)
{
    const Matching strip = build_matching(s, false, plane, 0, 0);
    const Matching nz = build_matching(s, true, plane, 0, 0);
    const std::size_t expect = static_cast<std::size_t>(s.block_size) * s.head_size / 2;
    std::printf("   plane %d (%s), page (block 0, head 0):\n", plane, plane == 0 ? "key" : "value");
    describe(s, "legacy strip", strip);
    describe(s, "NZ ingress", nz);
    std::size_t common = 0;
    for (std::size_t e = 0; e < strip.partner.size(); ++e) {
        if (strip.partner[e] >= 0 && strip.partner[e] == nz.partner[e]) ++common;
    }
    common /= 2;
    std::printf("     co-byte pairs in common: %zu of %zu\n", common, expect);
    check(nz.pairs == expect && nz.channel_axis == expect && nz.token_axis == 0,
          "the NZ ingress pairs two channels of one token in a byte");
    if (plane == 0) {
        check(strip.channel_axis == expect && strip.token_axis == 0,
              "the legacy key plane pairs two channels of one token in a byte");
    } else {
        check(strip.token_axis == expect && strip.channel_axis == 0,
              "the legacy value plane pairs two tokens of one channel in a byte");
    }
    check(common == 0,
          "the legacy strip layout is NOT reachable from the NZ one by any byte permutation");
}

// Check 5. The scale slot width this header defines must be the number the kernels compute.
// NzScaleSlotFloats is the one definition; ScaleSlotFloats in common/turboquant_common.h is
// __aicore__ and cannot be included here, so its formula is re-evaluated and compared.
void check_scale_slot_formula()
{
    for (unsigned heads = 1; heads <= 8; ++heads) {
        const unsigned ceil_div = (2u * heads + tq::kFp32PerBlock - 1u) / tq::kFp32PerBlock;
        const unsigned kernel = ceil_div * tq::kFp32PerBlock;
        const unsigned shared = tq::NzScaleSlotFloats(heads);
        if (heads <= 4) {
            std::printf("     heads %u: ScaleSlotFloats %u, NzScaleSlotFloats %u\n", heads, kernel,
                        shared);
        }
        check(kernel == shared, "NzScaleSlotFloats matches the kernels' ScaleSlotFloats");
    }
}

void run(int num_blocks, int num_kv_heads)
{
    Shape s;
    s.num_requests = 1;
    s.num_kv_heads = num_kv_heads;
    s.head_size = 256;
    s.block_size = 64;  // what the fused decode's valid_launch admits, and what the task asks for
    s.num_blocks = num_blocks;
    check(s.valid(), "the geometry is a valid ingress shape");
    check(s.block_size % static_cast<int>(tq::kCubeTileRows) == 0,
          "block_size is a whole number of Cube tiles");

    std::printf("\n== geometry: %d page(s), %d kv head(s), block_size %d, head_size %d, "
                "%zu cache bytes\n",
                s.num_blocks, s.num_kv_heads, s.block_size, s.head_size, s.cache_bytes());

    std::printf("\n1. THE GATE -- NZ ingress writer vs fused decode reader, same logical element\n");
    check_agreement(s);

    std::printf("\n2. the NZ ingress map is a bijection onto the cache\n");
    check_bijective(s);

    std::printf("\n3. why the layout had to change: the legacy strip contract, for contrast\n");
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane) {
        check_legacy_contrast(s, plane);
    }
}

}  // namespace

int main()
{
    std::printf("turboquant KV layout bridge: NZ ingress contract vs the kv4fp8 fused decode\n");
    std::printf("  strip height R %d, Cube tile rows %u, operand C0 %u, pack factor %u, "
                "fp32 per block %u\n",
                tq_scatter::kStripHeight, tq::kCubeTileRows, tq::kOperandC0, tq::kPackFactor,
                tq::kFp32PerBlock);
    std::printf("\n0. the scale slot width is one definition\n");
    check_scale_slot_formula();
    run(1, 1);
    run(2, 1);
    run(2, 4);
    run(3, 2);
    if (g_failures) {
        std::printf("\nturboquant KV layout bridge: FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("\nturboquant KV layout bridge: PASS\n");
    std::printf("  The NZ ingress contract and the fused decode agree on every byte, every\n");
    std::printf("  nibble and every scale float, so a write -> read round trip EXISTS for a\n");
    std::printf("  writer that targets it. The legacy strip contract remains unreachable from\n");
    std::printf("  it by any address permutation (check 3), which is why the writer needs a\n");
    std::printf("  layout mode rather than an offset fix.\n");
    return 0;
}
