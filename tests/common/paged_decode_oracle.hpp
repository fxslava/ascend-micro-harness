#pragma once

// CPU golden oracle, layout contract and fixture for the Ascend 950PR fused
// paged decode attention kernel.
//
//   batch / query tokens  M = 16
//   head dimension        d = 256
//   vLLM page             128 tokens = 8 strips of 16 tokens
//   KV heads              4
//   temporal blocks       8   (one 128-token page each -> 1024-token context)
//   grid                  4 * 8 = 32 tasks, one per physical AIC
//
// This header is the single definition of
//
//   * the randomised Hadamard rotation and its resident 32x32 block form,
//   * the int4 KV cache packing and the planar scale cache,
//   * the arithmetic the device is required to reproduce,
//   * the fixture, and the properties the fixture must have for the hazard
//     checks to mean anything (guidelines section 6.4).
//
// Nothing here is device code. `kernels/ops/turboquant/turboquant_paged_decode
// _attention.cpp` restates the geometry constants in its own namespace, the
// way every other operator in this repository does, because a device kernel
// cannot include a test header.
//
// That duplication is checked BEHAVIOURALLY, not by a static_assert the two
// translation units cannot share. Every constant that could disagree is one
// the kernel turns into a Global Memory address: `kNumBlocks` here is the
// scale cache's page count, and the kernel's `kNumPages` is the stride between
// its two scale planes. If they disagreed the kernel would read the wrong
// plane, and under the canary protocol the wrong plane is `0x7FC0BEEF` -- a
// quiet NaN -- so the bit-exact score panel check fails immediately rather
// than subtly. The same holds for the strip, head and page strides. This is
// why the canary fill covers the inputs and not only the outputs.
//
// Exactness, and where it stops
// -----------------------------
// Two quantities are checked bit-exactly and the rest in ULP, which is a
// property of the arithmetic rather than a choice:
//
//   Q_rot      64 Mmads of bf16 operands into an FP32 accumulator. Every
//              table entry is +-1/16 and the fixture keeps the integer
//              numerator of each result below 2^8, so the FP32 sum is exact
//              and its bf16 image is exact. Asserted, not assumed:
//              `ValidateFixture` recomputes it in double and rejects a
//              fixture that would round.
//   R[q][t]    the raw score panel, one Mmad over k = 256 of exact small
//              integers against Q_rot. Exact multiple of 1/16.
//
//   m, l, Acc, the egress rotation and the gated context all pass through
//              `Exp` (<= 1 ULP per call, documented) and a cancelling
//              reduction, so they are bounded in ULP against the largest
//              intermediate per guidelines section 5, never bit-compared.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../include/sim_micro_testbed.hpp"
#include "../../routines/routine_stage_page_zn_bf16.hpp"

namespace paged_decode {

// ------------------------------------------------------------------ geometry
constexpr int kTokens = 16;      // M: query tokens, and tokens per strip
constexpr int kHeadSize = 256;   // d
constexpr int kGroup = 32;       // Hadamard group == Cube n tile of the rotation
constexpr int kGroups = kHeadSize / kGroup;           // 8
constexpr int kStrips = 8;                            // strips per page
constexpr int kBlockSize = kStrips * kTokens;         // 128 tokens per page
constexpr int kKvHeads = 4;
constexpr int kTemporalBlocks = 8;
constexpr int kGrid = kKvHeads * kTemporalBlocks;     // 32
constexpr int kPanel = kTokens * kHeadSize;           // 4096 elements per strip
constexpr int kPackedBytes = kPanel / 2;              // 2048 bytes of nibbles
constexpr int kScores = kTokens * kTokens;            // 256 entries per panel
constexpr int kTableBlocks = 2 * kGroups;             // 16 resident 32x32 blocks
constexpr int kTableElems = kTableBlocks * kGroup * kGroup;  // 16384

// Number of physical pages in the fixture's cache. One more than the context
// needs, so a block table that points away from the identity mapping is
// distinguishable from one that does not.
constexpr int kNumBlocks = kTemporalBlocks + 1;

static_assert(kHeadSize == kGroups * kGroup, "the groups must tile the head");
static_assert(kPackedBytes % 32 == 0, "a strip burst must be whole 32-byte blocks");

// --------------------------------------------------------------- bit helpers
using ascend_micro::sim::bits_float;
using ascend_micro::sim::float_bits;
using ascend_micro::sim::float_to_bfloat16_rne;
using ascend_micro::sim::require;

inline float bfloat16_to_float(std::uint16_t raw)
{
    return bits_float(static_cast<std::uint32_t>(raw) << 16);
}

inline bool bfloat16_is_exact(double value)
{
    const float as_float = static_cast<float>(value);
    if (static_cast<double>(as_float) != value) return false;
    return bfloat16_to_float(float_to_bfloat16_rne(as_float)) == as_float;
}

inline int Parity(unsigned value)
{
    int parity = 0;
    while (value != 0u) {
        parity ^= static_cast<int>(value & 1u);
        value >>= 1;
    }
    return parity;
}

// H_n[a][b] = (-1)^popcount(a & b), the canonical Sylvester sign.
inline int HadamardSign(int a, int b)
{
    return Parity(static_cast<unsigned>(a & b)) ? -1 : +1;
}

// ------------------------------------------------------- the rotation Pi
// Per head of kHeadSize channels the transform is the randomised Hadamard
//
//   Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / sqrt(kHeadSize)
//
// which with ch = c1 * kGroup + c0 and in = g * kGroup + i has entries
//
//   Pi[c1*32 + c0][g*32 + i]
//       = H8[c1][g] . piSigns[g*32 + c0] . H32[c0][i] / 16              (1)
//
// Pi Pi^T = I exactly: the 1/16 is folded in here rather than applied later,
// so every entry is +-1/16, which is exact in bf16 and in FP32.
//
// This is the same transform `turboquant_reshape_and_cache_a8.cpp` applies
// when it writes the KV cache, which is why the decode path can rotate Q with
// it and recover the true context by rotating the accumulator back.
constexpr float kInvSqrtLen = 1.0f / 16.0f;  // 1 / sqrt(256)

// The sign pattern. Deterministic so the host fixture, the host oracle and the
// device's resident table are built from one function; it is the only place
// the "randomised" part of the rotation is defined.
inline int PiSign(int channel)
{
    const unsigned x = static_cast<unsigned>(channel);
    return ((x * 2654435761u) >> 13) & 1u ? -1 : +1;
}

// Entry (1), as a float. |Pi[c][i]| == 1/16 for every entry.
inline float PiEntry(int out_channel, int in_channel)
{
    const int c1 = out_channel / kGroup;
    const int c0 = out_channel % kGroup;
    const int g = in_channel / kGroup;
    const int i = in_channel % kGroup;
    const int sign = HadamardSign(c1, g) * PiSign(g * kGroup + c0) *
                     HadamardSign(c0, i);
    return static_cast<float>(sign) * kInvSqrtLen;
}

// ------------------------------------------------- the 16 resident blocks
// The dependence of (1) on the output group c1 is a single sign, so 16 blocks
// of 32x32 cover all 64 (output group, source group) pairs:
//
//   T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i] / 16  (2)
//   s = parity(c1 & g) = (H8[c1][g] < 0)
//
// so Pi[c1*32 + c0][g*32 + i] == T[parity(c1 & g)][g][c0][i]. The Cube
// multiplies the Q panel's group-g fractal by block `TableBlockIndex(c1, g)`
// and accumulates the eight products for one c1 into one L0C column group:
// 8 x 8 = 64 Mmads for the whole 256 x 256 rotation.
inline int TableBlockIndex(int c1, int g)
{
    return Parity(static_cast<unsigned>(c1 & g)) * kGroups + g;
}

inline float TableEntry(int block, int c0, int i)
{
    const int s = block / kGroups;
    const int g = block % kGroups;
    const int sign = (s ? -1 : +1) * PiSign(g * kGroup + c0) *
                     HadamardSign(c0, i);
    return static_cast<float>(sign) * kInvSqrtLen;
}

// The table as the device reads it: kTableBlocks blocks of 32x32 bf16, each
// block already in its Nz image, because at 16 bits a row-major 32x32 blob is
// *not* its own Nz image (C0 drops to 16; see
// routines/routine_stage_page_zn_bf16.hpp and reports/micro_layout_bf16.md).
//
// `row_major` is what the device stages through `stage_blocks_nz`, which does
// the ND -> Nz conversion at ingress; `nz` is the image that conversion must
// produce and is what the host oracle models the Cube operand as. The runner
// uploads `row_major` and the probe checks the device against an oracle built
// from `nz`, so a change to either side of that ingress shows up.
struct RotationTable {
    std::vector<std::uint16_t> row_major;  // kTableElems, GM upload image
    std::vector<std::uint16_t> nz;         // kTableElems, the L1/L0 image
};

inline RotationTable BuildRotationTable()
{
    RotationTable table;
    table.row_major.assign(kTableElems, 0);
    table.nz.assign(kTableElems, 0);
    for (int block = 0; block < kTableBlocks; ++block) {
        const int base = block * kGroup * kGroup;
        for (int c0 = 0; c0 < kGroup; ++c0) {
            for (int i = 0; i < kGroup; ++i) {
                const float entry = TableEntry(block, c0, i);
                require(bfloat16_is_exact(static_cast<double>(entry)),
                        "a rotation table entry is not exact in bf16");
                const std::uint16_t raw = float_to_bfloat16_rne(entry);
                table.row_major[base + c0 * kGroup + i] = raw;
                const std::uint32_t nz_index = ascend_micro::Bf16NzBlockIndex(
                    static_cast<std::uint32_t>(kGroup),
                    static_cast<std::uint32_t>(c0),
                    static_cast<std::uint32_t>(i));
                table.nz[base + static_cast<int>(nz_index)] = raw;
            }
        }
    }
    return table;
}

// Composition check: the 16 blocks, indexed the way the Cube indexes them,
// must reproduce all 65536 entries of Pi, and Pi Pi^T must be the identity.
// `test_paged_decode_probe.cpp` runs this; it is the only thing that checks
// the resident-block factorisation itself rather than its use.
inline void ValidateRotation()
{
    for (int c = 0; c < kHeadSize; ++c) {
        for (int in = 0; in < kHeadSize; ++in) {
            const int c1 = c / kGroup;
            const int c0 = c % kGroup;
            const int g = in / kGroup;
            const int i = in % kGroup;
            const float composed = TableEntry(TableBlockIndex(c1, g), c0, i);
            require(composed == PiEntry(c, in),
                    "the resident blocks do not compose back to Pi");
        }
    }
    for (int a = 0; a < kHeadSize; ++a) {
        for (int b = 0; b < kHeadSize; ++b) {
            double sum = 0.0;
            for (int i = 0; i < kHeadSize; ++i)
                sum += static_cast<double>(PiEntry(a, i)) *
                       static_cast<double>(PiEntry(b, i));
            const double expect = (a == b) ? 1.0 : 0.0;
            require(sum == expect, "Pi Pi^T is not the identity");
        }
    }
}

// ------------------------------------------------------- cache layouts
// key_cache   [num_blocks][kv_heads][strips][kPackedBytes]
//             nibble index inside a strip = token * kHeadSize + channel,
//             token major, so the dequantised UB page is the ND matrix
//             K[token][channel] whose Nz image is the Zn image of K^T that
//             the Cube B operand needs. One 2048-byte contiguous burst.
//
// value_cache [num_blocks][kv_heads][strips][kPackedBytes]
//             nibble index inside a strip = channel * kTokens + token,
//             channel major, so the dequantised page is [256 channels x 16
//             tokens] with the token axis contiguous -- exactly the operand a
//             per-channel ReduceRepeat over 16 lanes wants for the weighted
//             value accumulation. 2048-byte contiguous burst per strip, as
//             the operator's cache contract requires.
//
// scale_cache [2][num_blocks][kv_heads][kBlockSize], FP32, planar:
//             plane 0 is the key scale per token, plane 1 the value scale.
//             Scales are powers of two, so applying them outside the Cube is
//             exact (see `ScoreColumnScale` below).
constexpr int kStripBytes = kPackedBytes;

inline std::size_t KeyStripOffset(int block, int head, int strip)
{
    return ((static_cast<std::size_t>(block) * kKvHeads + head) * kStrips + strip) *
           kStripBytes;
}
inline std::size_t ValueStripOffset(int block, int head, int strip)
{
    return KeyStripOffset(block, head, strip);
}
inline std::size_t ScaleOffset(int plane, int block, int head, int token_in_page)
{
    return ((static_cast<std::size_t>(plane) * kNumBlocks + block) * kKvHeads + head) *
               kBlockSize +
           token_in_page;
}
constexpr std::size_t kKeyCacheBytes =
    static_cast<std::size_t>(kNumBlocks) * kKvHeads * kStrips * kStripBytes;
constexpr std::size_t kValueCacheBytes = kKeyCacheBytes;
constexpr std::size_t kScaleCacheFloats =
    static_cast<std::size_t>(2) * kNumBlocks * kKvHeads * kBlockSize;

inline int KeyNibbleIndex(int token, int channel) { return token * kHeadSize + channel; }
inline int ValueNibbleIndex(int channel, int token) { return channel * kTokens + token; }

// The cache holds SYMMETRIC int4, -8..+7, which is what `int4b_t` means and
// what the device's one-instruction `int4b_t -> bfloat16_t` cast delivers. An
// unsigned 0..15 convention would need a mask stage and a second conversion
// through a pair the 950PR Cast table does not list; see
// routines/routine_turboquant_dequant4.hpp.
constexpr int kNibbleMin = -8;
constexpr int kNibbleMax = 7;

inline void PackNibble(std::uint8_t* packed, int index, int value)
{
    require(value >= kNibbleMin && value <= kNibbleMax,
            "a symmetric int4 nibble must be -8..+7");
    const auto bits = static_cast<std::uint8_t>(value & 0x0f);
    std::uint8_t& byte = packed[index / 2];
    if (index % 2 == 0)
        byte = static_cast<std::uint8_t>((byte & 0xf0) | bits);
    else
        byte = static_cast<std::uint8_t>((byte & 0x0f) | (bits << 4));
}

inline int ReadNibble(const std::uint8_t* packed, int index)
{
    const std::uint8_t byte = packed[index / 2];
    const int bits = (index % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
    return bits >= 8 ? bits - 16 : bits;  // sign extend, as int4b_t does
}

// ------------------------------------------------------------- the fixture
// Properties the hazard checks depend on, all asserted by `ValidateFixture`:
//
//   * every strip's score maximum strictly greater than the previous strip's,
//     for every query, so the online rescale path (alpha < 1) runs on every
//     strip after the first. Guaranteed structurally rather than by luck: the
//     key scale carries a factor of two per strip, so
//     max_t S[q][t] = 2^strip * max_t (2^e0 R[q][t]) and the only thing to
//     check is that max_t R[q][t] > 0;
//   * per-strip probability sums pairwise distinct;
//   * every output channel of the final context distinct;
//   * Q_rot exact in bf16 and every score an exact multiple of 1/16, so the
//     panel is checked bit-for-bit instead of against a tolerance.
//
// A fixture whose strips are interchangeable cannot detect a wrong-slot or
// run-ahead fault, because the wrong operand yields the right answer. That is
// not hypothetical here: it is what let the online-softmax WAR hazard pass its
// original validation (guidelines section 0).

// Q is sparse by construction -- one nonzero channel per Hadamard group, value
// in {-2..2} -- which bounds the integer numerator of every rotated entry by
// 8 * 2 = 16 and so keeps Q_rot exact in bf16. Q_rot itself is dense, so the
// score panel is not sparse.
inline float QueryEntry(int query, int channel)
{
    const int group = channel / kGroup;
    const int inner = channel % kGroup;
    if (inner != (query * 5 + group * 7) % kGroup) return 0.0f;
    return static_cast<float>(((query + group * 3) % 5) - 2);
}

// The sign pattern of Q_rot, computed once from `QueryEntry` and `PiEntry`
// alone in double precision. The key fixture below reads it; nothing in the
// rotation or the golden model reads the key fixture, so there is no cycle.
inline const std::vector<signed char>& QRotSignTable()
{
    static const std::vector<signed char> table = [] {
        std::vector<signed char> signs(static_cast<std::size_t>(kTokens) * kHeadSize, 0);
        for (int q = 0; q < kTokens; ++q)
            for (int c = 0; c < kHeadSize; ++c) {
                double exact = 0.0;
                for (int i = 0; i < kHeadSize; ++i)
                    exact += static_cast<double>(QueryEntry(q, i)) *
                             static_cast<double>(PiEntry(c, i));
                signs[static_cast<std::size_t>(q) * kHeadSize + c] =
                    static_cast<signed char>(exact > 0.0 ? 1 : (exact < 0.0 ? -1 : 0));
            }
        return signs;
    }();
    return table;
}

// Key nibbles.
//
// Two jobs, and they pull in different directions. The per-strip score maximum
// has to rise strictly with the strip index for *every* query, or the online
// rescale path (alpha < 1) never runs and the recurrence's hardest path goes
// unexercised. The key scale carries a factor of two per strip, so
//
//   max_t S[strip][q][t] = 2^strip . max_t (2^e0 . R[q][t])
//
// is monotone as soon as max_t R[q][t] > 0 -- and R is a signed sum against
// the Hadamard-rotated query, whose entries cancel to a mean of zero, so
// positivity is not something the modular pattern below delivers on its own.
// The first revision of this fixture did not, and the Tier-1 probe rejected
// it rather than letting the CAModel runs quietly skip the rescale path.
//
// So token t carries an *anchor*: its nibbles are biased up on exactly the
// channels where Q_rot[t] is positive. R[t][t] then contains
// kKeyAnchorBias . (sum of the positive part of Q_rot[t]) and dominates the
// modular term, while R[q][t] for q != t sees a bias pattern uncorrelated
// with Q_rot[q] and stays small. max_t R[q][t] >= R[q][q] > 0 for every q.
// The diagonal is the cheapest anchor that is positive for all 16 queries at
// once; `ValidateFixture` checks the result rather than trusting the argument.
constexpr int kKeyAnchorBias = 7;

inline int KeyNibble(int head, int strip, int token, int channel)
{
    const int base = (token * 5 + channel * 3 + strip * 11 + head * 13) % 8;  // 0..7
    const signed char sign =
        QRotSignTable()[static_cast<std::size_t>(token) * kHeadSize + channel];
    // base + bias spans 0..14; shifting by 7 lands it in -7..+7, inside the
    // symmetric int4 range, and leaves the anchor's shape untouched.
    return base + (sign > 0 ? kKeyAnchorBias : 0) - 7;  // -7..+7
}

// Value nibbles.
//
// The `channel / kTokens` term is not decoration. Any pattern of the form
// (a . channel + b) % 16 has period 16 in the channel index, so the
// accumulator would hold 16 distinct channel values repeated 16 times, the
// egress Hadamard of a period-16 vector is almost entirely zero, and a
// permuted channel would be invisible. The first revision of this fixture was
// exactly that, and the Tier-1 probe rejected it. Mixing in the high half of
// the channel index gives the pattern period kHeadSize instead.
inline int ValueNibble(int head, int strip, int channel, int token)
{
    const int raw = (channel * 7 + (channel / kTokens) * 3 + token * 5 +
                     strip * 11 + head * 3) % 15;  // 0..14
    return raw - 7;                                // -7..+7
}

// Power-of-two scales: exact to apply outside the Cube, and the key scale's
// per-strip doubling is what forces alpha < 1 on every strip after the first.
//
// The base exponent is not free. The raw panel R spans roughly [-80, +215]
// (measured, not assumed -- the Tier-1 probe prints the range), and the strip
// ramp multiplies that by 2^7 across the page, so a base exponent that is too
// large drives the late strips into a hard max: every exp but the largest
// underflows to zero, every query's probability mass collapses to 1.0, and the
// per-strip sums stop being distinguishable. The first revision used -8 and the
// Tier-1 probe rejected it for exactly that -- a degenerate softmax, not a
// rounding problem. -16 puts the whole eight-strip score range in
// [-0.59, +1.60]: the early strips are nearly flat (mass near 16 per query),
// the late ones are peaked but not saturated, alpha falls to about 0.45 on the
// last strip, and no two strips share a probability mass.
constexpr int kKeyScaleBaseExponent = -16;

inline float KeyScale(int head, int strip, int token)
{
    // The head permutes which token gets which exponent rather than changing
    // the set of exponents, so the measured score range above holds for every
    // head while a wrong-head scale read still changes the panel.
    const int exponent = kKeyScaleBaseExponent + strip + ((token + head) % 3);
    return std::ldexp(1.0f, exponent);
}
inline float ValueScale(int head, int strip, int token)
{
    const int exponent = -4 - (token % 4) - (strip % 2) - (head % 2);
    return std::ldexp(1.0f, exponent);
}

// The gate the egress applies. Dense, and non-periodic in the channel index
// for the same reason `ValueNibble` is.
inline float GateEntry(int head, int query, int channel)
{
    const int raw =
        (query * 13 + channel * 5 + (channel / kTokens) * 7 + head * 11) % 17;
    return static_cast<float>(raw - 8) / 8.0f;
}

// Logical temporal block -> physical page. Deliberately not the identity, so a
// kernel that ignores the block table and indexes the cache by its own grid
// position reads the wrong page and fails.
inline int BlockTableEntry(int temporal) { return (temporal * 5 + 3) % kNumBlocks; }

// ------------------------------------------------------- the golden model
// What the device must reproduce, in the order it reproduces it.

// Q_rot = Q . Pi^T, i.e. Q_rot[q][c] = sum_i Q[q][i] . Pi[c][i], computed the
// way the Cube computes it: eight accumulating group contributions per output
// group, in group order, with an FP32 accumulator. The fixture makes the sum
// exact, so the order does not matter -- `ValidateFixture` is what establishes
// that rather than this comment.
inline std::vector<float> GoldenQRot(int /*head*/)
{
    std::vector<float> rot(static_cast<std::size_t>(kTokens) * kHeadSize, 0.0f);
    for (int q = 0; q < kTokens; ++q) {
        for (int c1 = 0; c1 < kGroups; ++c1) {
            for (int c0 = 0; c0 < kGroup; ++c0) {
                float sum = 0.0f;
                for (int g = 0; g < kGroups; ++g) {
                    const int block = TableBlockIndex(c1, g);
                    for (int i = 0; i < kGroup; ++i)
                        sum += QueryEntry(q, g * kGroup + i) *
                               TableEntry(block, c0, i);
                }
                rot[static_cast<std::size_t>(q) * kHeadSize + c1 * kGroup + c0] = sum;
            }
        }
    }
    return rot;
}

// The bf16 image of Q_rot that the Cube's Fixpipe leaves in L1, and therefore
// the A operand of every strip's score Mmad.
inline std::vector<float> GoldenQRotBf16(int head)
{
    std::vector<float> rot = GoldenQRot(head);
    for (float& value : rot)
        value = bfloat16_to_float(float_to_bfloat16_rne(value));
    return rot;
}

// Raw score panel: R[q][t] = sum_c Q_rot_bf16[q][c] . key_nibble[t][c].
// One Mmad with m = 16, n = 16, k = 256. Exact.
inline std::vector<float> GoldenRawScores(int head, int strip)
{
    const std::vector<float> rot = GoldenQRotBf16(head);
    std::vector<float> raw(static_cast<std::size_t>(kScores), 0.0f);
    for (int q = 0; q < kTokens; ++q)
        for (int t = 0; t < kTokens; ++t) {
            float sum = 0.0f;
            for (int c = 0; c < kHeadSize; ++c)
                sum += rot[static_cast<std::size_t>(q) * kHeadSize + c] *
                       static_cast<float>(KeyNibble(head, strip, t, c));
            raw[static_cast<std::size_t>(q) * kTokens + t] = sum;
        }
    return raw;
}

// S[q][t] = keyScale[t] . R[q][t]. The key's per-token scale is applied as a
// score column scale rather than to the 4096-element key page: one exact FP32
// multiply per column against 4096 bf16 multiplies, and it keeps the Cube's
// operand an exact small integer, which is what makes the panel bit-checkable.
inline float ScoreColumnScale(int head, int strip, int token)
{
    return KeyScale(head, strip, token);
}

struct GoldenState {
    std::vector<float> m;        // kTokens
    std::vector<float> l;        // kTokens
    std::vector<float> acc;      // kTokens * kHeadSize, rotated basis
    std::vector<float> context;  // kTokens * kHeadSize, true basis, gated
};

// The streaming online softmax over the selected strips, then the egress:
// rotate the accumulator back with Pi and gate it with sigmoid(gate).
//
// `strips` is the list of strips this task processes, in order. The canary
// protocol runs a single strip; the steady-state run processes 0..7.
inline GoldenState GoldenDecode(int head, const std::vector<int>& strips)
{
    GoldenState state;
    state.m.assign(kTokens, -std::numeric_limits<float>::infinity());
    state.l.assign(kTokens, 0.0f);
    state.acc.assign(static_cast<std::size_t>(kTokens) * kHeadSize, 0.0f);

    for (const int strip : strips) {
        const std::vector<float> raw = GoldenRawScores(head, strip);
        for (int q = 0; q < kTokens; ++q) {
            float scores[kTokens];
            float tile_max = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < kTokens; ++t) {
                scores[t] = raw[static_cast<std::size_t>(q) * kTokens + t] *
                            ScoreColumnScale(head, strip, t);
                if (scores[t] > tile_max) tile_max = scores[t];
            }
            const float m_new = std::max(state.m[q], tile_max);
            const float alpha = std::exp(state.m[q] - m_new);
            float p[kTokens];
            float p_sum = 0.0f;
            for (int t = 0; t < kTokens; ++t) {
                p[t] = std::exp(scores[t] - m_new);
                p_sum += p[t];
            }
            // The value's per-token scale folds into p, so the weighted
            // accumulation reads raw nibbles: 16 exact multiplies instead of
            // 4096, and the denominator still sees the unscaled p.
            float p_scaled[kTokens];
            for (int t = 0; t < kTokens; ++t)
                p_scaled[t] = p[t] * ValueScale(head, strip, t);
            for (int c = 0; c < kHeadSize; ++c) {
                float weighted = 0.0f;
                for (int t = 0; t < kTokens; ++t)
                    weighted += p_scaled[t] *
                                static_cast<float>(ValueNibble(head, strip, c, t));
                const std::size_t index = static_cast<std::size_t>(q) * kHeadSize + c;
                state.acc[index] = state.acc[index] * alpha + weighted;
            }
            state.l[q] = state.l[q] * alpha + p_sum;
            state.m[q] = m_new;
        }
    }

    // Egress. ctx = (acc / l) . Pi, then gated by sigmoid(gate). The rotation
    // is the inverse of the one the cache was written with: acc lives in the
    // rotated channel basis because the cached values do, and Pi Pi^T = I.
    state.context.assign(static_cast<std::size_t>(kTokens) * kHeadSize, 0.0f);
    for (int q = 0; q < kTokens; ++q) {
        std::vector<float> normalized(kHeadSize, 0.0f);
        for (int c = 0; c < kHeadSize; ++c)
            normalized[c] = state.l[q] == 0.0f
                                ? 0.0f
                                : state.acc[static_cast<std::size_t>(q) * kHeadSize + c] /
                                      state.l[q];
        for (int out = 0; out < kHeadSize; ++out) {
            float sum = 0.0f;
            for (int c = 0; c < kHeadSize; ++c)
                sum += normalized[c] * PiEntry(c, out);
            const float gate = GateEntry(head, q, out);
            const float gated = sum / (1.0f + std::exp(-gate));
            state.context[static_cast<std::size_t>(q) * kHeadSize + out] = gated;
        }
    }
    return state;
}

// The egress rotation as eight butterfly stages plus a sign multiply, which is
// how AIV0 computes it. Right-multiplying by
// Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16 is
//
//   strides 32, 64, 128   the 8-point Hadamard across groups  (H8 (x) I32)
//   elementwise           diag(piSigns)
//   strides 1, 2, 4, 8,16 the 32-point Hadamard within a group (I8 (x) H32)
//   scale                 1/16
//
// with a stage at stride s being out[i] = in[i ^ s] + sgn(i)[i] . in[i],
// sgn = +1 where (i & s) == 0 and -1 elsewhere. Exact in FP32 for dyadic
// inputs. `test_paged_decode_probe.cpp` checks this against the dense form, so
// the device's butterfly order is validated independently of the device.
constexpr int kEgressStageCount = 8;
constexpr int kEgressStrides[kEgressStageCount] = {32, 64, 128, 1, 2, 4, 8, 16};
// The sign multiply sits between the group stages and the inner stages.
constexpr int kEgressSignAfterStage = 3;

// The three stages whose partner lane i ^ s lies inside a 32-byte DataBlock.
// Those cannot be expressed as an offset operand -- a 4, 8 or 16 byte tensor
// offset is not 32-byte aligned -- so the device runs them as a Gather against
// a per-stage offset vector plus a per-stage sign vector, and the runner
// uploads both. Must match `kEgressGatherStrideOf` in
// routines/routine_vector_egress_rotate.hpp.
constexpr int kEgressGatherCount = 3;
constexpr int kEgressGatherStride[kEgressGatherCount] = {1, 2, 4};

inline std::vector<float> EgressButterfly(const float* row)
{
    std::vector<float> value(row, row + kHeadSize);
    std::vector<float> next(kHeadSize, 0.0f);
    for (int stage = 0; stage < kEgressStageCount; ++stage) {
        if (stage == kEgressSignAfterStage)
            for (int i = 0; i < kHeadSize; ++i)
                value[i] *= static_cast<float>(PiSign(i));
        const int stride = kEgressStrides[stage];
        for (int i = 0; i < kHeadSize; ++i) {
            const float sign = (i & stride) ? -1.0f : 1.0f;
            next[i] = value[i ^ stride] + sign * value[i];
        }
        value.swap(next);
    }
    for (int i = 0; i < kHeadSize; ++i) value[i] *= kInvSqrtLen;
    return value;
}

// --------------------------------------------------------- fixture validation
struct FixtureReport {
    std::vector<float> strip_max;   // kStrips, max score over q and t
    std::vector<float> strip_psum;  // kStrips, sum of p over q and t
    double max_qrot_magnitude = 0.0;
};

inline FixtureReport ValidateFixture(int head)
{
    FixtureReport report;

    // Q_rot must be exact in FP32 and in bf16, computed in double here so the
    // check is independent of the float accumulation it is checking.
    for (int q = 0; q < kTokens; ++q) {
        for (int c = 0; c < kHeadSize; ++c) {
            double exact = 0.0;
            for (int i = 0; i < kHeadSize; ++i)
                exact += static_cast<double>(QueryEntry(q, i)) *
                         static_cast<double>(PiEntry(c, i));
            require(bfloat16_is_exact(exact),
                    "Q_rot is not exact in bf16: the fixture would round and "
                    "the score panel could not be checked bit-exactly");
            const double magnitude = exact < 0 ? -exact : exact;
            if (magnitude > report.max_qrot_magnitude)
                report.max_qrot_magnitude = magnitude;
        }
    }
    const std::vector<float> rot_bf16 = GoldenQRotBf16(head);
    for (int q = 0; q < kTokens; ++q)
        for (int c = 0; c < kHeadSize; ++c) {
            double exact = 0.0;
            for (int i = 0; i < kHeadSize; ++i)
                exact += static_cast<double>(QueryEntry(q, i)) *
                         static_cast<double>(PiEntry(c, i));
            require(static_cast<double>(
                        rot_bf16[static_cast<std::size_t>(q) * kHeadSize + c]) == exact,
                    "the float Q_rot accumulation disagrees with the double one");
        }

    // Every raw score an exact multiple of 1/16, and max_t R[q][t] > 0 for
    // every query, which is what makes the scaled maxima monotone in strip.
    for (int strip = 0; strip < kStrips; ++strip) {
        const std::vector<float> raw = GoldenRawScores(head, strip);
        for (int q = 0; q < kTokens; ++q) {
            float best = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < kTokens; ++t) {
                const float value = raw[static_cast<std::size_t>(q) * kTokens + t];
                const double scaled = static_cast<double>(value) * 16.0;
                require(scaled == std::floor(scaled),
                        "a raw score is not an exact multiple of 1/16");
                if (value > best) best = value;
            }
            require(best > 0.0f,
                    "a query has no positive raw score, so doubling the key "
                    "scale would not raise its strip maximum");
        }
    }

    // Monotone maxima and distinct per-strip probability mass.
    report.strip_max.assign(kStrips, -std::numeric_limits<float>::infinity());
    report.strip_psum.assign(kStrips, 0.0f);
    for (int strip = 0; strip < kStrips; ++strip) {
        const std::vector<float> raw = GoldenRawScores(head, strip);
        for (int q = 0; q < kTokens; ++q) {
            float tile_max = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < kTokens; ++t) {
                const float score = raw[static_cast<std::size_t>(q) * kTokens + t] *
                                    ScoreColumnScale(head, strip, t);
                if (score > tile_max) tile_max = score;
            }
            if (tile_max > report.strip_max[strip]) report.strip_max[strip] = tile_max;
            for (int t = 0; t < kTokens; ++t) {
                const float score = raw[static_cast<std::size_t>(q) * kTokens + t] *
                                    ScoreColumnScale(head, strip, t);
                report.strip_psum[strip] += std::exp(score - tile_max);
            }
        }
    }
    for (int strip = 1; strip < kStrips; ++strip)
        require(report.strip_max[strip] > report.strip_max[strip - 1],
                "per-strip score maxima are not strictly increasing, so the "
                "online rescale path does not run on every strip");
    for (int a = 0; a < kStrips; ++a)
        for (int b = a + 1; b < kStrips; ++b)
            require(report.strip_psum[a] != report.strip_psum[b],
                    "two strips have identical probability mass");

    // The output must distinguish a permuted channel. Demanding that all 256
    // FP32 values of a row be pairwise distinct is both too strong (two of 256
    // floats may coincide by arithmetic coincidence without hiding anything)
    // and too weak (it says nothing about the permutations this kernel could
    // actually get wrong). So check the two properties that matter:
    //
    //   * a large fraction of each row's channels carry distinct values, which
    //     is what fails when the value cache is periodic in the channel index
    //     and the egress Hadamard collapses to near-zero;
    //   * the row is not invariant under c -> c ^ stride for any of the eight
    //     egress butterfly strides, nor under the group/inner index swap. Those
    //     are the specific confusions a wrong Gather offset, a dropped stage or
    //     a transposed group layout would produce.
    std::vector<int> strips(kStrips);
    for (int i = 0; i < kStrips; ++i) strips[i] = i;
    const GoldenState state = GoldenDecode(head, strips);
    for (int q = 0; q < kTokens; ++q) {
        const float* row = state.context.data() + static_cast<std::size_t>(q) * kHeadSize;
        std::vector<std::uint32_t> values(kHeadSize);
        for (int c = 0; c < kHeadSize; ++c) values[c] = float_bits(row[c]);
        std::sort(values.begin(), values.end());
        const auto distinct =
            static_cast<int>(std::unique(values.begin(), values.end()) - values.begin());
        require(distinct * 4 >= kHeadSize * 3,
                "fewer than three quarters of the output channels are distinct, so "
                "a permuted channel could go unobserved");
        for (int stage = 0; stage < kEgressStageCount; ++stage) {
            const int stride = kEgressStrides[stage];
            int changed = 0;
            for (int c = 0; c < kHeadSize; ++c)
                if (float_bits(row[c]) != float_bits(row[c ^ stride])) ++changed;
            require(changed * 2 >= kHeadSize,
                    "the output row is near-invariant under an egress butterfly "
                    "stride, so a dropped or mis-strided stage would not show");
        }
        int swapped = 0;
        for (int c = 0; c < kHeadSize; ++c) {
            const int mirrored = (c % kGroup) * kGroups + (c / kGroup);
            if (float_bits(row[c]) != float_bits(row[mirrored])) ++swapped;
        }
        require(swapped * 2 >= kHeadSize,
                "the output row is near-invariant under the group/inner index "
                "swap, so a transposed group layout would not show");
    }
    return report;
}

// ------------------------------------------------------- cache construction
// Fills a packed cache image. Only the strips of the pages the block table
// points at are written; everything else keeps whatever the caller put there,
// which under the canary protocol is 0x7FC0BEEF.
struct CacheFill {
    int block = 0;
    int head = 0;
    int strip = 0;
};

inline void WriteKeyStrip(std::uint8_t* key_cache, const CacheFill& where, int strip_data)
{
    std::uint8_t* strip = key_cache + KeyStripOffset(where.block, where.head, where.strip);
    for (int token = 0; token < kTokens; ++token)
        for (int channel = 0; channel < kHeadSize; ++channel)
            PackNibble(strip, KeyNibbleIndex(token, channel),
                       KeyNibble(where.head, strip_data, token, channel));
}

inline void WriteValueStrip(std::uint8_t* value_cache, const CacheFill& where,
                            int strip_data)
{
    std::uint8_t* strip =
        value_cache + ValueStripOffset(where.block, where.head, where.strip);
    for (int channel = 0; channel < kHeadSize; ++channel)
        for (int token = 0; token < kTokens; ++token)
            PackNibble(strip, ValueNibbleIndex(channel, token),
                       ValueNibble(where.head, strip_data, channel, token));
}

inline void WriteScales(float* scale_cache, const CacheFill& where, int strip_data)
{
    for (int token = 0; token < kTokens; ++token) {
        const int token_in_page = where.strip * kTokens + token;
        scale_cache[ScaleOffset(0, where.block, where.head, token_in_page)] =
            KeyScale(where.head, strip_data, token);
        scale_cache[ScaleOffset(1, where.block, where.head, token_in_page)] =
            ValueScale(where.head, strip_data, token);
    }
}

}  // namespace paged_decode
