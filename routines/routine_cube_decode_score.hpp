#pragma once

// Cube leaves for the paged decode attention operator, Ascend 950PR / arch35 /
// dav-3510 / CANN 9.2. Two of them:
//
//   rotate_query     the ingress rotation Q . Pi^T as 64 accumulating bf16
//                    Mmads against 16 resident 32x32 blocks, run once in the
//                    Cube prologue, with the result landing back in L1 as the
//                    bf16 A operand every strip's score GEMM then reuses.
//   score_strip      one strip's S = Q_rot . K_strip^T: a single Mmad with
//                    k = 256 and one Fixpipe into a Vector core's UB.
//
// Both are asynchronous leaves in the sense of
// MICROROUTINE_DESIGN_GUIDELINES.md section 3.4: they dispatch MTE1 / M / FIX
// work and issue no HardEvent, no PipeBarrier and no cross-core flag. The
// caller owns every edge.
//
// Why 64 Mmads and not 16
// -----------------------
// The rotation is the randomised Hadamard
//
//   Pi = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) / 16,
//   Pi[c1*32 + c0][g*32 + i] = H8[c1][g] . piSigns[g*32 + c0] . H32[c0][i] / 16
//
// whose dependence on the output group c1 is a single sign, so 16 blocks of
// 32x32 cover all 64 (output group, source group) pairs:
//
//   T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i] / 16,
//   s = parity(c1 & g).
//
// Folding the sign diagonal and the outer factor into the resident operand
// costs eight times the multiply-adds of the factored form and buys back every
// Vector-side pass the factored form needed. That trade was measured, not
// argued, on the prefill operator -- the Cube waited on a Vector core for
// 33.6% of its attributed cycles against 2.8% spent multiplying -- and is
// recorded in reports/reshape_and_cache_a8.md. This leaf is the decode-side
// reuse of that result, applied once to a 16 x 256 query panel instead of once
// per cached page.
//
// The 16-bit fractal, and the operand image it demands
// ----------------------------------------------------
// The arch-3510 fractal is 16 x (32B / sizeof(T)), so C0 is 16 for bf16 and a
// 32-wide axis is TWO fractals, not one. Two consequences, both settled by
// measurement in reports/micro_layout_bf16.md rather than by reasoning here:
//
//   * the 16 x 256 panel's Nz image is still the Zn image of its transpose, so
//     one L1 image serves both operand roles and the page needs no new load
//     path;
//   * a row-major 32 x 32 resident block is NOT its own Nz image. Its L1 image
//     has to be produced by an ND -> Nz ingress conversion
//     (`stage_blocks_nz` in routine_stage_page_zn_bf16.hpp). MTE1 cannot
//     repair it: its copy unit is the 512-byte fractal and its only
//     rearrangement is an intra-fractal transpose.
//
// This leaf therefore takes L1 base addresses that are already correct images
// and performs byte-identity L1 -> L0 loads, exactly as
// routine_fractal_load_zn_bf16.hpp documents.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "routine_fractal_load_zn_bf16.hpp"
#endif

namespace ascend_micro {

// Geometry of the rotation, shared with the operator and the host oracle.
constexpr std::uint32_t kDecodeTokens = 16;
constexpr std::uint32_t kDecodeHeadSize = 256;
constexpr std::uint32_t kDecodeGroup = 32;
constexpr std::uint32_t kDecodeGroups = kDecodeHeadSize / kDecodeGroup;  // 8
constexpr std::uint32_t kDecodeTableBlocks = 2u * kDecodeGroups;         // 16

// Element counts inside L0, in bf16 elements unless stated.
constexpr std::uint32_t kDecodeBlockElems = kDecodeGroup * kDecodeGroup;        // 1024
constexpr std::uint32_t kDecodeTableElems = kDecodeTableBlocks * kDecodeBlockElems;
constexpr std::uint32_t kDecodeGroupFractalElems = kDecodeTokens * kDecodeGroup;  // 512
constexpr std::uint32_t kDecodePanelElems = kDecodeTokens * kDecodeHeadSize;      // 4096

// H8[c1][g] = (-1)^popcount(c1 & g), so the block for (c1, g) is the one whose
// sign flag is that parity. Resolved at compile time, once per Mmad site.
constexpr std::uint32_t DecodeParityOf(std::uint32_t value)
{
    std::uint32_t parity = 0;
    while (value != 0u) {
        parity ^= (value & 1u);
        value >>= 1;
    }
    return parity;
}

// Variable templates rather than constexpr functions: in the CCE dialect a
// plain constexpr function is __host__, so an __aicore__ function may not call
// one. A variable template is a compile-time value and carries no such colour.
template <std::uint32_t C1, std::uint32_t G>
constexpr std::uint32_t kDecodeSignBlock = DecodeParityOf(C1 & G) * kDecodeGroups + G;

#if defined(__CCE_AICORE__)

// ===================================================== the ingress rotation
// L0 occupancy during the prologue, in bf16 elements:
//
//   L0A  [0, 8 * 512)            the eight Q group fractals, 4096 elements
//   L0B  [0, kDecodeTableElems)  the 16 resident blocks, 16384 elements
//   L0C  [0, 4096) FP32          the 16 x 256 rotated panel
//
// The resident blocks live only in L0B here: unlike the prefill operator, this
// leaf never needs them in the A role, because the panel being rotated is
// always the A operand. L0A therefore holds nothing but the eight Q group
// fractals, 4096 elements, which is exactly the size Q_rot needs afterwards --
// so one L0A allocation serves the prologue and then the whole strip sequence.
constexpr std::uint32_t kDecodeL0aQGroups = 0u;

// MTE1: the 16 resident blocks into L0B in the B role, byte-identity.
template <std::uint32_t Block>
__aicore__ inline void LoadRotationBlock(std::uint32_t l1_table_base,
                                         __cb__ bfloat16_t* l0b)
{
    load_block_b_zn<kDecodeGroup>(
        l1_table_base + Block * kDecodeBlockElems * kZnBf16ElemBytes,
        l0b + Block * kDecodeBlockElems);
}

template <std::uint32_t... Blocks>
__aicore__ inline void LoadRotationBlocks(std::uint32_t l1_table_base,
                                          __cb__ bfloat16_t* l0b,
                                          std::integer_sequence<std::uint32_t, Blocks...>)
{
    (LoadRotationBlock<Blocks>(l1_table_base, l0b), ...);
}

// MTE1: Q group G into L0A in the A role. A[q][i] = Q[q][G*32 + i].
template <std::uint32_t G>
__aicore__ inline void LoadQueryGroup(std::uint32_t l1_q_base, __ca__ bfloat16_t* l0a)
{
    load_page_group_a_nz<kDecodeTokens, kDecodeGroup>(
        l1_q_base + G * kDecodeGroupFractalElems * kZnBf16ElemBytes,
        l0a + kDecodeL0aQGroups + G * kDecodeGroupFractalElems);
}

template <std::uint32_t... Gs>
__aicore__ inline void LoadQueryGroups(std::uint32_t l1_q_base, __ca__ bfloat16_t* l0a,
                                       std::integer_sequence<std::uint32_t, Gs...>)
{
    (LoadQueryGroup<Gs>(l1_q_base, l0a), ...);
}

// One accumulating Mmad. Output group C1 takes the contribution of source
// group G through resident block kDecodeSignBlock<C1, G>; G == 0 initialises
// the L0C region and the other seven accumulate into it, so one region ends up
// holding
//
//   sum_G sum_i T[s(C1,G)][G][c0][i] . Q[q][G*32 + i]
//
// which is column C1*32 + c0 of Q . Pi^T. cmatrixInitVal is the compile-time
// flag that picks "C = A B" over "C += A B"; the Cube pipe is in order, so
// nothing else has to order the eight accumulations.
template <std::uint32_t C1, std::uint32_t G>
__aicore__ inline void RotateBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                   __cc__ float* l0c)
{
    using namespace AscendC::Te;
    constexpr bool init = (G == 0u);
    auto a = MakeTensor(
        MakeMemPtr(l0a + kDecodeL0aQGroups + G * kDecodeGroupFractalElems),
        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kDecodeTokens, kDecodeGroup));
    auto b = MakeTensor(
        MakeMemPtr(l0b + kDecodeSignBlock<C1, G> * kDecodeBlockElems),
        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kDecodeGroup, kDecodeGroup));
    auto c = MakeTensor(MakeMemPtr(l0c + C1 * kDecodeTokens * kDecodeGroup),
                        MakeFrameLayout<NZLayoutPtn, 16>(kDecodeTokens, kDecodeGroup));
    AscendC::Te::MmadParams mp(static_cast<std::uint16_t>(kDecodeTokens),
                               static_cast<std::uint16_t>(kDecodeGroup),
                               static_cast<std::uint16_t>(kDecodeGroup), 0, init);
    Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
}

template <std::uint32_t C1, std::uint32_t... Gs>
__aicore__ inline void RotateBlockRow(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                      __cc__ float* l0c,
                                      std::integer_sequence<std::uint32_t, Gs...>)
{
    (RotateBlock<C1, Gs>(l0a, l0b, l0c), ...);
}

template <std::uint32_t... C1s>
__aicore__ inline void RotateBlockRows(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                       __cc__ float* l0c,
                                       std::integer_sequence<std::uint32_t, C1s...>)
{
    (RotateBlockRow<C1s>(l0a, l0b, l0c,
                         std::make_integer_sequence<std::uint32_t, kDecodeGroups>{}),
     ...);
}

// MTE1 for the whole rotation: 16 block loads plus 8 query group loads. One
// MTE1_M edge covers all 24, because every operand is in place before the
// first Mmad issues.
__aicore__ inline void rotate_query_stage_l0(std::uint32_t l1_table_base,
                                             std::uint32_t l1_q_base,
                                             __ca__ bfloat16_t* l0a,
                                             __cb__ bfloat16_t* l0b)
{
    LoadRotationBlocks(l1_table_base, l0b,
                       std::make_integer_sequence<std::uint32_t, kDecodeTableBlocks>{});
    LoadQueryGroups(l1_q_base, l0a,
                    std::make_integer_sequence<std::uint32_t, kDecodeGroups>{});
}

// The 64 Mmads. Nothing else; the caller owns the M_FIX edge to the transfer
// below, because a group of 64 accumulating Mmads feeds one transfer and
// unitFlag pairs one Mmad with one Fixpipe.
__aicore__ inline void rotate_query_compute(__ca__ bfloat16_t* l0a,
                                            __cb__ bfloat16_t* l0b,
                                            __cc__ float* l0c)
{
    AscendC::SetMMRowMajor();
    RotateBlockRows(l0a, l0b, l0c,
                    std::make_integer_sequence<std::uint32_t, kDecodeGroups>{});
}

// FIX: the finished FP32 panel in L0C out to Global Memory as bf16, in ND.
//
// The F32 -> BF16 conversion is NOT requested through a parameter. The Tensor
// API derives it at compile time from the trait's round mode and the operand
// types -- `GetQuantMode<trait.roundMode, T, U>()` in
// impl/tensor_api/arch/cube/l0c_to_*/copy_impl/data_copy.h. `Te::FixpipeParams`
// carries only `unitFlag` and `subBlockId`; the `quantPre` field belongs to the
// ISASI `FixpipeParamsArch3510`, which this path does not take. Passing a bf16
// destination over an FP32 L0C source IS the cast request. The fixture keeps
// every rotated entry exact in bf16, so the cast is a relabelling and not a
// rounding -- the host probe asserts that rather than this comment claiming it.
//
// Why Global Memory and not straight to L1
// ----------------------------------------
// The first revision sent L0C to L1 in one Fixpipe (`CopyL0C2L1`, with a bf16
// Nz destination) and fed L0A from there, which is the obvious on-chip route
// and needs no round trip. **It does not work on this part as written, and the
// run that established that is the one worth recording:** with that path the
// score panel came back all 256 entries zero -- not transposed, not scrambled,
// no NaN, not canary -- while in the SAME run the same L0C contents reached
// Global Memory bit-exactly through `CopyL0C2GM`. So L0C held the right 4096
// values and the L1 branch delivered nothing L0A could read.
//
// The mechanism was not isolated, and the honest statement is that: whether
// `CopyL0C2L1` needs a trait or round mode this code did not set, whether a
// bf16 Nz destination in L1 is a supported combination at all, or whether the
// subsequent `CopyL12L0A` of a 16-fractal panel is what failed, is untested
// here. What is tested is that this route -- L0C -> GM through the Fixpipe the
// runner already checks bit-exactly, then the same GM -> L1 ND -> Nz ingress
// the query itself uses, then the same byte-identity L1 -> L0A load -- is
// built entirely from paths this harness has verified, and produces a
// bit-exact score panel.
//
// The cost is one Global Memory round trip, in the prologue, once per launch.
// The destination is `out_qrot`, a declared output the runner checks, not a
// workspace: the eight-strip pipeline that follows still touches Global Memory
// only to read its packed pages and scales, and writes it only once at the
// end. "No GM workspace" is a claim about the strip loop and stays true; it is
// not a claim about the prologue any more, and the report says so.
__aicore__ inline void rotate_query_to_gm(__gm__ bfloat16_t* gm_qrot,
                                          __cc__ float* l0c)
{
    using namespace AscendC::Te;
    auto c = MakeTensor(MakeMemPtr(l0c),
                        MakeFrameLayout<NZLayoutPtn, 16>(kDecodeTokens, kDecodeHeadSize));
    auto y = MakeTensor(MakeMemPtr(gm_qrot),
                        MakeFrameLayout<NDExtLayoutPtn>(kDecodeTokens, kDecodeHeadSize));
    // 64 accumulating Mmads feed this one transfer, so no unitFlag pairing.
    const FixpipeParams fp(0, false);
    Copy(MakeCopy(CopyL0C2GM{}).with(fp), y, c);
}

// MTE1: Q_rot from L1 into L0A as Nz(16, 256), resident for the whole strip
// sequence. It overwrites the eight Q group fractals, which the Mmads above
// have finished with; the caller owns the M_MTE1 edge that makes that safe.
__aicore__ inline void load_query_rotated(std::uint32_t l1_qrot_base,
                                          __ca__ bfloat16_t* l0a)
{
    load_page_group_a_nz<kDecodeTokens, kDecodeHeadSize>(l1_qrot_base, l0a);
}

// ======================================================== the per-strip GEMM
// S = Q_rot . K_strip^T, as eight accumulating Mmads over 32-channel groups.
//
// A is the resident Q_rot, Nz(16, 256). B is the strip's dequantised key page
// read as Zn(256, 16): B[c][t] = K[t][c], the same L1 bytes the staging core
// wrote as the ND page K[token][channel], because at 16 bits the page's Nz
// image is still the Zn image of its transpose:
//
//   Nz(16, 256)  offset = n1*256 + t*16 + n0,   channel = n1*16 + n0
//   Zn(256, 16)  offset = k1*256 + t*16 + k0,   channel = k1*16 + k0
//
// the same map with the second index renamed, so one L1 image and two
// byte-identity L1 -> L0 loads cover both roles.
//
// Why eight Mmads of k = 32 and not one of k = 256
// ------------------------------------------------
// `MmadParams` takes m, n and k as uint16 extents, so a single instruction
// with k = 256 is expressible and was the first revision here. It is **not**
// what this part computes correctly: the CAModel run with one k = 256 Mmad
// produced a score panel with 251 of 256 entries wrong, while the ingress
// rotation in the same kernel -- 64 accumulating Mmads of k = 32 through the
// identical operand path, Fixpipe and fixture -- came out bit-exact in the
// same run. The contraction depth was the only difference between them.
//
// So the depth stays at the group width the rest of this operator family uses,
// 32 channels, which is one INT8 K-fractal and two bf16 ones, and the eight
// group products accumulate in L0C with `cmatrixInitVal` set on the first.
// The Cube pipe is in order, so nothing else has to order the accumulation.
//
// This also means the Mmads can no longer be paired to the Fixpipe by
// `unitFlag`: eight accumulating Mmads feed one transfer, so the caller owns
// an explicit M_FIX edge between `score_strip_compute` and
// `score_strip_fixpipe` -- the same split `turboquant_reshape_and_cache_a8.cpp`
// makes for its 64-Mmad nest, and it lets the score-window back edge be waited
// after the multiplies instead of in front of them.
__aicore__ inline void score_strip_stage_l0(std::uint32_t l1_key_base,
                                            __cb__ bfloat16_t* l0b_slot)
{
    load_page_group_b_zn<kDecodeTokens, kDecodeHeadSize>(l1_key_base, l0b_slot);
}

// Channel group G of the contraction. Both operands' group slices sit at the
// same element offset, because both images share the k-fractal stride.
template <std::uint32_t G>
__aicore__ inline void ScoreBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b_slot,
                                  __cc__ float* l0c_window)
{
    using namespace AscendC::Te;
    constexpr bool init = (G == 0u);
    constexpr std::uint32_t offset = G * kDecodeTokens * kDecodeGroup;
    auto a = MakeTensor(
        MakeMemPtr(l0a + offset),
        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kDecodeTokens, kDecodeGroup));
    auto b = MakeTensor(
        MakeMemPtr(l0b_slot + offset),
        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kDecodeGroup, kDecodeTokens));
    auto c = MakeTensor(MakeMemPtr(l0c_window),
                        MakeFrameLayout<NZLayoutPtn, 16>(kDecodeTokens, kDecodeTokens));
    AscendC::Te::MmadParams mp(static_cast<std::uint16_t>(kDecodeTokens),
                               static_cast<std::uint16_t>(kDecodeTokens),
                               static_cast<std::uint16_t>(kDecodeGroup), 0, init);
    Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
}

template <std::uint32_t... Gs>
__aicore__ inline void ScoreBlocks(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b_slot,
                                   __cc__ float* l0c_window,
                                   std::integer_sequence<std::uint32_t, Gs...>)
{
    (ScoreBlock<Gs>(l0a, l0b_slot, l0c_window), ...);
}

__aicore__ inline void score_strip_compute(__ca__ bfloat16_t* l0a,
                                           __cb__ bfloat16_t* l0b_slot,
                                           __cc__ float* l0c_window)
{
    // Results first along N, then along M: the orientation every Cube leaf in
    // this harness sets before an Mmad whose panel leaves L0C through an
    // Nz -> ND Fixpipe.
    AscendC::SetMMRowMajor();
    ScoreBlocks(l0a, l0b_slot, l0c_window,
                std::make_integer_sequence<std::uint32_t, kDecodeGroups>{});
}

// FIX: the finished 16 x 16 FP32 window into a Vector core's UB, as ND.
// `subBlockId` is how the Cube addresses a chosen Vector core's Unified
// Buffer; the recurrence lives on sub block 0. No unitFlag, because eight
// accumulating Mmads feed this one transfer.
__aicore__ inline void score_strip_fixpipe(__cc__ float* l0c_window,
                                           __ubuf__ float* ub_scores,
                                           bool to_sub_block_one)
{
    using namespace AscendC::Te;
    auto src = MakeTensor(MakeMemPtr(l0c_window),
                          MakeFrameLayout<NZLayoutPtn, 16>(kDecodeTokens, kDecodeTokens));
    auto dst = MakeTensor(MakeMemPtr(ub_scores),
                          MakeFrameLayout<NDExtLayoutPtn>(kDecodeTokens, kDecodeTokens));
    const FixpipeParams fp(0, to_sub_block_one);
    Copy(MakeCopy(CopyL0C2UB{}).with(fp), dst, src);
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
