#pragma once

// Approach 1 -- MTE1 fractal loads that supply 16-bit Cube operands in Zn
// order on Ascend 950PR / arch35 / dav-3510, CANN 9.2.
//
// What MTE1 can and cannot do on this path
// ----------------------------------------
// `Copy(MakeCopy(CopyL12L0A{}), ...)` and `CopyL12L0B` move **512-byte
// fractals** -- "This path uses a 512-byte data fractal as the basic copy
// unit", with the b16 fractal cube 16 x 16
// (cube_compute_TensorAPI/cube_compute_load/Copy_L1ToL0B.md). Exactly two
// layout combinations exist per destination:
//
//   L0A:  Nz -> Nz  (non-transpose)      Zn -> Nz  (transposes each fractal)
//   L0B:  Zn -> Zn  (non-transpose)      Nz -> Zn  (transposes each fractal)
//
// Both transposing forms transpose *inside* a fractal. Neither reorders
// fractals, and no form gathers a sub-fractal out of a source that is not
// already fractal-organised. So the question the BF16 port raises --
// "can MTE1 interleave the two 16 x 16 sub-fractals of a 32-wide Hadamard
// group out of a flat L1 image?" -- has a definite answer: **no**, and it is a
// property of the path's copy unit rather than of any parameter choice. The
// low-level ISASI `LoadData` is the same instruction underneath
// (LOAD_2Dv2 addresses L1 by fractal index), so it does not widen the
// envelope either.
//
// What MTE1 does supply is the duality the INT8 kernel relied on, and it
// survives the widening intact for the *page*:
//
//   * Nz -> Nz of the group slice gives A[t][i] = x[t][G*32 + i];
//   * Zn -> Zn of the SAME L1 bytes gives B[i][t] = x[t][G*32 + i].
//
// Both readings walk the group's two sub-fractals in ascending channel order,
// because the page's Nz image has M1 = 1 and the Zn reading of a (32, 16)
// operand has N1 = 1, so in each case the only fractal-grid axis is the
// channel axis. No transpose, no permutation, two byte-identity loads.
//
// The operand this does NOT fix is the 32 x 32 resident rotation block, whose
// row-major GM/L1 image stops being its own Nz image once C0 drops to 16. The
// loads below take an L1 image that is already correct; producing it is
// `routines/routine_stage_page_zn_bf16.hpp`'s `stage_blocks_nz`, at ingress,
// which is the only stage on this path that can reorder sub-fractals.
//
// Contract
// --------
//   * `l1_byte_base` is a raw L1 byte address and must be 32-byte aligned.
//   * The L0A/L0B destination pointers are typed, so the offsets added to
//     them count elements, and the resulting address must be 512-byte
//     aligned. Both are static_asserted where the value is a constant and
//     stated here where it is not.
//   * These are leaves: they dispatch MTE1 work and issue no HardEvent. The
//     caller owns MTE2_MTE1 in front and MTE1_M behind
//     (MICROROUTINE_DESIGN_GUIDELINES.md section 3.4).

#include <cstdint>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#endif

namespace ascend_micro {

constexpr std::uint32_t kZnBf16C0 = 16u;
constexpr std::uint32_t kZnBf16ElemBytes = 2u;

#if defined(__CCE_AICORE__)

// ------------------------------------------------------- page group, A role
// L1 Nz(Tokens, Channels) sliced at one Hadamard group -> L0A Nz(Tokens,
// Group). A[t][i] = x[t][G*Group + i].
template <std::uint32_t Tokens, std::uint32_t Group>
__aicore__ inline void load_page_group_a_nz(std::uint32_t l1_byte_base,
                                            __ca__ bfloat16_t* dst)
{
    using namespace AscendC::Te;
    static_assert(Tokens % kZnBf16C0 == 0u, "the M axis must tile the 16-row fractal");
    static_assert(Group % kZnBf16C0 == 0u,
                  "a Hadamard group must be a whole number of 16-wide fractals");

    auto* src_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = MakeTensor(MakeMemPtr(src_ptr),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Tokens, Group));
    auto d = MakeTensor(MakeMemPtr(dst),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Tokens, Group));
    Copy(MakeCopy(CopyL12L0A{}), d, s);
}

// ------------------------------------------------------- page group, B role
// The same L1 bytes, read as Zn(Group, Tokens) -> L0B. B[i][t] = x[t][G*Group
// + i]. This is the load the BF16 port was expected to break; it does not,
// and the probe in tests/camodel/test_micro_layout_bf16.cpp is what settles
// that rather than this comment.
template <std::uint32_t Tokens, std::uint32_t Group>
__aicore__ inline void load_page_group_b_zn(std::uint32_t l1_byte_base,
                                            __cb__ bfloat16_t* dst)
{
    using namespace AscendC::Te;
    static_assert(Tokens % kZnBf16C0 == 0u, "the N axis must tile the 16-row fractal");
    static_assert(Group % kZnBf16C0 == 0u,
                  "a Hadamard group must be a whole number of 16-wide fractals");

    auto* src_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = MakeTensor(MakeMemPtr(src_ptr),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(Group, Tokens));
    auto d = MakeTensor(MakeMemPtr(dst),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(Group, Tokens));
    Copy(MakeCopy(CopyL12L0B{}), d, s);
}

// --------------------------------------------- resident block, both roles
// `l1_byte_base` must address the Nz(Dim, Dim) image of the block, not its
// row-major image. At 8 bits those are the same bytes; at 16 bits they are
// not, and the Nz image is what `stage_blocks_nz` produces.
//
// One image serves both roles. A wants Nz(Dim, Dim) of W; B wants
// Zn(Dim, Dim) of B[k][n] = W[n][k], and
//
//   Nz: offset k1*Dim*16 + m1*256 + m0*16 + k0  <- W[m1*16 + m0][k1*16 + k0]
//   Zn: offset k1*Dim*16 + n1*256 + n0*16 + k0  <- W[n1*16 + n0][k1*16 + k0]
//
// are the same map with the second index renamed. Both loads are therefore
// byte identities and neither needs a transposing form.
template <std::uint32_t Dim>
__aicore__ inline void load_block_a_nz(std::uint32_t l1_byte_base, __ca__ bfloat16_t* dst)
{
    using namespace AscendC::Te;
    static_assert(Dim % kZnBf16C0 == 0u, "a resident block must tile the fractal");
    static_assert((Dim * Dim * kZnBf16ElemBytes) % 512u == 0u,
                  "a resident block must stay 512-byte aligned in L0A");

    auto* src_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = MakeTensor(MakeMemPtr(src_ptr),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Dim, Dim));
    auto d = MakeTensor(MakeMemPtr(dst),
                        MakeFrameLayout<NZLayoutPtn, bfloat16_t>(Dim, Dim));
    Copy(MakeCopy(CopyL12L0A{}), d, s);
}

template <std::uint32_t Dim>
__aicore__ inline void load_block_b_zn(std::uint32_t l1_byte_base, __cb__ bfloat16_t* dst)
{
    using namespace AscendC::Te;
    static_assert(Dim % kZnBf16C0 == 0u, "a resident block must tile the fractal");
    static_assert((Dim * Dim * kZnBf16ElemBytes) % 512u == 0u,
                  "a resident block must stay 512-byte aligned in L0B");

    auto* src_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(l1_byte_base);
    auto s = MakeTensor(MakeMemPtr(src_ptr),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(Dim, Dim));
    auto d = MakeTensor(MakeMemPtr(dst),
                        MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(Dim, Dim));
    Copy(MakeCopy(CopyL12L0B{}), d, s);
}

#endif  // __CCE_AICORE__

}  // namespace ascend_micro
