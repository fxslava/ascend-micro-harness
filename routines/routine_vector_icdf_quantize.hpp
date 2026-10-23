#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

// Inverse-CDF vector quantization leaf: FP16/BF16 -> packed signed int4x2.
//
// Transform. The upstream Fixpipe QF322B16 export leaves the KV block clustered
// on subnormal magnitudes. The inverse CDF of the logistic distribution is the
// sigmoid, so the alignment stage is the scaled sigmoid
//
//     y = 14 / (1 + exp(-x)) - 7 = 14 * sigmoid(x) - 7,
//
// which maps x -> [-7, 7] monotonically and sends the subnormal cluster around
// zero to the centre of the uniform output range (y(0) = 0). 14 and 7 are
// exactly representable in half, and the form 14/(1+e) - 7 folds the output
// scale into the division numerator, so the per-lane chain is five vector
// instructions after the negate:
//
//     vneg  e = -x
//     vexp  e = exp(-x)            (Exp INTRINSIC: one instruction, FTZ)
//     vadds d = e + 1.0h
//     vdiv  r = 14.0h / d          (numerator is a hoisted vdup broadcast)
//     vadds y = r - 7.0h
//
// Range safety: x > 11 gives exp(-x) -> 0 (FTZ) => y = +7; x < -11 gives
// exp(-x) -> +inf => r = 14/inf = 0 => y = -7. The e^x/(1+e^x) form would
// produce inf/inf = NaN there, which is why this direction is used.
//
// Cast and packing. Cast<half -> int4x2_t> is the single vcvt_f162s4
// instruction, and on dav-3510 it packs ADJACENT source pairs natively: with
// RegLayout::ZERO the converted values of source lanes 2k and 2k+1 land in
// nibbles 0 and 1 of the k-th b32-sized destination group, i.e. one packed
// byte per two input lanes, low nibble first -- the exact order the dequant4
// leaf consumes. (Measured on CAModel: a cast at RegLayout::ONE puts the pair
// in nibbles 4..5, outside the window a compression store reads, so a
// layout-ZERO + layout-ONE pair of casts on de-interleaved registers loses the
// odd stream entirely. One cast on the contiguous stream is both simpler and
// correct.) The compression store is StoreDist::DIST_PACK4_B32, which writes
// the low byte of every b32 group contiguously: 128 input lanes -> 64 dense
// int4x2 bytes. No manual shifts, no UB round-trip between stages.
//
// Zero-scalar invariant (guidelines section 1): every stage here is a
// Reg-level vector instruction. No ReduceSum, no GetValue, no TPipe, no event
// IDs; ordering is entirely caller-owned.

namespace ascend_micro {

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#define ASCEND_MICRO_ICDF_INLINE __aicore__ inline
#else
#define ASCEND_MICRO_ICDF_INLINE inline
#endif

// One VF iteration consumes 2 * VECTOR_REG_WIDTH input bytes (256 even + 256
// odd lanes interleaved) and emits VECTOR_REG_WIDTH / 2 packed bytes.
constexpr std::uint32_t kIcdfVecReg = 256;                           // VECTOR_REG_WIDTH, dav-3510
constexpr std::uint32_t kIcdfLanes = kIcdfVecReg / 2;                // 128 half lanes per register
constexpr std::uint32_t kIcdfInLanesPerRepeat = kIcdfLanes;          // 128 elements
constexpr std::uint32_t kIcdfPackedBytesPerRepeat = kIcdfLanes / 2;  // 64 B out per repeat
// Default VF iterations per tile; the CAModel wrapper and the host fixture
// both derive their sizes from these four constants. 8 repeats keep one tile
// at 2048 B in / 512 B out.
constexpr std::uint32_t kIcdfTileRepeats = 8;

// Scaled-sigmoid constants, both exact in half.
constexpr float kIcdfScale = 14.0f;   // y = scale * sigmoid(x) - shift, range [-7, 7]
constexpr float kIcdfShift = 7.0f;

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)

// Nibble-position traits for the two int4x2 casts. Saturation is configured on
// the instruction (SatMode::SAT); on Ascend 950PR the CTRL[60]/CTRL[59]
// reset default already selects global saturation for float->integer
// conversion, so the trait is consistent with, not dependent on, CTRL state.
// CAST_RINT = round to nearest, ties to even, matching the oracle.
static constexpr AscendC::Reg::CastTrait kIcdfCastNibbleLo = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
// bfloat16 -> half widening ahead of the chain; SAT clamps a BF16 outside the
// half range to +/-65504 instead of producing inf that would poison vexp.
static constexpr AscendC::Reg::CastTrait kIcdfBf16ToHalf = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

// The whole transform for `repeat` VF iterations, all under one full-lane
// mask: 256-byte continuous, unmasked loads and stores throughout.
// Input is 2 * kIcdfLanes elements per repeat; output is kIcdfLanes packed
// bytes (two int4 values per byte).
template <class Input>
__simd_vf__ inline void icdf_quantize_vf(
    __ubuf__ Input* inAddr, __ubuf__ int4x2_t* outAddr, std::uint16_t repeats)
{
    AscendC::Reg::MaskReg mask =
        AscendC::Reg::CreateMask<half, AscendC::Reg::MaskPattern::ALL>();
    // Loop-invariant numerator: 14.0h broadcast once per VF entry. vdup keeps
    // the constant in the vector pipe; no UB operand, no scalar trip.
    AscendC::Reg::RegTensor<half> numReg;
    AscendC::Reg::Duplicate(numReg, static_cast<half>(kIcdfScale), mask);
    for (std::uint16_t i = 0; i < repeats; ++i) {
        __ubuf__ Input* in = inAddr + i * kIcdfInLanesPerRepeat;
        __ubuf__ int4x2_t* out = outAddr + i * kIcdfPackedBytesPerRepeat;
        AscendC::Reg::RegTensor<half> x;
        if constexpr (std::is_same_v<Input, bfloat16_t>) {
            AscendC::Reg::RegTensor<bfloat16_t> xBf;
            AscendC::Reg::LoadAlign<bfloat16_t,
                                    AscendC::Reg::LoadDist::DIST_NORM>(xBf, in);
            AscendC::Reg::Cast<half, bfloat16_t, kIcdfBf16ToHalf>(x, xBf, mask);
        } else {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_NORM>(x, in);
        }
        // Sigmoid chain, register-resident, in place.
        AscendC::Reg::Neg<half>(x, x, mask);
        AscendC::Reg::Exp<half>(x, x, mask);
        AscendC::Reg::Adds<half, half, AscendC::Reg::MaskMergeMode::ZEROING>(
            x, x, static_cast<half>(1.0f), mask);
        AscendC::Reg::Div<half>(x, numReg, x, mask);
        AscendC::Reg::Adds<half, half, AscendC::Reg::MaskMergeMode::ZEROING>(
            x, x, static_cast<half>(-kIcdfShift), mask);
        // One cast packs adjacent source pairs into one byte each (layout
        // ZERO: lane 2k -> nibble 0, lane 2k+1 -> nibble 1 of the k-th b32
        // group); PACK4_B32 stores those group low bytes contiguously.
        AscendC::Reg::RegTensor<int4x2_t> s4;
        AscendC::Reg::Cast<int4x2_t, half, kIcdfCastNibbleLo>(s4, x, mask);
        AscendC::Reg::StoreAlign<int4x2_t,
                                 AscendC::Reg::StoreDist::DIST_PACK4_B32>(out, s4, mask);
    }
}

#endif  // native

#undef ASCEND_MICRO_ICDF_INLINE
}  // namespace ascend_micro
