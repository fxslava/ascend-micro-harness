#pragma once

#include <cstdint>
#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
#include "kernel_operator.h"
#endif

// N(0,1) inputs, already scaled by the caller after Hadamard. No norm or
// variance estimation is hidden here. Finite inputs only; equality goes to
// the upper bin. Codes are unsigned, packed low bits first in lane order.
namespace ascend_micro {
template <unsigned Bits> struct LloydMaxCodebook;
template <> struct LloydMaxCodebook<2> {
    inline static constexpr float centroids[4] = {
        -1.510418f, -0.452794f, 0.452794f, 1.510418f};
    inline static constexpr float thresholds[3] = {-0.981606f, 0.f, 0.981606f};
};
template <> struct LloydMaxCodebook<4> {
    // Gaussian conditional-mean fixed point; midpoint thresholds. See Tier 1.
    inline static constexpr float centroids[16] = {
        -2.732589571f, -2.069017227f, -1.618046386f, -1.256231197f,
        -0.942340456f, -0.656759119f, -0.388048299f, -0.128395030f,
         0.128395030f,  0.388048299f,  0.656759119f,  0.942340456f,
         1.256231197f,  1.618046386f,  2.069017227f,  2.732589571f};
    inline static constexpr float thresholds[15] = {
        -2.400803399f, -1.843531806f, -1.437138792f, -1.099285827f,
        -0.799549788f, -0.522403709f, -0.258221665f, 0.f,
         0.258221665f,  0.522403709f,  0.799549788f, 1.099285827f,
         1.437138792f,  1.843531806f,  2.400803399f};
};

template <unsigned Bits> inline unsigned lloydmax_code(float x)
{
    static_assert(Bits == 2 || Bits == 4);
    unsigned q = 0;
    for (float t : LloydMaxCodebook<Bits>::thresholds) q += x >= t;
    return q;
}

enum class LloydMaxTable : unsigned { Immediate, UbHeader };
constexpr unsigned kLloydMaxVectorBytes = 256;
constexpr unsigned kLloydMaxHeaderBytes = 512;
// Live FP32 centroids occupy at most 64 B. One spare DataBlock before the
// thresholds gives distinct bank phases; both unmasked reads stay in header.
constexpr unsigned kLloydMaxThresholdWord = 24; // (64 + 32) / sizeof(float)
static_assert(kLloydMaxThresholdWord * 4 % 32 == 0);
static_assert(kLloydMaxThresholdWord * 4 % 256 != 0);
static_assert(kLloydMaxThresholdWord * 4 + 256 <= kLloydMaxHeaderBytes);

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace lloydmax_detail {
namespace R = AscendC::Reg;
using F = R::RegTensor<float>;
using U = R::RegTensor<std::uint32_t>;
using I = R::RegTensor<std::int32_t>;
using M = R::MaskReg;

// Compile-time immediates only, never a device pointer to a C++ constant array.
// Unused lanes contain zero, so both padded header loads are fully defined.
template <unsigned Bits, bool Threshold, unsigned Lane = 0, class Book = LloydMaxCodebook<Bits>>
__simd_callee__ inline void table(F& out, I& lane, M& all)
{
    constexpr unsigned count = (1u << Bits) - unsigned(Threshold);
    if constexpr (Lane == 0) R::Duplicate(out, 0.f, all);
    F value;
    M selected;
    constexpr float c = Threshold ? Book::thresholds[Lane] : Book::centroids[Lane];
    R::Duplicate(value, c, all);
    R::Compares<std::int32_t, AscendC::CMPMODE::EQ>(selected, lane, int(Lane), all);
    R::Select(out, value, out, selected);
    if constexpr (Lane + 1 < count) table<Bits, Threshold, Lane + 1, Book>(out, lane, all);
}

template <unsigned Step>
__simd_callee__ inline void search(F& x, F& thresholds, U& q, M& all)
{
    U idx, next;
    F threshold;
    M ge;
    R::Adds(idx, q, Step - 1u, all);
    R::Gather(threshold, thresholds, idx);
    R::Compare<float, AscendC::CMPMODE::GE>(ge, x, threshold, all);
    R::Adds(next, q, Step, all);
    R::Select(q, next, q, ge);
    if constexpr (Step > 1) search<Step / 2>(x, thresholds, q, all);
}

template <unsigned Bits>
__simd_callee__ inline void quantize(F& x, F& thresholds, U& q, M& all)
{
    if constexpr (Bits == 2) {
        // Exactly two data comparisons: selected threshold and sign. The asymmetric
        // tie on the negative threshold must go to the inner (upper) bin.
        U inner, outer, pos, neg;
        M positive, far;
        R::Compares<float, AscendC::CMPMODE::GE>(positive, x, 0.f, all);
        // A third comparison at -t is avoided by using sign-conditioned t:
        // compare x against {-t,+t}, then select the two signed bin pairs.
        F tp, tn, t;
        R::Duplicate(tp, 0.981606f, all);
        R::Duplicate(tn, -0.981606f, all);
        R::Select(t, tp, tn, positive);
        R::Compare<float, AscendC::CMPMODE::GE>(far, x, t, all);
        R::Duplicate(inner, 2u, all);
        R::Duplicate(outer, 3u, all);
        R::Select(pos, outer, inner, far);
        R::Duplicate(inner, 1u, all);
        R::Duplicate(outer, 0u, all);
        R::Select(neg, inner, outer, far);
        R::Select(q, pos, neg, positive);
    } else {
        R::Duplicate(q, 0u, all);
        search<8>(x, thresholds, q, all);
    }
}

// A register reduction tree concatenates adjacent codes into b32 words.
// After stride S, lanes divisible by 2*S contain 2*S codes in order. Other
// lanes are irrelevant; all reads remain inside the register (Gather wraps).
// Three/four stages replace eight/sixteen individual-code Gather operations.
template <unsigned Bits, unsigned Stride = 1>
__simd_callee__ inline void pack_tree(U& q, U& lane, M& all)
{
    U idx, value;
    R::Adds(idx, lane, Stride, all);
    R::Gather(value, q, idx);
    R::ShiftLefts(value, value, std::int16_t(Stride * Bits), all);
    R::Or(q, q, value, all);
    if constexpr (Stride * 2 < 32u / Bits) pack_tree<Bits, Stride * 2>(q, lane, all);
}
template <unsigned Bits>
__simd_callee__ inline void pack_pair(__ubuf__ std::uint32_t* dst,
                                    U& q0, U& q1, I& lane, M& all)
{
    U base, p0, p1, low_bits;
    auto& ulane = reinterpret_cast<U&>(lane);
    R::Duplicate(low_bits, Bits == 2 ? 3u : 7u, all);
    R::And(base, ulane, low_bits, all);
    R::ShiftLefts(base, base, std::int16_t(Bits == 2 ? 4 : 3), all);
    pack_tree<Bits>(q0, ulane, all);
    pack_tree<Bits>(q1, ulane, all);
    R::Gather(p0, q0, base);
    R::Gather(p1, q1, base);
    std::uint32_t count = 8;
    M store = R::UpdateMask<std::uint32_t>(count);
    if constexpr (Bits == 2) {
        M low;
        R::Compares<std::int32_t, AscendC::CMPMODE::LT>(low, lane, 4, all);
        R::Select(p0, p0, p1, low);
        R::StoreAlign(dst, p0, store);
    } else {
        R::StoreAlign(dst, p0, store);
        R::StoreAlign(dst + 8, p1, store);
    }
}
} // namespace lloydmax_detail

// Decode an unsigned register of codes. The table is already in a register;
// no address generation or memory access is part of this single Gather.
__simd_callee__ inline void lloydmax_dequantize_reg(
    lloydmax_detail::F& dst, lloydmax_detail::F& centroids,
    lloydmax_detail::U& code)
{
    AscendC::Reg::Gather(dst, centroids, code);
}

// Scheme B: caller reserves 512 B, aligned to 32 B, with full read padding.
// One vector-only initialization, zero DMA and zero GM constant workspace.
template <unsigned Bits>
__simd_vf__ inline void lloydmax_init_header_vf(__ubuf__ float* header)
{
    using namespace lloydmax_detail;
    M all = R::CreateMask<float, R::MaskPattern::ALL>();
    I lane; F c, t;
    R::Arange(lane, 0);
    table<Bits, false>(c, lane, all);
    table<Bits, true>(t, lane, all);
    // Masked stores write only the live rounded DataBlocks. Padded loads may
    // overlap adjacent slots, as A3 permits, but writes never overlap them.
    std::uint32_t count = Bits == 2 ? 8u : 16u;
    M live = R::UpdateMask<float>(count);
    R::StoreAlign(header, c, live);
    R::StoreAlign(header + kLloydMaxThresholdWord, t, live);
}

// Fused quantize + decode + packed-code export. Standalone dequantization can
// call the one-instruction Gather below after unpacking the caller's codes.
// Each iteration: 128 FP32 in, 128 FP32 reconstructed, 32/64 packed bytes.
// All pointers 32 B aligned; in/recon have whole 256 B registers accessible.
template <unsigned Bits, LloydMaxTable Mode = LloydMaxTable::Immediate>
__simd_vf__ inline void lloydmax_quantize_vf(
    __ubuf__ float* in, __ubuf__ float* recon, __ubuf__ std::uint32_t* packed,
    __ubuf__ float* header, std::uint16_t pairs)
{
    static_assert(Bits == 2 || Bits == 4);
    using namespace lloydmax_detail;
    M all = R::CreateMask<float, R::MaskPattern::ALL>();
    I lane; F c, t;
    R::Arange(lane, 0);
    if constexpr (Mode == LloydMaxTable::UbHeader) {
        R::LoadAlign(c, header);
        if constexpr (Bits == 4) R::LoadAlign(t, header + kLloydMaxThresholdWord);
    } else {
        table<Bits, false>(c, lane, all);
        if constexpr (Bits == 4) table<Bits, true>(t, lane, all);
    }
    for (std::uint16_t i = 0; i < pairs; ++i) {
        F x0, x1, y0, y1; U q0, q1;
        R::LoadAlign(x0, in + i * 128);
        R::LoadAlign(x1, in + i * 128 + 64);
        quantize<Bits>(x0, t, q0, all);
        quantize<Bits>(x1, t, q1, all);
        lloydmax_dequantize_reg(y0, c, q0); // one RV_VSELR per 64 FP32 lanes
        lloydmax_dequantize_reg(y1, c, q1);
        R::StoreAlign(recon + i * 128, y0, all);
        R::StoreAlign(recon + i * 128 + 64, y1, all);
        pack_pair<Bits>(packed + i * (4 * Bits), q0, q1, lane, all);
    }
}

// Standalone packed-code decoder. A3: each packed + i*(4*Bits) word base
// needs 256 accessible bytes even though only 32/64 bytes hold live codes.
template <unsigned Bits, LloydMaxTable Mode = LloydMaxTable::Immediate>
__simd_vf__ inline void lloydmax_dequantize_vf(
    __ubuf__ std::uint32_t* packed, __ubuf__ float* recon,
    __ubuf__ float* header, std::uint16_t pairs)
{
    static_assert(Bits == 2 || Bits == 4);
    using namespace lloydmax_detail;
    M all = R::CreateMask<float, R::MaskPattern::ALL>();
    I lane; F c; U idx, shifts, maskbits, wordmask;
    R::Arange(lane, 0);
    auto& ulane = reinterpret_cast<U&>(lane);
    if constexpr (Mode == LloydMaxTable::UbHeader) R::LoadAlign(c, header);
    else table<Bits, false>(c, lane, all);
    R::ShiftRights(idx, ulane, std::int16_t(Bits == 2 ? 4 : 3), all);
    R::Duplicate(wordmask, 32u / Bits - 1u, all);
    R::And(shifts, ulane, wordmask, all);
    R::ShiftLefts(shifts, shifts, std::int16_t(Bits == 2 ? 1 : 2), all);
    R::Duplicate(maskbits, (1u << Bits) - 1u, all);
    U idx1;
    R::Adds(idx1, idx, 2u * Bits, all); // word offset of input lane 64
    for (std::uint16_t i = 0; i < pairs; ++i) {
        U words, q0, q1; F y0, y1;
        R::LoadAlign(words, packed + i * (4 * Bits));
        R::Gather(q0, words, idx); R::Gather(q1, words, idx1);
        // Variable-shift counts are signed b32 in CANN 9.2 (0..30 here).
        auto& signed_shifts = reinterpret_cast<I&>(shifts);
        R::ShiftRight<std::uint32_t, std::int32_t>(q0, q0, signed_shifts, all);
        R::ShiftRight<std::uint32_t, std::int32_t>(q1, q1, signed_shifts, all);
        R::And(q0, q0, maskbits, all); R::And(q1, q1, maskbits, all);
        lloydmax_dequantize_reg(y0, c, q0); lloydmax_dequantize_reg(y1, c, q1);
        R::StoreAlign(recon + i * 128, y0, all);
        R::StoreAlign(recon + i * 128 + 64, y1, all);
    }
}
#endif
} // namespace ascend_micro
