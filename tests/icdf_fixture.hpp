#pragma once

// Fixture and oracle for the inverse-CDF int4x2 quantization probes.
//
// Shared by tests/test_icdf_quantize_probe.cpp (Tier 1) and
// tests/camodel/test_camodel_icdf_quantize.cpp (Tier 2) so both check against
// the same bytes.
//
// Oracle. Every FP16/BF16 input is a dyadic rational, so the scaled sigmoid is
// evaluated exactly in double and rounded once to the int4 lattice with
// round-to-nearest-even, saturating to [-8, 7]. The fixture admits only values
// whose exact y keeps a margin of at least kMinBoundaryMargin from every
// k+0.5 rounding boundary, so the documented <= 1 ULP deviation of the
// hardware vexp/vdiv (and up to 2 ULP of modelled slop, see the sensitivity
// check in the Tier 1 probe) cannot flip a nibble: Tier 2 is then expected to
// match the oracle byte-for-byte, with zero tolerance budget consumed.
//
// Packing order matches routine_vector_icdf_quantize.hpp: byte j holds element
// 2j in bits [3:0] and element 2j+1 in bits [7:4] (low nibble first), the same
// order the dequant4 leaf consumes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../routines/routine_vector_icdf_quantize.hpp"

namespace icdf {

// Tile geometry derives from the routine header so kernel and fixture cannot
// drift apart.
constexpr std::uint32_t kTileRepeats = ascend_micro::kIcdfTileRepeats;
constexpr std::uint32_t kTileElems =
    kTileRepeats * ascend_micro::kIcdfInLanesPerRepeat;             // 1024 inputs
constexpr std::uint32_t kTileInBytes = kTileElems * 2;               // 2048 B in
constexpr std::uint32_t kTileOutBytes =
    kTileRepeats * ascend_micro::kIcdfPackedBytesPerRepeat;          // 512 B packed out
constexpr float kMinBoundaryMargin = 0.0625f;                        // 1/16, dyadic

inline float bits_f32(std::uint32_t v)
{
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}
inline std::uint32_t f32_bits(float v)
{
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// Float -> FP16 bits, RNE, full range incl. subnormals/inf.
inline std::uint16_t f2h(float v)
{
    const std::uint32_t raw = f32_bits(v);
    const std::uint16_t sign = static_cast<std::uint16_t>((raw >> 16) & 0x8000u);
    const std::int32_t exp32 = static_cast<std::int32_t>((raw >> 23) & 0xffu) - 127;
    const std::uint32_t man = raw & 0x7fffffu;
    if (((raw >> 23) & 0xffu) == 0xffu) return static_cast<std::uint16_t>(sign | 0x7c00u);  // inf/nan
    if (exp32 > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);                      // overflow -> inf
    if (exp32 >= -14) {                                                                     // normal
        const std::int32_t e16 = exp32 + 15;
        const std::uint32_t rounded = man + 0xfffu + ((man >> 13) & 1u);
        std::uint32_t m16 = rounded >> 13;
        std::uint32_t e = static_cast<std::uint32_t>(e16);
        if (m16 & 0x400u) { m16 = 0; e += 1; }  // mantissa all-ones rounded up: next power of two
        if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | (e << 10) | (m16 & 0x3ffu));
    }
    // subnormal in half: significand = man | implicit 1, shifted.
    const std::uint32_t sig = man | 0x800000u;
    const std::int32_t shift = -14 - exp32 + 13;                                            // bits to drop from 24-bit sig
    if (shift > 24) return sign;                                                            // -> 0
    const std::uint32_t dropped = sig & ((1u << shift) - 1u);
    const std::uint32_t half_ulp = 1u << (shift - 1);
    std::uint32_t m = sig >> shift;
    if (dropped > half_ulp || (dropped == half_ulp && (m & 1u))) m += 1;
    return static_cast<std::uint16_t>(sign | m);                                            // may carry into normal min
}
inline float h2f(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t e = (h >> 10) & 0x1fu;
    const std::uint32_t m = h & 0x3ffu;
    if (e == 31) return bits_f32(sign | 0x7f800000u | (m << 13));
    if (e == 0) {
        if (m == 0) return bits_f32(sign);
        // subnormal: value = m * 2^-24
        return bits_f32(sign) + static_cast<float>(m) * 5.9604644775390625e-08f;  // m * 2^-24, exact in fp32
    }
    return bits_f32(sign | ((e + 112u) << 23) | (m << 13));
}
// Float -> BF16 bits, RNE.
inline std::uint16_t f2bf(float v)
{
    const std::uint32_t raw = f32_bits(v);
    const std::uint16_t r = static_cast<std::uint16_t>(raw >> 16);
    const std::uint16_t rounded =
        static_cast<std::uint16_t>((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
    return ((r & 0x7f80u) == 0x7f80u) ? r : rounded;  // keep inf/nan payload
}
inline float bf2f(std::uint16_t b)
{
    return bits_f32(static_cast<std::uint32_t>(b) << 16);
}

struct Fixture {
    bool bf16 = false;
    std::uint32_t tiles = 0;
    std::vector<std::uint16_t> in_bits;                  // tiles * kTileElems words
    std::vector<std::uint8_t> oracle_packed;             // tiles * kTileOutBytes bytes
    std::vector<float> margins;                          // per element, for assertions
    std::uint32_t subnormal_count = 0;
    std::uint32_t saturating_count = 0;                  // |y| rounds to 7
};

inline std::uint16_t encode(bool bf16, float v) { return bf16 ? f2bf(v) : f2h(v); }
inline float decode(bool bf16, std::uint16_t bits) { return bf16 ? bf2f(bits) : h2f(bits); }

inline int quantize_rne_sat(double y)
{
    double f = std::floor(y);
    double r = y - f;
    long long q;
    if (r > 0.5) q = static_cast<long long>(f) + 1;
    else if (r < 0.5) q = static_cast<long long>(f);
    else q = (static_cast<long long>(f) % 2 == 0) ? static_cast<long long>(f)
                                                  : static_cast<long long>(f) + 1;
    if (q > 7) q = 7;
    if (q < -8) q = -8;
    return static_cast<int>(q);
}

inline double sigmoid_scaled(double x)
{
    return 14.0 / (1.0 + std::exp(-x)) - 7.0;
}

// Deterministic 16-bit LCG so host probe and runner agree bit-for-bit.
struct Lcg {
    std::uint32_t state;
    explicit Lcg(std::uint32_t seed) : state(seed ? seed : 1u) {}
    std::uint32_t next() { state = state * 1664525u + 1013904223u; return state >> 8; }
};

// Anchor values: subnormal cluster members, zero, dyadic mid-range steps and
// both saturation tails (including |x| beyond the half exp overflow point so
// the inf/FTZ limbs of the chain run every tile). All are dyadic and verified
// against the margin filter at fixture build time.
inline const float* anchors(std::size_t& count)
{
    static const float kAnchors[] = {
        0.0f, 0.0f,
        5.9604645e-08f, -5.9604645e-08f,                    // +-2^-24, FP16 subnormal
        2.9802322e-07f, -2.9802322e-07f,                    // subnormal cluster
        1.1920929e-07f, -1.1920929e-07f,
        0.000244140625f, -0.000244140625f,                  // 2^-12
        0.0625f, -0.0625f, 0.5f, -0.5f, 1.0f, -1.0f,        // dyadic steps
        2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f,
        5.0f, -5.0f, 6.0f, -6.0f, 8.0f, -8.0f, 10.0f, -10.0f,
        11.0f, -11.0f,   // saturation tail; exp(11)=59874 stays finite in
                         // half, unlike exp(12)/exp(100) whose +inf trips
                         // CAModel's vec_err_idata_inf_nan input-data check.
                         // The inf limb (x <= -11) is covered by the Tier 1
                         // mock only; see reports/icdf_quantize.md.
        9.8f, -9.8f,     // vexp subnormal-output FTZ limb (exp(-9.8) < 2^-14)
        9.1835496206239346e-41f, -9.1835496206239346e-41f,  // 2^-133, BF16 subnormal
        1.4702718316940622e-39f, -1.4702718316940622e-39f,  // 2^-129, BF16 subnormal
    };
    count = sizeof(kAnchors) / sizeof(kAnchors[0]);
    return kAnchors;
}

inline Fixture make_fixture(bool bf16, std::uint32_t tiles)
{
    Fixture fx;
    fx.bf16 = bf16;
    fx.tiles = tiles;
    fx.in_bits.resize(static_cast<std::size_t>(tiles) * kTileElems);
    fx.oracle_packed.resize(static_cast<std::size_t>(tiles) * kTileOutBytes);
    fx.margins.resize(static_cast<std::size_t>(tiles) * kTileElems);

    std::size_t anchor_count = 0;
    const float* pa = anchors(anchor_count);
    for (std::uint32_t t = 0; t < tiles; ++t) {
        Lcg rng(0x9e3779b9u + 2654435761u * (t + 1));
        for (std::uint32_t i = 0; i < kTileElems; ++i) {
            // Every third lane is a random dyadic k/1024 in [-10, 10]; the
            // rest rotate through the anchor palette with a per-tile phase so
            // each tile's distribution is distinct.
            float v;
            const std::uint32_t sel = rng.next() % 3u;
            if (sel == 0) {
                const std::int32_t k =
                    static_cast<std::int32_t>(rng.next() % 20481u) - 10240;  // k/1024
                v = static_cast<float>(k) / 1024.0f;
            } else {
                v = pa[(i * 7u + t * 11u + 3u) % anchor_count];
            }
            const std::uint16_t bits = encode(bf16, v);
            const double x = static_cast<double>(decode(bf16, bits));
            const double y = sigmoid_scaled(x);
            // RNE-to-int boundaries are the half-integers; the distance that
            // matters is to the *nearest* one: |frac(y) - 0.5|.
            const double margin = std::fabs(y - std::floor(y) - 0.5);
            if (margin < kMinBoundaryMargin) {
                // Too close to a rounding boundary: replace with a verified
                // anchor so the bit-exact claim keeps holding.
                const std::uint32_t idx = (i + t + 5u) % anchor_count;
                v = pa[idx];
            }
            const std::uint16_t final_bits = encode(bf16, v);
            const double xf = static_cast<double>(decode(bf16, final_bits));
            const double yf = sigmoid_scaled(xf);
            const double marginf = std::fabs(yf - std::floor(yf) - 0.5);
            if (marginf < kMinBoundaryMargin)
                throw std::runtime_error("fixture anchor inside margin band");
            const std::uint32_t flat = t * kTileElems + i;
            fx.in_bits[flat] = final_bits;
            fx.margins[flat] = static_cast<float>(marginf);
            const double min_normal =
                bf16 ? 1.1754943508222875e-38 : 6.103515625e-05;  // 2^-126 vs 2^-14
            const bool is_subnormal = std::fabs(xf) > 0.0 && std::fabs(xf) < min_normal;
            if (is_subnormal) ++fx.subnormal_count;
            // "tail" = inputs whose quantized code is the extreme +-7 (the
            // sigmoid saturates; for |x| >= 5.3 the code is 7).
            if (std::fabs(sigmoid_scaled(xf)) >= 6.5) ++fx.saturating_count;
        }
        // Pack the oracle for this tile, low nibble first.
        for (std::uint32_t b = 0; b < kTileOutBytes; ++b) {
            const int q0 = quantize_rne_sat(sigmoid_scaled(
                static_cast<double>(decode(bf16, fx.in_bits[t * kTileElems + 2 * b]))));
            const int q1 = quantize_rne_sat(sigmoid_scaled(
                static_cast<double>(decode(bf16, fx.in_bits[t * kTileElems + 2 * b + 1]))));
            fx.oracle_packed[static_cast<std::size_t>(t) * kTileOutBytes + b] =
                static_cast<std::uint8_t>((q1 << 4) | (q0 & 0xf));
        }
    }
    // Guard band beyond the payload stays zero; runner guards it separately.
    return fx;
}

inline std::uint64_t tile_checksum(const Fixture& fx, std::uint32_t t)
{
    std::uint64_t sum = 1469598103934665603ull;
    for (std::uint32_t b = 0; b < kTileOutBytes; ++b)
        sum = (sum ^ fx.oracle_packed[static_cast<std::size_t>(t) * kTileOutBytes + b]) * 1099511628211ull;
    return sum;
}

// Properties the fixture must hold (guidelines 6.4: assert, do not describe).
inline void assert_properties(const Fixture& fx)
{
    if (fx.subnormal_count == 0)
        throw std::runtime_error("fixture covers no subnormal inputs");
    if (fx.saturating_count == 0)
        throw std::runtime_error("fixture covers no saturating tail inputs");
    float min_margin = 1e9f;
    for (float m : fx.margins) min_margin = std::fmin(min_margin, m);
    if (min_margin < kMinBoundaryMargin) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "fixture min margin %.5f below %.5f",
                      static_cast<double>(min_margin), static_cast<double>(kMinBoundaryMargin));
        throw std::runtime_error(buf);
    }
    for (std::uint32_t a = 0; a < fx.tiles; ++a)
        for (std::uint32_t b = a + 1; b < fx.tiles; ++b)
            if (tile_checksum(fx, a) == tile_checksum(fx, b))
                throw std::runtime_error("fixture tiles are interchangeable");
}

}  // namespace icdf
