#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include "../../include/sim_micro_testbed.hpp"

namespace wta_test {
using namespace ascend_micro::sim;
using Tile = std::array<std::uint16_t, 256>;

// Task contract: W = (1 + alpha) I - (beta/16) 1 1^T, alpha = 0.25, beta = 0.4.
// L0B stores W as FP16, so every consumer uses the FP16-rounded entries:
// diagonal 1.224609375 (= 1254/1024) and off-diagonal -0.024993896484375
// (= -819/2^15). Both are exact dyadic rationals, so the closed-form double
// oracle below is exact on FP16 inputs and agrees with an exact product-sum
// over the same FP16 entries bit for bit. The real 1.225 / 0.025 never enter
// any comparison.
inline constexpr double kAlpha = 0.25;
inline constexpr double kBeta = 0.4;
inline double wta_diag()
{ return half_to_float(float_to_half_rne(float(1 + kAlpha - kBeta / 16))); }
inline double wta_off()
{ return half_to_float(float_to_half_rne(float(-kBeta / 16))); }

inline Tile weights()
{
    Tile w{};
    const auto diag = float_to_half_rne(float(1 + kAlpha - kBeta / 16));
    const auto off = float_to_half_rne(float(-kBeta / 16));
    for (int k = 0; k < 16; ++k)
        for (int col = 0; col < 16; ++col)
            w[k * 16 + col] = k == col ? diag : off;
    return w;
}

// Correct rounding at the FP16 subnormal underflow midpoint as well.
inline std::uint16_t half_rne(float x)
{
    if (std::fabs(x) < std::ldexp(1.0f, -24)) {
        const auto sign = std::uint16_t((float_bits(x) >> 16) & 0x8000);
        return sign | (std::fabs(x) > std::ldexp(1.0f, -25) ? 1 : 0);
    }
    return float_to_half_rne(x);
}

inline int round_even(double x)
{
    const double base = std::floor(x);
    return static_cast<int>(base) +
        ((x - base > 0.5) || (x - base == 0.5 && static_cast<int>(base) % 2));
}

inline int quantize(float raw, std::uint32_t scale_bits, bool is_signed)
{
    const float scale = bits_float(scale_bits & 0xffffe000u);
    const double value = std::max(0.0, double(raw)) * scale;
    const int ceiling = is_signed ? 127 : 255;
    if (value >= ceiling) return ceiling;
    return round_even(value);
}

// Direct random input: sixteen independent rows of sixteen standard-normal
// competitors, the winner raised to second + delta, rounded once to FP16 on
// storage. No CDF transform and no quantizer preprocessing; the stored margin
// can differ from the requested real margin by up to one FP16 ulp per side.
inline Tile fixture(unsigned seed, float delta, std::array<unsigned, 16>* winners = nullptr)
{
    std::mt19937 rng(seed);
    Tile x{};
    for (int row = 0; row < 16; ++row) {
        std::array<double, 16> z{};
        for (unsigned c = 0; c < 16; ++c) {
            const double u = (double(rng()) + .5) / 4294967296.0;
            const double v = (double(rng()) + .5) / 4294967296.0;
            z[c] = std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v);
        }
        const auto winner = std::max_element(z.begin(), z.end()) - z.begin();
        z[winner] = -INFINITY;
        const double second = *std::max_element(z.begin(), z.end());
        z[winner] = second + delta;
        if (winners) (*winners)[row] = static_cast<unsigned>(winner);
        for (unsigned c = 0; c < 16; ++c)
            x[row * 16 + c] = half_rne(static_cast<float>(z[c]));
    }
    return x;
}

struct Reference {
    std::array<std::array<float, 256>, 8> raw{};
    std::array<Tile, 8> feedback{};
    std::array<std::array<int, 256>, 8> quantized{};
};

// Independent closed-form inhibition oracle: raw_j = d*x_j + off*(S - x_j)
// per row, S the row sum, d/off the exact FP16 W entries, evaluated in double
// and rounded once to FP32 (the L0C value the Fixpipe consumes). Every
// product and sum of FP16 inputs fits in double exactly for this fixture
// domain, so the FP16 feedback and B8 rounding are deterministic.
inline Reference reference(const Tile& input, std::uint32_t scale, bool sign)
{
    Reference result;
    const double d = wta_diag(), off = wta_off();
    Tile v = input;
    for (int t = 0; t < 8; ++t) {
        for (int r = 0; r < 16; ++r) {
            double sum = 0;
            for (int c = 0; c < 16; ++c) sum += half_to_float(v[r * 16 + c]);
            for (int c = 0; c < 16; ++c) {
                const int i = r * 16 + c;
                const double x = half_to_float(v[i]);
                const float raw = static_cast<float>(d * x + off * (sum - x));
                result.raw[t][i] = raw;
                result.feedback[t][i] = half_rne(std::max(0.0f, raw));
                result.quantized[t][i] = quantize(raw, scale, sign);
            }
        }
        v = result.feedback[t];
    }
    return result;
}

// The same recurrence with each inner product accumulated sequentially in
// FP32 - one plausible Mmad accumulation order, not a claim about hardware.
// Diagnostic only: it bounds how far an alternate accumulation order can move
// the emitted bytes relative to the exact oracle, which justifies the Tier-2
// tolerance. CAModel, not this model, establishes the real device order.
inline Reference reference_seq32(const Tile& input, std::uint32_t scale, bool sign)
{
    Reference result;
    const auto w = weights();
    std::array<float, 256> b{};  // Zn operand: b[col * 16 + k]
    for (int col = 0; col < 16; ++col)
        for (int k = 0; k < 16; ++k) b[col * 16 + k] = half_to_float(w[k * 16 + col]);
    Tile v = input;
    for (int t = 0; t < 8; ++t) {
        for (int r = 0; r < 16; ++r)
            for (int col = 0; col < 16; ++col) {
                float sum = 0;
                for (int k = 0; k < 16; ++k)
                    sum += half_to_float(v[r * 16 + k]) * b[col * 16 + k];
                const int i = r * 16 + col;
                result.raw[t][i] = sum;
                result.feedback[t][i] = half_rne(std::max(0.0f, sum));
                result.quantized[t][i] = quantize(sum, scale, sign);
            }
        v = result.feedback[t];
    }
    return result;
}

struct Convergence {
    unsigned rows = 0, support_one = 0, literal_one = 0, zero = 0, multi = 0;
    unsigned wrong_winner = 0, input_ties = 0;
    void add(const Tile& x, const std::array<int, 256>& q,
             const std::array<unsigned, 16>* winners = nullptr)
    {
        for (int r = 0; r < 16; ++r) {
            int argmax = 0, survivor = -1, count = 0;
            for (int c = 0; c < 16; ++c) {
                if (half_to_float(x[r * 16 + c]) > half_to_float(x[r * 16 + argmax]))
                    argmax = c;
                if (q[r * 16 + c] != 0) { survivor = c; ++count; }
            }
            unsigned ties = 0;
            for (int c = 0; c < 16; ++c)
                if (x[r * 16 + c] == x[r * 16 + argmax]) ++ties;
            if (ties > 1) ++input_ties;
            if (winners) argmax = static_cast<int>((*winners)[r]);
            ++rows;
            if (!count) ++zero;
            else if (count > 1) ++multi;
            else if (survivor != argmax) ++wrong_winner;
            else {
                ++support_one;
                if (q[r * 16 + survivor] == 1) ++literal_one;
            }
        }
    }
};

// Empirical convergence rate of the recurrence itself: first step, of the
// first 64, at which a row has exactly one live competitor on the intended
// winner. Evaluated on the FP16 feedback chain, independent of quantization.
struct Stability {
    std::array<unsigned, 65> histogram{}; // 0: not stable by 64, t: first stable step.
    unsigned rows = 0;
    void add(Tile v, const std::array<unsigned, 16>& winners)
    {
        const double d = wta_diag(), off = wta_off();
        std::array<unsigned, 16> first{};
        for (unsigned t = 1; t <= 64; ++t) {
            Tile next{};
            for (unsigned r = 0; r < 16; ++r) {
                double sum = 0;
                for (unsigned c = 0; c < 16; ++c) sum += half_to_float(v[r * 16 + c]);
                unsigned live = 0, index = 0;
                for (unsigned c = 0; c < 16; ++c) {
                    const unsigned i = r * 16 + c;
                    const double x = half_to_float(v[i]);
                    next[i] = half_rne(std::max(0.0f,
                        static_cast<float>(d * x + off * (sum - x))));
                    if (next[i]) { ++live; index = c; }
                }
                if (!first[r] && live == 1 && index == winners[r]) first[r] = t;
            }
            v = next;
        }
        for (auto t : first) ++histogram[t];
        rows += 16;
    }
};
}  // namespace wta_test
