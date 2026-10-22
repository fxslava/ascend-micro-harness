// Tier-1 host probe for turboquant_reshape_and_cache_a8.
//
// The CAModel runner next door already compares the device output to the
// factored oracle bit for bit, so repeating that here would prove nothing. This
// probe checks the three things that comparison *cannot* see, because they are
// properties of the oracle itself rather than of the device:
//
//   1. the transform really is an orthogonal rotation. The explicit 256x256
//      matrix M = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32) satisfies
//      M M^T = headSize * I exactly, which is what makes
//      invSqrtLen = 1/sqrt(headSize) the right normalisation and what makes
//      "randomised Hadamard" a claim rather than a label;
//   2. the factored oracle - inner Hadamard, sign diagonal, three butterfly
//      stages over the channel group index - equals a dense application of M,
//      in exact integers. A butterfly cascade in the wrong order, or a stage
//      that transforms the wrong bit of the group index, still composes to
//      *some* orthogonal map and would agree with a device that made the same
//      mistake; it would not agree with M;
//   3. the INT4 nibble packing round-trips through the unpacking contract the
//      dequantisation leaf implements (two nibbles per byte, low nibble first,
//      signed INT4 in [-8, 7]), and the power-of-two quantiser stays inside
//      that range with the error bound its scale implies.
//
// Build and run:
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude
//       tests/test_reshape_and_cache_a8_probe.cpp
//       -o bin/test_reshape_and_cache_a8_probe
//   bin/test_reshape_and_cache_a8_probe

#include "sim_micro_testbed.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using ascend_micro::sim::GuardedBuffer;
using ascend_micro::sim::require;

constexpr int kTokens = 16;
constexpr int kHeadSize = 256;
constexpr int kGroup = 32;
constexpr int kGroups = kHeadSize / kGroup;
constexpr int kPanel = kTokens * kHeadSize;
constexpr int kPackedBytes = kPanel / 2;
constexpr int kBlockSize = kTokens;
constexpr float kInvSqrtLen = 1.0f / 16.0f;

int parity(unsigned v)
{
    int p = 0;
    while (v) {
        p ^= static_cast<int>(v & 1u);
        v >>= 1;
    }
    return p;
}
int hadamard(int i, int j) { return parity(static_cast<unsigned>(i & j)) ? -1 : 1; }
int pi_sign(int ch) { return ((ch * ch + 3 * ch + 1) % 7 < 3) ? -1 : 1; }

// The device transform, written as one dense matrix. Row ch_out, column ch_in.
//
//   M = (H8 (x) I32) . diag(sgn) . (I8 (x) H32)
//
// (H8 (x) I32)[ch_out][mid] is non-zero only where c0 matches and
// (I8 (x) H32)[mid][ch_in] only where c1 matches, so exactly one intermediate
// channel survives the product: mid = (c1_in, c0_out). The sign diagonal is
// therefore indexed by that intermediate channel, which is the whole point of
// putting it between the two factors.
int dense_entry(int ch_out, int ch_in)
{
    const int c1_out = ch_out / kGroup, c0_out = ch_out % kGroup;
    const int c1_in = ch_in / kGroup, c0_in = ch_in % kGroup;
    const int mid = c1_in * kGroup + c0_out;
    return hadamard(c1_out, c1_in) * pi_sign(mid) * hadamard(c0_out, c0_in);
}

long long rint_even_shift(long long y, int shift)
{
    if (shift <= 0) return y << (-shift);
    const long long den = 1LL << shift;
    long long q = y / den;
    const long long rem = y % den;
    if (rem == 0) return q;
    const long long twice = 2 * (rem < 0 ? -rem : rem);
    const bool away = (twice > den) || (twice == den && (q % 2 != 0));
    if (away) q += (y >= 0 ? 1 : -1);
    return q;
}

// The runner's factored oracle, verbatim in structure: inner Hadamard per
// channel group, then the sign diagonal, then the outer factor expressed the
// way the kernel runs it - three radix-2 stages over the group index.
void factored(const int* x, long long* y)
{
    static long long z[kPanel];
    for (int t = 0; t < kTokens; ++t)
        for (int c1 = 0; c1 < kGroups; ++c1)
            for (int o = 0; o < kGroup; ++o) {
                long long acc = 0;
                for (int i = 0; i < kGroup; ++i)
                    acc += hadamard(o, i) * x[t * kHeadSize + c1 * kGroup + i];
                z[t * kHeadSize + c1 * kGroup + o] = acc;
            }
    // Sign diagonal, in place.
    for (int i = 0; i < kPanel; ++i) z[i] *= pi_sign(i % kHeadSize);
    // Three radix-2 stages over c1, sums into the low half of each period and
    // differences into the high half, exactly as the kernel emits them.
    static long long a[kPanel], b[kPanel];
    for (int i = 0; i < kPanel; ++i) a[i] = z[i];
    for (int stride = 4; stride >= 1; stride /= 2) {
        const int period = 2 * stride * kGroup;
        for (int t = 0; t < kTokens; ++t) {
            const long long* src = a + t * kHeadSize;
            long long* dst = b + t * kHeadSize;
            for (int base = 0; base < kHeadSize; base += period)
                for (int k = 0; k < period / 2; ++k) {
                    dst[base + k] = src[base + k] + src[base + period / 2 + k];
                    dst[base + period / 2 + k] = src[base + k] - src[base + period / 2 + k];
                }
        }
        for (int i = 0; i < kPanel; ++i) a[i] = b[i];
    }
    for (int i = 0; i < kPanel; ++i) y[i] = a[i];
}

unsigned mix(unsigned h)
{
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

}  // namespace

int main()
{
    const auto started = std::chrono::steady_clock::now();
    try {
        // ---- 1. M M^T == headSize * I, so the rotation is orthogonal -----
        GuardedBuffer<int> matrix(kHeadSize * kHeadSize);
        for (int r = 0; r < kHeadSize; ++r)
            for (int c = 0; c < kHeadSize; ++c) {
                const int entry = dense_entry(r, c);
                require(entry == 1 || entry == -1,
                        "the dense rotation must be a signed Hadamard matrix");
                matrix[r * kHeadSize + c] = entry;
            }
        for (int r = 0; r < kHeadSize; ++r)
            for (int s = 0; s < kHeadSize; ++s) {
                long long dot = 0;
                for (int c = 0; c < kHeadSize; ++c)
                    dot += static_cast<long long>(matrix[r * kHeadSize + c]) *
                           matrix[s * kHeadSize + c];
                require(dot == (r == s ? kHeadSize : 0),
                        "M M^T is not headSize * I: the transform is not orthogonal");
            }
        ASSERT_CANARY_INTACT(matrix);

        // ---- 2. the factored oracle equals a dense application of M ------
        GuardedBuffer<long long> factored_out(kPanel);
        GuardedBuffer<long long> dense_out(kPanel);
        std::vector<int> input(kPanel);
        int cases = 0;
        long long largest = 0;
        for (int variant = 0; variant < 6; ++variant) {
            for (int t = 0; t < kTokens; ++t)
                for (int c = 0; c < kHeadSize; ++c) {
                    int v;
                    switch (variant) {
                        case 0: v = (c == 0 && t == 0) ? 127 : 0; break;      // unit spike
                        case 1: v = -128; break;                              // INT8 floor
                        case 2: v = 127; break;                               // INT8 ceiling
                        case 3: v = ((t + c) % 2) ? 127 : -128; break;        // worst case swing
                        case 4: v = (c % 32) - 16; break;                     // sawtooth
                        default:
                            v = static_cast<int>(mix(static_cast<unsigned>(t) * 0x85EBCA6Bu +
                                                     static_cast<unsigned>(c) * 0xC2B2AE35u) %
                                                 255u) - 127;
                    }
                    input[t * kHeadSize + c] = v;
                }
            factored(input.data(), factored_out.data());
            for (int t = 0; t < kTokens; ++t)
                for (int r = 0; r < kHeadSize; ++r) {
                    long long acc = 0;
                    for (int c = 0; c < kHeadSize; ++c)
                        acc += static_cast<long long>(matrix[r * kHeadSize + c]) *
                               input[t * kHeadSize + c];
                    dense_out[t * kHeadSize + r] = acc;
                }
            for (int i = 0; i < kPanel; ++i) {
                require(factored_out[i] == dense_out[i],
                        "the factored rotation disagrees with the dense matrix");
                const long long mag = factored_out[i] < 0 ? -factored_out[i] : factored_out[i];
                if (mag > largest) largest = mag;
                ++cases;
            }
        }
        // 128 * 32 * 8 is the bound the kernel header claims for |Y|.
        require(largest <= 32768, "|Y| exceeded the INT32 and FP32 exactness bound");
        ASSERT_CANARY_INTACT(factored_out);
        ASSERT_CANARY_INTACT(dense_out);

        // ---- 3. quantiser range and nibble round trip --------------------
        // Reuse the last variant's rotation, quantise it the way the kernel
        // does, pack it the way the kernel does, and unpack it the way
        // routine_turboquant_dequant4.hpp does.
        GuardedBuffer<std::uint8_t> packed(kPackedBytes);
        for (int i = 0; i < kPackedBytes; ++i) packed[i] = 0;
        int nibbles = 0;
        double worst_relative = 0.0;
        for (int t = 0; t < kTokens; ++t) {
            long long abs_max = 1;
            for (int ch = 0; ch < kHeadSize; ++ch) {
                const long long a = factored_out[t * kHeadSize + ch] < 0
                                        ? -factored_out[t * kHeadSize + ch]
                                        : factored_out[t * kHeadSize + ch];
                if (a > abs_max) abs_max = a;
            }
            int e = 0;
            while ((1LL << (e + 1)) <= abs_max) ++e;
            const float scale = std::ldexp(1.0f, e - 2) * kInvSqrtLen;
            for (int ch = 0; ch < kHeadSize; ++ch) {
                long long q = rint_even_shift(factored_out[t * kHeadSize + ch], e - 2);
                if (q > 7) q = 7;
                require(q >= -8 && q <= 7, "quantised value outside signed INT4");
                // Key-major packing: index key * headSize + d.
                const int index = t * kHeadSize + ch;
                const auto nib = static_cast<std::uint8_t>(q & 0xF);
                packed[index / 2] |= static_cast<std::uint8_t>(
                    (index % 2 == 0) ? nib : static_cast<std::uint8_t>(nib << 4));
                // The reconstruction error of a 2^(e-2) scale is at most half a
                // step against the orthonormal value Y / sqrt(headSize), except
                // where the upper clamp bites.
                const double normalised =
                    static_cast<double>(factored_out[t * kHeadSize + ch]) * kInvSqrtLen;
                const double recovered = static_cast<double>(q) * static_cast<double>(scale);
                const double step = static_cast<double>(scale);
                if (q != 7) {
                    require(std::abs(recovered - normalised) <= 0.5 * step + 1e-6,
                            "power-of-two quantisation error exceeds half a step");
                }
                const double denominator = std::abs(normalised) > step ? std::abs(normalised) : step;
                const double relative = std::abs(recovered - normalised) / denominator;
                if (relative > worst_relative) worst_relative = relative;
                ++nibbles;
            }
        }
        // Unpack with the dequant4 contract and compare to the quantised values.
        int round_tripped = 0;
        for (int t = 0; t < kTokens; ++t) {
            long long abs_max = 1;
            for (int ch = 0; ch < kHeadSize; ++ch) {
                const long long a = factored_out[t * kHeadSize + ch] < 0
                                        ? -factored_out[t * kHeadSize + ch]
                                        : factored_out[t * kHeadSize + ch];
                if (a > abs_max) abs_max = a;
            }
            int e = 0;
            while ((1LL << (e + 1)) <= abs_max) ++e;
            for (int ch = 0; ch < kHeadSize; ++ch) {
                const int index = t * kHeadSize + ch;
                const std::uint8_t byte = packed[index / 2];
                const int raw = (index % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                const int unpacked = raw >= 8 ? raw - 16 : raw;  // sign extend INT4
                long long q = rint_even_shift(factored_out[index], e - 2);
                if (q > 7) q = 7;
                require(unpacked == static_cast<int>(q),
                        "nibble did not survive the pack and unpack round trip");
                ++round_tripped;
            }
        }
        ASSERT_CANARY_INTACT(packed);

        const auto elapsed = std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        std::printf("PASS reshape_and_cache_a8 host probe in %.3f us\n", elapsed);
        std::printf("  orthogonality: M M^T == %d * I over %d x %d signed entries\n",
                    kHeadSize, kHeadSize, kHeadSize);
        std::printf("  factorisation: %d values over 6 input patterns identical to a "
                    "dense apply, largest |Y| %lld (bound 32768)\n",
                    cases, largest);
        std::printf("  quantiser: %d nibbles inside [-8, 7], %d round-tripped through "
                    "the dequant4 packing, worst relative error %.4f\n",
                    nibbles, round_tripped, worst_relative);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL reshape_and_cache_a8 host probe: %s\n", error.what());
        return 1;
    }
}
