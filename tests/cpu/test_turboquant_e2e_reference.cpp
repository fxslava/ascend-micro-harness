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

// What does the kv4fp8 fused decode MEAN by the bytes in its cache?
// =================================================================
// A round-trip test needs a golden attention value computed from the cache the INGRESS
// actually wrote, which means knowing the decode's reconstruction exactly: how a stored
// nibble and a stored scale become the number that enters the score GEMM. Getting that
// wrong would make a correct round trip look broken, or -- worse -- make a broken one look
// correct against a reference that shares the mistake.
//
// So this probe pins the reconstruction on the host, with no device and no simulator, by
// reproducing tests/common/fused_decode_opt_fixture.hpp's own `expected` from its own cache
// bytes using a model written in the codec's terms:
//
//   n           the SIGNED int4 in [-8, 7] that turboquant_dequant4_signed widens
//   operand     n + kSignedLevelOffset, where kSignedLevelOffset = kNibbleSignShift
//               - kAffineBias = 8 - 7.5 = 0.5  (common/turboquant_codec_mx.h,
//               ComputeNibbleOperands). Equivalently centroid b - 7.5 for the unsigned
//               bin b = n + 8, which is TurboQuantModeTraits<KV4_FP8>::kCentroids.
//   value       operand * stored_scale / kGain -- the gain is factored OUT of the operand
//               and folded into the score scale, which is what kGain is for.
//
// If the model below reproduces `expected`, it IS the decode's semantics, and
// test_camodel_turboquant_e2e_roundtrip can use it as the golden for a cache no fixture
// wrote.
//
// THE HALF STEP, WHICH IS A FINDING AND NOT A DETAIL
// --------------------------------------------------
// `operand = n + 0.5` is a MIDRISE reconstruction: the levels sit at the half-integers, so
// zero is not representable. The prefill ingress quantiser is MIDTREAD -- it stores
// n = rint(y / s) and means y ~= n * s, with zero representable, and its Tier-1 probe checks
// exactly that (|q.scale - y| <= half a step, tests/cpu/test_reshape_scatter_probe.cpp). The
// two conventions differ by half a quantisation step on every element.
//
// That is a real semantic gap between the two operators, separate from the layout one, and
// this probe measures it rather than papering over it: check 2 reports the reconstruction
// error of the ingress's own intent under the decode's rule. Which convention TurboQuant
// wants is a quantiser decision with its own numerical gates, so nothing here changes
// either side; the round-trip test's golden uses the DECODE's rule, because the decode is
// what produces the output being checked.

#include "../common/fused_decode_opt_fixture.hpp"
#include "../common/turboquant_nz_ingress_oracle.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

namespace tq = vllm_ascend::turboquant;
using Mode = tq::TurboQuantModeTraits<tq::TurboQuantMode::KV4_FP8>;

int g_failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        std::printf("  FAIL %s\n", what);
        ++g_failures;
    }
}

// kSignedLevelOffset of common/turboquant_codec_mx.h, restated from the traits this header
// can see: kNibbleSignShift is half the radix (16 / 2 = 8) and kAffineBias is 7.5.
constexpr double kNibbleSignShift = 8.0;
constexpr double kSignedLevelOffset = kNibbleSignShift - Mode::kAffineBias;  // 0.5

// The decode's reconstruction of one stored element.
double dequantise(int signed_nibble, float stored_scale)
{
    return (static_cast<double>(signed_nibble) + kSignedLevelOffset) *
           static_cast<double>(stored_scale) / static_cast<double>(Mode::kGain);
}

// Read the signed int4 at (global row, kv head, channel) of an NZ-tiled plane. Same
// arithmetic as tq_nz::read_code, expressed against a flat cache so the fixture's buffers
// can be read directly.
int read_signed_nibble(const std::uint8_t *plane, std::uint64_t slot, unsigned head,
                       unsigned channel, unsigned kv_heads, unsigned head_size)
{
    const std::uint64_t byte = tq::NzTiledChannelByte(slot, head, channel, kv_heads, head_size);
    const std::uint32_t half = tq::NzTiledChannelNibble(channel, head_size);
    const std::uint8_t raw = half == 0u ? static_cast<std::uint8_t>(plane[byte] & 0x0F)
                                        : static_cast<std::uint8_t>(plane[byte] >> 4);
    return raw >= 8u ? static_cast<int>(raw) - 16 : static_cast<int>(raw);
}

// The whole decode, in FP64, from a cache and a query. `score_scale` is the multiplier the
// operator applies to q . k before the softmax; which launch field carries it is what
// check 1 determines.
std::vector<double> attention(const fused_decode_opt_test::Fixture &f, int request,
                              int query_head, double score_scale)
{
    const unsigned dim = 256;
    const unsigned group = static_cast<unsigned>(f.group_heads);
    const unsigned head = static_cast<unsigned>(query_head) / group;
    const unsigned kv_heads = static_cast<unsigned>(f.kv_heads);
    const auto *q = f.query.data() +
                    (static_cast<std::size_t>(request) * f.kv_heads * f.group_heads +
                     query_head) * dim;

    const int context = f.context;
    std::vector<double> score(static_cast<std::size_t>(context), 0.0);
    std::vector<std::uint64_t> slot(static_cast<std::size_t>(context), 0);
    for (int token = 0; token < context; ++token) {
        const int page = token / f.page_tokens;
        const int row = token % f.page_tokens;
        const int physical = f.table[static_cast<std::size_t>(request) * f.pages + page];
        slot[static_cast<std::size_t>(token)] =
            static_cast<std::uint64_t>(physical) * f.page_tokens + row;
        const float ks = f.scales[tq::NzTiledScaleIndex(slot[static_cast<std::size_t>(token)],
                                                        head, 0, kv_heads)];
        double dot = 0.0;
        for (unsigned c = 0; c < dim; ++c) {
            const int n = read_signed_nibble(f.key.data(), slot[static_cast<std::size_t>(token)],
                                             head, c, kv_heads, dim);
            dot += static_cast<double>(q[c]) * dequantise(n, ks);
        }
        score[static_cast<std::size_t>(token)] = dot * score_scale;
    }

    double m = score[0];
    for (double s : score) m = s > m ? s : m;
    double z = 0.0;
    std::vector<double> p(static_cast<std::size_t>(context), 0.0);
    for (int i = 0; i < context; ++i) {
        p[static_cast<std::size_t>(i)] = std::exp(score[static_cast<std::size_t>(i)] - m);
        z += p[static_cast<std::size_t>(i)];
    }

    std::vector<double> out(dim, 0.0);
    for (int i = 0; i < context; ++i) {
        const float vs = f.scales[tq::NzTiledScaleIndex(slot[static_cast<std::size_t>(i)], head,
                                                        1, kv_heads)];
        for (unsigned c = 0; c < dim; ++c) {
            const int n = read_signed_nibble(f.value.data(), slot[static_cast<std::size_t>(i)],
                                             head, c, kv_heads, dim);
            out[c] += p[static_cast<std::size_t>(i)] * dequantise(n, vs);
        }
    }
    for (unsigned c = 0; c < dim; ++c) out[c] /= z;
    return out;
}

// `expected` is stored as float, computed in double, so the tightest meaningful agreement is
// float rounding and not zero. Anything above this is a model error, not arithmetic noise.
constexpr double kFloatTol = 1e-6;

double worst_error(const fused_decode_opt_test::Fixture &f, int request, int query_head,
                   double score_scale)
{
    const std::vector<double> got = attention(f, request, query_head, score_scale);
    const auto base = (static_cast<std::size_t>(request) * f.kv_heads * f.group_heads +
                       query_head) * 256;
    double worst = 0.0;
    for (unsigned c = 0; c < 256; ++c) {
        const double want = f.expected[base + c];
        const double err = std::fabs(got[c] - want) /
                           (std::fabs(want) > 1e-12 ? std::fabs(want) : 1.0);
        worst = err > worst ? err : worst;
    }
    return worst;
}

// Check 1. Which score multiplier reproduces the fixture's own expectation?
//
// This has to run on a context of SEVERAL tiles. On one tile every row of the tile shares a
// key level, so every score is the same number, the softmax is uniform and the result is
// independent of the multiplier entirely -- all candidates "pass" and the check discriminates
// nothing. Two tiles with different levels is the smallest geometry where the weights differ
// and the multiplier is observable.
void check_score_scale()
{
    const fused_decode_opt_test::Fixture one(64, 1, 1, 1, 64);
    const fused_decode_opt_test::Fixture many(192, 1, 1, 1, 64);
    const double candidates[] = {1.0, 0.0625, 16.0};
    const char *names[] = {"1 (control)", "0.0625 = inv_sqrt_len", "16 (control)"};
    int winner = -1, degenerate = 0;
    for (int i = 0; i < 3; ++i) {
        const double w_one = worst_error(one, 0, 0, candidates[i]);
        const double w_many = worst_error(many, 0, 0, candidates[i]);
        std::printf("     score scale %-24s 1 tile %.3e   3 tiles %.3e\n", names[i], w_one,
                    w_many);
        if (w_one < kFloatTol) ++degenerate;
        if (w_many < kFloatTol) winner = i;
    }
    check(degenerate == 3,
          "one tile cannot discriminate the score scale: its softmax is uniform");
    check(winner == 1, "over several tiles only inv_sqrt_len reproduces the expectation");
}

// Check 2. The model reproduces `expected` over several geometries, which is what makes it
// usable as the round-trip golden rather than a one-case coincidence.
void check_model(int context, int requests, int kv_heads, int group, int page)
{
    const fused_decode_opt_test::Fixture f(context, requests, kv_heads, group, page);
    double worst = 0.0;
    for (int r = 0; r < requests; ++r) {
        for (int qh = 0; qh < kv_heads * group; ++qh) {
            const double err = worst_error(f, r, qh, 0.0625);
            worst = err > worst ? err : worst;
        }
    }
    std::printf("     context %4d, %d request(s), %d kv head(s), group %2d, page %3d: "
                "worst relative error %.3e\n",
                context, requests, kv_heads, group, page, worst);
    check(worst < kFloatTol, "the reconstruction model reproduces the fixture's expectation");
}

// Check 3. The half step, measured. For a midtread ingress that meant y ~= n * s, what does
// the decode's midrise rule reconstruct? Exactly half a step away, every time -- so this
// reports the gap as a number rather than asserting it away.
void check_half_step()
{
    const float s = 0.25f;
    double worst_rel = 0.0;
    std::printf("     n   ingress means n*s   decode reconstructs   difference\n");
    for (int n = -8; n <= 7; ++n) {
        const double meant = static_cast<double>(n) * s;
        const double got = dequantise(n, s * static_cast<float>(Mode::kGain));
        if (n >= -8 && n <= -6) {
            std::printf("     %-3d %17.6f %21.6f %12.6f\n", n, meant, got, got - meant);
        }
        const double diff = got - meant;
        check(std::fabs(diff - 0.5 * s) < 1e-6,
              "the decode's reconstruction is exactly half a step above the ingress's");
        const double rel = std::fabs(diff) / (std::fabs(meant) > 1e-12 ? std::fabs(meant) : 1.0);
        worst_rel = rel > worst_rel ? rel : worst_rel;
    }
    std::printf("     ... (all 16 codes checked)\n");
    std::printf("     the gap is a CONSTANT +0.5 * scale on every code: a midrise rule read\n");
    std::printf("     against a midtread quantiser. Worst relative size %.3f at the codes\n",
                worst_rel);
    std::printf("     nearest zero, where the ingress means a small number and the decode\n");
    std::printf("     cannot represent zero at all.\n");
}

}  // namespace

int main()
{
    std::printf("turboquant kv4fp8 reconstruction: what the fused decode means by a cache byte\n");
    std::printf("  kGain %.10f, kAffineBias %.1f, kSignedLevelOffset %.1f, levels %d\n",
                static_cast<double>(Mode::kGain), static_cast<double>(Mode::kAffineBias),
                kSignedLevelOffset, Mode::kLevels);

    std::printf("\n1. which score multiplier does the operator apply?\n");
    check_score_scale();

    std::printf("\n2. the reconstruction model against the fixture's own expectation\n");
    check_model(64, 1, 1, 1, 64);
    check_model(16, 1, 4, 1, 64);
    check_model(64, 1, 4, 4, 64);
    check_model(128, 2, 2, 2, 64);
    check_model(100, 1, 1, 1, 64);  // a ragged tail: context not a whole tile

    std::printf("\n3. the ingress/decode half-step gap, measured\n");
    check_half_step();

    if (g_failures) {
        std::printf("\nturboquant kv4fp8 reconstruction probe: FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("\nturboquant kv4fp8 reconstruction probe: PASS\n");
    std::printf("  value = (signed_nibble + 0.5) * stored_scale / kGain, score scaled by\n");
    std::printf("  inv_sqrt_len. This is the golden model the round-trip test uses.\n");
    return 0;
}
