// Runtime page-count runner for fused TurboQuant paged attention. Three pages
// preserve the numerics baseline; eight pages wrap the depth-3 ring twice.
//
// Checks, in order:
//
//   0. the fixture properties section 6.4 of the guidelines demands, asserted
//      rather than asserted-in-a-comment: nibbles in range, strictly
//      increasing per-page maxima, alpha < 1 on every page after the first,
//      pairwise distinct per-page p_sum, pairwise distinct output channels,
//      and every reference score entry a multiple of 1/128 so the panel check
//      can demand bit equality. A fixture whose pages are interchangeable
//      cannot detect a wrong-slot read, which is the whole point at this page
//      count;
//   1. every 16x16 score panel of every page, bit-exactly. The kernel's
//      Fixpipe ring is only three panels deep; in the eight-page run its
//      first five panels are overwritten before the sequence ends. It keeps a
//      per-page archive in UB and the single end-of-sequence outflow carries
//      all tiles of them. A wrong-slot read anywhere on the
//      dequant4 -> UB -> L1(Zn) -> L0B -> Mmad -> Fixpipe path lands here;
//   2. the online-softmax state (m, l) and the FP32 normalised context against
//      a CPU attention oracle, in ULP;
//   3. the FP16 context actually stored by the outflow leaf.

#include "camodel_common.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>

extern "C" void launch_camodel_fused_attention(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, std::uint32_t*,
    float*, float*, float*, std::uint16_t*, std::uint32_t);

namespace {

constexpr int kStageDepth = 3;
constexpr int kMaxTiles = 8;
constexpr int kDim = 16;
constexpr int kKeys = 16;
constexpr int kPanel = kDim * kKeys;
constexpr int kPackedBytes = kPanel / 2;
constexpr float kKeyScale = 0.5f;
constexpr float kValueScale = 0.25f;

// Fixture generators. key_nibble and value_nibble are indexed by meaning, not
// by packing order: key_nibble(tiles, t, d, key) is the quantised K[key][d] of page t
// and value_nibble(t, ch, key) the quantised V[key][ch]. The oracle below is
// therefore the same one the GM-workspace revision used.
//
// Device packing contract: k nibble index (key * kDim + d), so the dequantised
// UB page is the ND matrix K[key][d], whose Nz image is byte for byte the Zn
// image of K^T that the Cube B operand needs; v nibble index (ch * kKeys + key),
// the channel-major order update_accumulator expects for v_by_channel.
//
// Why the key generator is not the N = 3 one
// ------------------------------------------
// The three-page fixture varies the page through (d + 1) * 14 * t mod 16.
// Modular wrap is fine for three pages -- its maxima are 0, 1/16, 2/16 -- but
// over eight it folds: the maxima come out 0, 1/16, 2/16, 7/16, 0, 2/16, 0, 0,
// so pages 4..7 never exceed the running maximum, alpha is exactly 1 on four
// of the eight pages and the rescale path stops running. A page count this
// large needs a ramp that cannot wrap, so the page index enters as a
// non-modular slope instead:
//
//   key_nibble(tiles, t, d, key) = base(d, key) + t * key_ramp(d, key)
//
// with base reduced modulo 16 - (tiles - 1) so that base + (tiles - 1) * 1
// still fits in a nibble. Row 0 of the query weights head dimension d by
// ((5 d mod 9) - 4) / 64, i.e. by
//
//   d     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
//   w*64 -4  +1  -3  +2  -2  +3  -1  +4   0  -4  +1  -3  +2  -2  +3  -1
//
// so a score entry is an exact multiple of 1/128 with integer numerator
// sum_d w*64(d) * nibble, and the slope of key k in units of 1/128 per page is
// G(k) = sum over the ramping dimensions of w*64(d).
int key_ramp(int d, int key)
{
    // The two largest positive weights ramp on every key: +4 and +3 give every
    // key a slope floor of +7/128 per page, which is what keeps the per-page
    // maximum strictly increasing.
    if (d == 7 || d == 14) return 1;
    // Each bit of the key index switches on the ramp of one negative-weight
    // dimension (-4, -3, -2, -1), so G(k) = 7 - (4 b0 + 3 b1 + 2 b2 + b3)
    // ranges from +7 down to -3 and almost every key has its own slope. The
    // panel therefore changes shape from page to page, not just height, which
    // is what makes the per-page p_sum values distinct.
    if (d == 0 || d == 2 || d == 4 || d == 6) return (key >> (d / 2)) & 1;
    return 0;
}
int key_nibble(int tiles, int t, int d, int key)
{
    // Preserve the original three-page numerics fixture exactly. Its modular
    // ramp folds after page 3, so longer runs use the monotone ramp below.
    if (tiles == kStageDepth)
        return (d * 8 + key + t * (d + 1) * 14) % 16;
    return (d * 8 + key) % (16 - (tiles - 1)) + t * key_ramp(d, key);
}
int value_nibble(int t, int ch, int key) { return (ch + key + t * (ch + 3) * 3) % 16; }
float query(int row, int d) { return static_cast<float>(((row * 3 + d * 5) % 9) - 4) / 64.0f; }

std::uint32_t bits(float value)
{
    std::uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

// Normal-range round-to-nearest-even float -> half. The fixture keeps every
// converted value inside the half normal range.
std::uint16_t half_rne(float value)
{
    if (value == 0.0f) return 0;
    const auto raw = bits(value);
    const auto sign = static_cast<std::uint16_t>((raw >> 16) & 0x8000u);
    const int exponent = static_cast<int>((raw >> 23) & 0xffu) - 127 + 15;
    const auto mantissa = raw & 0x7fffffu;
    if (exponent <= 0 || exponent >= 31)
        throw std::runtime_error("fixture value outside half normal range");
    const auto rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    return static_cast<std::uint16_t>(sign | ((exponent + (rounded >> 23)) << 10) |
                                      ((rounded >> 13) & 0x3ffu));
}

float half_to_float(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const int exponent = (value >> 10) & 0x1f;
    const std::uint32_t mantissa = value & 0x3ffu;
    std::uint32_t raw;
    if (exponent == 0) {
        if (mantissa == 0) { raw = sign; }
        else {
            int shift = 0;
            std::uint32_t m = mantissa;
            while (!(m & 0x400u)) { m <<= 1; ++shift; }
            raw = sign | (static_cast<std::uint32_t>(127 - 15 - shift) << 23) |
                  ((m & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        raw = sign | 0x7f800000u | (mantissa << 13);
    } else {
        raw = sign | (static_cast<std::uint32_t>(exponent - 15 + 127) << 23) |
              (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &raw, sizeof(out));
    return out;
}

float ulp_of(float value)
{
    const float magnitude = std::abs(value);
    if (magnitude == 0.0f) return std::nextafter(0.0f, 1.0f);
    return std::nextafter(magnitude, HUGE_VALF) - magnitude;
}

void usage(FILE* output, const char* program)
{
    std::fprintf(output, "Usage: %s [--tiles N]\n"
                         "  N must be 3..8 (default: 3).\n", program);
}

}  // namespace

int main(int argc, char** argv)
{
    int tiles = kStageDepth;
    if (argc > 1) {
        const std::string_view arg = argv[1];
        if ((arg == "-h" || arg == "--help") && argc == 2) {
            usage(stdout, argv[0]);
            return 0;
        }
        if (arg == "--tiles" && argc == 3) {
            const std::string_view value = argv[2];
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), tiles);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                tiles < kStageDepth || tiles > kMaxTiles) {
                usage(stderr, argv[0]);
                return 2;
            }
        } else {
            usage(stderr, argv[0]);
            return 2;
        }
    }
    try {
        std::printf("Fused attention tiles: %d\n", tiles);
        camodel::Session session;
        camodel::GuardedDeviceBuffer q(kDim * kKeys * 2);
        camodel::GuardedDeviceBuffer k_packed(tiles * kPackedBytes);
        camodel::GuardedDeviceBuffer v_packed(tiles * kPackedBytes);
        camodel::GuardedDeviceBuffer offsets(kPanel * 4);
        camodel::GuardedDeviceBuffer out_scores(tiles * kPanel * 4);
        camodel::GuardedDeviceBuffer out_state(kDim * 4);
        camodel::GuardedDeviceBuffer out_normalized(kDim * 4);
        camodel::GuardedDeviceBuffer out_context(kDim * 2);

        auto* qv = static_cast<std::uint16_t*>(q.host_payload());
        auto* kv = static_cast<std::uint8_t*>(k_packed.host_payload());
        auto* vv = static_cast<std::uint8_t*>(v_packed.host_payload());
        auto* ov = static_cast<std::uint32_t*>(offsets.host_payload());

        for (int row = 0; row < kDim; ++row)
            for (int d = 0; d < kKeys; ++d)
                qv[row * kKeys + d] = half_rne(query(row, d));

        // A nibble out of range would be silently truncated by the packer and
        // would then disagree with the oracle for a reason that has nothing to
        // do with the kernel.
        for (int t = 0; t < tiles; ++t)
            for (int a = 0; a < kDim; ++a)
                for (int b = 0; b < kKeys; ++b)
                    if (key_nibble(tiles, t, a, b) < 0 || key_nibble(tiles, t, a, b) > 15 ||
                        value_nibble(t, a, b) < 0 || value_nibble(t, a, b) > 15) {
                        std::fprintf(stderr, "fixture nibble out of range at "
                                             "t=%d a=%d b=%d: k=%d v=%d\n",
                                     t, a, b, key_nibble(tiles, t, a, b),
                                     value_nibble(t, a, b));
                        throw std::runtime_error("fixture nibble outside 0..15");
                    }

        // Pack two nibbles per byte, low nibble first. The key page is packed
        // key-major (index key * kDim + d) and the value page channel-major
        // (index ch * kKeys + key).
        for (int t = 0; t < tiles; ++t)
            for (int i = 0; i < kPackedBytes; ++i) {
                const int lo = 2 * i, hi = 2 * i + 1;
                const int k_lo = key_nibble(tiles, t, lo % kDim, lo / kDim);
                const int k_hi = key_nibble(tiles, t, hi % kDim, hi / kDim);
                const int v_lo = value_nibble(t, lo / kKeys, lo % kKeys);
                const int v_hi = value_nibble(t, hi / kKeys, hi % kKeys);
                kv[t * kPackedBytes + i] = static_cast<std::uint8_t>((k_hi << 4) | k_lo);
                vv[t * kPackedBytes + i] = static_cast<std::uint8_t>((v_hi << 4) | v_lo);
            }
        for (int ch = 0; ch < kDim; ++ch)
            for (int key = 0; key < kKeys; ++key)
                ov[ch * kKeys + key] = static_cast<std::uint32_t>(key * sizeof(float));

        q.upload(); k_packed.upload(); v_packed.upload(); offsets.upload();
        out_scores.upload(); out_state.upload();
        out_normalized.upload(); out_context.upload();

        launch_camodel_fused_attention(
            session.stream, static_cast<std::uint16_t*>(q.payload()),
            static_cast<std::int8_t*>(k_packed.payload()),
            static_cast<std::int8_t*>(v_packed.payload()),
            static_cast<std::uint32_t*>(offsets.payload()),
            static_cast<float*>(out_scores.payload()),
            static_cast<float*>(out_state.payload()),
            static_cast<float*>(out_normalized.payload()),
            static_cast<std::uint16_t*>(out_context.payload()),
            static_cast<std::uint32_t>(tiles));
        camodel::check(aclrtSynchronizeStream(session.stream), "fused attention synchronize");

        q.download(); k_packed.download(); v_packed.download(); offsets.download();
        out_scores.download(); out_state.download();
        out_normalized.download(); out_context.download();
        q.assert_guards(); k_packed.assert_guards(); v_packed.assert_guards();
        offsets.assert_guards(); out_scores.assert_guards();
        out_state.assert_guards(); out_normalized.assert_guards();
        out_context.assert_guards();

        // ---- Oracle -------------------------------------------------------
        // Q and the dequantised key page are both half and exactly
        // representable, and every product and partial sum is a multiple of
        // 1/128 well inside FP32, so the reference score panel is exact.
        static float reference_scores[kMaxTiles][kDim][kKeys];
        for (int t = 0; t < tiles; ++t)
            for (int row = 0; row < kDim; ++row)
                for (int key = 0; key < kKeys; ++key) {
                    float sum = 0.0f;
                    for (int d = 0; d < kKeys; ++d)
                        sum += half_to_float(qv[row * kKeys + d]) *
                               (static_cast<float>(key_nibble(tiles, t, d, key)) * kKeyScale);
                    reference_scores[t][row][key] = sum;
                }

        float m = 0.0f, l = 0.0f, acc[kDim] = {};
        float page_max[kMaxTiles], page_psum[kMaxTiles], page_alpha[kMaxTiles];
        for (int t = 0; t < tiles; ++t) {
            const float* s0 = reference_scores[t][0];
            const float tile_max = *std::max_element(s0, s0 + kKeys);
            const float next_m = std::max(m, tile_max);
            const float alpha = std::exp(m - next_m);
            float p[kKeys], p_sum = 0.0f;
            for (int key = 0; key < kKeys; ++key) {
                p[key] = std::exp(s0[key] - next_m);
                p_sum += p[key];
            }
            l = l * alpha + p_sum;
            for (int ch = 0; ch < kDim; ++ch) {
                float weighted = 0.0f;
                for (int key = 0; key < kKeys; ++key)
                    weighted += p[key] *
                                (static_cast<float>(value_nibble(t, ch, key)) * kValueScale);
                acc[ch] = acc[ch] * alpha + weighted;
            }
            m = next_m;
            page_max[t] = tile_max;
            page_psum[t] = p_sum;
            page_alpha[t] = alpha;
        }
        float context[kDim];
        for (int ch = 0; ch < kDim; ++ch) context[ch] = acc[ch] / l;

        // ---- 0. the fixture properties of guidelines section 6.4 ----------
        for (int t = 0; t < tiles; ++t)
            for (int row = 0; row < kDim; ++row)
                for (int key = 0; key < kKeys; ++key) {
                    const float scaled = reference_scores[t][row][key] * 128.0f;
                    if (scaled != std::floor(scaled)) {
                        std::fprintf(stderr,
                                     "score t=%d row=%d key=%d = %.9g is not a "
                                     "multiple of 1/128\n",
                                     t, row, key, reference_scores[t][row][key]);
                        throw std::runtime_error("fixture score not dyadic");
                    }
                }
        for (int t = 1; t < tiles; ++t)
            if (!(page_max[t] > page_max[t - 1]) || !(page_alpha[t] < 1.0f)) {
                std::fprintf(stderr,
                             "page %d max %.9g is not above page %d max %.9g "
                             "(alpha %.9g)\n",
                             t, page_max[t], t - 1, page_max[t - 1], page_alpha[t]);
                throw std::runtime_error("page maxima are not strictly increasing");
            }
        for (int t = 0; t < tiles; ++t)
            for (int u = t + 1; u < tiles; ++u)
                if (page_psum[t] == page_psum[u]) {
                    std::fprintf(stderr, "pages %d and %d share p_sum %.9g\n",
                                 t, u, page_psum[t]);
                    throw std::runtime_error("per-page p_sum values are not distinct");
                }
        for (int ch = 0; ch < kDim; ++ch)
            for (int other = ch + 1; other < kDim; ++other)
                if (context[ch] == context[other]) {
                    std::fprintf(stderr, "channels %d and %d share context %.9g\n",
                                 ch, other, context[ch]);
                    throw std::runtime_error("output channels are not distinct");
                }

        // ---- 1. Score panels, bit exact -----------------------------------
        // On a mismatch, say *which* page's data each panel is carrying. At
        // this page count the interesting failure is a wrong-slot read, and
        // "panel 0 holds page 3" is a diagnosis where "panel 0 entry 0 is
        // 0.195 not 0.031" is only a symptom.
        const auto* got_scores = static_cast<const float*>(out_scores.host_payload());
        const float* reference = &reference_scores[0][0][0];  // tiles panels
        int mismatches = 0;
        for (int i = 0; i < tiles * kPanel; ++i)
            if (bits(got_scores[i]) != bits(reference[i])) ++mismatches;
        if (mismatches) {
            std::fprintf(stderr, "%d of %d score entries differ\n",
                         mismatches, tiles * kPanel);
            for (int t = 0; t < tiles; ++t) {
                int carries = -1, wrong = 0;
                for (int u = 0; u < tiles; ++u) {
                    int same = 0;
                    for (int i = 0; i < kPanel; ++i)
                        same += bits(got_scores[t * kPanel + i]) ==
                                bits(reference[u * kPanel + i]);
                    if (same == kPanel) carries = u;
                }
                for (int i = 0; i < kPanel; ++i)
                    wrong += bits(got_scores[t * kPanel + i]) !=
                             bits(reference[t * kPanel + i]);
                if (carries >= 0)
                    std::fprintf(stderr, "  panel %d: holds page %d's panel "
                                         "(%d/%d entries differ from page %d)\n",
                                 t, carries, wrong, kPanel, t);
                else
                    std::fprintf(stderr,
                                 "  panel %d: matches no page; %d/%d entries "
                                 "differ, row 0 key 0 got %.9g expected %.9g\n",
                                 t, wrong, kPanel,
                                 static_cast<double>(got_scores[t * kPanel]),
                                 static_cast<double>(reference[t * kPanel]));
            }
            std::fprintf(stderr, "  page row 0 key 0 reference:");
            for (int u = 0; u < tiles; ++u)
                std::fprintf(stderr, " %d:%.9g", u,
                             static_cast<double>(reference[u * kPanel]));
            std::fprintf(stderr, "\n  panel row 0 key 0 device   :");
            for (int t = 0; t < tiles; ++t)
                std::fprintf(stderr, " %d:%.9g", t,
                             static_cast<double>(got_scores[t * kPanel]));
            std::fprintf(stderr, "\n");
            throw std::runtime_error("Cube score panel mismatch");
        }

        // ---- 2. Recurrence state and FP32 context, in ULP -----------------
        const auto* got_state = static_cast<const float*>(out_state.host_payload());
        const auto* got_norm = static_cast<const float*>(out_normalized.host_payload());
        float max_ulp = 0.0f;
        const char* worst = "none";
        auto check_ulp = [&](const char* label, int index, float have, float want,
                             float budget) {
            if (!std::isfinite(have)) throw std::runtime_error("non-finite device result");
            const float ulps = std::abs(have - want) / ulp_of(want);
            if (ulps > max_ulp) { max_ulp = ulps; worst = label; }
            if (ulps > budget) {
                std::fprintf(stderr, "%s index=%d got=%.9g expected=%.9g (%.2f ULP)\n",
                             label, index, have, want, static_cast<double>(ulps));
                throw std::runtime_error("fused attention precision budget exceeded");
            }
        };
        // The replicated-state invariant: every lane of the m and l blocks
        // must carry the same value.
        for (int lane = 1; lane < 8; ++lane) {
            if (bits(got_state[lane]) != bits(got_state[0]) ||
                bits(got_state[8 + lane]) != bits(got_state[8])) {
                std::fprintf(stderr, "state lane %d not replicated\n", lane);
                throw std::runtime_error("replicated-state invariant violated");
            }
        }
        check_ulp("m", 0, got_state[0], m, 8.0f);
        check_ulp("l", 0, got_state[8], l, 8.0f);
        for (int ch = 0; ch < kDim; ++ch)
            check_ulp("context", ch, got_norm[ch], context[ch], 8.0f);

        // ---- 3. FP16 outflow ----------------------------------------------
        const auto* got_ctx = static_cast<const std::uint16_t*>(out_context.host_payload());
        for (int ch = 0; ch < kDim; ++ch) {
            const float have = half_to_float(got_ctx[ch]);
            const float want = half_to_float(half_rne(context[ch]));
            const float tolerance = std::abs(want) * 0x1p-10f;  // one half ULP
            if (!(std::abs(have - want) <= tolerance)) {
                std::fprintf(stderr,
                             "context FP16 ch=%d got=0x%04x (%.7g) expected=0x%04x (%.7g)\n",
                             ch, got_ctx[ch], have, half_rne(context[ch]), want);
                throw std::runtime_error("FP16 context mismatch");
            }
        }

        const std::size_t guard_bytes = 8 * 2 * camodel::kGuardBytes;
        std::printf("PASS fused attention: %d pages over a depth-%d ring "
                    "(%d slot reuses), %d/%d score entries bit-exact, "
                    "max %.3f ULP (worst operand %s), %d/%d FP16 context within 1 half ULP, "
                    "%zu/%zu GM guard bytes intact\n",
                    tiles, kStageDepth, tiles - kStageDepth,
                    tiles * kPanel, tiles * kPanel,
                    static_cast<double>(max_ulp), worst, kDim, kDim,
                    guard_bytes, guard_bytes);
        std::printf("  final m=%.9g l=%.9g  context[0]=%.9g context[15]=%.9g\n",
                    static_cast<double>(got_state[0]), static_cast<double>(got_state[8]),
                    static_cast<double>(got_norm[0]), static_cast<double>(got_norm[15]));
        std::printf("  fixture: page maxima");
        for (int t = 0; t < tiles; ++t)
            std::printf(" %.6g", static_cast<double>(page_max[t]));
        std::printf("\n           alpha       ");
        for (int t = 0; t < tiles; ++t)
            std::printf(" %.6g", static_cast<double>(page_alpha[t]));
        std::printf("\n           p_sum       ");
        for (int t = 0; t < tiles; ++t)
            std::printf(" %.6g", static_cast<double>(page_psum[t]));
        std::printf("\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL fused attention: %s\n", error.what());
        return 1;
    }
}
