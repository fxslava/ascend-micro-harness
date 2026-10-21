// Boundary runner for the fused TurboQuant paged-attention kernel at N = 3
// pages. Checks three independent things:
//
//   1. the whole 16x16 score panel of every page, bit-exactly, which verifies
//      the dequant4 -> GM workspace -> L1(ZN) -> L0B -> Mmad -> Fixpipe path;
//   2. the online-softmax state (m, l) and the FP32 normalised context against
//      a CPU attention oracle, in ULP;
//   3. the FP16 context actually stored by the outflow leaf.
//
// The fixture is chosen so the score panel is exactly representable in FP32
// (every entry is a multiple of 1/128), so step 1 can demand bit equality, and
// so the three pages are numerically distinct with strictly increasing page
// maxima. That last property matters: it forces alpha < 1 on every page after
// the first, and it means a run-ahead or buffer-reuse hazard that feeds a page
// the wrong tile's data cannot hide behind identical operands.

#include "camodel_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

extern "C" void launch_camodel_fused_attention(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, std::uint32_t*,
    std::uint16_t*, float*, float*, float*, std::uint16_t*);

namespace {

constexpr int kTiles = 3;
constexpr int kDim = 16;
constexpr int kKeys = 16;
constexpr int kPanel = kDim * kKeys;
constexpr int kPackedBytes = kPanel / 2;
constexpr float kKeyScale = 0.5f;
constexpr float kValueScale = 0.25f;

// Fixture generators. These mirror the device-side packing contract exactly:
// k nibble index (d * kKeys + key), v nibble index (ch * kKeys + key).
int key_nibble(int t, int d, int key) { return (d * 8 + key + t * (d + 1) * 14) % 16; }
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

}  // namespace

int main()
{
    try {
        camodel::Session session;
        camodel::GuardedDeviceBuffer q(kDim * kKeys * 2);
        camodel::GuardedDeviceBuffer k_packed(kTiles * kPackedBytes);
        camodel::GuardedDeviceBuffer v_packed(kTiles * kPackedBytes);
        camodel::GuardedDeviceBuffer offsets(kPanel * 4);
        camodel::GuardedDeviceBuffer ws_k(kTiles * kPanel * 2);
        camodel::GuardedDeviceBuffer out_scores(kTiles * kPanel * 4);
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

        // Pack two nibbles per byte, low nibble first.
        for (int t = 0; t < kTiles; ++t)
            for (int i = 0; i < kPackedBytes; ++i) {
                const int lo_index = 2 * i, hi_index = 2 * i + 1;
                const int k_lo = key_nibble(t, lo_index / kKeys, lo_index % kKeys);
                const int k_hi = key_nibble(t, hi_index / kKeys, hi_index % kKeys);
                const int v_lo = value_nibble(t, lo_index / kKeys, lo_index % kKeys);
                const int v_hi = value_nibble(t, hi_index / kKeys, hi_index % kKeys);
                kv[t * kPackedBytes + i] = static_cast<std::uint8_t>((k_hi << 4) | k_lo);
                vv[t * kPackedBytes + i] = static_cast<std::uint8_t>((v_hi << 4) | v_lo);
            }
        for (int ch = 0; ch < kDim; ++ch)
            for (int key = 0; key < kKeys; ++key)
                ov[ch * kKeys + key] = static_cast<std::uint32_t>(key * sizeof(float));

        q.upload(); k_packed.upload(); v_packed.upload(); offsets.upload();
        ws_k.upload(); out_scores.upload(); out_state.upload();
        out_normalized.upload(); out_context.upload();

        launch_camodel_fused_attention(
            session.stream, static_cast<std::uint16_t*>(q.payload()),
            static_cast<std::int8_t*>(k_packed.payload()),
            static_cast<std::int8_t*>(v_packed.payload()),
            static_cast<std::uint32_t*>(offsets.payload()),
            static_cast<std::uint16_t*>(ws_k.payload()),
            static_cast<float*>(out_scores.payload()),
            static_cast<float*>(out_state.payload()),
            static_cast<float*>(out_normalized.payload()),
            static_cast<std::uint16_t*>(out_context.payload()));
        camodel::check(aclrtSynchronizeStream(session.stream), "fused attention synchronize");

        q.download(); k_packed.download(); v_packed.download(); offsets.download();
        ws_k.download(); out_scores.download(); out_state.download();
        out_normalized.download(); out_context.download();
        q.assert_guards(); k_packed.assert_guards(); v_packed.assert_guards();
        offsets.assert_guards(); ws_k.assert_guards(); out_scores.assert_guards();
        out_state.assert_guards(); out_normalized.assert_guards();
        out_context.assert_guards();

        // ---- Oracle -------------------------------------------------------
        // Q and the dequantised key page are both half and exactly
        // representable, and every product and partial sum is a multiple of
        // 1/128 well inside FP32, so the reference score panel is exact.
        static float reference_scores[kTiles][kDim][kKeys];
        for (int t = 0; t < kTiles; ++t)
            for (int row = 0; row < kDim; ++row)
                for (int key = 0; key < kKeys; ++key) {
                    float sum = 0.0f;
                    for (int d = 0; d < kKeys; ++d)
                        sum += half_to_float(qv[row * kKeys + d]) *
                               (static_cast<float>(key_nibble(t, d, key)) * kKeyScale);
                    reference_scores[t][row][key] = sum;
                }

        float m = 0.0f, l = 0.0f, acc[kDim] = {};
        for (int t = 0; t < kTiles; ++t) {
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
        }
        float context[kDim];
        for (int ch = 0; ch < kDim; ++ch) context[ch] = acc[ch] / l;

        // ---- 1. Score panels, bit exact -----------------------------------
        const auto* got_scores = static_cast<const float*>(out_scores.host_payload());
        for (int t = 0; t < kTiles; ++t)
            for (int row = 0; row < kDim; ++row)
                for (int key = 0; key < kKeys; ++key) {
                    const float want = reference_scores[t][row][key];
                    const float have = got_scores[t * kPanel + row * kKeys + key];
                    if (bits(have) != bits(want)) {
                        std::fprintf(stderr,
                                     "score tile=%d row=%d key=%d got=%.9g expected=%.9g\n",
                                     t, row, key, have, want);
                        throw std::runtime_error("Cube score panel mismatch");
                    }
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

        const std::size_t guard_bytes = 9 * 2 * camodel::kGuardBytes;
        std::printf("PASS fused attention: %d pages, %d/%d score entries bit-exact, "
                    "max %.3f ULP (worst operand %s), %d/%d FP16 context within 1 half ULP, "
                    "%zu/%zu GM guard bytes intact\n",
                    kTiles, kTiles * kPanel, kTiles * kPanel,
                    static_cast<double>(max_ulp), worst, kDim, kDim,
                    guard_bytes, guard_bytes);
        std::printf("  final m=%.9g l=%.9g  context[0]=%.9g context[15]=%.9g\n",
                    static_cast<double>(got_state[0]), static_cast<double>(got_state[8]),
                    static_cast<double>(got_norm[0]), static_cast<double>(got_norm[15]));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL fused attention: %s\n", error.what());
        return 1;
    }
}
