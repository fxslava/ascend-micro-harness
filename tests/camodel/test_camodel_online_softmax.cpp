#include "camodel_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

extern "C" void launch_camodel_online_softmax(void*, float*, float*,
                                               std::uint32_t*, float*);

int main()
{
    try {
        camodel::Session session;
        camodel::GuardedDeviceBuffer scores(4 * 16 * 4), values(4 * 256 * 4);
        camodel::GuardedDeviceBuffer offsets(256 * 4), output(32 * 4);
        auto* sv = static_cast<float*>(scores.host_payload());
        auto* vv = static_cast<float*>(values.host_payload());
        auto* ov = static_cast<std::uint32_t*>(offsets.host_payload());
        for (int tile = 0; tile < 4; ++tile) {
            for (int j = 0; j < 16; ++j)
                sv[tile * 16 + j] = 0.125f * static_cast<float>((tile * 7 + j * 3) % 17 - 8);
            for (int col = 0; col < 16; ++col)
                for (int j = 0; j < 16; ++j)
                    vv[tile * 256 + col * 16 + j] =
                        0.0625f * static_cast<float>((tile + col * 3 + j * 5) % 19 - 9);
        }
        for (int col = 0; col < 16; ++col)
            for (int j = 0; j < 16; ++j)
                ov[col * 16 + j] = j * sizeof(float);
        scores.upload(); values.upload(); offsets.upload();
        launch_camodel_online_softmax(session.stream,
            static_cast<float*>(scores.payload()), static_cast<float*>(values.payload()),
            static_cast<std::uint32_t*>(offsets.payload()),
            static_cast<float*>(output.payload()));
        camodel::check(aclrtSynchronizeStream(session.stream), "softmax synchronize");
        scores.download(); values.download(); offsets.download(); output.download();
        scores.assert_guards(); values.assert_guards();
        offsets.assert_guards(); output.assert_guards();
        float m = 0.0f, l = 0.0f, acc[16]{};
        for (int tile = 0; tile < 4; ++tile) {
            const float tile_max = *std::max_element(sv + tile * 16, sv + tile * 16 + 16);
            const float next_m = std::max(m, tile_max);
            const float alpha = std::exp(m - next_m);
            float p[16], p_sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                p[j] = std::exp(sv[tile * 16 + j] - next_m);
                p_sum += p[j];
            }
            l = l * alpha + p_sum;
            for (int col = 0; col < 16; ++col) {
                float weighted = 0.0f;
                for (int j = 0; j < 16; ++j)
                    weighted += p[j] * vv[tile * 256 + col * 16 + j];
                acc[col] = acc[col] * alpha + weighted;
            }
            m = next_m;
        }
        const auto* got = static_cast<const float*>(output.host_payload());
        float max_error = 0.0f;
        float max_ulp = 0.0f;
        const char* max_ulp_label = "none";
        auto compare = [&](const char* label, int index, float actual, float expected) {
            const float error = std::abs(actual - expected);
            max_error = std::max(max_error, error);
            // One ULP at the oracle's magnitude, so the reported figure is
            // comparable across m (order 1) and l (order 30).
            const float magnitude = std::abs(expected);
            const float ulp = std::nextafter(magnitude, HUGE_VALF) - magnitude;
            if (ulp > 0.0f) {
                const float ulps = error / ulp;
                if (ulps > max_ulp) { max_ulp = ulps; max_ulp_label = label; }
            }
            if (!std::isfinite(actual) || error > 2e-4f) {
                std::fprintf(stderr, "%s index=%d got=%.9g expected=%.9g error=%.9g\n",
                             label, index, actual, expected, error);
                throw std::runtime_error("native softmax precision mismatch");
            }
        };
        compare("m", 0, got[0], m);
        compare("l", 0, got[8], l);
        for (int col = 0; col < 16; ++col)
            compare("acc", col, got[16 + col], acc[col]);
        std::printf("PASS native online softmax: 4 tiles, max absolute error %.9g, "
                    "max %.3f ULP (worst operand %s), "
                    "512/512 GM guard bytes intact\n",
                    max_error, static_cast<double>(max_ulp), max_ulp_label);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native online softmax: %s\n", error.what());
        return 1;
    }
}
