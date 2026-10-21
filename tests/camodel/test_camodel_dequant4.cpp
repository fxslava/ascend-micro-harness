#include "camodel_common.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

extern "C" void launch_camodel_dequant4(void*, std::int8_t*, float*);

int main()
{
    try {
        camodel::Session session;
        camodel::GuardedDeviceBuffer input(32), output(64 * sizeof(float));
        auto* packed = static_cast<std::uint8_t*>(input.host_payload());
        for (int i = 0; i < 32; ++i)
            packed[i] = static_cast<std::uint8_t>((((i * 7 + 3) & 15) << 4) | ((i * 5 + 1) & 15));
        input.upload();
        launch_camodel_dequant4(session.stream,
                                static_cast<std::int8_t*>(input.payload()),
                                static_cast<float*>(output.payload()));
        camodel::check(aclrtSynchronizeStream(session.stream), "dequant synchronize");
        input.download();
        output.download();
        input.assert_guards();
        output.assert_guards();
        const auto* result = static_cast<const float*>(output.host_payload());
        for (int i = 0; i < 64; ++i) {
            const auto nibble = (i & 1) ? packed[i / 2] >> 4 : packed[i / 2] & 15;
            const float expected = static_cast<float>(nibble) * 0.5f;
            if (result[i] != expected) {
                std::fprintf(stderr, "dequant index %d got %.9g expected %.9g\n", i,
                             result[i], expected);
                return 1;
            }
        }
        std::puts("PASS native dequant4: low nibble first, 64/64 exact, 128/128 GM guard bytes intact");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native dequant4: %s\n", error.what());
        return 1;
    }
}
