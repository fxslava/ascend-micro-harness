#include "camodel_common.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

extern "C" void launch_camodel_cube16(void*, std::uint16_t*, std::uint16_t*, float*);
extern "C" void launch_camodel_cube32(void*, std::uint16_t*, std::uint16_t*, float*);

namespace {
std::uint16_t half_integer(int value)
{
    constexpr std::uint16_t values[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400};
    return values[value];
}

void run(int m, camodel::Session& session)
{
    camodel::GuardedDeviceBuffer a(m * 16 * 2), b(16 * 16 * 2), output(m * 16 * 4);
    auto* av = static_cast<std::uint16_t*>(a.host_payload());
    auto* bv = static_cast<std::uint16_t*>(b.host_payload());
    for (int row = 0; row < m; ++row)
        for (int k = 0; k < 16; ++k)
            av[row * 16 + k] = half_integer(1 + (row + k) % 2);
    for (int k = 0; k < 16; ++k)
        for (int col = 0; col < 16; ++col)
            bv[k * 16 + col] = half_integer(1 + (k + 2 * col) % 4);
    a.upload();
    b.upload();
    if (m == 16)
        launch_camodel_cube16(session.stream, static_cast<std::uint16_t*>(a.payload()),
                              static_cast<std::uint16_t*>(b.payload()),
                              static_cast<float*>(output.payload()));
    else
        launch_camodel_cube32(session.stream, static_cast<std::uint16_t*>(a.payload()),
                              static_cast<std::uint16_t*>(b.payload()),
                              static_cast<float*>(output.payload()));
    camodel::check(aclrtSynchronizeStream(session.stream), "cube synchronize");
    a.download(); b.download(); output.download();
    a.assert_guards(); b.assert_guards(); output.assert_guards();
    const auto* got = static_cast<const float*>(output.host_payload());
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < 16; ++col) {
            int expected = 0;
            for (int k = 0; k < 16; ++k)
                expected += (1 + (row + k) % 2) * (1 + (k + 2 * col) % 4);
            if (got[row * 16 + col] != static_cast<float>(expected)) {
                std::fprintf(stderr, "cube %dx16 row=%d col=%d got=%g expected=%d\n",
                             m, row, col, got[row * 16 + col], expected);
                throw std::runtime_error("cube result mismatch");
            }
        }
    std::printf("PASS native cube %dx16: %d/%d exact, 384/384 GM guard bytes intact\n",
                m, m * 16, m * 16);
}
}  // namespace

int main()
{
    try {
        camodel::Session session;
        run(16, session);
        run(32, session);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native cube: %s\n", error.what());
        return 1;
    }
}
