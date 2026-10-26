// Tier 2 CAModel runner for the inverse-CDF int4x2 quantization kernel.
//
// Launches the real camodel_icdf_quantize_{fp16,bf16} symbols on the guarded
// fixture, then checks every packed output byte against the double oracle,
// per tile (not just end-of-run), plus GM guard bands head and tail.
//
// The Tier 1 sensitivity analysis (tests/cpu/test_icdf_quantize_probe.cpp) bounds
// the hardware deviation of the chain below the fixture's 1/16 boundary
// margin, so this check runs with zero tolerance: every byte must match.
// ICDF_QUANT_TILES is a build parameter (4 = ring-depth boundary, 8 = wraps
// the depth-2 ring twice for V3).

#include "../cpu/icdf_fixture.hpp"
#include "camodel_common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef ICDF_QUANT_TILES
#define ICDF_QUANT_TILES 4
#endif

extern "C" void launch_camodel_icdf_quantize_fp16(void*, std::uint16_t*, std::uint8_t*);
extern "C" void launch_camodel_icdf_quantize_bf16(void*, std::uint16_t*, std::uint8_t*);

namespace {

void run(bool bf16, camodel::Session& session)
{
    const std::uint32_t tiles = ICDF_QUANT_TILES;
    icdf::Fixture fx = icdf::make_fixture(bf16, tiles);
    icdf::assert_properties(fx);

    camodel::GuardedDeviceBuffer in(static_cast<std::size_t>(tiles) * icdf::kTileInBytes);
    camodel::GuardedDeviceBuffer out(static_cast<std::size_t>(tiles) * icdf::kTileOutBytes);
    std::memcpy(in.host_payload(), fx.in_bits.data(), fx.in_bits.size() * 2);

    in.upload();
    if (bf16)
        launch_camodel_icdf_quantize_bf16(session.stream,
            static_cast<std::uint16_t*>(in.payload()),
            static_cast<std::uint8_t*>(out.payload()));
    else
        launch_camodel_icdf_quantize_fp16(session.stream,
            static_cast<std::uint16_t*>(in.payload()),
            static_cast<std::uint8_t*>(out.payload()));
    camodel::check(aclrtSynchronizeStream(session.stream), "icdf synchronize");
    in.download(); out.download();
    in.assert_guards(); out.assert_guards();

    const auto* got = static_cast<const std::uint8_t*>(out.host_payload());
    std::uint64_t exact = 0;
    std::uint32_t mismatches = 0;
    // Tile-0 head dump: enough context to read a layout permutation directly
    // out of a failure pattern.
    std::fprintf(stderr, "%s tile 0 first 16 got:    ", bf16 ? "BF16" : "FP16");
    for (std::uint32_t b = 0; b < 16; ++b)
        std::fprintf(stderr, "%02x", got[b]);
    std::fprintf(stderr, "\n%s tile 0 first 16 oracle: ", bf16 ? "BF16" : "FP16");
    for (std::uint32_t b = 0; b < 16; ++b)
        std::fprintf(stderr, "%02x", fx.oracle_packed[b]);
    std::fprintf(stderr, "\n");
    const std::uint64_t total =
        static_cast<std::uint64_t>(tiles) * icdf::kTileOutBytes;
    for (std::uint32_t t = 0; t < tiles; ++t) {
        for (std::uint32_t b = 0; b < icdf::kTileOutBytes; ++b) {
            const std::uint64_t flat =
                static_cast<std::uint64_t>(t) * icdf::kTileOutBytes + b;
            if (got[flat] != fx.oracle_packed[flat]) {
                const std::uint32_t elem0 = t * icdf::kTileElems + 2 * b;
                if (++mismatches <= 32) {
                    std::fprintf(stderr,
                                 "%s tile=%u byte=%u got=0x%02x want=0x%02x "
                                 "(elements %u,%u margins %.4f,%.4f)\n",
                                 bf16 ? "BF16" : "FP16", t, b, got[flat],
                                 fx.oracle_packed[flat], elem0, elem0 + 1,
                                 static_cast<double>(fx.margins[elem0]),
                                 static_cast<double>(fx.margins[elem0 + 1]));
                }
            } else {
                ++exact;
            }
        }
    }
    if (mismatches)
        throw std::runtime_error("icdf quantize byte mismatch");
    std::printf("PASS native icdf %s %u tiles: %llu/%llu packed bytes bit-exact "
                "(zero tolerance), %u/%u GM guard bytes intact\n",
                bf16 ? "BF16" : "FP16", tiles,
                static_cast<unsigned long long>(exact),
                static_cast<unsigned long long>(total),
                static_cast<unsigned>(2 * camodel::kGuardBytes),
                static_cast<unsigned>(2 * camodel::kGuardBytes));
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        camodel::Session session;
        const std::string only = argc > 1 ? argv[1] : "both";
        if (only == "fp16" || only == "both") run(false, session);
        if (only == "bf16" || only == "both") run(true, session);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native icdf quantize: %s\n", error.what());
        return 1;
    }
}
