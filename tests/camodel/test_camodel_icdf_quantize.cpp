// Tier 2 CAModel runner for the inverse-CDF int4x2 quantization kernel.
//
// Launches the real camodel_icdf_quantize_{fp16,bf16} symbols on the guarded
// fixture, then checks every packed output byte against the double oracle,
// per tile (not just end-of-run), plus GM guard bands head and tail.
//
// The Tier 1 sensitivity analysis (tests/cpu/test_icdf_quantize_probe.cpp) bounds
// the hardware deviation of the chain below the fixture's 1/16 boundary
// margin, so this check runs with zero tolerance: every byte must match.
// Runtime --tiles 4 is the ring-depth boundary; 8 wraps the depth-2 ring twice.

#include "../cpu/icdf_fixture.hpp"
#include "camodel_common.hpp"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>

extern "C" void launch_camodel_icdf_quantize_fp16(void*, std::uint16_t*,
                                                   std::uint8_t*, std::uint32_t);
extern "C" void launch_camodel_icdf_quantize_bf16(void*, std::uint16_t*,
                                                   std::uint8_t*, std::uint32_t);

namespace {

void usage(FILE* output, const char* program)
{
    std::fprintf(output, "Usage: %s [--tiles N] [fp16|bf16|both]\n"
                         "  N must be at least 2 (default: 4).\n", program);
}

void run(bool bf16, camodel::Session& session, std::uint32_t tiles)
{
    icdf::Fixture fx = icdf::make_fixture(bf16, tiles);
    icdf::assert_properties(fx);

    camodel::GuardedDeviceBuffer in(static_cast<std::size_t>(tiles) * icdf::kTileInBytes);
    camodel::GuardedDeviceBuffer out(static_cast<std::size_t>(tiles) * icdf::kTileOutBytes);
    std::memcpy(in.host_payload(), fx.in_bits.data(), fx.in_bits.size() * 2);

    in.upload();
    if (bf16)
        launch_camodel_icdf_quantize_bf16(session.stream,
            static_cast<std::uint16_t*>(in.payload()),
            static_cast<std::uint8_t*>(out.payload()), tiles);
    else
        launch_camodel_icdf_quantize_fp16(session.stream,
            static_cast<std::uint16_t*>(in.payload()),
            static_cast<std::uint8_t*>(out.payload()), tiles);
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
    std::uint32_t tiles = 4;
    std::string_view only = "both";
    bool saw_tiles = false;
    bool saw_mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argc == 2 ? stdout : stderr, argv[0]);
            return argc == 2 ? 0 : 2;
        }
        if (arg == "--tiles" && !saw_tiles && i + 1 < argc) {
            const std::string_view value = argv[++i];
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), tiles);
            if (parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                tiles >= 2) {
                saw_tiles = true;
                continue;
            }
        } else if (!saw_mode && (arg == "fp16" || arg == "bf16" || arg == "both")) {
            only = arg;
            saw_mode = true;
            continue;
        }
        usage(stderr, argv[0]);
        return 2;
    }
    try {
        camodel::Session session;
        std::printf("ICDF quantize tiles: %u\n", tiles);
        if (only == "fp16" || only == "both") run(false, session, tiles);
        if (only == "bf16" || only == "both") run(true, session, tiles);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native icdf quantize: %s\n", error.what());
        return 1;
    }
}
