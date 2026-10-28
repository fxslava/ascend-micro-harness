// CAModel runner for the three butterfly implementations.
// Ascend950PR_9599, CANN 9.2.0-beta.2, KERNEL_TYPE_AIV_ONLY, one AIV.
//
//   --variant blocked|gather|shuffle   which leaf to launch (default: gather)
//   --rows N                           rows per launch, 1..16 (default 16)
//   --reps N                           repetitions of the transform (default 1)
//
// One variant per process. The directly linked CAModel drops its per-core
// instruction dumps in the current directory and overwrites them on the next
// run, so the comparison is three processes and three archived dump sets, not
// three launches in one.
//
// The gate is bit equality against rot_oracle::staged, not a tolerance. Every
// stage is an FP32 add or subtract of exactly representable values, the
// fixture is dyadic, and the Tier-1 probe has already established that the
// stage cascade reproduces the dense 256 x 256 matrix -- so a device that
// disagrees in one bit disagrees about the transform, not about rounding.
//
// Poison, not zero: the output panel is filled with 0x7FC0BEEF before the
// launch. It is a quiet NaN, so a row the kernel never wrote shows up as NaN
// in the comparison rather than as a plausible small number, and the sweep
// below can tell "not written" from "written wrongly".

#include "camodel_common.hpp"
#include "../common/rotate_butterfly_oracle.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

extern "C" void launch_camodel_rotate_butterfly_blocked(void*, float*, float*, float*,
                                                        std::uint32_t*, float*,
                                                        std::uint32_t, std::uint32_t);
extern "C" void launch_camodel_rotate_butterfly_gather(void*, float*, float*, float*,
                                                       std::uint32_t*, float*, std::uint32_t,
                                                       std::uint32_t);
extern "C" void launch_camodel_rotate_butterfly_shuffle(void*, float*, float*, float*,
                                                        std::uint32_t*, float*,
                                                        std::uint32_t, std::uint32_t);

namespace {

using namespace rot_oracle;

constexpr std::uint32_t kMaxRows = 16;
constexpr std::uint32_t kPoison = 0x7FC0BEEFu;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t bits_of(float value)
{
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

std::uint32_t mix(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Dyadic, distinct per (row, channel), and distinct per row so a leaf that
// transformed the wrong row cannot produce the right answer.
float activation(std::uint32_t row, int channel)
{
    const std::uint32_t h =
        mix(0x5bd1e995u ^ (row * 0x9E3779B9u) ^ (static_cast<std::uint32_t>(channel) * 0x27D4EB2Du));
    const int n = static_cast<int>(h % 253u) - 126;
    return std::ldexp(static_cast<float>(n), -3);
}

int parse_u32(std::string_view text, std::uint32_t& out)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc() ? 0 : -1;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string_view variant = "gather";
    std::uint32_t rows = kMaxRows;
    std::uint32_t reps = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--variant" && i + 1 < argc) {
            variant = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            if (parse_u32(argv[++i], rows) != 0 || rows == 0 || rows > kMaxRows) {
                std::fprintf(stderr, "--rows must be 1..%u\n", kMaxRows);
                return 2;
            }
        } else if (arg == "--reps" && i + 1 < argc) {
            if (parse_u32(argv[++i], reps) != 0 || reps == 0) {
                std::fprintf(stderr, "--reps must be positive\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %.*s\n", static_cast<int>(arg.size()),
                         arg.data());
            return 2;
        }
    }
    const bool is_blocked = variant == "blocked";
    const bool is_gather = variant == "gather";
    const bool is_shuffle = variant == "shuffle";
    if (!is_blocked && !is_gather && !is_shuffle) {
        std::fprintf(stderr, "--variant must be blocked, gather or shuffle\n");
        return 2;
    }

    std::printf("rotate butterfly CAModel: variant %.*s, rows %u, reps %u, d %d\n",
                static_cast<int>(variant.size()), variant.data(), rows, reps, kChannels);

    const std::vector<float> signs = canonical_signs();
    const std::vector<std::uint32_t> offsets = blocked_gather_offsets();
    const std::vector<float> gsigns = blocked_gather_signs();

    std::vector<float> host_in(static_cast<std::size_t>(kMaxRows) * kChannels, 0.0f);
    for (std::uint32_t r = 0; r < kMaxRows; ++r)
        for (int c = 0; c < kChannels; ++c)
            host_in[static_cast<std::size_t>(r) * kChannels + c] = activation(r, c);

    camodel::Session session;
    camodel::GuardedDeviceBuffer d_in(host_in.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_out(host_in.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_sign(signs.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_offsets(offsets.size() * sizeof(std::uint32_t));
    camodel::GuardedDeviceBuffer d_gsigns(gsigns.size() * sizeof(float));

    std::memcpy(d_in.host_payload(), host_in.data(), host_in.size() * sizeof(float));
    std::memcpy(d_sign.host_payload(), signs.data(), signs.size() * sizeof(float));
    std::memcpy(d_offsets.host_payload(), offsets.data(),
                offsets.size() * sizeof(std::uint32_t));
    std::memcpy(d_gsigns.host_payload(), gsigns.data(), gsigns.size() * sizeof(float));
    {
        auto* poison = static_cast<std::uint32_t*>(d_out.host_payload());
        for (std::size_t i = 0; i < d_out.payload_bytes / 4; ++i) poison[i] = kPoison;
    }
    d_in.upload();
    d_out.upload();
    d_sign.upload();
    d_offsets.upload();
    d_gsigns.upload();

    auto* p_in = static_cast<float*>(d_in.payload());
    auto* p_out = static_cast<float*>(d_out.payload());
    auto* p_sign = static_cast<float*>(d_sign.payload());
    auto* p_offsets = static_cast<std::uint32_t*>(d_offsets.payload());
    auto* p_gsigns = static_cast<float*>(d_gsigns.payload());

    if (is_blocked)
        launch_camodel_rotate_butterfly_blocked(session.stream, p_in, p_out, p_sign,
                                                p_offsets, p_gsigns, rows, reps);
    else if (is_gather)
        launch_camodel_rotate_butterfly_gather(session.stream, p_in, p_out, p_sign, p_offsets,
                                               p_gsigns, rows, reps);
    else
        launch_camodel_rotate_butterfly_shuffle(session.stream, p_in, p_out, p_sign,
                                                p_offsets, p_gsigns, rows, reps);
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    d_out.download();
    d_in.download();
    d_sign.download();
    d_in.assert_guards();
    d_out.assert_guards();
    d_sign.assert_guards();
    d_offsets.download();
    d_offsets.assert_guards();
    d_gsigns.download();
    d_gsigns.assert_guards();

    const auto* device = static_cast<const float*>(d_out.host_payload());
    std::size_t mismatches = 0;
    std::size_t still_poison = 0;
    int first_row = -1;
    int first_channel = -1;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::vector<float> want =
            staged(Direction::Ingress, host_in.data() + static_cast<std::size_t>(r) * kChannels,
                   signs.data());
        for (int c = 0; c < kChannels; ++c) {
            const float got = device[static_cast<std::size_t>(r) * kChannels + c];
            if (bits_of(got) == kPoison) ++still_poison;
            if (bits_of(got) != bits_of(want[c])) {
                if (first_row < 0) {
                    first_row = static_cast<int>(r);
                    first_channel = c;
                    std::fprintf(stderr,
                                 "  first divergence row %u channel %d: device %.6f "
                                 "(0x%08x) oracle %.6f (0x%08x)\n",
                                 r, c, static_cast<double>(got), bits_of(got),
                                 static_cast<double>(want[c]), bits_of(want[c]));
                }
                ++mismatches;
            }
        }
    }
    expect(mismatches == 0, "the rotated panel is bit-exact against the stage cascade");
    expect(still_poison == 0, "every owned output word was written");

    // Rows the launch did not claim must still hold the poison pattern: a leaf
    // that ran its row loop past `rows` is a wrong answer the comparison above
    // would never see, because it only reads the rows it asked for.
    std::size_t beyond_written = 0;
    for (std::uint32_t r = rows; r < kMaxRows; ++r)
        for (int c = 0; c < kChannels; ++c)
            if (bits_of(device[static_cast<std::size_t>(r) * kChannels + c]) != kPoison)
                ++beyond_written;
    expect(beyond_written == 0, "no row beyond the launch's row count was written");

    // The input panel and the tables are read-only operands.
    {
        const auto* back = static_cast<const float*>(d_in.host_payload());
        std::size_t changed = 0;
        for (std::size_t i = 0; i < host_in.size(); ++i)
            if (bits_of(back[i]) != bits_of(host_in[i])) ++changed;
        expect(changed == 0, "the source panel was not modified");
    }

    std::printf("  rows checked %u, mismatching words %zu, unwritten %zu, beyond-range %zu\n",
                rows, mismatches, still_poison, beyond_written);
    if (first_row >= 0)
        std::printf("  first divergence at (row %d, channel %d)\n", first_row, first_channel);
    if (failures == 0) {
        std::printf("rotate butterfly CAModel (%.*s): PASS\n", static_cast<int>(variant.size()),
                    variant.data());
        return 0;
    }
    std::fprintf(stderr, "rotate butterfly CAModel (%.*s): %d FAILURES\n",
                 static_cast<int>(variant.size()), variant.data(), failures);
    return 1;
}
