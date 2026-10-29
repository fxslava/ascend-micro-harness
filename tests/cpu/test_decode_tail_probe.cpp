// Tier-1 host probe for the unaligned-context fixture and its masked oracle.
//
// Why this exists, and what the device test cannot do without it
// -------------------------------------------------------------
// `tests/npu/test_npu_decode_single_token_unaligned.cpp` compares a kernel
// against `splitk_decode_test::Fixture::expected` at an unaligned `seq_len`.
// That comparison is only evidence if two things are true of the fixture
// itself, and neither is visible from the device side:
//
//   1. The oracle does not read the dead slots. They hold 0x7FC0BEEF, a quiet
//      NaN, so if the reference touched them `expected` would be NaN and the
//      device test would fail for a reason that has nothing to do with the
//      kernel.
//
//   2. The mask is NUMERICALLY OBSERVABLE. If including the dead tokens made
//      no difference to the result, a kernel that ignored `seq_len` entirely
//      would pass the device test and the whole exercise would prove nothing.
//      This is the same argument the aligned suite's `page_begin` coverage
//      rests on, and it has to be made on the host: the device test is handed
//      one oracle and can only agree or disagree with it.
//
// So this probe asserts that the masked reference is finite, that the slots
// past `seq_len` really do hold the canary in both scale planes, and that the
// masked and unmasked references DIFFER at every unaligned rung -- by a
// margin far wider than the BF16 tolerance the device comparison uses.
//
// No NPU, no simulator, no toolkit: plain C++17 over the same fixture header
// the device test includes.

#include "../common/splitk_decode_fixture.hpp"
#include "../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tail = turboquant_decode_tail;

namespace {

int failures = 0;

void report(const char* what, int seq_len)
{
    std::printf("FAIL %s at S=%d\n", what, seq_len);
    ++failures;
}

bool IsCanary(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits == 0x7FC0BEEFu;
}

}  // namespace

int main()
{
    // Small geometry: the point is the masking, not the parallel shape, and a
    // single head of two rows keeps the FP64 reference cheap enough to run
    // every rung of the ladder.
    constexpr int kRequests = 1, kHeads = 1, kRows = 2;
    constexpr int kPage = splitk_decode_test::kPageTokens;

    const std::vector<int> ladder = {1, 7, 15, 17, 31, 33, 127, 129, 255, 257, 1023};

    std::printf("=== unaligned fixture and masked oracle ===\n");
    std::printf("  %-6s %-7s %-9s %-12s %s\n", "S", "pages", "dead", "max |diff|",
                "tolerance");

    int rungs = 0;
    for (const int seq_len : ladder) {
        splitk_decode_test::Fixture masked(seq_len, kRequests, kHeads, kRows, kPage);
        // The same capacity with every token live: what an oracle that ignored
        // seq_len would produce.
        splitk_decode_test::Fixture whole(masked.tokens, kRequests, kHeads, kRows,
                                          kPage);

        if (masked.seq_len != seq_len) report("fixture ignored the live count", seq_len);
        if (masked.tokens != whole.tokens || masked.pages != whole.pages)
            report("capacity differs between the two fixtures", seq_len);
        if (masked.pages != static_cast<int>(tail::PagesForTokens(
                                static_cast<std::uint32_t>(seq_len), kPage)))
            report("fixture page count disagrees with the tail geometry", seq_len);

        // 1. the canary is really there, in both planes, for every dead slot,
        //    and is NOT there for any live one.
        const std::size_t half = masked.scales.size() / 2;
        int dead = 0;
        for (int page = 0; page < masked.pages; ++page) {
            const int physical = masked.pages - 1 - page;  // the fixture's permutation
            for (int t = 0; t < kPage; ++t) {
                const int logical = page * kPage + t;
                const auto at =
                    (static_cast<std::size_t>(physical) * kHeads + 0) * kPage + t;
                const bool want_canary = logical >= seq_len;
                if (want_canary) ++dead;
                if (IsCanary(masked.scales[at]) != want_canary ||
                    IsCanary(masked.scales[half + at]) != want_canary) {
                    report("a scale slot is not canary exactly when it is dead",
                           seq_len);
                    break;
                }
            }
        }
        if (dead != masked.tokens - seq_len)
            report("wrong number of dead slots", seq_len);

        // 2. the masked oracle never saw them.
        for (const float value : masked.expected)
            if (!std::isfinite(value)) {
                report("the masked reference is not finite -- the oracle read a "
                       "dead slot", seq_len);
                break;
            }

        // 3. the mask is observable, by a margin the device tolerance cannot
        //    hide. The device comparison allows 0.005 * row_max + 1e-6.
        float max_diff = 0, row_max = 0;
        for (std::size_t at = 0; at < masked.expected.size(); ++at) {
            max_diff = std::max(max_diff,
                                std::abs(masked.expected[at] - whole.expected[at]));
            row_max = std::max(row_max, std::abs(masked.expected[at]));
        }
        const float tolerance = 0.005f * row_max + 1e-6f;
        if (!(max_diff > tolerance))
            report("masking the tail changes nothing the device test could see",
                   seq_len);

        std::printf("  %-6d %-7d %-9d %-12.6g %.6g\n", seq_len, masked.pages, dead,
                    static_cast<double>(max_diff), static_cast<double>(tolerance));
        ++rungs;
    }

    // A negative control for the observability check itself: at an ALIGNED
    // length there are no dead slots, the two fixtures are identical, and the
    // difference must be exactly zero. If this reported a difference, the
    // comparison above would be measuring fixture noise rather than masking.
    {
        const int aligned = 4 * kPage;
        splitk_decode_test::Fixture a(aligned, kRequests, kHeads, kRows, kPage);
        splitk_decode_test::Fixture b(aligned, kRequests, kHeads, kRows, kPage);
        float diff = 0;
        for (std::size_t at = 0; at < a.expected.size(); ++at)
            diff = std::max(diff, std::abs(a.expected[at] - b.expected[at]));
        if (a.seq_len != aligned || a.tokens != aligned || diff != 0.0f) {
            std::printf("FAIL negative control: an aligned length is not a no-op "
                        "(seq_len=%d tokens=%d diff=%.9g)\n",
                        a.seq_len, a.tokens, static_cast<double>(diff));
            ++failures;
        } else {
            std::printf("PASS rejected negative control: at S=%d, a multiple of the "
                        "page, there are no dead slots and the masked and unmasked "
                        "references are bit-identical, so the differences above are "
                        "the mask and not fixture noise\n",
                        aligned);
        }
    }

    if (failures != 0) {
        std::printf("FAIL unaligned fixture probe: %d problems\n", failures);
        return 1;
    }
    std::printf("PASS unaligned fixture probe: %d rungs; every slot past seq_len "
                "carries 0x7FC0BEEF in both scale planes, the masked FP64 reference "
                "is finite (so it never read one), and masking changes the gated "
                "context by more than the device tolerance at every rung -- a "
                "kernel that ignored seqLen could not pass\n",
                rungs);
    return 0;
}
