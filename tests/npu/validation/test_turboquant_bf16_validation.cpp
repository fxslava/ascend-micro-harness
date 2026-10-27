// Physical-device validation for turboquant_reshape_and_cache_bf16.
//
// The BF16 instance is checked against the *same* golden oracle as the INT8
// instance, with no BF16-specific path at all. That is deliberate and it is
// the strongest available check: the fixture's activations are integers, a
// BF16 significand holds every one of them exactly, a product of two is exact
// in FP32 and a sum of 256 such products is an integer below 2^24, so the
// Cube's accumulation order cannot matter. For the same inputs the BF16 kernel
// must therefore produce byte-identical pages and bit-identical scales to the
// INT8 kernel, and tests/common/turboquant_oracle.hpp predicts both.
//
// With general (non-integer) BF16 activations none of that holds -- the FP32
// accumulation order is unspecified and no integer oracle can reproduce it --
// so such a test would have to degrade to a cosine plus a relative-L2 bound,
// which no longer detects a wrong-strip read the way nibble comparison does.
// That test, if it is ever wanted, belongs in a separate clearly labelled
// runner; it must not replace this one.
//
// What the ladder covers, and why 33 is on it
// -------------------------------------------
// Strips are driven in windows of at most `--window` (default 32), because the
// kernel allows one strip per grid block per launch and a grid wider than the
// device's AI core count does not dispatch. The rungs are not arbitrary:
//
//    1   cold start: one block, one launch, no window boundary at all.
//   32   one full wave in a single launch -- the widest grid that dispatches
//        at the default window, and the rung where every mode-4 flag id is
//        used once by 32 different physical cores.
//   33   ONE strip in the second wave. This is the rung that matters: it is a
//        ragged tail launch of a single block immediately after a full wave,
//        so a physical core that has already used its mode-4 ids must re-arm
//        them, and a token leaked by the first launch shows up as a hang or a
//        wrong page here and nowhere else.
//   64   two full waves, so every core runs twice and the tail is not ragged.
//        33 and 64 fail differently: 33 catches a leak that only a *partial*
//        second launch exposes, 64 catches one that needs a full reuse cycle.
//
// Pages are fragmented (numBlocks = 2 * strips + 2, strip s -> page 2s + 1),
// so the page index is never the grid block index and a stray write lands on a
// canary page rather than on a neighbour's correct page. The verifier reports
// per page and, when a page holds the wrong data, names the strip the data
// actually came from -- which is the signature of a block-index error and is
// invisible in an aggregate byte count.
//
// Usage:
//   test_turboquant_bf16_validation [--strips N]... [--tokens N]... [--window W]
//
// --strips and --tokens may each be repeated to build a custom ladder; with
// neither, the ladder is 1, 32, 33, 64 strips.

#include "../turboquant_npu_common.hpp"

#include <charconv>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string_view>
#include <vector>

namespace {

constexpr int kDefaultWindow = 32;
constexpr int kMaxStrips = 4096;

void usage(std::FILE* out, const char* argv0)
{
    std::fprintf(out,
                 "Usage: %s [--strips N]... [--tokens N]... [--window W]\n"
                 "  --strips N  validate a context of N aligned 16-token strips\n"
                 "              (repeatable; 1..%d)\n"
                 "  --tokens N  the same, given in tokens; N must be a multiple of %d\n"
                 "  --window W  at most W strips per kernel launch (default %d).\n"
                 "              A context wider than W is driven as several\n"
                 "              launches, which is how the second wave is reached.\n"
                 "  With no --strips or --tokens the ladder is 1, 32, 33, 64 strips.\n",
                 argv0, kMaxStrips, tq_npu::kBlockSize, kDefaultWindow);
}

bool parse_int(std::string_view text, int& out)
{
    const char* begin = text.data();
    const char* end = begin + text.size();
    int value = 0;
    const auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) return false;
    out = value;
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<int> ladder;
    int window = kDefaultWindow;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const bool is_strips = arg == "--strips";
        const bool is_tokens = arg == "--tokens";
        if (is_strips || is_tokens || arg == "--window") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", argv[i]);
                usage(stderr, argv[0]);
                return 2;
            }
            int value = 0;
            if (!parse_int(argv[i + 1], value)) {
                std::fprintf(stderr, "%s: not an integer: %s\n", argv[i], argv[i + 1]);
                return 2;
            }
            ++i;
            if (is_tokens) {
                if (value <= 0 || value % tq_npu::kBlockSize != 0) {
                    std::fprintf(stderr,
                                 "--tokens must be a positive multiple of %d, got %d\n",
                                 tq_npu::kBlockSize, value);
                    return 2;
                }
                value /= tq_npu::kBlockSize;
            }
            if (is_strips || is_tokens) {
                if (value <= 0 || value > kMaxStrips) {
                    std::fprintf(stderr, "strip count out of range: %d\n", value);
                    return 2;
                }
                ladder.push_back(value);
            } else {
                if (value <= 0) {
                    std::fprintf(stderr, "--window must be positive, got %d\n", value);
                    return 2;
                }
                window = value;
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(stdout, argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (ladder.empty()) ladder = {1, 32, 33, 64};

    try {
        tq_npu::Session session;
        std::printf("turboquant_reshape_and_cache_bf16 NPU validation, "
                    "window %d strip%s per launch\n",
                    window, window == 1 ? "" : "s");
        for (const int strips : ladder) {
            tq_npu::Fixture fixture(strips * tq_npu::kBlockSize);
            tq_npu::Bf16Device device(fixture);
            const int launches = device.launch(session.stream, fixture, window);
            tq_npu::check(aclrtSynchronizeStream(session.stream),
                          "aclrtSynchronizeStream");
            const int tail = strips % window;
            std::printf("\n%d strips in %d launch%s%s\n", strips, launches,
                        launches == 1 ? "" : "es",
                        launches > 1 && tail != 0 ? " (ragged tail wave)" : "");
            tq_npu::verify(fixture, device);
        }
        std::printf("\nPASS turboquant_reshape_and_cache_bf16: %zu ladder rungs, "
                    "bit-exact against the shared integer oracle\n",
                    ladder.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant BF16 NPU validation: %s\n", e.what());
        return 1;
    }
}
