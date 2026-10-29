// Physical-device validation for turboquant_reshape_and_cache_a8.
//
// Same fixture, same golden oracle and same per-page verifier as the BF16
// instance next door; only the input encoding differs. It now runs the same
// wave ladder too. It used to run a fixed {16, 512, 1024}-token list with no
// CLI at all, which covered 1, 32 and 64 strips -- but the 64-strip rung went
// through a kernel whose grid was capped per launch, so a second wave was only
// ever reached as a *separate dispatch*. The INT8 instance carries the
// inter-wave protocol now (turboquant_wave_protocol.hpp, guidelines section
// 6.5), so each rung here is one monolithic launch, and 33 and 48 are on the
// ladder because a ragged second wave inside a single dispatch is the case
// neither 32 nor 64 reaches. tq_npu::default_ladder() documents every rung.
//
// Usage:
//   test_turboquant_a8_validation [--strips N]... [--tokens N]... [--single]

#include "../turboquant_npu_common.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv)
{
    tq_npu::Options opts;
    switch (tq_npu::parse_options(argc, argv, opts)) {
    case tq_npu::ParseResult::kExitOk: return 0;
    case tq_npu::ParseResult::kExitUsage: return 2;
    case tq_npu::ParseResult::kRun: break;
    }
    const std::vector<int> ladder =
        opts.single ? tq_npu::single_rung(opts.ladder) : opts.ladder;

    try {
        tq_npu::Session session;
        std::printf("turboquant_reshape_and_cache_a8 NPU validation, "
                    "one monolithic dispatch per rung, %s%s\n",
                    tq_npu::mode4_name(opts.mode4_reuse),
                    opts.single ? " (--single: last rung only)" : "");
        for (const int strips : ladder) {
            tq_npu::Fixture fixture(strips * tq_npu::kBlockSize);
            tq_npu::A8Device device(fixture);
            device.launch(session.stream, fixture, opts.mode4_reuse);
            tq_npu::check(aclrtSynchronizeStream(session.stream),
                          "aclrtSynchronizeStream");
            std::printf("\n%d strips in 1 launch of %d grid blocks\n", strips, strips);
            tq_npu::verify(fixture, device);
        }
        std::printf("\nPASS turboquant_reshape_and_cache_a8: %zu ladder rung%s, "
                    "single dispatch each, %s, bit-exact against the shared "
                    "integer oracle\n",
                    ladder.size(), ladder.size() == 1 ? "" : "s",
                    tq_npu::mode4_name(opts.mode4_reuse));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant A8 NPU validation: %s\n", e.what());
        return 1;
    }
}
