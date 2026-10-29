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
// What the ladder covers, and why it is one dispatch per rung
// -----------------------------------------------------------
// Every rung is a SINGLE monolithic kernel launch of that many grid blocks.
// There is no --window, no cyclic slice and no staggered sub-launch: those
// existed to keep each physical wave a fresh kernel entry, because the kernel
// could not survive its own wave boundary. It can now -- boundary guard,
// bounded recycled channels, mandatory epilogue drain
// (turboquant_wave_protocol.hpp, guidelines section 6.5) -- and a capped
// launch would hide exactly the behaviour this ladder exists to test. The
// rungs and what each one is for are in tq_npu::default_ladder().
//
// Pages are fragmented (numBlocks = 2 * strips + 2, strip s -> page 2s + 1),
// so the page index is never the grid block index and a stray write lands on a
// canary page rather than on a neighbour's correct page. The verifier reports
// per page and, when a page holds the wrong data, names the strip the data
// actually came from -- which is the signature of a block-index error and is
// invisible in an aggregate byte count.
//
// Usage:
//   test_turboquant_bf16_validation [--strips N]... [--tokens N]... [--single]

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
        std::printf("turboquant_reshape_and_cache_bf16 NPU validation, "
                    "one monolithic dispatch per rung, %s%s\n",
                    tq_npu::mode4_name(opts.mode4_reuse),
                    opts.single ? " (--single: last rung only)" : "");
        for (const int strips : ladder) {
            tq_npu::Fixture fixture(strips * tq_npu::kBlockSize);
            tq_npu::Bf16Device device(fixture);
            device.launch(session.stream, fixture, opts.mode4_reuse);
            tq_npu::check(aclrtSynchronizeStream(session.stream),
                          "aclrtSynchronizeStream");
            // The wave arithmetic is reported, not assumed: the physical block
            // count is a property of the part, so the harness states the grid
            // it dispatched and lets the reader map it onto waves.
            std::printf("\n%d strips in 1 launch of %d grid blocks\n", strips, strips);
            tq_npu::verify(fixture, device);
        }
        std::printf("\nPASS turboquant_reshape_and_cache_bf16: %zu ladder rung%s, "
                    "single dispatch each, %s, bit-exact against the shared "
                    "integer oracle\n",
                    ladder.size(), ladder.size() == 1 ? "" : "s",
                    tq_npu::mode4_name(opts.mode4_reuse));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL turboquant BF16 NPU validation: %s\n", e.what());
        return 1;
    }
}
