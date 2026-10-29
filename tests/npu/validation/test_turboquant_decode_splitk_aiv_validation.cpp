// Validation ladder for turboquant_single_token_decode_splitk_aiv: AIV-only,
// M <= 8 query rows, d = 256 channels (directive 3B).
//
// One launch per rung. Every rung checks, through
// tq_splitk_rig::AivRig::Validate (shared with the benchmark, see
// tests/npu/turboquant_splitk_rigs.hpp):
//
//   * the L2 AtomicHeadSlot protocol -- every worker acquired the lock, none
//     hit the watchdog, the arrival tickets over the K workers form a
//     permutation of 1..K, exactly one worker is the tail, and each head slot
//     ends drained (lock released, arrived_count == K);
//   * the per-worker metric payload is not torn across its eight replicated
//     lanes, and each worker's page count equals Partition's for its split;
//   * the final context: normalized, butterfly-rotated and gated, in BF16,
//     against splitk_decode_test::Fixture::expected -- the independent FP64
//     full-context oracle, which applies DenseRotate rather than the kernel's
//     factorization, so the egress ordering is checked rather than assumed.
//
// This test's own part is the matrix. M runs over 1, 2, 4, 6 and 8: 1 is the
// single-token boundary, 8 is the kernel's maximum, and 6 is the default the
// rest of the harness uses. K runs over 2, 4 and 8, which is where the atomic
// contention lives -- K workers per (request, head) racing for one 128-byte
// aligned slot. K = 1 is deliberately absent: with one worker there is no
// contention and the tail is the first arrival, so it would not exercise the
// protocol this test exists for. The benchmark covers K = 1.
//
// Contention is made real rather than nominal by running several requests and
// all four heads at once, so requests * heads slots are contended concurrently
// by requests * heads * K workers in one launch.

#include "../turboquant_splitk_rigs.hpp"

#include <cstdio>
#include <exception>
#include <vector>

namespace {

using tq_npu::check;

// Real operator geometry: 128-token pages, four KV heads, 256 channels. Four
// pages so that every split of K = 8 still owns a page at M = 8, and so the
// uneven partition at K = 4 and 8 is exercised too.
constexpr int kRequests = 2;
constexpr int kHeads = splitk_decode_test::kHeads;
constexpr int kPageTokens = splitk_decode_test::kPageTokens;
constexpr int kPages = 4;
constexpr int kWatchdog = 1000000;

int run()
{
    tq_npu::Session session;

    const std::vector<int> rows = {1, 2, 4, 6, 8};
    const std::vector<int> splits = {2, 4, 8};

    int launches = 0, contended_slots = 0, workers = 0;
    for (const int m : rows) {
        // One fixture per M; K changes only the page partition, not the data.
        splitk_decode_test::Fixture f(kPages * kPageTokens, kRequests, kHeads, m,
                                      kPageTokens);
        for (const int k : splits) {
            tq_splitk_rig::AivRig rig(f, k);
            rig.Reset();
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            rig.Launch(session.stream, k, kWatchdog);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            const auto census = rig.Validate(k);

            // Validate() already requires acquired == 1 and timeout == 0 for
            // every worker, so these are the aggregate counts, reported rather
            // than re-asserted. A nonzero failed-exchange count is not a
            // failure -- it is the compare-and-swap retrying, which is what
            // contention looks like from inside -- but it is the number that
            // says the race actually happened.
            contended_slots += kRequests * kHeads;
            workers += kRequests * kHeads * k;
            ++launches;
            std::printf("  ok  M=%-2d K=%-2d  %2d slots, %3d workers, "
                        "%llu failed exchanges, %llu timeouts\n",
                        m, k, kRequests * kHeads, kRequests * kHeads * k,
                        static_cast<unsigned long long>(census.failed),
                        static_cast<unsigned long long>(census.timeouts));
        }
    }

    std::printf("PASS AIV split-K decode validation: %d launches over M=1/2/4/6/8 "
                "x K=2/4/8, %d contended head slots and %d workers in total; "
                "lock acquisition, arrival permutation, single tail and drained "
                "slots all checked, and the gated BF16 context matches the FP64 "
                "full-context oracle (d=%d, %d pages of %d tokens)\n",
                launches, contended_slots, workers, splitk_decode_test::kChannels,
                kPages, kPageTokens);
    return 0;
}

}  // namespace

int main()
{
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL AIV split-K decode validation: %s\n", error.what());
        return 1;
    }
}
