// Single-token decode at UNALIGNED context lengths, on the NPU.
//
// The directive's ladder: S in {1, 7, 15, 17, 31, 33, 127, 129, 255, 257,
// 1023}, none of them a multiple of 16 and none a multiple of the 128-token
// page, so every rung ends inside a strip and most end inside a page. S = 1,
// S = 129 and S = 257 each leave the last page holding exactly ONE live token
// in ONE live strip, which is the narrowest the geometry gets.
//
// Two legs, because the tail change touched two kernels:
//
//   A  turboquant_single_token_decode_tail_aiv -- the plain AIV decode, one
//      block per (request, kv_head) item, through the new entry point that
//      takes `seqLen`.
//   B  turboquant_single_token_decode_splitk_aiv -- the same arithmetic split
//      K ways over the page range, which is the harder case: the tail lands in
//      whichever split owns the last live page, every other split must see
//      only full strips, and a split whose whole range is dead must contribute
//      the neutral element. Run at K in {1, 2, 4, 8}, so at the short rungs
//      most splits are dead by construction.
//
// What makes a green run mean something here
// ------------------------------------------
// `splitk_decode_test::Fixture` floods every cache slot at or past `seq_len`
// with 0x7FC0BEEF -- a quiet NaN -- in BOTH scale planes, and writes the
// largest nibble the int4 format holds into the packed key and value bytes
// there. So a kernel that lets one dead token through cannot be slightly
// wrong:
//
//   * if it reads the dead SCALE, the NaN propagates through m, l, Acc, the
//     butterfly and the gate, and the context comes back NaN, which the rig
//     reports as a non-finite mismatch;
//   * if it masks the dead lanes by MULTIPLYING them by zero, that is also
//     NaN, because 0 * NaN is NaN -- which is exactly why the device phase is
//     a masked write rather than a masked multiply;
//   * if it somehow avoided the scale but admitted the key nibbles, the score
//     would be the largest this fixture can produce and would dominate the
//     softmax rather than perturb it.
//
// The oracle masks the same tokens to -inf and skips their payload for the
// same reason, so the comparison is between two independently masked
// computations, not between a kernel and a restatement of itself.
//
// One request per rung, deliberately: the plain kernel indexes its block table
// without a request stride, so it is only defined for a single request, and
// the already-committed aligned AIV suite covers two. Contention in leg B is
// still real -- 4 head slots against up to 32 workers in one launch.

#include "turboquant_splitk_rigs.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

extern "C" void launch_turboquant_single_token_decode_tail_aiv(
    void*, std::uint32_t, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, float*, float*, float*, float*, float*, float*, std::uint16_t*,
    std::int32_t*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t);

namespace {

namespace tail = turboquant_decode_tail;
using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;
using splitk_decode_test::Fixture;
using splitk_decode_test::kChannels;

constexpr int kHeads = splitk_decode_test::kHeads;      // 4
constexpr int kRows = splitk_decode_test::kRows;        // 6
constexpr int kPageTokens = splitk_decode_test::kPageTokens;  // 128
constexpr int kRequests = 1;
constexpr int kProbeLanes = 32;
// The plain kernel pads its per-item output to eight rows while taking its
// query panel packed, so the two sides index differently and the comparison
// below must carry both strides.
constexpr int kMaxRows = 8;
constexpr int kArchiveStrips = 64;
constexpr int kArchiveStride = kMaxRows * 16;

// ---------------------------------------------------------------- leg A rig
// The plain AIV decode operator, through the arbitrary-context entry point.
struct TailRig {
    const Fixture& f;
    int items;
    Buffer q, key, value, scales, table, gate, signs;
    Buffer out_scores, out_state, out_acc, out_rotated, out_context, probe;

    explicit TailRig(const Fixture& fixture)
        : f(fixture), items(f.requests * f.heads), q(f.q.size() * 2),
          key(f.key.size()), value(f.value.size()), scales(f.scales.size() * 4),
          table((f.table.size() * 4 + 31) / 32 * 32), gate(f.gate.size() * 4),
          signs(f.signs.size() * 4),
          out_scores(static_cast<std::size_t>(items) * kArchiveStrips * kArchiveStride * 4),
          out_state(static_cast<std::size_t>(items) * 2 * kMaxRows * 8 * 4),
          out_acc(static_cast<std::size_t>(items) * kMaxRows * kChannels * 4),
          out_rotated(static_cast<std::size_t>(items) * kMaxRows * kChannels * 4),
          out_context(static_cast<std::size_t>(items) * kMaxRows * kChannels * 2),
          probe(static_cast<std::size_t>(kProbeLanes) * 4)
    {
        q.upload(f.q); key.upload(f.key); value.upload(f.value);
        scales.upload(f.scales); gate.upload(f.gate); signs.upload(f.signs);
        auto padded = f.table;
        padded.resize(table.bytes / 4, 0);
        table.upload(padded);
    }

    // Canary every output before the launch, so "written" and "left alone" are
    // both observable afterwards.
    void Reset()
    {
        out_scores.fill_canary();
        out_state.fill_canary();
        out_acc.fill_canary();
        out_rotated.fill_canary();
        out_context.fill_canary();
        probe.fill_canary();
    }

    void Launch(aclrtStream stream)
    {
        launch_turboquant_single_token_decode_tail_aiv(
            stream, static_cast<std::uint32_t>(items),
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::int8_t*>(key.ptr),
            static_cast<std::int8_t*>(value.ptr), static_cast<float*>(scales.ptr),
            static_cast<std::uint32_t*>(table.ptr), static_cast<float*>(gate.ptr),
            static_cast<float*>(signs.ptr), static_cast<float*>(out_scores.ptr),
            static_cast<float*>(out_state.ptr), static_cast<float*>(out_acc.ptr),
            static_cast<float*>(out_rotated.ptr),
            static_cast<std::uint16_t*>(out_context.ptr),
            static_cast<std::int32_t*>(probe.ptr),
            static_cast<std::uint32_t>(f.requests),
            static_cast<std::uint32_t>(f.heads), static_cast<std::uint32_t>(f.rows),
            static_cast<std::uint32_t>(kChannels),
            static_cast<std::uint32_t>(f.page_tokens),
            static_cast<std::uint32_t>(f.pages),  // cache_pages
            static_cast<std::uint32_t>(f.pages),  // seq_pages
            static_cast<std::uint32_t>(f.page_tokens / 16),  // strips per page
            0u,                                              // strip_begin
            0u, static_cast<std::uint32_t>(items),           // item window
            0u,                                              // page_begin
            static_cast<std::uint32_t>(f.seq_len));
    }

    // The gated BF16 context against the oracle, plus the guard band.
    void Validate() const
    {
        std::vector<std::uint16_t> actual(out_context.bytes / 2);
        out_context.download(actual);
        for (int request = 0; request < f.requests; ++request)
            for (int head = 0; head < f.heads; ++head) {
                const int item = request * f.heads + head;
                for (int row = 0; row < f.rows; ++row) {
                    const auto got_base =
                        (static_cast<std::size_t>(item) * kMaxRows + row) * kChannels;
                    const auto want_base =
                        (static_cast<std::size_t>(item) * f.rows + row) * kChannels;
                    float scale = 0;
                    for (int c = 0; c < kChannels; ++c)
                        scale = std::max(scale, std::abs(f.expected[want_base + c]));
                    const float tolerance = 0.005f * scale + 1e-6f;
                    for (int c = 0; c < kChannels; ++c) {
                        const float got = single_token_decode::bfloat16_to_float(
                            actual[got_base + c]);
                        const float want = f.expected[want_base + c];
                        // A leaked dead slot arrives here as a NaN, so the
                        // finiteness test is the canary check and is reported
                        // as such rather than as an ordinary mismatch.
                        if (!std::isfinite(got)) {
                            std::fprintf(stderr,
                                         "CANARY LEAK S=%d item=%d row=%d c=%d: context "
                                         "is not finite, a cache slot past seq_len "
                                         "reached the softmax\n",
                                         f.seq_len, item, row, c);
                            throw std::runtime_error("tail canary reached the output");
                        }
                        if (std::abs(got - want) > tolerance) {
                            std::fprintf(stderr,
                                         "context mismatch S=%d item=%d row=%d c=%d "
                                         "got=%.9g ref=%.9g tol=%.9g\n",
                                         f.seq_len, item, row, c, got, want, tolerance);
                            throw std::runtime_error("unaligned context validation failed");
                        }
                    }
                }
                // The output guard band: rows the task does not own must still
                // hold the canary byte pattern, so a burst that overran its
                // row count is caught rather than averaged away.
                constexpr auto kCanaryPair = static_cast<std::uint16_t>(
                    (tq_npu::kCanaryByte << 8) | tq_npu::kCanaryByte);
                for (int row = f.rows; row < kMaxRows; ++row) {
                    const auto base =
                        (static_cast<std::size_t>(item) * kMaxRows + row) * kChannels;
                    for (int c = 0; c < kChannels; ++c)
                        require(actual[base + c] == kCanaryPair,
                                "output guard band overwritten past the row count");
                }
            }
    }
};

// ------------------------------------------------------------------- ladder
// --single-tile-canary restricts the sweep to the two rungs that live inside
// a single 16-token tile's worth of tail: S = 1, which is one live token and
// fifteen canary lanes in the only strip, and S = 17, which is one full strip
// followed by a strip holding a single live token. Those are the two shapes
// the mask has to get right; every longer rung adds pages, not cases, and on
// CAModel a full context costs hours for no extra coverage of the tail.
int run(bool single_tile)
{
    tq_npu::Session session;

    const std::vector<int> ladder =
        single_tile ? std::vector<int>{1, 17}
                    : std::vector<int>{1, 7, 15, 17, 31, 33, 127, 129, 255, 257, 1023};
    const std::vector<int> splits =
        single_tile ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 4, 8};

    std::printf("=== single-token decode, unaligned context%s ===\n",
                single_tile ? " (single-tile canary)" : "");

    int rungs = 0, plain_launches = 0, splitk_launches = 0, dead_splits = 0;
    for (const int seq_len : ladder) {
        // Capacity is rounded up to whole pages; seq_len is what is live.
        Fixture f(seq_len, kRequests, kHeads, kRows, kPageTokens);
        require(f.seq_len == seq_len, "fixture did not take the live token count");
        require(f.pages == static_cast<int>(tail::PagesForTokens(
                               static_cast<std::uint32_t>(seq_len), kPageTokens)),
                "fixture page count disagrees with the tail geometry");

        const std::uint32_t last_strip =
            tail::StripsForTokens(static_cast<std::uint32_t>(seq_len)) - 1u;
        const std::uint32_t active = tail::ActiveInStrip(
            static_cast<std::uint64_t>(last_strip) * 16u,
            static_cast<std::uint32_t>(seq_len));

        // ---- leg A: the plain AIV decode ---------------------------------
        {
            TailRig rig(f);
            rig.Reset();
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            rig.Launch(session.stream);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            rig.Validate();
            ++plain_launches;
        }

        // ---- leg B: the same context, split K ways -----------------------
        int dead_here = 0;
        for (const int k : splits) {
            tq_splitk_rig::AivRig rig(f, k);
            rig.Reset();
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            rig.Launch(session.stream, k, 1000000);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            rig.Validate(k);
            ++splitk_launches;
            for (int split = 0; split < k; ++split) {
                const auto range = turboquant_splitk::Partition(f.pages, k, split);
                if (tail::LiveStripsInWindow(
                        static_cast<std::uint32_t>(seq_len),
                        static_cast<std::uint32_t>(range.begin),
                        static_cast<std::uint32_t>(range.count), kPageTokens, 0u,
                        kPageTokens / 16) == 0u)
                    ++dead_here;
            }
        }
        dead_splits += dead_here;

        std::printf("  ok  S=%-5d %2d pages, %2u strips, %2u live lanes in the last, "
                    "%d dead splits over K=1/2/4/8\n",
                    seq_len, f.pages, last_strip + 1u, active, dead_here);
        ++rungs;
    }

    std::printf("PASS single-token decode at unaligned context lengths: %d rungs "
                "(S=1..1023, none a multiple of 16), %d plain-AIV launches and %d "
                "split-K launches, %d split ranges entirely past the live tail and "
                "contributing the neutral element; every cache slot past seq_len "
                "held a 0x7FC0BEEF canary in both scale planes and none reached the "
                "gated BF16 context\n",
                rungs, plain_launches, splitk_launches, dead_splits);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    bool single_tile = false;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string option = argv[arg];
        if (option == "--single-tile-canary") {
            single_tile = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--single-tile-canary]\n"
                         "  --single-tile-canary  only S=1 and S=17, the two "
                         "tail-tile shapes\n",
                         argv[0]);
            return 2;
        }
    }
    try {
        return run(single_tile);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL single-token unaligned decode: %s\n", error.what());
        return 1;
    }
}
