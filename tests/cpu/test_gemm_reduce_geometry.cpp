// Host gate for the Cube GEMV reduction's geometry and budget contract.
//
// §6.1's rule, applied to this stage: the arithmetic lives in a header with no CCE dependency,
// the kernel includes it, and this translation unit compiles the SAME declaration with
// -Wall -Wextra -Werror. It is worth something only because the kernel and this gate share the
// header rather than each carrying a copy. Every static_assert in the header fires at this
// compile; what is below is the part that needs values rather than constants, plus the
// reference combine the numerical comparison is scored against.

#include "../../include/turboquant_gemm_reduce.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace tq = ascend_micro::gemm_reduce;

namespace {

int failures = 0;

void expect(bool condition, const char *what)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// Where split s's row starts, and where its two statistics lanes sit inside it. The DN2NZ
// fetch and the statistics gather must agree with this or they read different matrices.
void check_workspace_addressing()
{
    const std::uint32_t heads = 4, splits = 8;
    for (std::uint32_t token = 0; token < 3; ++token) {
        for (std::uint32_t head = 0; head < heads; ++head) {
            const std::uint64_t base = tq::kPartialRowOffset(token, head, 0, heads, splits);
            for (std::uint32_t s = 1; s < splits; ++s) {
                const std::uint64_t row = tq::kPartialRowOffset(token, head, s, heads, splits);
                expect(row - base == std::uint64_t{s} * tq::kPartialStride,
                       "split rows are kPartialStride apart, which is what srcDValue must be");
            }
            // The next (token, head) must start past this one's last statistics lane.
            const std::uint64_t next = (head + 1 == heads)
                ? tq::kPartialRowOffset(token + 1, 0, 0, heads, splits)
                : tq::kPartialRowOffset(token, head + 1, 0, heads, splits);
            expect(next >= base + std::uint64_t{splits} * tq::kPartialStride,
                   "no (token, head) overlaps the next one's rows");
        }
    }
    expect(tq::kPartialSumLane - tq::kPartialMaxLane == 8,
           "max and sum are 8 floats apart, so one 64-byte burst carries both");
    expect(tq::kPartialTail >= 16, "the tail holds both statistics blocks");
}

// The Zn image DN2NZ builds, and the Load2DV2 parameters that read it, computed the way the
// kernel computes them. An off-by-one here is a wrong-matrix bug, not a slowdown.
void check_operand_layout()
{
    const std::uint32_t n = tq::kHeadSize;
    expect(tq::kPartialL1Bytes == n * tq::kC0Bytes, "the L1 cube is one C0 block per N row");
    // Non-transposed Load2DV2: mStep walks N in 16s, kStep walks K in 32-byte blocks.
    const std::uint32_t mStep = n / tq::kFractalRows;
    const std::uint32_t kStep = tq::kAlignedSplits(8) / tq::kC0Floats;
    expect(mStep == 16, "headSize 256 is 16 fractal rows of N");
    expect(kStep == 1, "every legal split count is a single C0 block, so one Mmad and no K tiling");
    expect(mStep * tq::kFractalBytes == tq::kPartialL1Bytes,
           "mStep fractals of 512 bytes is exactly the cube");
    // The transposed b32 path is the one that does NOT work here, and the gate says why rather
    // than leaving it to a comment: it would need kStep even AND a source Zn cube with
    // Col == 16, and K = S <= 8 cannot be 16.
    expect(kStep % 2 != 0, "a transposed b32 load would need an even kStep; this one is 1");
    expect(tq::kAlignedSplits(tq::kMaxSplits) < tq::kFractalRows,
           "K = S never reaches the 16 columns a transposed b32 merge demands");
    for (std::uint32_t s = 1; s <= tq::kMaxSplits; ++s) {
        expect(tq::kAlignedSplits(s) == tq::kC0Floats, "every split count is one C0 block");
        expect(tq::kValidReduceGeometry(tq::kHeadSize, s), "every split count is accepted");
    }
    expect(!tq::kValidReduceGeometry(tq::kHeadSize, tq::kMaxSplits + 1),
           "a split count past the cap is rejected rather than silently K-tiled");
}

void check_flag_budget()
{
    // The stage recycles the attention schedule's ids, so the only thing a budget object can
    // still prove is that it stays out of the library's reserved range and that no two places
    // collide. Both are checked as values here and as static_asserts in the header.
    std::vector<int> used(tq::kReduceFlagReservedFirst, 0);
    for (std::uint32_t slot = 0; slot < tq::kReduceSlots; ++slot) {
        for (std::uint16_t id : {static_cast<std::uint16_t>(tq::kReduceFlagWeightsReady + slot),
                                 static_cast<std::uint16_t>(tq::kReduceFlagProductReady + slot)}) {
            expect(id < tq::kReduceFlagReservedFirst, "every id is outside the reserved range");
            expect(used[id] == 0, "no two places share an id");
            used[id] = 1;
        }
    }
    std::uint32_t count = 0;
    for (int u : used) count += static_cast<std::uint32_t>(u);
    expect(count == tq::kReduceFlagCount, "the id count is the place count");
}

void check_task_slicing()
{
    // Production's Reduce() clamp, and the prologue depth that follows from it. The empty
    // worker is a required rung, not an edge case: §4 asks for one explicitly.
    const std::uint32_t total = 10, perBlock = 4;
    std::uint32_t seen = 0;
    for (std::uint32_t block = 0; block < 4; ++block) {
        const std::uint32_t tasks = tq::kReduceTaskCount(total, block, perBlock);
        expect(tq::kReducePrologue(tasks) <= tasks, "a prologue never primes more than it runs");
        expect(tq::kReducePrologue(tasks) <= tq::kReduceSlots, "a prologue never exceeds the ring");
        seen += tasks;
    }
    expect(seen == total, "the slicing covers every task exactly once");
    expect(tq::kReduceTaskCount(total, 3, perBlock) == 0, "the last block is empty and legal");
    expect(tq::kReducePrologue(0) == 0, "an empty worker primes nothing, so it retires nothing");
}

// The reference combine, in double, from the same workspace layout the kernel reads. This is
// the oracle the CAModel comparison scores both device legs against; it is NOT a substitute
// for either, and a host model is not CAModel evidence.
void check_reference_combine()
{
    const std::uint32_t splits = 8, d = 16;
    std::vector<float> m(splits), l(splits);
    std::vector<std::vector<float>> o(splits, std::vector<float>(d));
    for (std::uint32_t s = 0; s < splits; ++s) {
        m[s] = -2.0f + 0.5f * static_cast<float>(s);
        l[s] = 1.0f + 0.3f * static_cast<float>(s);
        for (std::uint32_t i = 0; i < d; ++i) {
            // Deliberately NOT a scalar multiple of l_s, and s-dependent in shape rather than
            // in magnitude only. A fixture where o_s is proportional to l_s cancels the
            // weights out of the quotient entirely and scores every weighting as correct.
            o[s][i] = 1.0f + 0.25f * static_cast<float>(i) +
                      static_cast<float>(s) * (2.0f - 0.1f * static_cast<float>(i));
        }
    }
    double star = -std::numeric_limits<double>::infinity();
    for (std::uint32_t s = 0; s < splits; ++s) star = star > m[s] ? star : m[s];
    double denom = 0.0;
    std::vector<double> num(d, 0.0);
    for (std::uint32_t s = 0; s < splits; ++s) {
        const double w = std::exp(static_cast<double>(m[s]) - star);
        denom += w * l[s];
        for (std::uint32_t i = 0; i < d; ++i) num[i] += w * o[s][i];
    }
    // The sequential online recurrence the vector reducer runs, over the same inputs. Same
    // value, different floating-point expression: this is the identity the GEMV form relies
    // on, so it is checked rather than asserted in prose.
    double runMax = -std::numeric_limits<double>::infinity(), runSum = 0.0;
    std::vector<double> acc(d, 0.0);
    for (std::uint32_t s = 0; s < splits; ++s) {
        const double newMax = runMax > m[s] ? runMax : m[s];
        const double alpha = std::exp(runMax - newMax), beta = std::exp(m[s] - newMax);
        runSum = runSum * alpha + l[s] * beta;
        for (std::uint32_t i = 0; i < d; ++i) acc[i] = acc[i] * alpha + o[s][i] * beta;
        runMax = newMax;
    }
    expect(std::fabs(runSum - denom) <= 1e-12 * denom, "the two denominators agree");
    for (std::uint32_t i = 0; i < d; ++i) {
        const double a = num[i] / denom, b = acc[i] / runSum;
        expect(std::fabs(a - b) <= 1e-12 * std::fabs(a), "the two numerators agree per channel");
    }
    // w_s is exp(m_s - m*) and NOT exp(m_s - m*) * l_s: these partials are unnormalised, so
    // folding l_s into the weight would weight every split by its own mass twice. The two
    // conventions are not equivalent and the fixture shows it, so the kernel's choice is
    // pinned by a test rather than by the comment next to it.
    double wrongDenom = 0.0;
    std::vector<double> wrongNum(d, 0.0);
    for (std::uint32_t s = 0; s < splits; ++s) {
        const double w = std::exp(static_cast<double>(m[s]) - star) * l[s];
        wrongDenom += w * l[s];
        for (std::uint32_t i = 0; i < d; ++i) wrongNum[i] += w * o[s][i];
    }
    bool differs = false;
    for (std::uint32_t i = 0; i < d; ++i) {
        const double a = num[i] / denom, b = wrongNum[i] / wrongDenom;
        differs = differs || std::fabs(a - b) > 1e-6 * std::fabs(a);
    }
    expect(differs, "folding l_s into the weight changes the answer on this fixture");
}

// ---------------------------------------------------------------------------------------------
// The dense batched form
// ---------------------------------------------------------------------------------------------

// The block-diagonal A cube, built the way the kernel builds it and read the way Mmad reads it.
// This is the check that makes the packing claim concrete: Out[q, c] must depend on K block q
// and on no other, because that containment is what makes a ragged batch's dead fractals safe.
void check_block_diagonal_containment()
{
    constexpr std::uint32_t rows = tq::kGemmBatchRows;
    constexpr std::uint32_t k = tq::kGemmKMax;
    // The cube as the kernel leaves it: zero everywhere, then each row's own 8 weight lanes
    // written at kWeightRowOffsetFloats(row).
    std::vector<float> cube(rows * tq::kSliceFractalFloats, 0.0f);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t s = 0; s < tq::kMaxSplits; ++s) {
            cube[tq::kWeightRowOffsetFloats(row) + s] =
                1.0f + static_cast<float>(row) + 0.125f * static_cast<float>(s);
        }
    }
    // A[m, 8 * kBlock + s] as the Cube addresses it: fractal kBlock at float kBlock * 128,
    // row m at + m * 8.
    auto A = [&](std::uint32_t m, std::uint32_t kIndex) {
        const std::uint32_t block = kIndex / tq::kC0Floats;
        const std::uint32_t lane = kIndex % tq::kC0Floats;
        return cube[block * tq::kSliceFractalFloats + m * tq::kC0Floats + lane];
    };
    bool offDiagonalZero = true, diagonalRight = true;
    for (std::uint32_t m = 0; m < rows; ++m) {
        for (std::uint32_t kIndex = 0; kIndex < k; ++kIndex) {
            const std::uint32_t block = kIndex / tq::kC0Floats;
            const std::uint32_t lane = kIndex % tq::kC0Floats;
            if (block == m) {
                diagonalRight = diagonalRight &&
                    A(m, kIndex) == 1.0f + static_cast<float>(m) + 0.125f * static_cast<float>(lane);
            } else if (A(m, kIndex) != 0.0f) {
                offDiagonalZero = false;
            }
        }
    }
    expect(diagonalRight, "row q's weights land in K block q, which is what Mmad will read");
    expect(offDiagonalZero,
           "every off-diagonal lane is zero, so Out[q, c] depends on K block q alone -- the "
           "containment a ragged batch's dead fractals rely on");

    // No two rows' weight blocks overlap, and none escapes the cube.
    bool disjoint = true;
    for (std::uint32_t a = 0; a < rows; ++a) {
        const std::uint32_t pa = tq::kWeightRowOffsetFloats(a);
        disjoint = disjoint && pa + tq::kC0Floats <= cube.size();
        for (std::uint32_t b = a + 1; b < rows; ++b) {
            const std::uint32_t pb = tq::kWeightRowOffsetFloats(b);
            disjoint = disjoint && (pa + tq::kC0Floats <= pb || pb + tq::kC0Floats <= pa);
        }
    }
    expect(disjoint, "the weight blocks are disjoint and inside the cube");
}

// The subcore split. Both halves must have work for every live count above one, the live rows
// must partition exactly, and each subcore's rows must be a contiguous run of tasks -- which is
// what lets one WriteHeads burst cover a half.
void check_subcore_split()
{
    for (std::uint32_t live = 1; live <= tq::kGemmBatchRows; ++live) {
        const std::uint32_t a = tq::kGemmLiveRowsOf(live, 0);
        const std::uint32_t b = tq::kGemmLiveRowsOf(live, 1);
        expect(a + b == live, "the two halves partition the batch's live rows exactly");
        expect(a <= tq::kGemmPanelRows && b <= tq::kGemmPanelRows,
               "neither half overflows its panel");
        expect(a >= 1, "subcore 0 always has at least one row");
        if (live >= 2) {
            expect(b >= 1, "both subcores have real work for every live count above one");
        } else {
            expect(b == 0, "live = 1 is the one idle-subcore geometry");
        }
        expect(tq::kGemmFirstTaskOf(live, 0) == 0 && tq::kGemmFirstTaskOf(live, 1) == a,
               "each half's first task continues where the other's rows end");
        // The static halves: subcore 1's rows start at the Fixpipe's split point whatever the
        // live count, which is what keeps each core's Init zero-fill to its own fractals.
        expect(tq::kGemmFirstRowOf(0) == 0 && tq::kGemmFirstRowOf(1) == tq::kGemmPanelRows,
               "the half boundary does not move with the live count");
    }
}

// Batch formation over a block's share, with and without fused gaps, compared against the
// kernel's own loop. A batch that spanned a task-id gap would make dnNum walk the wrong stride.
void check_batch_formation()
{
    const std::uint32_t heads = 2;
    for (std::uint32_t total = 1; total <= 40; ++total) {
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<std::uint32_t> first, count;
            std::uint32_t rows = 0, previous = 0;
            for (std::uint32_t task = 0; task < total; ++task) {
                // mode 1 drops every other token, exactly as --mixed does.
                if (mode == 1 && (task / heads) % 2 == 1) continue;
                if (first.empty() || tq::kGemmBatchBreaks(previous, task, rows)) {
                    first.push_back(task);
                    count.push_back(1);
                    rows = 1;
                } else {
                    ++count.back();
                    ++rows;
                }
                previous = task;
            }
            bool consecutive = true, bounded = true;
            for (std::size_t i = 0; i < first.size(); ++i) {
                bounded = bounded && count[i] >= 1 && count[i] <= tq::kGemmBatchRows;
                // Every row of a batch is the previous row's task plus one. This is the
                // property dnNum depends on, so it is checked and not assumed.
                if (i + 1 < first.size()) {
                    consecutive = consecutive && first[i] + count[i] <= first[i + 1];
                }
            }
            expect(bounded, "no batch exceeds kGemmBatchRows or is empty");
            expect(consecutive, "batches do not overlap and run in task order");
            std::uint32_t covered = 0;
            for (std::uint32_t c : count) covered += c;
            std::uint32_t live = 0;
            for (std::uint32_t task = 0; task < total; ++task) {
                if (!(mode == 1 && (task / heads) % 2 == 1)) ++live;
            }
            expect(covered == live, "every live task lands in exactly one batch");
            if (mode == 0) {
                expect(first.size() == tq::kGemmDenseBatches(total),
                       "a dense share needs ceil(tasks / kGemmBatchRows) batches");
            }
        }
    }
}

// The batched ingest's strides, and the one thing the statistics burst needs: that a half's
// (row, split) tails form a single arithmetic progression in GM.
void check_batched_ingest()
{
    const std::uint32_t heads = 4, splits = 8;
    // Consecutive tasks are exactly srcDnMatrixStride apart, which is what dnNum walks.
    for (std::uint32_t task = 0; task + 1 < 12; ++task) {
        const std::uint64_t here =
            tq::kPartialRowOffset(task / heads, task % heads, 0, heads, splits);
        const std::uint64_t next =
            tq::kPartialRowOffset((task + 1) / heads, (task + 1) % heads, 0, heads, splits);
        expect(next - here == tq::kSliceSrcMatrixStrideFloats(splits),
               "consecutive tasks are one srcDnMatrixStride apart");
    }
    // The destination: one fractal per packed row, contiguous, so Load2DV2 walks them in order.
    expect(tq::kSliceDn2NzDstMatrixStrideFloats == tq::kSliceFractalFloats,
           "dstNzMatrixStride is one 512-byte fractal, so the slice is fractal-contiguous");
    expect(tq::kSliceDn2NzDstNzC0Stride == tq::kGemmSliceCols &&
               tq::kSliceDn2NzNValue == tq::kGemmSliceCols,
           "a slice's destination is [16 channels][C0 splits], one fractal");
    expect(tq::kGemmBatchRows * tq::kSliceFractalFloats * 4 == tq::kSliceL1Bytes,
           "the slice slot holds exactly kGemmBatchRows fractals");
    // The statistics burst: tail(row, split) must be a single progression over the half.
    for (std::uint32_t row = 0; row + 1 < tq::kGemmPanelRows; ++row) {
        for (std::uint32_t s = 0; s < splits; ++s) {
            const std::uint64_t here =
                tq::kPartialRowOffset(row / heads, row % heads, s, heads, splits) + tq::kHeadSize;
            const std::uint32_t nextIndex = row * splits + s + 1;
            const std::uint32_t nr = nextIndex / splits, ns = nextIndex % splits;
            const std::uint64_t next =
                tq::kPartialRowOffset(nr / heads, nr % heads, ns, heads, splits) + tq::kHeadSize;
            expect(next - here == tq::kPartialStride,
                   "every (row, split) tail of a half is one kPartialStride from the next, so "
                   "ONE strided burst gathers the batch");
        }
    }
}

// The batched form's budgets, and the UB arithmetic the integration finding rests on.
void check_batched_budgets()
{
    expect(tq::kGemmL1Bytes == tq::kBatchSlots * tq::kWeightCubeBytes +
                                   tq::kSliceSlots * tq::kSliceL1Bytes,
           "the L1 claim is the two cubes plus the two slice slots");
    expect(tq::kAttentionL1Bytes + tq::kGemmL1Bytes <= 512u * 1024u, "L1 fits");
    expect(tq::kGemmL0aBytes <= 64u * 1024u && tq::kGemmL0bBytes <= 64u * 1024u, "L0A/L0B fit");
    expect(tq::kGemmL0cBytes <= 128u * 1024u, "L0C fits");
    expect(tq::kGemmPanelPayloadBytes == tq::kGemmPanelRows * tq::kHeadSize * 4,
           "a panel is its subcore's half of the batch, dense at headSize");
    expect(tq::kGemmPanelUbBytes > tq::kGemmPanelPayloadBytes,
           "the AKA3003 skew is tail past the payload, never between rows");
    // The honest integration number: what the batched panels add on top of the production
    // operator's existing claim. Reported rather than asserted away -- see the report.
    const std::uint32_t added = tq::kBatchSlots * tq::kGemmPanelUbBytes + tq::kGemmOutStageBytes;
    std::printf("  batched UB added over the M = 1 form: %u bytes "
                "(%u panel slots of %u + %u fp16 staging)\n",
                added, tq::kBatchSlots, tq::kGemmPanelUbBytes, tq::kGemmOutStageBytes);
    expect(added > 0, "the batched form does cost UB, and the number is stated");
}

}  // namespace

int main()
{
    check_workspace_addressing();
    check_operand_layout();
    check_flag_budget();
    check_task_slicing();
    check_reference_combine();
    check_block_diagonal_containment();
    check_subcore_split();
    check_batch_formation();
    check_batched_ingest();
    check_batched_budgets();
    if (failures) {
        std::fprintf(stderr, "%d geometry check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("PASS gemm reduce geometry: workspace addressing, Zn operand, id budget, "
                "task slicing, reference combine, block-diagonal containment, subcore split, "
                "batch formation, batched ingest strides, batched budgets\n");
    return EXIT_SUCCESS;
}
