#pragma once
// The two split-K decode device rigs: buffer layout, launch, and the full
// correctness census for each kernel, shared by the benchmark and by the
// NPU/simulator validation tests.
//
// These were lifted out of bench_turboquant_decode_splitk.cpp, which is still
// one of the drivers. They live here so the benchmark and the validation tests
// run the *same* checks rather than two copies that can drift: a tolerance, a
// neutral element or a torn-payload rule that differs between the two would
// make a bench pass and a validation failure (or the reverse) say nothing.
//
// What each rig establishes is written at the rig. What neither can establish
// is that it ran: both need a device or the simulator, so a host build of a
// driver only proves the harness compiles. See reports/decode_splitk_validation.md.

#include "turboquant_npu_common.hpp"
#include "../common/aic_splitk_decode_fixture.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tq_splitk_rig {

// One ULP of `value`, and the distance from `got` to `ref` measured in them.
// Same convention as tests/camodel/test_camodel_fused_attention.cpp, so a
// budget quoted in one place means the same thing in the other: the spacing is
// taken at the REFERENCE, which is what makes a stray canary word in a zero
// channel read as an enormous ULP count rather than as a pass.
inline float UlpOf(float value)
{
    const float magnitude = std::abs(value);
    if (magnitude == 0.0f) return std::nextafter(0.0f, 1.0f);
    return std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
}

inline double UlpDistance(float got, float ref)
{
    if (got == ref) return 0.0;
    if (!std::isfinite(got) || !std::isfinite(ref))
        return std::numeric_limits<double>::infinity();
    return std::abs(static_cast<double>(got) - static_cast<double>(ref)) /
           static_cast<double>(UlpOf(ref));
}

// The float the 0xAA canary fill spells, so "never written" stays a separate
// diagnosis from "written wrong". -3.0316488e-13: finite, so std::isfinite
// does not catch it, and small, so neither does an additive tolerance floor.
inline float CanaryFloat()
{
    std::uint32_t raw = 0;
    for (int byte = 0; byte < 4; ++byte)
        raw |= static_cast<std::uint32_t>(tq_npu::kCanaryByte) << (8 * byte);
    float out = 0;
    std::memcpy(&out, &raw, sizeof out);
    return out;
}

using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;
namespace contract = turboquant_splitk;
using splitk_decode_test::Fixture;
using splitk_decode_test::kChannels;

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, float*, std::uint32_t*,
    float*, float*, float*, turboquant_splitk::WorkerMetrics*, std::uint16_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, std::uint16_t*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

struct AivRig {
    const Fixture& f;
    Buffer q, key, value, scales, table, gate, signs, workspace, metrics, out;
    AivRig(const Fixture& fixture, int splits)
        : f(fixture), q(f.q.size() * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          gate(f.gate.size() * 4), signs(f.signs.size() * 4),
          workspace(contract::WorkspaceFloats(f.requests, f.heads, splits) * 4),
          metrics(f.requests * f.heads * splits * sizeof(contract::WorkerMetrics)),
          out(f.q.size() * 2)
    {
        q.upload(f.q); key.upload(f.key); value.upload(f.value);
        scales.upload(f.scales); gate.upload(f.gate); signs.upload(f.signs);
        auto padded = f.table;
        padded.resize(table.bytes / 4, 0);
        table.upload(padded);
        require(reinterpret_cast<std::uintptr_t>(workspace.ptr) % 32 == 0,
                "the partition workspace must be 32-byte aligned");
    }
    // The workspace is NOT zeroed. Under the atomic protocol the shared head
    // slot had to start clean, because a worker merged into whatever it found
    // there; a partition is written before it is read and by one task only, so
    // a run that needs it pre-cleared is a run with a hole in it. Flooding it
    // with the canary instead is what turns such a hole into a NaN in the
    // context rather than a plausible number.
    void Reset()
    {
        workspace.fill_canary();
        check(aclrtMemset(metrics.ptr, metrics.bytes, 0, metrics.bytes), "reset metrics");
        out.fill_canary();
    }
    void Launch(aclrtStream stream, int splits)
    {
        launch_turboquant_single_token_decode_splitk_aiv(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::int8_t*>(key.ptr),
            static_cast<std::int8_t*>(value.ptr), static_cast<float*>(scales.ptr),
            static_cast<std::uint32_t*>(table.ptr), static_cast<float*>(gate.ptr),
            static_cast<float*>(signs.ptr), static_cast<float*>(workspace.ptr),
            static_cast<contract::WorkerMetrics*>(metrics.ptr),
            static_cast<std::uint16_t*>(out.ptr), f.requests, f.heads, f.rows,
            f.page_tokens, f.requests * f.pages, f.pages, splits,
            // The fixture's live token count, not its page capacity. On an
            // aligned fixture the two are equal and this is the pre-tail
            // behaviour; on an unaligned one it is what tells the kernel where
            // the sequence actually ends.
            static_cast<std::uint32_t>(f.seq_len));
    }
    // The emitted context exactly as the device left it, for callers that
    // compare two launches against each other rather than against an oracle.
    std::vector<std::uint16_t> Context() const
    {
        std::vector<std::uint16_t> raw(out.bytes / 2);
        out.download(raw);
        return raw;
    }
    // What the census establishes now that there is no lock to audit.
    //
    // Under the atomic protocol the questions were "did every worker acquire"
    // and "were the arrival tickets a permutation". Neither exists here. The
    // protocol is instead: every task published a partition of its own, every
    // partition is where its index says it is, exactly one worker per head
    // emitted the context, and nothing a reducer read was still canary. The
    // last two are what a missing or misaddressed partition would break, and
    // they are checked against the workspace itself rather than against a
    // counter the kernel maintains.
    //
    // `export_ticks` is the direct successor of the atomic protocol's
    // `lock_ticks`: the region from the state pack to the retired MTE3 burst.
    // It is the figure the 43,000-cycle lock region should be compared
    // against, and it is why this field was renamed rather than dropped.
    struct Census {
        std::vector<double> tail_ticks, export_ticks;
    };
    Census Validate(int splits)
    {
        std::vector<contract::WorkerMetrics> m(metrics.bytes / sizeof(contract::WorkerMetrics));
        metrics.download(m);
        std::vector<float> w(workspace.bytes / 4);
        workspace.download(w);
        const int heads = f.requests * f.heads;
        Census result;
        // The emitting worker is no longer "the last arrival at this head":
        // it is block h of the grid, which reduces head h after the barrier
        // whatever split it happened to decode. So the tail is a property of
        // the GRID, not of a head's K workers, and it is checked as one.
        std::vector<std::uint64_t> finish(static_cast<std::size_t>(heads) * splits);
        std::uint64_t earliest = std::numeric_limits<std::uint64_t>::max();
        int tails = 0;
        for (int head = 0; head < heads; ++head) {
            for (int split = 0; split < splits; ++split) {
                const int item = head * splits + split;
                const auto& metric = m[item];
                for (int lane = 1; lane < 8; ++lane)
                    require(metric.exported[lane] == metric.exported[0] &&
                            metric.partition[lane] == metric.partition[0] &&
                            metric.reduced[lane] == metric.reduced[0] &&
                            metric.pages[lane] == metric.pages[0] &&
                            metric.tail[lane] == metric.tail[0] &&
                            metric.finish_hi[lane] == metric.finish_hi[0] &&
                            metric.finish_lo[lane] == metric.finish_lo[0] &&
                            metric.export_hi[lane] == metric.export_hi[0] &&
                            metric.export_lo[lane] == metric.export_lo[0],
                            "torn metric payload");
                require(metric.exported[0] == 1, "worker published nothing");
                const auto range = contract::Partition(f.pages, splits, split);
                require(metric.pages[0] == range.count, "wrong per-split page count");

                if (splits == 1) {
                    require(metric.partition[0] == contract::kNoPartition,
                            "a single-split worker must bypass the workspace");
                    require(metric.reduced[0] == 0,
                            "a single-split launch must not enter the reduction");
                } else {
                    require(metric.partition[0] == static_cast<unsigned>(item),
                            "worker wrote someone else's partition");
                    // The partition this task owns must have been written:
                    // every live row of it, in every lane of the record.
                    const std::size_t base =
                        static_cast<std::size_t>(item) * contract::kPartitionFloats;
                    for (int row = 0; row < f.rows; ++row) {
                        const std::size_t at = base + static_cast<std::size_t>(row) *
                                                          contract::kRecordFloats;
                        for (unsigned lane = 0; lane < contract::kRecordFloats; ++lane)
                            require(std::isfinite(w[at + lane]),
                                    "partition lane left unwritten or nonfinite");
                        for (unsigned lane = 1; lane < contract::kStateLanes; ++lane)
                            require(w[at + lane] == w[at] &&
                                    w[at + contract::kStateLanes + lane] ==
                                        w[at + contract::kStateLanes],
                                    "exported m/l state is not replicated");
                    }
                    // A reducer is a block of the first `heads` of the grid,
                    // and it is the one that emitted this head's context.
                    require(metric.reduced[0] == static_cast<unsigned>(item < heads),
                            "the wrong blocks ran the reduction");
                }
                require(metric.tail[0] ==
                            static_cast<unsigned>(splits == 1 || item < heads),
                        "the wrong worker emitted the context");

                const std::uint64_t stamp =
                    (std::uint64_t{metric.finish_hi[0]} << 32) | metric.finish_lo[0];
                require(stamp != 0, "missing completion timestamp");
                const std::uint64_t export_region =
                    (std::uint64_t{metric.export_hi[0]} << 32) | metric.export_lo[0];
                require(export_region > 0, "missing export interval");
                result.export_ticks.push_back(static_cast<double>(export_region));
                finish[static_cast<std::size_t>(item)] = stamp;
                earliest = std::min(earliest, stamp);
                tails += static_cast<int>(metric.tail[0] != 0);
            }
        }
        require(tails == heads, "every head must be emitted exactly once");
        // The tail delta, in the same shape the atomic census reported it:
        // from the first worker of the launch retiring to the worker that
        // emitted this head finishing. Under Split-K that emitter is block h.
        for (int head = 0; head < heads; ++head) {
            const std::size_t emitter =
                splits == 1 ? static_cast<std::size_t>(head) * splits
                            : static_cast<std::size_t>(head);
            require(finish[emitter] >= earliest, "noncomparable completion timestamps");
            result.tail_ticks.push_back(static_cast<double>(finish[emitter] - earliest));
        }
        std::vector<std::uint16_t> actual(f.q.size());
        out.download(actual);
        for (std::size_t row = 0; row < actual.size() / kChannels; ++row) {
            float scale = 0;
            for (int c = 0; c < kChannels; ++c)
                scale = std::max(scale, std::abs(f.expected[row * kChannels + c]));
            // BF16 rounding consumes at most 1/256 of a row's maximum.
            const float tolerance = 0.005f * scale + 1e-6f;
            for (int c = 0; c < kChannels; ++c) {
                const auto at = row * kChannels + c;
                const float got = single_token_decode::bfloat16_to_float(actual[at]);
                if (!std::isfinite(got) || std::abs(got - f.expected[at]) > tolerance) {
                    std::fprintf(stderr, "context mismatch row=%zu c=%d got=%.9g ref=%.9g tol=%.9g\n",
                                 row, c, got, f.expected[at], tolerance);
                    throw std::runtime_error("CPU reference validation failed");
                }
            }
        }
        return result;
    }
};

struct AicRig {
    const Fixture& f;
    int splits, begin, pages;
    Buffer q, pi, key, value, scales, table, qrot, workspace;
    std::vector<splitk_decode_test::AicPartial> reference;
    AicRig(const Fixture& fixture, int k, int first, int count)
        : f(fixture), splits(k), begin(first), pages(count), q(f.q.size() * 2),
          pi(paged_decode::kTableElems * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          qrot(f.requests * f.heads * k * splitk_decode_test::kAicAccFloats * 2),
          workspace(contract::WorkspaceFloats(f.requests, f.heads, k) * 4)
    {
        q.upload(splitk_decode_test::AicRawQuery(f));
        pi.upload(paged_decode::BuildRotationTable().row_major);
        key.upload(f.key); value.upload(f.value); scales.upload(f.scales);
        auto padded = f.table; padded.resize(table.bytes / 4, 0); table.upload(padded);
        for (int request = 0; request < f.requests; ++request)
            for (int head = 0; head < f.heads; ++head)
                for (int split = 0; split < k; ++split) {
                    const auto r = contract::Partition(pages, k, split);
                    reference.push_back(splitk_decode_test::ReferenceAicPartial(
                        f, request, head, begin + r.begin, r.count));
                }
    }
    void Reset() { qrot.fill_canary(); workspace.fill_canary(); }
    void Launch(aclrtStream stream, int)
    {
        launch_turboquant_paged_decode_splitk_aic(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::uint16_t*>(pi.ptr),
            static_cast<std::int8_t*>(key.ptr), static_cast<std::int8_t*>(value.ptr),
            static_cast<float*>(scales.ptr), static_cast<std::uint32_t*>(table.ptr),
            static_cast<std::uint16_t*>(qrot.ptr), static_cast<float*>(workspace.ptr),
            f.requests, f.heads, f.page_tokens,
            f.requests * f.pages, f.pages, splits, begin, pages,
            // The fixture's live token count. Equal to its capacity on an
            // aligned fixture, so this is the pre-tail behaviour there.
            static_cast<std::uint32_t>(f.seq_len));
    }
    // The Cube path now exports into the SAME partition layout the Vector
    // path does -- [m(8) | l(8) | acc(256)] per row, one partition per task --
    // so this reads the record rather than two separate tensors. That the
    // interleave arrives correctly is exactly what makes the two paths
    // interchangeable to a reducer, so it is checked lane by lane.
    // The worst deviation each checked quantity reached, as a FRACTION of its
    // own tolerance. Reported on success as well as failure: a check that
    // passes at 0.001x of its bound is not measuring anything, and a check
    // that passes at 0.98x is one fixture away from a false alarm. Without
    // these numbers the page-depth ladder cannot say whether a deeper rung
    // got worse or merely crossed a line that was always nearly crossed.
    struct Margins {
        double m = 0, l = 0, acc = 0;
        std::size_t acc_item = 0;
        int acc_row = 0, acc_channel = 0;
        double acc_got = 0, acc_ref = 0;
        // Where a failure actually lives. Throwing at the first channel past
        // its bound answers none of the questions that matter next: whether
        // the query rotation is intact, whether m and l hold for the fifteen
        // rows after the first, whether one channel is wrong or all of them,
        // and whether the deviations share a sign. One channel of one row is
        // a different defect from a whole accumulator, so the survey runs to
        // completion and the diagnosis is printed before anything is raised.
        int acc_over = 0, rows_touched = 0, qrot_wrong = 0, state_torn = 0;
        int m_over = 0, l_over = 0;
        double acc_signed_sum = 0, acc_abs_sum = 0;
        int acc_checked = 0;
        // The same deviations in ULP, which is the reading a relative bound
        // cannot give near zero. `state_ulp` is the worst over m and l
        // together, because they share one budget.
        double state_ulp = 0, acc_ulp = 0;
        int state_ulp_row = 0, acc_ulp_row = 0, acc_ulp_channel = 0;
        double acc_ulp_got = 0, acc_ulp_ref = 0;
        const char* state_ulp_which = "-";
        // `acc_ulp` above is the deviation in ULP OF THE CHANNEL's own value,
        // which for an accumulator is a magnitude artifact rather than an
        // error measure: a row of 256 channels is a sum with cancellation, so
        // whichever channel lands nearest zero has the finest ULP and so the
        // largest ULP distance, for the same absolute deviation. Measured on
        // CAModel (reports/decode_splitk_aic_npu_multipage.md section 5) the
        // absolute deviations stayed flat at 4.5e-08 .. 1.8e-07 across 4, 8
        // and 32 strips while this number went 48 -> 98 -> 247, purely because
        // the unluckiest of 4096 channels got smaller. It is unbounded for a
        // bit-correct kernel, so it is REPORTED and not capped.
        //
        // `acc_scaled_ulp` takes the spacing at the row's accumulation scale
        // (max |ref| over the row) instead, which is the magnitude the
        // arithmetic actually worked at. That one is bounded by the summation
        // error, and is what a budget can be held against.
        double acc_scaled_ulp = 0;
        int acc_scaled_ulp_row = 0, acc_scaled_ulp_channel = 0;
        double acc_scaled_ulp_got = 0, acc_scaled_ulp_ref = 0;
        double acc_scaled_ulp_scale = 0;
        // The smallest |ref| any checked channel had, which is what makes the
        // raw ULP number as large as it is.
        double acc_ref_min = 0;
        // Words still holding the 0xAA fill in a partition a worker was
        // supposed to write, and nonfinite words anywhere in one.
        int canary_words = 0, nonfinite_words = 0;
    };
    Margins margins;

    // Opt-in ULP budgets. Zero means "report, do not enforce", which is what
    // the page-count ladder and the benchmark want -- their margin fractions
    // are quoted against the relative bounds, and holding them to a ULP cap
    // they were never written for would be a change of gate, not an addition.
    // A rung that sets these is held to them.
    // `acc_ulp_cap` caps the raw per-channel number; see Margins for why no
    // rung sets it. `acc_scaled_ulp_cap` caps the scale-referenced one.
    double state_ulp_cap = 0, acc_ulp_cap = 0, acc_scaled_ulp_cap = 0;

    AivRig::Census Validate(int)
    {
        std::vector<float> w(workspace.bytes / 4);
        std::vector<std::uint16_t> qr(qrot.bytes / 2);
        workspace.download(w); qrot.download(qr);
        margins = Margins{};
        for (std::size_t item = 0; item < reference.size(); ++item) {
            const auto& ref = reference[item];
            const auto range = contract::Partition(pages, splits, item % splits);
            for (int row = 0; row < 16; ++row) {
                const auto record = item * contract::kPartitionFloats +
                                    static_cast<std::size_t>(row) * contract::kRecordFloats;
                const auto acc_at = record + 2u * contract::kStateLanes;
                const auto acc_base = item * splitk_decode_test::kAicAccFloats + row * kChannels;
                const float m = w[record], l = w[record + contract::kStateLanes];
                require(std::isfinite(m) && std::isfinite(l), "nonfinite AIC partial state");
                // The whole record, not just m and l: a worker that wrote its
                // state and skipped part of its accumulator leaves the fill
                // behind exactly there.
                for (unsigned lane = 0; lane < contract::kRecordFloats; ++lane) {
                    const float word = w[record + lane];
                    margins.canary_words += word == CanaryFloat();
                    margins.nonfinite_words += !std::isfinite(word);
                }
                const double m_ulp = UlpDistance(m, static_cast<float>(ref.m[row]));
                if (m_ulp > margins.state_ulp) {
                    margins.state_ulp = m_ulp;
                    margins.state_ulp_row = row;
                    margins.state_ulp_which = "m";
                }
                const double l_ulp = UlpDistance(l, static_cast<float>(ref.l[row]));
                if (l_ulp > margins.state_ulp) {
                    margins.state_ulp = l_ulp;
                    margins.state_ulp_row = row;
                    margins.state_ulp_which = "l";
                }
                const double m_tol = 2e-6 * std::max(1.0, std::abs(ref.m[row]));
                const double l_tol = 1e-4 * std::max(1.0, ref.l[row]);
                margins.m = std::max(margins.m, std::abs(m - ref.m[row]) / m_tol);
                margins.l = std::max(margins.l, std::abs(l - ref.l[row]) / l_tol);
                margins.m_over += std::abs(m - ref.m[row]) > m_tol;
                margins.l_over += std::abs(l - ref.l[row]) > l_tol;
                if (!range.count)
                    require(m == -3e38f && l == 0.0f, "empty AIC split did not write exact neutral state");
                for (unsigned lane = 1; lane < contract::kStateLanes; ++lane)
                    margins.state_torn += w[record + lane] != m ||
                                          w[record + contract::kStateLanes + lane] != l;
                double scale = 0;
                for (int c = 0; c < kChannels; ++c)
                    scale = std::max(scale, std::abs(ref.acc[row * kChannels + c]));
                const double acc_tol = 2e-4 * scale + 1e-5 * ref.l[row] + 1e-6;
                // Spacing at the row's accumulation scale, for acc_scaled_ulp.
                const double scale_ulp = UlpOf(static_cast<float>(scale));
                int over_in_row = 0;
                for (int c = 0; c < kChannels; ++c) {
                    const float got = w[acc_at + c];
                    if (!range.count)
                        require(got == 0.0f && qr[acc_base + c] == 0xAAAAu,
                                "empty AIC split must zero its accumulator and leave query rotation untouched");
                    const double deviation = std::abs(got - ref.acc[row * kChannels + c]);
                    if (std::isfinite(got) && deviation / acc_tol > margins.acc) {
                        margins.acc = deviation / acc_tol;
                        margins.acc_item = item;
                        margins.acc_row = row;
                        margins.acc_channel = c;
                        margins.acc_got = got;
                        margins.acc_ref = ref.acc[row * kChannels + c];
                    }
                    margins.acc_checked++;
                    margins.acc_signed_sum += got - ref.acc[row * kChannels + c];
                    margins.acc_abs_sum += deviation;
                    const double ulps =
                        UlpDistance(got, static_cast<float>(ref.acc[row * kChannels + c]));
                    if (ulps > margins.acc_ulp) {
                        margins.acc_ulp = ulps;
                        margins.acc_ulp_row = row;
                        margins.acc_ulp_channel = c;
                        margins.acc_ulp_got = got;
                        margins.acc_ulp_ref = ref.acc[row * kChannels + c];
                    }
                    const double ref_abs = std::abs(ref.acc[row * kChannels + c]);
                    if (margins.acc_checked == 1 || ref_abs < margins.acc_ref_min)
                        margins.acc_ref_min = ref_abs;
                    const double scaled_ulps =
                        scale_ulp > 0 && std::isfinite(got) ? deviation / scale_ulp
                                                            : 0.0;
                    if (!std::isfinite(got) || scaled_ulps > margins.acc_scaled_ulp) {
                        margins.acc_scaled_ulp =
                            std::isfinite(got)
                                ? scaled_ulps
                                : std::numeric_limits<double>::infinity();
                        margins.acc_scaled_ulp_row = row;
                        margins.acc_scaled_ulp_channel = c;
                        margins.acc_scaled_ulp_got = got;
                        margins.acc_scaled_ulp_ref = ref.acc[row * kChannels + c];
                        margins.acc_scaled_ulp_scale = scale;
                    }
                    if (!std::isfinite(got) || deviation > acc_tol) {
                        if (margins.acc_over < 8)
                            std::fprintf(stderr, "AIC partial mismatch item=%zu row=%d c=%d "
                                         "got=%.9g ref=%.9g deviation=%.6e tolerance=%.6e (%.3fx)\n",
                                         item, row, c, got, ref.acc[row * kChannels + c],
                                         deviation, acc_tol, deviation / acc_tol);
                        margins.acc_over++;
                        over_in_row++;
                    }
                    if (range.count)
                        margins.qrot_wrong +=
                            qr[acc_base + c] !=
                            f.q[(item / splits) * 16 * kChannels + row * kChannels + c];
                }
                margins.rows_touched += over_in_row != 0;
            }
        }
        std::printf("AIC partial margins (fraction of each tolerance): m=%.5fx l=%.5fx "
                    "acc=%.5fx at item=%zu row=%d c=%d got=%.9g ref=%.9g\n",
                    margins.m, margins.l, margins.acc, margins.acc_item,
                    margins.acc_row, margins.acc_channel, margins.acc_got, margins.acc_ref);
        // The localisation. `rows_touched` and `acc_over` say whether this is
        // one channel or a whole accumulator; the mean SIGNED deviation
        // against the mean absolute one says whether the error has a
        // direction -- a biased error points at a scale or a rescale factor,
        // a symmetric one at noise in the data path.
        std::printf("AIC partial profile: acc over tolerance %d of %d channels in %d of "
                    "%d rows; m over %d, l over %d; replicas torn %d; qrot wrong %d; "
                    "mean signed deviation %.6e, mean |deviation| %.6e\n",
                    margins.acc_over, margins.acc_checked, margins.rows_touched,
                    static_cast<int>(reference.size()) * 16, margins.m_over,
                    margins.l_over, margins.state_torn, margins.qrot_wrong,
                    margins.acc_checked ? margins.acc_signed_sum / margins.acc_checked : 0.0,
                    margins.acc_checked ? margins.acc_abs_sum / margins.acc_checked : 0.0);
        std::printf("AIC partial ULP: worst_state_ulp=%.2f (%s, row %d) "
                    "worst_acc_ulp=%.2f (row %d, c %d, got=%.9g ref=%.9g); "
                    "canary words %d, nonfinite words %d; budgets %s\n",
                    margins.state_ulp, margins.state_ulp_which, margins.state_ulp_row,
                    margins.acc_ulp, margins.acc_ulp_row, margins.acc_ulp_channel,
                    margins.acc_ulp_got, margins.acc_ulp_ref,
                    margins.canary_words, margins.nonfinite_words,
                    (state_ulp_cap > 0 || acc_ulp_cap > 0 ||
                     acc_scaled_ulp_cap > 0) ? "enforced" : "reported only");
        // The scale-referenced number, with the two magnitudes that explain
        // the raw one: the row scale the budget is taken at, and the smallest
        // |ref| in the launch, which is what inflates the raw ULP.
        std::printf("AIC partial scaled ULP: worst_acc_scaled_ulp=%.2f of %.0f "
                    "(row %d, c %d, got=%.9g ref=%.9g, row scale=%.9g); "
                    "min |ref| over %d channels = %.9g\n",
                    margins.acc_scaled_ulp, acc_scaled_ulp_cap,
                    margins.acc_scaled_ulp_row, margins.acc_scaled_ulp_channel,
                    margins.acc_scaled_ulp_got, margins.acc_scaled_ulp_ref,
                    margins.acc_scaled_ulp_scale, margins.acc_checked,
                    margins.acc_ref_min);
        require(margins.canary_words == 0,
                "a published partition still holds the 0xAA canary fill: a worker "
                "left part of its record unwritten");
        require(margins.nonfinite_words == 0,
                "nonfinite word in a published partition");
        if (state_ulp_cap > 0 && margins.state_ulp > state_ulp_cap) {
            std::fprintf(stderr, "worst_state_ulp %.2f exceeds the %.0f ULP budget "
                         "(%s, row %d)\n", margins.state_ulp, state_ulp_cap,
                         margins.state_ulp_which, margins.state_ulp_row);
            throw std::runtime_error("AIC partial state outside its ULP budget");
        }
        if (acc_ulp_cap > 0 && margins.acc_ulp > acc_ulp_cap) {
            std::fprintf(stderr, "worst_acc_ulp %.2f exceeds the %.0f ULP budget "
                         "(row %d, c %d, got=%.9g ref=%.9g)\n", margins.acc_ulp,
                         acc_ulp_cap, margins.acc_ulp_row, margins.acc_ulp_channel,
                         margins.acc_ulp_got, margins.acc_ulp_ref);
            throw std::runtime_error("AIC accumulator outside its ULP budget");
        }
        if (acc_scaled_ulp_cap > 0 && margins.acc_scaled_ulp > acc_scaled_ulp_cap) {
            std::fprintf(stderr, "worst_acc_scaled_ulp %.2f exceeds the %.0f ULP "
                         "budget (row %d, c %d, got=%.9g ref=%.9g, row scale=%.9g)\n",
                         margins.acc_scaled_ulp, acc_scaled_ulp_cap,
                         margins.acc_scaled_ulp_row, margins.acc_scaled_ulp_channel,
                         margins.acc_scaled_ulp_got, margins.acc_scaled_ulp_ref,
                         margins.acc_scaled_ulp_scale);
            throw std::runtime_error("AIC accumulator outside its scaled ULP budget");
        }
        require(margins.qrot_wrong == 0, "AIC query rotation differs from BF16 reference");
        require(margins.state_torn == 0, "AIC replicated state torn");
        require(margins.m_over == 0, "AIC partial maximum differs from CPU reference");
        require(margins.l_over == 0, "AIC partial denominator differs from CPU reference");
        require(margins.acc_over == 0, "AIC accumulator reference failed");
        return {};
    }
};

}  // namespace tq_splitk_rig
