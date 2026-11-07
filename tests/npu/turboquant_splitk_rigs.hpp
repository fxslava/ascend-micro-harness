#pragma once
// The AIC split-K decode device rig: buffer layout, launch, and the full
// correctness census for that kernel, shared by the benchmark and by the
// NPU/simulator validation tests.
//
// It was lifted out of bench_turboquant_decode_splitk.cpp, which is still one
// of the drivers. It lives here so the benchmark and the validation tests run
// the *same* checks rather than two copies that can drift: a tolerance, a
// neutral element or a torn-payload rule that differed between the two would
// make a bench pass and a validation failure (or the reverse) say nothing.
//
// AivRig used to sit beside it, over
// kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp. That
// kernel is deleted and the rig went with it; `Census`, which both rigs
// reported, is now a namespace-level type rather than a member of the rig that
// no longer exists. The fused decode operator's Split-K path has its own rig,
// tests/npu/turboquant_fused_decode_opt_rig.hpp.
//
// What the rig establishes is written at the rig. What it cannot establish is
// that it ran: it needs a device or the simulator, so a host build of a driver
// only proves the harness compiles. See reports/decode_splitk_validation.md.

#include "turboquant_npu_common.hpp"
#include "../common/aic_splitk_decode_fixture.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tq_splitk_rig {

using tq_npu::Buffer;
using tq_npu::check;
using tq_npu::require;
namespace contract = turboquant_splitk;
using splitk_decode_test::Fixture;
using splitk_decode_test::kChannels;

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, std::uint16_t*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

// The per-launch timing census a rig reports. It was `AivRig::Census` until
// the AIV split-K decoder was deleted; AicRig reports the same two vectors, so
// the type outlived the rig it was declared in and now sits here.
//
// `export_ticks` is the direct successor of the atomic protocol's `lock_ticks`:
// the region from the state pack to the retired MTE3 burst. It is the figure
// the 43,000-cycle lock region should be compared against, which is why the
// field was renamed rather than dropped.
struct Census {
    std::vector<double> tail_ticks, export_ticks;
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
    };
    Margins margins;

    Census Validate(int)
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
        require(margins.qrot_wrong == 0, "AIC query rotation differs from BF16 reference");
        require(margins.state_torn == 0, "AIC replicated state torn");
        require(margins.m_over == 0, "AIC partial maximum differs from CPU reference");
        require(margins.l_over == 0, "AIC partial denominator differs from CPU reference");
        require(margins.acc_over == 0, "AIC accumulator reference failed");
        return {};
    }
};

}  // namespace tq_splitk_rig
