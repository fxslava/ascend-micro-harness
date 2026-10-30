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

extern "C" void launch_turboquant_single_token_decode_splitk_aiv(
    void*, std::uint16_t*, std::int8_t*, std::int8_t*, float*, std::uint32_t*,
    float*, float*, turboquant_splitk::AtomicHeadSlot*,
    turboquant_splitk::WorkerMetrics*, std::uint16_t*, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_paged_decode_splitk_aic(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, std::uint16_t*, float*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

struct AivRig {
    const Fixture& f;
    Buffer q, key, value, scales, table, gate, signs, slots, metrics, out;
    AivRig(const Fixture& fixture, int splits)
        : f(fixture), q(f.q.size() * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          gate(f.gate.size() * 4), signs(f.signs.size() * 4),
          slots(f.requests * f.heads * sizeof(contract::AtomicHeadSlot)),
          metrics(f.requests * f.heads * splits * sizeof(contract::WorkerMetrics)),
          out(f.q.size() * 2)
    {
        q.upload(f.q); key.upload(f.key); value.upload(f.value);
        scales.upload(f.scales); gate.upload(f.gate); signs.upload(f.signs);
        auto padded = f.table;
        padded.resize(table.bytes / 4, 0);
        table.upload(padded);
        require(reinterpret_cast<std::uintptr_t>(slots.ptr) % 128 == 0,
                "head slot allocation must be 128-byte aligned");
    }
    void Reset()
    {
        check(aclrtMemset(slots.ptr, slots.bytes, 0, slots.bytes), "reset head slots");
        check(aclrtMemset(metrics.ptr, metrics.bytes, 0, metrics.bytes), "reset metrics");
        out.fill_canary();
    }
    void Launch(aclrtStream stream, int splits, int watchdog)
    {
        launch_turboquant_single_token_decode_splitk_aiv(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::int8_t*>(key.ptr),
            static_cast<std::int8_t*>(value.ptr), static_cast<float*>(scales.ptr),
            static_cast<std::uint32_t*>(table.ptr), static_cast<float*>(gate.ptr),
            static_cast<float*>(signs.ptr), static_cast<contract::AtomicHeadSlot*>(slots.ptr),
            static_cast<contract::WorkerMetrics*>(metrics.ptr),
            static_cast<std::uint16_t*>(out.ptr), f.requests, f.heads, f.rows,
            f.page_tokens, f.requests * f.pages, f.pages, splits, watchdog,
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
    struct Census {
        std::uint64_t failed = 0, timeouts = 0;
        std::vector<double> tail_ticks, lock_ticks;
    };
    Census Validate(int splits)
    {
        std::vector<contract::WorkerMetrics> m(metrics.bytes / sizeof(contract::WorkerMetrics));
        metrics.download(m);
        std::vector<contract::AtomicHeadSlot> s(f.requests * f.heads);
        slots.download(s);
        Census result;
        for (int head = 0; head < f.requests * f.heads; ++head) {
            std::vector<bool> arrivals(splits + 1, false);
            std::uint64_t first = std::numeric_limits<std::uint64_t>::max(), tail_end = 0;
            int tails = 0;
            for (int split = 0; split < splits; ++split) {
                const auto& w = m[head * splits + split];
                result.failed += w.failed_exchanges;
                result.timeouts += w.timeout;
                if (!w.acquired || w.timeout)
                    std::fprintf(stderr, "INVALID lock: head=%d split=%d acquired=%u "
                                 "timeouts=%u failed_exchanges=%u\n", head, split,
                                 w.acquired, w.timeout, w.failed_exchanges);
                require(w.acquired == 1 && w.timeout == 0, "INVALID: lock watchdog expired");
                require(w.arrival >= 1 && w.arrival <= static_cast<unsigned>(splits) &&
                        !arrivals[w.arrival], "arrival tickets are not a permutation");
                arrivals[w.arrival] = true;
                const auto range = contract::Partition(f.pages, splits, split);
                require(w.pages[0] == range.count, "wrong per-split page count");
                for (int lane = 1; lane < 8; ++lane)
                    require(w.finish_hi[lane] == w.finish_hi[0] &&
                            w.finish_lo[lane] == w.finish_lo[0] &&
                            w.tail[lane] == w.tail[0] && w.pages[lane] == w.pages[0] &&
                            w.lock_region_hi[lane] == w.lock_region_hi[0] &&
                            w.lock_region_lo[lane] == w.lock_region_lo[0],
                            "torn metric payload");
                const std::uint64_t finish = (std::uint64_t{w.finish_hi[0]} << 32) |
                                              w.finish_lo[0];
                require(finish != 0, "missing completion timestamp");
                const std::uint64_t lock_region = (std::uint64_t{w.lock_region_hi[0]} << 32) |
                                                    w.lock_region_lo[0];
                require(lock_region > 0, "missing lock interval");
                result.lock_ticks.push_back(static_cast<double>(lock_region));
                first = std::min(first, finish);
                require(w.tail[0] == static_cast<unsigned>(w.arrival == static_cast<unsigned>(splits)),
                        "wrong tail worker");
                if (w.tail[0]) { ++tails; tail_end = finish; }
            }
            require(tails == 1 && s[head].lock == 0 &&
                    s[head].arrived_count == static_cast<unsigned>(splits), "unfinished head slot");
            require(tail_end >= first, "noncomparable completion timestamps");
            result.tail_ticks.push_back(static_cast<double>(tail_end - first));
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
    Buffer q, pi, key, value, scales, table, qrot, state, acc;
    std::vector<splitk_decode_test::AicPartial> reference;
    AicRig(const Fixture& fixture, int k, int first, int count)
        : f(fixture), splits(k), begin(first), pages(count), q(f.q.size() * 2),
          pi(paged_decode::kTableElems * 2), key(f.key.size()), value(f.value.size()),
          scales(f.scales.size() * 4), table((f.table.size() * 4 + 31) / 32 * 32),
          qrot(f.requests * f.heads * k * splitk_decode_test::kAicAccFloats * 2),
          state(f.requests * f.heads * k * splitk_decode_test::kAicStateFloats * 4),
          acc(f.requests * f.heads * k * splitk_decode_test::kAicAccFloats * 4)
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
    void Reset() { qrot.fill_canary(); state.fill_canary(); acc.fill_canary(); }
    void Launch(aclrtStream stream, int, int)
    {
        launch_turboquant_paged_decode_splitk_aic(stream,
            static_cast<std::uint16_t*>(q.ptr), static_cast<std::uint16_t*>(pi.ptr),
            static_cast<std::int8_t*>(key.ptr), static_cast<std::int8_t*>(value.ptr),
            static_cast<float*>(scales.ptr), static_cast<std::uint32_t*>(table.ptr),
            static_cast<std::uint16_t*>(qrot.ptr), static_cast<float*>(state.ptr),
            static_cast<float*>(acc.ptr), f.requests, f.heads, f.page_tokens,
            f.requests * f.pages, f.pages, splits, begin, pages,
            // The fixture's live token count. Equal to its capacity on an
            // aligned fixture, so this is the pre-tail behaviour there.
            static_cast<std::uint32_t>(f.seq_len));
    }
    AivRig::Census Validate(int)
    {
        std::vector<float> s(state.bytes / 4), a(acc.bytes / 4);
        std::vector<std::uint16_t> qr(qrot.bytes / 2);
        state.download(s); acc.download(a); qrot.download(qr);
        for (std::size_t item = 0; item < reference.size(); ++item) {
            const auto& ref = reference[item];
            const auto range = contract::Partition(pages, splits, item % splits);
            for (int row = 0; row < 16; ++row) {
                const auto state_base = item * splitk_decode_test::kAicStateFloats;
                const auto acc_base = item * splitk_decode_test::kAicAccFloats + row * kChannels;
                const float m = s[state_base + row * 8], l = s[state_base + 128 + row * 8];
                require(std::isfinite(m) && std::isfinite(l), "nonfinite AIC partial state");
                require(std::abs(m - ref.m[row]) <= 2e-6 * std::max(1.0, std::abs(ref.m[row])),
                        "AIC partial maximum differs from CPU reference");
                require(std::abs(l - ref.l[row]) <= 1e-4 * std::max(1.0, ref.l[row]),
                        "AIC partial denominator differs from CPU reference");
                if (!range.count)
                    require(m == -3e38f && l == 0.0f, "empty AIC split did not write exact neutral state");
                for (int lane = 1; lane < 8; ++lane)
                    require(s[state_base + row * 8 + lane] == m &&
                            s[state_base + 128 + row * 8 + lane] == l, "AIC replicated state torn");
                double scale = 0;
                for (int c = 0; c < kChannels; ++c)
                    scale = std::max(scale, std::abs(ref.acc[row * kChannels + c]));
                for (int c = 0; c < kChannels; ++c) {
                    const float got = a[acc_base + c];
                    if (!range.count)
                        require(got == 0.0f && qr[acc_base + c] == 0xAAAAu,
                                "empty AIC split must zero its accumulator and leave query rotation untouched");
                    if (!std::isfinite(got) || std::abs(got - ref.acc[row * kChannels + c]) >
                        2e-4 * scale + 1e-5 * ref.l[row] + 1e-6) {
                        std::fprintf(stderr, "AIC partial mismatch item=%zu row=%d c=%d got=%.9g ref=%.9g\n",
                                     item, row, c, got, ref.acc[row * kChannels + c]);
                        throw std::runtime_error("AIC accumulator reference failed");
                    }
                    if (range.count)
                        require(qr[acc_base + c] == f.q[(item / splits) * 16 * kChannels + row * kChannels + c],
                                "AIC query rotation differs from BF16 reference");
                }
            }
        }
        return {};
    }
};

}  // namespace tq_splitk_rig
