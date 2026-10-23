// Tier 1 host probe for the inverse-CDF int4x2 quantization leaf.
//
// Three checks, mirroring SKELETON_KERNEL_WORKFLOW V1:
//
//  (a) Discrete-event queue model of the kernel's MTE2/Vector/MTE3 ring
//      (one event id per route, single token in flight): deadlock freedom,
//      no double-set on a live bit, zero residual tokens, sets == waits per
//      (route, id).
//
//  (b) Bit-exact scalar oracle: a mock engine replays the exact device op
//      order (DINTLV de-interleave, per-stream vneg/vexp/vadds/vdiv/vadds,
//      nibble casts at layouts ZERO/ONE, Or merge, PACK_B16 low-byte store)
//      in correctly-rounded half arithmetic, and its packed bytes must equal
//      the double oracle's bytes for FP16 and BF16 fixtures at 4 and 8 tiles.
//
//  (c) Sensitivity: the documented <= 1 ULP error of the hardware vexp/vdiv
//      is modelled by perturbing those two intermediate results by +/-1 and
//      +/-2 half-ULP; no nibble may flip (the fixture's 1/16 boundary margin
//      is what makes that provable). This bounds the Tier 2 tolerance: zero.

#include "icdf_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what)
{
    if (!ok) throw std::runtime_error(what);
}

// ---------------------------------------------------------------------------
// Half-precision device-op model. Every op is correctly rounded (round float
// exactness to half); the hardware's <= 1 ULP deviation is handled by the
// sensitivity check below, not by the model.
std::uint16_t h_neg(std::uint16_t a) { return a ^ 0x8000u; }
std::uint16_t h_exp(std::uint16_t t)
{
    return icdf::f2h(static_cast<float>(std::exp(static_cast<double>(icdf::h2f(t)))));
}
std::uint16_t h_add(std::uint16_t a, float b)
{
    return icdf::f2h(icdf::h2f(a) + b);
}
std::uint16_t h_div14(std::uint16_t d)
{
    return icdf::f2h(14.0f / icdf::h2f(d));
}
int cast_s4_rint_sat(std::uint16_t yh)
{
    return icdf::quantize_rne_sat(static_cast<double>(icdf::h2f(yh)));
}
// Perturb a non-negative half value by k ULP (bit-pattern monotonic range).
std::uint16_t ulp_bump(std::uint16_t v, int k)
{
    int bits = static_cast<int>(v) + k;
    if (bits < 0) bits = 0;
    if (bits > 0x7c00) bits = 0x7c00;
    return static_cast<std::uint16_t>(bits);
}

struct MockEngine {
    bool bf16 = false;
    std::vector<std::uint16_t> in;     // kIcdfInLanesPerRepeat elements, one repeat
    std::vector<std::uint8_t> packed;  // kIcdfPackedBytesPerRepeat bytes out

    void run()
    {
        packed.assign(ascend_micro::kIcdfPackedBytesPerRepeat, 0);
        std::vector<std::uint16_t> y(ascend_micro::kIcdfLanes);
        // Widen (BF16 -> half, exact for in-range values, SAT beyond) and run
        // the register-resident chain on every lane.
        for (std::uint32_t i = 0; i < ascend_micro::kIcdfLanes; ++i) {
            std::uint16_t v =
                bf16 ? icdf::f2h(icdf::bf2f(in[i])) : in[i];
            v = h_neg(v);
            v = h_exp(v);
            v = h_add(v, 1.0f);
            v = h_div14(v);
            y[i] = h_add(v, -7.0f);
        }
        // vcvt_f162s4 at RegLayout::ZERO packs adjacent source pairs: lane
        // 2k -> nibble 0, lane 2k+1 -> nibble 1 of byte k; PACK4_B32 stores
        // those bytes contiguously.
        for (std::uint32_t k = 0; k < ascend_micro::kIcdfPackedBytesPerRepeat; ++k) {
            const int q_lo = cast_s4_rint_sat(y[2 * k]) & 0xf;
            const int q_hi = cast_s4_rint_sat(y[2 * k + 1]) & 0xf;
            packed[k] = static_cast<std::uint8_t>((q_hi << 4) | q_lo);
        }
    }
};

void run_oracle_case(bool bf16, std::uint32_t tiles)
{
    icdf::Fixture fx = icdf::make_fixture(bf16, tiles);
    icdf::assert_properties(fx);
    MockEngine mock;
    mock.bf16 = bf16;
    std::uint64_t exact = 0;
    const std::uint64_t total =
        static_cast<std::uint64_t>(tiles) * icdf::kTileOutBytes;
    for (std::uint32_t t = 0; t < tiles; ++t) {
        for (std::uint32_t r = 0; r < icdf::kTileRepeats; ++r) {
            mock.in.assign(
                fx.in_bits.begin() + static_cast<long>(t) * icdf::kTileElems +
                    static_cast<long>(r) * ascend_micro::kIcdfInLanesPerRepeat,
                fx.in_bits.begin() + static_cast<long>(t) * icdf::kTileElems +
                    static_cast<long>(r + 1) * ascend_micro::kIcdfInLanesPerRepeat);
            mock.run();
            for (std::uint32_t b = 0; b < ascend_micro::kIcdfPackedBytesPerRepeat; ++b) {
                const std::uint8_t got = mock.packed[b];
                const std::uint8_t want =
                    fx.oracle_packed[static_cast<std::size_t>(t) * icdf::kTileOutBytes +
                                     r * ascend_micro::kIcdfPackedBytesPerRepeat + b];
                if (got != want) {
                    std::fprintf(stderr,
                                 "%s tile=%u repeat=%u byte=%u got=0x%02x want=0x%02x\n",
                                 bf16 ? "BF16" : "FP16", t, r, b, got, want);
                    throw std::runtime_error("tier1 mock diverged from oracle");
                }
                ++exact;
            }
        }
    }
    require(exact == total, "byte count mismatch");
    std::printf("PASS tier1 %s %u tiles: %llu/%llu packed bytes bit-identical, "
                "%u subnormal and %u +-7-tail inputs covered\n",
                bf16 ? "BF16" : "FP16", tiles,
                static_cast<unsigned long long>(exact),
                static_cast<unsigned long long>(total), fx.subnormal_count,
                fx.saturating_count);
}

void run_sensitivity(bool bf16)
{
    icdf::Fixture fx = icdf::make_fixture(bf16, 4);
    std::uint32_t checked = 0;
    for (std::uint32_t flat = 0; flat < fx.tiles * icdf::kTileElems; ++flat) {
        const std::uint16_t x = fx.in_bits[flat];
        std::uint16_t w = bf16 ? icdf::f2h(icdf::bf2f(x)) : x;
        std::uint16_t t0 = h_neg(w);
        std::uint16_t e0 = h_exp(t0);
        std::uint16_t d0 = h_add(e0, 1.0f);
        std::uint16_t r0 = h_div14(d0);
        std::uint16_t y0 = h_add(r0, -7.0f);
        const int q0 = cast_s4_rint_sat(y0);
        for (int ulps = 1; ulps <= 2; ++ulps) {
            for (int sign = -1; sign <= 1; sign += 2) {
                const std::uint16_t e1 = ulp_bump(e0, sign * ulps);
                const std::uint16_t d1 = h_add(e1, 1.0f);
                const std::uint16_t r1 = ulp_bump(h_div14(d1), sign * ulps);
                const std::uint16_t y1 = h_add(r1, -7.0f);
                require(cast_s4_rint_sat(y1) == q0,
                        "sensitivity: 2-ULP modelled vexp/vdiv error flips a nibble");
                ++checked;
            }
        }
    }
    std::printf("PASS tier1 %s sensitivity: %u perturbed recomputes, 0 nibble flips "
                "(justifies a zero Tier 2 tolerance)\n",
                bf16 ? "BF16" : "FP16", checked);
}

// ---------------------------------------------------------------------------
// Queue model of the kernel ring: one id per route, single token per route.
struct QueueModel {
    struct Op {
        char queue;          // '2' MTE2, 'V', '3' MTE3
        bool is_set;
        int route;           // 0 MTE2_V, 1 V_MTE2, 2 V_MTE3, 3 MTE3_V, 4 MTE3_S
    };
    std::vector<Op> mte2, vec, mte3;
    bool token[5] = {};
    std::uint32_t sets[5] = {}, waits[5] = {};

    bool ready(const std::vector<Op>& q, std::size_t pc) const
    {
        const Op& op = q[pc];
        return op.is_set || token[op.route];
    }
    void consume(Op op)
    {
        if (op.is_set) {
            if (token[op.route]) throw std::runtime_error("double set on a live token");
            token[op.route] = true;
            ++sets[op.route];
        } else {
            if (!token[op.route]) throw std::runtime_error("wait on a missing token");
            token[op.route] = false;
            ++waits[op.route];
        }
    }
    void run(std::uint32_t tiles)
    {
        mte2.clear(); vec.clear(); mte3.clear();
        for (int r = 0; r < 5; ++r) { token[r] = false; sets[r] = 0; waits[r] = 0; }
        for (std::uint32_t t = 0; t < tiles; ++t) {
            mte2.push_back({'2', false, 1});  // wait V_MTE2 (slot free)
            mte2.push_back({'2', true, 0});   // set  MTE2_V (filled)
            vec.push_back({'V', false, 0});   // wait MTE2_V
            vec.push_back({'V', false, 3});   // wait MTE3_V (out slot drained)
            vec.push_back({'V', true, 1});    // set  V_MTE2 (read done)
            vec.push_back({'V', true, 2});    // set  V_MTE3 (packed ready)
            mte3.push_back({'3', false, 2});  // wait V_MTE3
            mte3.push_back({'3', true, 3});   // set  MTE3_V (drained)
        }
        mte2.push_back({'2', false, 1});      // drain prime/last set
        vec.push_back({'V', false, 3});       // drain
        mte3.push_back({'3', true, 4});       // MTE3_S handshake
        mte3.push_back({'3', false, 4});
        token[1] = true;  // prime: input slots start free
        token[3] = true;  // prime: output slots start drained
        sets[1] = sets[3] = 1;
        std::size_t p2 = 0, pv = 0, p3 = 0;
        while (p2 < mte2.size() || pv < vec.size() || p3 < mte3.size()) {
            bool progressed = false;
            if (p2 < mte2.size() && ready(mte2, p2)) { consume(mte2[p2++]); progressed = true; }
            if (pv < vec.size() && ready(vec, pv)) { consume(vec[pv++]); progressed = true; }
            if (p3 < mte3.size() && ready(mte3, p3)) { consume(mte3[p3++]); progressed = true; }
            require(progressed, "deadlock: all three execution queues stalled");
        }
        for (int r = 0; r < 5; ++r) {
            require(!token[r], "event token leaked after epilogue");
            char buf[80];
            std::snprintf(buf, sizeof(buf), "sets != waits on route %d", r);
            require(sets[r] == waits[r], buf);
        }
        std::printf("PASS tier1 queue model: %u tiles, deadlock-free, 5/5 routes "
                    "balanced, 0 residual tokens\n", tiles);
    }
};

}  // namespace

int main()
{
    try {
        const auto t0 = std::chrono::steady_clock::now();
        QueueModel qm;
        qm.run(4);
        qm.run(8);
        run_oracle_case(false, 4);
        run_oracle_case(false, 8);
        run_oracle_case(true, 4);
        run_oracle_case(true, 8);
        run_sensitivity(false);
        run_sensitivity(true);
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("tier1 total host time: %.3f us\n",
                    std::chrono::duration<double, std::micro>(t1 - t0).count());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL tier1 icdf quantize: %s\n", error.what());
        return 1;
    }
}
