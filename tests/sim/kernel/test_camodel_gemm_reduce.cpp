// CAModel runner for the Cube Split-K reduction, in both of its forms.
//
// Five legs over one workspace: the M = 1 GEMV stage, that stage with Mmad's GEMV mode left
// enabled, the dense batched GEMM (M = kGemmBatchRows packed query rows against 16-channel
// slices), the batched form with the same knob, and the shipping TurboQuantPartialReducer as
// the numerical reference. Each run floods every GM payload, uploads only the slices it means
// to assign, downloads after synchronisation and compares bitwise, so a stray write is a canary
// failure and a wrong read is a numerical one.
//
//   --splits S        the split count, 1..8
//   --tokens N        tokens
//   --heads H         query heads per token
//   --grid G          blocks; the per-block share is ceil(tokens*heads / G)
//   --leg gemm|gemv|vector|batched|batched-gemv
//   --empty-worker    size the grid so the last block's share is empty
//   --mixed           drive a lengths vector so some tokens are fused and skipped
//   --fixture exact|varied
//
// Usage is through scripts/run_gemm_reduce_camodel.sh; nothing here reads a default geometry
// from anywhere but the shared contract header.

#include "../../../include/turboquant_gemm_reduce.hpp"
#include "fused_decode_opt_sim_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace tq = ascend_micro::gemm_reduce;

extern "C" void launch_gemm_reduce_sim(void *stream, const tq::ReduceLaunch *,
                                       const tq::ReduceBuffers *);

namespace {

// §3's flood: an fp32 quiet NaN, so a write shows up as a canary failure. The int4/BF16 views
// of this word are byte patterns and not guaranteed NaNs, which is why canaries detect writes
// and the numerical oracle detects wrong reads -- neither claim covers the other.
constexpr std::uint32_t kFlood = 0x7FC0BEEFu;
constexpr std::uint32_t kHalfBytes = 2;

struct Options {
    std::uint32_t splits = 8, tokens = 2, heads = 4, grid = 1;
    bool empty_worker = false, mixed = false, exact = true;
    tq::ReduceLeg leg = tq::ReduceLeg::kGemm;
};

[[noreturn]] void die(const std::string &why)
{
    throw std::runtime_error(why);
}

std::uint32_t number(const char *s)
{
    return static_cast<std::uint32_t>(std::strtoul(s, nullptr, 10));
}

Options parse(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--splits") o.splits = number(value());
        else if (a == "--tokens") o.tokens = number(value());
        else if (a == "--heads") o.heads = number(value());
        else if (a == "--grid") o.grid = number(value());
        else if (a == "--empty-worker") o.empty_worker = true;
        else if (a == "--mixed") o.mixed = true;
        else if (a == "--fixture") o.exact = std::string(value()) == "exact";
        else if (a == "--leg") {
            const std::string leg = value();
            if (leg == "gemm") o.leg = tq::ReduceLeg::kGemm;
            else if (leg == "gemv") o.leg = tq::ReduceLeg::kGemmGemvMode;
            else if (leg == "vector") o.leg = tq::ReduceLeg::kVector;
            else if (leg == "batched") o.leg = tq::ReduceLeg::kBatched;
            else if (leg == "batched-gemv") o.leg = tq::ReduceLeg::kBatchedGemvMode;
            else die("unknown leg: " + leg);
        } else die("unknown option: " + a);
    }
    if (!tq::kValidReduceGeometry(tq::kHeadSize, o.splits)) die("unsupported split count");
    if (!o.tokens || !o.heads || !o.grid) die("tokens, heads and grid must be positive");
    return o;
}

// The exact fixture: every value a small dyadic rational, so the whole combine is exact in
// fp32 and the comparison is bitwise rather than tolerance-bounded. Statistics are equal
// across splits, which makes every weight exactly 1 and the combine a plain sum -- the one
// case where a reordered summation cannot hide behind a tolerance.
//
// The varied fixture is the arithmetic regression: distinct per-split maxima, so the weights
// are genuine exponentials and the two legs' summation orders differ. It is scored against the
// double-precision reference with a stated budget, and the exact fixture's bitwise gate is
// never relaxed to make it pass.
struct Fixture {
    std::uint32_t tokens, heads, splits;
    bool exact;

    float partial(std::uint32_t token, std::uint32_t head, std::uint32_t split,
                  std::uint32_t channel) const
    {
        if (exact) {
            // Dyadic, and dependent on all four indices so a transposed or slot-confused read
            // lands on a different value rather than on a plausible one.
            const float base = 1.0f + 0.25f * static_cast<float>(channel % 8);
            const float shift = 0.5f * static_cast<float>(split) +
                                0.125f * static_cast<float>((token * heads + head) % 4);
            return base + shift + ((channel % 2) ? 0.0625f : 0.0f);
        }
        const float x = 0.37f * static_cast<float>(channel + 1) +
                        0.11f * static_cast<float>(split + 1) +
                        0.07f * static_cast<float>(token * heads + head + 1);
        return std::sin(x) * (1.0f + 0.5f * static_cast<float>(split));
    }

    float maximum(std::uint32_t token, std::uint32_t head, std::uint32_t split) const
    {
        if (exact) return 0.0f;  // every weight exactly exp(0) = 1
        return -0.5f + 0.75f * static_cast<float>(split) +
               0.25f * static_cast<float>((token * heads + head) % 3);
    }

    float mass(std::uint32_t token, std::uint32_t head, std::uint32_t split) const
    {
        if (exact) return 1.0f + 0.5f * static_cast<float>(split);
        return 0.5f + 0.3f * static_cast<float>(split) +
               0.1f * static_cast<float>((token * heads + head) % 5);
    }
};

// The oracle, in double, from the same layout the kernel reads.
std::vector<double> reference(const Fixture &f, std::uint32_t token, std::uint32_t head)
{
    double star = -std::numeric_limits<double>::infinity();
    for (std::uint32_t s = 0; s < f.splits; ++s) {
        star = std::max(star, static_cast<double>(f.maximum(token, head, s)));
    }
    double denom = 0.0;
    std::vector<double> num(tq::kHeadSize, 0.0);
    for (std::uint32_t s = 0; s < f.splits; ++s) {
        const double w = std::exp(static_cast<double>(f.maximum(token, head, s)) - star);
        denom += w * static_cast<double>(f.mass(token, head, s));
        for (std::uint32_t c = 0; c < tq::kHeadSize; ++c) {
            num[c] += w * static_cast<double>(f.partial(token, head, s, c));
        }
    }
    // The same 1/(l* + eps) the shared NormalizeHeads applies, so the oracle is comparable to
    // both legs rather than to an idealised quotient neither computes.
    const double inv = 1.0 / (denom + 1e-20);
    for (double &v : num) v *= inv;
    return num;
}

float half_to_float(std::uint16_t h)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) << 31;
    std::uint32_t exponent = (h >> 10) & 0x1F;
    std::uint32_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) {
            const std::uint32_t bits = sign;
            float out;
            std::memcpy(&out, &bits, 4);
            return out;
        }
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            --exponent;
        }
        ++exponent;
        mantissa &= 0x3FF;
    } else if (exponent == 31) {
        const std::uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
        float out;
        std::memcpy(&out, &bits, 4);
        return out;
    }
    const std::uint32_t bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

const char *leg_name(tq::ReduceLeg leg)
{
    switch (leg) {
        case tq::ReduceLeg::kGemm: return "gemm";
        case tq::ReduceLeg::kGemmGemvMode: return "gemv";
        case tq::ReduceLeg::kVector: return "vector";
        case tq::ReduceLeg::kBatched: return "batched";
        case tq::ReduceLeg::kBatchedGemvMode: return "batched-gemv";
    }
    return "?";
}

int run(const Options &o)
{
    const Fixture fixture{o.tokens, o.heads, o.splits, o.exact};
    const std::uint32_t total = o.tokens * o.heads;
    // An empty worker is a required rung (§4): one extra block whose share starts past the end.
    const std::uint32_t grid = o.empty_worker ? o.grid + 1 : o.grid;
    const std::uint32_t per_block = (total + o.grid - 1) / o.grid;

    const std::size_t workspace_floats =
        static_cast<std::size_t>(total) * o.splits * tq::kPartialStride;
    const std::size_t output_elems = static_cast<std::size_t>(total) * tq::kHeadSize;

    camodel::Session session;
    camodel::GuardedDeviceBuffer workspace(workspace_floats * sizeof(float));
    camodel::GuardedDeviceBuffer output(output_elems * kHalfBytes);
    // tokens + 1, not tokens: the overrun check below reads the word just past the live tokens
    // and requires it to still hold the flood, so that word has to be inside the PAYLOAD. At a
    // token count that is an exact multiple of 8 the old rounding put it in the trailing guard,
    // where it holds the canary instead -- a spurious "read or written past its live tokens"
    // at tokens = 8 and 16 and nowhere else, which is why the GEMV ladder's rungs (1, 2, 5)
    // never showed it.
    camodel::GuardedDeviceBuffer lengths(((o.tokens + 1 + 7) / 8) * 8 * sizeof(std::int32_t));

    // Flood every payload before assigning only the slices this run means to write.
    std::fill(workspace.host.begin(), workspace.host.end(), kFlood);
    std::fill(output.host.begin(), output.host.end(), kFlood);
    std::fill(lengths.host.begin(), lengths.host.end(), kFlood);
    // The guards are the canary word, not the flood: a payload NaN and a guard sentinel have to
    // be distinguishable or an overrun into the guard reads as intact.
    const std::size_t guard_words = camodel::kGuardBytes / 4;
    auto reset_guards = [&](camodel::GuardedDeviceBuffer &b) {
        for (std::size_t i = 0; i < guard_words; ++i) b.host[i] = camodel::kCanary;
        for (std::size_t i = b.host.size() - guard_words; i < b.host.size(); ++i) {
            b.host[i] = camodel::kCanary;
        }
    };
    reset_guards(workspace);
    reset_guards(output);
    reset_guards(lengths);

    auto *ws = static_cast<float *>(workspace.host_payload());
    auto *len = static_cast<std::int32_t *>(lengths.host_payload());

    const std::int32_t fused_limit = o.mixed ? 4096 : -1;
    std::vector<bool> combined(o.tokens, true);
    for (std::uint32_t t = 0; t < o.tokens; ++t) {
        // Mixed: every other token stays under the fused limit and must be left untouched, so
        // its output slice is a canary for a task-slicing error.
        const bool fused = o.mixed && (t % 2 == 1);
        combined[t] = !fused;
        len[t] = fused ? 1024 : 8192;
    }

    for (std::uint32_t t = 0; t < o.tokens; ++t) {
        for (std::uint32_t h = 0; h < o.heads; ++h) {
            for (std::uint32_t s = 0; s < o.splits; ++s) {
                const std::size_t row =
                    tq::kPartialRowOffset(t, h, s, o.heads, o.splits);
                for (std::uint32_t c = 0; c < tq::kHeadSize; ++c) {
                    ws[row + c] = fixture.partial(t, h, s, c);
                }
                // Only the two statistics lanes are assigned. The rest of the tail keeps the
                // flood, so a reducer that reads the wrong tail lane reads a NaN.
                ws[row + tq::kHeadSize + tq::kPartialMaxLane] = fixture.maximum(t, h, s);
                ws[row + tq::kHeadSize + tq::kPartialSumLane] = fixture.mass(t, h, s);
            }
        }
    }

    const std::vector<std::uint32_t> workspace_snapshot = workspace.host;
    workspace.upload();
    output.upload();
    lengths.upload();

    tq::ReduceLaunch launch;
    launch.grid = grid;
    launch.tokens = o.tokens;
    launch.heads = o.heads;
    launch.head_size = tq::kHeadSize;
    launch.splits = o.splits;
    launch.per_block = per_block;
    launch.fused_limit = fused_limit;
    launch.leg = o.leg;

    tq::ReduceBuffers buffers;
    buffers.workspace = workspace.payload();
    buffers.output = output.payload();
    // Always a real allocation: the leg is chosen by fused_limit, never by a null pointer.
    buffers.lengths = lengths.payload();

    launch_gemm_reduce_sim(session.stream, &launch, &buffers);
    camodel::check(aclrtSynchronizeStream(session.stream), "synchronize reduce");

    workspace.download();
    output.download();
    lengths.download();

    workspace.assert_guards();
    output.assert_guards();
    lengths.assert_guards();
    // The reduction reads the workspace and must not write a byte of it.
    if (workspace.host != workspace_snapshot) {
        die("the reduction wrote to its own input workspace");
    }
    if (lengths.host[guard_words + o.tokens] != kFlood) {
        die("the length vector was read or written past its live tokens");
    }

    const auto *out = static_cast<const std::uint16_t *>(
        static_cast<const void *>(static_cast<const std::uint8_t *>(
            static_cast<const void *>(output.host.data())) + camodel::kGuardBytes));

    double worst_abs = 0.0, worst_rel = 0.0;
    std::uint32_t worst_ulp = 0, checked = 0, skipped = 0;
    // The first few disagreements, with the ring slot they landed in: "which task" and "which
    // slot" is the difference between an arithmetic bug and a buffer-lifetime one.
    struct Bad {
        std::uint32_t task, token, head, channel;
        float got;
        double want;
        std::uint16_t bits;
    };
    std::vector<Bad> bad;
    for (std::uint32_t t = 0; t < o.tokens; ++t) {
        for (std::uint32_t h = 0; h < o.heads; ++h) {
            const std::size_t base =
                (static_cast<std::size_t>(t) * o.heads + h) * tq::kHeadSize;
            if (!combined[t]) {
                // A fused token's slice was never assigned and must still hold the flood. Its
                // low and high halves are checked at their own word positions: accepting
                // either would miss a halfword swap.
                for (std::uint32_t c = 0; c < tq::kHeadSize; ++c) {
                    const std::uint16_t want =
                        (c % 2 == 0) ? static_cast<std::uint16_t>(kFlood & 0xFFFF)
                                     : static_cast<std::uint16_t>(kFlood >> 16);
                    if (out[base + c] != want) {
                        die("a skipped token's output slot was written at channel " +
                            std::to_string(c));
                    }
                }
                ++skipped;
                continue;
            }
            const std::vector<double> want = reference(fixture, t, h);
            for (std::uint32_t c = 0; c < tq::kHeadSize; ++c) {
                const float got = half_to_float(out[base + c]);
                const double expect = want[c];
                const double abs = std::fabs(static_cast<double>(got) - expect);
                worst_abs = std::max(worst_abs, abs);
                if (std::fabs(expect) > 1e-6) {
                    worst_rel = std::max(worst_rel, abs / std::fabs(expect));
                }
                // ULP in the fp16 the operator actually exports, which is the unit the output
                // is quantised to: a double-precision residual below half an fp16 ULP is not
                // an error the output can represent.
                const float nearest = half_to_float(out[base + c]);
                float ideal = static_cast<float>(expect);
                std::uint16_t ideal_half = 0;
                {
                    // Round the oracle to fp16 the same way the kernel's Cast does (RINT).
                    const float scaled = ideal;
                    std::uint32_t bits;
                    std::memcpy(&bits, &scaled, 4);
                    const std::uint32_t sign = bits >> 31;
                    const std::int32_t exponent =
                        static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127;
                    std::uint32_t mantissa = bits & 0x7FFFFF;
                    if (exponent < -14) {
                        ideal_half = static_cast<std::uint16_t>(sign << 15);
                    } else if (exponent > 15) {
                        ideal_half = static_cast<std::uint16_t>((sign << 15) | 0x7C00);
                    } else {
                        const std::uint32_t shifted = mantissa >> 13;
                        const std::uint32_t remainder = mantissa & 0x1FFF;
                        std::uint32_t rounded = shifted;
                        if (remainder > 0x1000 || (remainder == 0x1000 && (shifted & 1))) {
                            ++rounded;
                        }
                        std::uint32_t exp_bits =
                            static_cast<std::uint32_t>(exponent + 15) + (rounded >> 10);
                        ideal_half = static_cast<std::uint16_t>(
                            (sign << 15) | (exp_bits << 10) | (rounded & 0x3FF));
                    }
                }
                const std::int32_t ulp =
                    static_cast<std::int32_t>(out[base + c] & 0x7FFF) -
                    static_cast<std::int32_t>(ideal_half & 0x7FFF);
                worst_ulp = std::max(worst_ulp, static_cast<std::uint32_t>(std::abs(ulp)));
                static_cast<void>(nearest);
                if (std::abs(ulp) > 2 && bad.size() < 8) {
                    bad.push_back(Bad{t * o.heads + h, t, h, c, got, expect, out[base + c]});
                }
                ++checked;
            }
        }
    }

    if (!bad.empty()) {
        // Which task, which ring slot, and -- for the batched legs -- which packed row, which
        // subcore and which 16-channel slice. That is the difference between an arithmetic bug,
        // a buffer-lifetime one and a packing one, and guessing between them is expensive.
        std::printf("first disagreeing channels:\n");
        for (const Bad &b : bad) {
            const std::uint32_t batch = b.task / tq::kGemmBatchRows;
            const std::uint32_t inBatch = b.task % tq::kGemmBatchRows;
            const std::uint32_t rows0 = tq::kGemmLiveRowsOf(tq::kGemmBatchRows, 0);
            const std::uint32_t sub = inBatch >= rows0 ? 1u : 0u;
            std::printf("  task=%u token=%u head=%u channel=%u | gemv_slot=%u | "
                        "batch=%u batch_slot=%u row=%u subcore=%u slice=%u "
                        "| got=%g want=%g bits=%04x\n",
                        b.task, b.token, b.head, b.channel, b.task % tq::kReduceSlots, batch,
                        batch % tq::kBatchSlots, inBatch, sub, b.channel / tq::kGemmSliceCols,
                        static_cast<double>(b.got), b.want, static_cast<unsigned>(b.bits));
        }
    }
    std::printf("leg=%s fixture=%s splits=%u tokens=%u heads=%u grid=%u per_block=%u "
                "checked=%u skipped_heads=%u\n",
                leg_name(o.leg), o.exact ? "exact" : "varied", o.splits, o.tokens, o.heads,
                grid, per_block, checked, skipped);
    std::printf("worst_abs=%.6e worst_rel=%.6e worst_fp16_ulp=%u\n", worst_abs, worst_rel,
                worst_ulp);

    // The exact fixture is a bitwise gate: every weight is exactly 1, every partial a dyadic
    // rational, so both the sum and the fp16 cast are exact and the only admissible ULP is 0.
    // This threshold is never relaxed to accommodate the varied fixture.
    const std::uint32_t budget = o.exact ? 0u : 2u;
    if (worst_ulp > budget) {
        std::fprintf(stderr, "FAIL worst fp16 ULP %u exceeds the %s budget of %u\n", worst_ulp,
                     o.exact ? "exact" : "varied", budget);
        return 1;
    }
    std::printf("PASS gemm reduce: canaries intact, workspace unmodified, ULP within budget\n");
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        return run(parse(argc, argv));
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
