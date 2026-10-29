// CAModel runner for the Split-K synchronisation probe.
// Ascend950PR_9599, CANN 9.2.0-beta.2, KERNEL_TYPE_AIV_ONLY, K concurrent AIVs.
//
//   --mode a|b|c                   A collision-free, B shared+spinlock, C tail egress
//   --splits K                     concurrent splits, 1..8 (default 4)
//   --reps N                       acquisitions per split, 1..8 (default 4)
//   --fixture uniform|staggered    default uniform
//
// One mode per process: the directly linked CAModel drops its per-core
// instruction dumps in the current directory and overwrites them on the next
// run, so a comparison across modes is several processes and several archived
// dump sets rather than several launches in one.
//
// What each mode's gate actually is
// ---------------------------------
// Mode A  every split's own slice holds its own state, bit-exactly, and no
//         slice holds another split's. The contributions are 2^k . (c + 1),
//         so a slice written by the wrong core is identifiable and not merely
//         detectable.
//
// Mode B  three independent gates, and the first is the one that matters:
//
//           * the ticket log is a PERMUTATION of 0 .. K.reps-1. Each
//             acquisition takes a ticket from a Global Memory counter while
//             holding the lock, so a duplicate ticket means two cores were
//             inside the critical section at once. This witnesses mutual
//             exclusion without reference to the arithmetic, which matters
//             because an arithmetic check can be passed by a broken lock
//             whose races happen not to overlap on this particular run.
//           * the combined accumulator. On the uniform fixture every rescale
//             factor is exactly 1.0f, so the combine is an exact integer sum
//             and the comparison is bit equality -- a LOST UPDATE detector,
//             and the shortfall names the split that lost it. On the
//             staggered fixture the oracle replays the device's own ticket
//             log, so the arrival order is the one that happened rather than
//             an assumed one, and the comparison is a stated tolerance.
//           * the atomically reduced maximum equals the maximum over the
//             splits, bit-exactly. The fixtures use NEGATIVE maxima
//             deliberately: `asc_atomic_max` is documented as taking a
//             `__gm__ float*` but not as to how it compares, and a
//             sign-magnitude integer comparison would order negatives
//             backwards. This probe reports that rather than assuming it.
//
// Mode C  the tail split's egress against the host reference, to a bf16
//         tolerance, and the shared record left unmodified -- the tail reads
//         the combined state, it does not combine.
//
// Poison, not zero: every output buffer is filled with a quiet NaN pattern
// before the launch, so a word the kernel never wrote reads as NaN in the
// comparison rather than as a plausible small number.

#include "camodel_common.hpp"
#include "../common/l2_atomic_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

extern "C" void launch_camodel_l2_atomic_mode_a(void*, std::uint32_t, float*, float*,
                                                std::uint32_t*, std::uint32_t*,
                                                std::uint32_t*, std::uint32_t*, float*,
                                                float*, float*, std::uint16_t*,
                                                std::int32_t*, std::uint32_t);
extern "C" void launch_camodel_l2_atomic_mode_b(void*, std::uint32_t, float*, float*,
                                                std::uint32_t*, std::uint32_t*,
                                                std::uint32_t*, std::uint32_t*, float*,
                                                float*, float*, std::uint16_t*,
                                                std::int32_t*, std::uint32_t);
extern "C" void launch_camodel_l2_atomic_mode_c(void*, std::uint32_t, float*, float*,
                                                std::uint32_t*, std::uint32_t*,
                                                std::uint32_t*, std::uint32_t*, float*,
                                                float*, float*, std::uint16_t*,
                                                std::int32_t*, std::uint32_t);

namespace {

using namespace l2_probe_oracle;

constexpr std::uint32_t kPoison = 0x7FC0BEEFu;
constexpr std::uint32_t kPoisonU32 = 0xFEEDFACEu;
constexpr std::uint16_t kPoisonB16 = 0x7FC0u;
constexpr std::uint32_t kMaxSplits = 8;
constexpr std::uint32_t kMaxReps = 8;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t bits_of(float value)
{
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof raw);
    return raw;
}

int parse_u32(std::string_view text, std::uint32_t& out)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc() ? 0 : -1;
}

void fill_poison_u32(camodel::GuardedDeviceBuffer& buffer, std::uint32_t pattern)
{
    auto* words = static_cast<std::uint32_t*>(buffer.host_payload());
    for (std::size_t i = 0; i < buffer.payload_bytes / 4; ++i) words[i] = pattern;
}

// A metric slot's value, and whether its eight replicated lanes agree. A slot
// whose lanes disagree was not written by one `Duplicate`, which is a
// stronger statement than "the value looks wrong".
struct MetricRead {
    std::int32_t value = 0;
    bool consistent = false;
    bool written = false;
};

MetricRead read_metric(const std::int32_t* slice, std::uint32_t slot)
{
    MetricRead out;
    const std::int32_t* lanes = slice + slot * kMetricLanesPerSlot;
    out.value = lanes[0];
    out.consistent = true;
    for (std::uint32_t i = 1; i < kMetricLanesPerSlot; ++i)
        if (lanes[i] != lanes[0]) out.consistent = false;
    out.written = static_cast<std::uint32_t>(lanes[0]) != kPoisonU32;
    return out;
}

const char* mode_name(char mode)
{
    switch (mode) {
        case 'a': return "A collision-free workspace";
        case 'b': return "B shared slot under a GM spinlock";
        default: return "C tail-split egress flush";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    char mode = 'b';
    std::uint32_t splits = 4;
    std::uint32_t reps = 4;
    Fixture fixture = Fixture::Uniform;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            const std::string_view value = argv[++i];
            if (value != "a" && value != "b" && value != "c") {
                std::fprintf(stderr, "--mode must be a, b or c\n");
                return 2;
            }
            mode = value[0];
        } else if (arg == "--splits" && i + 1 < argc) {
            if (parse_u32(argv[++i], splits) != 0 || splits == 0 || splits > kMaxSplits) {
                std::fprintf(stderr, "--splits must be 1..%u\n", kMaxSplits);
                return 2;
            }
        } else if (arg == "--reps" && i + 1 < argc) {
            if (parse_u32(argv[++i], reps) != 0 || reps == 0 || reps > kMaxReps) {
                std::fprintf(stderr, "--reps must be 1..%u\n", kMaxReps);
                return 2;
            }
        } else if (arg == "--fixture" && i + 1 < argc) {
            const std::string_view value = argv[++i];
            if (value == "uniform") {
                fixture = Fixture::Uniform;
            } else if (value == "staggered") {
                fixture = Fixture::Staggered;
            } else {
                std::fprintf(stderr, "--fixture must be uniform or staggered\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %.*s\n", static_cast<int>(arg.size()),
                         arg.data());
            return 2;
        }
    }

    const bool uniform = fixture == Fixture::Uniform;
    std::printf(
        "L2 atomic Split-K probe: mode %s, K=%u split(s), %u rep(s), %s fixture, "
        "d=%u, record %u B\n",
        mode_name(mode), splits, reps, uniform ? "uniform" : "staggered", kChannels,
        kRecordElems * 4u);

    // ------------------------------------------------------------- fixtures
    const std::vector<float> tiles = tile_array(fixture, splits);
    const std::vector<float> signs = rot_oracle::canonical_signs();
    const std::vector<float> gate = gate_vector();
    // Mode C reads a combined record; the other modes must find an identity
    // there, so the first combine does not fold in a stale accumulator.
    const State seeded =
        mode == 'c' ? uniform_expected(splits, reps) : identity();

    const std::size_t workspace_elems =
        static_cast<std::size_t>(kRecordElems) * kMaxSplits;
    const std::size_t log_elems = static_cast<std::size_t>(kMaxSplits) * kMaxReps;

    camodel::Session session;
    camodel::GuardedDeviceBuffer d_tiles(tiles.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_workspace(workspace_elems * sizeof(float));
    camodel::GuardedDeviceBuffer d_lock(32);
    camodel::GuardedDeviceBuffer d_ticket(32);
    camodel::GuardedDeviceBuffer d_ticket_log(log_elems * sizeof(std::uint32_t));
    camodel::GuardedDeviceBuffer d_spins(log_elems * sizeof(std::uint32_t));
    camodel::GuardedDeviceBuffer d_atomic_m(32);
    camodel::GuardedDeviceBuffer d_sign(signs.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_gate(gate.size() * sizeof(float));
    camodel::GuardedDeviceBuffer d_context(kChannels * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer d_metrics(
        static_cast<std::size_t>(kMaxSplits) * kMetricElems * sizeof(std::int32_t));

    std::memcpy(d_tiles.host_payload(), tiles.data(), tiles.size() * sizeof(float));
    std::memcpy(d_sign.host_payload(), signs.data(), signs.size() * sizeof(float));
    std::memcpy(d_gate.host_payload(), gate.data(), gate.size() * sizeof(float));

    // Every workspace slice starts at the combine identity, including the
    // slices Mode A will overwrite: a slice that stays at the identity is
    // then distinguishable from one written with a zero state.
    //
    // Mode C is the exception. It reads slice 0 and publishes its two
    // divides' operands into slices 1..3 for the Inf/NaN diagnosis below, so
    // those three start as poison and an unwritten word there is
    // distinguishable from a written one.
    {
        auto* ws = static_cast<float*>(d_workspace.host_payload());
        for (std::uint32_t k = 0; k < kMaxSplits; ++k) {
            float* rec = ws + static_cast<std::size_t>(k) * kRecordElems;
            if (mode == 'c' && k >= 1 && k <= 3) {
                auto* words = reinterpret_cast<std::uint32_t*>(rec);
                for (std::uint32_t i = 0; i < kRecordElems; ++i) words[i] = kPoison;
                continue;
            }
            for (std::uint32_t i = 0; i < kStateLanes; ++i) {
                rec[kMOffset + i] = seeded.m;
                rec[kLOffset + i] = seeded.l;
            }
            for (std::uint32_t c = 0; c < kChannels; ++c)
                rec[kAccOffset + c] = seeded.acc[c];
        }
    }
    {
        auto* lock = static_cast<std::uint32_t*>(d_lock.host_payload());
        for (std::size_t i = 0; i < d_lock.payload_bytes / 4; ++i) lock[i] = 0u;
        auto* ticket = static_cast<std::uint32_t*>(d_ticket.host_payload());
        for (std::size_t i = 0; i < d_ticket.payload_bytes / 4; ++i) ticket[i] = 0u;
        auto* atomic_m = static_cast<float*>(d_atomic_m.host_payload());
        for (std::size_t i = 0; i < d_atomic_m.payload_bytes / 4; ++i)
            atomic_m[i] = kMNegative;
    }
    fill_poison_u32(d_ticket_log, kPoisonU32);
    fill_poison_u32(d_spins, kPoisonU32);
    fill_poison_u32(d_metrics, kPoisonU32);
    {
        auto* words = static_cast<std::uint16_t*>(d_context.host_payload());
        for (std::size_t i = 0; i < d_context.payload_bytes / 2; ++i) words[i] = kPoisonB16;
    }

    d_tiles.upload();
    d_workspace.upload();
    d_lock.upload();
    d_ticket.upload();
    d_ticket_log.upload();
    d_spins.upload();
    d_atomic_m.upload();
    d_sign.upload();
    d_gate.upload();
    d_context.upload();
    d_metrics.upload();

    auto* p_tiles = static_cast<float*>(d_tiles.payload());
    auto* p_workspace = static_cast<float*>(d_workspace.payload());
    auto* p_lock = static_cast<std::uint32_t*>(d_lock.payload());
    auto* p_ticket = static_cast<std::uint32_t*>(d_ticket.payload());
    auto* p_ticket_log = static_cast<std::uint32_t*>(d_ticket_log.payload());
    auto* p_spins = static_cast<std::uint32_t*>(d_spins.payload());
    auto* p_atomic_m = static_cast<float*>(d_atomic_m.payload());
    auto* p_sign = static_cast<float*>(d_sign.payload());
    auto* p_gate = static_cast<float*>(d_gate.payload());
    auto* p_context = static_cast<std::uint16_t*>(d_context.payload());
    auto* p_metrics = static_cast<std::int32_t*>(d_metrics.payload());

    auto launcher = mode == 'a'   ? launch_camodel_l2_atomic_mode_a
                    : mode == 'b' ? launch_camodel_l2_atomic_mode_b
                                  : launch_camodel_l2_atomic_mode_c;
    launcher(session.stream, splits, p_tiles, p_workspace, p_lock, p_ticket, p_ticket_log,
             p_spins, p_atomic_m, p_sign, p_gate, p_context, p_metrics, reps);
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    d_tiles.download();
    d_workspace.download();
    d_lock.download();
    d_ticket.download();
    d_ticket_log.download();
    d_spins.download();
    d_atomic_m.download();
    d_sign.download();
    d_gate.download();
    d_context.download();
    d_metrics.download();
    d_tiles.assert_guards();
    d_workspace.assert_guards();
    d_lock.assert_guards();
    d_ticket.assert_guards();
    d_ticket_log.assert_guards();
    d_spins.assert_guards();
    d_atomic_m.assert_guards();
    d_sign.assert_guards();
    d_gate.assert_guards();
    d_context.assert_guards();
    d_metrics.assert_guards();

    const auto* ws = static_cast<const float*>(d_workspace.host_payload());
    const auto* log = static_cast<const std::uint32_t*>(d_ticket_log.host_payload());
    const auto* spins = static_cast<const std::uint32_t*>(d_spins.host_payload());
    const auto* metrics = static_cast<const std::int32_t*>(d_metrics.host_payload());

    // ----------------------------------------------------- the read-only set
    {
        std::size_t changed = 0;
        const auto* back = static_cast<const float*>(d_tiles.host_payload());
        for (std::size_t i = 0; i < tiles.size(); ++i)
            if (bits_of(back[i]) != bits_of(tiles[i])) ++changed;
        expect(changed == 0, "the seeded tile states were not modified");
    }

    // --------------------------------------------------------------- Mode A
    if (mode == 'a') {
        std::size_t wrong = 0, poisoned = 0;
        int first_split = -1;
        for (std::uint32_t k = 0; k < splits; ++k) {
            const State want = tile_state(fixture, k);
            const float* rec = ws + static_cast<std::size_t>(k) * kRecordElems;
            for (std::uint32_t i = 0; i < kStateLanes; ++i) {
                if (bits_of(rec[kMOffset + i]) != bits_of(want.m)) ++wrong;
                if (bits_of(rec[kLOffset + i]) != bits_of(want.l)) ++wrong;
            }
            for (std::uint32_t c = 0; c < kChannels; ++c) {
                const float got = rec[kAccOffset + c];
                if (bits_of(got) == kPoison) ++poisoned;
                if (bits_of(got) != bits_of(want.acc[c])) {
                    if (first_split < 0) {
                        first_split = static_cast<int>(k);
                        std::fprintf(stderr,
                                     "  first divergence split %u channel %u: device "
                                     "%.1f oracle %.1f\n",
                                     k, c, static_cast<double>(got),
                                     static_cast<double>(want.acc[c]));
                    }
                    ++wrong;
                }
            }
        }
        expect(wrong == 0, "every split's slice holds its own state, bit-exactly");
        expect(poisoned == 0, "no owned word was left unwritten");

        // The slices beyond the launch must still hold the identity: a core
        // that addressed the wrong slice is otherwise invisible, because the
        // comparison above only reads the slices it expects to be written.
        std::size_t beyond = 0;
        for (std::uint32_t k = splits; k < kMaxSplits; ++k) {
            const float* rec = ws + static_cast<std::size_t>(k) * kRecordElems;
            for (std::uint32_t c = 0; c < kChannels; ++c)
                if (bits_of(rec[kAccOffset + c]) != bits_of(seeded.acc[c])) ++beyond;
        }
        expect(beyond == 0, "no slice beyond the launch's split count was written");
        std::printf("  Mode A: %u slice(s) exact, %zu beyond-range write(s)\n", splits,
                    beyond);
    }

    // --------------------------------------------------------------- Mode B
    if (mode == 'b') {
        const std::uint32_t acquisitions = splits * reps;

        // ---- gate 1: the ticket log is a permutation --------------------
        std::vector<std::uint32_t> arrival(acquisitions, 0xFFFFFFFFu);
        std::vector<std::uint32_t> seen(acquisitions, 0);
        std::size_t out_of_range = 0, unwritten = 0;
        for (std::uint32_t k = 0; k < splits; ++k) {
            for (std::uint32_t r = 0; r < reps; ++r) {
                const std::uint32_t slot = k * reps + r;
                const std::uint32_t ticket = log[slot];
                if (ticket == kPoisonU32) {
                    ++unwritten;
                    continue;
                }
                if (ticket >= acquisitions) {
                    ++out_of_range;
                    continue;
                }
                ++seen[ticket];
                arrival[ticket] = k;
            }
        }
        expect(unwritten == 0, "every acquisition logged a ticket");
        expect(out_of_range == 0, "every ticket is inside the acquisition count");
        const std::size_t duplicates =
            static_cast<std::size_t>(std::count_if(seen.begin(), seen.end(),
                                                   [](std::uint32_t n) { return n > 1; }));
        const std::size_t missing = static_cast<std::size_t>(
            std::count(seen.begin(), seen.end(), 0u));
        expect(duplicates == 0,
               "no ticket was issued twice: the critical section is mutually exclusive");
        expect(missing == 0, "every ticket was issued");
        const auto* ticket_end =
            static_cast<const std::uint32_t*>(d_ticket.host_payload());
        expect(ticket_end[0] == acquisitions,
               "the ticket counter ends at the acquisition count");
        const auto* lock_end = static_cast<const std::uint32_t*>(d_lock.host_payload());
        expect(lock_end[0] == 0u, "the lock was released by the last holder");

        // ---- gate 2: the combined accumulator ---------------------------
        const float* rec = ws;
        if (uniform) {
            const State want = uniform_expected(splits, reps);
            std::size_t wrong = 0;
            for (std::uint32_t c = 0; c < kChannels; ++c)
                if (bits_of(rec[kAccOffset + c]) != bits_of(want.acc[c])) ++wrong;
            bool l_exact = true;
            for (std::uint32_t i = 0; i < kStateLanes; ++i)
                l_exact = l_exact && bits_of(rec[kLOffset + i]) == bits_of(want.l);
            expect(wrong == 0,
                   "the combined accumulator is bit-exact: no update was lost");
            expect(l_exact, "the combined denominator is bit-exact");
            if (wrong != 0)
                std::fprintf(stderr, "  %s\n",
                             uniform_shortfall_note(rec[kAccOffset], splits, reps).c_str());
            std::printf("  Mode B uniform: acc[0] %.1f (want %.1f), l %.1f (want %.1f)\n",
                        static_cast<double>(rec[kAccOffset]),
                        static_cast<double>(want.acc[0]),
                        static_cast<double>(rec[kLOffset]),
                        static_cast<double>(want.l));
        } else {
            // Replay the device's own interleaving.
            std::vector<std::uint32_t> order;
            for (std::uint32_t t = 0; t < acquisitions; ++t)
                if (arrival[t] != 0xFFFFFFFFu) order.push_back(arrival[t]);
            const State want = combine_sequence(fixture, order);
            double worst = 0.0;
            std::uint32_t worst_channel = 0;
            for (std::uint32_t c = 0; c < kChannels; ++c) {
                const double denom =
                    std::max(1.0, std::abs(static_cast<double>(want.acc[c])));
                const double rel =
                    std::abs(static_cast<double>(rec[kAccOffset + c] - want.acc[c])) / denom;
                if (rel > worst) {
                    worst = rel;
                    worst_channel = c;
                }
            }
            // Each combine step costs about one Exp (<= 1 ULP) and three
            // multiplies, so K.reps steps carry a few ULP; 1e-5 relative is
            // two orders of magnitude of headroom over that and still three
            // orders tighter than a wrong rescale would land.
            expect(worst < 1e-5,
                   "the combined accumulator matches the replayed interleaving");
            std::printf(
                "  Mode B staggered: worst relative difference %.3g at channel %u "
                "over %u acquisition(s)\n",
                worst, worst_channel, acquisitions);
        }

        // ---- gate 3: the atomic scalar maximum --------------------------
        {
            float want = kMNegative;
            for (std::uint32_t k = 0; k < splits; ++k)
                want = std::max(want, tile_state(fixture, k).m);
            const auto* got = static_cast<const float*>(d_atomic_m.host_payload());
            expect(bits_of(got[0]) == bits_of(want),
                   "the atomically reduced maximum equals the maximum over the splits");
            if (bits_of(got[0]) != bits_of(want))
                std::fprintf(stderr,
                             "  atomic max: device %.6g (0x%08x), oracle %.6g (0x%08x) "
                             "-- negative-float ordering is the first thing to suspect\n",
                             static_cast<double>(got[0]), bits_of(got[0]),
                             static_cast<double>(want), bits_of(want));
            // The combine's own carried maximum must agree with it.
            expect(bits_of(rec[kMOffset]) == bits_of(want),
                   "the combine's carried maximum agrees with the atomic reduction");
            std::printf("  Mode B maximum: %.6g, atomically reduced and carried\n",
                        static_cast<double>(want));
        }

        // ---- contention evidence ---------------------------------------
        std::uint64_t spin_total = 0;
        std::uint32_t spin_max = 0;
        for (std::uint32_t slot = 0; slot < acquisitions; ++slot) {
            if (spins[slot] == kPoisonU32) continue;
            spin_total += spins[slot];
            spin_max = std::max(spin_max, spins[slot]);
        }
        std::printf(
            "  Mode B contention: %llu failed exchange(s) over %u acquisition(s), "
            "worst %u, mean %.2f\n",
            static_cast<unsigned long long>(spin_total), acquisitions, spin_max,
            acquisitions ? static_cast<double>(spin_total) / acquisitions : 0.0);
        if (splits == 1)
            expect(spin_total == 0,
                   "an uncontended lock never spins, so K=1 is the contention baseline");
    }

    // --------------------------------------------------------------- Mode C
    if (mode == 'c') {
        const std::vector<float> want = egress_reference(seeded);
        const auto* got = static_cast<const std::uint16_t*>(d_context.host_payload());
        std::size_t poisoned = 0;
        double worst = 0.0;
        std::uint32_t worst_channel = 0;
        for (std::uint32_t c = 0; c < kChannels; ++c) {
            if (got[c] == kPoisonB16) {
                ++poisoned;
                continue;
            }
            const float device = bf16_to_float(got[c]);
            const float oracle = round_bf16(want[c]);
            const double denom = std::max(1.0, std::abs(static_cast<double>(oracle)));
            const double rel =
                std::abs(static_cast<double>(device) - static_cast<double>(oracle)) / denom;
            if (rel > worst) {
                worst = rel;
                worst_channel = c;
            }
        }
        expect(poisoned == 0, "every context channel was written");
        // One Exp in the gate, one 0-ULP Div, one bf16 round: about 2 ULP of
        // FP32 before a rounding step whose own quantum is 2^-8 relative. The
        // bound is therefore set by bf16, not by the rotation, which is exact.
        expect(worst < 8e-3, "the tail egress matches the host reference in bf16");
        std::printf("  Mode C: worst relative difference %.3g at channel %u\n", worst,
                    worst_channel);

        // The tail split reads the combined record; it must not write it.
        std::size_t changed = 0;
        for (std::uint32_t c = 0; c < kChannels; ++c)
            if (bits_of(ws[kAccOffset + c]) != bits_of(seeded.acc[c])) ++changed;
        expect(changed == 0, "the tail split did not modify the combined record");

        // The two divides' operands, published into slices 1..3 after the
        // measured region. The model raises `vec_err_idata_inf_nan_t0` on the
        // 0-ULP divide's Inf/NaN classification path in this mode, and the
        // only way to tell an operand that IS non-finite from a classification
        // constant the divide materialises for itself is to look at the
        // operands. This reports rather than asserts: a non-finite operand
        // here would be a defect in the probe, and its absence localises the
        // flag to the divide's own expansion.
        struct Operand { const char* name; std::uint32_t slice; };
        for (const Operand& operand : {Operand{"l_broadcast (divide 1 denominator)", 1},
                                       Operand{"rotated (divide 2 numerator)", 2},
                                       Operand{"gate_tmp (divide 2 denominator)", 3}}) {
            const float* slice =
                ws + static_cast<std::size_t>(operand.slice) * kRecordElems + kAccOffset;
            std::size_t nonfinite = 0, poisoned = 0;
            float lo = slice[0], hi = slice[0];
            for (std::uint32_t c = 0; c < kChannels; ++c) {
                if (bits_of(slice[c]) == kPoison) {
                    ++poisoned;
                    continue;
                }
                if (!std::isfinite(slice[c])) ++nonfinite;
                lo = std::min(lo, slice[c]);
                hi = std::max(hi, slice[c]);
            }
            std::printf("    %-38s non-finite %zu, unwritten %zu, range "
                        "[%.6g, %.6g]\n",
                        operand.name, nonfinite, poisoned, static_cast<double>(lo),
                        static_cast<double>(hi));
            expect(nonfinite == 0, "a divide operand is finite in every channel");
        }

        // Exactly one core did the work.
        std::uint32_t worked = 0;
        for (std::uint32_t k = 0; k < kMaxSplits; ++k) {
            const MetricRead block = read_metric(metrics + k * kMetricElems, kMetricBlock);
            if (block.written) ++worked;
        }
        expect(worked == 1, "exactly one core ran the tail egress");
    }

    // ------------------------------------------------------------- metrics
    std::printf("\n  per-split cycle counts (system counter, pipeline drained at every "
                "boundary)\n");
    const char* const phase_names[3][5] = {
        {"store", "", "", "", ""},
        {"", "atomic max", "acquire", "critical", "release"},
        {"normalise", "rotate", "gate", "store", ""},
    };
    const std::uint32_t mode_index = mode == 'a' ? 0u : mode == 'b' ? 1u : 2u;
    for (std::uint32_t k = 0; k < splits; ++k) {
        const std::int32_t* slice = metrics + k * kMetricElems;
        const MetricRead block = read_metric(slice, kMetricBlock);
        if (!block.written) {
            std::printf("    split %u: no metrics (core did not run this mode)\n", k);
            continue;
        }
        expect(block.consistent, "a metric slot's eight lanes agree");
        expect(block.value == static_cast<std::int32_t>(k),
               "a metric slice was written by the split it belongs to");
        const MetricRead region = read_metric(slice, kMetricRegion);
        std::printf("    split %u: region %7d", k, region.value);
        for (std::uint32_t phase = 0; phase < 5; ++phase) {
            if (phase_names[mode_index][phase][0] == '\0') continue;
            const MetricRead value = read_metric(slice, kMetricPhase0 + phase);
            std::printf("  %s %6d", phase_names[mode_index][phase], value.value);
        }
        std::printf("\n      per-rep:");
        for (std::uint32_t r = 0; r < reps; ++r)
            std::printf(" %d", read_metric(slice, kMetricRepBase + r).value);
        std::printf("\n");
    }

    if (failures == 0) {
        std::printf("\nl2 atomic Split-K probe (mode %c, K=%u): PASS\n", mode, splits);
        return 0;
    }
    std::fprintf(stderr, "\nl2 atomic Split-K probe (mode %c, K=%u): %d FAILURES\n", mode,
                 splits, failures);
    return 1;
}
