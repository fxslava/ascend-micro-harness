// CAModel runner and single-tile canary protocol for the paged decode
// attention operator. Ascend950PR_9599, CANN 9.2.0-beta.2.
//
//   --strips N        strips this launch processes, 1..8 (default 8)
//   --strip-begin T   first strip of that window, 0..7 (default 0)
//   --block B         run only grid block B, 0..31 (default: all 32)
//   --canary T H      shorthand for the isolation run: one strip T, one head
//                     H, one temporal block, and every Global Memory region
//                     the active tile does not own left as a canary
//   --no-fixture-check  skip the host fixture validation (it is ~1 s)
//
// Two regimes, and each catches what the other cannot
// ---------------------------------------------------
// The full run (`--strips 8`) wraps every two-slot ring four times, so every
// buffer-reuse back edge carries real contention and the per-strip interval is
// measurable. It cannot tell you whether the kernel read outside its own tile,
// because every tile's operands are present and plausible.
//
// The canary run (`--canary T H`) answers exactly that. Every Global Memory
// buffer is filled with 0x7FC0BEEF and the fixture is written into ONE
// (page, head, strip) slice; the kernel's own block table decides which page.
// A read outside that slice returns the canary, and 0x7FC0BEEF is a quiet NaN,
// so a stray scale read poisons the whole output rather than perturbing it. A
// write outside the active block's output slices leaves a canary word behind
// that the final sweep finds. What it cannot do is contend a ring: with one
// strip no ring ever wraps, which is why both regimes are run.
//
// The guard bands are a SECOND, different pattern. `camodel::GuardedDeviceBuffer`
// surrounds every payload with 64 bytes of 0xDEADBEEF, which catches a burst
// that runs off the end of a buffer; 0x7FC0BEEF catches a burst that lands in
// the wrong place inside one. Neither is 0x00, which a half-issued burst also
// leaves behind.
//
// What is checked bit-exactly, and what in ULP
// --------------------------------------------
//   out_qrot    bit-exact. 64 bf16 Mmads into an FP32 accumulator, with a
//               fixture whose rotated entries are exact in bf16.
//   out_scores  bit-exact, every panel of every strip. One Mmad over k = 256
//               of exact small integers against Q_rot, then a power-of-two
//               column scale. This is the check that locates a fault anywhere
//               on the int4 -> UB -> L1(Nz) -> L0B -> Mmad -> Fixpipe path.
//   m, l, acc, context  in ULP. Every one of them passes through `Exp`, which
//               is documented to permit 1 ULP per call, and `acc` is a
//               cancelling reduction, so the budget is stated against the
//               largest intermediate per guidelines section 5, not the final
//               value.

#include "camodel_common.hpp"
#include "../common/paged_decode_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_camodel_paged_decode_attention(
    void*, std::uint16_t*, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*, float*, std::uint32_t*, float*, float*, std::uint16_t*, float*,
    float*, float*, std::uint16_t*, float*, std::uint32_t, std::uint32_t,
    std::uint32_t);

namespace {

using namespace paged_decode;
using ascend_micro::sim::float_to_bfloat16_rne;

// The in-bounds canary. A quiet NaN when read as FP32, so a stray read of the
// scale plane propagates rather than perturbs.
constexpr std::uint32_t kTileCanary = 0x7FC0BEEFu;
constexpr std::uint32_t kStateLanes = 8;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t bits_of(float value)
{
    std::uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

// ULP distance between two finite FP32 values, monotone across zero.
std::uint64_t ulp_distance(float a, float b)
{
    auto ordered = [](float value) -> std::int64_t {
        const std::uint32_t raw = bits_of(value);
        return (raw & 0x80000000u) ? -static_cast<std::int64_t>(raw & 0x7fffffffu)
                                   : static_cast<std::int64_t>(raw);
    };
    if (!std::isfinite(a) || !std::isfinite(b)) return UINT64_MAX;
    const std::int64_t da = ordered(a);
    const std::int64_t db = ordered(b);
    return static_cast<std::uint64_t>(da > db ? da - db : db - da);
}

void fill_payload(camodel::GuardedDeviceBuffer& buffer, std::uint32_t word)
{
    auto* payload = static_cast<std::uint32_t*>(buffer.host_payload());
    for (std::size_t i = 0; i < buffer.payload_bytes / 4; ++i) payload[i] = word;
}

// Every output word of block `block` must have been written; every word of an
// inactive block must still hold the canary.
struct CanarySweep {
    std::size_t checked = 0;
    std::size_t mutated = 0;
    std::size_t intact = 0;
};

CanarySweep sweep_inactive(const std::uint32_t* words, std::size_t words_per_block,
                           std::uint32_t active_block)
{
    CanarySweep report;
    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(kGrid); ++block) {
        if (block == active_block) continue;
        for (std::size_t i = 0; i < words_per_block; ++i) {
            ++report.checked;
            if (words[block * words_per_block + i] == kTileCanary)
                ++report.intact;
            else
                ++report.mutated;
        }
    }
    return report;
}

// ------------------------------------------------------------------- options
struct Options {
    std::uint32_t strips = static_cast<std::uint32_t>(kStrips);
    std::uint32_t strip_begin = 0;
    std::uint32_t block = static_cast<std::uint32_t>(kGrid);  // kGrid == every block
    bool canary = false;
    bool fixture_check = true;
};

std::uint32_t parse_u32(std::string_view text, const char* what)
{
    std::uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::invalid_argument(what);
    return value;
}

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        auto next = [&](const char* what) {
            if (i + 1 >= argc) throw std::invalid_argument(what);
            return std::string_view{argv[++i]};
        };
        if (flag == "--strips") {
            options.strips = parse_u32(next("--strips"), "--strips");
        } else if (flag == "--strip-begin") {
            options.strip_begin = parse_u32(next("--strip-begin"), "--strip-begin");
        } else if (flag == "--block") {
            options.block = parse_u32(next("--block"), "--block");
        } else if (flag == "--no-fixture-check") {
            options.fixture_check = false;
        } else if (flag == "--canary") {
            const auto strip = parse_u32(next("--canary strip"), "--canary strip");
            const auto head = parse_u32(next("--canary head"), "--canary head");
            if (strip >= static_cast<std::uint32_t>(kStrips) ||
                head >= static_cast<std::uint32_t>(kKvHeads))
                throw std::invalid_argument("--canary T H out of range");
            options.canary = true;
            options.strips = 1;
            options.strip_begin = strip;
            // One temporal block, the first, so the block table's non-identity
            // mapping still decides which physical page is read.
            options.block = head * static_cast<std::uint32_t>(kTemporalBlocks);
        } else {
            throw std::invalid_argument("unknown option");
        }
    }
    if (options.strips < 1 || options.strips > static_cast<std::uint32_t>(kStrips) ||
        options.strip_begin + options.strips > static_cast<std::uint32_t>(kStrips))
        throw std::invalid_argument("strips window out of range");
    if (options.block > static_cast<std::uint32_t>(kGrid))
        throw std::invalid_argument("--block out of range");
    return options;
}

}  // namespace

int main(int argc, char** argv)
{
    Options options;
    try {
        options = parse(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "usage error: %s\n", error.what());
        std::fprintf(stderr,
                     "usage: %s [--strips N] [--strip-begin T] [--block B]\n"
                     "          [--canary T H] [--no-fixture-check]\n",
                     argv[0]);
        return 2;
    }

    std::printf("paged decode attention on CAModel: strips %u from %u, ",
                options.strips, options.strip_begin);
    if (options.block < static_cast<std::uint32_t>(kGrid))
        std::printf("block %u (head %u, temporal %u)%s\n", options.block,
                    options.block / kTemporalBlocks, options.block % kTemporalBlocks,
                    options.canary ? ", canary isolation" : "");
    else
        std::printf("all %d blocks\n", kGrid);

    try {
        if (options.fixture_check) {
            ValidateRotation();
            for (int head = 0; head < kKvHeads; ++head) (void)ValidateFixture(head);
            std::printf("  fixture: rotation and all four heads validated on the host\n");
        }

        // ------------------------------------------------------- allocation
        camodel::Session session;

        const std::size_t qrot_words = static_cast<std::size_t>(kTokens) * kHeadSize / 2;
        const std::size_t score_words = static_cast<std::size_t>(kStrips) * kScores;
        const std::size_t state_words = 2u * kTokens * kStateLanes;
        const std::size_t acc_words = static_cast<std::size_t>(kTokens) * kHeadSize;
        const std::size_t context_words =
            static_cast<std::size_t>(kTokens) * kHeadSize / 2;

        camodel::GuardedDeviceBuffer q(static_cast<std::size_t>(kKvHeads) * kTokens *
                                       kHeadSize * 2);
        camodel::GuardedDeviceBuffer pi_table(static_cast<std::size_t>(kTableElems) * 2);
        camodel::GuardedDeviceBuffer key_cache(kKeyCacheBytes);
        camodel::GuardedDeviceBuffer value_cache(kValueCacheBytes);
        camodel::GuardedDeviceBuffer scale_cache(kScaleCacheFloats * 4);
        camodel::GuardedDeviceBuffer block_table(static_cast<std::size_t>(kTemporalBlocks) *
                                                 4);
        camodel::GuardedDeviceBuffer gate(static_cast<std::size_t>(kKvHeads) * kTokens *
                                          kHeadSize * 4);
        camodel::GuardedDeviceBuffer gather_offsets(
            static_cast<std::size_t>(kEgressGatherCount) * kHeadSize * 4);
        camodel::GuardedDeviceBuffer gather_signs(
            static_cast<std::size_t>(kEgressGatherCount) * kHeadSize * 4);
        camodel::GuardedDeviceBuffer pi_sign_buffer(static_cast<std::size_t>(kHeadSize) * 4);

        camodel::GuardedDeviceBuffer out_qrot(static_cast<std::size_t>(kGrid) *
                                              qrot_words * 4);
        camodel::GuardedDeviceBuffer out_scores(static_cast<std::size_t>(kGrid) *
                                                score_words * 4);
        camodel::GuardedDeviceBuffer out_state(static_cast<std::size_t>(kGrid) *
                                               state_words * 4);
        camodel::GuardedDeviceBuffer out_acc(static_cast<std::size_t>(kGrid) * acc_words *
                                             4);
        camodel::GuardedDeviceBuffer out_context(static_cast<std::size_t>(kGrid) *
                                                 context_words * 4);
        camodel::GuardedDeviceBuffer out_rotated(static_cast<std::size_t>(kGrid) *
                                                 acc_words * 4);

        // -------------------------------------------------------- the inputs
        // Caches start as canary everywhere. In the full run every slice the
        // grid reads is then overwritten with the fixture; in the canary run
        // only the active tile's slice is, so any other read returns 0x7FC0BEEF.
        fill_payload(key_cache, kTileCanary);
        fill_payload(value_cache, kTileCanary);
        fill_payload(scale_cache, kTileCanary);

        auto* key_bytes = static_cast<std::uint8_t*>(key_cache.host_payload());
        auto* value_bytes = static_cast<std::uint8_t*>(value_cache.host_payload());
        auto* scale_floats = static_cast<float*>(scale_cache.host_payload());

        const bool one_tile = options.block < static_cast<std::uint32_t>(kGrid);
        std::size_t slices_written = 0;
        for (int temporal = 0; temporal < kTemporalBlocks; ++temporal) {
            const int page = BlockTableEntry(temporal);
            for (int head = 0; head < kKvHeads; ++head) {
                const std::uint32_t block =
                    static_cast<std::uint32_t>(head * kTemporalBlocks + temporal);
                if (one_tile && block != options.block) continue;
                for (std::uint32_t i = 0; i < options.strips; ++i) {
                    const int strip = static_cast<int>(options.strip_begin + i);
                    const CacheFill where{page, head, strip};
                    WriteKeyStrip(key_bytes, where, strip);
                    WriteValueStrip(value_bytes, where, strip);
                    WriteScales(scale_floats, where, strip);
                    ++slices_written;
                }
            }
        }

        // Q, the rotation table, the gate, the block table and the egress
        // tables are read whole by whichever block runs, so they are always
        // real. Head separation comes from the cache slice and the gate.
        auto* q_half = static_cast<std::uint16_t*>(q.host_payload());
        for (int head = 0; head < kKvHeads; ++head)
            for (int row = 0; row < kTokens; ++row)
                for (int channel = 0; channel < kHeadSize; ++channel)
                    q_half[(static_cast<std::size_t>(head) * kTokens + row) * kHeadSize +
                           channel] = float_to_bfloat16_rne(QueryEntry(row, channel));

        const RotationTable table = BuildRotationTable();
        std::memcpy(pi_table.host_payload(), table.row_major.data(),
                    table.row_major.size() * 2);

        auto* gate_floats = static_cast<float*>(gate.host_payload());
        for (int head = 0; head < kKvHeads; ++head)
            for (int row = 0; row < kTokens; ++row)
                for (int channel = 0; channel < kHeadSize; ++channel)
                    gate_floats[(static_cast<std::size_t>(head) * kTokens + row) *
                                    kHeadSize +
                                channel] = GateEntry(head, row, channel);

        auto* table_words = static_cast<std::uint32_t*>(block_table.host_payload());
        for (int temporal = 0; temporal < kTemporalBlocks; ++temporal)
            table_words[temporal] = static_cast<std::uint32_t>(BlockTableEntry(temporal));

        // The three sub-DataBlock egress stages: byte offsets of i ^ stride,
        // and the sign (i & stride) ? -1 : +1.
        auto* offsets = static_cast<std::uint32_t*>(gather_offsets.host_payload());
        auto* signs = static_cast<float*>(gather_signs.host_payload());
        for (int stage = 0; stage < kEgressGatherCount; ++stage) {
            const int stride = kEgressGatherStride[stage];
            for (int i = 0; i < kHeadSize; ++i) {
                offsets[static_cast<std::size_t>(stage) * kHeadSize + i] =
                    static_cast<std::uint32_t>((i ^ stride) * 4);
                signs[static_cast<std::size_t>(stage) * kHeadSize + i] =
                    (i & stride) ? -1.0f : 1.0f;
            }
        }
        auto* pi_signs = static_cast<float*>(pi_sign_buffer.host_payload());
        for (int i = 0; i < kHeadSize; ++i)
            pi_signs[i] = static_cast<float>(PiSign(i));

        // Outputs start as canary so an unwritten word is distinguishable from
        // a zero the kernel might legitimately produce.
        fill_payload(out_qrot, kTileCanary);
        fill_payload(out_scores, kTileCanary);
        fill_payload(out_state, kTileCanary);
        fill_payload(out_acc, kTileCanary);
        fill_payload(out_context, kTileCanary);
        fill_payload(out_rotated, kTileCanary);

        for (auto* buffer : {&q, &pi_table, &key_cache, &value_cache, &scale_cache,
                             &block_table, &gate, &gather_offsets, &gather_signs,
                             &pi_sign_buffer, &out_qrot, &out_scores, &out_state,
                             &out_acc, &out_context, &out_rotated})
            buffer->upload();

        std::printf("  uploaded: %zu cache slice(s) real, the rest canary 0x%08X\n",
                    slices_written, kTileCanary);

        // Snapshot the read-only inputs BEFORE the launch. The download below
        // overwrites the same host buffers, so a comparison taken afterwards
        // against those buffers would be comparing the device's bytes with
        // themselves and would pass unconditionally.
        std::vector<std::uint8_t> key_snapshot(kKeyCacheBytes);
        std::vector<std::uint8_t> value_snapshot(kValueCacheBytes);
        std::vector<float> scale_snapshot(kScaleCacheFloats);
        std::memcpy(key_snapshot.data(), key_bytes, kKeyCacheBytes);
        std::memcpy(value_snapshot.data(), value_bytes, kValueCacheBytes);
        std::memcpy(scale_snapshot.data(), scale_floats, kScaleCacheFloats * 4);

        // --------------------------------------------------------- the launch
        launch_camodel_paged_decode_attention(
            session.stream, static_cast<std::uint16_t*>(q.payload()),
            static_cast<std::uint16_t*>(pi_table.payload()),
            static_cast<std::int8_t*>(key_cache.payload()),
            static_cast<std::int8_t*>(value_cache.payload()),
            static_cast<float*>(scale_cache.payload()),
            static_cast<std::uint32_t*>(block_table.payload()),
            static_cast<float*>(gate.payload()),
            static_cast<std::uint32_t*>(gather_offsets.payload()),
            static_cast<float*>(gather_signs.payload()),
            static_cast<float*>(pi_sign_buffer.payload()),
            static_cast<std::uint16_t*>(out_qrot.payload()),
            static_cast<float*>(out_scores.payload()),
            static_cast<float*>(out_state.payload()),
            static_cast<float*>(out_acc.payload()),
            static_cast<std::uint16_t*>(out_context.payload()),
            static_cast<float*>(out_rotated.payload()), options.strips,
            options.strip_begin, options.block);
        camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

        for (auto* buffer : {&out_qrot, &out_scores, &out_state, &out_acc, &out_context,
                             &out_rotated, &key_cache, &value_cache, &scale_cache, &q,
                             &pi_table, &gate, &block_table, &gather_offsets,
                             &gather_signs, &pi_sign_buffer})
            buffer->download();

        // -------------------------------------------------- guard bands first
        for (auto* buffer : {&out_qrot, &out_scores, &out_state, &out_acc, &out_context,
                             &key_cache, &value_cache, &scale_cache, &q, &pi_table,
                             &gate, &block_table, &gather_offsets, &gather_signs,
                             &pi_sign_buffer})
            buffer->assert_guards();
        std::printf("  guard bands: all 15 buffers intact, head and tail\n");

        // ------------------------------------------- inputs are not written to
        // The kernel only reads the three caches. Every byte must come back as
        // it went in -- both the fixture slices and the canary everywhere else.
        // This is what rules out a stray MTE3 landing in an input buffer, which
        // the output-side canary sweep cannot see.
        {
            std::size_t key_touched = 0;
            std::size_t value_touched = 0;
            std::size_t scale_touched = 0;
            const auto* key_now =
                static_cast<const std::uint8_t*>(key_cache.host_payload());
            const auto* value_now =
                static_cast<const std::uint8_t*>(value_cache.host_payload());
            const auto* scale_now = static_cast<const float*>(scale_cache.host_payload());
            for (std::size_t i = 0; i < kKeyCacheBytes; ++i)
                if (key_now[i] != key_snapshot[i]) ++key_touched;
            for (std::size_t i = 0; i < kValueCacheBytes; ++i)
                if (value_now[i] != value_snapshot[i]) ++value_touched;
            for (std::size_t i = 0; i < kScaleCacheFloats; ++i)
                if (bits_of(scale_now[i]) != bits_of(scale_snapshot[i])) ++scale_touched;
            expect(key_touched == 0 && value_touched == 0 && scale_touched == 0,
                   "the kernel wrote nothing to any input cache");
            std::printf("  inputs: key %zu, value %zu, scale %zu word(s) changed\n",
                        key_touched, value_touched, scale_touched);
        }

        // --------------------------------------------------- active-tile checks
        const std::uint32_t first_block = one_tile ? options.block : 0u;
        const std::uint32_t last_block =
            one_tile ? options.block + 1u : static_cast<std::uint32_t>(kGrid);

        const auto* qrot_words_out =
            static_cast<const std::uint16_t*>(out_qrot.host_payload());
        const auto* score_out = static_cast<const float*>(out_scores.host_payload());
        const auto* state_out = static_cast<const float*>(out_state.host_payload());
        const auto* acc_out = static_cast<const float*>(out_acc.host_payload());
        const auto* context_out =
            static_cast<const std::uint16_t*>(out_context.host_payload());
        const auto* rotated_out = static_cast<const float*>(out_rotated.host_payload());

        std::size_t qrot_bad = 0;
        std::size_t score_bad = 0;
        std::uint64_t worst_state_ulp = 0;
        std::uint64_t worst_acc_ulp = 0;
        double worst_context_abs = 0.0;
        double worst_rotated_abs = 0.0;
        double largest_rotated_magnitude = 0.0;
        double largest_acc_intermediate = 0.0;
        double largest_context_magnitude = 0.0;

        std::vector<int> window;
        for (std::uint32_t i = 0; i < options.strips; ++i)
            window.push_back(static_cast<int>(options.strip_begin + i));

        for (std::uint32_t block = first_block; block < last_block; ++block) {
            const int head = static_cast<int>(block / kTemporalBlocks);
            const GoldenState golden = GoldenDecode(head, window);

            // 1. the 64-Mmad ingress rotation, bit-exactly
            const std::vector<float> rot = GoldenQRot(head);
            for (std::size_t i = 0; i < rot.size(); ++i) {
                const std::uint16_t want = float_to_bfloat16_rne(rot[i]);
                if (qrot_words_out[block * rot.size() + i] != want) ++qrot_bad;
            }

            // 2. every raw score panel of every strip in the window
            for (std::uint32_t i = 0; i < options.strips; ++i) {
                const std::vector<float> raw =
                    GoldenRawScores(head, static_cast<int>(options.strip_begin + i));
                for (int entry = 0; entry < kScores; ++entry) {
                    const float got =
                        score_out[block * score_words + i * kScores + entry];
                    if (bits_of(got) != bits_of(raw[entry])) ++score_bad;
                }
            }

            // 3. the carried state, read out of lane 0 of each replicated block
            for (int query = 0; query < kTokens; ++query) {
                const float m_got =
                    state_out[block * state_words + query * kStateLanes];
                const float l_got =
                    state_out[block * state_words + (kTokens + query) * kStateLanes];
                worst_state_ulp = std::max(worst_state_ulp,
                                           ulp_distance(m_got, golden.m[query]));
                worst_state_ulp = std::max(worst_state_ulp,
                                           ulp_distance(l_got, golden.l[query]));
            }

            // 4. the raw accumulator, still in the cache's rotated basis
            for (std::size_t i = 0; i < acc_words; ++i) {
                const float got = acc_out[block * acc_words + i];
                worst_acc_ulp = std::max(worst_acc_ulp, ulp_distance(got, golden.acc[i]));
                largest_acc_intermediate =
                    std::max(largest_acc_intermediate,
                             static_cast<double>(std::fabs(golden.acc[i])));
            }

            // 5a. the egress rotation on its own, before the gate. Splitting
            //     the two is what makes a failure here actionable: the gate is
            //     elementwise and the rotation is eight strided stages, and
            //     their errors are indistinguishable in the gated output.
            std::vector<float> normalized(kHeadSize);
            for (int query = 0; query < kTokens; ++query) {
                for (int c = 0; c < kHeadSize; ++c)
                    normalized[c] =
                        golden.acc[static_cast<std::size_t>(query) * kHeadSize + c] /
                        golden.l[query];
                const std::vector<float> want_row = EgressButterfly(normalized.data());
                for (int c = 0; c < kHeadSize; ++c) {
                    const float got =
                        rotated_out[block * acc_words +
                                    static_cast<std::size_t>(query) * kHeadSize + c];
                    worst_rotated_abs =
                        std::max(worst_rotated_abs,
                                 static_cast<double>(std::fabs(got - want_row[c])));
                    largest_rotated_magnitude =
                        std::max(largest_rotated_magnitude,
                                 static_cast<double>(std::fabs(want_row[c])));
                }
            }

            // 5b. the fused egress: Acc . Pi gated by sigmoid(gate), in bf16
            for (std::size_t i = 0; i < golden.context.size(); ++i) {
                const float want = bfloat16_to_float(float_to_bfloat16_rne(
                    golden.context[i]));
                const float got =
                    bfloat16_to_float(context_out[block * context_words * 2 + i]);
                worst_context_abs =
                    std::max(worst_context_abs,
                             static_cast<double>(std::fabs(got - want)));
                largest_context_magnitude =
                    std::max(largest_context_magnitude,
                             static_cast<double>(std::fabs(want)));
            }
        }

        // A failing panel costs a six-minute CAModel run to observe, so print
        // enough of it to tell the failure modes apart rather than only that
        // it failed: a transpose, a wrong channel group, a NaN from a canary
        // read and a wrong contraction depth look nothing alike side by side.
        if (score_bad != 0) {
            const int head = static_cast<int>(first_block / kTemporalBlocks);
            const std::vector<float> raw =
                GoldenRawScores(head, static_cast<int>(options.strip_begin));
            const float* got = score_out + first_block * score_words;
            std::fprintf(stderr, "  panel diagnosis, block %u strip %u:\n",
                         first_block, options.strip_begin);
            for (int q = 0; q < 2; ++q) {
                std::fprintf(stderr, "    q%-2d want:", q);
                for (int t = 0; t < 8; ++t)
                    std::fprintf(stderr, " %10.4f", raw[q * kTokens + t]);
                std::fprintf(stderr, "\n    q%-2d got :", q);
                for (int t = 0; t < 8; ++t)
                    std::fprintf(stderr, " %10.4f", got[q * kTokens + t]);
                std::fprintf(stderr, "\n");
            }
            // Is the device panel the transpose of the golden one?
            std::size_t transposed_match = 0;
            std::size_t nan_count = 0;
            std::size_t canary_count = 0;
            for (int q = 0; q < kTokens; ++q)
                for (int t = 0; t < kTokens; ++t) {
                    const float value = got[q * kTokens + t];
                    if (std::isnan(value)) ++nan_count;
                    if (bits_of(value) == kTileCanary) ++canary_count;
                    if (bits_of(value) == bits_of(raw[t * kTokens + q]))
                        ++transposed_match;
                }
            std::fprintf(stderr,
                         "    %zu/256 match the TRANSPOSE, %zu NaN, %zu still canary\n",
                         transposed_match, nan_count, canary_count);
        }

        expect(qrot_bad == 0, "out_qrot is bit-exact against the host rotation");
        expect(score_bad == 0, "every raw score panel is bit-exact");
        std::printf("  rotation: %zu/%zu Q_rot entries wrong\n", qrot_bad,
                    (last_block - first_block) *
                        static_cast<std::size_t>(kTokens) * kHeadSize);
        std::printf("  scores:   %zu/%zu panel entries wrong (bit-exact check)\n",
                    score_bad, (last_block - first_block) *
                                   static_cast<std::size_t>(options.strips) * kScores);

        // The ULP budget is against the largest intermediate, not the final
        // value: acc is a cancelling sum of 16 exp-weighted terms per strip and
        // Exp is documented to permit 1 ULP per call.
        constexpr std::uint64_t kStateUlpBudget = 64;
        constexpr std::uint64_t kAccUlpBudget = 256;
        // The context is a cancelling sum: symmetric int4 values make `acc` a
        // signed reduction, and the egress Hadamard cancels it further, so the
        // final magnitudes are a small fraction of the intermediates. An
        // absolute budget would therefore be meaningless -- it has to be stated
        // against the largest value in the row, which is guidelines section 5's
        // rule for exactly this shape of computation. bf16 carries 8 mantissa
        // bits, so 2^-6 of the largest magnitude leaves two bits of headroom
        // for the Exp (1 ULP per call, documented) and the Div in front of it.
        const double context_budget =
            largest_context_magnitude / 64.0 + 1.0e-7;
        // The egress is eight butterfly stages, a sign diagonal, a scale and a
        // gate, and a structural mistake in any of them looks the same from
        // the error magnitude alone. So when it fails, score the device output
        // against the alternatives rather than spending a six-minute run per
        // guess: whichever variant the device matches names the bug.
        if (worst_context_abs > context_budget) {
            const int head = static_cast<int>(first_block / kTemporalBlocks);
            const GoldenState golden = GoldenDecode(head, window);
            const float* dev = nullptr;
            std::vector<float> device_row(kHeadSize);
            for (int c = 0; c < kHeadSize; ++c)
                device_row[c] = bfloat16_to_float(
                    context_out[first_block * context_words * 2 + c]);
            dev = device_row.data();

            // Rebuild query 0's egress input exactly as the kernel has it.
            std::vector<float> normalized(kHeadSize);
            for (int c = 0; c < kHeadSize; ++c)
                normalized[c] = golden.acc[c] / golden.l[0];

            auto butterflies = [&](const std::vector<int>& strides, int sign_at) {
                std::vector<float> v = normalized;
                std::vector<float> n(kHeadSize);
                for (std::size_t s = 0; s < strides.size(); ++s) {
                    if (static_cast<int>(s) == sign_at)
                        for (int i = 0; i < kHeadSize; ++i)
                            v[i] *= static_cast<float>(PiSign(i));
                    const int st = strides[s];
                    for (int i = 0; i < kHeadSize; ++i)
                        n[i] = v[i ^ st] + ((i & st) ? -1.0f : 1.0f) * v[i];
                    v.swap(n);
                }
                if (sign_at == static_cast<int>(strides.size()))
                    for (int i = 0; i < kHeadSize; ++i)
                        v[i] *= static_cast<float>(PiSign(i));
                return v;
            };
            const std::vector<int> fwd{32, 64, 128, 1, 2, 4, 8, 16};
            const std::vector<int> rev{1, 2, 4, 8, 16, 32, 64, 128};

            struct Variant { const char* name; std::vector<float> value; };
            std::vector<Variant> variants;
            auto gate_of = [&](std::vector<float> y, bool scale, int mode) {
                for (int c = 0; c < kHeadSize; ++c) {
                    if (scale) y[c] *= 1.0f / 16.0f;
                    const float g = GateEntry(head, 0, c);
                    const float sig = 1.0f / (1.0f + std::exp(-g));
                    if (mode == 0) y[c] *= sig;
                    else if (mode == 1) y[c] *= (1.0f + std::exp(-g));
                    else if (mode == 2) y[c] *= 1.0f / (1.0f + std::exp(g));
                }
                return y;
            };
            variants.push_back({"specified (sign before stride 1)",
                                gate_of(butterflies(fwd, 3), true, 0)});
            variants.push_back({"no 1/16 scale",
                                gate_of(butterflies(fwd, 3), false, 0)});
            variants.push_back({"sign AFTER all 8 stages",
                                gate_of(butterflies(fwd, 8), true, 0)});
            variants.push_back({"sign BEFORE all 8 stages",
                                gate_of(butterflies(fwd, 0), true, 0)});
            variants.push_back({"inner stages first (Pi transposed)",
                                gate_of(butterflies(rev, 5), true, 0)});
            variants.push_back({"ungated", butterflies(fwd, 3)});
            variants.push_back({"gate multiplied not divided",
                                gate_of(butterflies(fwd, 3), true, 1)});
            variants.push_back({"sigmoid of -gate",
                                gate_of(butterflies(fwd, 3), true, 2)});
            variants.push_back({"no rotation at all",
                                gate_of(normalized, true, 0)});

            std::fprintf(stderr, "  egress diagnosis, block %u query 0:\n", first_block);
            for (const Variant& v : variants) {
                double worst = 0.0;
                for (int c = 0; c < kHeadSize; ++c)
                    worst = std::max(worst, static_cast<double>(std::fabs(
                                                dev[c] - v.value[c])));
                std::fprintf(stderr, "    %-38s max|dev - variant| = %.6e\n",
                             v.name, worst);
            }
            // The rotation is checked on its own above, so when only the gate
            // is left the useful number is the factor the device actually
            // applied -- measured as context / rotated, both device values --
            // against the sigmoid it should have applied. No inference.
            std::fprintf(stderr, "    %-4s %10s %10s %9s %9s %9s\n", "ch",
                         "dev rot", "dev ctx", "factor", "sigmoid", "gate");
            for (int c = 0; c < 10; ++c) {
                const float rot = rotated_out[first_block * acc_words + c];
                const float ctx = dev[c];
                const float gate_value = GateEntry(head, 0, c);
                const float want_sig = 1.0f / (1.0f + std::exp(-gate_value));
                const double factor = (rot == 0.0f)
                                          ? 0.0
                                          : static_cast<double>(ctx) / rot;
                std::fprintf(stderr, "    %-4d %10.5f %10.5f %9.5f %9.5f %9.5f\n",
                             c, rot, ctx, factor, want_sig, gate_value);
            }
        }

        expect(worst_state_ulp <= kStateUlpBudget, "m and l within the ULP budget");
        expect(worst_acc_ulp <= kAccUlpBudget, "acc within the ULP budget");
        const double rotated_budget = largest_rotated_magnitude / 1024.0 + 1.0e-7;
        expect(worst_rotated_abs <= rotated_budget,
               "the egress rotation, before the gate, within its relative budget");
        expect(worst_context_abs <= context_budget,
               "the gated bf16 context within the relative budget");
        std::printf("  egress:   rotation err %.3e vs budget %.3e "
                    "(largest |rot| %.6f)\n",
                    worst_rotated_abs, rotated_budget, largest_rotated_magnitude);
        std::printf("  numerics: m,l <= %llu ULP, acc <= %llu ULP (largest |acc| "
                    "%.6f), context err %.3e vs budget %.3e (largest |ctx| %.6f)\n",
                    static_cast<unsigned long long>(worst_state_ulp),
                    static_cast<unsigned long long>(worst_acc_ulp),
                    largest_acc_intermediate, worst_context_abs, context_budget,
                    largest_context_magnitude);

        // ------------------------------------------------ the canary sweep
        if (one_tile) {
            const auto* qrot_raw = static_cast<const std::uint32_t*>(
                static_cast<const void*>(out_qrot.host_payload()));
            const auto* score_raw = static_cast<const std::uint32_t*>(
                static_cast<const void*>(out_scores.host_payload()));
            const auto* state_raw = static_cast<const std::uint32_t*>(
                static_cast<const void*>(out_state.host_payload()));
            const auto* acc_raw = static_cast<const std::uint32_t*>(
                static_cast<const void*>(out_acc.host_payload()));
            const auto* context_raw = static_cast<const std::uint32_t*>(
                static_cast<const void*>(out_context.host_payload()));

            struct Region {
                const std::uint32_t* words;
                std::size_t words_per_block;
            };
            const Region regions[] = {{qrot_raw, qrot_words},
                                      {score_raw, score_words},
                                      {state_raw, state_words},
                                      {acc_raw, acc_words},
                                      {context_raw, context_words}};
            std::size_t mutated = 0;
            std::size_t checked = 0;
            for (const Region& region : regions) {
                const CanarySweep report = sweep_inactive(
                    region.words, region.words_per_block, options.block);
                mutated += report.mutated;
                checked += report.checked;
            }
            expect(mutated == 0,
                   "no output word outside the active tile was mutated");
            std::printf("  isolation: %zu/%zu inactive output words still canary\n",
                        checked - mutated, checked);

            // The strips the active block did NOT process must have left their
            // archive panels canary too, which is the temporal half of the
            // isolation claim.
            std::size_t temporal_mutated = 0;
            std::size_t temporal_checked = 0;
            for (std::uint32_t i = options.strips;
                 i < static_cast<std::uint32_t>(kStrips); ++i)
                for (int entry = 0; entry < kScores; ++entry) {
                    ++temporal_checked;
                    if (score_raw[options.block * score_words + i * kScores + entry] !=
                        kTileCanary)
                        ++temporal_mutated;
                }
            expect(temporal_mutated == 0,
                   "the active block wrote no archive panel beyond its strip window");
            std::printf("  isolation: %zu/%zu unprocessed archive words still canary\n",
                        temporal_checked - temporal_mutated, temporal_checked);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what());
        return 1;
    }

    if (failures != 0) {
        std::fprintf(stderr, "FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS paged decode attention: strips %u from %u",
                options.strips, options.strip_begin);
    if (options.block < static_cast<std::uint32_t>(kGrid))
        std::printf(", block %u isolated\n", options.block);
    else
        std::printf(", all %d blocks\n", kGrid);
    return 0;
}
