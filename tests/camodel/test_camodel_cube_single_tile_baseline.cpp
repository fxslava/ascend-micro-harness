// Single-tile Cube baseline for the Split-K crossover.
// Ascend950PR_9599, CANN 9.2.0-beta.2, MIX_AIC_1_2.
//
//   --strip T    the one strip to run, 0..7 (default 0)
//   --block B    the grid block to isolate, 0..31 (default 0)
//   --strips N   strips per launch, 1..8 (default 1) -- see below
//
// What this harness is, and what it is NOT
// ----------------------------------------
// It links the FROZEN production operator
// `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp`, unmodified,
// and launches it over exactly one 16-token strip on exactly one block. Its
// product is T_compute_strip, the Cube-side cost of one strip, which is the
// denominator of the Split-K overhead ratio
//
//   T_compute(T_tile) = (T_tile / 16) . T_compute_strip
//   overhead %        = T_sync(K) / T_compute(T_tile) . 100
//
// It is NOT a second correctness runner for that operator:
// `test_camodel_paged_decode_attention.cpp` is, and it covers twelve rungs
// including the strip ladder and the nine canary coordinates. What this one
// checks is only enough to establish that the run it TIMED was a correct run
// -- the ingress rotation and the strip's raw score panel, both bit-exactly --
// because a cycle count taken from a broken execution is worthless and a
// cycle count nobody checked is indistinguishable from one.
//
// The number itself comes from the instruction dumps, not from this process:
// the operator is frozen, so it carries no counter instrumentation and none
// may be added. `scripts/analyze_cube_tile_baseline.py` reads the Cube core's
// dumps and reports the spans. This harness's job is to produce a run whose
// dumps mean what the analyser assumes.
//
// Why --strips exists on a single-tile harness
// --------------------------------------------
// Because a single strip measures a LATENCY and the crossover needs a RATE,
// and they are different numbers on a pipelined operator. With one strip
// nothing overlaps: the measured span is the rotation prologue plus one
// strip's serial path through MTE2, the staging quorum, MTE1, eight Mmads and
// the Fixpipe. With eight strips the Cube issues a strip's Mmad group every
// ~19,900 ticks while the recurrence on AIV0 runs behind it, and THAT is the
// interval at which a split retires work. The report uses the steady-state
// interval as the crossover's denominator and the single-tile span as the
// cold-start term, and it needs both from the same build and the same
// fixture, so one harness produces both. Defaulting to 1 keeps the cheap run
// cheap.

#include "camodel_common.hpp"
#include "../common/paged_decode_oracle.hpp"

#include <algorithm>
#include <charconv>
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
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof raw);
    return raw;
}

void fill_payload(camodel::GuardedDeviceBuffer& buffer, std::uint32_t word)
{
    auto* payload = static_cast<std::uint32_t*>(buffer.host_payload());
    for (std::size_t i = 0; i < buffer.payload_bytes / 4; ++i) payload[i] = word;
}

int parse_u32(std::string_view text, std::uint32_t& out)
{
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc() ? 0 : -1;
}

}  // namespace

int main(int argc, char** argv)
{
    std::uint32_t strip_begin = 0;
    std::uint32_t block = 0;
    std::uint32_t strips = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--strip" && i + 1 < argc) {
            if (parse_u32(argv[++i], strip_begin) != 0 ||
                strip_begin >= static_cast<std::uint32_t>(kStrips)) {
                std::fprintf(stderr, "--strip must be 0..%d\n", kStrips - 1);
                return 2;
            }
        } else if (arg == "--block" && i + 1 < argc) {
            if (parse_u32(argv[++i], block) != 0 ||
                block >= static_cast<std::uint32_t>(kGrid)) {
                std::fprintf(stderr, "--block must be 0..%d\n", kGrid - 1);
                return 2;
            }
        } else if (arg == "--strips" && i + 1 < argc) {
            if (parse_u32(argv[++i], strips) != 0 || strips == 0 ||
                strips > static_cast<std::uint32_t>(kStrips)) {
                std::fprintf(stderr, "--strips must be 1..%d\n", kStrips);
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %.*s\n", static_cast<int>(arg.size()),
                         arg.data());
            return 2;
        }
    }
    if (strip_begin + strips > static_cast<std::uint32_t>(kStrips)) {
        std::fprintf(stderr, "--strip %u with --strips %u runs past strip %d\n",
                     strip_begin, strips, kStrips - 1);
        return 2;
    }

    const int head = static_cast<int>(block / kTemporalBlocks);
    const int temporal = static_cast<int>(block % kTemporalBlocks);
    std::printf(
        "cube single-tile baseline: block %u isolated (head %d, temporal %d), "
        "%u strip(s) from %u\n",
        block, head, temporal, strips, strip_begin);

    try {
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
        camodel::GuardedDeviceBuffer block_table(
            static_cast<std::size_t>(kTemporalBlocks) * 4);
        camodel::GuardedDeviceBuffer gate(static_cast<std::size_t>(kKvHeads) * kTokens *
                                          kHeadSize * 4);
        camodel::GuardedDeviceBuffer gather_offsets(
            static_cast<std::size_t>(kEgressGatherCount) * kHeadSize * 4);
        camodel::GuardedDeviceBuffer gather_signs(
            static_cast<std::size_t>(kEgressGatherCount) * kHeadSize * 4);
        camodel::GuardedDeviceBuffer pi_sign_buffer(static_cast<std::size_t>(kHeadSize) *
                                                    4);
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

        // The caches are canary everywhere except the slices this one tile
        // owns, which is the canary isolation regime the production runner
        // uses: a read outside the tile returns a quiet NaN and poisons the
        // output rather than perturbing it, so a timed run that silently read
        // the wrong page cannot also look correct.
        fill_payload(key_cache, kTileCanary);
        fill_payload(value_cache, kTileCanary);
        fill_payload(scale_cache, kTileCanary);

        auto* key_bytes = static_cast<std::uint8_t*>(key_cache.host_payload());
        auto* value_bytes = static_cast<std::uint8_t*>(value_cache.host_payload());
        auto* scale_floats = static_cast<float*>(scale_cache.host_payload());
        const int page = BlockTableEntry(temporal);
        for (std::uint32_t i = 0; i < strips; ++i) {
            const int strip = static_cast<int>(strip_begin + i);
            const CacheFill where{page, head, strip};
            WriteKeyStrip(key_bytes, where, strip);
            WriteValueStrip(value_bytes, where, strip);
            WriteScales(scale_floats, where, strip);
        }

        auto* q_half = static_cast<std::uint16_t*>(q.host_payload());
        for (int h = 0; h < kKvHeads; ++h)
            for (int row = 0; row < kTokens; ++row)
                for (int channel = 0; channel < kHeadSize; ++channel)
                    q_half[(static_cast<std::size_t>(h) * kTokens + row) * kHeadSize +
                           channel] = float_to_bfloat16_rne(QueryEntry(row, channel));

        const RotationTable table = BuildRotationTable();
        std::memcpy(pi_table.host_payload(), table.row_major.data(),
                    table.row_major.size() * 2);

        auto* gate_floats = static_cast<float*>(gate.host_payload());
        for (int h = 0; h < kKvHeads; ++h)
            for (int row = 0; row < kTokens; ++row)
                for (int channel = 0; channel < kHeadSize; ++channel)
                    gate_floats[(static_cast<std::size_t>(h) * kTokens + row) *
                                    kHeadSize +
                                channel] = GateEntry(h, row, channel);

        auto* table_words = static_cast<std::uint32_t*>(block_table.host_payload());
        for (int t = 0; t < kTemporalBlocks; ++t)
            table_words[t] = static_cast<std::uint32_t>(BlockTableEntry(t));

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
        for (int i = 0; i < kHeadSize; ++i) pi_signs[i] = static_cast<float>(PiSign(i));

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
            static_cast<float*>(out_rotated.payload()), strips, strip_begin, block);
        camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

        for (auto* buffer : {&out_qrot, &out_scores, &out_state, &out_acc, &out_context,
                             &out_rotated, &key_cache, &value_cache, &scale_cache})
            buffer->download();
        for (auto* buffer : {&out_qrot, &out_scores, &out_state, &out_acc, &out_context,
                             &key_cache, &value_cache, &scale_cache})
            buffer->assert_guards();

        // ---- the two bit-exact gates -----------------------------------
        // Everything the Cube produced, and nothing else. m, l, acc and the
        // context go through `Exp` and are checked in ULP by the production
        // runner; repeating that here would add nothing to a timing baseline,
        // and these two are what prove the Cube's path -- int4 -> UB ->
        // L1(Nz) -> L0B -> Mmad -> Fixpipe -- executed correctly on the run
        // that was timed.
        const auto* qrot_out = static_cast<const std::uint16_t*>(out_qrot.host_payload());
        const auto* score_out = static_cast<const float*>(out_scores.host_payload());

        const std::vector<float> rot = GoldenQRot(head);
        std::size_t qrot_bad = 0;
        for (std::size_t i = 0; i < rot.size(); ++i) {
            const std::uint16_t want = float_to_bfloat16_rne(rot[i]);
            if (qrot_out[block * rot.size() + i] != want) ++qrot_bad;
        }
        expect(qrot_bad == 0, "the 64-Mmad ingress rotation is bit-exact");

        std::size_t score_bad = 0;
        std::size_t score_canary = 0;
        for (std::uint32_t i = 0; i < strips; ++i) {
            const std::vector<float> raw =
                GoldenRawScores(head, static_cast<int>(strip_begin + i));
            for (int entry = 0; entry < kScores; ++entry) {
                const float got = score_out[block * score_words + i * kScores + entry];
                if (bits_of(got) == kTileCanary) ++score_canary;
                if (bits_of(got) != bits_of(raw[entry])) ++score_bad;
            }
        }
        expect(score_bad == 0, "every timed strip's raw score panel is bit-exact");
        expect(score_canary == 0, "every timed strip's score panel was written");

        // The archive slots beyond the launch's window must still be canary:
        // the operator copies out exactly the panels it filled, and a run that
        // published uninitialised Unified Buffer would also have executed a
        // different number of bursts.
        std::size_t beyond = 0;
        for (std::uint32_t i = strips; i < static_cast<std::uint32_t>(kStrips); ++i)
            for (int entry = 0; entry < kScores; ++entry)
                if (bits_of(score_out[block * score_words + i * kScores + entry]) !=
                    kTileCanary)
                    ++beyond;
        expect(beyond == 0, "no score panel beyond the launch's window was published");

        // No other block ran: the grid is 32 and 31 blocks take the early
        // return, which is what makes the Cube dump of core `block` the dump
        // of this measurement and nothing else.
        std::size_t other_blocks = 0;
        for (std::uint32_t b = 0; b < static_cast<std::uint32_t>(kGrid); ++b) {
            if (b == block) continue;
            for (std::size_t i = 0; i < score_words; ++i)
                if (bits_of(score_out[b * score_words + i]) != kTileCanary)
                    ++other_blocks;
        }
        expect(other_blocks == 0, "only the isolated block wrote any output");

        std::printf(
            "  gates: qrot %zu bad, scores %zu bad / %zu canary, beyond-window %zu, "
            "other blocks %zu\n",
            qrot_bad, score_bad, score_canary, beyond, other_blocks);
        std::printf("  block %u isolated on core %u\n", block, block);
        std::printf(
            "  the tick spans are in the Cube dump of core %u; run\n"
            "  scripts/analyze_cube_tile_baseline.py over it\n",
            block);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FATAL %s\n", error.what());
        return 1;
    }

    if (failures == 0) {
        std::printf("cube single-tile baseline: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "cube single-tile baseline: %d FAILURES\n", failures);
    return 1;
}
