// Tier-1 host probe for the paged decode attention operator.
//
// Stage V1 of SKELETON_KERNEL_WORKFLOW.md: a bit-exact scalar oracle over
// guarded host buffers, with no device, no simulator and no toolchain. It
// checks the things the CAModel runs cannot check, because a CAModel run
// compares the device against this oracle and so cannot tell a wrong device
// from a wrong oracle:
//
//   1. the randomised Hadamard Pi satisfies Pi Pi^T = I exactly, and the 16
//      resident 32x32 blocks, indexed the way the Cube indexes them,
//      reproduce all 65536 entries of Pi;
//   2. the bf16 Nz image of a resident block is a permutation of its
//      row-major image and the index map round-trips -- the property that
//      `stage_blocks_nz` exists to establish and that a flat ND copy breaks
//      at 16 bits (reports/micro_layout_bf16.md);
//   3. the eight-stage butterfly the egress runs on AIV0 equals the dense
//      256x256 right-multiplication by Pi, bit for bit;
//   4. the int4 packing round-trips for both cache orders;
//   5. every fixture property the hazard checks depend on (guidelines
//      section 6.4): Q_rot exact in bf16, every raw score an exact multiple
//      of 1/16, strictly increasing per-strip maxima, distinct per-strip
//      probability mass, distinct output channels;
//   6. single-strip isolation: the golden state for one strip depends on that
//      strip alone, which is the invariant the canary protocol asserts on the
//      device.
//
// What it does NOT establish: anything about the device. It has no model of
// the pipeline, the flags, the UB layout or the engines. Deadlock freedom and
// token balance are the queue model's job
// (tests/cpu/test_pipeline_deadlock_probe.cpp); addressing and scheduling are
// CAModel's.

#include "../common/paged_decode_oracle.hpp"

#include <cstdio>
#include <vector>

namespace {

using namespace paged_decode;
using ascend_micro::sim::GuardedBuffer;

int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

// ----------------------------------------------------------------- check 1+2
void CheckRotationAndTableImage()
{
    ValidateRotation();
    std::printf("  rotation: Pi Pi^T == I and 16 resident blocks compose to Pi\n");

    const RotationTable table = BuildRotationTable();
    expect(table.row_major.size() == static_cast<std::size_t>(kTableElems) &&
               table.nz.size() == static_cast<std::size_t>(kTableElems),
           "rotation table size");

    // The Nz image must be a permutation of the row-major image: same
    // multiset, and at 16 bits a *different* arrangement, which is the whole
    // reason the ingress conversion exists.
    std::vector<std::uint16_t> a = table.row_major;
    std::vector<std::uint16_t> b = table.nz;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    expect(a == b, "Nz image is a permutation of the row-major image");
    expect(table.nz != table.row_major,
           "Nz and row-major images differ at 16 bits (if equal, the ingress "
           "conversion would be a no-op and C0 would still be 32)");

    // And the index map round-trips: reading the Nz image back through
    // Bf16NzBlockIndex must return the row-major entry.
    int mismatches = 0;
    for (int block = 0; block < kTableBlocks; ++block) {
        const int base = block * kGroup * kGroup;
        for (int r = 0; r < kGroup; ++r)
            for (int c = 0; c < kGroup; ++c) {
                const auto nz_index = ascend_micro::Bf16NzBlockIndex(
                    static_cast<std::uint32_t>(kGroup),
                    static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(c));
                if (table.nz[base + static_cast<int>(nz_index)] !=
                    table.row_major[base + r * kGroup + c])
                    ++mismatches;
            }
    }
    expect(mismatches == 0, "Nz index map round-trips");

    // How far apart the two images are, reported rather than asserted: it is
    // the size of the fault a flat ingress would introduce.
    int differing = 0;
    for (int i = 0; i < kTableElems; ++i)
        if (table.nz[i] != table.row_major[i]) ++differing;
    std::printf("  table image: %d/%d entries differ between Nz and row-major\n",
                differing, kTableElems);
}

// ------------------------------------------------------------------- check 3
void CheckEgressButterfly()
{
    // Dyadic, dense, and distinct per channel so a dropped stage shows.
    std::vector<float> row(kHeadSize);
    for (int c = 0; c < kHeadSize; ++c)
        row[c] = static_cast<float>(((c * 37) % 29) - 14) / 8.0f;

    const std::vector<float> butterfly = EgressButterfly(row.data());

    std::vector<float> dense(kHeadSize, 0.0f);
    for (int out = 0; out < kHeadSize; ++out) {
        float sum = 0.0f;
        for (int c = 0; c < kHeadSize; ++c) sum += row[c] * PiEntry(c, out);
        dense[out] = sum;
    }

    int differing = 0;
    for (int out = 0; out < kHeadSize; ++out)
        if (float_bits(butterfly[out]) != float_bits(dense[out])) ++differing;
    expect(differing == 0, "eight-stage butterfly equals dense Acc . Pi bit for bit");

    // A dropped stage must be detectable, or the check above is vacuous.
    std::vector<float> value(row);
    std::vector<float> next(kHeadSize, 0.0f);
    for (int stage = 0; stage < kEgressStageCount - 1; ++stage) {
        if (stage == kEgressSignAfterStage)
            for (int i = 0; i < kHeadSize; ++i) value[i] *= static_cast<float>(PiSign(i));
        const int stride = kEgressStrides[stage];
        for (int i = 0; i < kHeadSize; ++i) {
            const float sign = (i & stride) ? -1.0f : 1.0f;
            next[i] = value[i ^ stride] + sign * value[i];
        }
        value.swap(next);
    }
    for (int i = 0; i < kHeadSize; ++i) value[i] *= kInvSqrtLen;
    bool any_difference = false;
    for (int out = 0; out < kHeadSize; ++out)
        if (float_bits(value[out]) != float_bits(dense[out])) any_difference = true;
    expect(any_difference, "dropping the last butterfly stage changes the result");
    std::printf("  egress: 8 butterfly stages == dense 256x256 Pi, exactly\n");
}

// ------------------------------------------------------------------- check 4
void CheckPacking()
{
    GuardedBuffer<std::uint8_t> key(kPackedBytes);
    GuardedBuffer<std::uint8_t> value(kPackedBytes);
    for (int i = 0; i < kPackedBytes; ++i) {
        key[static_cast<std::size_t>(i)] = 0;
        value[static_cast<std::size_t>(i)] = 0;
    }

    const CacheFill where{0, 0, 0};
    std::vector<std::uint8_t> key_image(kKeyCacheBytes, 0);
    std::vector<std::uint8_t> value_image(kValueCacheBytes, 0);
    WriteKeyStrip(key_image.data(), where, 3);
    WriteValueStrip(value_image.data(), where, 3);

    int key_bad = 0;
    int value_bad = 0;
    const std::uint8_t* key_strip = key_image.data() + KeyStripOffset(0, 0, 0);
    const std::uint8_t* value_strip = value_image.data() + ValueStripOffset(0, 0, 0);
    for (int token = 0; token < kTokens; ++token)
        for (int channel = 0; channel < kHeadSize; ++channel) {
            if (ReadNibble(key_strip, KeyNibbleIndex(token, channel)) !=
                KeyNibble(where.head, 3, token, channel))
                ++key_bad;
            if (ReadNibble(value_strip, ValueNibbleIndex(channel, token)) !=
                ValueNibble(where.head, 3, channel, token))
                ++value_bad;
        }
    expect(key_bad == 0, "key cache nibble packing round-trips (token major)");
    expect(value_bad == 0, "value cache nibble packing round-trips (channel major)");

    // The two orders must not coincide, or a kernel that confuses them passes.
    expect(key_image != value_image,
           "token-major and channel-major packings of the same strip differ");

    // The scale cache planes must not overlap.
    expect(ScaleOffset(0, kNumBlocks - 1, kKvHeads - 1, kBlockSize - 1) <
               ScaleOffset(1, 0, 0, 0),
           "scale cache key and value planes are disjoint");
    expect(ScaleOffset(1, kNumBlocks - 1, kKvHeads - 1, kBlockSize - 1) + 1 ==
               kScaleCacheFloats,
           "scale cache planar extent matches its declared size");

    ASSERT_CANARY_INTACT(key);
    ASSERT_CANARY_INTACT(value);
    std::printf("  packing: both cache orders round-trip and are distinguishable\n");
}

// ------------------------------------------------------------------- check 5
void CheckFixture()
{
    for (int head = 0; head < kKvHeads; ++head) {
        const FixtureReport report = ValidateFixture(head);
        std::printf("  fixture head %d: |Q_rot| <= %.6f, strip maxima %.4f .. %.4f\n",
                    head, report.max_qrot_magnitude, report.strip_max.front(),
                    report.strip_max.back());
    }

    // Heads must be separable, or the canary protocol's H = 0, 2, 3 targets
    // prove nothing: a kernel that read head 0's cache slice for every head
    // would pass them all. Q itself is head invariant here -- a decode query
    // is one tensor -- so the separation has to come from the cache slice, the
    // scale plane and the gate, and this is where that is established.
    std::vector<int> all_strips(kStrips);
    for (int i = 0; i < kStrips; ++i) all_strips[i] = i;
    for (const int head : {0, 2, 3})
        for (const int other : {0, 2, 3}) {
            if (other == head) continue;
            const GoldenState mine = GoldenDecode(head, all_strips);
            const GoldenState theirs = GoldenDecode(other, all_strips);
            int differing = 0;
            for (std::size_t i = 0; i < mine.context.size(); ++i)
                if (float_bits(mine.context[i]) != float_bits(theirs.context[i]))
                    ++differing;
            expect(differing * 4 >= static_cast<int>(mine.context.size()) * 3,
                   "two boundary heads differ in most output channels");
        }
    // And the same one strip at a time, which is the regime the canary
    // protocol actually runs in.
    for (const int strip : {0, 4, 7}) {
        const GoldenState a = GoldenDecode(0, {strip});
        const GoldenState b = GoldenDecode(2, {strip});
        int differing = 0;
        for (std::size_t i = 0; i < a.context.size(); ++i)
            if (float_bits(a.context[i]) != float_bits(b.context[i])) ++differing;
        expect(differing * 4 >= static_cast<int>(a.context.size()) * 3,
               "heads 0 and 2 differ on a single strip as well");
    }
    std::printf("  fixture: heads 0, 2 and 3 separable over 8 strips and "
                "one strip at a time\n");

    // The block table must not be the identity.
    bool table_is_identity = true;
    for (int t = 0; t < kTemporalBlocks; ++t)
        if (BlockTableEntry(t) != t) table_is_identity = false;
    expect(!table_is_identity,
           "block table is not the identity, so ignoring it is detectable");
}

// ------------------------------------------------------------------- check 6
void CheckSingleStripIsolation()
{
    // The canary protocol runs one strip. Its golden state must be a function
    // of that strip alone -- the device-side claim is spatial and temporal
    // isolation, and this is the host half of it.
    for (const int strip : {0, 4, 7}) {
        const GoldenState single = GoldenDecode(2, {strip});
        const GoldenState again = GoldenDecode(2, {strip});
        int differing = 0;
        for (std::size_t i = 0; i < single.context.size(); ++i)
            if (float_bits(single.context[i]) != float_bits(again.context[i]))
                ++differing;
        expect(differing == 0, "single-strip golden state is deterministic");

        // And distinct from every other strip's, or "the active tile matched"
        // would not identify which tile ran.
        for (const int other : {0, 4, 7}) {
            if (other == strip) continue;
            const GoldenState rival = GoldenDecode(2, {other});
            bool differs = false;
            for (std::size_t i = 0; i < single.context.size(); ++i)
                if (float_bits(single.context[i]) != float_bits(rival.context[i]))
                    differs = true;
            expect(differs, "two boundary strips produce different contexts");
        }
    }

    // A single strip has l > 0 and a finite m for every query, so the
    // normalisation the egress performs is well defined with one strip only.
    const GoldenState single = GoldenDecode(0, {7});
    for (int q = 0; q < kTokens; ++q) {
        expect(single.l[q] > 0.0f, "single-strip l is positive");
        expect(std::isfinite(single.m[q]), "single-strip m is finite");
    }
    std::printf("  isolation: strips 0, 4 and 7 give three distinct, "
                "deterministic single-strip states\n");
}

// The full eight-strip state, reported so the CAModel runner's expected
// magnitudes have a host-side provenance.
void ReportSteadyState()
{
    std::vector<int> strips(kStrips);
    for (int i = 0; i < kStrips; ++i) strips[i] = i;
    const GoldenState state = GoldenDecode(0, strips);
    float acc_max = 0.0f;
    float ctx_max = 0.0f;
    for (float value : state.acc) acc_max = std::max(acc_max, std::fabs(value));
    for (float value : state.context) ctx_max = std::max(ctx_max, std::fabs(value));
    std::printf("  head 0, 8 strips: m[0] = %.6f, l[0] = %.6f, "
                "max|acc| = %.6f, max|context| = %.6f\n",
                state.m[0], state.l[0], acc_max, ctx_max);
}

}  // namespace

int main()
{
    std::printf("paged decode attention: Tier-1 host probe\n");
    try {
        CheckRotationAndTableImage();
        CheckEgressButterfly();
        CheckPacking();
        CheckFixture();
        CheckSingleStripIsolation();
        ReportSteadyState();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS paged decode Tier-1: rotation, table image, egress "
                "butterfly, packing, fixture and single-strip isolation\n");
    return 0;
}
