// CAModel runner for turboquant_reshape_and_cache_a8: `--slices N` aligned
// 16-token strips, each one grid block, eight chunks per strip through a
// four-stage depth-4 MIX_AIC_1_2 pipeline, checked against the shared CPU
// oracle in tests/common/turboquant_oracle.hpp.
//
// Why the grid is collapsed to 1, 2 and 3 strips
// ---------------------------------------------
// The device shapes this operator is built for run 32 strips (512 tokens) and
// 64 (1024). A cycle-accurate model of either is hours, and there is no NPU on
// this workstation, so the grid is the collapsed dimension: the obligation a
// multi-strip run discharges is that strip t's addressing is a function of t
// alone, uniform in t, with state carried only through the per-core L1 and UB
// rings. For that, the sample set is ring-depth aware rather than fixed:
//
//   1 strip   the cold start, and the only grid this kernel had ever run at;
//   2 strips  the induction step -- two blocks dispatch, each one's Vector
//             cores must address its own strip and its own physical page;
//   3 strips  one interior step past the induction.
//
// The device shapes themselves, 32 and 64 strips, are reachable here too: with
// ASCEND_MICRO_SPARSE_SAMPLING the kernel runs only the first, middle and last
// four blocks of each 32-block wave (tests/common/turboquant_sampling.hpp),
// every other block returns from the entry having touched nothing, and this
// runner then requires those blocks' pages to still hold their canary. 24 of
// 64 blocks execute, so a 1024-token grid costs about what 24 strips cost
// while still covering the highest block indices, an interior step per wave,
// and a second execution wave. Build with -DASCEND_MICRO_SPARSE_SAMPLING=OFF
// for the full-count soak.
//
// What it does not prove: nothing here bounds error growth with the strip
// count (there is none -- each strip's quantiser is independent), and a
// collapsed grid understates full-count resource use. It is "executes on the
// cycle-accurate model over S = {0, 1, 2} and matches the audited golden",
// and that is all it is written up as.
//
// The addressing this is pointed at
// --------------------------------
// GetBlockIdx() on a Vector core of a MIX launch is the AIV index, not the
// grid block index: at a 1:2 AIC:AIV ratio it runs over [0, 2 * numBlocks).
// Taken raw it made the two Vector cores of block b address strips 2b and
// 2b + 1 while their Cube read strip b. At one strip the quotient is 0 either
// way, so the single-block runs this probe used to do could not see it.
//
// Checks, in order:
//
//   0.  the fixture properties, from the shared oracle's self-audit: nibble
//       coverage, the INT4 upper clamp taken, exponent spread, and pairwise
//       distinctness of chunks inside a strip, of tokens inside a chunk and of
//       the same (plane, head) across strips -- a fixture whose strips are
//       interchangeable cannot detect a wrong-strip read, which is the whole
//       point of running more than one;
//   0b. the 16 resident rotation blocks in `tables`, composed the way the Cube
//       composes them, reproduce all 65536 entries of the 256x256 matrix the
//       oracle applies;
//   1.  head and tail Global Memory canaries on all eight buffers;
//   2.  every cache byte and scale float on a page no block wrote still
//       0xDEADBEEF -- numBlocks is 2 * strips + 2 and strip s owns page
//       2s + 1, so a stray burst or a mis-scaled page offset has somewhere to
//       land where it is caught rather than ignored. Under sampling this set
//       also contains the pages of every skipped block, which is what makes a
//       skipped block's silence checkable rather than assumed;
//   2b. every written page has an unwritten page on both sides of it, so a
//       burst that overran its 2 KiB -- interconnect or L2 write combining
//       included -- corrupts a canary instead of a neighbour's real data;
//   3.  keyCache, key-major, bit-exact, per strip, chunk and token;
//   4.  valueCache, channel-major, bit-exact, per strip, chunk and channel;
//   5.  scaleCache, bit-exact FP32, per strip, chunk and token.

#include "camodel_common.hpp"

#include "../common/turboquant_oracle.hpp"
#if defined(ASCEND_MICRO_SPARSE_SAMPLING)
#include "../common/turboquant_sampling.hpp"
#endif

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_a8(
    void* stream, const std::int8_t* key, const std::int8_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens,
    std::uint32_t num_kv_heads, std::uint32_t head_size, std::uint32_t block_size,
    std::uint32_t num_blocks, std::uint32_t tokens_per_core, float inv_sqrt_len);

namespace {

constexpr int kHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kBlockSize = 16;
// 64 strips is the 1024-token device shape, and twice the model's 32 AI Cores,
// so --slices 64 is the one grid here that runs a second execution wave on a
// physical core. Measured cost on this model is ~8 s of wall clock per added
// strip on top of a ~45 s fixed cost, which puts even the full shapes inside an
// edit-test cycle: 1/2/3 strips for iteration, 32 and 64 before a claim.
constexpr int kMaxSlices = 64;

using tq_oracle::ChunkRef;
using tq_oracle::Shape;

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FIXTURE/CHECK FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

void usage(std::FILE* output, const char* argv0)
{
    std::fprintf(output,
                 "Usage: %s [--slices N]\n"
                 "  N aligned 16-token strips, one grid block each (default 1,"
                 " 1..%d).\n"
                 "  The inductive set for multi-strip addressing is 1, 2, 3;\n"
                 "  32 and 64 are the device shapes (512 and 1024 tokens).\n"
#if defined(ASCEND_MICRO_SPARSE_SAMPLING)
                 "  Built WITH sparse boundary sampling: at N > %d only the\n"
                 "  first %d, middle %d and last %d blocks of each %d-block wave\n"
                 "  execute; the rest must leave their pages canary.\n",
                 argv0, kMaxSlices, tq_sampling::kMinSampledWidth,
                 tq_sampling::kSampleEdge, tq_sampling::kSampleCenter,
                 tq_sampling::kSampleEdge, tq_sampling::kBlocksPerWave);
#else
                 "  Built WITHOUT sparse sampling: every block executes.\n",
                 argv0, kMaxSlices);
#endif
}

// The kernel's chunk numbering: 0..kHeads-1 are the value heads and the rest
// the key heads, which is the order its four-stage schedule runs them in.
int chunk_plane(int chunk) { return chunk >= kHeads ? 0 : 1; }
int chunk_head(int chunk) { return chunk % kHeads; }
const char* chunk_label(int chunk) { return chunk >= kHeads ? "key" : "value"; }

// How wrong is one (strip, chunk) page, and in which of the ways it can be
// wrong? Stopping at the first differing byte says nothing about *how* a page
// is wrong, and the plausible causes -- a stale landing panel, a wrong-strip
// read, a layout swap, a scale exponent off by one, a page never written at
// all -- all look identical at one byte.
int diagnose(const Shape& shape, const tq_oracle::Reference& ref, int strip, int chunk,
             const std::uint8_t* got, const float* got_scales)
{
    const int plane = chunk_plane(chunk);
    const int head = chunk_head(chunk);
    const ChunkRef& want = ref.chunk(strip, plane, head);
    const int head_bytes = shape.head_bytes();

    int bad_bytes = 0, first_bad = -1, canary_bytes = 0;
    for (int i = 0; i < head_bytes; ++i) {
        const auto lane = static_cast<std::uint8_t>((camodel::kCanary >> (8 * (i % 4))) & 0xFFu);
        if (got[i] != want.packed[static_cast<std::size_t>(i)]) {
            if (first_bad < 0) first_bad = i;
            ++bad_bytes;
        }
        canary_bytes += (got[i] == lane);
    }
    int bad_scales = 0;
    for (int t = 0; t < shape.block_size; ++t)
        if (tq_oracle::float_bits(got_scales[t]) !=
            tq_oracle::float_bits(want.scale[static_cast<std::size_t>(t)]))
            ++bad_scales;
    if (bad_bytes == 0 && bad_scales == 0) return 0;

    std::fprintf(stderr,
                 "\nstrip %d page %d chunk %d (%s head %d): %d/%d bytes differ "
                 "(first byte %d: device 0x%02x oracle 0x%02x), %d/%d scales differ, "
                 "%d/%d bytes still canary\n",
                 strip, want.page, chunk, chunk_label(chunk), head, bad_bytes,
                 head_bytes, first_bad,
                 first_bad < 0 ? 0u : static_cast<unsigned>(got[first_bad]),
                 first_bad < 0 ? 0u
                               : static_cast<unsigned>(want.packed[static_cast<std::size_t>(first_bad)]),
                 bad_scales, shape.block_size, canary_bytes, head_bytes);

    // Which oracle page, from which strip, in which layout, does this look
    // like? A stale panel, a wrong-slot read or a block-index error shows up
    // here as a high byte-match count against a page other than its own.
    for (int s = 0; s < shape.strips(); ++s)
        for (int c = 0; c < 2 * kHeads; ++c) {
            const ChunkRef& other = ref.chunk(s, chunk_plane(c), chunk_head(c));
            if (other.packed.empty()) continue;
            for (int lay = 0; lay < 2; ++lay) {
                const std::vector<std::uint8_t>& candidate =
                    lay == 0 ? other.packed : other.packed_alt;
                int same = 0;
                for (int i = 0; i < head_bytes; ++i)
                    same += (got[i] == candidate[static_cast<std::size_t>(i)]);
                if (same * 4 > head_bytes || (s == strip && c == chunk))
                    std::fprintf(stderr,
                                 "   looks like strip %d chunk %d %s layout: %d/%d bytes\n",
                                 s, c, lay == 0 ? "own" : "swapped", same, head_bytes);
            }
        }

    // Per token: how many nibbles differ, and is there an exponent that would
    // reproduce the device's nibbles from this chunk's own rotated values? If
    // there is, the rotation is right and only the scale is wrong.
    std::vector<int> device_q;
    tq_oracle::unpack_page(shape, plane, got, device_q);
    for (int t = 0; t < shape.block_size; ++t) {
        const std::size_t row = static_cast<std::size_t>(t) * shape.head_size;
        int diff = 0, lo = 99, hi = -99;
        for (int ch = 0; ch < shape.head_size; ++ch) {
            const int delta = device_q[row + ch] - want.q[row + ch];
            if (delta != 0) ++diff;
            if (delta < lo) lo = delta;
            if (delta > hi) hi = delta;
        }
        int fit = -999;
        for (int e2 = 0; e2 < 32 && fit == -999; ++e2) {
            bool all = true;
            for (int ch = 0; ch < shape.head_size && all; ++ch) {
                long long v = tq_oracle::rint_even_shift(want.y[row + ch], e2 - 2);
                if (v > tq_oracle::kNibbleMax) v = tq_oracle::kNibbleMax;
                if (v < tq_oracle::kNibbleMin || v != device_q[row + ch]) all = false;
            }
            if (all) fit = e2;
        }
        if (diff == 0 && bad_scales == 0) continue;
        std::fprintf(stderr,
                     "   token %2d: %3d/%d nibbles differ, delta [%d, %d], device "
                     "scale %.9g oracle %.9g (oracle e=%d, exponent reproducing "
                     "device: %d)\n",
                     t, diff, shape.head_size, diff ? lo : 0, diff ? hi : 0,
                     static_cast<double>(got_scales[t]),
                     static_cast<double>(want.scale[static_cast<std::size_t>(t)]),
                     want.exponent[static_cast<std::size_t>(t)], fit);
    }
    return 1;
}

int run(int slices)
{
    Shape shape;
    shape.num_tokens = slices * kBlockSize;
    shape.num_kv_heads = kHeads;
    shape.head_size = kHeadSize;
    shape.block_size = kBlockSize;
    shape.num_blocks = tq_oracle::fragmented_num_blocks(slices);
    require(shape.valid(), "unsupported shape");
    const int head_bytes = shape.head_bytes();
    const int channels = kHeads * kHeadSize;

    // ---- which grid blocks actually execute --------------------------------
    // With sampling compiled in, only the sampled blocks run; every other
    // block returns from the kernel entry having touched nothing, so its page
    // must still hold its canary afterwards. The flags come from the same
    // header the kernel consults, so host and device cannot disagree.
    std::vector<char> runs(static_cast<std::size_t>(shape.strips()), 1);
#if defined(ASCEND_MICRO_SPARSE_SAMPLING)
    for (int s = 0; s < shape.strips(); ++s)
        runs[static_cast<std::size_t>(s)] =
            tq_sampling::is_sampled(s, shape.strips()) ? 1 : 0;
#endif
    int sampled = 0;
    for (char r : runs) sampled += (r != 0);

    // ---- fixture and oracle, both from the shared header -------------------
    std::vector<std::int8_t> key_in, value_in;
    tq_oracle::build_activations(shape, key_in, value_in);
    const std::vector<std::int32_t> slot_map = tq_oracle::build_slot_mapping(shape);
    const std::vector<std::int8_t> table_blob = tq_oracle::build_tables(shape);
    const tq_oracle::Reference ref =
        tq_oracle::compute_oracle_reference(shape, key_in.data(), value_in.data(),
                                            slot_map.data(), runs.data());
    require(ref.bad_slot_strip < 0,
            "the fixture's slot mapping breaks the one-strip-one-page contract");
    require(sampled > 0, "the sample set is empty: nothing would execute");

    // ---- check 0: the fixture can actually detect a wrong read -------------
    // The activations are std::int8_t, so "inside INT8" is a type invariant
    // rather than something to assert. What has to be asserted is that the
    // fixture is *distinguishable*, which is what the audit covers.
    const tq_oracle::Audit audit = tq_oracle::audit_reference(ref);
    require(audit.nibbles_in_range, "the oracle produced a value outside INT4");
    require(audit.nibbles_covered == 16,
            "the fixture does not exercise all sixteen nibbles");
    require(audit.clamped_chunks > 0, "the INT4 upper clamp is never taken");
    require(audit.mixed_exponent_chunks > 0,
            "no chunk carries two different scale exponents");
    require(audit.distinct_exponents >= 3, "the scale exponent barely varies");
    // Not "no two chunks share a maximum": see Audit::chunk_max_collisions.
    // What has to hold is that the *pages* differ, which is what a wrong-chunk
    // read would have to reproduce, and that not every pair collides.
    require(audit.pages_distinct, "two chunks of one strip pack to identical pages");
    require(audit.chunk_pairs > 0 && audit.chunk_max_collisions < audit.chunk_pairs,
            "every pair of chunks in a strip shares a maximum: the fixture is degenerate");
    require(audit.tokens_distinct, "two tokens of one chunk quantise identically");
    require(audit.strips_distinct, "two strips pack to identical pages");
    require(audit.sign_negatives >= kHeadSize / 4 &&
                audit.sign_negatives <= 3 * kHeadSize / 4,
            "piSigns is degenerate, a dropped sign multiply would not show");

    // ---- check 0b: the resident blocks compose to the oracle's matrix ------
    const long long composed = tq_oracle::audit_tables(shape, table_blob.data());
    require(composed == static_cast<long long>(kHeadSize) * kHeadSize,
            "the resident rotation blocks do not compose to the dense matrix the "
            "oracle applies");

    // ---- device buffers ---------------------------------------------------
    camodel::Session session;
    camodel::GuardedDeviceBuffer key(key_in.size());
    camodel::GuardedDeviceBuffer value(value_in.size());
    camodel::GuardedDeviceBuffer key_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer value_cache(shape.cache_bytes());
    camodel::GuardedDeviceBuffer scale_cache(shape.scale_floats() * sizeof(float));
    camodel::GuardedDeviceBuffer slot_mapping(slot_map.size() * sizeof(std::int32_t));
    camodel::GuardedDeviceBuffer pi_signs(kHeadSize * sizeof(float));
    camodel::GuardedDeviceBuffer tables(table_blob.size());

    std::memcpy(key.host_payload(), key_in.data(), key_in.size());
    std::memcpy(value.host_payload(), value_in.data(), value_in.size());
    std::memcpy(slot_mapping.host_payload(), slot_map.data(),
                slot_map.size() * sizeof(std::int32_t));
    std::memcpy(tables.host_payload(), table_blob.data(), table_blob.size());
    auto* ps = static_cast<float*>(pi_signs.host_payload());
    // piSigns still reaches the device in the published signature, and the
    // runner still asserts it is non-degenerate, but the device does not read
    // it: the signs are inside the rotation blocks, and check 0b is what ties
    // the two together.
    for (int ch = 0; ch < kHeadSize; ++ch) ps[ch] = static_cast<float>(tq_oracle::pi_sign(ch));

    key.upload();
    value.upload();
    slot_mapping.upload();
    pi_signs.upload();
    tables.upload();

    launch_turboquant_reshape_and_cache_a8(
        session.stream, static_cast<const std::int8_t*>(key.payload()),
        static_cast<const std::int8_t*>(value.payload()),
        static_cast<std::int8_t*>(key_cache.payload()),
        static_cast<std::int8_t*>(value_cache.payload()),
        static_cast<float*>(scale_cache.payload()),
        static_cast<const std::int32_t*>(slot_mapping.payload()),
        static_cast<const float*>(pi_signs.payload()),
        static_cast<const std::int32_t*>(tables.payload()),
        static_cast<std::uint32_t>(shape.num_tokens),
        static_cast<std::uint32_t>(shape.num_kv_heads),
        static_cast<std::uint32_t>(shape.head_size),
        static_cast<std::uint32_t>(shape.block_size),
        static_cast<std::uint32_t>(shape.num_blocks),
        static_cast<std::uint32_t>(shape.block_size), shape.inv_sqrt_len());
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    key_cache.download();
    value_cache.download();
    scale_cache.download();
    key.download();
    value.download();
    slot_mapping.download();
    pi_signs.download();
    tables.download();

    // ---- check 1: canaries ------------------------------------------------
    key.assert_guards();
    value.assert_guards();
    key_cache.assert_guards();
    value_cache.assert_guards();
    scale_cache.assert_guards();
    slot_mapping.assert_guards();
    pi_signs.assert_guards();
    tables.assert_guards();

    const auto* kc = static_cast<const std::uint8_t*>(key_cache.host_payload());
    const auto* vc = static_cast<const std::uint8_t*>(value_cache.host_payload());
    const auto* sc = static_cast<const float*>(scale_cache.host_payload());

    // ---- check 2: nothing outside a written page moved --------------------
    std::size_t untouched = 0;
    for (int page = 0; page < shape.num_blocks; ++page) {
        if (ref.page_written[static_cast<std::size_t>(page)] != 0) continue;
        const std::size_t base = ref.page_offset(page);
        for (int i = 0; i < shape.page_bytes(); ++i) {
            const std::size_t at = base + static_cast<std::size_t>(i);
            // The payload base is 4-byte aligned inside a buffer filled with
            // the 32-bit canary, so byte `at` must equal canary lane at % 4.
            const auto lane =
                static_cast<std::uint8_t>((camodel::kCanary >> (8 * (at % 4))) & 0xFFu);
            require(kc[at] == lane, "keyCache byte on a page no strip owns was overwritten");
            require(vc[at] == lane, "valueCache byte on a page no strip owns was overwritten");
            ++untouched;
        }
        for (int plane = 0; plane < tq_oracle::kPlanes; ++plane)
            for (int head = 0; head < kHeads; ++head) {
                const std::size_t at = ref.scale_offset(plane, page, head);
                for (int t = 0; t < shape.block_size; ++t)
                    require(tq_oracle::float_bits(sc[at + static_cast<std::size_t>(t)]) ==
                                camodel::kCanary,
                            "scaleCache float on a page no strip owns moved");
            }
    }

    // ---- check 2b: a written page's neighbours are canary -----------------
    // The point of sampling the wave's interior rather than only its ends: a
    // center block's neighbours in the grid are skipped, so the pages adjacent
    // to the page it writes belong to strips that never ran. Any burst that
    // overran a page boundary -- a mis-scaled offset, an interconnect or L2
    // write-combining effect that wrote past the 2 KiB it was asked for -- has
    // to land on one of those and be named. Checked in physical page order, so
    // it covers both the even pages no strip owns and the odd pages of skipped
    // strips.
    int isolated_pages = 0;
    for (int page = 0; page < shape.num_blocks; ++page) {
        if (ref.page_written[static_cast<std::size_t>(page)] == 0) continue;
        const bool low_free =
            page == 0 || ref.page_written[static_cast<std::size_t>(page - 1)] == 0;
        const bool high_free = page + 1 >= shape.num_blocks ||
                               ref.page_written[static_cast<std::size_t>(page + 1)] == 0;
        require(low_free && high_free,
                "two written pages are physically adjacent: the fixture cannot "
                "show a burst that overran a page boundary");
        ++isolated_pages;
    }

    // ---- checks 3-5: bit-exact payloads, per strip and per chunk ----------
    std::size_t key_bytes = 0, value_bytes = 0, scales = 0;
    int failing = 0;
    for (int strip = 0; strip < shape.strips(); ++strip) {
        if (runs[static_cast<std::size_t>(strip)] == 0) continue;
        for (int chunk = 0; chunk < 2 * kHeads; ++chunk) {
            const int plane = chunk_plane(chunk);
            const int head = chunk_head(chunk);
            const int page = ref.chunk(strip, plane, head).page;
            const std::uint8_t* got = (plane == 0 ? kc : vc) + ref.head_offset(page, head);
            const float* got_scales = sc + ref.scale_offset(plane, page, head);
            if (diagnose(shape, ref, strip, chunk, got, got_scales) == 0) {
                if (plane == 0)
                    key_bytes += static_cast<std::size_t>(head_bytes);
                else
                    value_bytes += static_cast<std::size_t>(head_bytes);
                scales += static_cast<std::size_t>(shape.block_size);
                continue;
            }
            if (++failing >= 3) {
                std::fprintf(stderr, "   (further failing chunks not detailed)\n");
                strip = shape.strips();
                break;
            }
        }
    }
    if (failing != 0) throw std::runtime_error("packed cache page mismatch");

    // Everything below counts the blocks that actually executed, never the
    // grid: a sampled run that reported the full grid's byte count would be
    // claiming coverage it does not have.
    const int chunks = sampled * 2 * kHeads;
    std::printf("PASS turboquant_reshape_and_cache_a8: %d of %d strips x 16 tokens x "
                "%d channels, %d chunks over a depth-4 L1 ring and depth-2 UB rings\n",
                sampled, shape.strips(), channels, chunks);
    std::printf("  grid %d blocks, strip s -> physical page 2s+1 of %d\n",
                shape.strips(), shape.num_blocks);
#if defined(ASCEND_MICRO_SPARSE_SAMPLING)
    if (sampled != shape.strips()) {
        std::printf("  sparse boundary sampling: %d blocks per wave, first %d, "
                    "%d straddling the midpoint and last %d of each wave; blocks run:",
                    tq_sampling::kBlocksPerWave, tq_sampling::kSampleEdge,
                    tq_sampling::kSampleCenter, tq_sampling::kSampleEdge);
        for (int s = 0; s < shape.strips(); ++s)
            if (runs[static_cast<std::size_t>(s)] != 0) std::printf(" %d", s);
        std::printf("\n  %d blocks returned from the kernel entry having touched "
                    "nothing; their pages are checked as canary below\n",
                    shape.strips() - sampled);
    }
#endif
    std::printf("  keyCache   %zu/%d bytes bit-exact, key-major\n", key_bytes,
                sampled * kHeads * head_bytes);
    std::printf("  valueCache %zu/%d bytes bit-exact, channel-major\n", value_bytes,
                sampled * kHeads * head_bytes);
    std::printf("  scaleCache %zu/%d FP32 bit-exact\n", scales,
                chunks * shape.block_size);
    std::printf("  %zu cache bytes on the %d pages no block wrote untouched, "
                "%d/%d guard bytes intact\n",
                untouched, shape.num_blocks - sampled,
                8 * 2 * static_cast<int>(camodel::kGuardBytes),
                8 * 2 * static_cast<int>(camodel::kGuardBytes));
    std::printf("  every one of the %d written pages has an unwritten page on both "
                "sides, so a burst past a page boundary lands on a canary\n",
                isolated_pages);
    std::printf("  strip 0 per-chunk max |y|:");
    for (int chunk = 0; chunk < 2 * kHeads; ++chunk)
        std::printf(" %lld", ref.chunk(0, chunk_plane(chunk), chunk_head(chunk)).chunk_max);
    std::printf("\n  %d/%d in-strip chunk pairs share a maximum (harmless: nibbles are "
                "scale invariant, and all %d pages are pairwise distinct)\n",
                audit.chunk_max_collisions, audit.chunk_pairs, chunks);
    std::printf("  scale exponents: %d distinct, %d/%d chunks carry two or more\n",
                audit.distinct_exponents, audit.mixed_exponent_chunks, chunks);
    std::printf("  nibble values exercised: %d/16 (range [%d, %d]), INT4 upper clamp "
                "taken on %d/%d chunks\n",
                audit.nibbles_covered, audit.min_q, audit.max_q, audit.clamped_chunks,
                chunks);
    std::printf("  rotation table: %d blocks of %dx%d INT8 (%zu B), %lld composed "
                "entries match the oracle's matrix\n",
                2 * shape.groups(), tq_oracle::kGroup, tq_oracle::kGroup,
                table_blob.size(), composed);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    int slices = 1;
    bool saw_slices = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argc == 2 ? stdout : stderr, argv[0]);
            return argc == 2 ? 0 : 2;
        }
        if (arg == "--slices" && !saw_slices && i + 1 < argc) {
            const std::string_view value = argv[++i];
            const auto parsed =
                std::from_chars(value.data(), value.data() + value.size(), slices);
            if (parsed.ec == std::errc() && parsed.ptr == value.data() + value.size() &&
                slices >= 1 && slices <= kMaxSlices) {
                saw_slices = true;
                continue;
            }
        }
        usage(stderr, argv[0]);
        return 2;
    }
    try {
        return run(slices);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL turboquant_reshape_and_cache_a8 (%d slices): %s\n",
                     slices, error.what());
        return 1;
    }
}
