// Tier-1 host gate for the staged TurboQuant integration layer.
//
// WHAT THIS IS, AND WHAT IT IS NOT
//
// This executable needs no NPU, no CANN runtime and no CAModel: it links nothing but the
// C++17 headers of the staged subtree, so it runs on any build host in milliseconds. That is
// its entire reason to exist. Every other verification target of this subtree --
// test_integration_decode, test_integration_batch, test_integration_scatter -- opens an ACL
// device session in main(), so on a host with neither a 950PR nor the simulator libraries
// they cannot run at all and the staged oracles ship on the strength of a clean compile.
//
// It is NOT kernel evidence and must never be reported as such (AGENTS.md: a host model is
// not CAModel evidence). No kernel is launched here and none could be. What it establishes is
// that the ORACLES and the SHARED CONTRACT HEADERS the device runners compare against are
// themselves sound -- which is a prerequisite for their verdicts meaning anything, because a
// device runner cannot tell a wrong kernel from a wrong oracle.
//
// THE GATES, in the order they run:
//
//  1. THE BATCH ORACLE (tests/common/turboquant_oracle.hpp), over the same strip ladder
//     test_integration_batch drives: 1, 2, 3, 32, 33, 48 and 64 strips. The fixture's own
//     audit -- all sixteen nibbles reached, the INT4 clamp taken, mixed scale exponents,
//     pairwise-distinct pages, tokens and strips -- plus the resident rotation blocks
//     composing to the oracle's dense 256x256 matrix, plus the page-ownership property the
//     device runner's canary check rests on: strip s owns physical page 2s+1 and NOTHING
//     writes an even page. A fixture that quietly went degenerate (as an earlier revision of
//     it did, clustering every nibble on 0 and +-1) passes a device run and proves nothing;
//     this is where that is caught.
//
//  2. THE SCATTER LAYOUT (tests/common/turboquant_scatter_oracle.hpp). The two cache index
//     maps are checked for being BIJECTIONS onto the nibble space, not merely in range: a
//     key-major and a channel-major map that collide on one lane would make the device's
//     read-modify-write silently lossy, and a bijection is the property that forbids it.
//     Every value nibble is also required to land inside its own 2048-byte strip window, the
//     one the kernel actually DMAs, and the (word, bit) merge arithmetic is round-tripped
//     over all sixteen codes in both nibble parities with the neighbouring nibble required
//     to survive. The quantiser is checked for the two branches that carry the contract:
//     scale == 2^(e-2) * invSqrtLen on a live row, and the explicit absmax == 0 branch
//     (scale 1.0f, all-zero codes) that no integer fixture reaches by accident.
//
//  3. THE FUSED DECODE FIXTURE (tests/common/fused_decode_opt_fixture.hpp), over the aligned
//     and the RAGGED geometries of the decode ladder, 128/1024/2048 and 79/257/1023. Its
//     tile and page arithmetic must account for exactly `context` live rows; its block table
//     must map only odd physical pages, distinctly and in range, which is what makes an
//     off-by-one in the paged addressing land on a canary page; its oracle must stay inside
//     the [1, 15] window reference_half admits; and its raw-query image must be fp16-exact,
//     which raw_query() enforces by throwing -- the property the raw-basis rungs' bit-exact
//     comparison rests on.
//
//  4. THE SPLIT-K WORKSPACE CONTRACT (include/turboquant_splitk_contract.hpp). Partition()
//     must tile the pages: every live page covered exactly once, ranges ascending and
//     contiguous, empty splits only past the end -- checked over every (pages, splits) pair
//     of the operator's range, including the uneven non-power-of-2 partitions the decode
//     ladder rides on (S = 3, 5, 7) where a split can legitimately be empty. PartitionOffset
//     must be injective and inside WorkspaceFloats, or two workers publish partials over
//     each other.
//
//  5. THE CROSS-HEADER UB RESTATEMENT. include/turboquant_gemm_reduce.hpp deliberately does
//     not include turboquant_fused_decode_opt.hpp -- it is dependency-free so the host gate
//     over it never sees production's -Wextra warnings -- and therefore RESTATES the
//     operator's three UB figures as literals. Nothing in either production translation unit
//     checks the restatement, because no production TU includes both. This one does, so the
//     drift is a compile error here (static_assert below) instead of a reduction phase sized
//     against a UB budget that moved.
//
//  6. The header-internal static_assert suites of turboquant_wave_protocol.hpp,
//     turboquant_gemm_reduce.hpp and turboquant_fused_decode_opt.hpp, which fire on inclusion
//     alone, plus a runtime sweep of the mode-4 channel id predicate and the GEMM
//     reduction's split alignment.

#include "../common/turboquant_oracle.hpp"
#include "../common/turboquant_scatter_oracle.hpp"
#include "../common/fused_decode_opt_fixture.hpp"
#include "../common/decode_tiling_gate.hpp"
#include "../../include/turboquant_splitk_contract.hpp"
#include "../../include/turboquant_gemm_reduce.hpp"
#include "../../op_kernel/common/turboquant_wave_protocol.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <set>
#include <string>
#include <vector>

namespace {
namespace gr = ascend_micro::gemm_reduce;
namespace fd = ascend_micro::fused_decode_opt;

// Gate 5. The two headers are included above in ONE translation unit, which is the only place
// this comparison can be made at all.
static_assert(gr::kOperatorUbEndBytes == fd::kUbEnd,
              "turboquant_gemm_reduce.hpp restates kUbEnd; it has drifted");
static_assert(gr::kOperatorBasisUbBytes == fd::kBasisUbBytes,
              "turboquant_gemm_reduce.hpp restates kBasisUbBytes; it has drifted");
static_assert(gr::kVectorReducerUbBytes == fd::kReducerUbBytes,
              "turboquant_gemm_reduce.hpp restates kReducerUbBytes; it has drifted");

int failures = 0;
int checks = 0;

void expect(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

std::string with(const char* what, int a)
{
    return std::string(what) + " [" + std::to_string(a) + "]";
}

// ------------------------------------------------------------------ gate 1
void check_batch_oracle(int strips)
{
    tq_oracle::Shape shape;
    shape.num_tokens = strips * 16;
    shape.num_kv_heads = 4;
    shape.head_size = 256;
    shape.block_size = 16;
    shape.num_blocks = tq_oracle::fragmented_num_blocks(strips);
    expect(shape.valid(), with("batch oracle: the shape is supported", strips));
    expect(shape.strips() == strips, with("batch oracle: strip count", strips));

    std::vector<std::int8_t> key, value;
    tq_oracle::build_activations(shape, key, value);
    const std::vector<std::int32_t> slots = tq_oracle::build_slot_mapping(shape);
    const std::vector<std::int8_t> tables = tq_oracle::build_tables(shape);

    expect(tq_oracle::audit_tables(shape, tables.data()) ==
               static_cast<long long>(shape.head_size) * shape.head_size,
           with("batch oracle: the resident blocks compose to the dense matrix", strips));

    const tq_oracle::Reference ref =
        tq_oracle::compute_oracle_reference(shape, key.data(), value.data(), slots.data());
    expect(ref.bad_slot_strip < 0,
           with("batch oracle: the slot mapping honours one strip per page", strips));

    const tq_oracle::Audit audit = tq_oracle::audit_reference(ref);
    expect(audit.nibbles_in_range, with("batch oracle: every code is INT4", strips));
    expect(audit.nibbles_covered == 16, with("batch oracle: all sixteen nibbles", strips));
    expect(audit.clamped_chunks > 0, with("batch oracle: the INT4 clamp is taken", strips));
    expect(audit.mixed_exponent_chunks > 0,
           with("batch oracle: a chunk carries two exponents", strips));
    expect(audit.distinct_exponents >= 3, with("batch oracle: the exponent varies", strips));
    expect(audit.pages_distinct, with("batch oracle: pages are pairwise distinct", strips));
    expect(audit.tokens_distinct, with("batch oracle: tokens are pairwise distinct", strips));
    expect(audit.strips_distinct, with("batch oracle: strips are pairwise distinct", strips));
    expect(audit.chunk_pairs > 0 && audit.chunk_max_collisions < audit.chunk_pairs,
           with("batch oracle: the chunk maxima are not degenerate", strips));
    expect(audit.sign_negatives >= shape.head_size / 4 &&
               audit.sign_negatives <= 3 * shape.head_size / 4,
           with("batch oracle: piSigns is balanced", strips));
    expect(audit.ok(), with("batch oracle: the aggregate audit verdict", strips));

    // The page-ownership property the device runner's canary protocol rests on: odd pages
    // only, one strip each, nothing on an even page.
    std::vector<int> owner(static_cast<std::size_t>(shape.num_blocks), -1);
    bool mapping_ok = true;
    for (int strip = 0; strip < strips; ++strip) {
        const int page = tq_oracle::page_of_slots(shape, slots.data(), strip);
        if (page != tq_oracle::page_of_strip(strip) || page % 2 == 0 ||
            page >= shape.num_blocks || owner[static_cast<std::size_t>(page)] != -1) {
            mapping_ok = false;
            break;
        }
        owner[static_cast<std::size_t>(page)] = strip;
    }
    expect(mapping_ok, with("batch oracle: strip s owns the distinct odd page 2s+1", strips));

    bool written_ok = true;
    for (int page = 0; page < shape.num_blocks; ++page) {
        const bool owned = owner[static_cast<std::size_t>(page)] >= 0;
        const bool written = ref.page_written[static_cast<std::size_t>(page)] != 0;
        if (owned != written) written_ok = false;
    }
    expect(written_ok, with("batch oracle: exactly the owned pages are written", strips));
}

// ------------------------------------------------------------------ gate 2
void check_scatter_layout()
{
    tq_scatter::Shape shape;
    shape.num_requests = 1;
    shape.num_kv_heads = 4;
    shape.head_size = 256;
    shape.block_size = 16;
    shape.num_blocks = 8;
    expect(shape.valid(), "scatter oracle: the shape is supported");

    const std::size_t nibbles = shape.cache_bytes() * 2;
    std::vector<char> seen_key(nibbles, 0), seen_value(nibbles, 0);
    bool key_bijective = true, value_bijective = true, value_in_strip = true;

    for (int block = 0; block < shape.num_blocks; ++block)
        for (int head = 0; head < shape.num_kv_heads; ++head)
            for (int token = 0; token < shape.block_size; ++token) {
                const std::size_t window =
                    tq_scatter::strip_byte_offset(shape, block, head,
                                                  token / tq_scatter::kStripHeight) * 2;
                for (int channel = 0; channel < shape.head_size; ++channel) {
                    const std::size_t kn =
                        tq_scatter::key_nibble(shape, block, head, token, channel);
                    if (kn >= nibbles || seen_key[kn]) key_bijective = false;
                    else seen_key[kn] = 1;

                    const std::size_t vn =
                        tq_scatter::value_nibble(shape, block, head, token, channel);
                    if (vn >= nibbles || seen_value[vn]) value_bijective = false;
                    else seen_value[vn] = 1;

                    if (vn < window || vn >= window + shape.strip_bytes() * 2)
                        value_in_strip = false;
                }
            }

    expect(key_bijective, "scatter oracle: the key map is injective and in range");
    expect(value_bijective, "scatter oracle: the value map is injective and in range");
    expect(value_in_strip,
           "scatter oracle: every value nibble lands in its own 2048-byte strip window");

    std::size_t key_covered = 0, value_covered = 0;
    for (std::size_t i = 0; i < nibbles; ++i) {
        key_covered += static_cast<std::size_t>(seen_key[i] != 0);
        value_covered += static_cast<std::size_t>(seen_value[i] != 0);
    }
    expect(key_covered == nibbles && value_covered == nibbles,
           "scatter oracle: both maps are onto the whole nibble space");

    // The scale plane, over the same geometry.
    std::vector<char> seen_scale(shape.scale_floats(), 0);
    bool scale_bijective = true;
    for (int plane = 0; plane < tq_scatter::kPlanes; ++plane)
        for (int block = 0; block < shape.num_blocks; ++block)
            for (int head = 0; head < shape.num_kv_heads; ++head)
                for (int token = 0; token < shape.block_size; ++token) {
                    const std::size_t si =
                        tq_scatter::scale_index(shape, plane, block, head, token);
                    if (si >= seen_scale.size() || seen_scale[si]) scale_bijective = false;
                    else seen_scale[si] = 1;
                }
    expect(scale_bijective, "scatter oracle: the scale map is injective and in range");

    // The (word, bit) merge arithmetic, both parities, every code, with the neighbour
    // required to survive -- the read-modify-write property the sequential ladder exists to
    // check on the device.
    bool merge_ok = true;
    for (int code = tq_scatter::kNibbleMin; code <= tq_scatter::kNibbleMax; ++code)
        for (std::size_t parity = 0; parity < 2; ++parity) {
            std::vector<std::int8_t> cache(2, 0);
            tq_scatter::write_nibble(cache, 1u - parity, tq_scatter::kNibbleMax);
            tq_scatter::write_nibble(cache, parity, code);
            if (tq_scatter::read_nibble(cache.data(), parity) != code) merge_ok = false;
            if (tq_scatter::read_nibble(cache.data(), 1u - parity) != tq_scatter::kNibbleMax)
                merge_ok = false;
        }
    expect(merge_ok, "scatter oracle: the nibble merge preserves the neighbouring lane");

    // The quantiser's two contract branches.
    const std::vector<float> signs = rot_oracle::canonical_signs();
    std::vector<float> x(static_cast<std::size_t>(shape.head_size));
    bool activations_exact = true;
    for (int c = 0; c < shape.head_size; ++c) {
        const float raw = tq_scatter::activation(0, 0, 0, 0, c, -1);
        if (!tq_scatter::bf16_is_exact(raw)) activations_exact = false;
        x[static_cast<std::size_t>(c)] = raw;
    }
    expect(activations_exact, "scatter oracle: every fixture activation is BF16-exact");

    const tq_scatter::RowRef row = tq_scatter::quantise_row(shape, x, signs);
    expect(!row.zero_row && row.absmax > 0.0f,
           "scatter oracle: the live row is not degenerate");
    expect(row.scale == std::ldexp(1.0f, row.exponent - 2) * shape.inv_sqrt_len(),
           "scatter oracle: a live row stores 2^(e-2) * invSqrtLen");
    double peak = 0.0;
    bool codes_in_range = true;
    for (float value : row.rotated)
        peak = std::fmax(peak, std::fabs(static_cast<double>(value)));
    for (int code : row.q)
        if (code < tq_scatter::kNibbleMin || code > tq_scatter::kNibbleMax)
            codes_in_range = false;
    expect(codes_in_range, "scatter oracle: every live code is INT4");
    expect(peak < 16777216.0, "scatter oracle: the rotated panel is exact in FP32");

    const std::vector<float> zeros(static_cast<std::size_t>(shape.head_size), 0.0f);
    const tq_scatter::RowRef zero = tq_scatter::quantise_row(shape, zeros, signs);
    expect(zero.zero_row, "scatter oracle: an all-zero row reaches the absmax == 0 branch");
    expect(zero.scale == 1.0f, "scatter oracle: the absmax == 0 branch stores scale 1.0f");
    bool zero_codes = true;
    for (int code : zero.q)
        if (code != 0) zero_codes = false;
    expect(zero_codes, "scatter oracle: the absmax == 0 branch stores all-zero codes");
}

// ------------------------------------------------------------------ gate 3
void check_fused_fixture(int context, int requests, int kv_heads, int group, int block)
{
    const std::string tag = "fused fixture N=" + std::to_string(context) +
                            " B=" + std::to_string(requests) +
                            " H=" + std::to_string(kv_heads) +
                            " G=" + std::to_string(group) +
                            " S=" + std::to_string(block);
    const fused_decode_opt_test::Fixture f(context, requests, kv_heads, group, block);

    int live_rows = 0;
    bool full_but_last = true;
    for (int tile = 0; tile < f.tiles(); ++tile) {
        const int rows = f.valid_rows(tile);
        live_rows += rows;
        if (rows <= 0 || rows > 64) full_but_last = false;
        if (tile + 1 < f.tiles() && rows != 64) full_but_last = false;
    }
    expect(live_rows == context, tag + ": the tiles account for every live row");
    expect(full_but_last, tag + ": only the last 64-row tile may be short");

    int page_rows = 0;
    for (int page = 0; page < f.pages; ++page) page_rows += f.page_rows(page);
    expect(page_rows == context, tag + ": the pages account for every live row");

    // The interleaved block table: odd physical pages only, distinct, in range. This is what
    // makes an off-by-one in the paged addressing land on a canary page.
    std::set<std::int32_t> mapped;
    bool table_ok = true;
    for (int request = 0; request < requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            const std::int32_t physical =
                f.table[static_cast<std::size_t>(request) * f.pages + page];
            if (physical < 0 || physical >= f.physical_pages || physical % 2 == 0 ||
                !mapped.insert(physical).second)
                table_ok = false;
        }
    expect(table_ok, tag + ": the block table maps distinct odd physical pages");
    expect(static_cast<int>(mapped.size()) == requests * f.pages,
           tag + ": every logical page has its own physical page");

    bool oracle_in_window = true;
    for (float value : f.expected)
        if (!std::isfinite(value) || value < 1.0f || value > 15.0f) oracle_in_window = false;
    expect(oracle_in_window, tag + ": the oracle stays inside the fp16 reference window");

    bool lengths_ok = true;
    for (int request = 0; request < requests; ++request)
        if (f.lengths[static_cast<std::size_t>(request)] != context) lengths_ok = false;
    expect(lengths_ok, tag + ": every request carries the fixture's context length");

    // The raw-query basis. raw_query() throws unless every channel of Pi(query) is fp16-exact,
    // which is the property the raw-basis rungs compare bit for bit.
    const fused_decode_opt_test::RawQueryBasis basis;
    bool raw_exact = true;
    std::vector<std::uint16_t> raw;
    try {
        raw = f.raw_query(basis);
    } catch (const std::exception&) {
        raw_exact = false;
    }
    expect(raw_exact, tag + ": the raw-query image is fp16-exact");
    expect(raw.size() == f.query.size(), tag + ": the raw query covers every query vector");

    // Pi is an involution, so rotating the raw image back must reproduce the pre-rotated
    // query the decode oracle was built from.
    if (raw_exact && raw.size() >= fd::kHeadSize) {
        std::vector<float> decoded(fd::kHeadSize, 0.0f);
        for (std::size_t c = 0; c < fd::kHeadSize; ++c)
            decoded[c] = ascend_micro::sim::half_to_float(raw[c]);
        const std::vector<float> back = basis.apply(decoded.data());
        double worst = 0.0;
        for (std::size_t c = 0; c < fd::kHeadSize; ++c)
            worst = std::fmax(worst, std::fabs(static_cast<double>(back[c]) -
                                               static_cast<double>(f.query[c])));
        expect(worst <= 1e-6, tag + ": Pi is an involution on the fixture's query");
    }
}

// ------------------------------------------------------------------ gate 4
void check_splitk_contract()
{
    namespace contract = turboquant_splitk;
    bool tiling_ok = true, empty_only_past_end = true;
    for (std::uint32_t pages = 1; pages <= 64; ++pages)
        for (std::uint32_t splits = 1; splits <= contract::kMaxSplits; ++splits) {
            std::uint32_t covered = 0, expected_begin = 0;
            bool seen_empty = false;
            for (std::uint32_t split = 0; split < splits; ++split) {
                const contract::PageRange range = contract::Partition(pages, splits, split);
                if (range.count == 0) {
                    seen_empty = true;
                    continue;
                }
                // A nonempty split after an empty one would leave a hole.
                if (seen_empty) empty_only_past_end = false;
                if (range.begin != expected_begin) tiling_ok = false;
                if (range.begin + range.count > pages) tiling_ok = false;
                expected_begin = range.begin + range.count;
                covered += range.count;
            }
            if (covered != pages) tiling_ok = false;
        }
    expect(tiling_ok, "split-K contract: Partition tiles the live pages exactly once");
    expect(empty_only_past_end,
           "split-K contract: an empty split is only ever past the last page");

    // The non-power-of-2 partitions the decode ladder rides on, stated as cases rather than
    // swept, because these are the ones where a split is legitimately empty and a reduction
    // must still combine its neutral element.
    expect(contract::Partition(16u, 5u, 4u).count == 0u,
           "split-K contract: the fifth split of 16 pages is empty");
    expect(contract::Partition(16u, 7u, 6u).count == 0u,
           "split-K contract: the seventh split of 16 pages is empty");
    expect(contract::Partition(16u, 3u, 0u).count == 6u &&
               contract::Partition(16u, 3u, 2u).count == 4u,
           "split-K contract: S = 3 over 16 pages partitions 6/6/4");

    bool offsets_ok = true;
    for (std::uint32_t requests = 1; requests <= 4; ++requests)
        for (std::uint32_t heads = 1; heads <= 4; ++heads)
            for (std::uint32_t splits = 1; splits <= contract::kMaxSplits; ++splits) {
                const std::size_t floats = contract::WorkspaceFloats(requests, heads, splits);
                std::set<std::uint64_t> seen;
                for (std::uint32_t item = 0; item < requests * heads; ++item)
                    for (std::uint32_t split = 0; split < splits; ++split) {
                        const std::uint64_t offset =
                            contract::PartitionOffset(item, split, splits);
                        if (!seen.insert(offset).second) offsets_ok = false;
                        if (offset + contract::kPartitionFloats > floats) offsets_ok = false;
                    }
                if (seen.size() != static_cast<std::size_t>(requests) * heads * splits)
                    offsets_ok = false;
            }
    expect(offsets_ok,
           "split-K contract: PartitionOffset is injective and inside the workspace");

    expect(contract::GridFitsOneWave(1u, 1u, 1u) && contract::GridFitsOneWave(2u, 4u, 4u) &&
               !contract::GridFitsOneWave(4u, 4u, 8u),
           "split-K contract: GridFitsOneWave follows the 32-block co-residency limit");
}

// ------------------------------------------------------------------ gate 6
void check_headers()
{
    // The header static_asserts both production policies' id legality and uniqueness. What is
    // left to check at runtime is the predicate itself over the whole id space, so a change
    // that widened the reserved range cannot pass by being asserted only where it is used.
    bool legality_ok = true;
    for (std::uint32_t id = 0; id < 64u; ++id) {
        const bool legal = turboquant_wave::LegalIntraId(static_cast<std::uint16_t>(id));
        const bool want =
            id < 2u * turboquant_wave::kIntraNum &&
            (id % turboquant_wave::kIntraNum) <= turboquant_wave::kMaxSafeLocalId;
        if (legal != want) legality_ok = false;
    }
    expect(legality_ok, "wave protocol: LegalIntraId admits exactly the safe half-ranges");
    expect(turboquant_wave::IntraId(1u, 0u) == turboquant_wave::kIntraNum,
           "wave protocol: AIV1's channel lives in the high half");
    expect(turboquant_wave::ReshapeDefault::kRecycles == turboquant_wave::kReshapeDefaultReuse,
           "wave protocol: the default policy follows kReshapeDefaultReuse");
    expect(turboquant_wave::TotalStrips(528u, 16u) == 33u &&
               !turboquant_wave::BlockOutOfRange(32u, 528u, 16u) &&
               turboquant_wave::BlockOutOfRange(32u, 512u, 16u),
           "wave protocol: the strip boundary admits the first block of wave 1");
    expect(!turboquant_wave::WholeStrips(1000u, 16u),
           "wave protocol: a partial trailing page is refused");

    // The GEMM reduction's split alignment: every admitted split count packs into one
    // 32-byte C0 row, which is what lets the dense Mmad treat the partial grid as fractals.
    bool aligned_ok = true;
    for (std::uint32_t splits = 1; splits <= gr::kMaxSplits; ++splits)
        if (gr::kAlignedSplits(splits) != gr::kC0Floats) aligned_ok = false;
    expect(aligned_ok, "gemm reduce: every admitted split count aligns to one C0 row");

    // The staged decode entry's own geometry guard, over the rungs the decode ladder uses.
    expect(fd::valid_launch(1u, 1u, 16u, 1u, 256u, 128u, 8u, 1u, 16u, 1u, 16u),
           "fused decode: the single-split rung is admitted");
    expect(!fd::valid_launch(1u, 1u, 16u, 1u, 256u, 128u, 8u, 9u, 16u, 1u, 16u),
           "fused decode: nine splits are refused");
    expect(!fd::valid_launch(1u, 1u, 16u, 1u, 256u, 96u, 8u, 1u, 16u, 1u, 16u),
           "fused decode: a page that is not a whole number of 64-row tiles is refused");
}

}  // namespace

int main()
{
    try {
        for (int strips : {1, 2, 3, 32, 33, 48, 64}) check_batch_oracle(strips);
        check_scatter_layout();
        // Aligned, then ragged: 79 is one page with a 15-row tail, 257 is three pages whose
        // last holds one token, 1023 is a 63-row tail over eight 128-token pages.
        check_fused_fixture(128, 1, 1, 16, 128);
        check_fused_fixture(1024, 1, 1, 16, 128);
        check_fused_fixture(2048, 2, 1, 16, 128);
        check_fused_fixture(79, 1, 1, 16, 128);
        check_fused_fixture(257, 1, 1, 16, 128);
        check_fused_fixture(1023, 1, 4, 4, 64);
        // The host tiling the decode now runs on: the dispatch odometer against the six
        // integer div/mods it replaced, the descriptors against the page walk CountTiles used
        // to do, and the four geometries the closed-form tile cursor cannot express. This is
        // the one gate of the subtree that checks them and it needs no device.
        tiling_gate::run();
        ++checks;
        check_splitk_contract();
        check_headers();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL staged host gate threw: %s\n", error.what());
        return 1;
    }

    if (failures == 0) {
        std::printf("staged integration layer host gate: PASS (%d checks, no device)\n",
                    checks);
        return 0;
    }
    std::fprintf(stderr, "staged integration layer host gate: %d of %d checks FAILED\n",
                 failures, checks);
    return 1;
}
