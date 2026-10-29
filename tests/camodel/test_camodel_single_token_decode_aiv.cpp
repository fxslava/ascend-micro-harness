// CAModel runner and single-head canary protocol for the AIV-only
// single-token decode operator (roadmap No.3). Ascend950PR_9599, CANN 9.2.
//
//   --head H          isolate one (request, kv_head) item: ONE block in the
//                     grid, one head's cache slices and one head's query rows
//                     real, every other Global Memory word left as a canary
//   --pages P         logical pages the request spans, 1..8 (default 1)
//   --strips N        strips processed per page, 1..8 (default 8)
//   --strip-begin T   first strip of that window (default 0)
//   --all             every item in one launch, grid = requests x kv_heads
//   --rows R          rows per item, 1..8 (default 6)
//   --no-fixture-check   skip the host fixture validation (~2 s)
//
// The protocol, and what each part of it can catch
// ------------------------------------------------
// Every Global Memory payload -- inputs and outputs, not only outputs -- is
// flooded with 0x7FC0BEEF, and then the fixture is written into ONLY the
// slices the isolated item is entitled to read:
//
//   * the key, value and scale slices of (physical page, head, strip) for the
//     strips in the window, with the PHYSICAL page coming from the block
//     table, which is deliberately not the identity;
//   * that head's `rows` query rows and `rows` gate rows, and no others.
//
// 0x7FC0BEEF is a quiet NaN, so a read outside those slices propagates into
// everything downstream rather than perturbing it slightly -- a wrong page, a
// wrong head, a wrong strip or a wrong query row is a NaN panel, not a close
// one. Uploading only the active head's QUERY rows is stronger than the Cube
// operator's protocol, which uploads Q whole: there, head separation comes
// only from the cache slice and the gate, so a task that read another head's
// query rows would be invisible.
//
// Writes are checked the other way round. Every output buffer is canary
// before the launch, so three different claims become one sweep: the item's
// own slots must all be written, every OTHER item's slots must still be
// canary, and inside the active item the archive slots past the processed
// strips and the rows past `rows` must still be canary -- the output guard
// band. `camodel::GuardedDeviceBuffer` adds a second, different pattern
// (0xDEADBEEF) in 64 bytes either side of every payload, which catches a burst
// that runs off the end of a buffer rather than landing in the wrong place
// inside one. Neither pattern is 0x00, which a half-issued burst also leaves.
//
// What is checked bit-exactly, and what is not
// --------------------------------------------
//   out_scores  BIT-EXACT, every panel of every strip. The Tier-1 probe
//               establishes that all three summation orders agree on this
//               fixture, so this is a gate on the device and not on the
//               arithmetic. It is the check that locates a fault anywhere on
//               the int4 -> UB -> unpack -> GEMV path.
//   m           BIT-EXACT. A maximum of exactly representable score columns.
//   l, acc      ULP, against the largest intermediate: both pass through
//               `Exp` (1 ULP per call, documented) and `acc` is a cancelling
//               reduction.
//   rotated     relative, against the largest value in the row. Eight
//               butterfly stages over a cancelling sum.
//   context     relative, in bf16, after one more `Exp` and one more `Div`.
//
// What this runner does NOT establish: silicon, a grid wider than the items it
// launches, and anything about synchronisation -- that is the trace audit's
// job (scripts/analyze_single_token_decode_trace.py).

#include "camodel_common.hpp"
#include "../common/single_token_decode_oracle.hpp"
#include "../common/decode_boundary_fixture.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_single_token_decode_boundary_aiv(
    void*, std::uint32_t, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*,
    float*, float*, float*, float*, float*, float*, std::uint16_t*, std::int32_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t);

extern "C" void launch_turboquant_single_token_decode_aiv(
    void*, std::uint32_t, std::uint16_t*, std::int8_t*, std::int8_t*, float*,
    std::uint32_t*,
    float*, float*, float*, float*, float*, float*, std::uint16_t*, std::int32_t*,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t);

namespace {

using namespace single_token_decode;

// The in-bounds canary. A quiet NaN read as FP32.
constexpr std::uint32_t kTileCanary = 0x7FC0BEEFu;
// The kernel's compile-time row bound and archive geometry. Restated here
// because a runner cannot include a device header; every one of these is a
// stride the kernel turns into a Global Memory address, so a disagreement
// shows up as a canary word in the comparison rather than subtly.
constexpr int kMaxRows = 8;
constexpr int kArchiveStride = kMaxRows * kKeys;  // 128 floats
constexpr int kMaxArchiveStrips = 64;
constexpr int kProbeLanes = 32;

constexpr int kNumRequests = 1;
constexpr int kCachePages = paged_decode::kNumBlocks;  // 9 physical pages
constexpr int kBlockSize = paged_decode::kBlockSize;   // 128 tokens per page

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

struct CanarySweep {
    std::size_t checked = 0;
    std::size_t mutated = 0;
};

// ------------------------------------------------------------------- options
struct Options {
    std::uint32_t head = 0;
    std::uint32_t pages = 1;
    std::uint32_t page_begin = 0;
    std::uint32_t strips = static_cast<std::uint32_t>(kStripsPerPage);
    std::uint32_t strip_begin = 0;
    std::uint32_t rows = static_cast<std::uint32_t>(kRows);
    bool exact_boundary = false;
    bool all = false;
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
        if (flag == "--head") {
            options.head = parse_u32(next("--head"), "--head");
        } else if (flag == "--pages") {
            options.pages = parse_u32(next("--pages"), "--pages");
        } else if (flag == "--page-begin") {
            options.page_begin = parse_u32(next("--page-begin"), "--page-begin");
        } else if (flag == "--strips") {
            options.strips = parse_u32(next("--strips"), "--strips");
        } else if (flag == "--strip-begin") {
            options.strip_begin = parse_u32(next("--strip-begin"), "--strip-begin");
        } else if (flag == "--rows") {
            options.rows = parse_u32(next("--rows"), "--rows");
        } else if (flag == "--exact-boundary") {
            options.exact_boundary = true;
        } else if (flag == "--all") {
            options.all = true;
        } else if (flag == "--no-fixture-check") {
            options.fixture_check = false;
        } else {
            throw std::invalid_argument("unknown option");
        }
    }
    if (options.head >= static_cast<std::uint32_t>(kItems))
        throw std::invalid_argument("--head out of range");
    if (options.pages < 1 || options.page_begin >= static_cast<std::uint32_t>(kMaxPages) ||
        options.pages > static_cast<std::uint32_t>(kMaxPages) - options.page_begin)
        throw std::invalid_argument("--pages out of range");
    if (options.strips < 1 ||
        options.strip_begin + options.strips >
            static_cast<std::uint32_t>(kStripsPerPage))
        throw std::invalid_argument("strips window out of range");
    // Against `kRows`, the ORACLE's domain, and not against the kernel's
    // `kMaxRows` buffer bound of 8.
    //
    // The two are different numbers for a reason, and conflating them cost a
    // 822-second run. `kMaxRows` is how wide the kernel's Unified Buffer
    // panels are; `kRows` is how many rows this fixture models, and it is 6
    // because that is H_q / H_kv for the model — at 8 rows per KV head H_q
    // would be 32, which is not a shape this oracle describes. The oracle's
    // panel is kRows * kKeys floats and its `m`, `l` and `acc` are kRows
    // entries, so a run at `--rows 8` drove the device correctly and then
    // compared it against host memory past the end of three std::vectors:
    // 256 of 1024 panel entries "wrong" (exactly rows 6 and 7 over 8 strips)
    // and a largest |acc| of 1.7e35 read out of whatever followed the
    // allocation. The device was never at fault and the run established
    // nothing, which is the worst kind of failure to look at.
    if (options.rows < 1 || options.rows > static_cast<std::uint32_t>(kRows))
        throw std::invalid_argument(
            "--rows must be 1..6: the oracle models H_q / H_kv = 6 rows per "
            "(request, kv_head), and the kernel's kMaxRows = 8 is a Unified "
            "Buffer bound, not a shape this fixture can check");
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
                     "usage: %s [--page-begin P] [--exact-boundary] [--head H] [--pages P] [--strips N] [--strip-begin T]\n"
                     "          [--rows R] [--all] [--no-fixture-check]\n",
                     argv[0]);
        return 2;
    }

    const int rows = static_cast<int>(options.rows);
    std::printf("single-token decode (AIV only) on CAModel: %u row(s), %u page(s), "
                "%u strip(s) from %u, ",
                options.rows, options.pages, options.strips, options.strip_begin);
    if (options.all)
        std::printf("all %d items\n", kItems);
    else
        std::printf("item %u isolated (request 0, kv_head %u)\n", options.head,
                    options.head);

    // The window, as global strip indices, in the order the kernel walks it.
    std::vector<int> window;
    for (std::uint32_t page = 0; page < options.pages; ++page)
        for (std::uint32_t i = 0; i < options.strips; ++i)
            window.push_back(static_cast<int>((options.page_begin + page) * kStripsPerPage +
                                              options.strip_begin + i));

    try {
        if (options.fixture_check) {
            paged_decode::ValidateRotation();
            for (int head = 0; head < kKvHeads; ++head) {
                ValidateSharedFixture(head);
                (void)ValidateFixture(head, static_cast<int>(options.pages));
            }
            std::printf("  fixture: Pi, the shared one-page identity and all four "
                        "heads validated on the host\n");
        }

        // ------------------------------------------------------- allocation
        camodel::Session session;

        const std::size_t qrot_floats =
            static_cast<std::size_t>(kNumRequests) * kQHeads * kHeadSize;
        const std::size_t archive_floats =
            static_cast<std::size_t>(kItems) * kMaxArchiveStrips * kArchiveStride;
        const std::size_t state_floats =
            static_cast<std::size_t>(kItems) * 2 * kMaxRows * kStateLanes;
        const std::size_t panel_floats =
            static_cast<std::size_t>(kItems) * kMaxRows * kHeadSize;

        camodel::GuardedDeviceBuffer q_rot(qrot_floats * 2);
        camodel::GuardedDeviceBuffer key_cache(paged_decode::kKeyCacheBytes);
        camodel::GuardedDeviceBuffer value_cache(paged_decode::kValueCacheBytes);
        camodel::GuardedDeviceBuffer scale_cache(paged_decode::kScaleCacheFloats * 4);
        camodel::GuardedDeviceBuffer block_table(
            static_cast<std::size_t>(kMaxPages) * 4);
        camodel::GuardedDeviceBuffer gate(qrot_floats * 4);
        camodel::GuardedDeviceBuffer pi_sign_buffer(
            static_cast<std::size_t>(kHeadSize) * 4);

        camodel::GuardedDeviceBuffer out_scores(archive_floats * 4);
        camodel::GuardedDeviceBuffer out_state(state_floats * 4);
        camodel::GuardedDeviceBuffer out_acc(panel_floats * 4);
        camodel::GuardedDeviceBuffer out_rotated(panel_floats * 4);
        camodel::GuardedDeviceBuffer out_context(panel_floats * 2);
        camodel::GuardedDeviceBuffer probe(static_cast<std::size_t>(kProbeLanes) * 4);

        // -------------------------------------------------------- the inputs
        // Everything starts as canary -- the caches, the query panel and the
        // gate -- and only the slices the active item is entitled to read are
        // then overwritten.
        fill_payload(key_cache, kTileCanary);
        fill_payload(value_cache, kTileCanary);
        fill_payload(scale_cache, kTileCanary);
        fill_payload(q_rot, kTileCanary);
        fill_payload(gate, kTileCanary);
        fill_payload(block_table, kTileCanary);
        fill_payload(pi_sign_buffer, kTileCanary);

        auto* key_bytes = static_cast<std::uint8_t*>(key_cache.host_payload());
        auto* value_bytes = static_cast<std::uint8_t*>(value_cache.host_payload());
        auto* scale_floats = static_cast<float*>(scale_cache.host_payload());
        auto* qrot_values = static_cast<std::uint16_t*>(q_rot.host_payload());
        auto* gate_values = static_cast<float*>(gate.host_payload());

        std::size_t slices_written = 0;
        for (int head = 0; head < kKvHeads; ++head) {
            if (!options.all && head != static_cast<int>(options.head)) continue;
            for (const int strip : window) {
                const paged_decode::CacheFill where{PhysicalPageOfStrip(strip), head,
                                                    StripInPage(strip)};
                // The cache slices, written from THIS oracle's nibble and
                // scale functions rather than paged_decode's, because beyond
                // page 0 they differ -- that is what makes a page mix-up
                // observable at all.
                std::uint8_t* key_strip =
                    key_bytes + paged_decode::KeyStripOffset(where.block, where.head,
                                                             where.strip);
                std::uint8_t* value_strip =
                    value_bytes + paged_decode::ValueStripOffset(where.block, where.head,
                                                                 where.strip);
                for (int token = 0; token < kKeys; ++token)
                    for (int channel = 0; channel < kHeadSize; ++channel)
                        paged_decode::PackNibble(
                            key_strip, paged_decode::KeyNibbleIndex(token, channel),
                            options.exact_boundary ? 3 : KeyNibbleAt(head, strip, token, channel));
                for (int channel = 0; channel < kHeadSize; ++channel)
                    for (int token = 0; token < kKeys; ++token)
                        paged_decode::PackNibble(
                            value_strip, paged_decode::ValueNibbleIndex(channel, token),
                            options.exact_boundary ? decode_boundary::Value(strip, channel) :
                                ValueNibbleAt(head, strip, channel, token));
                for (int token = 0; token < kKeys; ++token) {
                    const int in_page = where.strip * kKeys + token;
                    scale_floats[paged_decode::ScaleOffset(0, where.block, where.head,
                                                           in_page)] =
                        options.exact_boundary ? 1.0f : KeyScale(head, strip, token);
                    scale_floats[paged_decode::ScaleOffset(1, where.block, where.head,
                                                           in_page)] =
                        options.exact_boundary ? 1.0f : ValueScaleAt(head, strip, token);
                }
                ++slices_written;
            }
            // Only this head's query and gate rows. Query head j = h * rows + r.
            for (int row = 0; row < rows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t at =
                        (static_cast<std::size_t>(head) * rows + row) * kHeadSize + c;
                    qrot_values[at] = float_to_bfloat16_rne(options.exact_boundary ? 1.0f / 256.0f : QRotEntry(head, row, c));
                    gate_values[at] = options.exact_boundary ? 0.0f : GateEntry(head, QueryRowOf(head, row), c);
                }
        }

        // Read whole by whichever block runs, so always real.
        auto* table_words = static_cast<std::uint32_t*>(block_table.host_payload());
        for (unsigned page = options.page_begin; page < options.page_begin + options.pages; ++page)
            table_words[page] =
                static_cast<std::uint32_t>(paged_decode::BlockTableEntry(page));
        auto* pi_signs = static_cast<float*>(pi_sign_buffer.host_payload());
        for (int c = 0; c < kHeadSize; ++c)
            pi_signs[c] = static_cast<float>(paged_decode::PiSign(c));

        // Outputs start as canary so an unwritten word is distinguishable from
        // a zero the kernel might legitimately produce.
        fill_payload(out_scores, kTileCanary);
        fill_payload(out_state, kTileCanary);
        fill_payload(out_acc, kTileCanary);
        fill_payload(out_rotated, kTileCanary);
        fill_payload(out_context, kTileCanary);
        fill_payload(probe, kTileCanary);

        for (auto* buffer : {&q_rot, &key_cache, &value_cache, &scale_cache,
                             &block_table, &gate, &pi_sign_buffer, &out_scores,
                             &out_state, &out_acc, &out_rotated, &out_context, &probe})
            buffer->upload();

        std::printf("  uploaded: %zu cache slice(s) and %d query row(s) real, "
                    "everything else canary 0x%08X\n",
                    slices_written, options.all ? rows * kKvHeads : rows, kTileCanary);

        const std::vector<camodel::GuardedDeviceBuffer*> read_only = {
            &q_rot, &key_cache, &value_cache, &scale_cache, &gate, &block_table, &pi_sign_buffer};
        std::vector<std::vector<std::uint32_t>> input_images;
        for (auto* buffer : read_only) input_images.push_back(buffer->host);

        // Snapshot the read-only inputs BEFORE the launch: the download below
        // overwrites the same host buffers, so a comparison taken afterwards
        // would compare the device's bytes with themselves.
        std::vector<std::uint8_t> key_snapshot(paged_decode::kKeyCacheBytes);
        std::vector<std::uint8_t> value_snapshot(paged_decode::kValueCacheBytes);
        std::vector<float> scale_snapshot(paged_decode::kScaleCacheFloats);
        std::vector<std::uint16_t> qrot_snapshot(qrot_floats);
        std::memcpy(key_snapshot.data(), key_bytes, paged_decode::kKeyCacheBytes);
        std::memcpy(value_snapshot.data(), value_bytes, paged_decode::kValueCacheBytes);
        std::memcpy(scale_snapshot.data(), scale_floats,
                    paged_decode::kScaleCacheFloats * 4);
        std::memcpy(qrot_snapshot.data(), qrot_values, qrot_floats * 2);

        // --------------------------------------------------------- the launch
        // One block per item. In the isolated run that is a ONE-BLOCK grid
        // processing exactly item `head`, which is what puts the work on
        // core 0's first Vector core and makes the trace audit's "the audited
        // core did the work" check meaningful.
        const std::uint32_t item_begin = options.all ? 0u : options.head;
        const std::uint32_t item_count =
            options.all ? static_cast<std::uint32_t>(kItems) : 1u;
        const std::uint32_t block_dim = item_count;
        const auto launch = [&](auto launch_fn, auto... extra) {
            launch_fn(
            session.stream, block_dim, static_cast<std::uint16_t*>(q_rot.payload()),
            static_cast<std::int8_t*>(key_cache.payload()),
            static_cast<std::int8_t*>(value_cache.payload()),
            static_cast<float*>(scale_cache.payload()),
            static_cast<std::uint32_t*>(block_table.payload()),
            static_cast<float*>(gate.payload()),
            static_cast<float*>(pi_sign_buffer.payload()),
            static_cast<float*>(out_scores.payload()),
            static_cast<float*>(out_state.payload()),
            static_cast<float*>(out_acc.payload()),
            static_cast<float*>(out_rotated.payload()),
            static_cast<std::uint16_t*>(out_context.payload()),
            static_cast<std::int32_t*>(probe.payload()),
            static_cast<std::uint32_t>(kNumRequests),
            static_cast<std::uint32_t>(kKvHeads), options.rows,
            static_cast<std::uint32_t>(kHeadSize), static_cast<std::uint32_t>(kBlockSize),
            static_cast<std::uint32_t>(kCachePages), options.pages, options.strips,
            options.strip_begin, item_begin, item_count, extra...);
        };
        if (options.page_begin == 0u && !options.exact_boundary)
            launch(&launch_turboquant_single_token_decode_aiv);
        else
            launch(&launch_turboquant_single_token_decode_boundary_aiv, options.page_begin);
        camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

        for (auto* buffer : {&out_scores, &out_state, &out_acc, &out_rotated,
                             &out_context, &probe, &key_cache, &value_cache,
                             &scale_cache, &q_rot, &gate, &block_table, &pi_sign_buffer})
            buffer->download();

        // -------------------------------------------------- guard bands first
        for (auto* buffer : {&out_scores, &out_state, &out_acc, &out_rotated,
                             &out_context, &probe, &key_cache, &value_cache,
                             &scale_cache, &q_rot, &gate, &block_table, &pi_sign_buffer})
            buffer->assert_guards();
        std::printf("  guard bands: all 13 buffers intact, head and tail\n");

        for (std::size_t i = 0; i < read_only.size(); ++i)
            expect(read_only[i]->host == input_images[i], "ASSERT_CANARY_INTACT: all input bytes unchanged");

        // ---------------------------------------- the launch observation
        {
            const auto* words = static_cast<const std::int32_t*>(probe.host_payload());
            std::printf("  launch: GetBlockIdx %d, GetBlockNum %d, GetSubBlockNum %d, "
                        "itemCount %d (grid %u)\n",
                        words[0], words[8], words[16], words[24], block_dim);
            expect(words[8] == static_cast<std::int32_t>(block_dim),
                   "the kernel observes GetBlockNum equal to the launched grid");
            expect(words[16] == 1,
                   "under KERNEL_TYPE_AIV_ONLY each AIV is its own block "
                   "(GetSubBlockNum == 1)");
        }

        // ------------------------------------------- inputs are not written to
        {
            std::size_t touched = 0;
            const auto* key_now =
                static_cast<const std::uint8_t*>(key_cache.host_payload());
            const auto* value_now =
                static_cast<const std::uint8_t*>(value_cache.host_payload());
            const auto* scale_now = static_cast<const float*>(scale_cache.host_payload());
            const auto* qrot_now =
                static_cast<const std::uint16_t*>(q_rot.host_payload());
            for (std::size_t i = 0; i < paged_decode::kKeyCacheBytes; ++i)
                if (key_now[i] != key_snapshot[i]) ++touched;
            for (std::size_t i = 0; i < paged_decode::kValueCacheBytes; ++i)
                if (value_now[i] != value_snapshot[i]) ++touched;
            for (std::size_t i = 0; i < paged_decode::kScaleCacheFloats; ++i)
                if (bits_of(scale_now[i]) != bits_of(scale_snapshot[i])) ++touched;
            for (std::size_t i = 0; i < qrot_floats; ++i)
                if (qrot_now[i] != qrot_snapshot[i]) ++touched;
            expect(touched == 0, "the kernel wrote nothing to any input buffer");
            std::printf("  inputs: %zu word(s) changed across the caches, the query "
                        "panel and the scale plane\n",
                        touched);
        }

        // --------------------------------------------------- active-item checks
        const auto* score_out = static_cast<const float*>(out_scores.host_payload());
        const auto* state_out = static_cast<const float*>(out_state.host_payload());
        const auto* acc_out = static_cast<const float*>(out_acc.host_payload());
        const auto* rotated_out = static_cast<const float*>(out_rotated.host_payload());
        const auto* context_out =
            static_cast<const std::uint16_t*>(out_context.host_payload());

        std::size_t score_bad = 0;
        std::size_t m_bad = 0;
        std::uint64_t worst_l_ulp = 0;
        std::uint64_t worst_acc_ulp = 0;
        double worst_acc_abs = 0.0;
        double worst_rotated_abs = 0.0;
        double worst_context_abs = 0.0;
        double largest_acc = 0.0;
        double largest_rotated = 0.0;
        double largest_context = 0.0;

        const int first_item = options.all ? 0 : static_cast<int>(options.head);
        const int last_item = options.all ? kItems : static_cast<int>(options.head) + 1;

        for (int item = first_item; item < last_item; ++item) {
            const int head = item % kKvHeads;
            const GoldenState golden = options.exact_boundary ? decode_boundary::Oracle(window) : GoldenDecode(head, window);

            // 1. every raw score panel, bit-exactly
            for (std::size_t i = 0; i < window.size(); ++i) {
                const std::vector<float> raw = options.exact_boundary ?
                    std::vector<float>(kRows * kKeys, 3.0f) : GoldenRawPanel(head, window[i]);
                for (int row = 0; row < rows; ++row)
                    for (int t = 0; t < kKeys; ++t) {
                        const float got =
                            score_out[(static_cast<std::size_t>(item) * kMaxArchiveStrips +
                                       i) *
                                          kArchiveStride +
                                      static_cast<std::size_t>(row) * kKeys + t];
                        if (bits_of(got) !=
                            bits_of(raw[static_cast<std::size_t>(row) * kKeys + t]))
                            ++score_bad;
                    }
            }

            // 2. the carried state, out of lane 0 of each replicated block.
            //    m bit-exactly, l in ULP.
            for (int row = 0; row < rows; ++row) {
                const std::size_t base =
                    static_cast<std::size_t>(item) * 2 * kMaxRows * kStateLanes;
                const float m_got = state_out[base + row * kStateLanes];
                const float l_got =
                    state_out[base + (kMaxRows + row) * kStateLanes];
                if (bits_of(m_got) != bits_of(golden.m[row])) ++m_bad;
                worst_l_ulp = std::max(worst_l_ulp, ulp_distance(l_got, golden.l[row]));
            }

            // 3. the accumulator, still in the cache's rotated basis
            const std::size_t panel_base =
                static_cast<std::size_t>(item) * kMaxRows * kHeadSize;
            for (int row = 0; row < rows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t at = static_cast<std::size_t>(row) * kHeadSize + c;
                    worst_acc_ulp = std::max(
                        worst_acc_ulp,
                        ulp_distance(acc_out[panel_base + at], golden.acc[at]));
                    worst_acc_abs = std::max(
                        worst_acc_abs,
                        static_cast<double>(std::fabs(acc_out[panel_base + at] -
                                                      golden.acc[at])));
                    largest_acc = std::max(largest_acc,
                                           static_cast<double>(std::fabs(golden.acc[at])));
                }

            // 4. the egress rotation on its own, before the gate. Splitting it
            //    from the gate is what makes a failure actionable: the gate is
            //    elementwise and the rotation is eight strided stages, and
            //    their errors are indistinguishable in the gated output.
            for (int row = 0; row < rows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t at = static_cast<std::size_t>(row) * kHeadSize + c;
                    worst_rotated_abs = std::max(
                        worst_rotated_abs,
                        static_cast<double>(std::fabs(rotated_out[panel_base + at] -
                                                      golden.rotated[at])));
                    largest_rotated =
                        std::max(largest_rotated,
                                 static_cast<double>(std::fabs(golden.rotated[at])));
                }

            if (options.exact_boundary) {
                for (int row = 0; row < rows; ++row) {
                    const auto state_base = static_cast<std::size_t>(item) * 2 * kMaxRows * kStateLanes;
                    for (int lane = 0; lane < kStateLanes; ++lane) {
                        expect(bits_of(state_out[state_base + row * kStateLanes + lane]) == bits_of(golden.m[row]), "exact boundary replicated m");
                        expect(bits_of(state_out[state_base + (kMaxRows + row) * kStateLanes + lane]) == bits_of(golden.l[row]), "exact boundary replicated l");
                    }
                    for (int c = 0; c < kHeadSize; ++c) {
                        const auto at = row * kHeadSize + c;
                        expect(bits_of(acc_out[panel_base + at]) == bits_of(golden.acc[at]), "exact boundary acc");
                        expect(bits_of(rotated_out[panel_base + at]) == bits_of(golden.rotated[at]), "exact boundary dense rotation");
                        expect(context_out[panel_base + at] == float_to_bfloat16_rne(golden.context[at]), "exact boundary BF16 context");
                    }
                }
            }

            // 5. the gated bf16 context
            for (int row = 0; row < rows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    const std::size_t at = static_cast<std::size_t>(row) * kHeadSize + c;
                    const float want =
                        bfloat16_to_float(float_to_bfloat16_rne(golden.context[at]));
                    const float got = bfloat16_to_float(context_out[panel_base + at]);
                    worst_context_abs =
                        std::max(worst_context_abs,
                                 static_cast<double>(std::fabs(got - want)));
                    largest_context =
                        std::max(largest_context, static_cast<double>(std::fabs(want)));
                }
        }

        // A failing panel costs a long CAModel run to observe, so print enough
        // of it to tell the failure modes apart rather than only that it
        // failed: a transpose, a NaN from a canary read, a clamped-mask
        // partial dot product and a wrong page look nothing alike side by side.
        if (score_bad != 0) {
            const int head = first_item % kKvHeads;
            const std::vector<float> raw = GoldenRawPanel(head, window.front());
            const float* got = score_out + static_cast<std::size_t>(first_item) *
                                               kMaxArchiveStrips * kArchiveStride;
            std::fprintf(stderr, "  panel diagnosis, item %d strip %d:\n", first_item,
                         window.front());
            for (int row = 0; row < 2 && row < rows; ++row) {
                std::fprintf(stderr, "    r%-2d want:", row);
                for (int t = 0; t < 8; ++t)
                    std::fprintf(stderr, " %10.4f", raw[row * kKeys + t]);
                std::fprintf(stderr, "\n    r%-2d got :", row);
                for (int t = 0; t < 8; ++t)
                    std::fprintf(stderr, " %10.4f", got[row * kKeys + t]);
                std::fprintf(stderr, "\n");
            }
            std::size_t nan_count = 0;
            std::size_t canary_count = 0;
            std::size_t clamped_match = 0;
            std::size_t other_page = 0;
            // The clamped-mask form the roadmap's sketch would have produced,
            // and the panel of the WRONG page, scored side by side: whichever
            // the device matches names the bug.
            for (int row = 0; row < rows; ++row)
                for (int t = 0; t < kKeys; ++t) {
                    const float value = got[row * kKeys + t];
                    if (std::isnan(value)) ++nan_count;
                    if (bits_of(value) == kTileCanary) ++canary_count;
                    float partial = 0.0f;
                    for (int c = 0; c < 64; ++c)
                        partial += QRotEntry(head, row, c) *
                                   static_cast<float>(
                                       KeyNibbleAt(head, window.front(), t, c));
                    if (bits_of(value) == bits_of(partial)) ++clamped_match;
                }
            if (window.size() > static_cast<std::size_t>(kStripsPerPage)) {
                const std::vector<float> wrong =
                    GoldenRawPanel(head, window.front() % kStripsPerPage);
                for (int row = 0; row < rows; ++row)
                    for (int t = 0; t < kKeys; ++t)
                        if (bits_of(got[row * kKeys + t]) ==
                            bits_of(wrong[row * kKeys + t]))
                            ++other_page;
            }
            std::fprintf(stderr,
                         "    %zu NaN, %zu still canary, %zu match a 64-channel "
                         "(clamped-mask) partial sum, %zu match page 0's panel\n",
                         nan_count, canary_count, clamped_match, other_page);
        }

        expect(score_bad == 0, "every raw score panel is bit-exact");
        expect(m_bad == 0, "every row maximum m is bit-exact");
        std::printf("  scores:   %zu/%zu panel entries wrong (bit-exact check)\n",
                    score_bad,
                    static_cast<std::size_t>(last_item - first_item) * window.size() *
                        static_cast<std::size_t>(rows) * kKeys);

        // The ULP budgets are against the largest intermediate, not the final
        // value: `acc` is a cancelling sum of 16 exp-weighted terms per strip
        // and `Exp` is documented to permit 1 ULP per call. The relative
        // budgets follow guidelines section 5 for a cancelling reduction --
        // an absolute budget on the egress would be meaningless, because the
        // Hadamard cancels `acc` down to a small fraction of its intermediates.
        constexpr std::uint64_t kLUlpBudget = 64;
        // `acc` is gated RELATIVE to the largest value in the panel, not in
        // ULP, and the four-page rung is what settled that. A ULP gate on a
        // cancelling reduction measures the wrong thing: symmetric int4 values
        // make `acc` a signed sum, so individual channels land arbitrarily
        // close to zero, and the ULP distance at such a channel is enormous
        // while the absolute error there is negligible. At 32 strips the worst
        // entry reached 1280 ULP against a 256 ULP budget while every
        // downstream quantity built from `acc` -- the egress rotation and the
        // gated context -- sat three orders of magnitude INSIDE its own
        // relative budget, which is not what an accumulator fault looks like.
        // Guidelines section 5 asks for a budget against the largest
        // intermediate for exactly this shape of computation, and the two
        // egress checks below already follow it.
        //
        // The divisor is derived, not fitted. `acc` sums at most
        // 8 strips/page x 8 pages x 16 keys = 1024 exp-weighted products, each
        // carrying up to 1 ULP from `Exp` plus FP32 rounding, so the
        // accumulated error is of order 2^10 x 2^-24 = 2^-14 of the largest
        // term. 2^-12 of the largest magnitude leaves four times that.
        const double acc_budget = largest_acc / 4096.0 + 1.0e-7;
        const double rotated_budget = largest_rotated / 1024.0 + 1.0e-7;
        // bf16 carries 8 mantissa bits, so 2^-6 of the largest magnitude
        // leaves two bits of headroom for the Exp and the Div in front of it.
        const double context_budget = largest_context / 64.0 + 1.0e-7;

        expect(worst_l_ulp <= kLUlpBudget, "l within the ULP budget");
        expect(worst_acc_abs <= acc_budget, "acc within its relative budget");
        expect(worst_rotated_abs <= rotated_budget,
               "the egress rotation, before the gate, within its relative budget");
        expect(worst_context_abs <= context_budget,
               "the gated bf16 context within the relative budget");
        std::printf("  numerics: m bit-exact, l <= %llu ULP, acc err %.3e vs "
                    "budget %.3e (%llu ULP, largest |acc| %.6f)\n",
                    static_cast<unsigned long long>(worst_l_ulp), worst_acc_abs,
                    acc_budget, static_cast<unsigned long long>(worst_acc_ulp),
                    largest_acc);
        std::printf("  egress:   rotation err %.3e vs budget %.3e (largest |rot| "
                    "%.6f), context err %.3e vs budget %.3e (largest |ctx| %.6f)\n",
                    worst_rotated_abs, rotated_budget, largest_rotated,
                    worst_context_abs, context_budget, largest_context);

        if (options.exact_boundary) std::printf("  byte-exact boundary oracle checked (scores, m/l replicas, acc, rotation, BF16)\n");

        // ------------------------------------------------ the canary sweep
        CanarySweep sweep;
        const auto* score_raw =
            static_cast<const std::uint32_t*>(out_scores.host_payload());
        const auto* state_raw =
            static_cast<const std::uint32_t*>(out_state.host_payload());
        const auto* acc_raw = static_cast<const std::uint32_t*>(out_acc.host_payload());
        const auto* rotated_raw =
            static_cast<const std::uint32_t*>(out_rotated.host_payload());
        const auto* context_raw =
            static_cast<const std::uint16_t*>(out_context.host_payload());

        auto check_word = [&](std::uint32_t word) {
            ++sweep.checked;
            if (word != kTileCanary) ++sweep.mutated;
        };

        for (int item = 0; item < kItems; ++item) {
            const bool active = item >= first_item && item < last_item;
            // Unassigned items: every output word of theirs must be untouched.
            // Active items: the archive slots past the processed strips, and
            // the rows past `rows`, are the output guard band.
            for (int slot = 0; slot < kMaxArchiveStrips; ++slot)
                for (int entry = 0; entry < kArchiveStride; ++entry) {
                    const bool written =
                        active && slot < static_cast<int>(window.size()) &&
                        entry < rows * kKeys;
                    if (written) continue;
                    check_word(score_raw[(static_cast<std::size_t>(item) *
                                              kMaxArchiveStrips +
                                          slot) *
                                             kArchiveStride +
                                         entry]);
                }
            for (int half = 0; half < 2; ++half)
                for (int row = 0; row < kMaxRows; ++row)
                    for (int lane = 0; lane < kStateLanes; ++lane) {
                        if (active && row < rows) continue;
                        check_word(state_raw[((static_cast<std::size_t>(item) * 2 + half) *
                                                  kMaxRows +
                                              row) *
                                                 kStateLanes +
                                             lane]);
                    }
            for (int row = 0; row < kMaxRows; ++row)
                for (int c = 0; c < kHeadSize; ++c) {
                    if (active && row < rows) continue;
                    const std::size_t at =
                        (static_cast<std::size_t>(item) * kMaxRows + row) * kHeadSize + c;
                    check_word(acc_raw[at]);
                    check_word(rotated_raw[at]);
                    // bf16: two canary halves per 32-bit canary word.
                    ++sweep.checked;
                    if (context_raw[at] != static_cast<std::uint16_t>(kTileCanary >> (16u * (at % 2u))))
                        ++sweep.mutated;
                }
        }
        expect(sweep.mutated == 0,
               "no unassigned item slot and no output guard band word was mutated");
        std::printf("  isolation: %zu/%zu unassigned and guard-band output words "
                    "still canary\n",
                    sweep.checked - sweep.mutated, sweep.checked);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what());
        return 1;
    }

    if (failures != 0) {
        std::fprintf(stderr, "FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS single-token decode: %u row(s), %u page(s), %u strip(s)",
                options.rows, options.pages, options.strips);
    if (options.all)
        std::printf(", all %d items\n", kItems);
    else
        std::printf(", item %u isolated\n", options.head);
    return 0;
}
