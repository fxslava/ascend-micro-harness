// Exhaustive host check of the decode tail geometry
// (kernels/ops/turboquant/turboquant_decode_tail.hpp).
//
// What this establishes, and why it is worth a test of its own
// -----------------------------------------------------------
// The masking of a partial strip is device code that no host can run. What CAN
// be settled here is the arithmetic that decides WHICH tokens are live: the
// trip count a kernel walks, the active lane count it masks to, and the
// agreement between those two and the oracle. Every one of those is a pure
// function of five integers, so they can be checked against a brute-force
// definition over a dense sweep rather than spot-checked -- and if they are
// wrong, no amount of correct masking saves the kernel.
//
// Three properties, in increasing strength:
//
//   1. `ActiveInStrip` equals a per-token count, over every (base, seq_len)
//      pair in a dense square. A clamp expression and a counting loop are
//      different enough formulations that agreeing on 4.2 million pairs is
//      evidence rather than restatement.
//
//   2. THE PARTITION. Walking the window and marking every token each strip
//      declares active must mark exactly the live tokens the window spans,
//      each exactly once. This is the property the kernel's correctness rests
//      on: no live token skipped (it would be missing from the softmax), none
//      visited twice (it would be counted twice in the denominator), and
//      nothing past `seq_len` touched at all. A bitmap comparison catches all
//      three, and none of them is visible from a trip count alone.
//
//   3. THE PREFIX CLAIM. `LiveStripsInWindow` stops at the first dead strip,
//      so it is a trip count only if the live strips are the walk's prefix.
//      That is checked here against a scan of the WHOLE window, so a geometry
//      where it failed would be caught rather than assumed.
//
// Four negative controls follow, each a plausible wrong implementation. They
// must be REJECTED. Without them a green run says only that the checks ran.

#include "../../kernels/ops/turboquant/turboquant_decode_tail.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace tail = turboquant_decode_tail;

namespace {

int failures = 0;

void fail(const char* what, long long a, long long b)
{
    if (failures < 10)
        std::printf("FAIL %s: %lld != %lld\n", what, a, b);
    ++failures;
}

// ---------------------------------------------------------------- reference
// Independent of the clamp: count the tokens of this strip that are live.
std::uint32_t BruteActive(std::uint64_t base, std::uint32_t seq_len)
{
    std::uint32_t live = 0;
    for (std::uint32_t t = 0; t < tail::kStripTokens; ++t)
        if (base + t < static_cast<std::uint64_t>(seq_len)) ++live;
    return live;
}

// ------------------------------------------------------------------ test 1
long long CheckActiveInStrip()
{
    constexpr std::uint32_t kSpan = 2048;
    long long pairs = 0;
    for (std::uint32_t seq_len = 0; seq_len <= kSpan; ++seq_len)
        for (std::uint64_t base = 0; base <= kSpan; ++base) {
            const std::uint32_t got = tail::ActiveInStrip(base, seq_len);
            const std::uint32_t want = BruteActive(base, seq_len);
            if (got != want) fail("ActiveInStrip", got, want);
            // The three predicates must partition every strip.
            const bool dead = tail::StripIsDead(base, seq_len);
            const bool full = tail::StripIsFull(base, seq_len);
            const bool part = tail::StripIsPartial(base, seq_len);
            if (static_cast<int>(dead) + static_cast<int>(full) +
                    static_cast<int>(part) != 1)
                fail("strip is not exactly one of dead/full/partial",
                     static_cast<int>(dead) * 100 + static_cast<int>(full) * 10 +
                         static_cast<int>(part),
                     1);
            ++pairs;
        }
    return pairs;
}

// ------------------------------------------------------------ tests 2 and 3
struct Window {
    std::uint32_t page_begin, seq_pages, block_size, strip_begin, strips;
};

// Mark the tokens the walk declares active, and the tokens that are actually
// live inside the window, then compare. Returns false on any disagreement.
bool CheckPartition(const Window& w, std::uint32_t seq_len)
{
    const std::uint64_t capacity =
        static_cast<std::uint64_t>(w.page_begin + w.seq_pages) * w.block_size;
    std::vector<int> walked(capacity + tail::kStripTokens, 0);
    std::vector<int> live(capacity + tail::kStripTokens, 0);

    // What the walk says, through the shared functions the kernel uses.
    const std::uint32_t count = tail::LiveStripsInWindow(
        seq_len, w.page_begin, w.seq_pages, w.block_size, w.strip_begin, w.strips);
    for (std::uint32_t index = 0; index < count; ++index) {
        const std::uint64_t base = tail::WalkTokenBase(
            index, w.strips, w.page_begin, w.strip_begin, w.block_size);
        const std::uint32_t active = tail::WalkActive(
            index, w.strips, w.page_begin, w.strip_begin, w.block_size, seq_len);
        if (active == 0u) {
            fail("a strip inside the trip count is dead", index, count);
            return false;
        }
        for (std::uint32_t t = 0; t < active; ++t) ++walked[base + t];
    }

    // What is actually live inside the window, enumerated directly.
    for (std::uint32_t p = 0; p < w.seq_pages; ++p)
        for (std::uint32_t s = 0; s < w.strips; ++s) {
            const std::uint64_t base =
                static_cast<std::uint64_t>(w.page_begin + p) * w.block_size +
                static_cast<std::uint64_t>(w.strip_begin + s) * tail::kStripTokens;
            for (std::uint32_t t = 0; t < tail::kStripTokens; ++t)
                if (base + t < static_cast<std::uint64_t>(seq_len)) ++live[base + t];
        }

    for (std::size_t token = 0; token < walked.size(); ++token) {
        if (walked[token] > 1) {
            fail("token visited more than once", static_cast<long long>(token),
                 walked[token]);
            return false;
        }
        if (walked[token] != live[token]) {
            fail("walked set differs from the live set at token",
                 static_cast<long long>(token), walked[token] - live[token]);
            return false;
        }
    }

    // The prefix claim: scanning the WHOLE window must find live strips only
    // below the trip count, and every index below it live.
    const std::uint64_t whole =
        static_cast<std::uint64_t>(w.seq_pages) * w.strips;
    for (std::uint64_t index = 0; index < whole; ++index) {
        const bool active =
            tail::WalkActive(static_cast<std::uint32_t>(index), w.strips,
                             w.page_begin, w.strip_begin, w.block_size, seq_len) != 0u;
        if (active != (index < count)) {
            fail("live strips are not the walk's prefix",
                 static_cast<long long>(index), count);
            return false;
        }
    }
    return true;
}

long long CheckWindows()
{
    long long cases = 0;
    const std::uint32_t block_sizes[] = {16, 32, 128};
    for (const std::uint32_t block_size : block_sizes) {
        const std::uint32_t strips_in_page = block_size / tail::kStripTokens;
        for (std::uint32_t strip_begin = 0; strip_begin < strips_in_page; ++strip_begin)
            for (std::uint32_t strips = 1; strip_begin + strips <= strips_in_page;
                 ++strips) {
                if (!tail::WalkIsMonotone(strip_begin, strips, block_size)) {
                    fail("a window inside the page must be monotone", strips,
                         strips_in_page);
                    continue;
                }
                for (std::uint32_t page_begin = 0; page_begin <= 2; ++page_begin)
                    for (std::uint32_t seq_pages = 1; seq_pages <= 8; ++seq_pages) {
                        const std::uint32_t span =
                            (page_begin + seq_pages) * block_size + tail::kStripTokens;
                        const Window w{page_begin, seq_pages, block_size, strip_begin,
                                       strips};
                        for (std::uint32_t seq_len = 0; seq_len <= span; ++seq_len) {
                            if (!CheckPartition(w, seq_len)) return cases;
                            ++cases;
                        }
                    }
            }
    }
    return cases;
}

// ------------------------------------------------------------------ test 4
// The back-compatibility property the aligned entry points rest on.
//
// Every caller from before the tail change meant "the whole window is live",
// and both kernels spell that default as `WholeWindowTokens`. If that value
// did not walk exactly `seq_pages * strips` strips, the aligned callers --
// including the frozen CAModel ladders -- would silently change trip count,
// which is the one regression this change must not have. Checked over every
// window geometry the operators admit, not argued from the formula.
long long CheckAlignedDefault()
{
    long long checked = 0;
    const std::uint32_t block_sizes[] = {16, 32, 128, 256};
    for (const std::uint32_t block_size : block_sizes) {
        const std::uint32_t strips_in_page = block_size / tail::kStripTokens;
        for (std::uint32_t strip_begin = 0; strip_begin < strips_in_page; ++strip_begin)
            for (std::uint32_t strips = 1; strip_begin + strips <= strips_in_page;
                 ++strips)
                for (std::uint32_t page_begin = 0; page_begin <= 3; ++page_begin)
                    for (std::uint32_t seq_pages = 1; seq_pages <= 16; ++seq_pages) {
                        const auto whole = static_cast<std::uint32_t>(
                            tail::WholeWindowTokens(page_begin, seq_pages, block_size,
                                                    strip_begin, strips));
                        const std::uint32_t live = tail::LiveStripsInWindow(
                            whole, page_begin, seq_pages, block_size, strip_begin,
                            strips);
                        if (live != seq_pages * strips)
                            fail("the aligned default changed the trip count", live,
                                 seq_pages * strips);
                        // And the last strip of such a window is full, so the
                        // masking phase is never entered on an aligned launch.
                        const std::uint32_t last = tail::WalkActive(
                            seq_pages * strips - 1u, strips, page_begin, strip_begin,
                            block_size, whole);
                        if (last != tail::kStripTokens)
                            fail("an aligned window ends on a partial strip", last,
                                 tail::kStripTokens);
                        ++checked;
                    }
    }
    return checked;
}

// ------------------------------------------------------- negative controls
// Each is a plausible wrong implementation. The property named must reject it,
// or the corresponding check above proves nothing.

// The pre-fix kernel: every strip is full.
std::uint32_t BadActiveAlwaysFull(std::uint64_t, std::uint32_t) { return 16u; }

// Rounds the partial strip up instead of masking it.
std::uint32_t BadActiveRoundUp(std::uint64_t base, std::uint32_t seq_len)
{
    const std::uint32_t active = tail::ActiveInStrip(base, seq_len);
    return active == 0u ? 0u : 16u;
}

// Drops the partial strip: the other way to be wrong, and the one a truncating
// `seq_len / 16` produces.
std::uint32_t BadLiveStripsTruncate(std::uint32_t seq_len, const Window& w)
{
    const std::uint64_t start =
        static_cast<std::uint64_t>(w.page_begin) * w.block_size +
        static_cast<std::uint64_t>(w.strip_begin) * tail::kStripTokens;
    if (static_cast<std::uint64_t>(seq_len) <= start) return 0u;
    return static_cast<std::uint32_t>((seq_len - start) / tail::kStripTokens);
}

// The rounding form that overflows near the top of the range.
std::uint32_t BadCeilDiv(std::uint32_t value, std::uint32_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

int RunControls()
{
    int rejected = 0;

    // Control 1 and 2: wrong active counts, caught by the dense sweep.
    bool caught_full = false, caught_round = false;
    for (std::uint32_t seq_len = 0; seq_len <= 64 && !(caught_full && caught_round);
         ++seq_len)
        for (std::uint64_t base = 0; base <= 64; ++base) {
            const std::uint32_t want = BruteActive(base, seq_len);
            if (BadActiveAlwaysFull(base, seq_len) != want) caught_full = true;
            if (BadActiveRoundUp(base, seq_len) != want) caught_round = true;
        }
    if (caught_full) {
        std::printf("PASS rejected negative control: every strip full -- the "
                    "pre-fix behaviour, which admits tokens past seq_len\n");
        ++rejected;
    } else {
        std::printf("FAIL negative control not rejected: every strip full\n");
        ++failures;
    }
    if (caught_round) {
        std::printf("PASS rejected negative control: partial strip rounded up "
                    "to 16 active lanes\n");
        ++rejected;
    } else {
        std::printf("FAIL negative control not rejected: partial rounded up\n");
        ++failures;
    }

    // Control 3: a truncating trip count, caught by the partition.
    const Window w{0u, 8u, 128u, 0u, 8u};
    bool caught_truncate = false;
    for (std::uint32_t seq_len = 1; seq_len <= 1024; ++seq_len)
        if (BadLiveStripsTruncate(seq_len, w) !=
            tail::LiveStripsInWindow(seq_len, w.page_begin, w.seq_pages, w.block_size,
                                     w.strip_begin, w.strips)) {
            caught_truncate = true;
            break;
        }
    if (caught_truncate) {
        std::printf("PASS rejected negative control: trip count truncated, "
                    "dropping the partial strip entirely\n");
        ++rejected;
    } else {
        std::printf("FAIL negative control not rejected: truncated trip count\n");
        ++failures;
    }

    // Control 4: the overflowing round-up.
    if (BadCeilDiv(0xFFFFFFFFu, 16u) != tail::StripsForTokens(0xFFFFFFFFu)) {
        std::printf("PASS rejected negative control: (v + d - 1) / d overflows "
                    "at the top of the range, giving %u strips for 2^32-1 "
                    "tokens instead of %u\n",
                    BadCeilDiv(0xFFFFFFFFu, 16u), tail::StripsForTokens(0xFFFFFFFFu));
        ++rejected;
    } else {
        std::printf("FAIL negative control not rejected: overflowing round-up\n");
        ++failures;
    }
    return rejected;
}

// ----------------------------------------------------------- the directive
// The ladder the validation harness launches, printed so the record shows what
// each rung asks of the kernel.
void PrintLadder()
{
    const std::uint32_t ladder[] = {1, 7, 15, 17, 31, 33, 127, 129, 255, 257, 1023};
    constexpr std::uint32_t kBlock = 128;
    std::printf("  %-6s %-7s %-7s %-8s %-7s %s\n", "S", "strips", "pages",
                "last", "tail", "last page");
    for (const std::uint32_t seq_len : ladder) {
        const std::uint32_t strips = tail::StripsForTokens(seq_len);
        const std::uint32_t pages = tail::PagesForTokens(seq_len, kBlock);
        const std::uint64_t last_base =
            static_cast<std::uint64_t>(strips - 1) * tail::kStripTokens;
        const std::uint32_t active = tail::ActiveInStrip(last_base, seq_len);
        std::printf("  %-6u %-7u %-7u %-8llu %-7u %u\n", seq_len, strips, pages,
                    static_cast<unsigned long long>(last_base), active,
                    tail::ActiveInPage(pages - 1, kBlock, seq_len));
    }
}

}  // namespace

int main()
{
    std::printf("=== decode tail geometry ===\n");
    PrintLadder();

    const long long pairs = CheckActiveInStrip();
    const long long cases = CheckWindows();
    const long long aligned = CheckAlignedDefault();
    const int rejected = RunControls();

    if (failures != 0) {
        std::printf("FAIL decode tail geometry: %d disagreements\n", failures);
        return 1;
    }
    std::printf("PASS decode tail geometry: %lld (base, seq_len) pairs agree with "
                "a per-token count, %lld windows partition their live tokens "
                "exactly once with the live strips forming the walk's prefix, "
                "%lld window geometries keep their pre-tail trip count under the "
                "aligned default, and %d negative controls are rejected\n",
                pairs, cases, aligned, rejected);
    return 0;
}
