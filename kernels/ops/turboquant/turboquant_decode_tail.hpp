#pragma once

// Tail geometry for paged decode at an ARBITRARY context length.
// Ascend 950PR / arch35 / dav-3510 / CANN 9.2.
//
// Why this header exists
// ----------------------
// In generative decoding the live token count S grows by one per step, so
// `S % 16 != 0` and `S % block_size != 0` are the common case, not the corner.
// Every decode kernel here walks whole 16-token strips of whole pages and
// derives its trip count from `seq_pages * strips`, which silently rounds the
// context UP to the next strip. The rounded-in tokens are cache slots that
// were never written for this sequence: in production they hold a previous
// sequence's keys, and under the isolated-canary protocol the harness uses
// they hold 0x7FC0BEEF, a quiet NaN. Either way they entered the softmax.
//
// This header is the arithmetic that separates allocated CAPACITY from LIVE
// tokens. It is pure C++17 over <cstdint>, with no AscendC and no kernel
// headers, for the same two reasons `turboquant_wave_protocol.hpp` is:
//
//   * the host launcher, the CPU oracle and the device kernel must agree on
//     which tokens are live, and the only way to guarantee that is one
//     definition compiled into all three;
//   * a host translation unit can then compile it, which is what makes the
//     static_asserts at the bottom and the exhaustive host test real evidence
//     rather than a claim about code no one can run.
//
// Every function is marked with TQ_TAIL_FN for the reason recorded in
// `turboquant_wave_protocol.hpp`: on this toolchain a function in a header
// shared between host and device TUs needs a CORE-TYPE attribute, not merely
// vague linkage, or the device link fails with an undefined symbol. C++17
// `constexpr` already implies `inline`, so linkage was never what was missing.
// Include this header AFTER `kernel_operator.h` in a device TU, so that
// `__aicore__` is defined when the macro is expanded.

#include <cstdint>

#ifndef TQ_TAIL_FN
#  if defined(__CCE_AICORE__)
#    if defined(__aicore__)
#      define TQ_TAIL_FN __aicore__ inline
#    else
#      define TQ_TAIL_FN [aicore] inline
#    endif
#  else
#    define TQ_TAIL_FN inline
#  endif
#endif

namespace turboquant_decode_tail {

// A strip is 16 tokens, everywhere in this operator family.
constexpr std::uint32_t kStripTokens = 16u;

// The score a masked lane carries into the online softmax.
//
// Large negative rather than -inf, matching what the kernels already seed
// `m_prev` with: the first strip's `Max` selects the tile maximum from the
// score row itself, so a true -inf would make `m_new - m_new` an invalid
// subtraction on an all-masked row. It is low enough that `exp(x - m)`
// underflows to +0 for any m a real score can take on this operator -- scores
// are bounded by 256 * 7 * max|scale| -- which is the property the mask rests
// on: a masked lane contributes exactly +0 to the denominator l and exactly +0
// to every channel of the accumulator Acc.
//
// PRECONDITION, and the reason `LiveStripsInWindow` exists: a strip with ZERO
// active tokens must never reach the masking phase. With every lane masked,
// `m_new` is kMaskedScore, the centred row is exactly 0, `exp(0)` is 1, and
// the strip would contribute 16 to l instead of 0 -- the one input that turns
// this mask into a wrong answer rather than a no-op. Dead strips are dropped
// from the trip count instead of being masked.
constexpr float kMaskedScore = -3.0e38f;

// Rounding up without the `(value + divisor - 1)` form, which overflows for
// values near the top of the range. A zero divisor yields 0 rather than
// trapping, so a caller that has not validated its geometry gets an empty
// walk instead of undefined behaviour.
TQ_TAIL_FN constexpr std::uint32_t CeilDiv(std::uint32_t value, std::uint32_t divisor)
{
    return divisor == 0u ? 0u
                         : value / divisor + (value % divisor != 0u ? 1u : 0u);
}

// Strips, and logical pages, needed to hold `seq_len` live tokens.
TQ_TAIL_FN constexpr std::uint32_t StripsForTokens(std::uint32_t seq_len)
{
    return CeilDiv(seq_len, kStripTokens);
}

TQ_TAIL_FN constexpr std::uint32_t PagesForTokens(std::uint32_t seq_len,
                                                  std::uint32_t block_size)
{
    return CeilDiv(seq_len, block_size);
}

// The logical token index at which (logical page, strip within page) begins.
// 64-bit because `page * block_size` is a cache-wide product and the callers
// already address global memory in 64 bits.
TQ_TAIL_FN constexpr std::uint64_t StripTokenBase(std::uint32_t page,
                                                  std::uint32_t block_size,
                                                  std::uint32_t strip)
{
    return static_cast<std::uint64_t>(page) * block_size +
           static_cast<std::uint64_t>(strip) * kStripTokens;
}

// Live tokens in the strip beginning at `token_base`, in 0..16. This is the
// lane count the masking phase keeps; the rest take kMaskedScore.
TQ_TAIL_FN constexpr std::uint32_t ActiveInStrip(std::uint64_t token_base,
                                                 std::uint32_t seq_len)
{
    const std::uint64_t live = static_cast<std::uint64_t>(seq_len);
    return token_base >= live
               ? 0u
               : (live - token_base >= kStripTokens
                      ? kStripTokens
                      : static_cast<std::uint32_t>(live - token_base));
}

TQ_TAIL_FN constexpr bool StripIsDead(std::uint64_t token_base,
                                      std::uint32_t seq_len)
{
    return ActiveInStrip(token_base, seq_len) == 0u;
}

TQ_TAIL_FN constexpr bool StripIsFull(std::uint64_t token_base,
                                      std::uint32_t seq_len)
{
    return ActiveInStrip(token_base, seq_len) == kStripTokens;
}

// The only strip that needs the mask: some tokens live, some not.
TQ_TAIL_FN constexpr bool StripIsPartial(std::uint64_t token_base,
                                         std::uint32_t seq_len)
{
    return ActiveInStrip(token_base, seq_len) != 0u &&
           ActiveInStrip(token_base, seq_len) != kStripTokens;
}

// Live tokens in one logical page, in 0..block_size.
TQ_TAIL_FN constexpr std::uint32_t ActiveInPage(std::uint32_t page,
                                                std::uint32_t block_size,
                                                std::uint32_t seq_len)
{
    const std::uint64_t base = static_cast<std::uint64_t>(page) * block_size;
    const std::uint64_t live = static_cast<std::uint64_t>(seq_len);
    return base >= live ? 0u
                        : (live - base >= block_size
                               ? block_size
                               : static_cast<std::uint32_t>(live - base));
}

// ---------------------------------------------------------------- the walk
// The kernels walk their window with one running index:
//
//   logical page  = page_begin  + index / strips
//   strip in page = strip_begin + index % strips
//
// so these functions are that walk, shared rather than restated at each call
// site. `WalkTokenBase` is monotonically increasing in `index` whenever
// `strip_begin + strips <= block_size / 16`, which every kernel here already
// refuses to run without -- and that monotonicity is what makes the live set a
// PREFIX of the walk, which is what lets a single trip count express the tail.
TQ_TAIL_FN constexpr std::uint32_t WalkPage(std::uint32_t index, std::uint32_t strips,
                                            std::uint32_t page_begin)
{
    return strips == 0u ? page_begin : page_begin + index / strips;
}

TQ_TAIL_FN constexpr std::uint32_t WalkStrip(std::uint32_t index, std::uint32_t strips,
                                             std::uint32_t strip_begin)
{
    return strips == 0u ? strip_begin : strip_begin + index % strips;
}

TQ_TAIL_FN constexpr std::uint64_t WalkTokenBase(std::uint32_t index,
                                                 std::uint32_t strips,
                                                 std::uint32_t page_begin,
                                                 std::uint32_t strip_begin,
                                                 std::uint32_t block_size)
{
    return StripTokenBase(WalkPage(index, strips, page_begin), block_size,
                          WalkStrip(index, strips, strip_begin));
}

TQ_TAIL_FN constexpr std::uint32_t WalkActive(std::uint32_t index, std::uint32_t strips,
                                              std::uint32_t page_begin,
                                              std::uint32_t strip_begin,
                                              std::uint32_t block_size,
                                              std::uint32_t seq_len)
{
    return ActiveInStrip(
        WalkTokenBase(index, strips, page_begin, strip_begin, block_size), seq_len);
}

// The token index one past the last the window can hold: the `seq_len` a
// caller with no tail means. Passing this is exactly the pre-tail behaviour,
// which is how the aligned entry points stay unchanged.
TQ_TAIL_FN constexpr std::uint64_t WholeWindowTokens(std::uint32_t page_begin,
                                                     std::uint32_t seq_pages,
                                                     std::uint32_t block_size,
                                                     std::uint32_t strip_begin,
                                                     std::uint32_t strips)
{
    return seq_pages == 0u || strips == 0u
               ? 0u
               : StripTokenBase(page_begin + seq_pages - 1u, block_size,
                                strip_begin + strips);
}

// Is the walk's token base strictly increasing in `index`?
//
// Load-bearing, and not only for the helper below. The kernels take
// `LiveStripsInWindow` as a TRIP COUNT and walk indices 0..count-1, which is
// correct only if the live strips are exactly the walk's prefix -- and that
// follows from monotonicity alone: once a strip starts at or past `seq_len`,
// every later one does too.
//
// Inside a page the base rises by 16 per index, always. Across a page boundary
// it rises by `block_size - (strips - 1) * 16`, which is positive exactly when
// `strips <= block_size / 16`. That is implied by the window check every
// kernel here already performs (`strip_begin + strips <= block_size / 16`), so
// this predicate is not a new restriction -- it is the existing one, named, so
// that the launchers can refuse a geometry that would make the trip count a
// silent truncation rather than an error.
TQ_TAIL_FN constexpr bool WalkIsMonotone(std::uint32_t strip_begin,
                                         std::uint32_t strips,
                                         std::uint32_t block_size)
{
    return strips != 0u && block_size != 0u && block_size % kStripTokens == 0u &&
           strip_begin + strips <= block_size / kStripTokens;
}

// Strips of the window that hold at least one live token, in walk order.
//
// REQUIRES `WalkIsMonotone(strip_begin, strips, block_size)`; the early exit
// below is that assumption made explicit, and without it this would stop at a
// dead strip that later live ones follow.
//
// A loop, run once per launch on the scalar unit, rather than a closed form:
// the live set is a prefix, so this stops at the first dead strip, and being a
// plain loop it is `constexpr`-evaluable, which is what lets the static_asserts
// below and the host test check it exhaustively instead of spot-checking an
// algebraic rearrangement. The bound is 64-bit because `seq_pages * strips` is
// a product of two runtime operands.
TQ_TAIL_FN constexpr std::uint32_t LiveStripsInWindow(std::uint32_t seq_len,
                                                      std::uint32_t page_begin,
                                                      std::uint32_t seq_pages,
                                                      std::uint32_t block_size,
                                                      std::uint32_t strip_begin,
                                                      std::uint32_t strips)
{
    const std::uint64_t count =
        static_cast<std::uint64_t>(seq_pages) * static_cast<std::uint64_t>(strips);
    std::uint32_t live = 0u;
    for (std::uint64_t index = 0; index < count; ++index) {
        if (WalkActive(static_cast<std::uint32_t>(index), strips, page_begin,
                       strip_begin, block_size, seq_len) == 0u)
            break;
        ++live;
    }
    return live;
}

// ------------------------------------------------------- caller validation
// `seq_len` describes the sequence, and the window describes what this launch
// was told to process. These are the two ways they can disagree, and both are
// silent-wrong-answer shaped rather than crash shaped, which is why the
// launchers refuse them instead of clamping.

// Live tokens past the end of the window: the launch would drop them and
// return a context computed from a truncated sequence.
TQ_TAIL_FN constexpr bool WindowCoversSeqLen(std::uint32_t seq_len,
                                             std::uint32_t page_begin,
                                             std::uint32_t seq_pages,
                                             std::uint32_t block_size,
                                             std::uint32_t strip_begin,
                                             std::uint32_t strips)
{
    return static_cast<std::uint64_t>(seq_len) <=
           WholeWindowTokens(page_begin, seq_pages, block_size, strip_begin, strips);
}

// More live tokens than the cache was allocated to hold.
TQ_TAIL_FN constexpr bool SeqLenInCapacity(std::uint32_t seq_len,
                                           std::uint32_t cache_pages,
                                           std::uint32_t block_size)
{
    return static_cast<std::uint64_t>(seq_len) <=
           static_cast<std::uint64_t>(cache_pages) * block_size;
}

// ------------------------------------------------------------ the contract
// Compile-time, so a build that includes this header has checked them.

static_assert(CeilDiv(0u, 16u) == 0u, "no tokens, no strips");
static_assert(CeilDiv(1u, 16u) == 1u && CeilDiv(16u, 16u) == 1u &&
                  CeilDiv(17u, 16u) == 2u,
              "CeilDiv must round up");
static_assert(CeilDiv(5u, 0u) == 0u, "a zero divisor must not trap");
// The overflow the `(value + divisor - 1)` form gets wrong.
static_assert(StripsForTokens(0xFFFFFFFFu) == 268435456u,
              "rounding up must not overflow at the top of the range");

// The directive's ladder, as strip and page counts.
static_assert(StripsForTokens(1u) == 1u && StripsForTokens(7u) == 1u &&
                  StripsForTokens(15u) == 1u && StripsForTokens(17u) == 2u &&
                  StripsForTokens(31u) == 2u && StripsForTokens(33u) == 3u &&
                  StripsForTokens(127u) == 8u && StripsForTokens(129u) == 9u &&
                  StripsForTokens(255u) == 16u && StripsForTokens(257u) == 17u &&
                  StripsForTokens(1023u) == 64u,
              "the unaligned ladder's strip counts");
static_assert(PagesForTokens(1u, 128u) == 1u && PagesForTokens(127u, 128u) == 1u &&
                  PagesForTokens(128u, 128u) == 1u &&
                  PagesForTokens(129u, 128u) == 2u &&
                  PagesForTokens(255u, 128u) == 2u &&
                  PagesForTokens(257u, 128u) == 3u &&
                  PagesForTokens(1023u, 128u) == 8u,
              "the unaligned ladder's page counts");

// One live token in the whole context: one strip, one active lane, fifteen
// masked. The rung the directive singles out.
static_assert(ActiveInStrip(0u, 1u) == 1u, "S = 1 leaves one active lane");
static_assert(StripIsPartial(0u, 1u), "S = 1 is a partial strip");
static_assert(LiveStripsInWindow(1u, 0u, 1u, 128u, 0u, 8u) == 1u,
              "S = 1 walks exactly one strip of an eight-strip window");

// A strip is full, partial or dead, and nothing else.
static_assert(ActiveInStrip(0u, 16u) == 16u && StripIsFull(0u, 16u) &&
                  !StripIsPartial(0u, 16u),
              "an exactly-filled strip is full, not partial");
static_assert(ActiveInStrip(16u, 17u) == 1u && StripIsPartial(16u, 17u),
              "S = 17 leaves one active lane in the second strip");
static_assert(ActiveInStrip(16u, 16u) == 0u && StripIsDead(16u, 16u) &&
                  !StripIsPartial(16u, 16u),
              "the strip after the last live token is dead, not partial");
static_assert(ActiveInStrip(0u, 0u) == 0u &&
                  LiveStripsInWindow(0u, 0u, 8u, 128u, 0u, 8u) == 0u,
              "a zero-length context has no live strips at all -- the "
              "all-masked strip the masking phase must never see");

// The last strip of each unaligned rung, at 128-token pages.
static_assert(ActiveInStrip(112u, 127u) == 15u, "S = 127: fifteen lanes in strip 7");
static_assert(ActiveInStrip(128u, 129u) == 1u, "S = 129: one lane in strip 8");
static_assert(ActiveInStrip(240u, 255u) == 15u, "S = 255: fifteen lanes in strip 15");
static_assert(ActiveInStrip(256u, 257u) == 1u, "S = 257: one lane in strip 16");
static_assert(ActiveInStrip(1008u, 1023u) == 15u, "S = 1023: fifteen in strip 63");

// The last page of each rung: capacity versus live.
static_assert(ActiveInPage(0u, 128u, 1u) == 1u &&
                  ActiveInPage(0u, 128u, 127u) == 127u &&
                  ActiveInPage(1u, 128u, 129u) == 1u &&
                  ActiveInPage(2u, 128u, 257u) == 1u &&
                  ActiveInPage(7u, 128u, 1023u) == 127u,
              "the last page holds only the live remainder");
static_assert(ActiveInPage(1u, 128u, 128u) == 0u,
              "a page past the last live token is entirely dead");

// The walk, over the whole 1023-token ladder rung: 8 pages x 8 strips of 16 is
// 1024 capacity, 64 live strips, the last one short by a token.
static_assert(LiveStripsInWindow(1023u, 0u, 8u, 128u, 0u, 8u) == 64u,
              "S = 1023 walks all 64 strips, the last one partial");
static_assert(WalkActive(63u, 8u, 0u, 0u, 128u, 1023u) == 15u,
              "and the 64th strip keeps fifteen lanes");
static_assert(LiveStripsInWindow(1024u, 0u, 8u, 128u, 0u, 8u) == 64u &&
                  WalkActive(63u, 8u, 0u, 0u, 128u, 1024u) == 16u,
              "the aligned neighbour walks the same strips with no mask");
static_assert(LiveStripsInWindow(129u, 0u, 8u, 128u, 0u, 8u) == 9u,
              "S = 129 stops after the ninth strip");

// The walk through a window that does not start at the origin. The live set is
// still a prefix OF THE WALK, which is the property the trip count needs.
static_assert(WalkTokenBase(0u, 8u, 1u, 0u, 128u) == 128u,
              "page_begin shifts the walk's first token");
static_assert(LiveStripsInWindow(129u, 1u, 1u, 128u, 0u, 8u) == 1u,
              "from page 1, S = 129 leaves one live strip");
static_assert(LiveStripsInWindow(129u, 1u, 1u, 128u, 2u, 4u) == 0u,
              "and none at all if the window starts past the live tail");
static_assert(WalkTokenBase(1u, 4u, 0u, 2u, 128u) == 48u,
              "strip_begin shifts within the page");

// Window and capacity validation.
static_assert(WholeWindowTokens(0u, 8u, 128u, 0u, 8u) == 1024u,
              "eight whole pages of eight strips span 1024 tokens");
static_assert(WindowCoversSeqLen(1023u, 0u, 8u, 128u, 0u, 8u) &&
                  WindowCoversSeqLen(1024u, 0u, 8u, 128u, 0u, 8u) &&
                  !WindowCoversSeqLen(1025u, 0u, 8u, 128u, 0u, 8u),
              "a window must cover every live token or the launch is refused");
static_assert(!WindowCoversSeqLen(1023u, 0u, 7u, 128u, 0u, 8u),
              "seven pages do not cover 1023 tokens");
static_assert(WholeWindowTokens(0u, 0u, 128u, 0u, 8u) == 0u &&
                  !WindowCoversSeqLen(1u, 0u, 0u, 128u, 0u, 8u),
              "an empty window covers nothing");
static_assert(SeqLenInCapacity(1023u, 8u, 128u) &&
                  SeqLenInCapacity(1024u, 8u, 128u) &&
                  !SeqLenInCapacity(1025u, 8u, 128u),
              "live tokens must fit the allocated cache");

// Monotonicity, which the trip count depends on.
static_assert(WalkIsMonotone(0u, 8u, 128u) && WalkIsMonotone(2u, 4u, 128u) &&
                  WalkIsMonotone(0u, 1u, 16u),
              "a window inside the page is monotone");
static_assert(!WalkIsMonotone(0u, 9u, 128u),
              "nine strips do not fit a 128-token page, and the walk would "
              "then revisit a page");
static_assert(!WalkIsMonotone(5u, 4u, 128u),
              "a window must end inside the page");
static_assert(!WalkIsMonotone(0u, 0u, 128u) && !WalkIsMonotone(0u, 1u, 0u) &&
                  !WalkIsMonotone(0u, 1u, 24u),
              "an empty window, a zero page and a page that is not a whole "
              "number of strips are all rejected");

}  // namespace turboquant_decode_tail
