#pragma once

// CPU golden oracle and fixture for the AIV-only single-token paged decode
// operator (roadmap kernel No.3).
//
//   query tokens per request   1          (query_tokens_per_req == 1)
//   rows per (request, kv_head) 6         H_q / H_kv = 24 / 4, at k = 0
//   head dimension             d = 256
//   vLLM page                  128 tokens = 8 strips of 16
//   KV heads                   H_kv = 4
//   grid                       (request, kv_head), so B scales the GRID
//
// This header adds exactly three things to `paged_decode`, and deliberately
// nothing else:
//
//   1. the 6-row query indexing, which is GQA's (q head = kv_head * 6 + r);
//   2. a GLOBAL strip index, so a request can span several pages;
//   3. the 6-row golden model, split at every boundary the device publishes.
//
// Everything that defines the data -- Pi, the resident blocks, the int4
// packing, the two cache orders, the planar scale cache, the nibble patterns,
// the value scale, the gate, the block table and the egress butterfly -- is
// INCLUDED from tests/common/paged_decode_oracle.hpp and not restated. That
// is not only to avoid duplication: at `pages == 1` the fixture below is
// byte-identical to the Cube decode operator's, so the two kernels can be run
// against the same cache image and their first six rows compared to each
// other. A second copy of the layout would quietly destroy that property.
//
// What is checked bit-exactly, and what is not
// --------------------------------------------
//   raw score panels  BIT-EXACT. See `routine_vector_decode_gemv.hpp`: on this
//                     fixture every partial sum of the 256-term dot product is
//                     an exact multiple of 1/16 bounded by 1792, so all three
//                     summation orders -- the host's loop, the device's binary
//                     tree inside a repeat, and the quarter fold across
//                     repeats -- give the same FP32 word.
//   m                 BIT-EXACT. m is a maximum of score columns, each an
//                     exact raw score times a power-of-two key scale. A
//                     maximum of exactly representable values is exact, so
//                     unlike the Cube operator's 2-ULP budget for the pair,
//                     this one is checkable to the bit. (That is a stronger
//                     gate than the roadmap asks for, and it is free.)
//   l, acc            ULP. Both pass through `Exp`, documented at up to 1 ULP
//                     per call, and `acc` is a cancelling reduction.
//   rotated, context  RELATIVE, against the largest magnitude in the row.
//                     `acc / l` is a 0-ULP `Div` but `l` already carries the
//                     `Exp` error; the eight butterfly stages then cancel, and
//                     the gate adds one more `Exp` and one more `Div`. The
//                     task statement asks for a bit-exact normalised context;
//                     that is not reachable through `Exp` and `Div` on any
//                     fixture, and asserting it would mean asserting something
//                     the arithmetic forbids. The budgets below are the
//                     decode report's, stated against the largest intermediate
//                     per guidelines section 5.

#include "paged_decode_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace single_token_decode {

using paged_decode::bfloat16_to_float;
using paged_decode::bfloat16_is_exact;
using paged_decode::EgressButterfly;
using paged_decode::GateEntry;
using paged_decode::GoldenQRot;
using paged_decode::GoldenQRotBf16;
using paged_decode::KeyNibble;
using paged_decode::PiEntry;
using paged_decode::PiSign;
using paged_decode::QueryEntry;
using paged_decode::ValueNibble;
using paged_decode::ValueScale;

using ascend_micro::sim::float_bits;
using ascend_micro::sim::float_to_bfloat16_rne;
using ascend_micro::sim::require;

// ------------------------------------------------------------------ geometry
constexpr int kHeadSize = paged_decode::kHeadSize;      // 256
constexpr int kKeys = paged_decode::kTokens;            // 16 tokens per strip
constexpr int kStripsPerPage = paged_decode::kStrips;   // 8
constexpr int kKvHeads = paged_decode::kKvHeads;        // 4
constexpr int kRows = 6;                                // H_q / H_kv at k = 0
constexpr int kQHeads = kKvHeads * kRows;               // 24
constexpr int kPanelEntries = kRows * kKeys;            // 96 raw scores
constexpr int kRowElems = kRows * kHeadSize;            // 1536
constexpr int kStateLanes = 8;                          // one FP32 DataBlock

// The grid at B = 1. The canary protocol isolates one (request, kv_head) item
// per launch, so the item index is the output slot and `kItems` is the sweep
// the unassigned-slot check covers.
constexpr int kItems = kKvHeads;

// How many pages the fixture's cache can serve. `paged_decode::kNumBlocks` is
// 9 physical pages and the block table maps 8 logical ones, which is the most
// a request may span here.
constexpr int kMaxPages = paged_decode::kTemporalBlocks;  // 8
constexpr int kMaxStrips = kMaxPages * kStripsPerPage;    // 64

static_assert(kRows * kKvHeads == kQHeads, "GQA 6:1 over four KV heads");
static_assert(kRows < paged_decode::kTokens,
              "the 6 rows are drawn from the 16-row query fixture");

// ------------------------------------------------------- the 6-row query
// GQA indexing: query head j = kv_head * 6 + r. The fixture's logical query
// row for head h row r is j mod 16, which gives each head a DIFFERENT set of
// six rows (h = 0 -> 0..5, h = 1 -> 6..11, h = 2 -> 12..15,0,1,
// h = 3 -> 2..7). That matters for the canary protocol: head separation in the
// Cube operator comes only from the cache slice and the gate, and a task that
// read the wrong head's query rows would be invisible there. Here it is not.
inline int QueryRowOf(int head, int row)
{
    return (head * kRows + row) % paged_decode::kTokens;
}

// Q_rot as the device reads it, FP32. The value is the bf16 image of the
// rotation -- the same words the Cube operator's Fixpipe leaves in L1 -- so
// the two operators contract against identical query operands, and FP32 is
// exact for a bf16 value. Kernel No.5 produces this panel on the Cube; until
// it exists the runner uploads it, which is also what makes this kernel
// testable on its own.
inline float QRotEntry(int head, int row, int channel)
{
    const std::vector<float>& rot = [] {
        static const std::vector<float> table = GoldenQRotBf16(0);
        return table;
    }();
    return rot[static_cast<std::size_t>(QueryRowOf(head, row)) * kHeadSize + channel];
}

// ------------------------------------------------- global strip addressing
// A request's strips are numbered along the sequence: global strip
// g = page_index * 8 + strip_in_page, with the PHYSICAL page coming from the
// block table, which is deliberately not the identity.
inline int PageOfStrip(int global_strip) { return global_strip / kStripsPerPage; }
inline int StripInPage(int global_strip) { return global_strip % kStripsPerPage; }
inline int PhysicalPageOfStrip(int global_strip)
{
    return paged_decode::BlockTableEntry(PageOfStrip(global_strip));
}

// The key scale's per-strip doubling, and why the ramp restarts per page.
//
// `paged_decode::KeyScale` ramps the exponent by one per strip from a base of
// -16, chosen so that eight strips land the whole score range in about
// [-0.6, +1.6]: too high a base and every exp but the largest underflows, the
// probability mass collapses to 1.0 per query, and the strips stop being
// distinguishable -- which the Cube operator's Tier-1 probe rejected a fixture
// for. Four pages is 32 strips, where that ramp would reach 2^15 and do
// exactly that.
//
// So the ramp is taken modulo the page: `(global_strip % 8)`. At one page this
// is `paged_decode::KeyScale` verbatim, which is what keeps the two fixtures
// byte-identical there. Across a page boundary the strip maximum DROPS back,
// and that is a feature rather than a compromise: it is the only way this
// family of fixtures exercises the branch where the CARRIED maximum wins
// (alpha == 1, a small tile contribution) instead of the rescale branch
// (alpha < 1). A single-page run never reaches it, because the ramp makes
// every strip maximum rise. `ValidateFixture` asserts both branches occur.
inline float KeyScale(int head, int global_strip, int token)
{
    const int exponent = paged_decode::kKeyScaleBaseExponent +
                         (global_strip % kStripsPerPage) + ((token + head) % 3);
    return std::ldexp(1.0f, exponent);
}

// The nibbles, extended to a global strip index -- and the page index has to
// enter the pattern, which is not what the first version of this oracle did.
//
// `paged_decode::KeyNibble`'s strip term is `strip * 11` taken mod 8, so it
// has PERIOD 8; its head term has period 8 too, and the whole pattern depends
// only on `(3 . strip + 5 . head) mod 8`. Eight distinct key payloads exist in
// that generator, full stop. Handing it a global strip index therefore gives
// page 1 strip 0 the SAME 4096 nibbles as page 0 strip 0, and a kernel that
// ignored the block table and read page 0 four times would have produced
// exactly the right answer -- the "wrong operand yields the right answer"
// failure mode that let the online-softmax WAR hazard pass its original
// validation. The Tier-1 probe caught it as `two strips share a row maximum`,
// before any simulator run.
//
// So the page index is mixed in, and mixed in so that it VANISHES on page 0:
// `page * (channel + 1)` is zero for every channel when page is zero, which
// keeps the single-page fixture byte-identical to the Cube operator's.
// `ValidateSharedFixture` asserts that identity rather than trusting this
// comment. The anchor term is reused verbatim from `paged_decode`, because it
// is what makes `max_t R[row][t] > 0` and therefore what makes the key
// scale's per-strip doubling raise the row maximum.
inline int KeyNibbleAt(int head, int global_strip, int token, int channel)
{
    const int page = PageOfStrip(global_strip);
    if (page == 0) return KeyNibble(head, global_strip, token, channel);
    const int base = (token * 5 + channel * 3 + global_strip * 11 + head * 13 +
                      page * (channel + 1)) %
                     8;  // 0..7
    const signed char sign = paged_decode::QRotSignTable()[
        static_cast<std::size_t>(token) * kHeadSize + channel];
    return base + (sign > 0 ? paged_decode::kKeyAnchorBias : 0) - 7;  // -7..+7
}

inline int ValueNibbleAt(int head, int global_strip, int channel, int token)
{
    const int page = PageOfStrip(global_strip);
    if (page == 0) return ValueNibble(head, global_strip, channel, token);
    const int raw = (channel * 7 + (channel / kKeys) * 3 + token * 5 +
                     global_strip * 11 + head * 3 + page * (token + 1)) %
                    15;  // 0..14
    return raw - 7;      // -7..+7
}
// The value scale's strip term is `strip % 2`, so it is page-periodic by
// construction and needs no extension: the value payload's page dependence is
// carried by the nibbles above.
inline float ValueScaleAt(int head, int global_strip, int token)
{
    return ValueScale(head, global_strip, token);
}

// ------------------------------------------------------- the golden model
// Raw score panel for one strip: R[row][t] = sum_c Q_rot[row][c] . K[t][c].
// No scale of any kind. This is the quantity the device archives and the
// runner compares bit for bit.
inline std::vector<float> GoldenRawPanel(int head, int global_strip)
{
    std::vector<float> raw(static_cast<std::size_t>(kPanelEntries), 0.0f);
    for (int row = 0; row < kRows; ++row)
        for (int t = 0; t < kKeys; ++t) {
            float sum = 0.0f;
            for (int c = 0; c < kHeadSize; ++c)
                sum += QRotEntry(head, row, c) *
                       static_cast<float>(KeyNibbleAt(head, global_strip, t, c));
            raw[static_cast<std::size_t>(row) * kKeys + t] = sum;
        }
    return raw;
}

struct GoldenState {
    std::vector<float> m;           // kRows
    std::vector<float> l;           // kRows
    std::vector<float> acc;         // kRowElems, the cache's rotated basis
    std::vector<float> normalized;  // kRowElems, acc / l
    std::vector<float> rotated;     // kRowElems, after the egress butterfly
    std::vector<float> context;     // kRowElems, gated by sigmoid(gate)
    // Diagnostics the fixture validation and the report read.
    std::vector<float> strip_max;   // per strip, max over rows and tokens
    int rescale_strips = 0;         // strips where alpha < 1 for some row
    int carried_strips = 0;         // strips where the carried max won for some row
};

// The streaming recurrence over `strips` (global indices, in order), then the
// egress. Written in the order the device runs it, and publishing every
// intermediate the device publishes -- the split is what makes a CAModel
// failure actionable instead of "the egress is wrong".
inline GoldenState GoldenDecode(int head, const std::vector<int>& strips)
{
    GoldenState state;
    state.m.assign(kRows, -std::numeric_limits<float>::infinity());
    state.l.assign(kRows, 0.0f);
    state.acc.assign(static_cast<std::size_t>(kRowElems), 0.0f);
    state.strip_max.assign(strips.size(), -std::numeric_limits<float>::infinity());

    for (std::size_t index = 0; index < strips.size(); ++index) {
        const int strip = strips[index];
        const std::vector<float> raw = GoldenRawPanel(head, strip);
        bool rescaled = false;
        bool carried = false;
        for (int row = 0; row < kRows; ++row) {
            float scores[kKeys];
            float tile_max = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < kKeys; ++t) {
                scores[t] = raw[static_cast<std::size_t>(row) * kKeys + t] *
                            KeyScale(head, strip, t);
                if (scores[t] > tile_max) tile_max = scores[t];
            }
            if (tile_max > state.strip_max[index]) state.strip_max[index] = tile_max;

            const float m_new = std::max(state.m[row], tile_max);
            const float alpha = std::exp(state.m[row] - m_new);
            if (alpha < 1.0f) rescaled = true;
            if (index > 0 && state.m[row] >= tile_max) carried = true;

            float p[kKeys];
            float p_sum = 0.0f;
            for (int t = 0; t < kKeys; ++t) {
                p[t] = std::exp(scores[t] - m_new);
                p_sum += p[t];
            }
            // The value scale folds into p -- 16 exact multiplies instead of
            // 4096 over the page -- and the denominator still sums the
            // UNSCALED p, exactly as in the Cube operator.
            float p_scaled[kKeys];
            for (int t = 0; t < kKeys; ++t)
                p_scaled[t] = p[t] * ValueScaleAt(head, strip, t);
            for (int c = 0; c < kHeadSize; ++c) {
                float weighted = 0.0f;
                for (int t = 0; t < kKeys; ++t)
                    weighted += p_scaled[t] *
                                static_cast<float>(ValueNibbleAt(head, strip, c, t));
                const std::size_t at = static_cast<std::size_t>(row) * kHeadSize + c;
                state.acc[at] = state.acc[at] * alpha + weighted;
            }
            state.l[row] = state.l[row] * alpha + p_sum;
            state.m[row] = m_new;
        }
        if (rescaled) ++state.rescale_strips;
        if (carried) ++state.carried_strips;
    }

    // Egress. normalized = acc / l; rotated = normalized . Pi as the eight
    // butterfly stages with the 1/16 applied last, which is the associativity
    // `rotate_butterfly_gather_vf<Egress>` uses (`rot_scale_row` runs after
    // the stages, and 1/16 is a power of two, so before or after is the same
    // FP32 word); context = rotated . sigmoid(gate).
    state.normalized.assign(static_cast<std::size_t>(kRowElems), 0.0f);
    state.rotated.assign(static_cast<std::size_t>(kRowElems), 0.0f);
    state.context.assign(static_cast<std::size_t>(kRowElems), 0.0f);
    for (int row = 0; row < kRows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kHeadSize;
        for (int c = 0; c < kHeadSize; ++c)
            state.normalized[base + c] =
                state.l[row] == 0.0f ? 0.0f : state.acc[base + c] / state.l[row];
        const std::vector<float> turned = EgressButterfly(&state.normalized[base]);
        for (int c = 0; c < kHeadSize; ++c) {
            state.rotated[base + c] = turned[c];
            const float gate = GateEntry(head, QueryRowOf(head, row), c);
            state.context[base + c] = turned[c] / (1.0f + std::exp(-gate));
        }
    }
    return state;
}

// The dense reference for the egress: ctx = normalized . Pi, summed over the
// input channel with Pi's own 1/16 inside each term. A DIFFERENT associativity
// from the butterfly, so the two agreeing to a relative tolerance is evidence
// about the rotation and not about one implementation of it.
inline std::vector<float> DenseRotate(const float* normalized)
{
    std::vector<float> out(static_cast<std::size_t>(kHeadSize), 0.0f);
    for (int o = 0; o < kHeadSize; ++o) {
        double sum = 0.0;
        for (int c = 0; c < kHeadSize; ++c)
            sum += static_cast<double>(normalized[c]) * static_cast<double>(PiEntry(c, o));
        out[o] = static_cast<float>(sum);
    }
    return out;
}

// --------------------------------------------------------- fixture validation
struct FixtureReport {
    double max_qrot_magnitude = 0.0;
    double max_partial_sum = 0.0;  // the bound the bit-exact panel gate rests on
    double max_score = 0.0;
    double min_score = 0.0;
    int rescale_strips = 0;
    int carried_strips = 0;
    double smallest_l = 0.0;
    // The closest pair of raw score panels in the window, in differing
    // entries out of kPanelEntries. 0 would mean two interchangeable strips.
    int closest_panels = 0;
};

// The properties the gates depend on. Each one has a gate that becomes
// vacuous without it, and that is the only reason any of them is here.
inline FixtureReport ValidateFixture(int head, int pages)
{
    require(pages >= 1 && pages <= kMaxPages, "page count out of range");
    FixtureReport report;
    report.min_score = std::numeric_limits<double>::infinity();

    std::vector<int> strips;
    for (int page = 0; page < pages; ++page)
        for (int s = 0; s < kStripsPerPage; ++s)
            strips.push_back(page * kStripsPerPage + s);

    // 1. The six query rows are exact in bf16, so Q_rot is an exact FP32
    //    operand and the panel can be compared to the bit.
    for (int row = 0; row < kRows; ++row)
        for (int c = 0; c < kHeadSize; ++c) {
            const int logical = QueryRowOf(head, row);
            double exact = 0.0;
            for (int i = 0; i < kHeadSize; ++i)
                exact += static_cast<double>(QueryEntry(logical, i)) *
                         static_cast<double>(PiEntry(c, i));
            require(bfloat16_is_exact(exact),
                    "a Q_rot entry is not exact in bf16, so the score panel "
                    "could not be compared bit-exactly");
            require(static_cast<double>(QRotEntry(head, row, c)) == exact,
                    "QRotEntry disagrees with the double-precision rotation");
            report.max_qrot_magnitude =
                std::max(report.max_qrot_magnitude, std::fabs(exact));
        }

    // 2. The six rows of a head are distinct logical rows, or two rows of the
    //    output would be interchangeable and a swapped row invisible.
    for (int a = 0; a < kRows; ++a)
        for (int b = a + 1; b < kRows; ++b)
            require(QueryRowOf(head, a) != QueryRowOf(head, b),
                    "two of a head's six query rows are the same logical row");

    // 3. THE BIT-EXACT PANEL GATE'S PRECONDITION. Every prefix sum of the
    //    256-term dot product, in EVERY order, must be exactly representable
    //    in FP32 -- that is what makes the host's loop, the device's binary
    //    tree inside a repeat and the quarter fold across repeats agree to the
    //    bit. Sufficient: each term is an integer multiple of 1/16 (Q_rot is,
    //    the nibbles are integers) and the running magnitude stays under
    //    2^24 / 16. Checked as a bound over all orders by bounding the sum of
    //    absolute values, which dominates every order.
    for (const int strip : strips)
        for (int row = 0; row < kRows; ++row)
            for (int t = 0; t < kKeys; ++t) {
                double absolute = 0.0;
                for (int c = 0; c < kHeadSize; ++c) {
                    const double term =
                        static_cast<double>(QRotEntry(head, row, c)) *
                        static_cast<double>(KeyNibbleAt(head, strip, t, c));
                    const double scaled = term * 16.0;
                    require(scaled == std::floor(scaled),
                            "a score term is not an exact multiple of 1/16");
                    absolute += std::fabs(term);
                }
                report.max_partial_sum = std::max(report.max_partial_sum, absolute);
            }
    require(report.max_partial_sum * 16.0 < 16777216.0,
            "a partial sum of the score dot product can exceed 2^24 / 16, so "
            "the summation order would be observable and the bit-exact panel "
            "gate would be wrong");

    // 4. The raw panel agrees with the double-precision computation, which is
    //    what says the FP32 accumulation in `GoldenRawPanel` is itself exact.
    for (const int strip : strips) {
        const std::vector<float> raw = GoldenRawPanel(head, strip);
        for (int row = 0; row < kRows; ++row)
            for (int t = 0; t < kKeys; ++t) {
                double exact = 0.0;
                for (int c = 0; c < kHeadSize; ++c)
                    exact += static_cast<double>(QRotEntry(head, row, c)) *
                             static_cast<double>(KeyNibbleAt(head, strip, t, c));
                require(static_cast<double>(
                            raw[static_cast<std::size_t>(row) * kKeys + t]) == exact,
                        "the FP32 raw panel disagrees with the double one");
            }
    }

    // 5. NO TWO STRIPS ARE INTERCHANGEABLE. A fixture whose strips can be
    //    swapped cannot detect a wrong-strip or wrong-page read, because the
    //    wrong operand yields the right answer -- which is what let the
    //    online-softmax WAR hazard pass its original validation, and what the
    //    page-periodic nibble generator above would have reintroduced.
    //
    //    Stated over whole panels, and that is deliberate. The first version
    //    of this check compared per-row score MAXIMA and required them
    //    pairwise distinct, which is not a structural property at all: a raw
    //    score is an integer multiple of 1/16 inside a range of some 4700
    //    values, so two strips sharing a scalar maximum is an expected
    //    coincidence over the ~2900 pairs this fixture has, and it says
    //    nothing about whether the strips can be told apart. Two strips whose
    //    96-entry panels differ ARE distinguishable, whatever their maxima do.
    std::vector<std::vector<float>> panels;
    panels.reserve(strips.size());
    for (const int strip : strips) {
        panels.push_back(GoldenRawPanel(head, strip));
        for (int row = 0; row < kRows; ++row)
            for (int t = 0; t < kKeys; ++t) {
                const float score =
                    panels.back()[static_cast<std::size_t>(row) * kKeys + t] *
                    KeyScale(head, strip, t);
                report.max_score = std::max(report.max_score, static_cast<double>(score));
                report.min_score = std::min(report.min_score, static_cast<double>(score));
            }
    }
    int closest = kPanelEntries + 1;
    for (std::size_t a = 0; a < panels.size(); ++a)
        for (std::size_t b = a + 1; b < panels.size(); ++b) {
            int differing = 0;
            for (int entry = 0; entry < kPanelEntries; ++entry)
                if (float_bits(panels[a][entry]) != float_bits(panels[b][entry]))
                    ++differing;
            require(differing > 0,
                    "two strips have identical raw score panels, so a "
                    "wrong-strip or wrong-page read would yield the right "
                    "answer");
            closest = std::min(closest, differing);
        }
    report.closest_panels = closest;

    // 6. Both recurrence branches run. Within a page the ramp makes every
    //    strip maximum rise, so alpha < 1; at a page boundary it drops back,
    //    so the carried maximum wins. A gate on `alpha` is vacuous if the
    //    fixture only ever takes one of the two.
    const GoldenState state = GoldenDecode(head, strips);
    report.rescale_strips = state.rescale_strips;
    report.carried_strips = state.carried_strips;
    require(state.rescale_strips > 0,
            "no strip rescales the accumulator, so the online softmax's "
            "hardest path never runs");
    if (pages > 1)
        require(state.carried_strips > 0,
                "no strip is dominated by the carried maximum, so the "
                "multi-page run exercises nothing the single page does not");

    // 7. The denominator is positive and finite, or the egress divides by zero
    //    and every later check is meaningless.
    report.smallest_l = std::numeric_limits<double>::infinity();
    for (int row = 0; row < kRows; ++row) {
        require(std::isfinite(state.l[row]) && state.l[row] > 0.0f,
                "a row's softmax denominator is not positive and finite");
        report.smallest_l = std::min(report.smallest_l,
                                     static_cast<double>(state.l[row]));
    }

    // 8. The output must distinguish a permuted channel and a dropped
    //    butterfly stage. Same two properties the Cube operator's fixture
    //    asserts, for the same reasons: three quarters of the channels of a
    //    row distinct, and the row not near-invariant under any of the eight
    //    egress strides or under the group/inner index swap.
    for (int row = 0; row < kRows; ++row) {
        const float* out = state.context.data() + static_cast<std::size_t>(row) * kHeadSize;
        std::vector<std::uint32_t> values(kHeadSize);
        for (int c = 0; c < kHeadSize; ++c) values[c] = float_bits(out[c]);
        std::sort(values.begin(), values.end());
        const auto distinct =
            static_cast<int>(std::unique(values.begin(), values.end()) - values.begin());
        require(distinct * 4 >= kHeadSize * 3,
                "fewer than three quarters of a row's output channels are "
                "distinct, so a permuted channel could go unobserved");
        for (int stage = 0; stage < paged_decode::kEgressStageCount; ++stage) {
            const int stride = paged_decode::kEgressStrides[stage];
            int changed = 0;
            for (int c = 0; c < kHeadSize; ++c)
                if (float_bits(out[c]) != float_bits(out[c ^ stride])) ++changed;
            require(changed * 2 >= kHeadSize,
                    "an output row is near-invariant under an egress butterfly "
                    "stride, so a dropped or mis-strided stage would not show");
        }
        int swapped = 0;
        for (int c = 0; c < kHeadSize; ++c) {
            const int mirrored = (c % paged_decode::kGroup) * paged_decode::kGroups +
                                 (c / paged_decode::kGroup);
            if (float_bits(out[c]) != float_bits(out[mirrored])) ++swapped;
        }
        require(swapped * 2 >= kHeadSize,
                "an output row is near-invariant under the group/inner index "
                "swap, so a transposed group layout would not show");
    }
    return report;
}

// At one page the fixture must BE the Cube operator's, which is what lets the
// same cache image serve both kernels and their first six rows be compared.
// Asserted rather than asserted-in-a-comment.
inline void ValidateSharedFixture(int head)
{
    for (int strip = 0; strip < kStripsPerPage; ++strip) {
        for (int token = 0; token < kKeys; ++token) {
            require(float_bits(KeyScale(head, strip, token)) ==
                        float_bits(paged_decode::KeyScale(head, strip, token)),
                    "the single-page key scale differs from the Cube operator's");
            for (int c = 0; c < kHeadSize; ++c)
                require(KeyNibbleAt(head, strip, token, c) ==
                            KeyNibble(head, strip, token, c),
                        "the single-page key nibbles differ from the Cube "
                        "operator's");
        }
        for (int c = 0; c < kHeadSize; ++c)
            for (int token = 0; token < kKeys; ++token)
                require(ValueNibbleAt(head, strip, c, token) ==
                            ValueNibble(head, strip, c, token),
                        "the single-page value nibbles differ from the Cube "
                        "operator's");
    }
}

}  // namespace single_token_decode
