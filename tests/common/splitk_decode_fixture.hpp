#pragma once
// Dynamically sized physical-page fixture and independent FP64 decode oracle.
#include "single_token_decode_oracle.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

namespace splitk_decode_test {
using ascend_micro::sim::require;
constexpr int kHeads = 4, kRows = 6, kChannels = 256, kPageTokens = 128;

// The pattern written into every cache slot at or past `seq_len`: 0x7FC0BEEF,
// the quiet NaN the isolated-canary protocol in
// tests/camodel/test_camodel_single_token_decode_aiv.cpp already uses. It is
// not decoration. A kernel that masks the dead lanes by MULTIPLYING them by
// zero still produces NaN, because 0 * NaN is NaN, and the NaN then reaches m,
// l, Acc and the context. So this constant is what separates a mask that
// overwrites the dead lanes from one that merely scales them, and it is the
// reason the device phase is a masked write rather than a masked multiply.
inline float FloatFromBits(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

inline float TailCanary() { return FloatFromBits(0x7FC0BEEFu); }

// What goes into a cache slot at or past `seq_len`.
//
//   kZero   every dead byte is 0. The benign baseline: whatever the kernel
//           does with a dead lane, zero data cannot move the answer much, so
//           this is the pass the dirty pass is compared AGAINST.
//   kDirty  finite adversarial garbage. Large magnitudes of both signs so a
//           leaked lane would dominate a softmax maximum rather than perturb
//           it, and the two alternating bit patterns so a leaked lane that is
//           reinterpreted rather than scaled is also wrong.
//   kNaN    the quiet NaN 0x7FC0BEEF, the loud canary.
enum class TailFill { kZero, kDirty, kNaN };

// How a request's logical pages map onto its distinct physical pages.
//
//   kReverse    logical page p lands at physical pages-1-p. The original and
//               the default. Already non-identity, which is what catches a
//               kernel that ignores the block table -- but MONOTONE, so a
//               kernel that walks its pages backwards, or negates its page
//               index somewhere, reads a correct-looking sequence out of it.
//   kScattered  monotone in neither direction, so the only way to read it
//               correctly is to consult the table per page. This is the
//               mapping the multi-page L1 reuse rungs want: the hazard they
//               cover is a kernel confusing one page's staging buffer for
//               another's, and a scattered table makes that confusion land on
//               unrelated data rather than on the neighbouring page.
enum class PageMap { kReverse, kScattered };

// The permutation of [0, pages) for one request, as a plain vector so the
// caller can assert its shape. Deterministic across runs and platforms: the
// point is a mapping a reader cannot predict from the page index, not
// randomness, and a fixture whose data placement moved between runs would make
// a failure unreproducible.
inline std::vector<int> PagePermutation(PageMap map, int pages, int request)
{
    std::vector<int> order(static_cast<std::size_t>(pages > 0 ? pages : 0));
    for (int page = 0; page < pages; ++page)
        order[static_cast<std::size_t>(page)] = pages - 1 - page;
    if (map == PageMap::kReverse) return order;
    // Fisher-Yates over a fixed LCG, seeded by the request so two requests do
    // not share a mapping.
    std::uint32_t state = 0x9E3779B9u ^
                          (static_cast<std::uint32_t>(request) * 0x85EBCA6Bu + 1u);
    for (int i = pages - 1; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        const int j = static_cast<int>((state >> 8) %
                                       static_cast<std::uint32_t>(i + 1));
        std::swap(order[static_cast<std::size_t>(i)],
                  order[static_cast<std::size_t>(j)]);
    }
    return order;
}

// The FP32 scale planes. Cycles large positive, large negative, a very large
// value, and the two bit patterns -- which as floats are about -2.9e-16 and
// +1.5e16, so the cycle spans sixteen orders of magnitude in both directions.
inline float DirtyScale(std::size_t index)
{
    switch (index % 5u) {
        case 0: return 999.0f;
        case 1: return -999.0f;
        case 2: return 1.0e4f;
        case 3: return FloatFromBits(0xA5A5A5A5u);
        default: return FloatFromBits(0x5A5A5A5Au);
    }
}

// The packed int4 planes. Alternating nibbles 0xA and 0x5 lay down exactly the
// 0xA5A5A5A5 / 0x5A5A5A5A byte patterns; as signed int4 they are -6 and +5.
inline int DirtyNibble(std::size_t index)
{
    return (index % 2u) == 0u ? -6 : 5;
}

inline float TailScaleFor(TailFill fill, std::size_t index)
{
    switch (fill) {
        case TailFill::kZero: return 0.0f;
        case TailFill::kDirty: return DirtyScale(index);
        default: return TailCanary();
    }
}

inline int TailNibbleFor(TailFill fill, std::size_t index)
{
    switch (fill) {
        case TailFill::kZero: return 0;
        case TailFill::kDirty: return DirtyNibble(index);
        default: return 7;  // the largest the format holds
    }
}

struct Fixture {
    // `tokens` is the cache CAPACITY this fixture allocates, rounded up to
    // whole pages; `seq_len` is how many of those tokens are live. They differ
    // for every context length that is not a multiple of the page size, which
    // in generative decoding is almost all of them.
    int tokens, seq_len, pages, requests, heads, rows, page_tokens;
    TailFill tail_fill;
    PageMap page_map;
    std::vector<std::uint16_t> q;
    std::vector<std::uint8_t> key, value;
    std::vector<float> scales, gate, signs;
    std::vector<std::uint32_t> table;
    std::vector<float> expected;

    // `live` defaults to -1, meaning "every allocated token is live", which
    // is what every caller from before the tail change meant. Pass a smaller
    // value for an unaligned context: the pages are still allocated whole (the
    // cache has no notion of a part-page), the slots past `live` are flooded
    // according to `fill`, and the oracle masks them out either way.
    //
    // An ALIGNED fixture has no dead slot, so `fill` cannot affect it at all.
    // That is what makes changing the default safe for every caller that does
    // not pass `live`.
    //
    // `map` is last for the same reason `live` and `fill` are where they are:
    // every caller from before it existed keeps the reverse permutation and so
    // keeps a byte-identical fixture.
    Fixture(int n, int batch, int head_count = kHeads, int row_count = kRows,
            int page_size = kPageTokens, int live = -1,
            TailFill fill = TailFill::kDirty,
            PageMap map = PageMap::kReverse)
        : tokens(((n + page_size - 1) / page_size) * page_size),
          seq_len(live < 0 ? n : live),
          pages((n + page_size - 1) / page_size), requests(batch),
          heads(head_count), rows(row_count), page_tokens(page_size),
          tail_fill(fill), page_map(map),
          q(batch * heads * rows * kChannels),
          key(static_cast<std::size_t>(batch) * pages * heads * page_tokens * 128),
          value(key.size()), scales(static_cast<std::size_t>(batch) * pages *
                                   heads * page_tokens * 2),
          gate(q.size()), signs(kChannels), table(batch * pages), expected(q.size())
    {
        for (int c = 0; c < kChannels; ++c)
            signs[c] = static_cast<float>(single_token_decode::PiSign(c));
        for (int request = 0; request < batch; ++request) {
            // A permutation over this request's distinct physical pages. The
            // table entry and the data offsets below both come from
            // `physical`, so any permutation is self-consistent: it moves
            // WHERE a logical token is stored without changing WHAT it holds,
            // because the content is keyed by `global_strip`, which is derived
            // from the LOGICAL page. That is what makes the oracle's output
            // independent of the mapping -- and a kernel that ignores the
            // table read a different logical token.
            const std::vector<int> order = PagePermutation(map, pages, request);
            for (int page = 0; page < pages; ++page) {
                const int physical = request * pages + order[static_cast<std::size_t>(page)];
                table[request * pages + page] = physical;
                for (int head = 0; head < heads; ++head) {
                    for (int strip = 0; strip < page_tokens / 16; ++strip) {
                        const int global_strip = (request * pages + page) * (page_tokens / 16) + strip;
                        const auto offset =
                            (static_cast<std::size_t>(physical) * heads + head) *
                            page_tokens * 128 + strip * 2048;
                        for (int t = 0; t < 16; ++t) {
                            // The logical token this slot holds, in the
                            // request's own numbering -- the same numbering
                            // turboquant_decode_tail works in.
                            const int logical = page * page_tokens + strip * 16 + t;
                            const bool live = logical < seq_len;
                            for (int c = 0; c < kChannels; ++c) {
                                const auto at_key =
                                    static_cast<std::size_t>(t) * kChannels + c;
                                const auto at_val =
                                    static_cast<std::size_t>(c) * 16 + t;
                                paged_decode::PackNibble(key.data() + offset, at_key,
                                    live ? single_token_decode::KeyNibbleAt(head, global_strip, t, c)
                                         : TailNibbleFor(fill, at_key));
                                paged_decode::PackNibble(value.data() + offset, at_val,
                                    live ? single_token_decode::ValueNibbleAt(head, global_strip, c, t)
                                         : TailNibbleFor(fill, at_val));
                            }
                            const auto at =
                                (static_cast<std::size_t>(physical) * heads + head) *
                                page_tokens + strip * 16 + t;
                            // A dead slot's scale decides whether a leak is
                            // visible, so it is the adversarial one: under
                            // kDirty a leaked lane gets a magnitude that
                            // dominates the softmax maximum rather than
                            // perturbing it, and under kZero it gets nothing.
                            // The two passes must agree regardless.
                            scales[at] = live
                                ? single_token_decode::KeyScale(head, global_strip, t)
                                : TailScaleFor(fill, at);
                            scales[scales.size() / 2 + at] = live
                                ? single_token_decode::ValueScaleAt(head, global_strip, t)
                                : TailScaleFor(fill, at + 1u);
                        }
                    }
                }
            }
            for (int head = 0; head < heads; ++head)
                for (int row = 0; row < rows; ++row)
                    for (int c = 0; c < kChannels; ++c) {
                        const auto at = ((request * heads + head) * rows + row) *
                                        kChannels + c;
                        q[at] = ascend_micro::sim::float_to_bfloat16_rne(
                            single_token_decode::QRotEntry(head, row, c));
                        gate[at] = single_token_decode::GateEntry(
                            head, single_token_decode::QueryRowOf(head, row), c) +
                            static_cast<float>(request) * 0.125f;
                    }
        }
        Reference();
    }

    static int Unpack(const std::uint8_t* data, int index)
    {
        const int bits = (data[index / 2] >> ((index % 2) * 4)) & 15;
        return bits >= 8 ? bits - 16 : bits;
    }

    // Independent full-context FP64 softmax, reading the uploaded packed bytes
    // through the uploaded block table. No split partition or online merge.
    // Independent full-context FP64 softmax over the LIVE tokens, with the
    // dead tail masked to -inf exactly as the kernel masks it to a large
    // negative. Reading a dead slot is avoided rather than neutralised: its
    // scale is a NaN, and `0 * NaN` is NaN here just as it is on the device, so
    // a zero weight would not be enough -- the same hazard, in the same shape,
    // on both sides of the comparison.
    void Reference()
    {
        std::vector<double> scores(tokens), weights(tokens);
        for (int request = 0; request < requests; ++request)
            for (int head = 0; head < heads; ++head)
                for (int row = 0; row < rows; ++row) {
                    const auto base = ((request * heads + head) * rows + row) * kChannels;
                    double maximum = -std::numeric_limits<double>::infinity();
                    for (int t = 0; t < tokens; ++t) {
                        if (t >= seq_len) {
                            scores[t] = -std::numeric_limits<double>::infinity();
                            continue;
                        }
                        const auto physical = table[request * pages + t / page_tokens];
                        const auto strip_base =
                            (static_cast<std::size_t>(physical) * heads + head) *
                            page_tokens * 128 + (t % page_tokens / 16) * 2048;
                        double dot = 0;
                        for (int c = 0; c < kChannels; ++c)
                            dot += static_cast<double>(single_token_decode::bfloat16_to_float(q[base + c])) *
                                   Unpack(key.data() + strip_base, (t % 16) * kChannels + c);
                        const auto scale_at =
                            (static_cast<std::size_t>(physical) * heads + head) *
                            page_tokens + t % page_tokens;
                        scores[t] = dot * scales[scale_at];
                        maximum = std::max(maximum, scores[t]);
                    }
                    double denominator = 0;
                    for (int t = 0; t < tokens; ++t) {
                        weights[t] = std::exp(scores[t] - maximum);
                        denominator += weights[t];
                    }
                    std::vector<double> acc(kChannels, 0);
                    for (int t = 0; t < tokens; ++t) {
                        if (t >= seq_len) continue;  // weights[t] is 0; the payload is NaN
                        const auto physical = table[request * pages + t / page_tokens];
                        const auto strip_base =
                            (static_cast<std::size_t>(physical) * heads + head) *
                            page_tokens * 128 + (t % page_tokens / 16) * 2048;
                        const auto scale_at =
                            (static_cast<std::size_t>(physical) * heads + head) *
                            page_tokens + t % page_tokens;
                        const double p = weights[t] * scales[scales.size() / 2 + scale_at];
                        for (int c = 0; c < kChannels; ++c)
                            acc[c] += p * Unpack(value.data() + strip_base, c * 16 + t % 16);
                    }
                    std::vector<float> normalized(kChannels);
                    for (int c = 0; c < kChannels; ++c)
                        normalized[c] = static_cast<float>(acc[c] / denominator);
                    // Dense matrix rotation checks the register butterfly's
                    // egress ordering independently of its factorization.
                    const auto rotated = single_token_decode::DenseRotate(normalized.data());
                    for (int c = 0; c < kChannels; ++c)
                        expected[base + c] = rotated[c] / (1.0f + std::exp(-gate[base + c]));
                }
    }
};

}  // namespace splitk_decode_test
