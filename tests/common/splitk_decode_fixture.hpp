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
inline float TailCanary()
{
    const std::uint32_t bits = 0x7FC0BEEFu;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

struct Fixture {
    // `tokens` is the cache CAPACITY this fixture allocates, rounded up to
    // whole pages; `seq_len` is how many of those tokens are live. They differ
    // for every context length that is not a multiple of the page size, which
    // in generative decoding is almost all of them.
    int tokens, seq_len, pages, requests, heads, rows, page_tokens;
    std::vector<std::uint16_t> q;
    std::vector<std::uint8_t> key, value;
    std::vector<float> scales, gate, signs;
    std::vector<std::uint32_t> table;
    std::vector<float> expected;

    // `live` defaults to -1, meaning "every allocated token is live", which
    // is what every caller from before the tail change meant. Pass a smaller
    // value for an unaligned context: the pages are still allocated whole (the
    // cache has no notion of a part-page), the slots past `live` are flooded
    // with TailCanary(), and the oracle masks them out.
    Fixture(int n, int batch, int head_count = kHeads, int row_count = kRows,
            int page_size = kPageTokens, int live = -1)
        : tokens(((n + page_size - 1) / page_size) * page_size),
          seq_len(live < 0 ? n : live),
          pages((n + page_size - 1) / page_size), requests(batch),
          heads(head_count), rows(row_count), page_tokens(page_size),
          q(batch * heads * rows * kChannels),
          key(static_cast<std::size_t>(batch) * pages * heads * page_tokens * 128),
          value(key.size()), scales(static_cast<std::size_t>(batch) * pages *
                                   heads * page_tokens * 2),
          gate(q.size()), signs(kChannels), table(batch * pages), expected(q.size())
    {
        for (int c = 0; c < kChannels; ++c)
            signs[c] = static_cast<float>(single_token_decode::PiSign(c));
        for (int request = 0; request < batch; ++request) {
            // Reverse permutation over this request's distinct physical pages.
            for (int page = 0; page < pages; ++page) {
                const int physical = request * pages + pages - 1 - page;
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
                                paged_decode::PackNibble(key.data() + offset,
                                    t * kChannels + c,
                                    live ? single_token_decode::KeyNibbleAt(head, global_strip, t, c)
                                         : 7);
                                paged_decode::PackNibble(value.data() + offset,
                                    c * 16 + t,
                                    live ? single_token_decode::ValueNibbleAt(head, global_strip, c, t)
                                         : 7);
                            }
                            const auto at =
                                (static_cast<std::size_t>(physical) * heads + head) *
                                page_tokens + strip * 16 + t;
                            // Dead slots get the largest nibble the format
                            // holds and a NaN scale, so a kernel that lets one
                            // through cannot produce a nearly-right answer:
                            // either the NaN propagates, or the score is the
                            // largest this fixture can make and dominates the
                            // softmax.
                            scales[at] = live
                                ? single_token_decode::KeyScale(head, global_strip, t)
                                : TailCanary();
                            scales[scales.size() / 2 + at] = live
                                ? single_token_decode::ValueScaleAt(head, global_strip, t)
                                : TailCanary();
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
