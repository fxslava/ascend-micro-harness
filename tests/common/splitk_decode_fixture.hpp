#pragma once
// Dynamically sized physical-page fixture and independent FP64 decode oracle.
#include "single_token_decode_oracle.hpp"
#include "../../include/turboquant_splitk_contract.hpp"

namespace splitk_decode_test {
using ascend_micro::sim::require;
constexpr int kHeads = 4, kRows = 6, kChannels = 256, kPageTokens = 128;

struct Fixture {
    int tokens, pages, requests, heads, rows, page_tokens;
    std::vector<std::uint16_t> q;
    std::vector<std::uint8_t> key, value;
    std::vector<float> scales, gate, signs;
    std::vector<std::uint32_t> table;
    std::vector<float> expected;

    Fixture(int n, int batch, int head_count = kHeads, int row_count = kRows,
            int page_size = kPageTokens)
        : tokens(n), pages(n / page_size), requests(batch), heads(head_count),
          rows(row_count), page_tokens(page_size),
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
                            for (int c = 0; c < kChannels; ++c) {
                                paged_decode::PackNibble(key.data() + offset,
                                    t * kChannels + c,
                                    single_token_decode::KeyNibbleAt(head, global_strip, t, c));
                                paged_decode::PackNibble(value.data() + offset,
                                    c * 16 + t,
                                    single_token_decode::ValueNibbleAt(head, global_strip, c, t));
                            }
                            const auto at =
                                (static_cast<std::size_t>(physical) * heads + head) *
                                page_tokens + strip * 16 + t;
                            scales[at] = single_token_decode::KeyScale(head, global_strip, t);
                            scales[scales.size() / 2 + at] =
                                single_token_decode::ValueScaleAt(head, global_strip, t);
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
    void Reference()
    {
        std::vector<double> scores(tokens), weights(tokens);
        for (int request = 0; request < requests; ++request)
            for (int head = 0; head < heads; ++head)
                for (int row = 0; row < rows; ++row) {
                    const auto base = ((request * heads + head) * rows + row) * kChannels;
                    double maximum = -std::numeric_limits<double>::infinity();
                    for (int t = 0; t < tokens; ++t) {
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
