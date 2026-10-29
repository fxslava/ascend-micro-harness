#pragma once
#include "splitk_decode_fixture.hpp"

namespace splitk_decode_test {
constexpr int kAicRows = 16;
constexpr int kAicStateFloats = 2 * kAicRows * 8;
constexpr int kAicAccFloats = kAicRows * kChannels;

struct AicPartial {
    std::vector<double> m, l, acc;
};

// Direct FP64 softmax over the uploaded bytes, independently for each M row.
// Empty ranges carry a finite neutral state so external log-sum-exp combines
// never form inf-inf. Rows share one KV cache, as packed GQA queries do.
inline AicPartial ReferenceAicPartial(const Fixture& f, int request, int head,
                                      unsigned begin, unsigned pages)
{
    AicPartial out;
    out.m.assign(kAicRows, -3e38f);
    out.l.assign(kAicRows, 0);
    out.acc.assign(kAicAccFloats, 0);
    const unsigned first = begin * f.page_tokens, count = pages * f.page_tokens;
    std::vector<double> scores(count);
    for (int row = 0; row < kAicRows && count; ++row) {
        const auto qbase = ((request * f.heads + head) * kAicRows + row) * kChannels;
        for (unsigned i = 0; i < count; ++i) {
            const unsigned t = first + i;
            const auto physical = f.table[request * f.pages + t / f.page_tokens];
            const auto base = (static_cast<std::size_t>(physical) * f.heads + head) *
                              f.page_tokens * 128 + (t % f.page_tokens / 16) * 2048;
            double dot = 0;
            for (int c = 0; c < kChannels; ++c)
                dot += single_token_decode::bfloat16_to_float(f.q[qbase + c]) *
                       static_cast<double>(Fixture::Unpack(f.key.data() + base, (t % 16) * kChannels + c));
            const auto scale = (static_cast<std::size_t>(physical) * f.heads + head) *
                                f.page_tokens + t % f.page_tokens;
            scores[i] = dot * f.scales[scale];
            out.m[row] = std::max(out.m[row], scores[i]);
        }
        for (unsigned i = 0; i < count; ++i) {
            const unsigned t = first + i;
            const auto physical = f.table[request * f.pages + t / f.page_tokens];
            const auto base = (static_cast<std::size_t>(physical) * f.heads + head) *
                              f.page_tokens * 128 + (t % f.page_tokens / 16) * 2048;
            const auto scale = (static_cast<std::size_t>(physical) * f.heads + head) *
                                f.page_tokens + t % f.page_tokens;
            const double p = std::exp(scores[i] - out.m[row]);
            out.l[row] += p;
            const double pv = p * f.scales[f.scales.size() / 2 + scale];
            for (int c = 0; c < kChannels; ++c)
                out.acc[row * kChannels + c] += pv * Fixture::Unpack(f.value.data() + base, c * 16 + t % 16);
        }
    }
    return out;
}

inline std::vector<std::uint16_t> AicRawQuery(const Fixture& f)
{
    std::vector<std::uint16_t> q(f.q.size());
    for (int request = 0; request < f.requests; ++request)
        for (int head = 0; head < f.heads; ++head)
            for (int row = 0; row < kAicRows; ++row)
                for (int c = 0; c < kChannels; ++c)
                    q[((request * f.heads + head) * kAicRows + row) * kChannels + c] =
                        ascend_micro::sim::float_to_bfloat16_rne(
                            paged_decode::QueryEntry(single_token_decode::QueryRowOf(head, row), c));
    return q;
}
}  // namespace splitk_decode_test
