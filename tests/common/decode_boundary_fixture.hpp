#pragma once
#include "single_token_decode_oracle.hpp"

namespace decode_boundary {
constexpr std::uint32_t kCanary = 0x7FC0BEEFu;
// Exactly representable inputs: equal score 3, exp(0)=1, gate=0. Page- and
// channel-dependent integer values expose stale slot/offset/rotation errors.
// Positive even baseline plus one odd channel keeps every dense butterfly
// output nonzero: a Walsh sign sum is even, so it cannot cancel the +1.
// This avoids CANN precise-Div's internal 0/0 diagnostic for zero numerators.
inline int Value(int strip, int channel)
{
    return 2 + 2 * (strip % 3) + (channel == 37 ? 1 : 0);
}
inline single_token_decode::GoldenState Oracle(const std::vector<int>& strips)
{
    using namespace single_token_decode;
    GoldenState out;
    out.m.assign(kRows, 3.0f);
    out.l.assign(kRows, static_cast<float>(16 * strips.size()));
    out.acc.resize(kRowElems);
    out.normalized.resize(kRowElems);
    out.rotated.resize(kRowElems);
    out.context.resize(kRowElems);
    for (int row = 0; row < kRows; ++row) {
        for (int c = 0; c < kHeadSize; ++c) {
            float sum = 0.0f;
            // Scalar oracle sums tokens, independently of the Vector tree.
            for (int strip : strips)
                for (int t = 0; t < kKeys; ++t) sum += Value(strip, c);
            out.acc[row * kHeadSize + c] = sum;
            out.normalized[row * kHeadSize + c] = sum / out.l[row];
        }
        const auto rotated = DenseRotate(out.normalized.data() + row * kHeadSize);
        for (int c = 0; c < kHeadSize; ++c) {
            out.rotated[row * kHeadSize + c] = rotated[c];
            out.context[row * kHeadSize + c] = rotated[c] * 0.5f;
        }
    }
    return out;
}
}  // namespace decode_boundary
