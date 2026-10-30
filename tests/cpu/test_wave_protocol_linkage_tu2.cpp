// Translation unit 2 of the protocol header's linkage audit. It exists so the
// header is included by more than one translation unit of the same program:
// that is what turns a missing `inline`/`constexpr` into a duplicate-symbol
// link error, and a missing definition into the undefined-symbol error
// directive 1 names.

#include "wave_protocol_linkage_body.hpp"

#include <cstdint>

std::uint32_t WaveLinkageTu2(std::uint32_t tokens, std::uint32_t per_core)
{
    return wave_linkage::Touch<2>(tokens, per_core);
}
