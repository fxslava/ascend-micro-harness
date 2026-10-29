// Translation unit 1 of the protocol header's linkage audit; the second is
// test_wave_protocol_linkage_tu2.cpp and both are driven by
// scripts/audit_device_symbols.py stage 2. See wave_protocol_linkage_body.hpp
// for what this does and does not establish.

#include "wave_protocol_linkage_body.hpp"

#include <cstdint>
#include <cstdio>

// Defined in the other translation unit.
std::uint32_t WaveLinkageTu2(std::uint32_t tokens, std::uint32_t per_core);

int main(int argc, char** argv)
{
    // argc-derived so that nothing here is a constant expression: these have
    // to be real calls, or the audit would pass on code that never referenced
    // the functions at all.
    const std::uint32_t tokens = 512u + static_cast<std::uint32_t>(argc);
    const std::uint32_t per_core = 16u + static_cast<std::uint32_t>(argc) - 1u;
    (void)argv;

    const std::uint32_t one = wave_linkage::Touch<1>(tokens, per_core);
    const std::uint32_t two = WaveLinkageTu2(tokens, per_core);

    // The two instantiations differ only by their tag, so this is the only
    // relation worth asserting and it pins that both really ran.
    if (two != one + 1u) {
        std::printf("FAIL wave protocol linkage: tu1=%u tu2=%u, expected tu2 == tu1 + 1\n",
                    one, two);
        return 1;
    }

    std::printf("PASS wave protocol linkage: two translation units reference every "
                "function in turboquant_wave_protocol.hpp and link with no undefined "
                "and no duplicate symbol (tu1=%u tu2=%u)\n", one, two);
    return 0;
}
