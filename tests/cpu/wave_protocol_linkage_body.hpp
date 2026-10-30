#pragma once
// One call site for every function in turboquant_wave_protocol.hpp, with
// arguments the compiler cannot fold, instantiated once per translation unit
// of the linkage audit (scripts/audit_device_symbols.py, stage 2).
//
// The point is not the value. It is that two translation units which both
// include the protocol header and both reference every function in it link
// together: no undefined symbol, which is the failure directive 1 names
// (`ld.lld: undefined symbol: turboquant_wave::TotalStrips`), and no duplicate
// definition, which is what dropping `constexpr`/`inline` from one of these
// functions would produce instead.
//
// What this cannot see: the *core type*. `__aicore__` only exists in a device
// pass, so a host link proves linkage and ODR and says nothing about whether
// a function is callable from device code. Stage 1 of the audit checks that
// property on the source, and stage 3 on the device objects when they exist.

#include "../../kernels/ops/turboquant/turboquant_wave_protocol.hpp"

#include <cstdint>

namespace wave_linkage {

// `Tag` keeps the two translation units' instantiations distinct, so each one
// really does emit its own set of references rather than sharing one.
template <int Tag>
std::uint32_t Touch(std::uint32_t tokens, std::uint32_t per_core)
{
    using namespace turboquant_wave;
    using Single = ReshapeSingleShot;
    using Recycled = ReshapeRecycled;

    const std::uint32_t owner = tokens & 1u;
    const std::uint32_t chunk = per_core & 7u;

    std::uint32_t acc = TotalStrips(tokens, per_core);
    acc += BlockOutOfRange(tokens, tokens, per_core) ? 1u : 0u;
    acc += IntraId(owner, chunk);
    acc += LegalIntraId(static_cast<std::uint16_t>(tokens & 15u)) ? 1u : 0u;
    acc += RoundUpBlock(tokens);
    acc += NextSlot(tokens, per_core);
    acc += SlotStrideSkewed(tokens) ? 1u : 0u;
    acc += BaseAligned(tokens) ? 1u : 0u;
    acc += AllIdsLegal<Single>() ? 1u : 0u;
    acc += AllIdsLegal<Recycled>() ? 1u : 0u;
    acc += RolesDisjoint<Single>() ? 1u : 0u;
    acc += RolesDisjoint<Recycled>() ? 1u : 0u;

    acc += Single::ProductReady(owner, chunk);
    acc += Single::PanelFree(owner, chunk);
    acc += Single::PanelFreeWait(owner, chunk);
    acc += Single::PanelFreeSet(owner, chunk);
    acc += Single::PanelFreePrime(owner, owner);
    acc += Single::EpilogueDrain(owner, owner);
    acc += Single::PageDrained(owner);

    acc += Recycled::ProductReady(owner, chunk);
    acc += Recycled::PanelFree(owner, chunk);
    acc += Recycled::PanelFreeWait(owner, chunk);
    acc += Recycled::PanelFreeSet(owner, chunk);
    acc += Recycled::PanelFreePrime(owner, owner);
    acc += Recycled::EpilogueDrain(owner, owner);
    acc += Recycled::PageDrained(owner);

    return acc + static_cast<std::uint32_t>(Tag);
}

}  // namespace wave_linkage
