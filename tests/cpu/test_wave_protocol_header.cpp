// Host compile-time and census gate for the MIX_AIC_1_2 inter-wave protocol.
//
// REFINEMENT_OF_COMBAT_KERNELS.md section 2 wants the compile-time memory and
// id checks run under -Wall -Werror. The two reshape-and-cache kernels can only
// be compiled by BiSheng against a CANN 9.2 toolkit, so on a host without one
// that gate is unavailable for the kernel bodies. What *is* available is the
// part of those checks that is pure arithmetic: every static_assert in
// turboquant_wave_protocol.hpp -- the flag-id budgets of both channel
// policies, the channel disjointness, the release/wait pairing, the strip
// boundary and the AKA3003 arena algebra -- lives in a header with no CCE
// dependency, and both kernels include that header rather than carrying their
// own copy. So this translation unit exercises the same declarations the device
// build uses.
//
// On top of the static_asserts it runs the census guidelines section 6.2c asks
// for by name, over the actual schedule, for each policy:
//
//   * the recycled policy -- the production default -- must keep at most ONE
//     outstanding notification per id, the property its 41.8M-state proof
//     rests on, which here means the per-id operation sequence alternates
//     SET, WAIT, SET, WAIT in the order the handshake chain forces;
//   * the single-shot policy, the regression fallback, must give every
//     cross-core event its own id, so no id ever carries two announcements.
//
// It is a *gate*, not a test of the kernels: it proves the protocol arithmetic
// the kernels are built on, and says nothing about their instruction
// sequences, their buffer lifetimes or their numerics. Those need
// tests/cpu/test_reshape_wave_model.py (token flow) and CAModel (everything
// else).
//
// Build and run:
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror -I.
//       tests/cpu/test_wave_protocol_header.cpp -o bin/test_wave_protocol_header
//   bin/test_wave_protocol_header

#include "../../kernels/ops/turboquant/turboquant_wave_protocol.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using turboquant_wave::kIntraNum;
using turboquant_wave::kMaxSafeLocalId;
using turboquant_wave::kReshapeChunks;
using turboquant_wave::kReshapeLandDepth;
using turboquant_wave::LegalIntraId;

int failures = 0;

void expect(bool condition, const std::string& what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++failures;
}

// One cross-core operation, in the order the handshake chain forces it. The
// chain is what makes this a total order per id rather than a guess: the
// Cube's set on productReady for generation j is behind its own wait on the
// panelFree credit of that landing slot, which is behind the owning Vector
// core's release of generation j - LandDepth, which is behind that core's wait
// on productReady for j - LandDepth.
struct Op {
    std::uint16_t id;
    bool is_set;
    std::string what;
};

// The cross-core operations of one block's body, per sub block, in that order.
// This mirrors both kernels' schedules: prologue priming, then per generation
// the Cube's announcement and the Vector core's release, then the Cube's
// epilogue drain of the releases nothing else waits for, then the pageDrained
// join.
template <class P>
std::vector<Op> schedule_ops(std::uint32_t owner)
{
    std::vector<Op> ops;
    const std::uint32_t generations = kReshapeChunks / 2u;
    for (std::uint32_t slot = 0; slot < kReshapeLandDepth; ++slot)
        ops.push_back({P::PanelFreePrime(owner, slot), true,
                       "prime panelFree slot " + std::to_string(slot)});
    for (std::uint32_t j = 0; j < generations; ++j) {
        // Chunk owned by this core in generation j.
        const std::uint32_t chunk = 2u * j + owner;
        ops.push_back({P::PanelFreeWait(owner, chunk), false,
                       "AIC waits panelFree for chunk " + std::to_string(chunk)});
        ops.push_back({P::ProductReady(owner, chunk), true,
                       "AIC sets productReady for chunk " + std::to_string(chunk)});
        ops.push_back({P::ProductReady(owner, chunk), false,
                       "AIV waits productReady for chunk " + std::to_string(chunk)});
        ops.push_back({P::PanelFreeSet(owner, chunk), true,
                       "AIV sets panelFree after chunk " + std::to_string(chunk)});
    }
    for (std::uint32_t k = 0; k < kReshapeLandDepth; ++k)
        ops.push_back({P::EpilogueDrain(owner, k), false,
                       "AIC epilogue drains panelFree " + std::to_string(k)});
    ops.push_back({P::PageDrained(owner), true, "AIC sets pageDrained"});
    ops.push_back({P::PageDrained(owner), false, "AIV waits pageDrained"});
    return ops;
}

// Properties every policy must have, whatever its id allocation.
template <class P>
void check_policy(const char* name)
{
    const std::string tag = std::string(name) + ": ";
    std::vector<Op> all;
    for (std::uint32_t owner = 0; owner < 2u; ++owner) {
        const std::vector<Op> ops = schedule_ops<P>(owner);
        all.insert(all.end(), ops.begin(), ops.end());
    }

    // 1. Every id is legal: inside its sub block's half and clear of the
    //    library's reserved range.
    for (const Op& op : all)
        expect(LegalIntraId(op.id), tag + "reserved id " + std::to_string(op.id) +
                                        " used by: " + op.what);

    // 2. The two sub blocks occupy disjoint halves, in both directions.
    for (const Op& op : all)
        expect((op.id < kIntraNum) == (op.id <= kMaxSafeLocalId),
               tag + "id straddles the sub-block halves: " + op.what);

    // 3. Sets and waits balance per id, so nothing leaks past the block.
    std::map<std::uint16_t, int> balance;
    for (const Op& op : all) balance[op.id] += op.is_set ? 1 : -1;
    for (const auto& entry : balance)
        expect(entry.second == 0,
               tag + "id " + std::to_string(entry.first) + " leaves " +
                   std::to_string(entry.second) + " token(s) at block exit");

    // 4. At most ONE outstanding notification per id: the per-id sequence
    //    alternates SET, WAIT, SET, WAIT. This is the property the recycled
    //    policy's proof rests on, and tests/research/kernel_cluster_sync
    //    _probe.cpp confirmed the consuming semantics it needs on silicon and
    //    on CAModel -- provided every announcement is behind its x_S drain,
    //    which scripts/audit_cross_core_fences.py checks separately. The
    //    single-shot policy satisfies it trivially because each id appears
    //    exactly twice.
    std::map<std::uint16_t, std::vector<const Op*>> per_id;
    for (const Op& op : all) per_id[op.id].push_back(&op);
    for (const auto& entry : per_id) {
        bool want_set = true;
        for (const Op* op : entry.second) {
            expect(op->is_set == want_set,
                   tag + "id " + std::to_string(entry.first) +
                       " breaks SET/WAIT alternation at: " + op->what);
            want_set = !want_set;
        }
    }

    // 5. The budget: the policy's highest local id, and the id count it costs.
    std::set<std::uint16_t> distinct;
    for (const Op& op : all) distinct.insert(op.id);
    expect(P::kMaxLocalId <= kMaxSafeLocalId,
           tag + "the policy's highest id is reserved");
    std::printf("  %-11s %2zu distinct ids over both sub blocks, highest local "
                "id %u of %u, %zu cross-core operations\n",
                name, distinct.size(), P::kMaxLocalId, kMaxSafeLocalId, all.size());
}

}  // namespace

int main()
{
    std::printf("turboquant wave protocol, %u chunks over a depth-%u landing "
                "ring, default policy %s\n",
                kReshapeChunks, kReshapeLandDepth,
                turboquant_wave::kReshapeDefaultReuse ? "recycled"
                                                      : "single-shot");
    check_policy<turboquant_wave::ReshapeRecycled>("recycled");
    check_policy<turboquant_wave::ReshapeSingleShot>("single-shot");

    // The single-shot policy's defining property, over and above the shared
    // ones: no id carries two announcements at all, so no reading of the flag
    // mechanism can make a second wait pass early.
    {
        std::set<std::uint16_t> seen;
        for (std::uint32_t owner = 0; owner < 2u; ++owner)
            for (const Op& op : schedule_ops<turboquant_wave::ReshapeSingleShot>(owner))
                if (op.is_set)
                    expect(seen.insert(op.id).second,
                           "single-shot: a mode-4 id carries two sets: " + op.what);
        expect(seen.size() == 22u,
               "single-shot: expected 22 distinct announcements, got " +
                   std::to_string(seen.size()));
    }

    // The recycled policy's defining property: a bounded id count that does
    // not grow with the schedule, which is the whole reason to want it.
    {
        std::set<std::uint16_t> ids;
        for (std::uint32_t owner = 0; owner < 2u; ++owner)
            for (const Op& op : schedule_ops<turboquant_wave::ReshapeRecycled>(owner))
                ids.insert(op.id);
        expect(ids.size() == 2u * (2u * kReshapeLandDepth + 1u),
               "recycled: expected 2 * (2 * LandDepth + 1) ids, got " +
                   std::to_string(ids.size()));
        expect(turboquant_wave::ReshapeRecycled::kMaxLocalId <
                   turboquant_wave::ReshapeSingleShot::kMaxLocalId,
               "recycled: the policy buys no id headroom");
    }

    // The canonical policy. A silent flip of kReshapeDefaultReuse would
    // otherwise pass every check above, because both policies are legal: what
    // decides which one ships is the evidence in guidelines section 6.2c, and
    // this is where that decision is pinned.
    expect(turboquant_wave::kReshapeDefaultReuse,
           "the default policy is no longer the recycled ring; if that is "
           "deliberate, update guidelines section 6.2c and this gate together");
    expect(turboquant_wave::ReshapeDefault::kRecycles ==
               turboquant_wave::kReshapeDefaultReuse,
           "the ReshapeDefault alias does not follow kReshapeDefaultReuse");

    // The boundary guard, on every rung of the wave ladder.
    for (const int strips : {1, 32, 33, 48, 64, 128}) {
        const auto s = static_cast<std::uint32_t>(strips);
        const std::uint32_t tokens = s * 16u;
        expect(turboquant_wave::TotalStrips(tokens, 16u) == s,
               "TotalStrips disagrees with ladder rung " + std::to_string(strips));
        expect(!turboquant_wave::BlockOutOfRange(s - 1u, tokens, 16u),
               "the last in-range block is guarded out at " + std::to_string(strips));
        expect(turboquant_wave::BlockOutOfRange(s, tokens, 16u),
               "the first out-of-range block is not guarded at " +
                   std::to_string(strips));
    }
    // A zero strip width must guard every block rather than divide by zero.
    expect(turboquant_wave::BlockOutOfRange(0u, 512u, 0u),
           "a zero tokensPerCore does not guard block 0");

    // A longer schedule must NOT fit the single-shot policy: Channels<10, 2>
    // would need id 11, the first one the library may claim, so the header
    // refuses to instantiate it. Checked as arithmetic, because a
    // static_assert cannot be asserted to fire.
    {
        constexpr std::uint32_t chunks = 10u;
        constexpr std::uint32_t needed = chunks / 2u + (chunks / 2u + kReshapeLandDepth);
        expect(needed > kMaxSafeLocalId,
               "a ten-chunk single-shot schedule appears to fit the legal id space");
        // The recycled policy is what such a schedule would need, and it fits.
        expect(2u * kReshapeLandDepth <= kMaxSafeLocalId,
               "the recycled policy does not fit either");
    }

    if (failures != 0) {
        std::fprintf(stderr, "FAIL turboquant wave protocol header: %d check(s)\n",
                     failures);
        return 1;
    }
    std::printf("PASS turboquant wave protocol header: both channel policies legal, "
                "balanced, alternating and within budget; boundary guard exact on "
                "1/32/33/48/64/128 strips and on a zero strip width\n");
    return 0;
}
