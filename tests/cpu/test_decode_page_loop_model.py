#!/usr/bin/env python3
"""Explore every engine interleaving of the AIC decode operator's PAGE loop.

`test_decode_ring_model.py` covers one page of the two-slot ring on the fused
AIV kernel and the Split-K reduction. It does not cover
`turboquant_paged_decode_splitk_aic`, where the same ring is primed and drained
once per page on three cores, and where a worker carries (m, l, Acc) across
page boundaries. The places that only exist at a boundary -- the two
full-page completion flags, the per-page prime/drain of the intra-core feedback
cycles, and the reuse of the per-strip L1 key page by the next page -- are the
ones this model adds.

Engines are one queue per (core, pipe), in scalar emission order, exactly as
the kernel issues them:

  AIC   MTE1 (L1 -> L0B staging), M (Mmad), FIX (Fixpipe)
  AIV0  MTE2 (packed value + both scale slices), V (dequant, recurrence)
  AIV1  MTE2 (packed key), V (dequant), MTE3 (UB -> L1 CV burst)

Intra-core HardEvent places have capacity one: a SET into an occupied place
loses a token on hardware, so it is an error here and not a disabled
transition. Cross-core mode-4 flags are counters, and the kernel uses them as
such on purpose -- `kFlagKeyReady` deliberately carries several unconsumed
announcements at once because AIV1 has no L1 back edge inside a page -- so each
cross-core place declares the bound it is allowed to reach and exceeding that
bound is the error.

What a page boundary must satisfy, and what the negative controls remove:

  * every intra-core feedback cycle is primed and drained inside the page, so
    no credit crosses the boundary;
  * the L1 key page of strip s, which page p + 1 overwrites, is not written
    until the Cube has announced that page p's scores were all consumed;
  * AIV0 does not prime page p + 1's score ring before the Cube has retired
    page p's, which is what keeps `kFlagScoreFree` inside its bound;
  * the marking is empty when all three cores retire.
"""
from collections import Counter, deque
import itertools
import sys

STAGE_DEPTH = 2
PIPELINE_LAG = 4
MAX_STRIPS = 8  # kStrips: the 128-token page geometry

# Declared bound for each cross-core mode-4 place. A HardEvent place is
# capacity one and is not listed here.
def cross_core_bound(strips):
    return {
        "ValueReady": 1,
        "ScoreReady": 1,
        "ScoreFree": 1,
        # AIV1 runs unthrottled inside a page: with no L1 back edge its only
        # brake is MTE3_V, so it can announce every strip that lands in a
        # given ring slot before the Cube reads the first of them. That is
        # deliberate, and `kFlagKeyReady`'s comment in the kernel records it.
        "KeyReady": (strips + STAGE_DEPTH - 1) // STAGE_DEPTH,
        "PageDrainedV0": 1,
        "PageDrainedV1": 1,
    }


# A HardEvent route names its producing and consuming PIPE, and that is what
# decides which queue each instruction sits in: a SET is issued into the
# producer's queue and a WAIT into the consumer's, so a WAIT stalls the
# consuming engine. Getting this backwards is not a detail -- modelling
# `WaitFlag<V_MTE2>` on the Vector queue instead of MTE2 lets MTE2 run past the
# page drain and steal the next page's credit, which reports a deadlock that
# the hardware does not have.
CORE_PIPES = {
    "AIC": ("MTE1", "M", "FIX"),
    "V0": ("MTE2", "V"),
    "V1": ("MTE2", "V", "MTE3"),
}

ROUTE_PIPES = {
    "V_MTE2": ("V", "MTE2"),
    "MTE2_V": ("MTE2", "V"),
    "MTE3_V": ("MTE3", "V"),
    "V_MTE3": ("V", "MTE3"),
    "M_MTE1": ("M", "MTE1"),
    "MTE1_M": ("MTE1", "M"),
    "FIX_M": ("FIX", "M"),
    "M_FIX": ("M", "FIX"),
}


def slot_of(column):
    return column % STAGE_DEPTH


def schedule(pages, strips, *, no_page_fence=False, carry_credit=False,
             l1_before_drain=False, prime_once=False, no_scalar_fence=False,
             aic_fix_only=False, page_fence_on_mte3=False):
    """Emit the three cores' pipe queues for `pages` pages of `strips` strips.

    Keyword flags are the negative controls, each removing one of the
    properties the docstring lists.
    """
    queues = {name: [] for name in
              ("AIC.MTE1", "AIC.M", "AIC.FIX", "V0.MTE2", "V0.V",
               "V1.MTE2", "V1.V", "V1.MTE3")}

    def emit(pipe, op, *args):
        assert pipe in queues, f"unmodelled pipe {pipe}"
        queues[pipe].append((op, *args))

    def hard(core, op, route, slot):
        """An intra-core HardEvent on `core`, routed to the pipe that issues it."""
        producer, consumer = ROUTE_PIPES[route.split(".")[-1]]
        emit(f"{core}.{producer if op == 'set' else consumer}",
             op, ("hard", f"{core}.{route}", slot))

    def cross(pipe, op, name, slot=0):
        emit(pipe, op, ("cross", name, slot))

    barriers = itertools.count()

    def scalar_drain(core, drained):
        """A `SetFlag<X_S>` / `WaitFlag<X_S>` pair for each pipe X in `drained`.

        Setting on pipe X and waiting on the SCALAR stream stalls instruction
        ISSUE until X has drained. So the fence is one-sided: it holds back
        every later instruction of that core, on every pipe, but it only
        guarantees the pipes actually named in `drained` have retired. A pipe
        not named keeps whatever it had in flight.

        This is what keeps the next page's `SetFlag<V_MTE2>` prime, issued on
        V, from overtaking this page's `WaitFlag<V_MTE2>` drain, which sits on
        the MTE2 queue. The `no_scalar_fence` control is what shows the prime
        double-SETs without it.
        """
        tag = next(barriers)
        for pipe in CORE_PIPES[core]:
            emit(f"{core}.{pipe}", "barrier", (core, tag, tuple(drained)))

    # ---------------------------------------------------------------- AIV1
    def key_page(page):
        # run_key_core: prime, the column schedule, drain.
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page):
                hard("V1", "set", "V_MTE2", slot)
                hard("V1", "set", "MTE3_V", slot)
        for column in range(strips + PIPELINE_LAG):
            strip = column - 1
            if not (0 <= strip < strips):
                continue
            slot = slot_of(strip)
            hard("V1", "wait", "V_MTE2", slot)
            emit("V1.MTE2", "load_key", slot, page, strip)
            hard("V1", "set", "MTE2_V", slot)
            hard("V1", "wait", "MTE2_V", slot)
            hard("V1", "wait", "MTE3_V", slot)
            emit("V1.V", "dequant_key", slot, page, strip)
            hard("V1", "set", "V_MTE2", slot)
            hard("V1", "set", "V_MTE3", slot)
            hard("V1", "wait", "V_MTE3", slot)
            # The CV burst writes L1 key page `strip`, which page - 1 wrote.
            emit("V1.MTE3", "stage_l1", strip, page)
            hard("V1", "set", "MTE3_V", slot)
            cross("V1.MTE3", "set", "KeyReady", slot)
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page + 1 < pages):
                hard("V1", "wait", "V_MTE2", slot)
                hard("V1", "wait", "MTE3_V", slot)

    def key_core():
        for page in range(pages):
            key_page(page)
            if not no_page_fence:
                # Waited on PIPE_V, with the V drain first. A mode-4 wait
                # blocks only its own pipe's issuance, so the pipe it sits on
                # decides nothing by itself -- what holds page p + 1 back is
                # the scalar drain behind it, and that drain has to be on the
                # same pipe as the wait to mean anything. `page_fence_on_mte3`
                # is the construction this replaces.
                lane = "V1.MTE3" if page_fence_on_mte3 else "V1.V"
                cross(lane, "wait", "PageDrainedV1")
                if not no_scalar_fence:
                    order = ("MTE3", "V", "MTE2") if page_fence_on_mte3 else \
                            ("V", "MTE3", "MTE2")
                    scalar_drain("V1", order)

    # ---------------------------------------------------------------- AIV0
    def value_page(page):
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page):
                hard("V0", "set", "V_MTE2", slot)
                cross("V0.V", "set", "ScoreFree", slot)
        for column in range(strips + PIPELINE_LAG):
            recur = column - PIPELINE_LAG
            if 0 <= recur < strips:
                slot = slot_of(recur)
                cross("V0.V", "wait", "ScoreReady", slot)
                # Reads the carried (m, l, Acc) of page `page`, strip `recur`.
                emit("V0.V", "recurrence", slot, page, recur)
                emit("V0.V", "archive", slot, page, recur)
                if not carry_credit:
                    hard("V0", "set", "V_MTE2", slot)
                cross("V0.V", "set", "ScoreFree", slot)
            dequant = column - 3
            if 0 <= dequant < strips:
                slot = slot_of(dequant)
                hard("V0", "wait", "MTE2_V", slot)
                emit("V0.V", "dequant_value", slot, page, dequant)
                cross("V0.V", "set", "ValueReady", slot)
            load = column - 2
            if 0 <= load < strips:
                slot = slot_of(load)
                hard("V0", "wait", "V_MTE2", slot)
                emit("V0.MTE2", "load_value", slot, page, load)
                hard("V0", "set", "MTE2_V", slot)
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page + 1 < pages) and not carry_credit:
                hard("V0", "wait", "V_MTE2", slot)

    def value_core():
        for page in range(pages):
            value_page(page)
            if not no_page_fence:
                cross("V0.V", "wait", "PageDrainedV0")
                if not no_scalar_fence:
                    scalar_drain("V0", ("V", "MTE2"))

    # ----------------------------------------------------------------- AIC
    def gemm_page(page):
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page):
                hard("AIC", "set", "M_MTE1", slot)
                hard("AIC", "set", "FIX_M", slot)
        for column in range(strips + PIPELINE_LAG):
            strip = column - 2
            if not (0 <= strip < strips):
                continue
            slot = slot_of(strip)
            cross("AIC.MTE1", "wait", "KeyReady", slot)
            cross("AIC.MTE1", "wait", "ValueReady", slot)
            hard("AIC", "wait", "M_MTE1", slot)
            emit("AIC.MTE1", "load_l0b", strip, page)
            hard("AIC", "set", "MTE1_M", slot)
            hard("AIC", "wait", "MTE1_M", slot)
            hard("AIC", "wait", "FIX_M", slot)
            emit("AIC.M", "mmad", slot, page, strip)
            hard("AIC", "set", "M_MTE1", slot)
            hard("AIC", "set", "M_FIX", slot)
            hard("AIC", "wait", "M_FIX", slot)
            cross("AIC.FIX", "wait", "ScoreFree", slot)
            emit("AIC.FIX", "fixpipe", slot, page, strip)
            hard("AIC", "set", "FIX_M", slot)
            cross("AIC.FIX", "set", "ScoreReady", slot)
        for slot in range(STAGE_DEPTH):
            if not (prime_once and page + 1 < pages):
                hard("AIC", "wait", "M_MTE1", slot)
                hard("AIC", "wait", "FIX_M", slot)
        if not no_page_fence:
            if l1_before_drain:
                # The control: announce reuse before the scores are consumed.
                cross("AIC.FIX", "set", "PageDrainedV0")
                cross("AIC.FIX", "set", "PageDrainedV1")
                for slot in range(STAGE_DEPTH):
                    cross("AIC.FIX", "wait", "ScoreFree", slot)
            else:
                for slot in range(STAGE_DEPTH):
                    cross("AIC.FIX", "wait", "ScoreFree", slot)
                # FIX_S drains the Fixpipe before announcing that the L1 key
                # pages and this page's mode-4 announcements may be reused.
                # That alone is not enough to let the NEXT page prime: its
                # `SetFlag<M_MTE1>` sits on M while this page's matching WAIT
                # sits on MTE1, and its `SetFlag<FIX_M>` sits on FIX while the
                # matching WAIT sits on M. So M and MTE1 have to drain too.
                # `HardEventAic` has no MTE1_S, so MTE1 drains through
                # MTE1_M into M and M then drains into the scalar stream.
                if not no_scalar_fence:
                    if aic_fix_only:
                        scalar_drain("AIC", ("FIX",))
                    else:
                        scalar_drain("AIC", ("FIX", "MTE1", "M"))
                cross("AIC.FIX", "set", "PageDrainedV0")
                cross("AIC.FIX", "set", "PageDrainedV1")
        else:
            for slot in range(STAGE_DEPTH):
                cross("AIC.FIX", "wait", "ScoreFree", slot)

    def cube_core():
        for page in range(pages):
            gemm_page(page)

    key_core()
    value_core()
    cube_core()
    return {name: tuple(q) for name, q in queues.items()}


class Violation(Exception):
    pass


def explore(queues, pages, strips, cap=4_000_000, sticky_aiv_wait=False):
    names = sorted(queues)
    lanes = [queues[name] for name in names]
    places = sorted({cmd[1] for lane in lanes for cmd in lane
                     if cmd[0] in ("set", "wait")})
    index = {place: i for i, place in enumerate(places)}
    lane_of = {name: i for i, name in enumerate(names)}
    # Where each pipe's copy of each scalar fence sits in its own queue.
    barrier_at = {}
    for name, lane in zip(names, lanes):
        pipe = name.split(".")[1]
        for pc, cmd in enumerate(lane):
            if cmd[0] == "barrier":
                barrier_at[(cmd[1][1], pipe)] = pc
    sets = Counter(cmd[1] for lane in lanes for cmd in lane if cmd[0] == "set")
    waits = Counter(cmd[1] for lane in lanes for cmd in lane if cmd[0] == "wait")
    if sets != waits:
        raise Violation("unbalanced token census: " + ", ".join(
            f"{p}: {sets[p]} set / {waits[p]} wait"
            for p in sorted(set(sets) | set(waits)) if sets[p] != waits[p]))

    bounds = cross_core_bound(strips)

    def capacity(place):
        kind, name, _ = place
        return 1 if kind == "hard" else bounds[name]

    caps = [capacity(place) for place in places]
    # Owner of each L1 key page, and the page whose (m, l, Acc) the recurrence
    # has most recently folded in. -1 is "no page has written it yet".
    # `sticky` is the set of cross-core places whose AIV-side wait has already
    # been released once, for the sticky_aiv_wait semantics knob below.
    start = (tuple([0] * len(lanes)), tuple([0] * len(places)),
             tuple([-1] * strips), -1, 0, 0)
    pending = deque([start])
    seen = {start}
    terminals = 0
    while pending:
        if len(seen) > cap:
            raise Violation(f"state space exceeded {cap} states")
        pcs, marking, l1, folded, recurs, sticky = pending.popleft()
        progress = False
        for lane_id, lane in enumerate(lanes):
            pc = pcs[lane_id]
            if pc == len(lane):
                continue
            op, *rest = lane[pc]
            tokens = list(marking)
            pages_l1 = list(l1)
            next_folded, next_recurs, next_sticky = folded, recurs, sticky
            if op == "set":
                place = rest[0]
                i = index[place]
                if tokens[i] + 1 > caps[i]:
                    raise Violation(
                        f"SET into a full place {place} (bound {caps[i]}) "
                        f"-- hardware loses this token")
                tokens[i] += 1
            elif op == "wait":
                place = rest[0]
                i = index[place]
                # SS6.3's knob, and this operator is where it bit. Under
                # `sticky_aiv_wait` an AIV-side mode-4 wait behaves as a level
                # that the first wait observes and no later wait blocks on --
                # which is what the four-page Split-K trace measured for
                # AIV1's `kFlagPageDrainedV1`: stalls of 23146, 3, 3 and 3
                # ticks against announcements ~20600 ticks apart. A model that
                # assumes the consuming semantics has assumed the answer, so
                # both settings are explored and both verdicts reported.
                if (sticky_aiv_wait and place[0] == "cross"
                        and names[lane_id].startswith("V") and (sticky >> i) & 1):
                    pass
                elif tokens[i] == 0:
                    continue  # the engine blocks here
                else:
                    tokens[i] -= 1
                    if (sticky_aiv_wait and place[0] == "cross"
                            and names[lane_id].startswith("V")):
                        next_sticky = sticky | (1 << i)
            elif op == "barrier":
                core, tag, drained = rest[0]
                # Instruction issue is stalled until every NAMED pipe has
                # retired its pre-fence work -- that is, has reached this same
                # fence. Pipes not named are held back but not drained.
                if any(pcs[lane_of[f"{core}.{p}"]] < barrier_at[(tag, p)]
                       for p in drained):
                    continue
            elif op == "stage_l1":
                strip, page = rest
                # Page p + 1 may only overwrite the L1 key page once the Cube
                # has read page p's -- which the page fence is what guarantees.
                if pages_l1[strip] not in (-1, page - 1):
                    raise Violation(
                        f"L1 key page {strip} rewritten by page {page} while "
                        f"page {pages_l1[strip]} was still live")
                pages_l1[strip] = page
            elif op == "load_l0b":
                strip, page = rest
                if pages_l1[strip] != page:
                    raise Violation(
                        f"Cube read L1 key page {strip} holding page "
                        f"{pages_l1[strip]} while multiplying page {page}")
            elif op == "recurrence":
                _, page, strip = rest
                # The carried state is a single sequential accumulator: the
                # recurrence of (page, strip) must follow its predecessor.
                expect = recurs
                actual = page * strips + strip
                if actual != expect:
                    raise Violation(
                        f"recurrence of page {page} strip {strip} ran out of "
                        f"order (expected sequence position {expect})")
                next_recurs = expect + 1
                next_folded = page
            state = (tuple(pcs[:lane_id] + (pc + 1,) + pcs[lane_id + 1:]),
                     tuple(tokens), tuple(pages_l1), next_folded, next_recurs,
                     next_sticky)
            progress = True
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not progress:
            if any(pcs[i] != len(lanes[i]) for i in range(len(lanes))):
                stuck = [names[i] for i in range(len(lanes))
                         if pcs[i] != len(lanes[i])]
                raise Violation("deadlock / unprimed cycle on " + ", ".join(stuck))
            if any(tokens for tokens in marking):
                raise Violation("non-empty final marking: " + ", ".join(
                    f"{places[i]}={marking[i]}" for i in range(len(places))
                    if marking[i]))
            if recurs != pages * strips:
                raise Violation(
                    f"only {recurs} of {pages * strips} recurrences retired")
            terminals += 1
    if not terminals:
        raise Violation("no terminal state reached")
    return len(seen)


# Exhaustively explorable geometries. One page is covered at the full
# eight-strip page geometry; the multi-page rungs are covered at one, two and
# four strips.
GRID = [(1, 1), (1, 2), (1, 4), (1, 8),
        (2, 1), (2, 2), (2, 4),
        (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (6, 1)]

# Beyond exhaustive interleaving exploration here: eight strips on three
# asynchronous cores over more than one page is tens of millions of states with
# no partial-order reduction in this model. It is NOT covered by this gate, and
# the CAModel run at `--page-tokens 128` over two pages is what carries that
# geometry instead. Recorded rather than silently dropped.
BEYOND_EXHAUSTIVE = [(2, 8), (4, 8)]

CONTROLS = {
    # The state this kernel was in before the page-boundary fix: the AIC
    # drained only its Fixpipe at the boundary, so the next page's M-queue and
    # FIX-queue primes could overtake this page's MTE1-queue and M-queue
    # drains. This control is the regression test for that fix.
    "aic_fix_only": "the AIC draining only FIX at the page boundary",
    "no_scalar_fence": "the per-page X_S scalar drains removed",
    "no_page_fence": "the full-page completion flags removed",
    "carry_credit": "a V_MTE2 credit carried across the page boundary",
    "l1_before_drain": "L1 reuse announced before the scores were consumed",
    "prime_once": "the feedback cycles primed once instead of per page",
}


def main():
    failures = 0
    for pages, strips in GRID:
        try:
            states = explore(schedule(pages, strips), pages, strips)
        except Violation as e:
            print(f"FAIL pages={pages} strips={strips}: {e}")
            failures += 1
            continue
        print(f"PASS pages={pages} strips={strips}: {states} reachable states, "
              f"empty final marking, {pages * strips} ordered recurrences")

    for pages, strips in BEYOND_EXHAUSTIVE:
        print(f"NOT COVERED pages={pages} strips={strips}: beyond exhaustive "
              f"interleaving exploration in this model; covered on CAModel only")

    # Each control must be rejected, and on a geometry where it can bite: a
    # page boundary, which pages=1 does not have.
    for control, description in CONTROLS.items():
        rejected = []
        for pages, strips in ((2, 1), (2, 2), (2, 8), (4, 1)):
            try:
                explore(schedule(pages, strips, **{control: True}), pages, strips)
            except Violation as e:
                rejected.append(f"pages={pages} strips={strips}: {e}")
                break
        if rejected:
            print(f"PASS rejected negative control: {control} "
                  f"({description}) -- {rejected[0]}")
        else:
            print(f"FAIL negative control {control} ({description}) was accepted")
            failures += 1

    # ---- the flag-semantics knob, reported and not gated -----------------
    # REFINEMENT_OF_COMBAT_KERNELS.md SS6.3: two hardware properties decide
    # whether channel recycling is usable, and a model that assumes either has
    # assumed the answer. Everything above assumes an AIV-side mode-4 wait
    # CONSUMES the flag it tested. The four-page Split-K trace measured the
    # other half for `kFlagPageDrainedV1` on PIPE_MTE3 -- stalls of 23146, 3,
    # 3 and 3 ticks against announcements ~20600 ticks apart -- so the verdict
    # under the sticky reading belongs here too.
    #
    # It is REPORTED rather than gated because the token structure cannot fix
    # it: under a sticky AIV wait no arrangement of these flags orders AIV1's
    # L1 reuse, including the arrangement the kernel now ships. What settles
    # it for a given toolkit is the stall census in
    # `scripts/audit_mode4_tick_order.py` over that revision's own trace, and
    # that is the gate. This section exists so the dependence is on the
    # record instead of implicit.
    print()
    for label, flags in (("as shipped (page fence on PIPE_V)", {}),
                         ("pre-fix (page fence on PIPE_MTE3)",
                          {"page_fence_on_mte3": True})):
        verdicts = []
        for pages, strips in ((2, 1), (4, 1)):
            try:
                explore(schedule(pages, strips, **flags), pages, strips,
                        sticky_aiv_wait=True)
                verdicts.append(f"pages={pages}: safe even so")
            except Violation as e:
                verdicts.append(f"pages={pages}: {e}")
        print(f"SEMANTICS sticky AIV mode-4 wait, {label}: " + "; ".join(verdicts))
    print("SEMANTICS the consuming reading is the one measured for this "
          "revision; see reports/decode_splitk_aic_page_race.md")

    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print(f"\nPASS AIC page loop: {len(GRID)} geometries 1-safe across page "
          f"boundaries, ordered carried state, finite drain; "
          f"{len(CONTROLS)} negative controls; "
          f"{len(BEYOND_EXHAUSTIVE)} geometries not covered here; "
          f"flag-semantics knob reported for both readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
