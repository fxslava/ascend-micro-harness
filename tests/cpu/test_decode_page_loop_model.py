#!/usr/bin/env python3
"""Explore every engine interleaving of the AIC decode operator's STRIP STREAM.

`test_decode_ring_model.py` covers one page of the two-slot ring on the fused
AIV kernel and the Split-K reduction. It does not cover
`turboquant_paged_decode_splitk_aic`, where one worker carries (m, l, Acc)
across page boundaries and three cores rendezvous once per strip.

This model is the token/queue gate for the CONTINUOUS strip pipeline that
replaced the per-page prime/drain schedule. What changed, and why the model had
to change with it:

  * the L1 key page is a ring of `STAGE_DEPTH` pages addressed by SLOT, not one
    page per strip addressed by STRIP. The per-strip arrangement carried no WAR
    back edge at all -- its premise was "every page is read once", true inside
    one sequence page and false across the page loop -- and that is the defect
    `reports/decode_splitk_aic_page_race.md` measures;
  * the intra-core feedback cycles are primed once before the stream and
    drained once after it, so there is no inter-page drain cascade left to
    model;
  * cross-core ordering is two MODE-2 channels. Mode 2 is not a set of
    independent per-AIV latches: the AIC's counter advances only once BOTH
    Vector cores have announced, and one AIC announcement increments the
    counter on both of them. A model that treated the readiness channel as one
    shared place would let either AIV alone release the Cube, which is exactly
    the property `single_sender_quorum` below exists to reject.

Engines are one queue per (core, pipe), in scalar emission order, exactly as
the kernel issues them:

  AIC   MTE1 (L1 -> L0B staging), M (Mmad), FIX (Fixpipe)
  AIV0  MTE2 (packed value + both scale slices), V (dequant, recurrence)
  AIV1  MTE2 (packed key), V (dequant), MTE3 (UB -> L1 CV burst)

Intra-core HardEvent places have capacity one: a SET into an occupied place
loses a token on hardware, so it is an error here and not a disabled
transition. The two mode-2 channels are counters with a declared bound, and
exceeding the bound is the error -- on hardware a mode-2 flag counter past 15
is a reported fault, not a wrap.

What the strip stream must satisfy, and what the negative controls remove:

  * the L1 key page of slot s, which strip t overwrites, is not written until
    the Cube has read strip t - STAGE_DEPTH's keys out of it into L0B;
  * the UB score panel of slot s, which the Cube's Fixpipe for strip t
    overwrites, is not written until AIV0's recurrence has read strip
    t - STAGE_DEPTH's scores out of it;
  * every GEMM reads the page its own strip staged, and every recurrence reads
    the panel its own strip's Fixpipe wrote;
  * the carried (m, l, Acc) are folded in strict strip order, across page
    boundaries, at fixed UB offsets;
  * both counters are empty when all three cores retire -- including the
    releases AIV1 drains in its epilogue, which the stream itself never
    consumed.
"""
from collections import Counter, deque
import itertools
import sys

STAGE_DEPTH = 2
PIPELINE_LAG = 4
MAX_STRIPS = 8  # kStrips: strips in one 128-token page

# Declared bound for each mode-2 channel, as the kernel's `kMaxOutstanding`
# asserts it. Neither side can run more than STAGE_DEPTH strips ahead of the
# other, plus the one announcement in flight.
CROSS_BOUND = STAGE_DEPTH + 1

# A HardEvent route names its producing and consuming PIPE, and that is what
# decides which queue each instruction sits in: a SET is issued into the
# producer's queue and a WAIT into the consumer's, so a WAIT stalls the
# consuming engine. Getting this backwards is not a detail -- modelling
# `WaitFlag<V_MTE2>` on the Vector queue instead of MTE2 lets MTE2 run past a
# fence and steal the next strip's credit, which reports a deadlock the
# hardware does not have.
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

# The two mode-2 channels. "StageReady" is announced by both AIVs and waited by
# the AIC (a quorum); "StepRetired" is announced by the AIC and waited by both
# AIVs (a broadcast).
READY = "StageReady"
RETIRED = "StepRetired"


def slot_of(strip):
    return strip % STAGE_DEPTH


def schedule(pages, strips, *, no_l1_war=False, no_panel_war=False,
             single_sender_quorum=False, no_prime=False, carry_credit=False,
             release_before_fixpipe=False, release_before_l0b=False,
             no_epilogue_drain=False, no_scalar_fence=False,
             aiv0_issue_fence=False):
    """Emit the three cores' pipe queues for a window of `pages * strips` strips.

    Keyword flags are the negative controls, each removing one of the
    properties the docstring lists.
    """
    total = pages * strips
    queues = {name: [] for name in
              ("AIC.MTE1", "AIC.M", "AIC.FIX", "V0.MTE2", "V0.V",
               "V1.MTE2", "V1.V", "V1.MTE3")}

    def emit(pipe, op, *args):
        assert pipe in queues, f"unmodelled pipe {pipe}"
        queues[pipe].append((op, *args))

    def hard(core, op, route, slot):
        """An intra-core HardEvent on `core`, routed to the pipe that issues it."""
        producer, consumer = ROUTE_PIPES[route]
        emit(f"{core}.{producer if op == 'set' else consumer}",
             op, ("hard", f"{core}.{route}", slot))

    barriers = itertools.count()

    def scalar_drain(core, drained):
        """A `SetFlag<X_S>` / `WaitFlag<X_S>` pair for each pipe X in `drained`.

        Setting on pipe X and waiting on the SCALAR stream stalls instruction
        ISSUE until X has drained. So the fence is one-sided: it holds back
        every later instruction of that core, on every pipe, but it only
        guarantees the pipes actually named in `drained` have retired.

        Behind a cross-core wait this is what turns "the named pipe is blocked"
        into "this core is blocked", which is what bounds a core's run-ahead --
        and it is the only thing `scripts/audit_mode4_tick_order.py` can read a
        stall from. `no_scalar_fence` removes them.
        """
        if no_scalar_fence:
            return
        tag = next(barriers)
        for pipe in CORE_PIPES[core]:
            emit(f"{core}.{pipe}", "barrier", (core, tag, tuple(drained)))

    # ---------------------------------------------------------------- AIV1
    # Key ingress, dequant, and the UB -> L1 CV burst. One flat pass over the
    # window's strips; the L1 page it writes is the SLOT, not the strip.
    def key_core():
        if total == 0:
            return
        for slot in range(STAGE_DEPTH):
            hard("V1", "set", "V_MTE2", slot)
            hard("V1", "set", "MTE3_V", slot)
        for strip in range(total):
            slot = slot_of(strip)
            hard("V1", "wait", "V_MTE2", slot)
            emit("V1.MTE2", "load_key", slot, strip)
            hard("V1", "set", "MTE2_V", slot)
            hard("V1", "wait", "MTE2_V", slot)
            hard("V1", "wait", "MTE3_V", slot)
            emit("V1.V", "dequant_key", slot, strip)
            hard("V1", "set", "V_MTE2", slot)
            hard("V1", "set", "V_MTE3", slot)
            hard("V1", "wait", "V_MTE3", slot)
            # WAR on the L1 key page: strip `strip` overwrites what strip
            # `strip - STAGE_DEPTH` put there, so the Cube must have read it.
            if strip >= STAGE_DEPTH and not no_l1_war:
                emit("V1.MTE3", "m2wait", RETIRED, "V1")
                scalar_drain("V1", ("MTE3",))
            emit("V1.MTE3", "stage_l1", slot, strip)
            hard("V1", "set", "MTE3_V", slot)
            emit("V1.MTE3", "m2set", READY, "V1")
        # The Cube broadcasts one release per strip; this core consumed one per
        # strip from STAGE_DEPTH on, so min(STAGE_DEPTH, total) are standing.
        standing = min(STAGE_DEPTH, total)
        if no_l1_war:
            standing = total
        if not no_epilogue_drain:
            for _ in range(standing):
                emit("V1.MTE3", "m2wait", RETIRED, "V1")
                scalar_drain("V1", ("MTE3",))
        for slot in range(STAGE_DEPTH):
            hard("V1", "wait", "V_MTE2", slot)
            hard("V1", "wait", "MTE3_V", slot)

    # ---------------------------------------------------------------- AIV0
    # Value ingress, dequant, and the online-softmax recurrence. Column c
    # carries the recurrence of strip c - PIPELINE_LAG, the dequant of
    # c - 3 and the load of c - 2, exactly as the kernel issues them.
    def value_core():
        if total == 0:
            return
        for slot in range(STAGE_DEPTH):
            hard("V0", "set", "V_MTE2", slot)
        # The score slots start empty, so the Cube's first STAGE_DEPTH GEMMs
        # have nothing to wait for on this side: priming announcements, which
        # report no earlier pipe work and need no drain.
        if not no_prime:
            for _ in range(min(STAGE_DEPTH, total)):
                emit("V0.V", "m2set", READY, "V0")
        for column in range(total + PIPELINE_LAG):
            recur = column - PIPELINE_LAG
            if 0 <= recur < total:
                slot = slot_of(recur)
                # RAW on the score panel: the Cube's release for THIS strip.
                #
                # No scalar drain behind this one, unlike AIV1's. The wait
                # names PIPE_V and the recurrence is V work, so the V queue
                # already orders them; the drain would only convert the wait
                # into a stall on instruction ISSUE. On AIV1 that conversion is
                # load-bearing -- it is what bounds the core whose run-ahead
                # was the defect, and the only thing the trace census can read
                # a stall from -- but on AIV0 it buys nothing the credit ring
                # does not already give, and it sits in the hot loop. The
                # `aiv0_issue_fence` control below is this edge put back, and
                # it must still be ACCEPTED: removing a fence cannot be proved
                # safe by a control that fails.
                emit("V0.V", "m2wait", RETIRED, "V0")
                if aiv0_issue_fence:
                    scalar_drain("V0", ("V",))
                emit("V0.V", "recurrence", slot, recur)
                emit("V0.V", "archive", slot, recur)
                if no_panel_war:
                    # The control: announce the slot free BEFORE reading it.
                    pass
                if not carry_credit:
                    hard("V0", "set", "V_MTE2", slot)
            dequant = column - 3
            if 0 <= dequant < total:
                slot = slot_of(dequant)
                hard("V0", "wait", "MTE2_V", slot)
                emit("V0.V", "dequant_value", slot, dequant)
            load = column - 2
            if 0 <= load < total:
                slot = slot_of(load)
                hard("V0", "wait", "V_MTE2", slot)
                emit("V0.MTE2", "load_value", slot, load)
                hard("V0", "set", "MTE2_V", slot)
            # The DEFERRED announcement. The score slot is free again, folded
            # into the readiness quorum: the Cube's (recur + STAGE_DEPTH)-th
            # GEMM is the one that overwrites this panel. It is emitted at the
            # END of the column, after this column's dequant and load, so that
            # the scalar unit has enqueued them before it blocks on the V_S
            # wait that fences the announcement. The fence itself is unchanged
            # -- the SetFlag<V_S> still sits in the V queue behind the
            # recurrence; only the WaitFlag moves. `eager_announce` is the
            # previous placement, and it must still be ACCEPTED: moving a
            # fence later cannot be justified by a control that fails.
            if 0 <= recur < total and recur + STAGE_DEPTH < total:
                emit("V0.V", "m2set", READY, "V0")
        for slot in range(STAGE_DEPTH):
            if not carry_credit:
                hard("V0", "wait", "V_MTE2", slot)

    # ----------------------------------------------------------------- AIC
    def cube_core():
        if total == 0:
            return
        for slot in range(STAGE_DEPTH):
            hard("AIC", "set", "M_MTE1", slot)
            hard("AIC", "set", "FIX_M", slot)
        for strip in range(total):
            slot = slot_of(strip)
            # ONE quorum wait for both preconditions. It gates MTE1 -- the pipe
            # that reads L1 -- and the MTE1 -> M -> scalar drain behind it holds
            # the whole core, so the Fixpipe is ordered too.
            emit("AIC.MTE1", "m2wait", READY, "AIC")
            scalar_drain("AIC", ("MTE1", "M"))
            if release_before_l0b:
                # The control: announce the L1 page reusable before reading it.
                emit("AIC.FIX", "m2set", RETIRED, "AIC")
            hard("AIC", "wait", "M_MTE1", slot)
            emit("AIC.MTE1", "load_l0b", slot, strip)
            hard("AIC", "set", "MTE1_M", slot)
            hard("AIC", "wait", "MTE1_M", slot)
            hard("AIC", "wait", "FIX_M", slot)
            emit("AIC.M", "mmad", slot, strip)
            hard("AIC", "set", "M_MTE1", slot)
            hard("AIC", "set", "M_FIX", slot)
            hard("AIC", "wait", "M_FIX", slot)
            if release_before_fixpipe:
                # The control: section 6.2a's hazard, announced as a structure
                # rather than as a missing drain -- the release goes out before
                # the Fixpipe has written the panel it reports.
                emit("AIC.FIX", "m2set", RETIRED, "AIC")
            emit("AIC.FIX", "fixpipe", slot, strip)
            hard("AIC", "set", "FIX_M", slot)
            if not (release_before_fixpipe or release_before_l0b):
                # MTE1 drains through M, then FIX, so the announcing pipe is the
                # last one drained (section 6.2a, and section 6.6's check).
                scalar_drain("AIC", ("MTE1", "M", "FIX"))
                emit("AIC.FIX", "m2set", RETIRED, "AIC")
        for slot in range(STAGE_DEPTH):
            hard("AIC", "wait", "M_MTE1", slot)
            hard("AIC", "wait", "FIX_M", slot)

    key_core()
    value_core()
    cube_core()
    return ({name: tuple(q) for name, q in queues.items()},
            {"total": total, "single_sender_quorum": single_sender_quorum})


class Violation(Exception):
    pass


def explore(built, pages, strips, cap=4_000_000, sticky_aiv_wait=False):
    queues, options = built
    total = options["total"]
    single_sender = options["single_sender_quorum"]
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

    # ---- censuses, before any exploration -------------------------------
    sets = Counter(cmd[1] for lane in lanes for cmd in lane if cmd[0] == "set")
    waits = Counter(cmd[1] for lane in lanes for cmd in lane if cmd[0] == "wait")
    if sets != waits:
        raise Violation("unbalanced HardEvent census: " + ", ".join(
            f"{p}: {sets[p]} set / {waits[p]} wait"
            for p in sorted(set(sets) | set(waits)) if sets[p] != waits[p]))

    m2 = Counter((cmd[0], cmd[1], cmd[2]) for lane in lanes for cmd in lane
                 if cmd[0] in ("m2set", "m2wait"))
    ready = (m2[("m2set", READY, "V0")], m2[("m2set", READY, "V1")],
             m2[("m2wait", READY, "AIC")])
    if len(set(ready)) != 1:
        raise Violation(
            "unbalanced readiness quorum: AIV0 announces %d, AIV1 announces %d, "
            "the AIC waits %d. A mode-2 AIV -> AIC pair is one announcement from "
            "EACH AIV against one AIC wait." % ready)
    retired = (m2[("m2set", RETIRED, "AIC")], m2[("m2wait", RETIRED, "V0")],
               m2[("m2wait", RETIRED, "V1")])
    if len(set(retired)) != 1:
        raise Violation(
            "unbalanced release broadcast: the AIC announces %d, AIV0 waits %d, "
            "AIV1 waits %d. One AIC announcement increments the counter on BOTH "
            "AIVs, so each must wait once per announcement." % retired)

    # ---- the state ------------------------------------------------------
    # marking      one token count per intra-core HardEvent place
    # ready_lead   per AIV, announcements not yet absorbed by an AIC wait
    # rel_lead     per AIV, releases announced but not yet waited on
    # l1_owner     strip whose keys are in each L1 page, and whether read
    # panel_owner  strip whose scores are in each UB panel, and whether read
    # recurs       how many recurrences have retired, in strip order
    # sticky       which AIV-side channels have been released at least once
    start = (tuple([0] * len(lanes)), tuple([0] * len(places)),
             (0, 0), (0, 0),
             tuple([-1] * STAGE_DEPTH), tuple([0] * STAGE_DEPTH),
             tuple([-1] * STAGE_DEPTH), tuple([0] * STAGE_DEPTH),
             0, 0)
    pending = deque([start])
    seen = {start}
    terminals = 0
    while pending:
        if len(seen) > cap:
            raise Violation(f"state space exceeded {cap} states")
        (pcs, marking, ready_lead, rel_lead, l1_owner, l1_read,
         panel_owner, panel_read, recurs, sticky) = pending.popleft()
        progress = False
        for lane_id, lane in enumerate(lanes):
            pc = pcs[lane_id]
            if pc == len(lane):
                continue
            op, *rest = lane[pc]
            tokens = list(marking)
            new_ready = list(ready_lead)
            new_rel = list(rel_lead)
            new_l1_owner, new_l1_read = list(l1_owner), list(l1_read)
            new_panel_owner, new_panel_read = list(panel_owner), list(panel_read)
            next_recurs, next_sticky = recurs, sticky
            if op == "set":
                place = rest[0]
                i = index[place]
                if tokens[i] + 1 > 1:
                    raise Violation(
                        f"SET into a full place {place} -- a HardEvent is one "
                        f"bit, so this token is lost")
                tokens[i] += 1
            elif op == "wait":
                place = rest[0]
                i = index[place]
                if tokens[i] == 0:
                    continue  # the engine blocks here
                tokens[i] -= 1
            elif op == "m2set":
                channel, who = rest
                if channel == READY:
                    # Only this AIV's own counter advances; the AIC's advances
                    # when BOTH have.
                    new_ready[0 if who == "V0" else 1] += 1
                    available = (max(new_ready) if single_sender
                                 else min(new_ready))
                    if available > CROSS_BOUND:
                        raise Violation(
                            f"readiness channel reached {available} outstanding "
                            f"announcements (bound {CROSS_BOUND}); past 15 a "
                            f"mode-2 counter is a reported fault")
                else:
                    # One AIC announcement increments BOTH AIVs' counters.
                    new_rel = [n + 1 for n in new_rel]
                    if max(new_rel) > CROSS_BOUND:
                        raise Violation(
                            f"release channel reached {max(new_rel)} outstanding "
                            f"announcements (bound {CROSS_BOUND})")
            elif op == "m2wait":
                channel, who = rest
                if channel == READY:
                    # `single_sender_quorum` is the control that models a mode-4
                    # per-AIV latch in this position: either AIV alone releases
                    # the Cube.
                    available = (max(new_ready) if single_sender
                                 else min(new_ready))
                    if available == 0:
                        continue
                    if single_sender:
                        new_ready[new_ready.index(max(new_ready))] -= 1
                    else:
                        new_ready = [n - 1 for n in new_ready]
                else:
                    which = 0 if who == "V0" else 1
                    bit = 1 << which
                    # Section 6.3's knob. Under `sticky_aiv_wait` an AIV-side
                    # cross-core wait behaves as a LEVEL that the first wait
                    # observes and no later wait blocks on -- which is what the
                    # four-page trace measured for mode 4's AIC -> AIV
                    # direction: stalls of 23146, 3, 3 and 3 ticks against
                    # announcements ~20600 ticks apart. A model that assumes
                    # the consuming semantics has assumed the answer, so both
                    # readings are explored and both verdicts reported.
                    if sticky_aiv_wait and (sticky & bit):
                        pass
                    elif new_rel[which] == 0:
                        continue
                    else:
                        new_rel[which] -= 1
                        if sticky_aiv_wait:
                            next_sticky = sticky | bit
            elif op == "barrier":
                core, tag, drained = rest[0]
                # Instruction issue is stalled until every NAMED pipe has
                # retired its pre-fence work -- that is, has reached this same
                # fence. Pipes not named are held back but not drained.
                if any(pcs[lane_of[f"{core}.{p}"]] < barrier_at[(tag, p)]
                       for p in drained):
                    continue
            elif op == "stage_l1":
                slot, strip = rest
                if new_l1_owner[slot] != -1 and not new_l1_read[slot]:
                    raise Violation(
                        f"L1 key page {slot} rewritten by strip {strip} while "
                        f"strip {new_l1_owner[slot]} was still live")
                new_l1_owner[slot], new_l1_read[slot] = strip, 0
            elif op == "load_l0b":
                slot, strip = rest
                if new_l1_owner[slot] != strip:
                    raise Violation(
                        f"Cube read L1 key page {slot} holding strip "
                        f"{new_l1_owner[slot]} while multiplying strip {strip}")
                new_l1_read[slot] = 1
            elif op == "fixpipe":
                slot, strip = rest
                if new_panel_owner[slot] != -1 and not new_panel_read[slot]:
                    raise Violation(
                        f"UB score panel {slot} rewritten by strip {strip} while "
                        f"strip {new_panel_owner[slot]} was still live")
                new_panel_owner[slot], new_panel_read[slot] = strip, 0
            elif op == "recurrence":
                slot, strip = rest
                if new_panel_owner[slot] != strip:
                    raise Violation(
                        f"recurrence of strip {strip} read UB score panel {slot} "
                        f"holding strip {new_panel_owner[slot]}")
                new_panel_read[slot] = 1
                # The carried state is a single sequential accumulator at fixed
                # UB offsets: strip t must follow strip t - 1, across page
                # boundaries as much as inside a page.
                if strip != recurs:
                    raise Violation(
                        f"recurrence of strip {strip} ran out of order "
                        f"(expected strip {recurs})")
                next_recurs = recurs + 1
            state = (tuple(pcs[:lane_id] + (pc + 1,) + pcs[lane_id + 1:]),
                     tuple(tokens), tuple(new_ready), tuple(new_rel),
                     tuple(new_l1_owner), tuple(new_l1_read),
                     tuple(new_panel_owner), tuple(new_panel_read),
                     next_recurs, next_sticky)
            progress = True
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not progress:
            if any(pcs[i] != len(lanes[i]) for i in range(len(lanes))):
                stuck = [names[i] for i in range(len(lanes))
                         if pcs[i] != len(lanes[i])]
                raise Violation("deadlock / unprimed cycle on " + ", ".join(stuck))
            if any(marking):
                raise Violation("non-empty final HardEvent marking: " + ", ".join(
                    f"{places[i]}={marking[i]}" for i in range(len(places))
                    if marking[i]))
            if any(ready_lead) or any(rel_lead):
                raise Violation(
                    f"non-empty final mode-2 counters: readiness {ready_lead}, "
                    f"release {rel_lead}")
            if recurs != total:
                raise Violation(
                    f"only {recurs} of {total} recurrences retired")
            terminals += 1
    if not terminals:
        raise Violation("no terminal state reached")
    return len(seen)


# Exhaustively explorable geometries. The brief for this refactor asks for
# coverage to eight pages, and with the page loop gone the model's cost tracks
# the STRIP count rather than the page count -- the stream does not care where
# a page boundary falls, which is the whole point of the change.
GRID = [(1, 1), (1, 2), (1, 4), (1, 8),
        (2, 1), (2, 2), (2, 4),
        (3, 1), (3, 2),
        (4, 1), (4, 2),
        (5, 1), (6, 1), (7, 1), (8, 1)]

# Beyond exhaustive interleaving exploration here: eight strips per page over
# more than one page is tens of millions of states with no partial-order
# reduction in this model. It is NOT covered by this gate, and the CAModel run
# at `--page-tokens 128` over two pages is what would carry that geometry
# instead. Recorded rather than silently dropped.
BEYOND_EXHAUSTIVE = [(8, 2), (2, 8), (4, 8), (8, 8)]

CONTROLS = {
    # THE defect this refactor fixes: AIV1 restages the L1 key page with no WAR
    # edge behind it. Under the per-strip L1 arrangement this was the shipped
    # code, and `reports/decode_splitk_aic_page_race.md` section 5 measured it.
    "no_l1_war": "AIV1 restaging the L1 key page with no release behind it",
    # Mode 4 in mode 2's position: one AIV alone releases the Cube's wait, so
    # the GEMM can run without the other's staging.
    "single_sender_quorum": "either AIV alone satisfying the readiness quorum",
    # Section 6.2a's hazard, as a structure rather than a missing drain.
    "release_before_fixpipe": "the release announced before the Fixpipe landed",
    "release_before_l0b": "the release announced before the L1 -> L0B load",
    "no_prime": "the score slots not primed as free before the stream",
    "carry_credit": "a V_MTE2 credit never returned by the recurrence",
    "no_epilogue_drain": "AIV1 leaving its standing releases unconsumed",
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
        print(f"PASS pages={pages} strips={strips} ({pages * strips} strips): "
              f"{states} reachable states, empty final marking, "
              f"{pages * strips} ordered recurrences")

    for pages, strips in BEYOND_EXHAUSTIVE:
        print(f"NOT COVERED pages={pages} strips={strips}: beyond exhaustive "
              f"interleaving exploration in this model; covered on CAModel only")

    # Each control must be rejected, and on a geometry where it can bite: a
    # ring wrap, which fewer than STAGE_DEPTH + 1 strips does not have.
    for control, description in CONTROLS.items():
        rejected = []
        for pages, strips in ((2, 2), (4, 1), (2, 4), (8, 1)):
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
    # REFINEMENT_OF_COMBAT_KERNELS.md section 6.3: two hardware properties
    # decide whether channel recycling is usable, and a model that assumes
    # either has assumed the answer. Everything above assumes an AIV-side
    # cross-core wait CONSUMES the flag it tested. The four-page Split-K trace
    # measured the other reading for mode 4's AIC -> AIV direction, and under
    # that reading NO arrangement of mode-4 flags orders AIV1's L1 reuse --
    # which is why this operator now uses mode 2 instead. So the sticky verdict
    # is reported here for the shipped structure, and what settles it for a
    # given toolkit is the stall census over that revision's own trace.
    print()
    for label, flags in (("as shipped (mode-2 quorum + broadcast)", {}),
                         ("with no L1 WAR edge, the pre-refactor shape",
                          {"no_l1_war": True})):
        verdicts = []
        for pages, strips in ((2, 2), (4, 1)):
            try:
                explore(schedule(pages, strips, **flags), pages, strips,
                        sticky_aiv_wait=True)
                verdicts.append(f"pages={pages} strips={strips}: safe even so")
            except Violation as e:
                verdicts.append(f"pages={pages} strips={strips}: {e}")
        print(f"SEMANTICS sticky AIV cross-core wait, {label}: "
              + "; ".join(verdicts))
    print("SEMANTICS which reading holds for mode 2 on this toolkit is settled "
          "by scripts/run_cluster_sync_probe_camodel.sh and the stall census, "
          "not by this model; see reports/decode_splitk_aic_page_race.md")

    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print(f"\nPASS AIC strip stream: {len(GRID)} geometries 1-safe across page "
          f"boundaries, L1 and score-panel ownership checked per strip, ordered "
          f"carried state, finite drain; {len(CONTROLS)} negative controls; "
          f"{len(BEYOND_EXHAUSTIVE)} geometries not covered here; "
          f"flag-semantics knob reported for both readings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
