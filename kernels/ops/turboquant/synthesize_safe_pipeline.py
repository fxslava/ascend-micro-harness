#!/usr/bin/env python3
"""Synthesize and exhaustively check a small DaVinci pipeline (Python >= 3.10).

No solver or third-party packages are needed. Run with --help for outputs.
This is a protocol synthesizer, NOT a parser or proof of the reference C++.
The executable semantics deliberately implement the requested *consuming*,
one-bit flags. A hardware wait that does not clear its flag violates this model.
See safe_pipeline.md for the abstraction/refinement and optimality boundaries.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
import time


RESERVED = frozenset(range(11, 15)) | frozenset(range(27, 31))
LEGAL_LOCAL = tuple(range(11)) + (15,)


class VerificationError(Exception):
    def __init__(self, message, trace=()):
        super().__init__(message)
        self.trace = list(trace)


@dataclass(frozen=True)
class Transition:
    name: str
    pre: int
    post: int
    require: int = 0
    reason: str = "missing ownership/data"


class Net:
    """Elementary 1-safe net encoded by bitsets, with diagnostic assertion arcs.

    pre = enabling/consuming arcs; post = producing arcs. require = assertions,
    NOT enabling arcs: a premature memory access or set must fail verification,
    not disappear from the graph. Likewise an occupied post place is an error,
    not an implicit capacity stall. Only explicit waits/queue availability block.
    Each 1-bit hardware latch has an implicit complementary EMPTY place.
    """

    def __init__(self):
        self.places = {}
        self.transitions = []
        self.initial = 0
        self.final = None
        self.register_mask = 0
        self.queue_mask = 0
        self.groups = []
        self.ends = []

    def bits(self, names):
        value = 0
        for name in names:
            if name not in self.places:
                self.places[name] = 1 << len(self.places)
            value |= self.places[name]
        return value

    def seed(self, *names):
        bits = self.bits(names)
        if self.initial & bits:
            raise VerificationError("prologue SetFlag overwrite / aliased credit")
        self.initial |= bits

    def add(self, name, pre=(), post=(), require=(), reason="missing ownership/data"):
        self.transitions.append(Transition(name, self.bits(pre), self.bits(post),
                                           self.bits(require), reason))

    def register(self, name):
        self.register_mask |= self.bits([name])
        return name

    def queue(self, name):
        idle = name + ".IDLE"
        self.seed(idle)
        self.queue_mask |= self.bits([idle])
        return idle

    def states(self, base, states):
        names = [base + "." + x for x in states]
        self.groups.append(self.bits(names))
        self.seed(names[0])


class Stream:
    def __init__(self, net, name, cyclic):
        self.net, self.name, self.cyclic = net, name, cyclic
        self.index = 0
        self.start = name + ".pc0"
        self.pc = self.start
        net.seed(self.pc)

    def step(self, name, pre=(), post=(), require=(), reason="missing ownership/data"):
        self.index += 1
        nxt = f"{self.name}.pc{self.index}"
        self.net.add(f"{self.name}: {name}", (self.pc, *pre), (nxt, *post),
                     require, reason)
        self.pc = nxt

    def end(self):
        if self.cyclic:
            # Identify the final control place with pc0: no epoch/global barrier.
            old, new = self.net.bits([self.pc]), self.net.bits([self.start])
            t = self.net.transitions[-1]
            self.net.transitions[-1] = Transition(t.name, t.pre,
                                                  (t.post & ~old) | new,
                                                  t.require, t.reason)
        else:
            self.net.ends.append(self.pc)


@dataclass(frozen=True)
class Policy:
    # Logical channels: READY[0], READY[1], FREE[0], FREE[1].
    colors: tuple[int, ...] = (0, 1, 2, 3)
    free_position: str = "late"
    ready_position: str = "late"
    fix_drain: bool = True
    vector_drain: bool = True
    free_wait: bool = True
    sticky_wait: bool = False
    pack_wait: bool = True

    def flag(self, owner, slot, ready):
        return owner * 16 + LEGAL_LOCAL[self.colors[slot + (0 if ready else 2)]]


def build(policy=Policy(), chunks=8, cyclic=False):
    """Build independent MTE2, AIC scalar, AIV scalar, and DMA engine streams.

    The cyclic net uses four chunk positions, with no iteration bound. Resource
    ownership is tied to positions; slot FREE->WRITE->VALID->READ->FREE is the
    generation invariant. Independent ingress is a conservative abstraction of
    scalar-issued MTE2. Cube is one ordered L1/MTE1/M macro-operation; it retains
    L1 through compute (conservative) and L0C through FIX_S (one accumulator).
    """
    if cyclic:
        chunks = 4
    if chunks < 1:
        raise ValueError("chunks must be positive")
    n = Net()
    # All 32 cross registers exist, even if unused. HardEvents are route/core
    # scoped, just as hardware: numeric ID 0 on FIX_S is distinct from V_S/0.
    for fid in range(32):
        n.register(f"CrossFlags[{fid}]")

    def hard(core, route, event):
        if event not in range(6):
            raise VerificationError(f"AKA2003: reserved HardEvent {event}")
        return n.register(f"{core}.HardEvents[{event}].{route}")

    def flag(owner, slot, ready):
        fid = policy.flag(owner, slot, ready)
        if fid in RESERVED or not 0 <= fid < 32:
            raise VerificationError(f"reserved mode-4 channel {fid}")
        return f"CrossFlags[{fid}]"

    def tag(owner, slot, ready):
        return f"credit.{owner}.{slot}.{'READY' if ready else 'FREE'}"

    def xset(stream, owner, slot, ready, require=()):
        stream.step(f"SetFlag {'READY' if ready else 'FREE_ACK'}[{slot}]",
                    post=(flag(owner, slot, ready), tag(owner, slot, ready)),
                    require=require, reason="notification before queue-to-scalar drain")

    def xwait(stream, owner, slot, ready):
        f, g = flag(owner, slot, ready), tag(owner, slot, ready)
        # Provenance assertion catches consuming another channel's aliased bit.
        # The ghost is removed only AFTER the assertion succeeds.
        stream.step(f"WaitFlag {'READY' if ready else 'FREE_ACK'}[{slot}]",
                    pre=(f,), post=(f,) if policy.sticky_wait else (),
                    require=(g,), reason="stale / wrong-channel flag consumed")
        t = n.transitions[-1]
        # require bits are not part of pre, so absent provenance cannot block.
        n.transitions[-1] = Transition(t.name, t.pre, t.post, t.require, t.reason)
        # Drop the ghost on firing; represented separately from enabling arcs.
        ghost_clears[len(n.transitions) - 1] = n.bits([g])

    ghost_clears = {}
    mte2_idle = n.queue("AIC.PIPE_MTE2")
    cube_idle = n.queue("AIC.PIPE_M")
    fix_idle = n.queue("AIC.PIPE_FIX")
    for slot in range(4):
        n.states(f"L1[{slot}]", ("FREE", "WRITING", "VALID", "READING"))
        n.seed(hard("AIC", "MTE1_MTE2", slot))
    for owner in range(2):
        for slot in range(2):
            n.states(f"AIV{owner}.UB_Land[{slot}]",
                     ("FREE", "WRITING", "VALID", "READING"))
            n.states(f"AIV{owner}.UB_Pack[{slot}]",
                     ("FREE", "WRITING", "VALID", "READING"))
            n.seed(flag(owner, slot, False), tag(owner, slot, False))
            n.seed(hard(f"AIV{owner}", "MTE3_V", slot))

    # Column dispatcher: StagePage(0), then StagePage(c+1), Gemm(c).
    # No StagePage(c+2) can issue until Gemm(c)'s scalar notification retires.
    # These are program-order permits, not additional hardware flags.
    for c in range(min(2, chunks)):
        n.seed(f"column.ingress_permit.{c}")
    ingress = Stream(n, "MTE2", cyclic)
    for c in range(chunks):
        slot = c % 4
        base = f"L1[{slot}]"
        work, done = f"MTE2.work.{c}", f"MTE2.done.{c}"
        ingress.step(f"Wait MTE1_MTE2 c{c}",
                     pre=(hard("AIC", "MTE1_MTE2", slot), f"column.ingress_permit.{c}"))
        ingress.step(f"DMA dispatch c{c}", pre=(mte2_idle,), post=(work,),
                     require=(base + ".FREE",), reason="L1 WAR")
        # Memory-state transformations are assertion consumes, not wait guards.
        ghost_clears[len(n.transitions) - 1] = n.bits([base + ".FREE"])
        t = n.transitions[-1]
        n.transitions[-1] = Transition(t.name, t.pre, t.post | n.bits([base + ".WRITING"]),
                                       t.require, t.reason)
        n.add(f"MTE2 DMA complete c{c}", (work, base + ".WRITING"),
              (done, base + ".VALID"))
        n.add(f"MTE2 Set MTE2_MTE1 c{c}", (done,),
              (mte2_idle, hard("AIC", "MTE2_MTE1", slot)))
    # Last transition added was an engine transition, so end() cannot assume
    # it is the stream's last. Append an explicit control-only boundary first.
    ingress.step("iteration boundary")
    ingress.end()

    aic = Stream(n, "AIC.S", cyclic)
    for c in range(chunks):
        owner, slot, l1 = c % 2, (c // 2) % 2, c % 4
        land, page = f"AIV{owner}.UB_Land[{slot}]", f"L1[{l1}]"
        work, done = f"M.work.{c}", f"M.done.{c}"
        drained = f"FIX_S.drained.{c}"
        aic.step(f"Wait MTE2_MTE1 c{c}", pre=(hard("AIC", "MTE2_MTE1", l1),))
        if policy.free_position == "early" and policy.free_wait:
            xwait(aic, owner, slot, False)
        aic.step(f"Cube dispatch c{c}", pre=(cube_idle,), post=(work, page + ".READING"),
                 require=(page + ".VALID",), reason="L1 RAW")
        ghost_clears[len(n.transitions) - 1] = n.bits([page + ".VALID"])
        n.add(f"Cube complete c{c}", (work, page + ".READING"), (done, page + ".FREE"))
        n.add(f"Cube Set M_FIX / L1 FREE c{c}", (done,),
              (cube_idle, hard("AIC", "M_FIX", 0), hard("AIC", "MTE1_MTE2", l1)))
        aic.step(f"Wait M_FIX c{c}", pre=(hard("AIC", "M_FIX", 0),))
        if policy.free_position == "late" and policy.free_wait:
            xwait(aic, owner, slot, False)
        aic.step(f"Fixpipe dispatch c{c}", pre=(fix_idle,),
                 post=(f"FIX.work.{c}", land + ".WRITING"),
                 require=(land + ".FREE",), reason="landing WAR / missing FREE_ACK")
        ghost_clears[len(n.transitions) - 1] = n.bits([land + ".FREE"])
        n.add(f"Fixpipe complete c{c}", (f"FIX.work.{c}", land + ".WRITING"),
              (f"FIX.done.{c}", land + ".VALID"))
        n.add(f"PIPE_FIX Set FIX_S c{c}", (f"FIX.done.{c}",),
              (fix_idle, hard("AIC", "FIX_S", 0)))
        if policy.fix_drain:
            aic.step(f"Wait FIX_S c{c}", pre=(hard("AIC", "FIX_S", 0),), post=(drained,))
        xset(aic, owner, slot, True, (drained,))
        ghost_clears[len(n.transitions) - 1] = n.bits([drained])
        if cyclic or c + 2 < chunks:
            following = (c + 2) % chunks if cyclic else c + 2
            t = n.transitions[-1]
            n.transitions[-1] = Transition(t.name, t.pre,
                t.post | n.bits([f"column.ingress_permit.{following}"]), t.require, t.reason)
    aic.step("iteration boundary")
    aic.end()

    for owner in range(2):
        core = f"AIV{owner}"
        v_idle, dma_idle = n.queue(core + ".PIPE_V"), n.queue(core + ".PIPE_MTE3")
        aiv = Stream(n, core + ".S", cyclic)
        store = Stream(n, core + ".MTE3", cyclic)
        for c in range(owner, chunks, 2):
            slot = (c // 2) % 2
            land, pack = f"{core}.UB_Land[{slot}]", f"{core}.UB_Pack[{slot}]"
            drained = f"V_S.drained.{c}"
            if policy.ready_position == "early":
                xwait(aiv, owner, slot, True)
            aiv.step(f"prepare addresses c{c}")
            if policy.ready_position == "late":
                xwait(aiv, owner, slot, True)
            aiv.step(f"Vector landing read dispatch c{c}", pre=(v_idle,),
                     post=(f"V.read.{c}", land + ".READING"),
                     require=(land + ".VALID",), reason="landing RAW / stale generation")
            ghost_clears[len(n.transitions) - 1] = n.bits([land + ".VALID"])
            n.add(f"Vector landing read complete c{c}", (f"V.read.{c}", land + ".READING"),
                  (f"V.read_done.{c}", land + ".FREE"))
            n.add(f"PIPE_V Set V_S c{c}", (f"V.read_done.{c}",),
                  (v_idle, hard(core, "V_S", 0)))
            if policy.vector_drain:
                aiv.step(f"Wait V_S c{c}", pre=(hard(core, "V_S", 0),), post=(drained,))
            xset(aiv, owner, slot, False, (drained,))
            ghost_clears[len(n.transitions) - 1] = n.bits([drained])
            if policy.pack_wait:
                aiv.step(f"Wait MTE3_V c{c}", pre=(hard(core, "MTE3_V", slot),))
            aiv.step(f"quantise / pack dispatch c{c}", pre=(v_idle,),
                     post=(f"V.pack.{c}", pack + ".WRITING"),
                     require=(pack + ".FREE",), reason="packed-output WAR")
            ghost_clears[len(n.transitions) - 1] = n.bits([pack + ".FREE"])
            n.add(f"quantise / pack complete c{c}", (f"V.pack.{c}", pack + ".WRITING"),
                  (f"V.pack_done.{c}", pack + ".VALID"))
            n.add(f"PIPE_V Set V_MTE3 / V_S c{c}", (f"V.pack_done.{c}",),
                  (v_idle, hard(core, "V_MTE3", slot), hard(core, "V_S", 0)))
            # Shared quantiser scratch is reused by the next chunk's Cast.
            aiv.step(f"Wait V_S scratch reusable c{c}", pre=(hard(core, "V_S", 0),))
            store.step(f"Wait V_MTE3 c{c}", pre=(hard(core, "V_MTE3", slot),))
            store.step(f"DMA store dispatch c{c}", pre=(dma_idle,),
                       post=(f"MTE3.work.{c}", pack + ".READING"),
                       require=(pack + ".VALID",), reason="packed-output RAW")
            ghost_clears[len(n.transitions) - 1] = n.bits([pack + ".VALID"])
            n.add(f"DMA store complete c{c}", (f"MTE3.work.{c}", pack + ".READING"),
                  (f"MTE3.done.{c}", pack + ".FREE"))
            n.add(f"PIPE_MTE3 Set MTE3_V c{c}", (f"MTE3.done.{c}",),
                  (dma_idle, hard(core, "MTE3_V", slot)))
        aiv.step("iteration boundary")
        aiv.end()
        store.step("iteration boundary")
        store.end()

    if not cyclic:
        # The epilogue is a sequence of waits. Model them atomically only after
        # every stream has issued its last operation; no work can depend on them.
        credits = [hard("AIC", "MTE1_MTE2", s) for s in range(4)]
        for o in range(2):
            for s in range(2):
                credits += [hard(f"AIV{o}", "MTE3_V", s), flag(o, s, False), tag(o, s, False)]
        n.add("EPILOGUE drain FREE_ACK and MTE3_V (host-visible completion)",
              (*n.ends, *credits), ("TERMINATED",))
        free = [name for name in n.places if name.endswith(".IDLE") or
                (name.endswith(".FREE") and not name.startswith("credit."))]
        n.final = n.bits(["TERMINATED", *free])
    return n, ghost_clears


def explore(net, clears, *, cyclic=False, max_states=64_000_000, progress=True):
    """Full BFS; no partial-order reduction, random sampling or depth bound.

    Save every predecessor edge in compact linked arrays. Reverse reachability
    proves AG EF terminal (finite) or AG EF initial (cyclic). Every finite
    transition fires at most once, so the finite graph has no livelock. For the
    cyclic net this is nonblocking/recurrent liveness, not unfair-scheduler
    starvation freedom. Hardware progress assumes weakly fair engine service.
    """
    started = time.monotonic()
    states, seen = [net.initial], {net.initial: 0}
    parents, via = array("i", [-1]), array("i", [-1])
    rev_head, rev_src, rev_next = array("i", [-1]), array("i"), array("i")
    fired = set()
    terminal = []
    cursor, last = 0, started
    def bits(value):
        while value:
            low = value & -value
            yield low
            value ^= low

    # Index by a necessary input place. This changes only lookup cost: every
    # enabled transition is still enumerated at every marking, with no POR.
    frequencies = Counter(b for t in net.transitions for b in bits(t.pre))
    buckets = {}
    dispatch_mask = 0
    always = []
    for i, t in enumerate(net.transitions):
        clear = clears.get(i, 0)
        changed = t.pre | t.post | clear
        groups = tuple(g for g in net.groups if changed & g)
        item = (t.pre, t.post, t.require, clear, i, groups)
        if not t.pre:
            always.append(item)
        else:
            pivot = min(bits(t.pre), key=frequencies.__getitem__)
            buckets.setdefault(pivot, []).append(item)
            dispatch_mask |= pivot

    def candidates(marking):
        yield from always
        for pivot in bits(marking & dispatch_mask):
            yield from buckets[pivot]

    if any((net.initial & g).bit_count() != 1 for g in net.groups):
        raise VerificationError("initial buffer conservation failed")

    def trace(index, tail=None):
        names = [] if tail is None else [tail]
        while parents[index] != -1:
            names.append(net.transitions[via[index]].name)
            index = parents[index]
        return list(reversed(names))

    while cursor < len(states):
        marking = states[cursor]
        successors = 0
        for pre, post, require, clear, tid, groups in candidates(marking):
            if marking & pre != pre:
                continue
            if marking & require != require:
                t = net.transitions[tid]
                raise VerificationError(t.reason + " at " + t.name, trace(cursor, t.name))
            remaining = marking & ~(pre | clear)
            if remaining & post:
                t = net.transitions[tid]
                occupied = [p for p, bit in net.places.items() if remaining & post & bit]
                raise VerificationError("1-safety overwrite: " + ", ".join(occupied)
                                        + " at " + t.name, trace(cursor, t.name))
            nxt = remaining | post
            # Unchanged groups retain their invariant by induction; check all
            # affected groups on EVERY successor, not just first discovery.
            if groups and any((nxt & g).bit_count() != 1 for g in groups):
                raise VerificationError("buffer state conservation failed",
                                        trace(cursor, net.transitions[tid].name))
            successors += 1
            fired.add(tid)
            dest = seen.get(nxt)
            if dest is None:
                if len(states) >= max_states:
                    raise VerificationError(f"INCONCLUSIVE: state limit {max_states}; no proof emitted")
                dest = len(states)
                seen[nxt] = dest
                states.append(nxt)
                parents.append(cursor)
                via.append(tid)
                rev_head.append(-1)
            rev_src.append(cursor)
            rev_next.append(rev_head[dest])
            rev_head[dest] = len(rev_src) - 1
        if not successors:
            if cyclic or marking != net.final:
                raise VerificationError("deadlock / dirty termination", trace(cursor))
            terminal.append(cursor)
        cursor += 1
        now = time.monotonic()
        if progress and now - last > 15:
            print(f"    BFS expanded {cursor:,}; discovered {len(states):,}; edges {len(rev_src):,}", flush=True)
            last = now
    targets = [0] if cyclic else terminal
    reached = bytearray(len(states))
    todo = deque(targets)
    for v in targets:
        reached[v] = 1
    while todo:
        edge = rev_head[todo.popleft()]
        while edge != -1:
            src = rev_src[edge]
            if not reached[src]:
                reached[src] = 1
                todo.append(src)
            edge = rev_next[edge]
    if not all(reached):
        bad = reached.index(0)
        raise VerificationError("reachable state cannot drain / return to initial marking", trace(bad))
    if len(fired) != len(net.transitions):
        missing = [t.name for i, t in enumerate(net.transitions) if i not in fired]
        raise VerificationError("dead transitions: " + ", ".join(missing))
    return {"states": len(states), "edges": len(rev_src), "transitions": len(fired),
            "places": len(net.places), "terminal_states": len(terminal),
            "safe_reachable_percent": 100, "overwrites": 0, "races": 0,
            "deadlocks": 0, "all_states_coreachable": True,
            "seconds": round(time.monotonic() - started, 3)}


def partitions(size):
    """All set partitions as restricted-growth strings, modulo ID renaming."""
    if size == 0:
        yield ()
        return
    def extend(prefix):
        if len(prefix) == size:
            yield prefix
            return
        for value in range(max(prefix) + 2):
            yield from extend((*prefix, value))
    yield from extend((0,))


def synthesize(max_states):
    rejected = []
    # Lexicographic objective: minimum channels, then latest legal waits.
    # All aliases with fewer colors are actually explored (or fail prologue).
    for colors in sorted(partitions(4), key=lambda c: (max(c), c)):
        policy = Policy(colors=colors)
        print(f"  candidate {colors}: {max(colors) + 1} IDs/sub-block", flush=True)
        try:
            net, clears = build(policy, cyclic=True)
            result = explore(net, clears, cyclic=True, max_states=max_states)
        except VerificationError as error:
            if str(error).startswith("INCONCLUSIVE"):
                raise
            rejected.append({"colors": colors, "reason": str(error), "trace": error.trace})
            print(f"    REJECT: {error}", flush=True)
            continue
        print(f"    ACCEPT: {result['states']:,} states, {result['edges']:,} edges", flush=True)
        return policy, result, rejected
    raise VerificationError("no legal assignment in the protocol grammar")


def synthesize_placements(policy):
    """Enumerate wait cuts in the dispatch dependency order.

    A wait at cut i is before operation i. Only cuts preceding the protected
    access are legal. Independent operations moved ahead of a wait cannot add a
    buffer dependency; latest cuts dominate earlier cuts in this slack metric.
    BFS verifies the selected dominating schedule, rather than repeating an
    expensive proof for dominated schedules. There is no clock-based objective.
    """
    choices = []
    for free_cut in range(3):  # before Cube, before FIX, after FIX
        for ready_cut in range(3):  # before prepare, before read, after read
            legal = free_cut <= 1 and ready_cut <= 1
            choices.append({"free_cut": free_cut, "ready_cut": ready_cut,
                            "legal": legal, "slack_score": free_cut + ready_cut,
                            "reason": "precedes both protected accesses" if legal
                                      else "wait would follow its protected memory access"})
    best = max((c for c in choices if c["legal"]), key=lambda c: c["slack_score"])
    return replace(policy, free_position=("early", "late")[best["free_cut"]],
                   ready_position=("early", "late")[best["ready_cut"]]), choices


def negative_tests(max_states):
    mutations = {
        "missing_FIX_S": Policy(fix_drain=False),
        "missing_V_S": Policy(vector_drain=False),
        "missing_FREE_ACK_wait": Policy(free_wait=False),
        "nonconsuming_sticky_wait": Policy(sticky_wait=True),
        "missing_MTE3_V_wait": Policy(pack_wait=False),
        "aliased_READY_slots": Policy(colors=(0, 0, 1, 2)),
    }
    results = {}
    for name, policy in mutations.items():
        try:
            net, clears = build(policy, cyclic=True)
            explore(net, clears, cyclic=True, max_states=max_states, progress=False)
        except VerificationError as error:
            if str(error).startswith("INCONCLUSIVE"):
                raise
            results[name] = {"reason": str(error), "trace": error.trace}
            print(f"  mutation {name}: CAUGHT ({len(error.trace)} steps): {error}", flush=True)
        else:
            raise VerificationError("checker failed to reject mutation: " + name)
    return results


def header(policy):
    ids = [LEGAL_LOCAL[i] for i in policy.colors]
    return """// Generated by synthesize_safe_pipeline.py; C++17, standalone declarations.
// Proof assumes consuming one-bit mode-4 flags and ordered local HardEvents.
// This header is an integration contract; it does not patch the reference kernel.
#pragma once
#include <cstdint>
namespace turboquant_safe_pipeline {
constexpr std::uint16_t kReady[2] = {READY0, READY1};
constexpr std::uint16_t kFree[2] = {FREE0, FREE1};
constexpr std::uint16_t AicId(std::uint32_t owner, std::uint16_t local) {
    return static_cast<std::uint16_t>(owner * 16u + local);
}
constexpr std::uint16_t AivLocalId(std::uint16_t canonical) { return canonical % 16u; }
constexpr std::uint32_t LandSlot(std::uint32_t chunk) { return (chunk / 2u) % 2u; }
// Canonical AIC-addressed IDs. Convert to AIV-local IDs at the AIV endpoint.
template <std::uint32_t C> constexpr std::uint16_t kFlagProductReady =
    AicId(C % 2u, kReady[LandSlot(C)]);
template <std::uint32_t C> constexpr std::uint16_t kFlagPanelFreeWait =
    AicId(C % 2u, kFree[LandSlot(C)]);
template <std::uint32_t C> constexpr std::uint16_t kFlagPanelFreeSet =
    AivLocalId(kFlagPanelFreeWait<C>);
template <std::uint32_t C> constexpr std::uint16_t kFlagProductReadyWait =
    AivLocalId(kFlagProductReady<C>);
template <std::uint32_t Owner, std::uint32_t Slot>
constexpr std::uint16_t kFlagPanelFreePrime = kFree[Slot]; // AIV-local
struct ModuloFlags {
    static constexpr std::uint16_t ProductReady(std::uint32_t c) {
        return AicId(c % 2u, kReady[LandSlot(c)]);
    }
    static constexpr std::uint16_t PanelFreeWait(std::uint32_t c) {
        return AicId(c % 2u, kFree[LandSlot(c)]);
    }
    static constexpr std::uint16_t PanelFreeSet(std::uint32_t c) {
        return AivLocalId(PanelFreeWait(c));
    }
};
constexpr std::int32_t kEvtFixScalar = 0; // AIC FIX_S route
constexpr std::int32_t kEvtVectorScalar = 0; // each AIV V_S route
constexpr std::int32_t kEvtL1[4] = {0, 1, 2, 3}; // MTE1_MTE2 / MTE2_MTE1
constexpr std::int32_t kEvtPack[2] = {0, 1}; // each AIV MTE3_V / V_MTE3
constexpr bool LegalCross(std::uint16_t id) {
    return id < 32u && !(id >= 11u && id <= 14u) && !(id >= 27u && id <= 30u);
}
static_assert(LegalCross(kFlagProductReady<7>) && LegalCross(kFlagPanelFreeWait<7>));
static_assert(kFlagPanelFreeSet<3> == AivLocalId(kFlagPanelFreeWait<7>));
static_assert(kFlagProductReady<3> == kFlagProductReady<7>);
static_assert(kEvtL1[3] < 6 && kEvtPack[1] < 6);

// INSERTION CONTRACT (applies to finite unrolling and the modulo loop):
// Prologue on each AIV: Set mode-4 FREE[0], FREE[1] exactly once, using local
// IDs. Prime MTE3_V[0..1]. On AIC prime MTE1_MTE2[0..3]. READY starts empty.
// AIC StagePage: Wait MTE1_MTE2[slot]; DMA; Set MTE2_MTE1[slot].
// AIC compute: Wait MTE2_MTE1; ordered MTE1/M compute; release L1 only after
// its final read. Keep the single L0C accumulator until FIX_S completes.
// AIC GemmChunk: issue/complete Cube compute, Wait M_FIX; then place
// CrossCoreWaitFlag<4, PIPE_FIX>(kFlagPanelFreeWait<C>) immediately BEFORE
// FixpipeChunk<C>(). Follow Fixpipe by SetFlag<HardEvent::FIX_S>(0),
// WaitFlag<HardEvent::FIX_S>(0), then
// CrossCoreSetFlag<4, PIPE_FIX>(kFlagProductReady<C>).
// AIV RotateAndQuantise: prepare addresses, then immediately BEFORE Cast use
// CrossCoreWaitFlag<4, PIPE_V>(kFlagProductReadyWait<C>).
// After the LAST landing-panel read issue SetFlag<HardEvent::V_S>(0),
// WaitFlag<HardEvent::V_S>(0), then
// CrossCoreSetFlag<4, PIPE_V>(kFlagPanelFreeSet<C>).
// Wait MTE3_V[pack] before overwriting packed bytes OR the scale slice.
// After quantisation Set V_MTE3[pack]; drain V_S before reusing shared scratch.
// MTE3: Wait V_MTE3[pack], write packed bytes AND scale, then Set MTE3_V[pack].
// Epilogue after last chunk: AIC waits both FREE slots from BOTH owners and
// all four L1 credits. Each AIV waits both MTE3_V slots. No READY remains.
// Queue-specific MTE3 completion must reach scalar before returning to host
// (MTE3_S Set/Wait if the final MTE3_V wait only orders the V queue).
// No extra global epoch barrier is used. Never seed a reused READY flag.
// Hardware endpoint encoding must be confirmed for the installed CANN build;
// see safe_pipeline.md: this is not a CAModel validation certificate.
} // namespace turboquant_safe_pipeline
""".replace("READY0", str(ids[0])).replace("READY1", str(ids[1])).replace("FREE0", str(ids[2])).replace("FREE1", str(ids[3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=8, help="finite chunk count (default: 8)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--max-states", type=int, default=64_000_000,
                        help="per-proof resource limit; exhaustion is INCONCLUSIVE, never PASS")
    args = parser.parse_args(argv)
    if args.chunks < 1 or args.max_states < 1:
        parser.error("chunks and max-states must be positive")
    print("DaVinci consuming-latch pipeline synthesis", flush=True)
    print("Model: 1 AIC + 2 AIV; L1[4], UB_Land[2]/AIV, UB_Pack[2]/AIV", flush=True)
    print("Proof scope: generated abstract protocol, not reference-kernel machine code.")
    print("Search: all partitions of READY[0..1], FREE[0..1]; no extra throttling.")
    try:
        policy, modulo, rejected = synthesize(args.max_states)
        placed, placements = synthesize_placements(policy)
        if placed != policy:
            raise VerificationError("placement search differs from verified candidate")
        print("Wait placement: FREE after Cube, directly before FIX; READY after address preparation, before read.")
        print(f"Finite BFS: {args.chunks} chunks", flush=True)
        net, clears = build(policy, chunks=args.chunks)
        finite = explore(net, clears, max_states=args.max_states)
        print("Negative controls:", flush=True)
        mutations = negative_tests(args.max_states)
    except VerificationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        for step, name in enumerate(error.trace):
            print(f"  {step:3}: {name}", file=sys.stderr)
        return 1
    report = {
        "schema": 1, "semantics": "consuming one-bit latch; Set-on-full is rejected",
        "scope": "generated abstract protocol only; no C++/CAModel equivalence claim",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "chunks": args.chunks, "assignment": list(policy.colors),
        "ids_per_sub_block": max(policy.colors) + 1,
        "optimality": "minimum within two-slot separate-channel/no-extra-throttle grammar",
        "wait_placement": {"free": policy.free_position, "ready": policy.ready_position},
        "placement_candidates": placements,
        "finite": finite, "modulo": modulo,
        "modulo_liveness": "all states can return to initial marking; every transition live",
        "rejected_assignments": rejected, "negative_controls": mutations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hpp = args.output_dir / "safe_pipeline_flags.hpp"
    result = args.output_dir / "safe_pipeline_proof.json"
    hpp.write_text(header(policy), encoding="utf-8")
    result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, stats in (("FINITE", finite), ("MODULO (unbounded cyclic net)", modulo)):
        print(f"{label}: PASS; {stats['states']:,} reachable states / {stats['edges']:,} edges; "
              "100% reachable-state safety; 0 overwrites, 0 races, 0 deadlocks; "
              f"all states co-reachable ({stats['seconds']:.3f}s)", flush=True)
    print("Finite termination: flags/events/queues drained; free resources restored.")
    print("Modulo: recurrent liveness; weak fairness required for scheduler progress.")
    print(f"Wrote {hpp}\nWrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
