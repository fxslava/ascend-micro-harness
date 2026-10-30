#!/usr/bin/env python3
"""Capacity-one token/queue model of the reshape-and-cache MIX_AIC_1_2 pipeline.

Gate 1 of REFINEMENT_OF_COMBAT_KERNELS.md, for the inter-wave protocol. It
models one physical AI Core block -- one Cube and two Vector cores -- running
one grid block's body, and then running a SECOND grid block's body on the same
physical block, which is what a grid wider than the physical block count
actually does.

What is modelled
----------------
Every engine is an in-order queue. The scalar stream dispatches into those
queues in program order, so a scalar stall (a cross-core wait, or an x_S drain
wait) blocks everything issued after it on that core, while work already
dispatched keeps retiring. HardEvent places have capacity one, keyed by
(core, route, id): a second SET before the consuming WAIT loses a token and is
an error, not a disabled transition. Mode-4 intra-block flags are places keyed
by absolute id and shared between the Cube and one Vector core.

Buffers carry generation ownership -- the L1 page ring, each Vector core's
landing panels and packed egress slots, the L0A/L0B page region and L0C -- so a
write into a live buffer or a read of the wrong generation is an assertion
failure rather than something the flag protocol is simply trusted to prevent.
That is the point: the flag protocol is what is on trial.

The two semantics on trial
--------------------------
`consuming` decides whether `wait_intra_block` clears the flag it tested.
`clear_on_entry` decides whether entering the kernel (`InitSocState`, or the
FFTS task dispatch) clears the block's mode-4 registers. Neither is established
on this hardware, and the four combinations give different verdicts, which is
exactly why the kernel ships both channel policies:

  consuming  clear_on_entry  single-shot        recycled
  True       True            safe               safe
  True       False           safe               safe
  False      True            safe               FAILS in wave 0
  False      False           FAILS in wave 1    FAILS in wave 0

Which row the hardware is in is now measured rather than inferred.
tests/research/kernel_cluster_sync_probe.cpp ran a cyclic bilateral handshake
over RECYCLED mode-4 ids on one physical block -- 1 Cube + 2 Vector cores, no
data, an x_S drain in front of every announcement -- and completed cleanly on
physical silicon and on CAModel with no early wait retirement. **Mode-4 waits
consume**, so the live rows are 1 and 2, and both policies are safe in both;
the recycled ring is the shipped default because it costs five ids at any
schedule length instead of eleven at eight chunks.

The sticky rows are kept, and they are not idle. Two CAModel runs on this
kernel, SoC and toolkit measured a second wait on a reused id passing before
its matching set (1032 cycles early in
kernels/ops/turboquant/safe_pipeline.md, 2179 cycles early in
MICROROUTINE_DESIGN_GUIDELINES.md section 6.2c) while producing bit-exact
output, and those runs look exactly like row 3. Their cause was an
announcement issued on the scalar unit ahead of the pipeline work it reported
(guidelines 6.2a), not a non-consuming latch; the fence that fixes it is
audited by scripts/audit_cross_core_fences.py. So row 3 is the signature to
recognise if a revision drops a drain or a toolkit regression reintroduces the
reordering -- the model says what it would cost, and the trace audit is what
would see it. Row 4 is excluded by this kernel's history of passing 32-strip
and repeated-launch runs.

This is a model of the synchronisation and of buffer lifetime. It says nothing
about numerics, alignment, bus burst splitting or compiler lowering, and a pass
here is not a CAModel result. Exhaustive exploration runs on bounded instances
(2 and 4 chunks); the production 8-chunk schedule is explored by randomised and
adversarial scheduling, which is a weaker claim and is reported as such.

Run:
    python3 tests/cpu/test_reshape_wave_model.py
"""

import random
import sys

INTRA_NUM = 16
MAX_SAFE_LOCAL_ID = 10

# Engines, per core role. 'S' is the scalar stream: it dispatches into the
# other queues in program order and executes the cross-core flag operations and
# the x_S drain waits itself.
AIC_ENGINES = ("S", "MTE2", "MTE1", "M", "FIX")
AIV_ENGINES = ("S", "V", "MTE3", "MTE2")


# --------------------------------------------------------------- flag policies
class SingleShot:
    """One mode-4 id per cross-core event, never reused inside a block.

    The regression fallback. Safe under every semantics except an uncleared
    sticky latch, at the price of saturating the legal id space at eight
    chunks.
    """

    name = "single-shot"
    recycles = False

    def __init__(self, chunks, land_depth):
        self.chunks = chunks
        self.land_depth = land_depth
        self.free_base = chunks // 2

    def product_ready(self, owner, chunk):
        return owner * INTRA_NUM + chunk // 2

    def panel_free(self, owner, index):
        return owner * INTRA_NUM + self.free_base + index

    def panel_free_wait(self, owner, chunk):
        return self.panel_free(owner, chunk // 2)

    def panel_free_set(self, owner, chunk):
        return self.panel_free(owner, chunk // 2 + self.land_depth)

    def panel_free_prime(self, owner, slot):
        return self.panel_free(owner, slot)

    def epilogue_drain(self, owner, k):
        return self.panel_free(owner, self.chunks // 2 + k)

    def page_drained(self, owner):
        return owner * INTRA_NUM + self.free_base + self.chunks // 2 + self.land_depth

    def max_local_id(self):
        return self.free_base + self.chunks // 2 + self.land_depth


class Recycled:
    """A bounded bilateral two-slot ring: 2 * land_depth + 1 ids per sub block."""

    name = "recycled"
    recycles = True

    def __init__(self, chunks, land_depth):
        self.chunks = chunks
        self.land_depth = land_depth

    def product_ready(self, owner, chunk):
        return owner * INTRA_NUM + (chunk // 2) % self.land_depth

    def panel_free(self, owner, index):
        return owner * INTRA_NUM + self.land_depth + index % self.land_depth

    def panel_free_wait(self, owner, chunk):
        return self.panel_free(owner, chunk // 2)

    def panel_free_set(self, owner, chunk):
        return self.panel_free(owner, chunk // 2)

    def panel_free_prime(self, owner, slot):
        return self.panel_free(owner, slot)

    def epilogue_drain(self, owner, k):
        return self.panel_free(owner, k)

    def page_drained(self, owner):
        return owner * INTRA_NUM + 2 * self.land_depth

    def max_local_id(self):
        return 2 * self.land_depth


# ------------------------------------------------------------------- programs
class Geometry:
    """The schedule's compile-time traits, as the kernels spell them."""

    def __init__(self, chunks=8, stage_depth=4, land_depth=2, pack_depth=2,
                 lookahead=None):
        self.chunks = chunks
        self.stage_depth = min(stage_depth, chunks)
        self.land_depth = land_depth
        self.pack_depth = pack_depth
        # How far MTE2 ingress runs ahead of the Cube, in chunks. The kernels
        # use kStageDepth - 1; 1 is the adjacent-column schedule they used
        # before, kept here so the change is re-proved rather than asserted.
        self.lookahead = self.stage_depth - 1 if lookahead is None else lookahead
        assert 1 <= self.lookahead <= self.stage_depth

    def owner(self, chunk):
        return chunk & 1

    def l1_slot(self, chunk):
        return chunk % self.stage_depth

    def land_slot(self, chunk):
        return (chunk // 2) % self.land_depth

    def pack_slot(self, chunk):
        return (chunk // 2) % self.pack_depth


def cube_program(geo, flags):
    """The Cube core's instruction stream, in program order."""
    ops = []

    def emit(engine, kind, *args):
        ops.append((engine, kind, args))

    # Prologue: the resident rotation blocks reach L1 and then both L0 buffers,
    # each copy fenced by a paired barrier, then every feedback cycle is primed.
    emit("MTE2", "set", "MTE2_MTE1", 0)
    emit("MTE1", "wait", "MTE2_MTE1", 0)
    emit("MTE1", "set", "MTE1_M", 0)
    emit("M", "wait", "MTE1_M", 0)
    for slot in range(geo.stage_depth):
        emit("MTE1", "set", "MTE1_MTE2", slot)
    emit("M", "set", "M_MTE1", 0)
    emit("FIX", "set", "FIX_M", 0)

    def stage_page(chunk):
        slot = geo.l1_slot(chunk)
        emit("MTE2", "wait", "MTE1_MTE2", slot)
        emit("MTE2", "stage_page", chunk)
        emit("MTE2", "set", "MTE2_MTE1", slot)

    def gemm_chunk(chunk):
        slot = geo.l1_slot(chunk)
        emit("MTE1", "wait", "MTE2_MTE1", slot)
        emit("MTE1", "wait", "M_MTE1", 0)
        emit("MTE1", "load_l0", chunk)
        emit("MTE1", "set", "MTE1_M", 0)
        emit("MTE1", "set", "MTE1_MTE2", slot)
        emit("M", "wait", "MTE1_M", 0)
        emit("M", "wait", "FIX_M", 0)
        emit("M", "mmads", chunk)
        emit("M", "set", "M_MTE1", 0)
        emit("M", "set", "M_FIX", 0)
        # The landing panel is waited AFTER the Mmads: only the Fixpipe touches
        # Unified Buffer, so the multiplies overlap the owning Vector core's
        # previous chunk. A cross-core wait stalls the scalar stream, so
        # everything issued after it is behind the flag.
        emit("S", "xwait", flags.panel_free_wait(geo.owner(chunk), chunk))
        emit("FIX", "wait", "M_FIX", 0)
        emit("FIX", "fixpipe", chunk)
        emit("FIX", "set", "FIX_M", 0)
        # A mode-4 set issues on the scalar stream, so the Fixpipe is drained
        # into it first; without this the announcement overtakes the transfer.
        emit("FIX", "set", "FIX_S", 0)
        emit("S", "wait", "FIX_S", 0)
        emit("S", "xset", flags.product_ready(geo.owner(chunk), chunk))

    for column in range(geo.chunks + geo.lookahead):
        if column < geo.chunks:
            stage_page(column)
        if column >= geo.lookahead:
            gemm_chunk(column - geo.lookahead)

    # Epilogue: drain every primed credit, then every release nothing else
    # waits for, then every asynchronous queue into the scalar stream, then
    # tell both Vector cores.
    for slot in range(geo.stage_depth):
        emit("MTE2", "wait", "MTE1_MTE2", slot)
    emit("MTE1", "wait", "M_MTE1", 0)
    emit("M", "wait", "FIX_M", 0)
    for owner in (0, 1):
        for k in range(geo.land_depth):
            emit("S", "xwait", flags.epilogue_drain(owner, k))
    emit("MTE2", "set", "MTE2_S", 0)
    emit("S", "wait", "MTE2_S", 0)
    emit("M", "set", "M_S", 0)
    emit("S", "wait", "M_S", 0)
    emit("FIX", "set", "FIX_S", 0)
    emit("S", "wait", "FIX_S", 0)
    for owner in (0, 1):
        emit("S", "xset", flags.page_drained(owner))
    return ops


def vector_program(geo, flags, core):
    """One Vector core's instruction stream, in program order."""
    ops = []

    def emit(engine, kind, *args):
        ops.append((engine, kind, args))

    # Prologue: the quantiser's immediate constants carry no tensor hazard, so
    # the only state here is the priming. Both packed slots start free and both
    # landing panels are free for the Cube to write.
    for slot in range(geo.pack_depth):
        emit("MTE3", "set", "MTE3_V", slot)
    for slot in range(geo.land_depth):
        emit("S", "xset", flags.panel_free_prime(core, slot))

    def rotate_and_quantise(chunk):
        pack = geo.pack_slot(chunk)
        emit("S", "xwait", flags.product_ready(core, chunk))
        emit("V", "cast", chunk)
        # The landing panel is dead from here on, so it is released before the
        # rest of the stage runs. The drain makes the scalar-stream set wait
        # for the Vector queue to retire the cast.
        emit("V", "set", "V_S", pack)
        emit("S", "wait", "V_S", pack)
        # The flag IS the release, so the ownership effect belongs to that set.
        emit("S", "xset_free", flags.panel_free_set(core, chunk), chunk)
        emit("V", "quantise", chunk)
        emit("V", "wait", "MTE3_V", pack)
        emit("V", "pack", chunk)
        emit("V", "set", "V_MTE3", pack)

    def store_chunk(chunk):
        pack = geo.pack_slot(chunk)
        emit("MTE3", "wait", "V_MTE3", pack)
        emit("MTE3", "store", chunk)
        emit("MTE3", "set", "MTE3_V", pack)

    for column in range(geo.chunks + 3):
        if column >= 2 and column - 2 < geo.chunks and geo.owner(column - 2) == core:
            rotate_and_quantise(column - 2)
        if column >= 3 and column - 3 < geo.chunks and geo.owner(column - 3) == core:
            store_chunk(column - 3)

    # Epilogue: drain the primed tokens, which also retires the last MTE3
    # bursts, then await the Cube's drain, then empty this core's own queues.
    # The two cores lead with different pipes, as the kernels do.
    for slot in range(geo.pack_depth):
        emit("V", "wait", "MTE3_V", slot)
    emit("S", "xwait", flags.page_drained(core))
    order = (("V", "V_S"), ("MTE2", "MTE2_S"), ("MTE3", "MTE3_S")) if core == 0 else \
            (("MTE3", "MTE3_S"), ("V", "V_S"), ("MTE2", "MTE2_S"))
    for engine, route in order:
        emit(engine, "set", route, 0)
        emit("S", "wait", route, 0)
    return ops


# ------------------------------------------------------------------ the model
class ModelError(AssertionError):
    pass


class Machine:
    """One physical AI Core block: three cores, their queues and its buffers."""

    def __init__(self, geo, flags, consuming, clear_on_entry):
        self.geo = geo
        self.flags = flags
        self.consuming = consuming
        self.clear_on_entry = clear_on_entry
        self.programs = [
            ("AIC", AIC_ENGINES, cube_program(geo, flags)),
            ("AIV0", AIV_ENGINES, vector_program(geo, flags, 0)),
            ("AIV1", AIV_ENGINES, vector_program(geo, flags, 1)),
        ]
        # Per core: for each engine, the op list, and for each of those ops the
        # number of scalar ops that precede it in program order. The scalar
        # stream dispatches in program order, so an op cannot execute until the
        # scalar stream has reached it.
        self.lanes = []
        for _name, engines, ops in self.programs:
            per_engine = {e: [] for e in engines}
            scalar_seen = 0
            for engine, kind, args in ops:
                per_engine[engine].append((kind, args, scalar_seen))
                if engine == "S":
                    scalar_seen += 1
            self.lanes.append(per_engine)
        # Place keys, in a fixed order, so a marking is a tuple.
        self.event_keys = []
        self.mode4_keys = []
        for core_index, (_name, engines, ops) in enumerate(self.programs):
            for engine, kind, args in ops:
                if kind in ("set", "wait"):
                    key = (core_index, args[0], args[1])
                    if key not in self.event_keys:
                        self.event_keys.append(key)
                elif kind in ("xset", "xset_free", "xwait"):
                    if args[0] not in self.mode4_keys:
                        self.mode4_keys.append(args[0])
        self.event_index = {k: i for i, k in enumerate(self.event_keys)}
        self.mode4_index = {k: i for i, k in enumerate(self.mode4_keys)}
        # Buffers, each holding a generation id or -1 for free.
        self.buffers = (
            [("l1", s) for s in range(geo.stage_depth)]
            + [("l0page", 0), ("l0c", 0)]
            + [("land", c, s) for c in (0, 1) for s in range(geo.land_depth)]
            + [("pack", c, s) for c in (0, 1) for s in range(geo.pack_depth)]
        )
        self.buffer_index = {b: i for i, b in enumerate(self.buffers)}

    # -- initial and terminal markings ------------------------------------
    def initial(self, carry=None):
        events = [0] * len(self.event_keys)
        mode4 = [0] * len(self.mode4_keys)
        buffers = [-1] * len(self.buffers)
        if carry is not None:
            events, mode4, buffers = (list(carry[0]), list(carry[1]), list(carry[2]))
            if self.clear_on_entry:
                mode4 = [0] * len(self.mode4_keys)
        pcs = tuple(0 for _ in self.programs)
        pipe = tuple(
            tuple(0 for _ in engines) for _name, engines, _ops in self.programs)
        return (pcs, pipe, tuple(events), tuple(mode4), tuple(buffers))

    def finished(self, state):
        pcs, pipe, _e, _m, _b = state
        for core_index, (_name, engines, _ops) in enumerate(self.programs):
            for e_index, engine in enumerate(engines):
                if pipe[core_index][e_index] != len(self.lanes[core_index][engine]):
                    return False
        return True

    # -- one transition ----------------------------------------------------
    def step(self, state, core_index, engine_index):
        pcs, pipe, events, mode4, buffers = state
        _name, engines, _ops = self.programs[core_index]
        engine = engines[engine_index]
        lane = self.lanes[core_index][engine]
        index = pipe[core_index][engine_index]
        if index == len(lane):
            return None
        kind, args, scalar_needed = lane[index]
        # The scalar stream dispatches in program order: a pipe op cannot run
        # before the scalar stream has issued it.
        if engine != "S" and pcs[core_index] < scalar_needed:
            return None

        events = list(events)
        mode4 = list(mode4)
        buffers = list(buffers)

        def buf(key):
            return buffers[self.buffer_index[key]]

        def claim(key, generation, what):
            if buf(key) != -1:
                raise ModelError(
                    "%s: %s is still live with generation %d" % (what, key, buf(key)))
            buffers[self.buffer_index[key]] = generation

        def require(key, generation, what):
            if buf(key) != generation:
                raise ModelError(
                    "%s: %s holds generation %s, not %d"
                    % (what, key, buf(key), generation))

        def release(key):
            buffers[self.buffer_index[key]] = -1

        geo = self.geo
        if kind == "set":
            slot = self.event_index[(core_index, args[0], args[1])]
            if events[slot] != 0:
                raise ModelError("double SET loses a token on %s.%s id %d"
                                 % (self.programs[core_index][0], args[0], args[1]))
            events[slot] = 1
        elif kind == "wait":
            slot = self.event_index[(core_index, args[0], args[1])]
            if events[slot] == 0:
                return None
            events[slot] = 0
        elif kind in ("xset", "xset_free"):
            slot = self.mode4_index[args[0]]
            if mode4[slot] != 0:
                if self.consuming:
                    raise ModelError(
                        "a second announcement on mode-4 id %d before the first "
                        "was consumed" % args[0])
                # A sticky latch simply stays set: the set is lost, which is
                # how a reused id stops carrying information.
            mode4[slot] = 1
            if kind == "xset_free":
                key = ("land", geo.owner(args[1]), geo.land_slot(args[1]))
                require(key, args[1], "panel release")
                release(key)
        elif kind == "xwait":
            slot = self.mode4_index[args[0]]
            if mode4[slot] == 0:
                return None
            if self.consuming:
                mode4[slot] = 0
        elif kind == "stage_page":
            claim(("l1", geo.l1_slot(args[0])), args[0], "MTE2 ingress")
        elif kind == "load_l0":
            require(("l1", geo.l1_slot(args[0])), args[0], "MTE1 fractal load")
            claim(("l0page", 0), args[0], "MTE1 fractal load")
            release(("l1", geo.l1_slot(args[0])))
        elif kind == "mmads":
            require(("l0page", 0), args[0], "Mmad")
            claim(("l0c", 0), args[0], "Mmad")
            release(("l0page", 0))
        elif kind == "fixpipe":
            require(("l0c", 0), args[0], "Fixpipe")
            claim(("land", geo.owner(args[0]), geo.land_slot(args[0])), args[0],
                  "Fixpipe egress")
            release(("l0c", 0))
        elif kind == "cast":
            require(("land", geo.owner(args[0]), geo.land_slot(args[0])), args[0],
                    "Vector cast of the landing panel")
        elif kind == "quantise":
            pass
        elif kind == "pack":
            claim(("pack", geo.owner(args[0]), geo.pack_slot(args[0])), args[0],
                  "INT4 nibble pack")
        elif kind == "store":
            require(("pack", geo.owner(args[0]), geo.pack_slot(args[0])), args[0],
                    "MTE3 cache burst")
            release(("pack", geo.owner(args[0]), geo.pack_slot(args[0])))
        else:
            raise AssertionError("unknown op %s" % kind)

        pipe = list(list(p) for p in pipe)
        pipe[core_index][engine_index] += 1
        pcs = list(pcs)
        if engine == "S":
            pcs[core_index] += 1
        return (tuple(pcs), tuple(tuple(p) for p in pipe), tuple(events),
                tuple(mode4), tuple(buffers))

    def enabled(self, state):
        out = []
        for core_index, (_name, engines, _ops) in enumerate(self.programs):
            for engine_index in range(len(engines)):
                nxt = self.step(state, core_index, engine_index)
                if nxt is not None:
                    out.append((core_index, engine_index, nxt))
        return out


def explore(machine, carry=None, max_states=4_000_000):
    """Exhaustive breadth-first exploration of every engine interleaving."""
    start = machine.initial(carry)
    seen = {start}
    frontier = [start]
    terminals = set()
    while frontier:
        state = frontier.pop()
        moves = machine.enabled(state)
        if not moves:
            if not machine.finished(state):
                raise ModelError("deadlock: no engine can advance and the "
                                 "programs are not finished")
            terminals.add((state[2], state[3], state[4]))
            continue
        for _core, _engine, nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                if len(seen) > max_states:
                    raise ModelError("state limit %d exceeded" % max_states)
                frontier.append(nxt)
    if not terminals:
        raise ModelError("no terminal state")
    return seen, terminals


def walk(machine, carry=None, seed=0, steps=400_000):
    """One randomised schedule, for geometries too large to explore fully."""
    rng = random.Random(seed)
    state = machine.initial(carry)
    for _ in range(steps):
        moves = machine.enabled(state)
        if not moves:
            if not machine.finished(state):
                raise ModelError("deadlock on randomised schedule seed %d" % seed)
            return state
        state = rng.choice(moves)[2]
    raise ModelError("randomised schedule did not terminate")


def check_terminal(machine, terminals, label):
    """A drained block leaves no event bit, no live buffer and, under consuming
    waits, no mode-4 flag either."""
    for events, mode4, buffers in terminals:
        if any(events):
            leaked = [machine.event_keys[i] for i, v in enumerate(events) if v]
            raise ModelError("%s: HardEvent tokens leak past the block: %s"
                             % (label, leaked))
        if any(b != -1 for b in buffers):
            live = [machine.buffers[i] for i, b in enumerate(buffers) if b != -1]
            raise ModelError("%s: buffers still live at block exit: %s" % (label, live))
        if machine.consuming and any(mode4):
            leaked = [machine.mode4_keys[i] for i, v in enumerate(mode4) if v]
            raise ModelError("%s: mode-4 flags leak past the block: %s" % (label, leaked))


def ids_legal(machine):
    for abs_id in machine.mode4_keys:
        local = abs_id % INTRA_NUM
        if local > MAX_SAFE_LOCAL_ID or abs_id >= 2 * INTRA_NUM:
            raise ModelError("mode-4 id %d is in the library's reserved range" % abs_id)


# ------------------------------------------------------------------ scenarios
class LimitExceeded(Exception):
    """The exhaustive search hit its state cap: not a verdict either way."""


def run_exhaustive(machine, cap):
    counts = []
    carry = None
    for wave in (0, 1):
        try:
            seen, terminals = explore(machine, carry=carry, max_states=cap)
        except ModelError as error:
            if "state limit" in str(error):
                raise
            raise ModelError("wave %d: %s" % (wave, error))
        check_terminal(machine, terminals, "wave %d" % wave)
        if len(terminals) != 1:
            raise ModelError("wave %d has %d distinct terminal markings"
                             % (wave, len(terminals)))
        carry = next(iter(terminals))
        counts.append(len(seen))
    return "exhaustive, %s states" % " + ".join(str(c) for c in counts)


def run_sampled(machine, seeds):
    carry = None
    for wave in (0, 1):
        ends = set()
        for seed in range(seeds):
            try:
                end = walk(machine, carry=carry, seed=seed + 1000 * wave)
            except ModelError as error:
                raise ModelError("wave %d: %s" % (wave, error))
            ends.add((end[2], end[3], end[4]))
        check_terminal(machine, ends, "wave %d" % wave)
        if len(ends) != 1:
            raise ModelError("wave %d reached %d distinct terminal markings"
                             % (wave, len(ends)))
        carry = next(iter(ends))
    return "randomised, %d+%d schedules" % (seeds, seeds)


def run_case(policy_class, geo, consuming, clear_on_entry, exhaustive, seeds=24,
             cap=600_000):
    """Wave 0 then wave 1 on the same physical block. Returns a note or raises.

    Exhaustive exploration is attempted where asked for and falls back to
    randomised scheduling when the state cap is reached, which is reported
    rather than silently downgraded.
    """
    flags = policy_class(geo.chunks, geo.land_depth)
    machine = Machine(geo, flags, consuming, clear_on_entry)
    ids_legal(machine)
    if exhaustive:
        try:
            return run_exhaustive(machine, cap)
        except ModelError as error:
            if "state limit" not in str(error):
                raise
    return run_sampled(machine, seeds)


SEMANTICS = (
    # (consuming, clear_on_entry, label)
    (True, True, "consuming wait, cleared at entry"),
    (True, False, "consuming wait, never cleared"),
    (False, True, "sticky latch, cleared at entry"),
    (False, False, "sticky latch, never cleared"),
)

def reuses_ids(policy, geo):
    """Whether this geometry actually makes the policy reuse an id.

    A recycled ring only recycles once a core has more landing generations than
    the ring has slots: at four chunks over a depth-2 ring each core runs
    exactly two generations, so even the recycled policy gives every event its
    own id and the two policies are indistinguishable. Only the eight-chunk
    schedule exercises reuse, which is why the small instances are the
    exhaustive ones and the eight-chunk one is where the two policies diverge.
    """
    return policy.recycles and (geo.chunks // 2) > geo.land_depth


def expectation(policy, geo, consuming, clear_on_entry):
    """What the model must conclude, given the semantics and the geometry.

    "pass" means two waves complete with every queue empty and every buffer
    free; "fail" means the model must raise, and a run in which it does not is
    itself a failure.
    """
    if not consuming and reuses_ids(policy, geo):
        # A sticky flag that a wait does not clear makes the SECOND
        # announcement on a reused id a no-op, so the consumer's second wait
        # passes on the first announcement. This is the shape both CAModel
        # traces had -- by a different cause, an undrained announcement -- and
        # so the failure mode to recognise rather than a hardware prediction.
        return "fail"
    if not consuming and not clear_on_entry:
        # Every id this block set is still set when the next wave's block
        # starts on the same physical core, so its first wait passes at once.
        return "fail"
    return "pass"


def main():
    failures = 0
    print("reshape-and-cache MIX_AIC_1_2 inter-wave model: two grid blocks on one "
          "physical block")

    # (chunks, land_depth, stage_depth, lookahead, exhaustive)
    # The 2- and 4-chunk instances are bounded and explored exhaustively; the
    # production 8-chunk schedule is sampled. Lookahead 1 is the adjacent-column
    # schedule this kernel used before and 3 is kStageDepth - 1, which it uses
    # now, so both marked graphs are checked rather than one assumed from the
    # other.
    for chunks, land_depth, stage_depth, lookahead, exhaustive in (
            (2, 1, 2, 1, True),
            (4, 2, 4, 1, True),
            (4, 2, 4, 3, True),
            (8, 2, 4, 1, False),
            (8, 2, 4, 3, False)):
        geo = Geometry(chunks=chunks, stage_depth=stage_depth,
                       land_depth=land_depth, lookahead=lookahead)
        print("\n%d chunks, depth-%d L1 ring, depth-%d landing ring, "
              "ingress lookahead %d" % (chunks, stage_depth, land_depth, lookahead),
              flush=True)
        for policy in (SingleShot, Recycled):
            flags = policy(chunks, land_depth)
            budget = flags.max_local_id()
            if policy.recycles and not reuses_ids(policy, geo):
                print("  ---- %s: this geometry gives every event its own id "
                      "anyway, so it does not exercise reuse" % policy.name,
                      flush=True)
            for consuming, clear_on_entry, label in SEMANTICS:
                want = expectation(policy, geo, consuming, clear_on_entry)
                try:
                    note = run_case(policy, geo, consuming, clear_on_entry, exhaustive)
                    got, detail = "pass", note
                except ModelError as error:
                    got, detail = "fail", str(error)
                verdict = "PASS" if got == want else "UNEXPECTED"
                if got != want:
                    failures += 1
                print("  %-10s %-11s ids<=%2d  %-30s %-4s  %s"
                      % (verdict, policy.name, budget, label, got, detail),
                      flush=True)

    # Negative controls: the model must reject a schedule that drops a rule the
    # protocol insists on. Each mutation is applied to the production geometry
    # under the safest semantics, so only the mutation can make it fail.
    print("\nnegative controls (consuming wait, cleared at entry, 4 chunks)")
    geo = Geometry(chunks=4, stage_depth=4, land_depth=2)
    for mutation, why in (
        ("no_prime", "the landing credits are never primed"),
        ("no_drain", "the epilogue does not drain the final releases"),
        ("double_announce", "one chunk is announced twice"),
    ):
        flags = SingleShot(geo.chunks, geo.land_depth)
        machine = Machine(geo, flags, True, True)
        mutate(machine, mutation)
        try:
            seen, terminals = explore(machine)
            check_terminal(machine, terminals, "mutated wave 0")
        except ModelError as error:
            print("  PASS rejected %-16s %s: %s" % (mutation, why, error))
        else:
            print("  UNEXPECTED accepted %s: %s" % (mutation, why))
            failures += 1

    print()
    if failures:
        print("FAIL reshape wave model: %d unexpected verdict(s)" % failures)
        return 1
    print("PASS reshape wave model: both policies agree with the semantics matrix, "
          "the recycled default is safe across waves under the measured consuming "
          "semantics, the single-shot fallback is safe under every semantics except "
          "an uncleared sticky latch, and three negative controls are rejected")
    return 0


def mutate(machine, mutation):
    """Rebuild one core's lanes with a protocol rule removed."""
    core = 1  # AIV0 primes the credits and the Cube drains them
    _name, engines, ops = machine.programs[core]
    if mutation == "no_prime":
        ops = [op for op in ops
               if not (op[1] == "xset"
                       and op[2][0] in {machine.flags.panel_free_prime(0, s)
                                        for s in range(machine.geo.land_depth)})]
    elif mutation == "no_drain":
        core = 0
        _name, engines, ops = machine.programs[0]
        drains = {machine.flags.epilogue_drain(o, k)
                  for o in (0, 1) for k in range(machine.geo.land_depth)}
        ops = [op for op in ops if not (op[1] == "xwait" and op[2][0] in drains)]
    elif mutation == "double_announce":
        core = 0
        _name, engines, ops = machine.programs[0]
        out = []
        for op in ops:
            out.append(op)
            if op[1] == "xset" and op[2][0] == machine.flags.product_ready(0, 0):
                out.append(op)
        ops = out
    else:
        raise AssertionError(mutation)
    machine.programs[core] = (_name, engines, ops)
    rebuilt = {e: [] for e in engines}
    scalar_seen = 0
    for engine, kind, args in ops:
        rebuilt[engine].append((kind, args, scalar_seen))
        if engine == "S":
            scalar_seen += 1
    machine.lanes[core] = rebuilt


if __name__ == "__main__":
    sys.exit(main())
