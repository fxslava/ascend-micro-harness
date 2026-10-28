#!/usr/bin/env python3
"""Audit the MIX_AIC_1_2 paged decode attention operator from its CAModel traces.

Usage:
  analyze_paged_decode_trace.py [--strips N] <aiv0_popped> <aiv0_complete>
                                <aiv1_popped> <aiv1_complete>
                                <aic_popped>  <aic_complete>

--strips is the strip count the kernel was launched with, 8 if omitted. Every
per-strip assertion is scaled by it; none is relaxed.

Nine numbered checks, each fatal (1b counts with 1):

  0. the audited core RAN. An idle core in this grid is not an empty one -- it
     issues a scalar prologue and an early return, and every balance check
     below is vacuously true on it -- so each core must first show the work it
     is defined by: AIV1 its UB -> L1 staging bursts, the Cube its rotation
     Mmads, AIV0 at least one vector instruction. The grid is 32 blocks and
     block N runs on core N, so this is also what catches auditing core0 on a
     run that isolated another block;
  1. intra-core SET_FLAG / WAIT_FLAG balance per core and per
     (pipe, trigger pipe, flag id), so no HardEvent token leaks, and no use of
     the reserved ids 6 and 7;
  1b. SET before WAIT in program order on every intra-core route. Instruction
     ids within one core ARE program order, and an intra-core SetFlag/WaitFlag
     pair is not a semaphore that waits for a set still to come: the set must
     be reached first. Counting cannot catch a violation -- the counts balance
     either way -- and the model's own diagnostic for it is
     "has no matched set_flag";
  2. cross-core SET_INTRA_BLOCK / WAIT_INTRA_BLOCK pairing per ABSOLUTE flag
     id. An AIV reports sync_id inside its own sub-block half while the AIC
     reports the absolute id, so AIV1's ids are rebased by INTRA_NUM before
     pairing -- that rebasing is itself the evidence that the sub-block
     convention holds (guidelines section 6.2d);
  3. **the L1 key pages, from the data instructions.** Every page is staged in
     16 MOV_UB_TO_L1 bursts and read by exactly one LOAD_2Dv2, after its last
     write, and there are as many pages as strips. See below for what this
     replaced and why;
  4. the double-handshake quorum: before each of its GEMMs the Cube waited one
     staging flag from each of the two Vector cores;
  5. the instruction budget: 64 rotation Mmads plus 8 per strip, one Fixpipe
     for the rotated query plus one per strip, and 16 UB -> L1 bursts per
     strip (a 256-channel bf16 panel is 16 C0 column strips);
  6. no Global Memory store anywhere on the staging core, which is how "the
     dequantised key page never leaves the chip" is proved rather than
     asserted;
  7. every Global Memory store on the recurrence core happens after the last
     Fixpipe, which is what "one outflow at the end of the sequence" means in
     the trace rather than in the source;
  8. both Vector cores executed, with a per-engine instruction census.

Then it reports the per-strip steady-state interval and merged engine overlap.
Ticks come from one CAModel clock in all three dumps, so they are directly
comparable.

Why check 3 asks about buffers and not about flags
-------------------------------------------------
Check 3 used to assert that each cross-core flag id's merged event sequence
alternates SET, WAIT, SET, WAIT -- the property that would rule out the stale
mode-4 set recorded in guidelines section 6.2c, where a reused id released a
wait 2179 cycles before the announcement it was waiting for. That check was
unsound and had to be removed.

The reason is in the API reference: CrossCoreWaitFlag blocks "the issuance of
subsequent instructions of the current core", so the wait instruction retires
as soon as the queue reaches it and arms a block on its pipe's NEXT
instruction. Its tick in either dump is therefore not the moment its flag
released it. Measured: on the 2-strip rung the Cube's drain wait on id 4 was
dispatched at tick 8274 and completed at 26699, against the AIV0 set that
released it at 26688 -- ordered by dispatch tick, the wait precedes its own
set. The check fired on every rung, including the numerically correct ones, so
it was a false-positive generator; moving it to the completion log moved the
complaints without fixing the unsoundness. The ordering simply is not
recoverable from flag instruction ticks.

So check 3 now tests the property the flags exist to PROVIDE, using the data
instructions whose ticks do mean what they say: if the staging handshake works,
each L1 key page is written exactly once as 16 bursts and read exactly once
afterwards. Run against the archived dumps of the runs that were wrong, it
passes 2 strips and fails 4 and 8, naming the page and the tick of each extra
read -- which is the only evidence that it works.

What it does NOT establish: anything about how the hardware clears a mode-4
level on a wait. It establishes that on these runs no L1 key page was ever
overwritten before its reader had read it.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)")
SYNC = re.compile(r"(SET|WAIT)_INTRA_BLOCK\s+PIPE:(\w+).*?sync_id:(\d+)")

# AscendC::INTRA_NUM: a mode-4 flag id space of 16 per sub block.
INTRA_NUM = 16

# paged_decode::kStageDepth. Two slots everywhere -- the "double window" the
# score panels are named for -- so two columns fill the pipeline whatever the
# strip count, and the steady state starts after them.
STAGE_DEPTH = 2

# The Cube gates its MTE1 loads on the staging quorum and its Fixpipe on the
# score-window back edge, so the wait pipe separates the two kinds of flag.
QUORUM_PIPE = "MTE1"

# Per-strip instruction budget. The rotation prologue is fixed.
ROTATION_MMADS = 64
SCORE_MMADS_PER_STRIP = 8
CV_BURSTS_PER_STRIP = 16
# kRingKey in the kernel: one bf16 [16 x 256] key page plus the 32-byte bank
# skew. The audit groups an L1 page's staging bursts by this window.
L1_KEY_PAGE_STRIDE = 8224   # a 256-channel bf16 panel is 16 C0 column strips
ROTATION_FIXPIPES = 1      # L0C -> GM, the rotated query

VECTOR_PIPES = {"RVECEX", "RVECST", "RVECLD", "RVECSU", "RVECLP", "VEC"}


def read(path):
    """Return {instruction id: (tick, pipe, operation)}."""
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            tick, pipe, instr_id, operation = match.groups()
            records[int(instr_id)] = (int(tick), pipe, operation)
    return records


def merge(intervals):
    spans = []
    for begin, end in sorted(intervals):
        if spans and begin <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([begin, end])
    return [(a, b) for a, b in spans]


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, stop = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < stop:
            out.append((start, stop))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def kernel_body_ran(v0, v1, c):
    """Fail unless all three audited cores actually executed the kernel body.

    Checks 1, 1b and 2 below are balance properties: on an empty instruction
    stream every one of them is vacuously true and prints "balanced on 0
    route(s)". That is how six rungs of a twelve-rung ladder came back green
    off a core that had done nothing. The grid is 32 blocks and the model
    dumps all 32 cores; block N runs on core N, so auditing core 0 while the
    runner isolated block 16 reads a core that took the early return. Only
    check 3 noticed, and only because it asks a question about data
    instructions rather than about a balance.

    So the audit states up front that it is reading the right core, in terms
    of the work each core is defined by: AIV1 stages the key fractal UB -> L1,
    the Cube issues Mmads, AIV0 runs the recurrence on the vector pipes.
    """
    counts = {}
    for name, records in (("AIV0", v0), ("AIV1", v1), ("AIC", c)):
        if name == "AIV1":
            want, what = CV_BURSTS_PER_STRIP, "UB -> L1 staging burst"
            saw = sum(1 for _, (_, _, op) in records.items()
                      if "MOV_UB_TO_L1" in op)
        elif name == "AIC":
            want, what = ROTATION_MMADS, "Mmad"
            saw = sum(1 for _, (_, _, op) in records.items() if "MMAD" in op)
        else:
            want, what = 1, "vector instruction"
            saw = sum(1 for _, (_, pipe, _) in records.items()
                      if pipe in VECTOR_PIPES)
        if saw < want:
            raise SystemExit(
                f"{name} issued {saw} {what}(s), expected at least {want}: "
                f"this core did not run the kernel body. The grid is 32 "
                f"blocks and block N runs on core N, so audit core<block> "
                f"rather than core0 on any run that isolated a block "
                f"other than 0.")
        counts[name] = saw
    print(f"  all three audited cores ran the kernel body: "
          f"{counts['AIV1']} staging burst(s), {counts['AIC']} Mmad(s), "
          f"{counts['AIV0']} vector instruction(s)")


def hard_event_balance(name, records):
    sets, waits = Counter(), Counter()
    for _, (_, _, operation) in records.items():
        flag = FLAG.search(operation)
        if not flag:
            continue
        key = (flag.group(1), flag.group(2), int(flag.group(3), 0))
        if "SET_FLAG" in operation:
            sets[key] += 1
        elif "WAIT_FLAG" in operation:
            waits[key] += 1
    if sets != waits:
        leaked = {k: (sets[k], waits[k]) for k in set(sets) | set(waits)
                  if sets[k] != waits[k]}
        raise SystemExit(
            f"{name}: intra-core event leak on (pipe, trigger, id) -> "
            f"(sets, waits): {leaked}"
        )
    reserved = [k for k in sets if k[2] in (6, 7)]
    if reserved:
        raise SystemExit(f"{name}: event IDs 6 and 7 are reserved, saw {reserved}")
    return sets


def cross_core(records, rebase, pipe=None):
    """{(kind, absolute flag id): count} for one core, optionally one pipe."""
    out = Counter()
    for _, (_, _, operation) in records.items():
        sync = SYNC.search(operation)
        if sync:
            kind, sync_pipe, sync_id = sync.groups()
            if pipe is not None and sync_pipe != pipe:
                continue
            out[(kind, int(sync_id) + rebase)] += 1
    return out


def cross_core_events(records, rebase, core):
    """[(tick, kind, absolute id, core)] for one core, in no particular order."""
    events = []
    for _, (tick, _, operation) in records.items():
        sync = SYNC.search(operation)
        if sync:
            kind, _, sync_id = sync.groups()
            events.append((tick, kind, int(sync_id) + rebase, core))
    return events


def hard_event_program_order(name, records):
    """Check 1b. Per intra-core route, SET must be ISSUED before its WAIT.

    An intra-core SetFlag/WaitFlag pair is not a blocking semaphore that will
    wait for a set still to come: the scalar stream drops the set into the
    producing pipe's queue and the wait into the consuming pipe's, so the set
    has to be reached FIRST in program order. Get that backwards and the model
    says `has no matched set_flag` and lets the wait through, after which the
    producer's later set collides with a flag still standing
    (`execute_set_flag already has same set_flag`).

    Counting is not enough to catch it -- the counts balance either way. What
    catches it is the order, and within one core program order is the order the
    instruction ids were assigned, which is what this walks.

    This is check 1b because it was added after the model caught, on AIV0, a
    WaitFlag<V_MTE2> in LoadValueStrip(s+2) issued one column before the
    SetFlag<V_MTE2> in RecurStrip(s) that releases it. The arithmetic balanced;
    the order did not. The run's numbers were correct anyway, which is the
    reason to check this rather than trust them.
    """
    routes = defaultdict(list)
    for instr_id, (_, _, operation) in sorted(records.items()):
        flag = FLAG.search(operation)
        if not flag:
            continue
        if "SET_FLAG" in operation:
            kind = "SET"
        elif "WAIT_FLAG" in operation:
            kind = "WAIT"
        else:
            continue
        pipe, trigger = flag.group(1), flag.group(2)
        routes[(pipe, trigger, int(flag.group(3), 0))].append((instr_id, kind))
    for route, seq in sorted(routes.items()):
        pipe, trigger, flag = route
        pending = 0
        for instr_id, kind in seq:
            if kind == "SET":
                pending += 1
                if pending > 1:
                    raise SystemExit(
                        f"{name} {trigger}_{pipe} id {flag}: a second SET at "
                        f"instruction {instr_id} while the previous one was "
                        f"still unconsumed -- the flag was already standing.")
            else:
                pending -= 1
                if pending < 0:
                    raise SystemExit(
                        f"{name} {trigger}_{pipe} id {flag}: WAIT at "
                        f"instruction {instr_id} is issued before any SET of "
                        f"that flag. An intra-core wait does not wait for a "
                        f"set still to come; the set must be reached first in "
                        f"program order.")
        if pending != 0:
            raise SystemExit(
                f"{name} {trigger}_{pipe} id {flag}: {pending} SET(s) left "
                f"standing at the end of the launch -- a leaked event bit.")
    return len(routes)


def l1_key_page_order(v1_records, c_records):
    """Check 3. Every L1 key page is written once and read after it was written.

    This replaces an earlier check that tried to prove the same thing from the
    cross-core flag instructions, which cannot work. CrossCoreWaitFlag is a
    FLOWCTRL instruction that retires as soon as it is reached and arms a block
    on the *subsequent* instructions of its pipe -- the API reference says
    "block the issuance of subsequent instructions of the current core" -- so
    neither dump's tick for a WAIT_INTRA_BLOCK is the moment the flag released
    it, and ordering a handshake by those ticks is meaningless. The old check
    duly reported a fault on every rung, including ones that were numerically
    correct.

    The property the flags exist to guarantee is checkable directly, and this
    is where the real fault was found: on the 8-strip rung AIV1 wrote L1
    0xc060 at ticks 6754, 8703, 10653 and 12608 while the Cube read that same
    address at 7088, 26994, 66481 and 106697, so two of the four reads
    returned a page that had been overwritten twice. Writes and reads of an L1
    address are MTE3 and MTE1 data instructions whose completion ticks mean
    what they say.
    """
    writes = defaultdict(list)
    for _, (tick, _, operation) in v1_records.items():
        if "MOV_UB_TO_L1" not in operation:
            continue
        m = re.search(r"XD:X\d+=0x([0-9a-f]+)", operation)
        if m:
            writes[int(m.group(1), 16)].append(tick)
    reads = defaultdict(list)
    for _, (tick, _, operation) in c_records.items():
        if "LOAD_2Dv2" not in operation:
            continue
        m = re.search(r"XN:X\d+=0x([0-9a-f]+)", operation)
        if m:
            reads[int(m.group(1), 16)].append(tick)

    # A key page base is an L1 address the Cube loaded a key fractal from,
    # which is an address AIV1 also staged into. The page's other
    # CV_BURSTS_PER_STRIP - 1 bursts land at higher addresses inside the same
    # L1_KEY_PAGE_STRIDE window, so group the writes by window rather than by
    # exact address.
    pages = sorted(set(writes) & set(reads))
    if not pages:
        raise SystemExit(
            "no L1 address was both staged by AIV1 and loaded by the Cube: "
            "the key path did not run, or the dump format changed")
    for base in pages:
        r = sorted(reads[base])
        window = [t for addr, ticks in writes.items()
                  if base <= addr < base + L1_KEY_PAGE_STRIDE for t in ticks]
        w = sorted(window)
        if len(r) != 1:
            raise SystemExit(
                f"L1 key page 0x{base:x} was loaded {len(r)} times at ticks "
                f"{r}; with one page per strip the Cube reads each page once. "
                f"More than one read means two strips share a page, and the "
                f"later staging overwrites keys an earlier read still needs.")
        if len(w) != CV_BURSTS_PER_STRIP:
            raise SystemExit(
                f"L1 key page 0x{base:x} took {len(w)} staging bursts, "
                f"expected {CV_BURSTS_PER_STRIP}. More than that many means a "
                f"second strip was staged over this page.")
        if r[0] < w[-1]:
            raise SystemExit(
                f"L1 key page 0x{base:x}: the Cube loaded it at tick {r[0]} "
                f"but AIV1's staging bursts ran to tick {w[-1]} -- the Cube "
                f"read the page before it was fully written.")
    print(f"  {len(pages)} L1 key page(s), each staged in "
          f"{CV_BURSTS_PER_STRIP} bursts and read exactly once, after its "
          f"last write")
    return pages


def dispatch_ticks(records, keep):
    """Dispatch ticks of the instructions keep() selects, in issue order."""
    return sorted(tick for _, (tick, pipe, operation) in records.items()
                  if keep(pipe, operation))


def cadence(name, ticks, strips, expected=None):
    if expected is not None and len(ticks) != expected:
        raise SystemExit(
            f"{name}: expected {expected} per-strip events, saw {len(ticks)}"
        )
    if len(ticks) < 2:
        print(f"  {name:<30}: {len(ticks)} event(s), no interval")
        return []
    deltas = [b - a for a, b in zip(ticks, ticks[1:])]
    steady = deltas[STAGE_DEPTH - 1:] if len(deltas) > STAGE_DEPTH else []
    print(f"  {name:<30}: {len(ticks)} events, deltas "
          f"{' '.join(str(d) for d in deltas)}")
    if steady:
        mid = sorted(steady)[len(steady) // 2]
        print(f"  {'':<30}  steady state ({len(steady)} intervals, columns "
              f"{STAGE_DEPTH}..{strips - 1}): min {min(steady)}, median {mid}, "
              f"max {max(steady)}")
    else:
        print(f"  {'':<30}  no steady-state interval: the ring never wrapped "
              f"at {strips} strip(s), so this run is fill and drain only")
    return steady


def busy(records, complete, keep):
    intervals = []
    for instr_id, (begin, pipe, operation) in records.items():
        if instr_id not in complete or not keep(pipe, operation):
            continue
        end = complete[instr_id][0]
        if end <= begin:
            end = begin + 1
        intervals.append((begin, end))
    return merge(intervals)


def main(aiv0_popped, aiv0_complete, aiv1_popped, aiv1_complete,
         aic_popped, aic_complete, strips):
    v0_in, v0_out = read(aiv0_popped), read(aiv0_complete)
    v1_in, v1_out = read(aiv1_popped), read(aiv1_complete)
    c_in, c_out = read(aic_popped), read(aic_complete)

    print(f"paged decode attention trace audit, {strips} strip(s), "
          f"ring depth {STAGE_DEPTH}")

    # ---- 0. the audited cores are the ones that ran -------------------
    kernel_body_ran(v0_in, v1_in, c_in)

    # ---- 1. intra-core HardEvent balance ------------------------------
    for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
        events = hard_event_balance(name, records)
        routes = hard_event_program_order(name, records)
        print(f"  {name} intra-core events balanced and set-before-wait on "
              f"all {routes} route(s): {sum(events.values())} set/wait pairs")

    # ---- 2. cross-core pairing on absolute ids ------------------------
    aiv = cross_core(v0_in, 0) + cross_core(v1_in, INTRA_NUM)
    aic = cross_core(c_in, 0)
    ids = {flag for _, flag in list(aiv) + list(aic)}
    for flag in sorted(ids):
        if aiv[("SET", flag)] != aic[("WAIT", flag)] or \
           aic[("SET", flag)] != aiv[("WAIT", flag)]:
            raise SystemExit(
                f"cross-core flag {flag} unpaired: AIV set "
                f"{aiv[('SET', flag)]} / AIC wait {aic[('WAIT', flag)]}, "
                f"AIC set {aic[('SET', flag)]} / AIV wait {aiv[('WAIT', flag)]}"
            )
    for flag in sorted(ids):
        if flag >= 11 and flag < INTRA_NUM or flag >= INTRA_NUM + 11:
            raise SystemExit(
                f"cross-core flag {flag} is at or above the library-reserved "
                f"11..14 band in its own sub-block half"
            )
    print(f"  cross-core flags paired on {len(ids)} absolute id(s): "
          f"{sorted(ids)}")

    # ---- 3. the L1 key pages, from the data instructions ---------------
    pages = l1_key_page_order(v1_out, c_out)
    if len(pages) != strips:
        raise SystemExit(
            f"{len(pages)} L1 key page(s) for {strips} strip(s): every strip "
            f"must own one page")

    # ---- 4. the double-handshake quorum -------------------------------
    quorum = cross_core(c_in, 0, pipe=QUORUM_PIPE)
    from_aiv0 = sum(v for k, v in quorum.items()
                    if k[0] == "WAIT" and k[1] < INTRA_NUM)
    from_aiv1 = sum(v for k, v in quorum.items()
                    if k[0] == "WAIT" and k[1] >= INTRA_NUM)
    if from_aiv0 != strips or from_aiv1 != strips:
        raise SystemExit(
            f"quorum not reached: before its {strips} GEMMs the Cube waited "
            f"{from_aiv0} staging flags from AIV0 and {from_aiv1} from AIV1, "
            f"expected {strips} of each"
        )
    print(f"  quorum: the Cube waited {strips} flag(s) from each Vector core")

    # ---- 5. instruction budget ----------------------------------------
    mmads = sum(1 for _, (_, _, op) in c_in.items() if "MMAD" in op)
    want_mmads = ROTATION_MMADS + SCORE_MMADS_PER_STRIP * strips
    if mmads != want_mmads:
        raise SystemExit(
            f"expected {want_mmads} Mmads ({ROTATION_MMADS} rotation + "
            f"{SCORE_MMADS_PER_STRIP} x {strips} score), saw {mmads}"
        )
    fixpipes = sum(1 for _, (_, _, op) in c_in.items() if "FIX_L0C_TO_DST" in op)
    want_fix = ROTATION_FIXPIPES + strips
    if fixpipes != want_fix:
        raise SystemExit(
            f"expected {want_fix} Fixpipes ({ROTATION_FIXPIPES} rotated query + "
            f"one per strip), saw {fixpipes}"
        )
    cv_moves = sum(1 for _, (_, _, op) in v1_in.items() if "MOV_UB_TO_L1" in op)
    want_cv = CV_BURSTS_PER_STRIP * strips
    if cv_moves != want_cv:
        raise SystemExit(
            f"expected {want_cv} UB->L1 bursts ({CV_BURSTS_PER_STRIP} C0 column "
            f"strips x {strips} strips), saw {cv_moves}"
        )
    print(f"  budget: {mmads} Mmads, {fixpipes} Fixpipes, {cv_moves} UB->L1 bursts")

    # ---- 6. the staging core never touches Global Memory --------------
    cv_stores = sum(1 for _, (_, _, op) in v1_in.items() if "Dst:OUT" in op)
    if cv_stores:
        raise SystemExit(
            f"the staging core wrote Global Memory {cv_stores} times: the "
            f"dequantised key page is not staying on chip"
        )
    print("  the staging core issued no Global Memory store")

    # ---- 7. one outflow, after the sequence ---------------------------
    last_fixpipe = max(
        (tick for _, (tick, _, op) in c_in.items() if "FIX_L0C_TO_DST" in op),
        default=None,
    )
    if last_fixpipe is None:
        raise SystemExit("no Fixpipe in the Cube trace")
    early = sorted(tick for _, (tick, _, op) in v0_in.items()
                   if "Dst:OUT" in op and tick < last_fixpipe)
    if early:
        raise SystemExit(
            f"{len(early)} Global Memory store(s) on the recurrence core before "
            f"the last Fixpipe at tick {last_fixpipe}: first at {early[0]}. "
            f"The outflow is not a single end-of-sequence write."
        )
    outflow = sum(1 for _, (_, _, op) in v0_in.items() if "Dst:OUT" in op)
    print(f"  outflow: {outflow} Global Memory store(s) on the recurrence core, "
          f"all after the last Fixpipe (tick {last_fixpipe})")

    # ---- 8. both Vector cores executed --------------------------------
    census = {}
    for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
        if not records:
            raise SystemExit(f"{name} executed no instructions")
        census[name] = Counter(pipe for _, (_, pipe, _) in records.items())
    if not any(p in census["AIV1"] for p in VECTOR_PIPES | {"MTE2", "MTE3"}):
        raise SystemExit("AIV1 issued no engine work: Vector core 1 is idle")
    print("  per-engine census:")
    for name in ("AIV0", "AIV1", "AIC"):
        top = ", ".join(f"{p}:{n}" for p, n in census[name].most_common(6))
        print(f"    {name:<5} {sum(census[name].values()):6d} instrs  {top}")

    # ---- cadence and overlap, reported not asserted --------------------
    print(f"\nper-strip cadence (ring depth {STAGE_DEPTH}; the first "
          f"{STAGE_DEPTH} columns are pipeline fill)")
    # The first UB->L1 burst of each strip's group of CV_BURSTS_PER_STRIP.
    cv_ticks = dispatch_ticks(v1_in, lambda p, op: "MOV_UB_TO_L1" in op)
    cadence("AIV1 UB->L1 staging bursts", cv_ticks[::CV_BURSTS_PER_STRIP], strips,
            expected=strips)
    # The score Mmads only: skip the rotation prologue's 64.
    mmad_ticks = dispatch_ticks(c_in, lambda p, op: "MMAD" in op)
    score_mmads = mmad_ticks[ROTATION_MMADS:]
    cadence("AIC score Mmad groups", score_mmads[::SCORE_MMADS_PER_STRIP], strips,
            expected=strips)
    fix_ticks = dispatch_ticks(c_in, lambda p, op: "FIX_L0C_TO_DST" in op)
    cadence("AIC Fixpipe to UB", fix_ticks[ROTATION_FIXPIPES:], strips,
            expected=strips)

    print("\nmeasured engine overlap (merged in-flight spans)")
    cube = busy(c_in, c_out, lambda p, op: "MMAD" in op)
    stage = busy(v1_in, v1_out, lambda p, op: "MOV_UB_TO_L1" in op)
    recur = busy(v0_in, v0_out, lambda p, op: p in VECTOR_PIPES)
    for name, spans in (("AIC Mmad", cube), ("AIV1 UB->L1", stage),
                        ("AIV0 Vector", recur)):
        total = sum(b - a for a, b in spans)
        print(f"  {name:<14} {len(spans):4d} span(s), {total} tick(s) in flight")
    for a_name, a, b_name, b in (("AIC Mmad", cube, "AIV1 UB->L1", stage),
                                 ("AIC Mmad", cube, "AIV0 Vector", recur),
                                 ("AIV1 UB->L1", stage, "AIV0 Vector", recur)):
        both = intersect(a, b)
        total = sum(y - x for x, y in both)
        print(f"  {a_name} & {b_name}: {len(both)} intersecting span(s), "
              f"{total} tick(s) concurrent")

    print("\nOK paged decode trace audit: 9 checks passed")


if __name__ == "__main__":
    args = sys.argv[1:]
    strips = 8
    if args and args[0] == "--strips":
        strips = int(args[1])
        args = args[2:]
    if len(args) != 6:
        raise SystemExit(__doc__)
    main(*args, strips)
