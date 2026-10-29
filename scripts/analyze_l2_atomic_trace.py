#!/usr/bin/env python3
"""Audit the Split-K synchronisation probe from its CAModel traces.

Usage:
  analyze_l2_atomic_trace.py --mode a|b|c --splits K --reps N <dump directory>

Where split k's work actually is
-------------------------------
The probe is KERNEL_TYPE_AIV_ONLY and every AIV is its own block -- but an AIV
is not an AI core. Each 950PR AI core carries TWO Vector cores, and the model
dumps them separately, so the mapping is

    split k  ->  core<k / 2>.veccore<k % 2>

and NOT `core<k>.veccore0`, which is the MIX mapping the decode operator's
audit uses. This was measured, not assumed: at K=2 the two splits' Global
Memory traffic is in `core0.veccore0` and `core0.veccore1`, and `core1` holds
nothing at all. An audit that read `core<k>.veccore0` would find an untouched
core for every odd split and report it as a core that did not run -- which is
what happened first, and is why this script now asserts the mapping with a
census over every dump instead of trusting it.

A consequence worth carrying into the numbers: at K=8 the eight splits are
four AI cores with two Vector cores each, so pairs of splits share whatever
their AI core shares on the way to L2. Any K>=2 contention figure here is
therefore a mixture of inter-core and intra-core sharing, and the report says
so rather than presenting it as eight independent cores.

Why this exists alongside the kernel's own cycle counters
---------------------------------------------------------
The kernel reads `GetSystemCycle()` around each phase and publishes the
deltas, which is the primary measurement. That is a device-reported number
about itself. This audit is the independent one: it reads the instruction
dumps, whose ticks are recorded by the model rather than by the kernel, and it
can see three things the counters cannot.

  * **Mutual exclusion, from the data instructions.** Each split's critical
    section is bracketed by its Global Memory read of the shared record and
    the retirement of its write-back. Across cores those windows must be
    DISJOINT. This is the property the lock exists to provide, measured from
    the instructions whose ticks mean what they say -- not from the lock's own
    bookkeeping. It is the trace-side counterpart of the runner's ticket-log
    permutation check, and the two fail independently: a lock that never
    excludes still produces a clean ticket log if its races happen not to
    overlap on one run, and a ticket counter that is itself broken would not
    perturb these windows.
  * **The spin census, from the atomics.** A failed test-and-set is a
    `SIMT_RED` that returned non-zero, so the atomic instruction count per
    acquisition is 1 + spins (+1 for the ticket). The kernel counts spins in a
    thread register; this counts the instructions that implement them.
  * **In-flight time per Global Memory burst.** Dispatch tick to completion
    tick: the interval during which the burst was outstanding.
  * **Issue-stall cycles, attributed to windows.** The model's per-core CCU
    log records one `[ISSUE INSTR] general hazard failed.` line per CYCLE in
    which the core could not issue, with the tick. Counting those inside the
    spin window and inside the critical-section window splits a split's lost
    cycles between waiting for the lock and waiting for its own memory.

What "stall" here does and does not mean
---------------------------------------
`general hazard failed` is an ISSUE-stage hazard: in that cycle the next
instruction could not be issued, for any reason -- an operand not ready, a
queue full, a flag not set, a memory access outstanding. It is **not** a bus
or L2 stall counter, and nothing in these dumps is: the CCU log models the
core's own issue pipeline and the dcache log is the per-core SIMT DataCache
tag array. The model emits no per-transaction L2 record, no bank-conflict
record and no queue-occupancy record, so "bus stall cycles" in the sense of
cycles lost inside the memory system cannot be measured from these runs at
all. What CAN be measured, and is reported below, is the outstanding time of
each Global Memory burst and the issue-stall cycles of each core, attributed
to the protocol's windows. The report says this plainly rather than
relabelling one as the other.

What the model does and does not show about contention
-----------------------------------------------------
Reported, not asserted: whether the in-flight time of a shared-line access
grows with K. If it does, the model is representing some form of contention on
that line and the growth is measurable. If it does not, then on THIS model the
only cost of sharing is the serialisation the lock imposes, and no claim about
bank or bus contention on silicon can be drawn from these runs. The report
states which of the two was observed; this script just prints the numbers.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)")

GM_READ = "Src:OUT, Dst:UB"
GM_WRITE = "Src:UB, Dst:OUT"
VECTOR_PIPES = {"RVECEX", "RVECST", "RVECLD", "RVECSU", "RVECLP", "VEC"}

# The probe's Global Memory instruction budget, per mode. Reads and writes are
# whole DataCopy instructions, not DataBlocks: the state pair is one 2-block
# burst and the accumulator one 32-block burst, so a record move is two
# instructions in each direction.
#
#   fixture_reads   before the measured region
#   reads_per_rep / writes_per_rep   inside it
#   trailing_writes the metric slice, once
BUDGET = {
    "a": dict(fixture_reads=2, reads_per_rep=0, writes_per_rep=2, trailing_writes=1,
              vf_per_rep=0),
    "b": dict(fixture_reads=2, reads_per_rep=2, writes_per_rep=2, trailing_writes=1,
              vf_per_rep=3),
    # Mode C's four trailing writes are the metric slice plus the three
    # divide-operand bursts it publishes for the Inf/NaN diagnosis, all of
    # them after the measured region.
    "c": dict(fixture_reads=4, reads_per_rep=0, writes_per_rep=1, trailing_writes=4,
              vf_per_rep=0),
}


def read(path):
    """{instruction id: (tick, pipe, operation)} from one dump."""
    records = {}
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
        match = LINE.search(line)
        if match:
            tick, pipe, instr_id, operation = match.groups()
            records[int(instr_id)] = (int(tick), pipe, operation)
    return records


def select(records, predicate):
    """[(id, tick, pipe, op)] in issue order for the matching instructions."""
    out = [(i, t, p, o) for i, (t, p, o) in records.items() if predicate(p, o)]
    return sorted(out, key=lambda row: row[0])


def mnemonic(operation):
    """The instruction's mnemonic, exactly.

    Substring matching does not work on this instruction set and the mistake
    is not a near miss: `SIMT_END` is a prefix of `SIMT_END_DVG`, `SIMT_ST` of
    `SIMT_START_DVG`, and a divergent `if (threadIdx.x != 0)` guard emits
    several of each. Counted by substring, a three-VF repetition reports six
    VF ends and eight stores. Every census below compares mnemonics for
    equality.
    """
    return operation.split()[0] if operation else ""


def count(records, name):
    return sum(1 for _, (_, _, o) in records.items() if mnemonic(o) == name)


def spans(dispatch, complete, predicate):
    """[(begin, end)] in-flight intervals, in issue order."""
    out = []
    for instr_id, tick, pipe, operation in select(dispatch, predicate):
        if instr_id not in complete:
            continue
        end = complete[instr_id][0]
        out.append((tick, max(end, tick + 1)))
    return out


def describe(name, values):
    if not values:
        return f"  {name:<34} none"
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    return (f"  {name:<34} n={len(values):3d}  min {min(values):6d}  "
            f"median {mid:6d}  max {max(values):6d}  total {sum(values):8d}")


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
        raise SystemExit(f"{name}: intra-core event leak on "
                         f"(pipe, trigger, id) -> (sets, waits): {leaked}")
    reserved = [k for k in sets if k[2] in (6, 7)]
    if reserved:
        raise SystemExit(f"{name}: event ids 6 and 7 are reserved, saw {reserved}")
    return sum(sets.values())


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dumps", type=Path)
    parser.add_argument("--mode", required=True, choices=("a", "b", "c"))
    parser.add_argument("--splits", required=True, type=int)
    parser.add_argument("--reps", required=True, type=int)
    args = parser.parse_args()
    budget = BUDGET[args.mode]

    # Mode C is the tail split's work alone, so only one split runs it.
    active = [args.splits - 1] if args.mode == "c" else list(range(args.splits))

    def dump_pair(split):
        """split -> (popped, completed) dump paths, via the AIV_ONLY mapping."""
        stem = f"core{split // 2}.veccore{split % 2}"
        return (args.dumps / f"{stem}.instr_popped_log.dump",
                args.dumps / f"{stem}.instr_log.dump")

    def aiv_name(split):
        return f"core{split // 2}.veccore{split % 2}"

    print(f"L2 atomic Split-K trace audit: mode {args.mode.upper()}, "
          f"K={args.splits}, {args.reps} rep(s)")
    print("  auditing split(s): " +
          ", ".join(f"{k} on {aiv_name(k)}" for k in active))

    # ---- the block -> AIV mapping, asserted over every dump -------------
    # Exactly the splits that were supposed to work may hold Global Memory
    # traffic, and they must be the ones the mapping names. A mapping change
    # otherwise shows up as a core that "did not run the probe body", which
    # points at the kernel instead of at the audit.
    carriers = []
    for path in sorted(args.dumps.glob("core*.veccore*.instr_popped_log.dump")):
        text = path.read_text(errors="replace")
        if GM_READ in text or GM_WRITE in text:
            carriers.append(path.name.split(".instr_")[0])
    expected = sorted(f"core{k // 2}.veccore{k % 2}" for k in active)
    if sorted(carriers) != expected:
        raise SystemExit(
            f"the block -> AIV mapping does not hold: Global Memory traffic is "
            f"in {sorted(carriers)}, expected exactly {expected}. Under "
            f"KERNEL_TYPE_AIV_ONLY split k runs on core<k/2>.veccore<k%2> -- "
            f"two Vector cores per AI core -- which is what this audit assumes "
            f"and what that list is checking.")
    print(f"  mapping: Global Memory traffic in exactly {len(carriers)} AIV "
          f"dump(s), as split k -> core<k/2>.veccore<k%2> predicts")

    dispatch, complete = {}, {}
    for core in active:
        popped, done = dump_pair(core)
        if not popped.exists() or not done.exists():
            raise SystemExit(f"missing dump for split {core}: {popped}")
        dispatch[core] = read(popped)
        complete[core] = read(done)

    # ---- 0. the audited cores are the ones that ran ---------------------
    # An idle core in this grid is not an empty one: it issues a scalar
    # prologue and an early return, and every balance check below is vacuously
    # true on it. So each core must first show the Global Memory traffic the
    # mode is defined by.
    print("\n0. the audited cores ran the probe body")
    for core in active:
        reads = select(dispatch[core], lambda p, o: GM_READ in o)
        writes = select(dispatch[core], lambda p, o: GM_WRITE in o)
        vector = select(dispatch[core], lambda p, o: p in VECTOR_PIPES)
        want_reads = budget["fixture_reads"] + budget["reads_per_rep"] * args.reps
        want_writes = budget["writes_per_rep"] * args.reps + budget["trailing_writes"]
        if len(reads) != want_reads or len(writes) != want_writes:
            raise SystemExit(
                f"split {core} ({aiv_name(core)}): {len(reads)} GM read(s) "
                f"and {len(writes)} GM "
                f"write(s), expected {want_reads} and {want_writes}. Either "
                f"this AIV did not run the probe body, or the burst budget "
                f"changed -- a mode's record is two instructions in each "
                f"direction, a 2-DataBlock state pair and a 32-DataBlock "
                f"accumulator.")
        if not vector:
            raise SystemExit(f"split {core} ({aiv_name(core)}) issued no "
                             f"vector instruction")
        print(f"   split {core} on {aiv_name(core)}: {len(reads)} GM read(s), "
              f"{len(writes)} GM write(s), {len(vector)} vector instruction(s)")

    # ---- 1. intra-core event balance ------------------------------------
    print("\n1. intra-core HardEvent balance")
    for core in active:
        total = hard_event_balance(f"split {core} ({aiv_name(core)})",
                                   dispatch[core])
        print(f"   split {core}: balanced on {total} set/wait pair(s), "
              f"no reserved id")

    # ---- 2. the SIMT census ---------------------------------------------
    # What the three VFs lower to, read off the trace:
    #
    #   the atomic maximum   SIMT_LDG (asc_ldcg of this split's m)
    #                        SIMT_RED (asc_atomic_max -- a REDUCTION, because
    #                                  the return value is discarded)
    #   the acquire          SIMT_ATOM per exchange attempt (asc_atomic_exch,
    #                                  whose return value decides the loop),
    #                        SIMT_ATOM for the ticket add, two SIMT_STG for
    #                        the two asc_stcg logs
    #   the release          SIMT_MEMBAR (asc_threadfence), SIMT_ATOM
    #
    # So per repetition: 3 SIMT_ATOM at the uncontended floor plus one per
    # FAILED exchange, which makes the atomic count an independent spin census
    # -- independent of the thread-register counter the kernel publishes.
    print("\n2. the SIMT census")
    want_vf = budget["vf_per_rep"] * args.reps
    atoms_by_split = {}
    for core in active:
        vf = count(dispatch[core], "VF_SIMT")
        end = count(dispatch[core], "SIMT_END")
        red = count(dispatch[core], "SIMT_RED")
        atom = count(dispatch[core], "SIMT_ATOM")
        ldg = count(dispatch[core], "SIMT_LDG")
        stg = count(dispatch[core], "SIMT_STG")
        membar = count(dispatch[core], "SIMT_MEMBAR")
        atoms_by_split[core] = atom
        if vf != want_vf:
            raise SystemExit(
                f"split {core}: {vf} SIMT VF dispatch(es), expected "
                f"{want_vf}. A VF that compiled to nothing is the first thing "
                f"to suspect: asc_vf_call expands to nothing unless "
                f"SPLIT_CORE_VEC is defined, which happens only in the AIV "
                f"leg at __NPU_ARCH__ 3510.")
        if end != vf:
            raise SystemExit(f"split {core}: {vf} VF dispatch(es) but "
                             f"{end} SIMT_END(s)")
        print(f"   split {core}: {vf} VF dispatch(es), {end} end(s), "
              f"{red} reduction(s), {atom} atomic(s), {ldg} load(s), "
              f"{stg} store(s), {membar} fence(s)")
        if args.mode == "b":
            for name, saw, want in (("SIMT_RED", red, args.reps),
                                    ("SIMT_LDG", ldg, args.reps),
                                    ("SIMT_STG", stg, 2 * args.reps),
                                    ("SIMT_MEMBAR", membar, args.reps)):
                if saw != want:
                    raise SystemExit(
                        f"split {core}: {saw} {name}, expected {want} "
                        f"({args.reps} repetition(s)). The VF bodies are "
                        f"fixed, so this is a lowering change and every "
                        f"census below is built on the old one.")
            if atom < 3 * args.reps:
                raise SystemExit(
                    f"split {core}: {atom} SIMT_ATOM, below the uncontended "
                    f"floor of {3 * args.reps} (one winning exchange, one "
                    f"ticket add and one release per repetition)")
    if args.mode == "b":
        floor = 3 * args.reps
        total = sum(atoms_by_split.values())
        failed = total - floor * len(active)
        print(f"   atomics: {total} over {len(active)} split(s); the "
              f"uncontended floor is {floor} each ({floor * len(active)} in "
              f"all), so {failed} exchange(s) failed and spun")
        print(f"   spin census from the trace: {failed} failed exchange(s) "
              f"over {args.splits * args.reps} acquisition(s), "
              f"{failed / (args.splits * args.reps):.2f} per acquisition")

    # ---- 3. Mode B: the critical sections do not overlap -----------------
    if args.mode == "b":
        print("\n3. mutual exclusion, from the data instructions")
        windows = []
        for core in active:
            reads = spans(dispatch[core], complete[core], lambda p, o: GM_READ in o)
            writes = spans(dispatch[core], complete[core], lambda p, o: GM_WRITE in o)
            # Drop the fixture reads and the trailing metric write: what is
            # left is one (fetch pair, store pair) per repetition.
            body_reads = reads[budget["fixture_reads"]:]
            body_writes = writes[:len(writes) - budget["trailing_writes"]]
            for rep in range(args.reps):
                begin = body_reads[2 * rep][0]
                end = body_writes[2 * rep + 1][1]
                windows.append((begin, end, core, rep))
        windows.sort()
        overlaps = []
        for earlier, later in zip(windows, windows[1:]):
            if later[0] < earlier[1] and later[2] != earlier[2]:
                overlaps.append((earlier, later))
        if overlaps:
            for earlier, later in overlaps[:8]:
                print(f"   OVERLAP split {earlier[2]} rep {earlier[3]} "
                      f"[{earlier[0]}, {earlier[1]}) and split {later[2]} rep "
                      f"{later[3]} [{later[0]}, {later[1]})")
            raise SystemExit(
                f"{len(overlaps)} pair(s) of critical sections from different "
                f"splits overlap in time: the lock did not exclude")
        print(f"   {len(windows)} critical section(s) across {args.splits} "
              f"split(s), none overlapping")
        lengths = [end - begin for begin, end, _, _ in windows]
        print(describe("critical section length", lengths))
        # The gaps between consecutive sections are the handover cost: one
        # core's release plus the next core's winning exchange.
        gaps = [later[0] - earlier[1] for earlier, later in zip(windows, windows[1:])
                if later[2] != earlier[2]]
        print(describe("handover gap (different splits)", gaps))
        if windows:
            total_span = windows[-1][1] - windows[0][0]
            held = sum(lengths)
            print(f"   the shared line was held {held} of {total_span} tick(s) "
                  f"({100.0 * held / total_span:.1f}%) between the first fetch "
                  f"and the last write-back")

    # ---- 4. in-flight time per Global Memory burst ----------------------
    print("\n4. Global Memory burst in-flight time (dispatch to completion)")
    for core in active:
        reads = spans(dispatch[core], complete[core], lambda p, o: GM_READ in o)
        writes = spans(dispatch[core], complete[core], lambda p, o: GM_WRITE in o)
        print(f"   split {core} on {aiv_name(core)}")
        print(describe("   GM read", [b - a for a, b in reads]))
        print(describe("   GM write", [b - a for a, b in writes]))
    if args.mode == "b":
        shared_reads, shared_writes = [], []
        for core in active:
            reads = spans(dispatch[core], complete[core], lambda p, o: GM_READ in o)
            writes = spans(dispatch[core], complete[core], lambda p, o: GM_WRITE in o)
            shared_reads += [b - a for a, b in reads[budget["fixture_reads"]:]]
            shared_writes += [b - a
                              for a, b in writes[:len(writes) -
                                                 budget["trailing_writes"]]]
        print("   the shared record only, pooled over every split")
        print(describe("   shared-line read", shared_reads))
        print(describe("   shared-line write", shared_writes))

    # ---- 5. the VF dispatch cost, per VF kind ---------------------------
    # Mode B dispatches the three VFs in a fixed order every repetition --
    # the atomic maximum, the acquire, the release -- so the dispatches split
    # into three series by position. Pooling them would hide the one that
    # actually varies with K: only the acquire spins.
    vf_windows = {}
    if budget["vf_per_rep"]:
        print("\n5. SIMT VF dispatch to end")
        kinds = (("atomic maximum", 0), ("acquire (spins)", 1), ("release", 2)) \
            if budget["vf_per_rep"] == 3 else (("VF", 0),)
        for core in active:
            vf = select(dispatch[core], lambda p, o: mnemonic(o) == "VF_SIMT")
            ends = select(dispatch[core], lambda p, o: mnemonic(o) == "SIMT_END")
            windows = []
            for (_, begin, _, _), (end_id, end_tick, _, _) in zip(vf, ends):
                finish = complete[core].get(end_id, (end_tick,))[0]
                windows.append((begin, max(finish, begin + 1)))
            vf_windows[core] = windows
            print(f"   split {core} on {aiv_name(core)}")
            for name, offset in kinds:
                series = windows[offset::budget["vf_per_rep"]]
                print(describe(f"   {name}", [b - a for a, b in series]))

    # ---- 6. issue-stall cycles, attributed to the protocol's windows -----
    # One CCU line per cycle the core could not issue. See the header for what
    # that does and does not mean: it is an issue-stage hazard of any kind,
    # not a bus stall, and these dumps carry no bus-level record at all.
    print("\n6. issue-stall cycles (CCU `general hazard failed`, one per cycle)")
    stall_line = re.compile(r"^\[info\]\s+(\d+):\s+\[ISSUE INSTR\] general hazard failed")
    for core in active:
        ccu = args.dumps / f"{aiv_name(core)}.ccu_log.dump"
        if not ccu.exists():
            print(f"   split {core}: no CCU log archived ({ccu.name}); the "
                  f"runner copies it for the probe and not for the decode "
                  f"baseline, whose CCU log is tens of megabytes")
            continue
        ticks = []
        for line in ccu.read_text(errors="replace").splitlines():
            match = stall_line.match(line)
            if match:
                ticks.append(int(match.group(1)))
        ticks.sort()

        def inside(windows):
            """Stall cycles inside the union of `windows`.

            The windows are MERGED first, and that is load-bearing rather than
            tidy: Global Memory bursts overlap each other freely -- nine
            write-backs are in flight across a Mode A run -- so summing the
            per-window counts charges a cycle once per window containing it.
            Measured before the merge: Mode A reported 4,726 cycles stalled
            with a burst outstanding against a total of 3,448 for the whole
            core, which is how the double count announced itself.
            """
            merged = []
            for begin, end in sorted(windows):
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([begin, end])
            total = 0
            for begin, end in merged:
                total += sum(1 for t in ticks if begin <= t < end)
            return total

        print(f"   split {core}: {len(ticks)} stall cycle(s) in all")
        if args.mode == "b" and core in vf_windows:
            spin = vf_windows[core][1::3]
            release = vf_windows[core][2::3]
            maximum = vf_windows[core][0::3]
            print(f"     in the acquire VF (spinning)      {inside(spin):7d}")
            print(f"     in the atomic-maximum VF          {inside(maximum):7d}")
            print(f"     in the release VF                 {inside(release):7d}")
        gm = spans(dispatch[core], complete[core],
                   lambda p, o: GM_READ in o or GM_WRITE in o)
        print(f"     while a GM burst was outstanding  {inside(gm):7d}")

    # ---- 7. the per-repetition interval, from the trace ------------------
    print("\n7. the per-repetition interval, from the trace")
    for core in active:
        writes = spans(dispatch[core], complete[core], lambda p, o: GM_WRITE in o)
        body = writes[:len(writes) - budget["trailing_writes"]]
        per_rep = budget["writes_per_rep"]
        marks = [body[rep * per_rep][0] for rep in range(args.reps)
                 if rep * per_rep < len(body)]
        deltas = [b - a for a, b in zip(marks, marks[1:])]
        print(f"   split {core}: write-back dispatch ticks {marks}")
        if deltas:
            print(f"            intervals {deltas}")

    print("\nOK L2 atomic Split-K trace audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
