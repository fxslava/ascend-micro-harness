#!/usr/bin/env python3
"""Audit the MIX_AIC_1_2 fused attention probe from its CAModel traces.

Usage:
  analyze_fused_trace.py <aiv0_popped> <aiv0_complete>
                         <aiv1_popped> <aiv1_complete>
                         <aic_popped>  <aic_complete>

Six checks, each fatal:

  1. intra-core SET_FLAG / WAIT_FLAG balance per core and per
     (pipe, trigger pipe, flag id), so no HardEvent token leaks;
  2. cross-core SET_INTRA_BLOCK / WAIT_INTRA_BLOCK pairing per absolute flag
     id. An AIV reports sync_id inside its own sub-block half while the AIC
     reports the absolute id, so AIV1 ids are rebased by INTRA_NUM before
     pairing -- that rebasing is itself the evidence that the sub-block id
     convention holds;
  3. the double-handshake quorum: the Cube waits the same number of flags from
     each of the two Vector cores, and never fewer than one GEMM worth;
  4. both Vector cores executed, with a per-engine instruction census;
  5. no Global Memory store anywhere on the CV staging core, and exactly one
     UB -> L1 hardware move per key page;
  6. every Global Memory store on the recurrence core happens after the last
     Fixpipe, which is what "one outflow at the end of the sequence" means in
     the trace rather than in the source.

Then it reports cycle spans and merged engine overlap. Ticks come from one
CAModel clock in all three dumps, so the intervals are directly comparable.
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
# The Cube gates its MTE1 loads on the staging quorum and its Fixpipe on the
# score-panel back edge, so the wait pipe separates the two kinds of flag.
QUORUM_PIPE = "MTE1"

# AscendC::INTRA_NUM: a mode-4 flag id space of 16 per sub block.
INTRA_NUM = 16
TILES = 3

VECTOR_PIPES = {"RVECEX", "RVECST", "RVECLD", "RVECSU", "RVECLP", "VEC"}
CUBE_PIPES = {"CUBE", "FIXP", "MTE1", "MTE2"}


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
        raise SystemExit(
            f"{name}: intra-core event leak: sets={dict(sets)} waits={dict(waits)}"
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


def busy(records, complete, keep):
    """Merged in-flight spans for the instructions keep() selects."""
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
         aic_popped, aic_complete):
    v0_in, v0_out = read(aiv0_popped), read(aiv0_complete)
    v1_in, v1_out = read(aiv1_popped), read(aiv1_complete)
    c_in, c_out = read(aic_popped), read(aic_complete)

    # ---- 1. intra-core HardEvent balance, per core --------------------
    per_core_events = {
        "AIV0": hard_event_balance("AIV0", v0_in),
        "AIV1": hard_event_balance("AIV1", v1_in),
        "AIC": hard_event_balance("AIC", c_in),
    }

    # ---- 2. cross-core pairing on absolute flag ids -------------------
    # An AIV prints sync_id relative to its own sub-block half; the AIC prints
    # the absolute id. Rebasing AIV1 by INTRA_NUM is what makes the two agree,
    # so a pairing failure here would mean the sub-block convention is wrong.
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
    cross_sets = sum(v for k, v in (aiv + aic).items() if k[0] == "SET")
    cross_waits = sum(v for k, v in (aiv + aic).items() if k[0] == "WAIT")
    if cross_sets != cross_waits:
        raise SystemExit(
            f"cross-core total mismatch: {cross_sets} sets, {cross_waits} waits"
        )

    # ---- 3. the double-handshake quorum -------------------------------
    # Only the waits that gate the Cube MTE1 loads are staging flags; the
    # Fixpipe waits are the score-panel back edge and are counted separately.
    quorum = cross_core(c_in, 0, pipe=QUORUM_PIPE)
    from_aiv0 = sum(v for k, v in quorum.items()
                    if k[0] == "WAIT" and k[1] < INTRA_NUM)
    from_aiv1 = sum(v for k, v in quorum.items()
                    if k[0] == "WAIT" and k[1] >= INTRA_NUM)
    if from_aiv0 != TILES or from_aiv1 != TILES:
        raise SystemExit(
            f"quorum not reached: before its {TILES} GEMMs the Cube waited "
            f"{from_aiv0} staging flags from AIV0 and {from_aiv1} from AIV1, "
            f"expected {TILES} of each"
        )

    # ---- 4. both Vector cores executed --------------------------------
    census = {}
    for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
        pipes = Counter(pipe for _, (_, pipe, _) in records.items())
        census[name] = pipes
        if not records:
            raise SystemExit(f"{name} executed no instructions")
    if not any(p in census["AIV1"] for p in VECTOR_PIPES | {"MTE2", "MTE3"}):
        raise SystemExit("AIV1 issued no engine work: Vector core 1 is idle")

    # ---- 5. the CV core never touches Global Memory --------------------
    cv_moves = sum(1 for _, (_, _, op) in v1_in.items() if "MOV_UB_TO_L1" in op)
    cv_stores = sum(1 for _, (_, _, op) in v1_in.items() if "Dst:OUT" in op)
    if cv_moves != TILES:
        raise SystemExit(
            f"expected {TILES} UB->L1 hardware moves on the staging core, saw {cv_moves}"
        )
    if cv_stores:
        raise SystemExit(
            f"the staging core wrote Global Memory {cv_stores} times: the key "
            f"page is not staying on chip"
        )

    # ---- 6. one outflow, after the sequence ---------------------------
    last_fixpipe = max(
        (tick for _, (tick, _, op) in c_in.items() if "FIX_L0C_TO_DST" in op),
        default=None,
    )
    if last_fixpipe is None:
        raise SystemExit("no Fixpipe in the Cube trace")
    gm_stores = sorted(tick for _, (tick, _, op) in v0_in.items()
                       if "Dst:OUT" in op)
    if not gm_stores:
        raise SystemExit("the recurrence core never stored its result")
    early = [t for t in gm_stores if t <= last_fixpipe]
    if early:
        raise SystemExit(
            f"{len(early)} Global Memory store(s) issued before the last "
            f"Fixpipe at cycle {last_fixpipe}: that is a per-tile dump, not a "
            f"single end-of-sequence outflow"
        )

    # ---- report -------------------------------------------------------
    total_events = sum(sum(c.values()) for c in per_core_events.values())
    pairs = sum(len(c) for c in per_core_events.values())
    print(f"intra-core HardEvents: {total_events} sets / {total_events} waits "
          f"across {pairs} (route, ID) pairs on 3 cores; zero residual")
    print(f"cross-core mode-4 flags: {cross_sets} sets / {cross_waits} waits, "
          f"every absolute id paired AIV<->AIC")
    print(f"double handshake: before each of {TILES} GEMMs the Cube waited one "
          f"staging flag from each Vector core ({from_aiv0} from AIV0 ids "
          f"0..15, {from_aiv1} from AIV1 ids 16..31)")
    for name in ("AIV0", "AIV1", "AIC"):
        total = sum(census[name].values())
        detail = ", ".join(f"{p} {n}" for p, n in census[name].most_common())
        print(f"  {name}: {total} instructions ({detail})")
    print(f"CV path: {cv_moves} MOV_UB_TO_L1 hardware moves, "
          f"{cv_stores} Global Memory stores on the staging core")
    print(f"outflow: {len(gm_stores)} GM stores, first at cycle {gm_stores[0]}, "
          f"all after the last Fixpipe at cycle {last_fixpipe}")

    # ---- cycle spans and engine overlap -------------------------------
    classes = {
        "MTE2": busy(v0_in, v0_out,
                     lambda p, o: p == "MTE2" and "MOV_SRC_TO_DST" in o)
                + busy(v1_in, v1_out,
                       lambda p, o: p == "MTE2" and "MOV_SRC_TO_DST" in o),
        "Vector": busy(v0_in, v0_out, lambda p, o: p in VECTOR_PIPES)
                  + busy(v1_in, v1_out, lambda p, o: p in VECTOR_PIPES),
        "MTE3": busy(v1_in, v1_out, lambda p, o: p == "MTE3" and "MOV_" in o)
                + busy(v0_in, v0_out,
                       lambda p, o: p == "MTE3" and "MOV_" in o),
        "Cube": busy(c_in, c_out, lambda p, o: p in CUBE_PIPES),
    }
    classes = {name: merge(spans) for name, spans in classes.items()}
    missing = [name for name, spans in classes.items() if not spans]
    if missing:
        raise SystemExit(f"no in-flight intervals recorded for {missing}")

    all_in = list(v0_in.items()) + list(v1_in.items()) + list(c_in.items())
    all_out = list(v0_out.items()) + list(v1_out.items()) + list(c_out.items())
    dispatch = [tick for _, (tick, _, _) in all_in]
    complete = [tick for _, (tick, _, _) in all_out]
    print(f"dispatch span: cycles {min(dispatch)}..{max(dispatch)} "
          f"({max(dispatch) - min(dispatch)} cycles, {len(all_in)} instructions)")
    print(f"completion span: cycles {min(complete)}..{max(complete)} "
          f"({max(complete) - min(complete)} cycles, {len(all_out)} instructions)")

    occupancy = {n: sum(b - a for a, b in s) for n, s in classes.items()}
    print("engine busy cycles: " +
          ", ".join(f"{n} {occupancy[n]}" for n in ("MTE2", "Cube", "Vector", "MTE3")))

    for left, right in (("Cube", "Vector"), ("Cube", "MTE2"), ("Cube", "MTE3"),
                        ("Vector", "MTE2"), ("Vector", "MTE3"), ("MTE2", "MTE3")):
        common = intersect(classes[left], classes[right])
        widest = max(common, key=lambda s: s[1] - s[0], default=None)
        detail = f"widest {widest[0]}..{widest[1]}" if widest else "none"
        print(f"  overlap {left:<6} + {right:<6}: "
              f"{sum(b - a for a, b in common):>5} cycles ({detail})")

    four = classes["Cube"]
    for name in ("Vector", "MTE2", "MTE3"):
        four = intersect(four, classes[name])
    if four:
        widest = max(four, key=lambda s: s[1] - s[0])
        print(f"four-engine overlap: {sum(b - a for a, b in four)} cycles, "
              f"widest CAModel cycles {widest[0]}..{widest[1]}")
    else:
        three = intersect(intersect(classes["Cube"], classes["Vector"]),
                          classes["MTE2"])
        if not three:
            raise SystemExit("no Cube+Vector+MTE2 concurrency in the trace")
        widest = max(three, key=lambda s: s[1] - s[0])
        print(f"three-engine overlap (Cube+Vector+MTE2): "
              f"{sum(b - a for a, b in three)} cycles, "
              f"widest CAModel cycles {widest[0]}..{widest[1]}")
        print("four-engine overlap: none; MTE3 never in flight with the other three")

    # Cross-core concurrency is the MIX claim, so report it on its own.
    both_vectors = intersect(
        merge(busy(v0_in, v0_out, lambda p, o: True)),
        merge(busy(v1_in, v1_out, lambda p, o: True)),
    )
    if not both_vectors:
        raise SystemExit("AIV0 and AIV1 are never in flight at the same time")
    widest = max(both_vectors, key=lambda s: s[1] - s[0])
    print(f"AIV0 + AIV1 concurrent: {sum(b - a for a, b in both_vectors)} cycles, "
          f"widest CAModel cycles {widest[0]}..{widest[1]}")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    main(*sys.argv[1:7])
