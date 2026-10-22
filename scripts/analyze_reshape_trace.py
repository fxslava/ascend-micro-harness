#!/usr/bin/env python3
"""Audit the MIX_AIC_1_2 reshape-and-cache probe from its CAModel traces.

Usage:
  analyze_reshape_trace.py <aiv0_popped> <aiv0_complete>
                           <aiv1_popped> <aiv1_complete>
                           <aic_popped>  <aic_complete>

The fused attention audit next door checks a three-stage ring whose Vector
cores have different roles (recurrence and CV staging). This kernel has a
four-stage ring whose two Vector cores have the *same* role on alternating
chunks and whose Cube page staging never touches the Unified Buffer, so the
flag graph and the engine census are different and reusing that script would
either pass vacuously or fail for the wrong reason.

Six checks, each fatal:

  1. intra-core SET_FLAG / WAIT_FLAG balance per core and per
     (pipe, trigger pipe, flag id), so no HardEvent token leaks, and event ids
     6 and 7 left alone;
  2. cross-core SET_INTRA_BLOCK / WAIT_INTRA_BLOCK pairing per absolute flag
     id. An AIV reports sync_id inside its own sub-block half while the AIC
     reports the absolute id, so AIV1 ids are rebased by INTRA_NUM before
     pairing -- that rebasing is itself the evidence that the sub-block id
     convention holds;
  3. chunk ownership: the Cube announced CHUNKS panels, half to each Vector
     core, and waited one panel-free flag per chunk from each half plus the two
     priming tokens, so neither Vector core was outrun and neither was skipped;
  4. both Vector cores executed, with a per-engine instruction census, and each
     one egressed its own half of the chunks;
  5. the Cube staged every page from Global Memory into L1 without a Unified
     Buffer hop: zero Global Memory stores anywhere on the AIC and at least one
     MTE2 move per page plus the rotation table;
  6. four-engine concurrency: MTE2, Cube, Vector and MTE3 in flight at the same
     time, which is what the four-stage schedule claims and what a serialised
     implementation would fail.

Then it reports per-chunk cadence, cycle spans and merged engine overlap.
Ticks come from one CAModel clock in all three dumps, so the intervals are
directly comparable.
"""

import re
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)")
SYNC = re.compile(r"(SET|WAIT)_INTRA_BLOCK\s+PIPE:(\w+).*?sync_id:(\d+)")

# AscendC::INTRA_NUM: a mode-4 flag id space of 16 per sub block.
INTRA_NUM = 16

# turboquant_a8 geometry.
CHUNKS = 8          # four value heads then four key heads
GROUPS = 8          # Mmads per chunk, one per 32-channel group
LAND_DEPTH = 2      # landing panels per Vector core
STAGE_DEPTH = 4     # L1 page ring depth, so four columns of pipeline fill
PER_CORE = CHUNKS // 2

# Flag ids, from the kernel: product-ready is Owner*16 + slot and panel-free is
# Owner*16 + 4 + slot.
PRODUCT_READY = {0, 1, INTRA_NUM, INTRA_NUM + 1}
PANEL_FREE = {4, 5, INTRA_NUM + 4, INTRA_NUM + 5}

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


def cross_core(records, rebase):
    """{(kind, absolute flag id): count} for one core."""
    out = Counter()
    for _, (_, _, operation) in records.items():
        sync = SYNC.search(operation)
        if sync:
            kind, _, sync_id = sync.groups()
            out[(kind, int(sync_id) + rebase)] += 1
    return out


def dispatch_ticks(records, keep):
    return sorted(tick for _, (tick, pipe, operation) in records.items()
                  if keep(pipe, operation))


def cadence(name, ticks, expected=None, fill=STAGE_DEPTH):
    if expected is not None and len(ticks) != expected:
        raise SystemExit(
            f"{name}: expected {expected} per-chunk events, saw {len(ticks)}"
        )
    if len(ticks) < 2:
        print(f"  {name:<28}: {len(ticks)} event(s), no interval")
        return []
    deltas = [b - a for a, b in zip(ticks, ticks[1:])]
    steady = deltas[fill - 1:] if len(deltas) > fill else []
    print(f"  {name:<28}: {len(ticks)} events, deltas "
          + " ".join(str(d) for d in deltas))
    if steady:
        mid = sorted(steady)[len(steady) // 2]
        print(f"  {'':<28}  steady state ({len(steady)} intervals): "
              f"min {min(steady)}, median {mid}, max {max(steady)}")
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
    unknown = sorted(ids - PRODUCT_READY - PANEL_FREE)
    if unknown:
        raise SystemExit(f"unexpected mode-4 flag ids in the trace: {unknown}")

    # ---- 3. chunk ownership, both ways --------------------------------
    # The Cube announces one panel per chunk and the owning Vector core hands
    # that panel back, so each half of the id space must carry exactly
    # PER_CORE product-ready flags and LAND_DEPTH + PER_CORE panel-free flags
    # (the extra two are the priming tokens, drained by the Cube epilogue).
    announced = {
        0: sum(v for k, v in aic.items()
               if k[0] == "SET" and k[1] in PRODUCT_READY and k[1] < INTRA_NUM),
        1: sum(v for k, v in aic.items()
               if k[0] == "SET" and k[1] in PRODUCT_READY and k[1] >= INTRA_NUM),
    }
    released = {
        0: sum(v for k, v in aiv.items()
               if k[0] == "SET" and k[1] in PANEL_FREE and k[1] < INTRA_NUM),
        1: sum(v for k, v in aiv.items()
               if k[0] == "SET" and k[1] in PANEL_FREE and k[1] >= INTRA_NUM),
    }
    for core in (0, 1):
        if announced[core] != PER_CORE:
            raise SystemExit(
                f"the Cube announced {announced[core]} panels to AIV{core}, "
                f"expected {PER_CORE}: chunk ownership is not half and half"
            )
        if released[core] != LAND_DEPTH + PER_CORE:
            raise SystemExit(
                f"AIV{core} released {released[core]} panels, expected "
                f"{LAND_DEPTH} primed + {PER_CORE} per chunk"
            )

    # ---- 4. both Vector cores executed, and both egressed -------------
    census = {}
    for name, records in (("AIV0", v0_in), ("AIV1", v1_in), ("AIC", c_in)):
        if not records:
            raise SystemExit(f"{name} executed no instructions")
        census[name] = Counter(pipe for _, (_, pipe, _) in records.items())
    stores = {
        "AIV0": sum(1 for _, (_, p, o) in v0_in.items()
                    if p == "MTE3" and "Dst:OUT" in o),
        "AIV1": sum(1 for _, (_, p, o) in v1_in.items()
                    if p == "MTE3" and "Dst:OUT" in o),
    }
    for name in ("AIV0", "AIV1"):
        if stores[name] < 2 * PER_CORE:
            raise SystemExit(
                f"{name} issued {stores[name]} Global Memory stores, expected "
                f"at least {2 * PER_CORE}: one packed page and one scale slice "
                f"per owned chunk"
            )

    # ---- 5. the Cube staged every page without a UB hop ---------------
    aic_stores = sum(1 for _, (_, _, o) in c_in.items() if "Dst:OUT" in o)
    if aic_stores:
        raise SystemExit(
            f"the Cube wrote Global Memory {aic_stores} times: the page "
            f"staging is not a pure GM -> L1 path"
        )
    # MOV_SPR_XN on the MTE2 pipe is a special-register write that configures
    # the next transfer, not a transfer, so counting every "MOV_" overstates
    # the data movement. The page copies are the ND -> NZ converter moves, and
    # there must be exactly one per chunk: the whole point of the layout is
    # that a 16 x 256 INT8 page is one burst, not a strip schedule.
    aic_pages = sum(1 for _, (_, p, o) in c_in.items()
                    if p == "MTE2" and "MOV_OUT_TO_L1" in o)
    aic_mte2_data = sum(1 for _, (_, p, o) in c_in.items()
                        if p == "MTE2" and "MOV_" in o and "MOV_SPR" not in o)
    if aic_pages != CHUNKS:
        raise SystemExit(
            f"the Cube issued {aic_pages} GM -> L1 page moves, expected exactly "
            f"{CHUNKS}, one ND -> NZ burst per chunk"
        )
    if aic_mte2_data != CHUNKS + 1:
        raise SystemExit(
            f"the Cube issued {aic_mte2_data} MTE2 data moves, expected "
            f"{CHUNKS + 1}: {CHUNKS} pages and the rotation table"
        )
    ub_moves = sum(1 for _, (_, _, o) in c_in.items() if "MOV_UB_TO_L1" in o)
    if ub_moves:
        raise SystemExit(
            f"{ub_moves} UB -> L1 moves on the Cube: this kernel stages pages "
            f"straight from Global Memory"
        )

    # ---- report -------------------------------------------------------
    total_events = sum(sum(c.values()) for c in per_core_events.values())
    pairs = sum(len(c) for c in per_core_events.values())
    print(f"intra-core HardEvents: {total_events} sets / {total_events} waits "
          f"across {pairs} (route, ID) pairs on 3 cores; zero residual")
    print(f"cross-core mode-4 flags: {cross_sets} sets / {cross_waits} waits, "
          f"every absolute id paired AIV<->AIC, ids "
          f"{sorted(ids)}")
    print(f"chunk ownership: the Cube announced {announced[0]} panels to AIV0 "
          f"(ids 0..15) and {announced[1]} to AIV1 (ids 16..31); the cores "
          f"released {released[0]} and {released[1]} panels "
          f"({LAND_DEPTH} primed each)")
    for name in ("AIV0", "AIV1", "AIC"):
        total = sum(census[name].values())
        detail = ", ".join(f"{p} {n}" for p, n in census[name].most_common())
        print(f"  {name}: {total} instructions ({detail})")
    print(f"egress: AIV0 {stores['AIV0']} GM stores, AIV1 {stores['AIV1']} GM "
          f"stores, AIC {aic_stores}")
    print(f"page staging: {aic_pages} GM->L1 ND->NZ page moves and "
          f"{aic_mte2_data} MTE2 data moves in all on the Cube, {ub_moves} "
          f"UB->L1 moves, so every page reached L1 from Global Memory in one "
          f"burst with no Unified Buffer hop")

    # ---- per-chunk cadence --------------------------------------------
    cube_ops = Counter(op.split()[0] for _, (_, pipe, op) in c_in.items()
                       if pipe == "CUBE" and "_FLAG" not in op)
    mmad_ops = [op for op, n in cube_ops.items() if n == CHUNKS * GROUPS]
    if not mmad_ops:
        mmad_ops = [op for op, _ in cube_ops.most_common(1)]
    mmad = dispatch_ticks(
        c_in, lambda p, o: p == "CUBE" and o.split()[0] in mmad_ops)
    fixpipe = dispatch_ticks(c_in, lambda p, o: "FIX_L0C_TO_DST" in o)
    print("per-chunk cadence. CUBE-pipe mnemonics: " +
          ", ".join(f"{op} x{n}" for op, n in cube_ops.most_common()))
    streams = [
        ("Mmad dispatch (per chunk)", mmad[::GROUPS] if mmad else [], None),
        ("Fixpipe dispatch (per chunk)", fixpipe[::GROUPS] if fixpipe else [], None),
        ("AIV0 egress", dispatch_ticks(
            v0_in, lambda p, o: p == "MTE3" and "Dst:OUT" in o)[::2], None),
        ("AIV1 egress", dispatch_ticks(
            v1_in, lambda p, o: p == "MTE3" and "Dst:OUT" in o)[::2], None),
    ]
    paced = []
    for name, ticks, expected in streams:
        fill = STAGE_DEPTH if "chunk" in name else LAND_DEPTH
        deltas = cadence(name, ticks, expected=expected, fill=fill)
        if deltas:
            paced.append((name, sorted(deltas)[len(deltas) // 2]))
    if paced:
        slowest = max(paced, key=lambda e: e[1])
        print(f"  steady-state interval: {slowest[1]} cycles per chunk, the "
              f"slowest of the measured streams ({slowest[0]})")

    # ---- cycle spans and engine overlap -------------------------------
    classes = {
        "MTE2": merge(busy(c_in, c_out, lambda p, o: p == "MTE2" and "MOV_" in o)),
        "Cube": merge(busy(c_in, c_out,
                           lambda p, o: p in ("CUBE", "FIXP", "MTE1"))),
        "Vector": merge(busy(v0_in, v0_out, lambda p, o: p in VECTOR_PIPES)
                        + busy(v1_in, v1_out, lambda p, o: p in VECTOR_PIPES)),
        "MTE3": merge(busy(v0_in, v0_out, lambda p, o: p == "MTE3" and "MOV_" in o)
                      + busy(v1_in, v1_out,
                             lambda p, o: p == "MTE3" and "MOV_" in o)),
    }
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

    # ---- 6. four-engine concurrency -----------------------------------
    four = classes["Cube"]
    for name in ("Vector", "MTE2", "MTE3"):
        four = intersect(four, classes[name])
    if not four:
        raise SystemExit(
            "no four-engine concurrency: MTE2, Cube, Vector and MTE3 are never "
            "in flight together, so the four-stage schedule is serialised"
        )
    widest = max(four, key=lambda s: s[1] - s[0])
    print(f"four-engine overlap: {sum(b - a for a, b in four)} cycles, "
          f"widest CAModel cycles {widest[0]}..{widest[1]}")

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
    main(*sys.argv[1:])
