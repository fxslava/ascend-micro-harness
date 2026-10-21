#!/usr/bin/env python3
"""Verify event balance and engine overlap for the fused attention probe.

Usage:
  analyze_fused_trace.py <vec_popped> <vec_complete> <cube_popped> <cube_complete>

Checks three things and fails loudly on any of them:

  1. intra-core SET_FLAG / WAIT_FLAG balance per (pipe, trigger pipe, flag id)
     on the AIV, so no HardEvent token leaks across the launch;
  2. cross-core SET_INTRA_BLOCK / WAIT_INTRA_BLOCK pairing per sync id, where
     a set on one core must be matched by a wait on the other;
  3. genuine concurrency, by merging each engine's in-flight intervals into
     busy spans and intersecting the four engine classes.

Ticks come from the same CAModel clock in both core dumps, so AIC and AIV
intervals are directly comparable.
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
    """Merge [(begin, end)] into disjoint busy spans."""
    spans = []
    for begin, end in sorted(intervals):
        if spans and begin <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([begin, end])
    return [(a, b) for a, b in spans]


def intersect(a, b):
    """Intersect two lists of disjoint sorted spans."""
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


def main(vec_popped, vec_complete, cube_popped, cube_complete):
    vec_in, vec_out = read(vec_popped), read(vec_complete)
    cube_in, cube_out = read(cube_popped), read(cube_complete)

    # ---- 1. intra-core HardEvent balance (AIV) -------------------------
    sets, waits = Counter(), Counter()
    for _, (_, _, operation) in vec_in.items():
        flag = FLAG.search(operation)
        if not flag:
            continue
        key = (flag.group(1), flag.group(2), int(flag.group(3), 0))
        if "SET_FLAG" in operation:
            sets[key] += 1
        elif "WAIT_FLAG" in operation:
            waits[key] += 1
    if sets != waits:
        raise SystemExit(f"intra-core event leak: sets={dict(sets)} waits={dict(waits)}")
    bad = [k for k in sets if k[2] > 5]
    if bad:
        raise SystemExit(f"event IDs 6 and 7 are reserved, saw {bad}")

    # ---- 2. cross-core pairing ----------------------------------------
    cross = defaultdict(Counter)
    for core, records in (("AIV", vec_in), ("AIC", cube_in)):
        for _, (_, _, operation) in records.items():
            sync = SYNC.search(operation)
            if sync:
                kind, _, sync_id = sync.groups()
                cross[core][(kind, int(sync_id))] += 1
    for sync_id in {k[1] for c in cross.values() for k in c}:
        aiv_set = cross["AIV"][("SET", sync_id)]
        aic_wait = cross["AIC"][("WAIT", sync_id)]
        aic_set = cross["AIC"][("SET", sync_id)]
        aiv_wait = cross["AIV"][("WAIT", sync_id)]
        if aiv_set != aic_wait or aic_set != aiv_wait:
            raise SystemExit(
                f"cross-core flag {sync_id} unpaired: "
                f"AIV set {aiv_set} / AIC wait {aic_wait}, "
                f"AIC set {aic_set} / AIV wait {aiv_wait}"
            )
    cross_sets = sum(v for c in cross.values() for k, v in c.items() if k[0] == "SET")
    cross_waits = sum(v for c in cross.values() for k, v in c.items() if k[0] == "WAIT")
    if cross_sets != cross_waits:
        raise SystemExit(f"cross-core total mismatch: {cross_sets} sets, {cross_waits} waits")

    # ---- 3. engine overlap ---------------------------------------------
    classes = defaultdict(list)
    for instr_id, (begin, pipe, operation) in vec_in.items():
        if instr_id not in vec_out:
            continue
        end = vec_out[instr_id][0]
        if end <= begin:
            end = begin + 1
        if "MOV_SRC_TO_DST_ALIGNv2" in operation and pipe == "MTE2":
            classes["MTE2"].append((begin, end))
        elif "MOV_SRC_TO_DST_ALIGNv2" in operation and pipe == "MTE3":
            classes["MTE3"].append((begin, end))
        elif pipe in VECTOR_PIPES:
            classes["Vector"].append((begin, end))
    for instr_id, (begin, pipe, operation) in cube_in.items():
        if instr_id not in cube_out or pipe not in CUBE_PIPES:
            continue
        end = cube_out[instr_id][0]
        if end <= begin:
            end = begin + 1
        classes["Cube"].append((begin, end))

    missing = [name for name in ("MTE2", "MTE3", "Vector", "Cube") if not classes[name]]
    if missing:
        raise SystemExit(f"no in-flight intervals recorded for {missing}")

    spans = {name: merge(values) for name, values in classes.items()}
    busy = {name: sum(b - a for a, b in value) for name, value in spans.items()}

    print(f"intra-core HardEvents: {sum(sets.values())} sets / {sum(waits.values())} "
          f"waits across {len(sets)} (route, ID) pairs; IDs 0..5; zero residual")
    print(f"cross-core mode-4 flags: {cross_sets} sets / {cross_waits} waits, "
          f"every sync id paired AIV<->AIC")
    print("engine busy cycles: " +
          ", ".join(f"{name} {busy[name]}" for name in ("MTE2", "Cube", "Vector", "MTE3")))

    pairs = [
        ("Cube", "Vector"), ("Cube", "MTE2"), ("Cube", "MTE3"),
        ("Vector", "MTE2"), ("Vector", "MTE3"), ("MTE2", "MTE3"),
    ]
    for left, right in pairs:
        common = intersect(spans[left], spans[right])
        total = sum(b - a for a, b in common)
        widest = max(common, key=lambda s: s[1] - s[0], default=None)
        detail = f"widest {widest[0]}..{widest[1]}" if widest else "none"
        print(f"  overlap {left:<6} + {right:<6}: {total:>5} cycles ({detail})")

    four = spans["Cube"]
    for name in ("Vector", "MTE2", "MTE3"):
        four = intersect(four, spans[name])
    if four:
        widest = max(four, key=lambda s: s[1] - s[0])
        print(f"four-engine overlap: {sum(b - a for a, b in four)} cycles, "
              f"widest CAModel cycles {widest[0]}..{widest[1]}")
    else:
        three = intersect(intersect(spans["Cube"], spans["Vector"]), spans["MTE2"])
        if not three:
            raise SystemExit("no Cube+Vector+MTE2 concurrency in the trace")
        widest = max(three, key=lambda s: s[1] - s[0])
        print(f"three-engine overlap (Cube+Vector+MTE2): "
              f"{sum(b - a for a, b in three)} cycles, "
              f"widest CAModel cycles {widest[0]}..{widest[1]}")
        print("four-engine overlap: none; MTE3 never in flight with the other three")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:5])
