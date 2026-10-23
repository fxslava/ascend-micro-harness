#!/usr/bin/env python3
"""Trace audit for the AIV inverse-CDF int4x2 quantization CAModel runs.

Reads the per-core dumps of the run directory (core*.veccore*.instr_popped_log.dump
= dispatch ticks, core*.veccore*.instr_log.dump = completion ticks) and reports:

  * per-engine instruction census,
  * the vector mnemonic histogram (the selected API chain, spelled out),
  * per-mnemonic cycle cost (completion - dispatch, joined by instruction ID),
  * vector-pipe occupancy over the inner-loop window: the union of VEC
    instruction in-flight intervals over the window span -- the measurable
    form of "vector ALU utilization", reported, not assumed,
  * per-tile steady-state cadence from the MTE3 outflow instructions,
  * SET_FLAG/WAIT_FLAG balance per (pipe, trigger pipe, flag id) -- exits
    non-zero on any imbalance or residual.

Exit code 0 means every structural check passed.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\]\s+\(PC:\s*\w+\)\s+(\w+)\s*:\s*\(Binary:\s*\w+\)\s+\(ID:\s*(\d+)\)\s+(\S+)")
FLAG = re.compile(
    r"(SET_FLAG|WAIT_FLAG)\s+PIPE:(\w+),\s*TRIGGER PIPE:(\w+),\s*FLAG ID:?\s*(\w+)", re.I)


def parse(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        tick, engine, ident, mnemonic = m.groups()
        fm = FLAG.search(line)
        event = None
        if fm:
            kind, pipe, trig, fid = fm.groups()
            event = (kind.upper(), pipe.upper(), trig.upper(),
                     int(fid, 0) if not fid.startswith("0x") else int(fid, 16))
        rows.append((int(tick), engine.upper(), int(ident), mnemonic.upper(), event))
    return rows


def union_len(intervals):
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, required=True)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    cores = defaultdict(lambda: {"popped": None, "done": None})
    for popped in sorted(args.run_dir.glob("core*.veccore*.instr_popped_log.dump")):
        stem = popped.name.replace(".instr_popped_log.dump", "")
        done = popped.parent / (stem + ".instr_log.dump")
        cores[stem]["popped"] = popped
        cores[stem]["done"] = done

    failures = []
    for stem in sorted(cores):
        disp = parse(cores[stem]["popped"])
        comp = parse(cores[stem]["done"]) if cores[stem]["done"] else []
        if not disp:
            continue
        print(f"== {stem}: {len(disp)} dispatched, {len(comp)} completed ==")
        census = Counter(r[1] for r in disp)
        for engine in sorted(census):
            print(f"  {engine:8s} {census[engine]:6d}")

        # Event balance per (kind, pipe, trigger, id) across both dumps.
        ev = Counter()
        for rows in (disp, comp):
            for _, _, _, _, event in rows:
                if event:
                    ev[event] += 1
        imbalance = 0
        for (kind, pipe, trig, fid) in sorted(ev, key=lambda k: (k[1], k[2], k[3], k[0])):
            key_set = ("SET_FLAG", pipe, trig, fid)
            key_wait = ("WAIT_FLAG", pipe, trig, fid)
            s, w = ev.get(key_set, 0), ev.get(key_wait, 0)
            if s != w:
                imbalance += 1
                print(f"  IMBALANCE {pipe}->{trig} id {fid}: {s} sets vs {w} waits")
        if imbalance:
            failures.append(f"{stem}: {imbalance} unbalanced event routes")
        else:
            routes = {(e[1], e[2], e[3]) for e in ev}
            print(f"  SET/WAIT_FLAG balanced on {len(routes)} (pipe,trigger,id) routes")

        # VEC compute analysis on the dispatch stream, joined by ID for cost.
        comp_by_id = {r[2]: r[0] for r in comp}
        vec = [r for r in disp if r[1] == "VEC" and r[4] is None]
        if not vec:
            continue
        hist = Counter(r[3] for r in vec)
        print("  VEC mnemonic census (inner chain):")
        costs = defaultdict(list)
        for tick, _, ident, mnemonic, _ in vec:
            if ident in comp_by_id:
                costs[mnemonic].append(comp_by_id[ident] - tick)
        for mnemonic, count in sorted(hist.items(), key=lambda kv: -kv[1]):
            c = costs.get(mnemonic, [])
            if c:
                c.sort()
                med = c[len(c) // 2]
                print(f"    {mnemonic:14s} x{count:5d}  cost med {med:4d}  "
                      f"min {c[0]:4d}  max {c[-1]:4d} cycles")
            else:
                print(f"    {mnemonic:14s} x{count:5d}  (no completion join)")

        # Vector occupancy over the inner-loop window.
        intervals = []
        for tick, _, ident, _, _ in vec:
            if ident in comp_by_id:
                intervals.append((tick, comp_by_id[ident]))
        if intervals:
            start = min(s for s, _ in intervals)
            end = max(e for _, e in intervals)
            span = end - start
            busy = union_len(intervals)
            pct = 100.0 * busy / span if span else 100.0
            print(f"  inner-loop window: dispatch {start}..{end} "
                  f"({span} cycles, {len(intervals)} VEC instructions)")
            print(f"  vector pipe occupancy over window: {busy}/{span} = {pct:.1f}%")

        # Per-tile cadence from MTE3 outflow (one DataCopy per tile).
        mte3 = [r for r in disp if r[1] == "MTE3" and r[4] is None]
        if len(mte3) >= 2:
            ticks = sorted(r[0] for r in mte3)
            gaps = [b - a for a, b in zip(ticks, ticks[1:])]
            gaps.sort()
            print(f"  MTE3 outflow: {len(mte3)} stores, steady-state interval "
                  f"median {gaps[len(gaps)//2]} min {gaps[0]} max {gaps[-1]} cycles")
            if len(mte3) != args.tiles:
                failures.append(
                    f"{stem}: expected {args.tiles} MTE3 stores, saw {len(mte3)}")

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    print("icdf trace audit: all structural checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
