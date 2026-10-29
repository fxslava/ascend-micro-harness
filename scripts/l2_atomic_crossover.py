#!/usr/bin/env python3
"""The Split-K crossover: where L2 synchronisation stops mattering.

Usage:
  l2_atomic_crossover.py <build directory>
  l2_atomic_crossover.py --strip-interval N --strip-span N --sync K=C [K=C ...]

With a build directory it reads the probe logs and the Cube baseline audits
that `scripts/run_l2_atomic_probe_camodel.sh` leaves there. The explicit form
exists so the table can be recomputed from numbers in a report without
re-running anything, and so a reader can substitute a different denominator
and see what changes.

The arithmetic
--------------
    T_compute(T_tile) = (T_tile / 16) . T_compute_strip
    overhead %        = T_sync(K) / T_compute(T_tile) . 100

with T_tile in tokens per split and 16 tokens per strip. The crossover tile
size is the smallest T_tile whose overhead is below the target (3% by
default), which inverts to

    T_tile* = 16 . T_sync(K) / (target . T_compute_strip)

rounded up to a whole strip, because a split cannot process a fraction of
one.

Which denominator, and why it is not a free choice
--------------------------------------------------
`T_compute_strip` has two measurements and they are not interchangeable:

  the INTERVAL  between consecutive score Mmad groups in the eight-strip
                steady state. A rate: a split retires T_tile/16 strips at
                this cadence. This is the denominator, because the question
                is what fraction of a split's RUNNING TIME the sync costs.
  the SPAN      one isolated strip from its first Mmad to its Fixpipe
                retiring, with nothing overlapping. This is the Cube's OWN
                arithmetic and nothing else, and on this operator it is two
                orders of magnitude smaller than the interval -- 180 ticks
                against 19,905 -- because the interval is set by the Vector
                recurrence that the Cube waits on.

Both are printed. The crossover is computed from the interval, because the
question is what fraction of a split's running time the sync costs. The
span column answers a different and bleaker question -- what the sync costs
measured against Cube arithmetic alone -- and its answer is that no tile size
in the sweep hides it: a K=8 reduction is worth about 150 strips of Cube work.
The only reason the interval column looks tolerable is that this pipeline is
recurrence-bound, and that is a property of the operator rather than a
property of the reduction.

Which T_sync, likewise
----------------------
  per-split    what ONE split waits: its atomic maximum, its spin, its
               critical section and its release. Includes time spent waiting
               for other splits, so it already carries the contention. The
               worst split governs the makespan and is the primary figure;
               the mean is printed beside it.
  serialised   K . (critical section + handover gap), from the trace. The
               added makespan if every split's compute overlapped perfectly
               and only the reduction serialised. This is the floor a better
               protocol would have to beat, and it is NOT what the per-split
               figure measures.
"""

import argparse
import math
import re
import sys
from pathlib import Path

TILES = (16, 32, 64, 128, 256, 512, 1024)
TARGET = 0.03

# "    split 3: region   16674  atomic max   2610  acquire   9605  critical
#   3049  release   1246"
SPLIT_LINE = re.compile(
    r"split\s+(\d+):\s+region\s+(-?\d+)"
    r"(?:\s+atomic max\s+(-?\d+))?"
    r"(?:\s+acquire\s+(-?\d+))?"
    r"(?:\s+critical\s+(-?\d+))?"
    r"(?:\s+release\s+(-?\d+))?")
REPS_LINE = re.compile(r"K=(\d+) split\(s\), (\d+) rep\(s\)")
INTERVAL_LINE = re.compile(r"T_compute_strip \(steady-state interval\)\s+(\d+)")
SPAN_LINE = re.compile(r"T_compute_strip \(span, one isolated strip\)\s+(\d+)")
ROTATION_LINE = re.compile(r"rotation prologue, once per split\s+(\d+)")
CS_LINE = re.compile(r"critical section length\s+n=\s*(\d+).*?median\s+(\d+)")
GAP_LINE = re.compile(r"handover gap \(different splits\)\s+n=\s*(\d+).*?median\s+(\d+)")


def parse_probe_log(path):
    """{'splits', 'reps', 'per_split': [sync cycles per repetition]}"""
    text = path.read_text(errors="replace")
    header = REPS_LINE.search(text)
    if not header:
        return None
    splits, reps = int(header.group(1)), int(header.group(2))
    per_split = []
    for match in SPLIT_LINE.finditer(text):
        phases = [int(g) for g in match.groups()[2:] if g is not None]
        if len(phases) != 4:
            continue
        per_split.append(sum(phases) / reps)
    if not per_split:
        return None
    return dict(splits=splits, reps=reps, per_split=per_split)


def parse_audit(path):
    text = path.read_text(errors="replace")
    out = {}
    cs = CS_LINE.search(text)
    gap = GAP_LINE.search(text)
    if cs:
        out["critical_median"] = int(cs.group(2))
    if gap:
        out["handover_median"] = int(gap.group(2))
    return out


def crossover_tile(sync, strip, target=TARGET):
    """Smallest whole-strip tile whose overhead is below `target`."""
    if strip <= 0:
        return None
    tokens = 16.0 * sync / (target * strip)
    return int(math.ceil(tokens / 16.0) * 16)


def report(strip_interval, strip_span, rotation, sync_by_k, serial_by_k):
    print(f"T_compute_strip, steady-state interval : {strip_interval} tick(s)"
          if strip_interval else
          "T_compute_strip, steady-state interval : UNKNOWN")
    print(f"T_compute_strip, isolated span         : {strip_span} tick(s)"
          if strip_span else "T_compute_strip, isolated span         : UNKNOWN")
    if rotation:
        print(f"rotation prologue, once per split      : {rotation} tick(s)")
    print(f"overhead target                        : {TARGET * 100:.0f}%")

    if not strip_interval:
        print("\nno steady-state interval: run the baseline with --strips 8 "
              "before computing a crossover")
        return 1

    print("\nT_sync(K), cycles per split per reduction (system counter, "
          "pipeline drained)")
    print(f"  {'K':>3}  {'worst split':>12}  {'mean':>9}  "
          f"{'serialised K.(cs+gap)':>22}")
    for k in sorted(sync_by_k):
        values = sync_by_k[k]
        serial = serial_by_k.get(k)
        print(f"  {k:>3}  {max(values):>12.0f}  {sum(values) / len(values):>9.0f}  "
              f"{(str(serial) if serial else 'n/a'):>22}")

    for k in sorted(sync_by_k):
        sync = max(sync_by_k[k])
        print(f"\nK = {k}: T_sync = {sync:.0f} tick(s) (worst split)")
        print(f"  {'T_tile':>7}  {'strips':>6}  {'T_compute':>10}  "
              f"{'overhead %':>10}  {'(by span)':>10}")
        for tile in TILES:
            strips = tile / 16.0
            compute = strips * strip_interval
            overhead = 100.0 * sync / compute
            mark = "  <- below target" if overhead < TARGET * 100 else ""
            by_span = (100.0 * sync / (strips * strip_span)) if strip_span else float("nan")
            print(f"  {tile:>7}  {strips:>6.0f}  {compute:>10.0f}  "
                  f"{overhead:>10.2f}  {by_span:>10.2f}{mark}")
        tile_star = crossover_tile(sync, strip_interval)
        print(f"  crossover: T_tile >= {tile_star} tokens "
              f"({tile_star // 16} strip(s)) puts the sync below "
              f"{TARGET * 100:.0f}%")
        if strip_span:
            print(f"             against Cube arithmetic alone ({strip_span} "
                  f"tick(s)/strip) it would take {crossover_tile(sync, strip_span)} "
                  f"tokens, i.e. the sync is worth about "
                  f"{sync / strip_span:.0f} strips of Cube work")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build", nargs="?", type=Path)
    parser.add_argument("--strip-interval", type=int)
    parser.add_argument("--strip-span", type=int)
    parser.add_argument("--rotation", type=int)
    parser.add_argument("--sync", action="append", default=[],
                        metavar="K=CYCLES",
                        help="an explicit T_sync for split count K; repeatable")
    args = parser.parse_args()

    sync_by_k, serial_by_k = {}, {}
    strip_interval = args.strip_interval
    strip_span = args.strip_span
    rotation = args.rotation

    for item in args.sync:
        k, _, cycles = item.partition("=")
        sync_by_k[int(k)] = [float(cycles)]

    if args.build:
        build = args.build
        for log in sorted(build.glob("camodel_l2atomic_mode_b_K*.log")):
            parsed = parse_probe_log(log)
            if not parsed:
                print(f"  (no per-split metrics in {log.name})", file=sys.stderr)
                continue
            sync_by_k[parsed["splits"]] = parsed["per_split"]
            audit = build / f"audit_l2atomic_{log.stem.split('camodel_l2atomic_')[1]}.txt"
            if audit.exists():
                numbers = parse_audit(audit)
                if "critical_median" in numbers and "handover_median" in numbers:
                    serial_by_k[parsed["splits"]] = parsed["splits"] * (
                        numbers["critical_median"] + numbers["handover_median"])
        for audit in sorted(build.glob("audit_cube_baseline_*.txt")):
            text = audit.read_text(errors="replace")
            interval = INTERVAL_LINE.search(text)
            span = SPAN_LINE.search(text)
            rot = ROTATION_LINE.search(text)
            if interval and strip_interval is None:
                strip_interval = int(interval.group(1))
            if span and strip_span is None:
                strip_span = int(span.group(1))
            if rot and rotation is None:
                rotation = int(rot.group(1))

    if not sync_by_k:
        raise SystemExit(
            "no T_sync figures: pass a build directory holding "
            "camodel_l2atomic_mode_b_K*.log, or give --sync K=CYCLES")

    print("Split-K synchronisation crossover, Ascend950PR_9599 CAModel")
    print("all figures are cycle-accurate model ticks; this host has no NPU, "
          "so none of this is a silicon measurement")
    print()
    return report(strip_interval, strip_span, rotation, sync_by_k, serial_by_k)


if __name__ == "__main__":
    sys.exit(main())
