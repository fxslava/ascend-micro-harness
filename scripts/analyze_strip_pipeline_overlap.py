#!/usr/bin/env python3
"""Engine occupancy and scalar-issue census for the decode strip pipeline.

Answers two questions about a CAModel run of
`turboquant_paged_decode_splitk_aic`, from the per-core completion logs:

1. **Where does the Vector core's scalar time go?** The discriminator is how
   many times each PC is hit. Code that runs once per STRIP shows up as many
   distinct PCs hit exactly `strips` times; an inner loop over the 16 query
   rows shows up as fewer PCs hit `16 * strips` times; the launch prologue
   shows up as PCs hit once. Splitting the scalar count that way says whether
   a per-strip prologue is worth hoisting or whether the cost is inside a
   shared routine, which a single total cannot.

2. **Do the Cube and the Vector cores overlap?** The span is binned and each
   engine's instructions are counted per bin, which is a text rendering of
   what a Perfetto track shows. `cube_idle_fraction` is the share of bins in
   which the Cube retired nothing at all -- the direct measure of starvation.

Both are reported as numbers rather than as a verdict: what counts as enough
overlap depends on the geometry, and at one strip per page the pipeline is
only three strips deep, so some idling is structural rather than a defect.

3. **Do the engines actually overlap, in time?** `--overlap` merges each
   engine's retirement ticks into busy INTERVALS -- two consecutive
   retirements join when the gap is at most `--gap` ticks -- and intersects
   them exactly. That replaces the binned picture with a number, at the cost
   of depending on `--gap`, which is therefore printed with every figure;
   `--gap-sweep` runs several values so a conclusion that only holds at one of
   them is visible as such.

   `VECTOR` on an AIV is the union of the RISC-V vector units and the classic
   VEC column, because the decode recurrence runs on the SIMT path: in these
   traces `VEC` holds 15 instructions against `RVECEX`'s 12,780.

usage: analyze_strip_pipeline_overlap.py EVIDENCE_DIR [--strips N] [--bins N]
                                         [--json] [--overlap] [--gap N]
                                         [--gap-sweep] [--timeline-json PATH]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TICK = re.compile(r"^\[\w+\]\s+\[(\d+)\]")
ENGINE = re.compile(r"\)\s+(\w+)\s*:")
PC = re.compile(r"PC: (0x[0-9a-f]+)")
CORE = re.compile(r"core(\d+)\.(\w+)\.instr_popped_log\.dump")

# The engines that do the operator's actual work, as the dumps name them.
WORK_ENGINES = ("CUBE", "FIXP", "MTE1", "MTE2", "MTE3", "VEC")

# Logical pipes, and the retirement engines that carry each. The AIV's Vector
# pipe spans the RISC-V vector units of the SIMT path as well as the classic
# VEC column; FLOWCTRL, SCALAR and PUSHQ are issue, not pipe work.
PIPE_GROUPS = {
    "VECTOR": ("VEC", "RVECEX", "RVECLD", "RVECST", "RVECSU", "RVECLP"),
    "MTE2": ("MTE2",),
    "MTE3": ("MTE3",),
    "MTE1": ("MTE1",),
    "CUBE": ("CUBE",),
    "FIXP": ("FIXP",),
}


def intervals_of(ticks, gap):
    """Merge sorted retirement ticks into [start, end] busy intervals.

    An instruction retiring at tick t is evidence the engine was busy AT t and
    says nothing about how long before t it had been. So an interval is a run
    of retirements no more than `gap` apart, and its length understates busy
    time by at most `gap` at the leading edge. That is why `gap` is reported
    rather than buried.
    """
    if not ticks:
        return []
    merged = []
    start = previous = ticks[0]
    for tick in ticks[1:]:
        if tick - previous <= gap:
            previous = tick
        else:
            merged.append((start, previous))
            start = previous = tick
    merged.append((start, previous))
    return merged


def covered(intervals):
    return sum(end - start for start, end in intervals)


def intersect(a, b):
    """Total length of the intersection of two interval lists."""
    total, i, j = 0, 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def pipe_intervals(ticks, block, gap):
    """-> {(core.unit, PIPE): intervals} for every logical pipe present."""
    out = {}
    for (core, unit, engine), values in ticks.items():
        for pipe, engines in PIPE_GROUPS.items():
            if engine in engines:
                key = ("core%d.%s" % (core, unit), pipe)
                out.setdefault(key, []).extend(values)
    return {key: intervals_of(sorted(values), gap) for key, values in out.items()}


def read(directory, block):
    """-> {(core, unit, engine): [ticks]}, {(core, unit): {pc: hits}}"""
    ticks = defaultdict(list)
    pcs = defaultdict(Counter)
    for path in sorted(directory.glob("core*.instr_popped_log.dump")):
        name = CORE.match(path.name)
        if not name:
            continue
        core, unit = int(name[1]), name[2]
        if block is not None and core != block:
            continue
        with path.open(errors="replace") as source:
            for raw in source:
                tick, engine = TICK.match(raw), ENGINE.search(raw)
                if not tick or not engine:
                    continue
                ticks[(core, unit, engine[1])].append(int(tick[1]))
                if engine[1] == "SCALAR":
                    found = PC.search(raw)
                    if found:
                        pcs[(core, unit)][found[1]] += 1
    return ticks, pcs


def scalar_shape(hits, strips):
    """Split a core's scalar PCs by how often each was hit.

    `once` is the launch prologue, `per_strip` is code the strip loop runs one
    time per strip, `per_row` is the 16-query-row loop inside the recurrence,
    and `other` is everything that matches no bucket.
    """
    buckets = {"once": 0, "per_strip": 0, "per_row": 0, "other": 0}
    distinct = {"once": 0, "per_strip": 0, "per_row": 0, "other": 0}
    for count in hits.values():
        if count == 1:
            key = "once"
        elif strips and count == strips:
            key = "per_strip"
        elif strips and count == 16 * strips:
            key = "per_row"
        else:
            key = "other"
        buckets[key] += count
        distinct[key] += 1
    return buckets, distinct


# The pairs worth asking about on this operator, as (core, pipe) against
# (core, pipe). The Cube against AIV0's Vector is the cross-core one: it is how
# much of the Cube's own busy time is hidden behind Vector work.
OVERLAP_PAIRS = (
    ("core0.veccore0", "MTE2", "core0.veccore0", "VECTOR"),
    ("core0.veccore0", "VECTOR", "core0.veccore0", "MTE3"),
    ("core0.veccore1", "MTE2", "core0.veccore1", "VECTOR"),
    ("core0.veccore1", "VECTOR", "core0.veccore1", "MTE3"),
    ("core0.cubecore0", "CUBE", "core0.veccore0", "VECTOR"),
    ("core0.cubecore0", "FIXP", "core0.veccore0", "VECTOR"),
    ("core0.cubecore0", "MTE1", "core0.veccore1", "MTE3"),
)


def report_overlap(ticks, args, span_start, span_end):
    gaps = [args.gap] if not args.gap_sweep else [16, 64, 256, 1024]
    payload = {"span": [span_start, span_end], "by_gap": {}}
    for gap in gaps:
        pipes = pipe_intervals(ticks, args.block, gap)
        print("=== gap %d ticks: busy time and bubbles ===" % gap)
        print("  %-26s %8s %8s %8s %9s %9s"
              % ("core.pipe", "instrs", "busy", "idle", "bubbles", "worst"))
        rows = {}
        for key, intervals in sorted(pipes.items()):
            name = "%s %s" % key
            busy = covered(intervals)
            first, last = intervals[0][0], intervals[-1][1]
            own_span = max(1, last - first)
            idle = own_span - busy
            holes = [intervals[i + 1][0] - intervals[i][1]
                     for i in range(len(intervals) - 1)]
            count = sum(1 for _ in ticks_for(ticks, key))
            print("  %-26s %8d %8d %8d %9d %9d"
                  % (name, count, busy, idle, len(holes), max(holes) if holes else 0))
            rows[name] = {"instructions": count, "busy": busy, "idle": idle,
                          "first": first, "last": last,
                          "bubbles": len(holes), "worst_bubble": max(holes) if holes else 0,
                          "intervals": intervals}
        print("  -- pairwise overlap (of each side's own busy time) --")
        pairs = {}
        for core_a, pipe_a, core_b, pipe_b in OVERLAP_PAIRS:
            a = pipes.get((core_a, pipe_a))
            b = pipes.get((core_b, pipe_b))
            if not a or not b:
                continue
            both = intersect(a, b)
            busy_a, busy_b = covered(a), covered(b)
            label = "%s %s  vs  %s %s" % (core_a, pipe_a, core_b, pipe_b)
            print("  %-54s %7d ticks  %5.1f%% of A  %5.1f%% of B"
                  % (label, both,
                     100.0 * both / max(1, busy_a), 100.0 * both / max(1, busy_b)))
            pairs[label] = {"overlap": both, "busy_a": busy_a, "busy_b": busy_b}
        payload["by_gap"][str(gap)] = {"pipes": rows, "pairs": pairs}
        print()
    if args.timeline_json:
        with open(args.timeline_json, "w") as handle:
            json.dump(payload, handle, indent=1)
        print("timeline written to %s" % args.timeline_json)
    return 0


def ticks_for(ticks, key):
    """Every retirement tick behind one (core.unit, PIPE) key."""
    name, pipe = key
    for (core, unit, engine), values in ticks.items():
        if "core%d.%s" % (core, unit) == name and engine in PIPE_GROUPS[pipe]:
            for value in values:
                yield value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--strips", type=int, default=0,
                        help="strips this launch walked; enables the scalar "
                             "shape census, which needs the trip count")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--overlap", action="store_true",
                        help="interval-based busy time, bubbles and pairwise "
                             "overlap, instead of the binned picture")
    parser.add_argument("--gap", type=int, default=64,
                        help="retirements this many ticks apart or less are one "
                             "busy interval (default 64)")
    parser.add_argument("--gap-sweep", action="store_true",
                        help="repeat the overlap census at several gap values, "
                             "so a conclusion that holds at only one is visible")
    parser.add_argument("--timeline-json", metavar="PATH",
                        help="write the busy intervals for plotting")
    args = parser.parse_args()

    ticks, pcs = read(args.directory, args.block)
    if not ticks:
        print("no usable *.instr_popped_log.dump in %s" % args.directory,
              file=sys.stderr)
        return 2

    span_end = max(max(t) for t in ticks.values())
    span_start = min(min(t) for t in ticks.values())
    span = max(1, span_end - span_start)

    if args.overlap:
        return report_overlap(ticks, args, span_start, span_end)

    totals = {}
    for (core, unit, engine), values in sorted(ticks.items()):
        totals.setdefault("core%d.%s" % (core, unit), {})[engine] = len(values)

    shapes = {}
    for (core, unit), hits in sorted(pcs.items()):
        buckets, distinct = scalar_shape(hits, args.strips)
        shapes["core%d.%s" % (core, unit)] = {
            "scalar_total": sum(hits.values()),
            "distinct_pcs": len(hits),
            "instructions": buckets,
            "distinct": distinct,
        }

    # Occupancy, binned. One row per (core, engine) that does real work.
    occupancy, chart = {}, {}
    for (core, unit, engine), values in sorted(ticks.items()):
        if engine not in WORK_ENGINES:
            continue
        bins = [0] * args.bins
        for tick in values:
            index = min(args.bins - 1, (tick - span_start) * args.bins // span)
            bins[index] += 1
        name = "core%d.%s %s" % (core, unit, engine)
        occupancy[name] = bins
        peak = max(bins) or 1
        chart[name] = "".join(" .:-=+*#@"[min(8, (count * 8 + peak - 1) // peak)]
                              for count in bins)

    cube_bins = [bins for name, bins in occupancy.items() if name.endswith("CUBE")]
    cube_idle = (sum(1 for i in range(args.bins)
                     if all(b[i] == 0 for b in cube_bins)) / args.bins
                 if cube_bins else None)

    report = {"span_ticks": span, "bins": args.bins,
              "instructions_per_engine": totals,
              "scalar_shape": shapes,
              "cube_idle_fraction": cube_idle,
              "occupancy": occupancy}

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("span %d ticks (%d .. %d), %d bins of %d ticks"
          % (span, span_start, span_end, args.bins, span // args.bins))
    for core, engines in totals.items():
        print("  %-18s %s" % (core, ", ".join(
            "%s=%d" % (engine, count) for engine, count in sorted(engines.items()))))
    if args.strips:
        print("  -- scalar instruction shape (trip count %d strips):" % args.strips)
        for core, shape in shapes.items():
            instructions, distinct = shape["instructions"], shape["distinct"]
            print("  %-18s total %6d over %4d PCs | once %5d/%3d  per-strip "
                  "%5d/%3d  per-row %6d/%3d  other %5d/%3d"
                  % (core, shape["scalar_total"], shape["distinct_pcs"],
                     instructions["once"], distinct["once"],
                     instructions["per_strip"], distinct["per_strip"],
                     instructions["per_row"], distinct["per_row"],
                     instructions["other"], distinct["other"]))
    print("  -- engine occupancy over the span, one column per bin:")
    for name, line in chart.items():
        print("  %-26s |%s|" % (name, line))
    if cube_idle is not None:
        print("  cube_idle_fraction %.3f (bins in which the Cube retired nothing)"
              % cube_idle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
