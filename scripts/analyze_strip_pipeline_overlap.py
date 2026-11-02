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

usage: analyze_strip_pipeline_overlap.py EVIDENCE_DIR [--strips N] [--bins N]
                                         [--json]
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
    args = parser.parse_args()

    ticks, pcs = read(args.directory, args.block)
    if not ticks:
        print("no usable *.instr_popped_log.dump in %s" % args.directory,
              file=sys.stderr)
        return 2

    span_end = max(max(t) for t in ticks.values())
    span_start = min(min(t) for t in ticks.values())
    span = max(1, span_end - span_start)

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
