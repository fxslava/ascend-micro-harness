#!/usr/bin/env python3
"""Extract T_compute_strip from the Cube dump of a single-tile decode run.

Usage:
  analyze_cube_tile_baseline.py --strips N [--core N] <dump directory>

Reads `core<N>.cubecore0.instr_{popped_,}log.dump` and, when it is there,
`core<N>.veccore0.*` for the recurrence context.

The two numbers, and why both exist
-----------------------------------
A pipelined operator has no single "cost of a strip". It has

  the SPAN      the time from a strip's first score Mmad to its Fixpipe
                retiring, with nothing overlapping it. This is a latency, it
                is what a single-strip launch measures, and it is the right
                term for a split's cold start -- the first strip of a split
                cannot overlap with a predecessor it does not have.

  the INTERVAL  the time between consecutive score Mmad groups in the steady
                state. This is a rate, it is what an eight-strip launch
                measures, and it is the right denominator for the Split-K
                overhead ratio: a split of T_tile tokens retires
                T_tile / 16 strips at this interval.

They differ by whatever the schedule overlaps, which on this operator is most
of it -- `reports/paged_decode_attention.md` section 6 measures the Vector
recurrence in flight two orders of magnitude longer than the Cube. This script
reports both and labels them, and the crossover report uses the interval as
the denominator and the span as the cold-start term. Reporting one number
called "T_compute_strip" without saying which it is would be the only way to
get the crossover wrong by a factor of several.

What is asserted and what is reported
-------------------------------------
Asserted: the audited core ran the Cube body (64 rotation Mmads are a
constant of the operator), the Mmad and Fixpipe counts match the strip count,
and no instruction id appears in the completion log without a dispatch. Those
are what make the spans below mean what they say. Everything else is reported.
"""

import argparse
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)

# Constants of the frozen operator, not of this script's choosing.
ROTATION_MMADS = 64
SCORE_MMADS_PER_STRIP = 8
ROTATION_FIXPIPES = 1
VECTOR_PIPES = {"RVECEX", "RVECST", "RVECLD", "RVECSU", "RVECLP", "VEC"}


def read(path):
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            tick, pipe, instr_id, operation = match.groups()
            records[int(instr_id)] = (int(tick), pipe, operation)
    return records


def select(records, needle):
    return sorted(((i, t) for i, (t, _, o) in records.items() if needle in o),
                  key=lambda row: row[0])


def describe(name, values):
    if not values:
        return f"  {name:<40} none"
    ordered = sorted(values)
    return (f"  {name:<40} n={len(values):3d}  min {min(values):7d}  "
            f"median {ordered[len(ordered) // 2]:7d}  max {max(values):7d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", type=Path)
    parser.add_argument("--strips", required=True, type=int)
    parser.add_argument("--core", default=0, type=int)
    args = parser.parse_args()

    cube_popped = args.dumps / f"core{args.core}.cubecore0.instr_popped_log.dump"
    cube_done = args.dumps / f"core{args.core}.cubecore0.instr_log.dump"
    for path in (cube_popped, cube_done):
        if not path.exists():
            raise SystemExit(f"missing Cube dump: {path}")
    dispatch, complete = read(cube_popped), read(cube_done)

    print(f"cube single-tile baseline: core {args.core}, {args.strips} strip(s)")

    # ---- the audited core ran the Cube body -----------------------------
    mmads = select(dispatch, "MMAD")
    fixpipes = select(dispatch, "FIX_L0C_TO_DST")
    want_mmads = ROTATION_MMADS + SCORE_MMADS_PER_STRIP * args.strips
    want_fix = ROTATION_FIXPIPES + args.strips
    if len(mmads) != want_mmads:
        raise SystemExit(
            f"core {args.core}: {len(mmads)} Mmad(s), expected {want_mmads} "
            f"({ROTATION_MMADS} rotation + {SCORE_MMADS_PER_STRIP} x "
            f"{args.strips} score). An idle core in this grid is not an empty "
            f"one -- the grid is 32 blocks and block N runs on core N, so "
            f"audit core<block>.")
    if len(fixpipes) != want_fix:
        raise SystemExit(f"core {args.core}: {len(fixpipes)} Fixpipe(s), "
                         f"expected {want_fix}")
    # A handful of completion records with no dispatch record is normal: the
    # model writes both logs when the process stops, and the last instructions
    # in flight can land in one and not the other. Measured on the three
    # baseline rungs: two orphans each, out of about three thousand
    # instructions. What would mean the dumps are not from one run is a large
    # fraction, so the check is a fraction and not a presence test -- the
    # presence test rejected three runs whose own output gates all passed.
    orphans = [i for i in complete if i not in dispatch]
    if complete and len(orphans) > max(8, len(complete) // 100):
        raise SystemExit(
            f"{len(orphans)} of {len(complete)} completion record(s) have no "
            f"dispatch record; at that fraction the two dumps are not from "
            f"one run")
    print(f"  budget: {len(mmads)} Mmad(s), {len(fixpipes)} Fixpipe(s), "
          f"{len(orphans)} completion record(s) without a dispatch")

    # ---- the rotation prologue ------------------------------------------
    rotation = mmads[:ROTATION_MMADS]
    rot_begin = rotation[0][1]
    rot_end = max(complete[i][0] for i, _ in rotation)
    print(f"\nthe ingress rotation prologue (strip invariant, once per launch)")
    print(f"  {ROTATION_MMADS} Mmad(s) span {rot_end - rot_begin} tick(s) "
          f"[{rot_begin}, {rot_end}]")

    # ---- per strip: the SPAN ---------------------------------------------
    score = mmads[ROTATION_MMADS:]
    score_fix = fixpipes[ROTATION_FIXPIPES:]
    print(f"\nper-strip SPAN: first score Mmad dispatched to that strip's "
          f"Fixpipe retired")
    print("  this is a LATENCY with nothing overlapping it; use it for a "
          "split's cold start")
    spans = []
    for strip in range(args.strips):
        group = score[strip * SCORE_MMADS_PER_STRIP:
                      (strip + 1) * SCORE_MMADS_PER_STRIP]
        begin = group[0][1]
        fix_id = score_fix[strip][0]
        end = complete.get(fix_id, (score_fix[strip][1],))[0]
        spans.append(end - begin)
        print(f"    strip {strip}: [{begin}, {end}]  span {end - begin}")
    print(describe("per-strip span", spans))

    # ---- per strip: the INTERVAL -----------------------------------------
    group_ticks = [score[strip * SCORE_MMADS_PER_STRIP][1]
                   for strip in range(args.strips)]
    print(f"\nper-strip INTERVAL: between consecutive score Mmad groups")
    print("  this is a RATE; use it as the Split-K overhead denominator")
    if len(group_ticks) < 2:
        print("    one strip, so there is no interval: this run gives the span "
              "only. Re-run with --strips 8 for the steady state.")
        interval = None
    else:
        deltas = [b - a for a, b in zip(group_ticks, group_ticks[1:])]
        print(f"    group ticks {group_ticks}")
        print(f"    intervals   {deltas}")
        # The operator's rings are two slots deep, so the first two columns are
        # pipeline fill and the steady state starts after them.
        steady = deltas[1:] if len(deltas) > 1 else deltas
        ordered = sorted(steady)
        interval = ordered[len(ordered) // 2]
        print(f"    steady state over {len(steady)} interval(s): min "
              f"{min(steady)}, median {interval}, max {max(steady)}")

    # ---- the whole launch, and the Vector side for context --------------
    launch_begin = min(t for _, (t, _, _) in dispatch.items())
    launch_end = max(t for _, (t, _, _) in complete.items())
    print(f"\nwhole Cube launch: [{launch_begin}, {launch_end}] = "
          f"{launch_end - launch_begin} tick(s)")

    vec_popped = args.dumps / f"core{args.core}.veccore0.instr_popped_log.dump"
    vec_done = args.dumps / f"core{args.core}.veccore0.instr_log.dump"
    if vec_popped.exists() and vec_done.exists():
        v_in, v_out = read(vec_popped), read(vec_done)
        vector = [(t, v_out[i][0]) for i, (t, p, _) in v_in.items()
                  if p in VECTOR_PIPES and i in v_out]
        if vector:
            merged, total = [], 0
            for begin, end in sorted(vector):
                end = max(end, begin + 1)
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([begin, end])
            total = sum(b - a for a, b in merged)
            print(f"AIV0 vector pipes in flight: {total} tick(s) over "
                  f"{len(merged)} span(s)")
            print("  the recurrence, for context: on this operator it is the "
                  "critical path and the Cube waits on it, which is why the "
                  "Mmad group interval above is the pipeline's strip rate and "
                  "not the Cube's own throughput")

    print("\n---- the numbers the crossover needs ----")
    print(f"T_compute_strip (span, one isolated strip)   "
          f"{spans[0] if spans else 'n/a'}")
    print(f"T_compute_strip (steady-state interval)      "
          f"{interval if interval is not None else 'n/a (needs --strips > 1)'}")
    print(f"rotation prologue, once per split             {rot_end - rot_begin}")
    print("\nOK cube single-tile baseline audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
