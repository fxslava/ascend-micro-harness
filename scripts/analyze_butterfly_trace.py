#!/usr/bin/env python3
"""Per-engine instruction and cycle census for the butterfly micro-benchmark.

Usage:
  analyze_butterfly_trace.py [--label NAME] [--rows N] [--reps N]
                             <popped_dump> <completed_dump>

  analyze_butterfly_trace.py --compare <summary.json> [<summary.json> ...]

Two modes, because the benchmark is a comparison and a comparison needs the
runs side by side.

Census mode reads one core's two instruction dumps and reports, per engine
queue, how many instructions it issued, over what tick span, and how many of
those ticks it was the only engine in flight. The engines that matter here are
the Reg-vector ones:

  RVECEX   register-vector execute   -- the arithmetic
  RVECLD   register-vector load      -- RV_VLD, a register filled from UB
  RVECST   register-vector store     -- RV_VST, a register spilled to UB
  RVECSU   register-vector scalar    -- address and loop bookkeeping inside VF
  RVECLP   register-vector loop      -- RV_VLOOP
  VEC      the level-2 vector queue  -- Approach A's whole arithmetic path

A register-resident leaf that is NOT register resident shows up here as
RVECST traffic it has no reason to issue: every store inside the measured
region beyond the four that publish the row is a spill. That is the zero-UB
-spill invariant stated as a number rather than as an intention.

The census writes a JSON summary beside the dumps so --compare can line the
variants up afterwards. Ticks come from one CAModel clock, so they are
directly comparable between runs of the same grid.

What this does NOT measure: silicon. These are cycle-accurate model ticks for
one AIV with no contention from any other core, and the model's issue rules
are not a timing guarantee for the part.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LINE = re.compile(
    r"\[(?P<tick>\d+)\]\s+\(PC: (?P<pc>0x[0-9a-f]+)\)\s+(?P<engine>[A-Z_0-9]+)\s*:"
    r".*?\(ID: (?P<id>\d+)\)\s+(?P<mnemonic>[A-Z_0-9]+)"
)

# Mnemonics that move a vector register to or from Unified Buffer. Everything
# else on a Reg-vector queue is arithmetic, a permute or bookkeeping.
LOAD_MNEMONICS = {"RV_VLD", "RV_VLDA", "RV_VLDU"}
STORE_MNEMONICS = {"RV_VST", "RV_VSTA", "RV_VSTU"}


def parse(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            rows.append(
                dict(
                    tick=int(match.group("tick")),
                    engine=match.group("engine"),
                    id=int(match.group("id")),
                    mnemonic=match.group("mnemonic"),
                )
            )
    if not rows:
        raise SystemExit(f"no instruction records in {path}")
    return rows


def census(rows):
    by_engine = defaultdict(list)
    for row in rows:
        by_engine[row["engine"]].append(row)
    out = {}
    for engine, items in by_engine.items():
        ticks = [item["tick"] for item in items]
        out[engine] = dict(
            instructions=len(items),
            first_tick=min(ticks),
            last_tick=max(ticks),
            span=max(ticks) - min(ticks),
            mnemonics=dict(Counter(item["mnemonic"] for item in items).most_common()),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--reps", type=int, default=0)
    parser.add_argument("--json", default="")
    parser.add_argument("--compare", nargs="*", default=None)
    parser.add_argument("dumps", nargs="*")
    args = parser.parse_args()

    if args.compare is not None:
        summaries = [json.loads(Path(p).read_text()) for p in args.compare]
        if not summaries:
            raise SystemExit("--compare needs at least one summary")
        keys = ["total_instructions", "vector_instructions", "rvecex", "rv_vld",
                "rv_vst", "vec_queue", "vector_span", "dispatch_span"]
        width = max(len(s["label"]) for s in summaries)
        header = f"{'variant':<{width}}  " + "  ".join(f"{k:>12}" for k in keys)
        print(header)
        print("-" * len(header))
        for s in summaries:
            print(
                f"{s['label']:<{width}}  "
                + "  ".join(f"{s.get(k, 0):>12}" for k in keys)
            )
        best = min(summaries, key=lambda s: s["vector_span"])
        print()
        print(f"shortest vector-engine span: {best['label']} "
              f"({best['vector_span']} model ticks over "
              f"{best.get('rows', '?')} rows x {best.get('reps', '?')} reps)")
        for s in summaries:
            if s["rv_vst"] > s.get("expected_stores", s["rv_vst"]):
                print(f"  note: {s['label']} issued {s['rv_vst']} RV_VST, more than the "
                      f"{s['expected_stores']} a spill-free row loop needs")
        return

    if len(args.dumps) != 2:
        raise SystemExit("census mode takes <popped_dump> <completed_dump>")

    popped = parse(args.dumps[0])
    completed = parse(args.dumps[1])
    pop_census = census(popped)
    com_census = census(completed)

    reg_engines = [e for e in pop_census if e.startswith("RVEC")]
    vector_engines = reg_engines + (["VEC"] if "VEC" in pop_census else [])

    counts = Counter()
    for row in popped:
        counts[row["mnemonic"]] += 1

    rv_vld = sum(counts[m] for m in LOAD_MNEMONICS)
    rv_vst = sum(counts[m] for m in STORE_MNEMONICS)

    vector_ticks = [row["tick"] for row in completed if row["engine"] in vector_engines]
    vector_span = (max(vector_ticks) - min(vector_ticks)) if vector_ticks else 0
    all_ticks = [row["tick"] for row in popped]

    summary = dict(
        label=args.label,
        rows=args.rows,
        reps=args.reps,
        total_instructions=len(popped),
        vector_instructions=sum(pop_census[e]["instructions"] for e in vector_engines),
        rvecex=pop_census.get("RVECEX", {}).get("instructions", 0),
        rvecsu=pop_census.get("RVECSU", {}).get("instructions", 0),
        rveclp=pop_census.get("RVECLP", {}).get("instructions", 0),
        vec_queue=pop_census.get("VEC", {}).get("instructions", 0),
        rv_vld=rv_vld,
        rv_vst=rv_vst,
        vector_span=vector_span,
        dispatch_span=max(all_ticks) - min(all_ticks),
        # A spill-free register-resident row loop loads four registers and
        # stores four per row, plus the one-off sign registers.
        expected_stores=4 * max(args.rows, 1) * max(args.reps, 1),
    )

    print(f"== {args.label}: rows {args.rows}, reps {args.reps} ==")
    print(f"{'engine':<10} {'instr':>8} {'first':>10} {'last':>10} {'span':>10}")
    for engine in sorted(pop_census, key=lambda e: -pop_census[e]["instructions"]):
        entry = pop_census[engine]
        print(f"{engine:<10} {entry['instructions']:>8} {entry['first_tick']:>10} "
              f"{entry['last_tick']:>10} {entry['span']:>10}")
    print()
    print("Reg-vector mnemonic census (dispatch):")
    for mnemonic, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if mnemonic.startswith("RV_") or mnemonic == "VF":
            kind = ("load" if mnemonic in LOAD_MNEMONICS
                    else "store" if mnemonic in STORE_MNEMONICS else "")
            print(f"  {mnemonic:<16} {count:>6}  {kind}")
    print()
    print(f"vector-engine completion span: {vector_span} model ticks "
          f"({summary['vector_instructions']} instructions over {vector_engines})")
    print(f"RV_VLD {rv_vld}, RV_VST {rv_vst} "
          f"(a spill-free row loop needs {summary['expected_stores']} stores)")
    print(f"whole-kernel dispatch span: {summary['dispatch_span']} model ticks "
          "-- fill plus drain, NOT a throughput figure")
    com_total = sum(e["instructions"] for e in com_census.values())
    print(f"completed-log instructions: {com_total}")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"summary written to {args.json}")


if __name__ == "__main__":
    main()
