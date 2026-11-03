#!/usr/bin/env python3
"""Instruction census across the decode rotation legs.

A bypass that agrees numerically has not been shown to BE a bypass: an
`if constexpr` that was compiled the wrong way round, or a leaf that still runs
and happens to be the identity on the fixture, produces the same numbers. What
distinguishes them is the instruction stream, so this reads each leg's own
`core*.instr_popped_log.dump` and reports what the compared legs issued.

Usage:

    python3 scripts/audit_decode_rotation_census.py \\
        --leg model=<dir> --leg rotated=<dir> [--core core0.veccore0] \\
        [--expect-fewer rotated:model] [--expect-absent MAD:rotated]

Each `--leg NAME=DIR` names one run's artifact directory. `--core` selects the
core whose stream is compared, by dump-file prefix; it may be repeated. The
report is JSON on stdout.

`--expect-fewer A:B` requires leg A to have issued strictly fewer instructions
than leg B on every compared core -- the gate that work was removed rather than
merely reordered. `--expect-absent MNEMONIC:LEG` requires a mnemonic to appear
zero times on that leg and at least once on some other leg, which is how an
eliminated stage is stated as a fact about the trace rather than about the
source. `--expect-delta MNEMONIC:A:B:N` requires leg B to have issued EXACTLY
N more of that mnemonic than leg A, summed over the compared cores, which is
how "the 64-Mmad prologue is gone" becomes a number the run either produces or
does not.

Simulator ticks are not silicon throughput and instruction counts are not
cycles; what this establishes is presence, absence and ordering of work.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# `[tick] (PC: 0x...) ... (ID: n) MNEMONIC ...` -- the same shape
# scripts/analyze_splitk_decode_trace.py keys off, mnemonic only.
LINE = re.compile(r"\(PC: 0x[0-9a-f]+\).*?\(ID: \d+\)\s+(\w+)")


def census(path):
    """Instruction mnemonics of one core's popped-instruction dump."""
    counts = Counter()
    total = 0
    for line in path.read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        counts[match.group(1)] += 1
        total += 1
    return total, counts


def cores_of(directory, prefixes):
    found = {}
    for prefix in prefixes:
        path = directory / f"{prefix}.instr_popped_log.dump"
        if not path.is_file():
            raise SystemExit(f"missing trace: {path}")
        found[prefix] = path
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", action="append", required=True, metavar="NAME=DIR",
                    help="a leg's name and its artifact directory")
    ap.add_argument("--core", action="append", default=None, metavar="PREFIX",
                    help="dump prefix to compare, e.g. core0.veccore0")
    ap.add_argument("--expect-fewer", action="append", default=[], metavar="A:B",
                    help="require leg A to issue strictly fewer than leg B")
    ap.add_argument("--expect-absent", action="append", default=[],
                    metavar="MNEMONIC:LEG",
                    help="require MNEMONIC absent on LEG and present elsewhere")
    ap.add_argument("--expect-delta", action="append", default=[],
                    metavar="MNEMONIC:A:B:N",
                    help="require leg B to issue exactly N more MNEMONIC than leg A")
    ap.add_argument("--top", type=int, default=12,
                    help="mnemonics to list per core (default 12)")
    args = ap.parse_args()

    prefixes = args.core or ["core0.veccore0"]
    legs = {}
    for spec in args.leg:
        if "=" not in spec:
            raise SystemExit(f"--leg wants NAME=DIR, got {spec!r}")
        name, _, directory = spec.partition("=")
        legs[name] = Path(directory)

    report = {"cores": prefixes, "legs": {}}
    for name, directory in legs.items():
        per_core = {}
        for prefix, path in cores_of(directory, prefixes).items():
            total, counts = census(path)
            per_core[prefix] = {
                "instructions": total,
                "top": dict(counts.most_common(args.top)),
                "mnemonics": len(counts),
            }
        report["legs"][name] = {"directory": str(directory), "per_core": per_core}

    failures = []
    for spec in args.expect_fewer:
        fewer, _, more = spec.partition(":")
        for leg in (fewer, more):
            if leg not in report["legs"]:
                raise SystemExit(f"--expect-fewer names an unknown leg: {leg}")
        for prefix in prefixes:
            a = report["legs"][fewer]["per_core"][prefix]["instructions"]
            b = report["legs"][more]["per_core"][prefix]["instructions"]
            entry = {"check": "fewer", "fewer": fewer, "more": more, "core": prefix,
                     "fewer_count": a, "more_count": b, "removed": b - a,
                     "pass": a < b}
            report.setdefault("checks", []).append(entry)
            if not entry["pass"]:
                failures.append(f"{fewer} did not issue fewer than {more} on {prefix}"
                                f" ({a} vs {b})")

    for spec in args.expect_absent:
        mnemonic, _, leg = spec.partition(":")
        if leg not in report["legs"]:
            raise SystemExit(f"--expect-absent names an unknown leg: {leg}")
        here = sum(report["legs"][leg]["per_core"][p]["top"].get(mnemonic, 0)
                   for p in prefixes)
        # Present somewhere else, or the absence says nothing: a mnemonic no
        # leg issues is not evidence that this leg eliminated it.
        elsewhere = max(
            (sum(report["legs"][other]["per_core"][p]["top"].get(mnemonic, 0)
                 for p in prefixes)
             for other in report["legs"] if other != leg), default=0)
        entry = {"check": "absent", "mnemonic": mnemonic, "leg": leg,
                 "on_leg": here, "elsewhere": elsewhere,
                 "pass": here == 0 and elsewhere > 0}
        report.setdefault("checks", []).append(entry)
        if not entry["pass"]:
            failures.append(
                f"{mnemonic} is {here} on {leg} and {elsewhere} elsewhere: "
                "absence not established")

    for spec in args.expect_delta:
        parts = spec.split(":")
        if len(parts) != 4:
            raise SystemExit(f"--expect-delta wants MNEMONIC:A:B:N, got {spec!r}")
        mnemonic, fewer, more, want = parts[0], parts[1], parts[2], int(parts[3])
        for leg in (fewer, more):
            if leg not in report["legs"]:
                raise SystemExit(f"--expect-delta names an unknown leg: {leg}")
        counted = {}
        for leg in (fewer, more):
            counted[leg] = sum(
                report["legs"][leg]["per_core"][p]["top"].get(mnemonic, 0)
                for p in prefixes)
        got = counted[more] - counted[fewer]
        entry = {"check": "delta", "mnemonic": mnemonic, "fewer": fewer,
                 "more": more, "fewer_count": counted[fewer],
                 "more_count": counted[more], "delta": got, "expected": want,
                 "pass": got == want}
        report.setdefault("checks", []).append(entry)
        if not entry["pass"]:
            failures.append(
                f"{more} issued {got} more {mnemonic} than {fewer}, expected {want}"
                f" ({counted[more]} vs {counted[fewer]})")

    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
