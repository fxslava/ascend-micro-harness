#!/usr/bin/env python3
"""Did a cross-core announcement overtake the pipe work it reports?

This is MICROROUTINE_DESIGN_GUIDELINES.md section 6.2a's hazard, measured
directly instead of assumed in either direction.

A `CrossCoreSetFlag` lowers to a FLOWCTRL instruction -- `set_intra_block` in
mode 4, `ffts_cross_core_sync` in mode 2 -- so its logged tick is when the
SCALAR/FLOWCTRL stream issued it, not when the work it reports retired. Section
6.2a's original measurement caught one issued 1583 cycles ahead of the MTE1
load it was supposed to follow. The remedy is a `SetFlag<x_S>` /
`WaitFlag<x_S>` pair in front of it, which stalls issue until pipe x drains.

The question this answers is whether the pair is REDUNDANT on a given
channel, which it is exactly when the hardware already defers the
announcement until that pipe retires. Per announcement:

  `retired_before`  instructions on the reported pipe that had retired when
                    the announcement was issued. This is the comparable
                    number: an announcement that follows its own work shows
                    one more strip's worth than its predecessor.
  `retired_after`   retirements on that pipe afterwards. NOT a verdict on its
                    own -- the next strip's work is in there too.
  `lead`            ticks from the last retirement on that pipe to the
                    announcement.

**There is deliberately no pass/fail here.** Nothing in the trace labels a
retirement with the strip it belongs to, so no absolute threshold separates
"overtook its own work" from "was followed by the next strip's work". This
operator makes that concrete: AIV0 stops announcing kStageDepth strips before
the end, because there is no further GEMM to release, so its last
announcement is legitimately followed by two whole strips of recurrence. The
comparison that does work is between two builds of the SAME geometry, at the
same announcement ordinal -- which is the caller's job, not this script's.

The pipe each announcement reports is taken from the instruction's own
`PIPE:` field, which is the pipe named in the template argument.

usage: analyze_announce_overtake.py EVIDENCE_DIR [--block N] [--json]
       analyze_announce_overtake.py --self-test
exit 0 on a readable trace, 2 on no usable trace.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TICK = re.compile(r"^\[\w+\]\s+\[(\d+)\]")
ENGINE = re.compile(r"\)\s+(\w+)\s*:")
OP = re.compile(r"\(ID: \d+\)\s+(\w+)")
PIPE = re.compile(r"PIPE:(\w+)")
XT = re.compile(r"XT:X\d+=(0x[0-9a-fA-F]+|\d+)")
CORE = re.compile(r"core(\d+)\.(\w+)\.instr_popped_log\.dump")

# Which retirement engines carry each pipe's work, as the dumps name them. On
# the AIV the Vector pipe retires on the RISC-V vector units of the SIMT path
# as well as on the classic VEC column, so PIPE_V means both.
PIPE_ENGINES = {
    "VEC": ("VEC", "RVECLD", "RVECST", "RVECEX", "RVECSU", "RVECLP"),
    "MTE1": ("MTE1",),
    "MTE2": ("MTE2",),
    "MTE3": ("MTE3",),
    "FIXP": ("FIXP",),
    "CUBE": ("CUBE",),
    "M": ("CUBE",),
    "FIX": ("FIXP",),
}


def analyse(directory, block):
    """-> [finding] for every cross-core announcement in the trace."""
    findings = []
    for path in sorted(directory.glob("core*.instr_popped_log.dump")):
        name = CORE.match(path.name)
        if not name:
            continue
        core, unit = int(name[1]), name[2]
        if block is not None and core != block:
            continue
        # (tick, engine) for every retirement, and the announcements.
        events, announces = [], []
        with path.open(errors="replace") as source:
            for raw in source:
                tick, engine, op = TICK.match(raw), ENGINE.search(raw), OP.search(raw)
                if not tick or not engine or not op:
                    continue
                tick, engine, op = int(tick[1]), engine[1], op[1]
                events.append((tick, engine))
                if op in ("SET_CROSS_CORE", "SET_INTRA_BLOCK"):
                    pipe = PIPE.search(raw)
                    flag = XT.search(raw)
                    identity = None
                    if op == "SET_CROSS_CORE" and flag:
                        value = int(flag[1], 0)
                        if value & 0x1 == 0x1:
                            identity = "m%d:%d" % ((value >> 4) & 0x3,
                                                   (value >> 8) & 0xF)
                    elif op == "SET_INTRA_BLOCK":
                        found = re.search(r"sync_id:(\d+)", raw)
                        if found:
                            identity = "m4:%s" % found[1]
                    announces.append((tick, pipe[1] if pipe else "?", identity))

        by_engine = defaultdict(list)
        for tick, engine in events:
            by_engine[engine].append(tick)

        for tick, pipe, identity in announces:
            engines = PIPE_ENGINES.get(pipe, ())
            before, after = 0, 0
            last_before = None
            for engine in engines:
                for other in by_engine.get(engine, ()):
                    if other < tick:
                        before += 1
                        last_before = other if last_before is None else max(last_before, other)
                    elif other > tick:
                        after += 1
            findings.append({
                "core": "core%d.%s" % (core, unit),
                "channel": identity or "?",
                "pipe": pipe,
                "tick": tick,
                "retired_before": before,
                "retired_after": after,
                "lead": None if last_before is None else tick - last_before,
            })
    return findings


def per_channel(findings):
    """-> {channel: [retired_before]} in announcement order.

    The sequence is what a build-to-build comparison reads: each announcement
    that follows its own work shows roughly one more strip's worth of pipe
    retirements than the one before it, so a build in which the announcement
    stopped following its work shows a flatter or lower sequence.
    """
    series = {}
    for finding in sorted(findings, key=lambda f: f["tick"]):
        key = "%s %s PIPE_%s" % (finding["core"], finding["channel"], finding["pipe"])
        series.setdefault(key, []).append(finding["retired_before"])
    return series


def self_test():
    """The census must count what retired on each side of an announcement."""
    import shutil
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="overtake-selftest-"))
    failures = 0

    def line(tick, engine, index, text):
        return ("[info] [%08d] (PC: 0x1000) %-8s : (Binary: 0x0) (ID: %06d) %s\n"
                % (tick, engine, index, text))

    def write(directory, events):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "core0.veccore0.instr_popped_log.dump").write_text(
            "".join(line(t, e, i, x) for i, (t, e, x) in enumerate(events)))

    try:
        cases = (
            ("two retirements before the announcement", [
                (100, "RVECEX", "RV_VMUL"),
                (200, "RVECST", "RV_VST"),
                (300, "FLOWCTRL", "SET_CROSS_CORE  PIPE:VEC, XT:X1=0x821,"),
            ], (2, 0)),
            ("one before, one after", [
                (100, "RVECEX", "RV_VMUL"),
                (200, "FLOWCTRL", "SET_CROSS_CORE  PIPE:VEC, XT:X1=0x821,"),
                (300, "RVECST", "RV_VST"),
            ], (1, 1)),
            ("the mode-2 id is decoded from GetffstMsg", [
                (100, "FLOWCTRL", "SET_CROSS_CORE  PIPE:VEC, XT:X1=0x921,"),
            ], (0, 0)),
        )
        for index, (label, events, want) in enumerate(cases):
            directory = root / ("case%d" % index)
            write(directory, events)
            found = analyse(directory, 0)
            got = (found[0]["retired_before"], found[0]["retired_after"])
            if got != want:
                print("  UNEXPECTED self-test %r: %s, wanted %s" % (label, got, want))
                failures += 1
            else:
                print("  %-46s before=%d after=%d" % (label, got[0], got[1]))
        # The id decode, checked on its own: 0x921 is mode 2, flag 9.
        directory = root / "case2"
        if analyse(directory, 0)[0]["channel"] != "m2:9":
            print("  UNEXPECTED self-test: GetffstMsg id decode")
            failures += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("%s: %d self-test failure(s)" % ("FAIL" if failures else "PASS", failures))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--strips", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("announcement overtake self-test")
        return self_test()
    if args.directory is None:
        parser.error("EVIDENCE_DIR is required unless --self-test is given")

    findings = analyse(args.directory, args.block)
    if not findings:
        print("no cross-core announcement in %s" % args.directory, file=sys.stderr)
        return 2
    series = per_channel(findings)

    if args.json:
        print(json.dumps({"announcements": findings,
                          "retired_before_by_channel": series}, indent=2))
        return 0

    print("%d cross-core announcement(s)" % len(findings))
    for finding in findings:
        print("  %-16s %-6s on PIPE_%-5s tick %-8d retired before %-6d "
              "after %-6d lead %s"
              % (finding["core"], finding["channel"], finding["pipe"],
                 finding["tick"], finding["retired_before"],
                 finding["retired_after"],
                 "-" if finding["lead"] is None else finding["lead"]))
    print("  -- retired_before, in announcement order, per channel:")
    for key, values in sorted(series.items()):
        print("  %-34s %s" % (key, values))
    print("CENSUS only: compare these against the same geometry's other build; "
          "see the module docstring for why there is no verdict here")
    return 0


if __name__ == "__main__":
    sys.exit(main())
