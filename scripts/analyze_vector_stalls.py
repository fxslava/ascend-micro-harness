#!/usr/bin/env python3
"""Where do a Vector core's fine-grained stalls actually sit?

The interval census at gap 64 says AIV0's Vector pipe is one unbroken busy
interval. At gap 16 the same pipe shows 842 gaps totalling 19,190 ticks, 27% of
its span -- so there IS real sub-16-tick stalling, and the question is which
phase of the recurrence it sits in front of.

This attributes each gap to the PC that retires immediately AFTER it: that is
the instruction the pipe was waiting to issue, so its address localises the
stall to a phase. Phases are then read off the PC histogram, since the per-row
body is one contiguous code region executed 16 times per strip.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

TICK = re.compile(r"^\[\w+\]\s+\[(\d+)\]")
ENGINE = re.compile(r"\)\s+(\w+)\s*:")
PC = re.compile(r"PC: (0x[0-9a-f]+)")
OP = re.compile(r"\(ID: \d+\)\s+(\w+)")

VECTOR = {"VEC", "RVECEX", "RVECLD", "RVECST", "RVECSU", "RVECLP"}

directory = Path(sys.argv[1])
gap = int(sys.argv[2]) if len(sys.argv) > 2 else 16

events = []
path = directory / "core0.veccore0.instr_popped_log.dump"
with path.open(errors="replace") as handle:
    for raw in handle:
        tick, engine, pc, op = (TICK.match(raw), ENGINE.search(raw),
                                PC.search(raw), OP.search(raw))
        if not (tick and engine and pc and op):
            continue
        if engine[1] in VECTOR:
            events.append((int(tick[1]), pc[1], op[1]))

events.sort()
print("vector retirements: %d" % len(events))

# Gaps larger than `gap`, attributed to the instruction that follows them.
holes = defaultdict(lambda: [0, 0])   # pc -> [count, total ticks]
by_op = defaultdict(lambda: [0, 0])
total = 0
for i in range(1, len(events)):
    delta = events[i][0] - events[i - 1][0]
    if delta > gap:
        holes[events[i][1]][0] += 1
        holes[events[i][1]][1] += delta
        by_op[events[i][2]][0] += 1
        by_op[events[i][2]][1] += delta
        total += delta

span = events[-1][0] - events[0][0]
print("span %d ticks, stalled %d ticks in %d gaps > %d ticks (%.1f%%)"
      % (span, total, sum(v[0] for v in holes.values()), gap, 100.0 * total / span))

print("\ntop PCs the pipe was waiting to issue (count, ticks, mnemonic):")
rows = sorted(holes.items(), key=lambda kv: -kv[1][1])[:14]
mnemonic = {}
for tick, pc, op in events:
    mnemonic.setdefault(pc, op)
for pc, (count, ticks) in rows:
    print("  %-12s %6d gaps  %8d ticks  %6.1f avg  %s"
          % (pc, count, ticks, ticks / count, mnemonic.get(pc, "?")))

print("\nby mnemonic:")
for op, (count, ticks) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[:12]:
    print("  %-14s %6d gaps  %8d ticks  %6.1f avg" % (op, count, ticks, ticks / count))
