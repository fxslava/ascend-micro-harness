#!/usr/bin/env python3
"""Audit completed TAS/ticket/release sequences and state DMA in dav-3510 dumps.

The kernel has exactly three atomic instruction PCs: acquire exchange, arrival
add, release exchange, in first-observed order. No opcode encoding is assumed.
Use a completed simulator run; partial dumps cannot establish a bound. The
reported interval spans completion of the successful exchange through completion
of the release exchange. It is a trace interval, not a silicon measurement.
"""
import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r"\[(\d+)\]\s+\(PC: (0x[0-9a-f]+)\).*?\(ID: \d+\)\s+(\w+)(.*)")


def audit(path):
    events = []
    pcs = []
    for line in path.read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        tick, pc, mnemonic, tail = match.groups()
        if mnemonic == "SIMT_ATOM":
            if pc not in pcs:
                pcs.append(pc)
            events.append((int(tick), pc, "atomic"))
        elif mnemonic.startswith("MOV_SRC_TO_DST"):
            direction = "read" if "Src:OUT, Dst:UB" in tail else (
                "write" if "Src:UB, Dst:OUT" in tail else "other")
            events.append((int(tick), pc, direction))
    if not pcs:
        return []
    if len(pcs) != 3:
        raise ValueError(f"{path.name}: expected three atomic PCs, saw {len(pcs)}")
    acquire, ticket, release = pcs
    last_acquire = None
    held = None
    attempts = 0
    intervals = []
    for tick, pc, kind in events:
        if kind != "atomic":
            continue
        if pc == acquire:
            last_acquire = tick
            attempts += 1
        elif pc == ticket:
            if last_acquire is None or held is not None:
                raise ValueError(f"{path.name}: malformed acquisition sequence")
            held = last_acquire
        elif pc == release:
            if held is None:
                raise ValueError(f"{path.name}: release without ticket")
            reads = sum(held <= t <= tick and k == "read" for t, _, k in events)
            writes = sum(held <= t <= tick and k == "write" for t, _, k in events)
            intervals.append(dict(core=path.name, begin=held, end=tick,
                                  hold_ticks=tick - held, failed_exchanges=attempts - 1,
                                  state_reads=reads, state_writes=writes))
            held = last_acquire = None
            attempts = 0
    if held is not None or last_acquire is not None:
        raise ValueError(f"{path.name}: unfinished or timed-out acquisition")
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--assert-batched", action="store_true")
    parser.add_argument("--max-hold-ticks", type=int)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("core*.veccore*.instr_log.dump")):
        rows.extend(audit(path))
    if not rows:
        raise ValueError("no completed lock intervals")
    print(json.dumps(rows, indent=2))
    if args.assert_batched and any(r["state_reads"] > 1 or r["state_writes"] > 1 for r in rows):
        raise ValueError("critical section has per-row state DMA")
    if args.max_hold_ticks is not None and any(r["hold_ticks"] >= args.max_hold_ticks for r in rows):
        raise ValueError(f"strict hold bound < {args.max_hold_ticks} ticks was not achieved")


if __name__ == "__main__":
    main()
