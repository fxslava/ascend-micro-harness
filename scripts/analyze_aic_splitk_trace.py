#!/usr/bin/env python3
"""Audit dynamic MIX decode page handoffs and once-per-worker query rotation.

Read completed *.instr_log.dump files. AIV1 reports local mode-4 IDs; rebase
them by 16 before pairing with the AIC's absolute IDs, as in the legacy audit.
With --workers/--pages/--strips, also check 64 rotation Mmads per active worker
plus 8 score Mmads per processed strip. Counts can span multiple launches.
"""
import argparse
import json
import re
from collections import Counter

LINE = re.compile(r"\[\d+\].*?\(ID: \d+\)\s+(\w+)(.*)")
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:0x([0-9a-f]+)")
MODE4 = re.compile(r"sync_id:(\d+)")


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pages", type=int)
    parser.add_argument("--strips", type=int)
    args = parser.parse_args()
    sets, waits, cross_sets, cross_waits, ops = (Counter() for _ in range(5))
    for path in args.directory.glob("core*.instr_log.dump"):
        core = int(re.match(r"core(\d+)", path.name)[1])
        rebase = 16 if ".veccore1." in path.name else 0
        with path.open(errors="replace") as source:
            for raw in source:
                match = LINE.search(raw)
                if not match:
                    continue
                op, tail = match.groups()
                ops[op] += 1
                if op in ("SET_FLAG", "WAIT_FLAG"):
                    flag = FLAG.search(tail)
                    if flag:
                        pipe, target, number = flag.groups()
                        number = int(number, 16)
                        if number >= 6:
                            raise ValueError(f"{path.name}: reserved HardEvent ID {number}")
                        key = path.name, pipe, target, number
                        (sets if op == "SET_FLAG" else waits)[key] += 1
                if op in ("SET_INTRA_BLOCK", "WAIT_INTRA_BLOCK"):
                    flag = MODE4.search(tail)
                    if flag:
                        key = core, int(flag[1]) + rebase
                        (cross_sets if op == "SET_INTRA_BLOCK" else cross_waits)[key] += 1
    if not sets or sets != waits:
        raise ValueError(f"HardEvent imbalance: {sets - waits}, {waits - sets}")
    if not cross_sets or cross_sets != cross_waits:
        raise ValueError(f"mode-4 imbalance: {cross_sets - cross_waits}, {cross_waits - cross_sets}")
    mode4 = Counter()
    for (_, flag), count in cross_sets.items():
        mode4[flag] += count
    mmads = sum(count for op, count in ops.items() if "MMAD" in op)
    if any(v is not None for v in (args.workers, args.pages, args.strips)):
        if any(v is None for v in (args.workers, args.pages, args.strips)):
            parser.error("supply --workers, --pages and --strips together")
        expected = args.workers * 64 + args.pages * args.strips * 8
        if mmads != expected or mode4[10] != args.pages or mode4[26] != args.pages:
            raise ValueError(f"query/page census mismatch: Mmads {mmads}/{expected}, handoffs {mode4}")
    print(json.dumps(dict(hardevent_pairs=sum(sets.values()), reserved_event_ids=0,
                         mode4_sets_and_waits=dict(mode4), cube_mmads=mmads), indent=2))


if __name__ == "__main__":
    main()
