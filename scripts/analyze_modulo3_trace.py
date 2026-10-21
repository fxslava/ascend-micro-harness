#!/usr/bin/env python3
"""Summarize the 950PR CAModel core-0 queue trace for the three-tile probe."""

import re
import sys
from collections import Counter
from pathlib import Path


LINE = re.compile(
    r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)"
)
FLAG = re.compile(
    r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)"
)


def read_log(path):
    records = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            tick, pipe, instr_id, operation = match.groups()
            records[int(instr_id)] = (int(tick), pipe, operation)
    return records


def main(popped_path, completed_path):
    popped = read_log(popped_path)
    completed = read_log(completed_path)
    sets = Counter()
    waits = Counter()
    intervals = {"MTE2": [], "Vector": [], "MTE3": []}
    last_wait_dispatch = 0
    for instr_id, (begin, pipe, operation) in popped.items():
        flag = FLAG.search(operation)
        if flag:
            source, target, slot = flag.groups()
            key = (source, target, int(slot, 0))
            if "SET_FLAG" in operation:
                sets[key] += 1
            elif "WAIT_FLAG" in operation:
                waits[key] += 1
                last_wait_dispatch = max(last_wait_dispatch, begin)
        if instr_id not in completed:
            continue
        end = completed[instr_id][0]
        if "MOV_SRC_TO_DST_ALIGNv2" in operation and pipe in ("MTE2", "MTE3"):
            intervals[pipe].append((begin, end, instr_id))
        elif "RV_VADDS" in operation:
            intervals["Vector"].append((begin, end, instr_id))

    if sets != waits or sum(sets.values()) != 18:
        raise SystemExit(f"event mismatch: sets={sets}, waits={waits}")
    if any(slot > 2 for _, _, slot in sets):
        raise SystemExit("event ID exceeds slot range 0..2")

    best = None
    for ingress in intervals["MTE2"]:
        for vector in intervals["Vector"]:
            for egress in intervals["MTE3"]:
                start = max(ingress[0], vector[0], egress[0])
                stop = min(ingress[1], vector[1], egress[1])
                if start < stop and (best is None or stop - start > best[1] - best[0]):
                    best = (start, stop, ingress[2], vector[2], egress[2])
    if best is None:
        raise SystemExit("no three-pipe overlap in hardware trace")

    last_mte3_retire = max(end for _, end, _ in intervals["MTE3"])
    print(f"hardware flags: {sum(sets.values())} sets / {sum(waits.values())} waits; "
          f"{len(sets)} route-ID pairs; IDs 0..2; zero count residual")
    print("three-pipe overlap: CAModel cycles "
          f"{best[0]}..{best[1]} (MTE2 ID {best[2]}, "
          f"Vector ID {best[3]}, MTE3 ID {best[4]})")
    print(f"Scalar queued final event waits by cycle {last_wait_dispatch}; "
          f"last MTE3 copy completed at cycle {last_mte3_retire}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: analyze_modulo3_trace.py popped.dump completed.dump")
    main(sys.argv[1], sys.argv[2])
