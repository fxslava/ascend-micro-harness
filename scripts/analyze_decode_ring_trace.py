#!/usr/bin/env python3
"""Audit capacity-one flags and actual MTE2/Vector in-flight overlap per core."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_single_token_decode_trace import flag_key, parse


def places(rows, order):
    marking = Counter()
    sets, waits = Counter(), Counter()
    for row in sorted(rows, key=order):
        op = row["mnemonic"]
        if op not in ("SET_FLAG", "WAIT_FLAG"):
            continue
        key = flag_key(row)
        if "?" in key or not 0 <= key[2] <= 5:
            raise ValueError(f"invalid/reserved flag route {key}")
        if op == "SET_FLAG":
            if marking[key]:
                raise ValueError(f"double SET of occupied place {key}, ID={row['id']}")
            marking[key] = 1
            sets[key] += 1
        else:
            if marking[key] != 1:
                raise ValueError(f"WAIT without token {key}, ID={row['id']}")
            marking[key] = 0
            waits[key] += 1
    if sets != waits or any(marking.values()):
        raise ValueError("unbalanced flags / nonempty final marking")
    return sets


def audit(popped_path, strips=None):
    complete_path = popped_path.with_name(popped_path.name.replace("instr_popped_log", "instr_log"))
    popped, completed = parse(popped_path), parse(complete_path)
    if not popped and not completed:
        return dict(core=popped_path.name, instructions=0, active=False, routes={},
                    residual_tokens=0, mte2_vector_intersections=0, overlap_examples=[], engines={})
    if not popped or not completed:
        raise ValueError(f"incomplete trace {popped_path}")
    flags = places(popped, lambda r: r["id"])
    actual = places(completed, lambda r: (r["tick"], r["id"]))
    if flags != actual:
        raise ValueError("dispatched/completed token census differs")
    end = {r["id"]: r for r in completed}
    intervals = []
    for row in popped:
        if row["id"] not in end:
            # CAModel omits some register barriers from the completion log.
            # Token census above still requires every flag to complete.
            continue
        finish = end[row["id"]]
        if finish["mnemonic"] != row["mnemonic"]:
            raise ValueError("instruction ID mismatch")
        intervals.append((row["tick"], finish["tick"], row["engine"], row["mnemonic"]))
    vector = [x for x in intervals if x[2].startswith(("VEC", "RVEC")) and
              x[3] not in ("SET_FLAG", "WAIT_FLAG")]
    if vector:
        for slot in range(2):
            generations = None if strips is None else (strips + 1 - slot) // 2
            for flag_id in (slot, 2 + slot):
                count = flags[("VEC", "MTE2", flag_id)]
                if count == 0 or (generations is not None and count != 1 + generations):
                    raise ValueError(f"missing/incorrect feedback census: slot={slot}, id={flag_id}, count={count}")
    ingress = [x for x in intervals if x[2] == "MTE2" and x[3].startswith("MOV_SRC_TO_DST")]
    overlaps = [(a[0], a[1], b[0], b[1]) for a in ingress for b in vector
                if max(a[0], b[0]) < min(a[1], b[1])]
    return dict(core=popped_path.name, instructions=len(popped), active=bool(vector),
                routes={str(k): v for k, v in sorted(flags.items())},
                residual_tokens=0, mte2_vector_intersections=len(overlaps),
                overlap_examples=overlaps[:3],
                engines=dict(Counter(r["engine"] for r in popped)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--strips", type=int, help="expected strips on each active core (isolated probes)")
    args = parser.parse_args()
    if args.strips is not None and args.strips < 0:
        raise ValueError("strip count must be nonnegative")
    paths = sorted(args.directory.glob("core*.instr_popped_log.dump"))
    if not paths:
        raise ValueError("no instruction traces; cannot audit")
    results = [audit(path, args.strips) for path in paths]
    if not any(r["active"] for r in results):
        raise ValueError("no active Vector core; vacuous audit")
    if args.require_overlap and not any(r["mte2_vector_intersections"] for r in results):
        raise ValueError("no observed MTE2/Vector overlap")
    print(json.dumps(dict(status="PASS", idle_cores=sum(not r["active"] for r in results),
                          cores=[r for r in results if r["active"]]), indent=2))


if __name__ == "__main__":
    main()
