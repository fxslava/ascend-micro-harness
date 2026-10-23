#!/usr/bin/env python3
"""Audit a single WTA launch; report popped-to-completed instruction latency.

The final S8 identity Mmad / GM Fixpipe belong to the observation wrapper and
are excluded from recurrent step cost. Source order is not timing evidence.
"""
import argparse
from collections import Counter
from pathlib import Path
import re

LINE = re.compile(r"\[(\d+)\].*?\)\s+(\S+)\s+:\s+\(Binary:.*?\)\s+\(ID:\s*(\d+)\)\s+(.+)")
FLAG = re.compile(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)")


def records(path):
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            tick, pipe, ident, op = m.groups()
            result[int(ident)] = (int(tick), pipe, op)
    return result


def audit(directory, steps):
    popped = records(directory / "core0.cubecore0.instr_popped_log.dump")
    completed = records(directory / "core0.cubecore0.instr_log.dump")
    assert popped and popped.keys() <= completed.keys(), "missing completion / truncated trace"
    # CAModel records barriers/DSB only at completion, without a popped entry.
    assert all(completed[i][2].startswith(("BAR ", "DSB "))
               for i in completed.keys() - popped.keys()), "unmatched completion"
    sets, waits = Counter(), Counter()
    for _, _, op in popped.values():
        assert "INTRA_BLOCK" not in op, "cross-core synchronization in AIC-only probe"
        m = FLAG.search(op)
        if m:
            key = (m[1], m[2], int(m[3], 0))
            assert key[2] <= 5, "reserved event ID"
            (sets if op.startswith("SET_FLAG") else waits)[key] += 1
    assert sets == waits, f"token count mismatch: {sets} vs {waits}"
    expected = {("FIXP", "MTE1", 0): steps + 1,
                ("MTE1", "CUBE", 0): steps, ("CUBE", "FIXP", 0): steps,
                ("FIXP", "SCALAR", 0): 1,
                ("MTE2", "MTE1", 1): 1, ("MTE1", "CUBE", 1): 1,
                ("CUBE", "FIXP", 1): 1, ("FIXP", "SCALAR", 1): 1}
    assert dict(sets) == expected, f"unexpected schedule: {sets}"
    # Check temporal one-bit ownership at retirement, not just aggregate counts.
    live = set()
    for _, (_, _, op) in sorted(completed.items(), key=lambda p: (p[1][0], p[0])):
        m = FLAG.search(op)
        if not m:
            continue
        key = (m[1], m[2], int(m[3], 0))
        if op.startswith("SET_FLAG"):
            assert key not in live, f"double set at retirement: {key}"
            live.add(key)
        else:
            assert key in live, f"wait without token: {key}"
            live.remove(key)
    assert not live
    for path in directory.glob("core*.instr_popped_log.dump"):
        if path.name != "core0.cubecore0.instr_popped_log.dump":
            assert not records(path), f"unexpected active core: {path.name}"
    for path in [directory / "run.log", *directory.glob("*excp_log.dump")]:
        assert not re.search(r"invalid ldst addr|su_ccu_mpu_err|\[(?:error|fatal)\]",
                             path.read_text(errors="replace"), re.I), f"fault in {path}"

    def select(predicate):
        return [(i, row) for i, row in sorted(popped.items()) if predicate(row[2])]

    loads = select(lambda op: op.startswith("LOAD_2D") and "Dst:L0A" in op and "Dtype:F16" in op)
    mmads = select(lambda op: op.startswith("MMAD ") and "dtype:F16F32" in op)
    fixes = select(lambda op: op.startswith("FIX_L0C_TO_DST") and "Dst:L1," in op)
    observer_mma = select(lambda op: op.startswith("MMAD ") and "dtype:S8 " in op)
    observer_fix = select(lambda op: op.startswith("FIX_L0C_TO_DST") and "Dst:OUT," in op)
    assert len(loads) == len(mmads) == len(fixes) == steps, "wrong recurrent instruction census"
    assert len(observer_mma) == len(observer_fix) == 1, "wrong observation census"
    assert len(select(lambda op: op.startswith("FIX_L0C_TO_DST"))) == steps + 1, "unexpected Fixpipe path"
    assert len(select(lambda op: op.startswith("MMAD "))) == steps + 1, "unexpected Cube work"
    assert not select(lambda op: "Src:UB" in op or "Dst:UB" in op), "UB bounce"
    sign_match = re.search(r"PASS native WTA T=\d+ signed=(\d)", (directory / "run.log").read_text())
    assert sign_match, "no numeric validation result"
    quant = select(lambda op: "SPR:QUANT_PRE," in op)
    assert len(quant) == 1, "scalar quantization must configure exactly once"
    quant_value = re.search(r"XN:X\d+=(0x[0-9a-f]+|\d+)", quant[0][1][2])
    scale_match = re.search(r"scale_bits=(0x[0-9a-f]+)", (directory / "run.log").read_text())
    assert scale_match, "missing scale calibration"
    assert quant_value and int(quant_value[1], 0) == (int(scale_match[1], 0) | (int(sign_match[1]) << 46)), "wrong QUANT_PRE bits"
    assert observer_mma[0][1][0] > completed[fixes[-1][0]][0], "observer preceded victory"
    print(f"PASS AIC-only: scalar addressing faults=0; {sum(sets.values())} event pairs; residual tokens=0")
    for route, count in sorted(sets.items()):
        print(f"  {route[0]}->{route[1]} id={route[2]} sets=waits={count}")
    print("step,mte1_cycles,mmad_cycles,fixpipe_cycles,sum_engine_cycles,load_to_fix_complete")
    fix_latencies = []
    for t, (load, mma, fix) in enumerate(zip(loads, mmads, fixes), 1):
        durations = [completed[i][0] - row[0] for i, row in (load, mma, fix)]
        assert all(d >= 0 for d in durations)
        assert completed[load[0]][0] <= mma[1][0]
        assert completed[mma[0]][0] <= fix[1][0]
        if t > 1:
            assert completed[fixes[t - 2][0]][0] <= load[1][0], "feedback RAW"
        fix_latencies.append(durations[-1])
        span = completed[fix[0]][0] - load[1][0]
        print(f"{t},{durations[0]},{durations[1]},{durations[2]},{sum(durations)},{span}")
    if steps > 1:
        print(f"Final B8 Fixpipe minus preceding FP16 Fixpipe: {fix_latencies[-1] - fix_latencies[-2]} cycles")
        print("Successive MTE1 load intervals: " + ",".join(
            str(b[1][0] - a[1][0]) for a, b in zip(loads, loads[1:])))
    print("Latency uses completion_tick - popped_tick; dispatch/event gaps are excluded from engine sum.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    audit(args.directory, args.steps)
