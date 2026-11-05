#!/usr/bin/env python3
"""Convert in-kernel phase boundary CSV records to Perfetto complete events.

Input CSV columns: block_idx,role,phase_id,cycle_timestamp. A phase record marks
the start of that phase; phase 6 is the terminal boundary after phase 5. Cycle
timestamps are only compared within one (block_idx, role) clock domain.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PHASES = {
    1: ("PROLOGUE_QUERY_FWHT", "AIV"),
    2: ("ATTN_TILE_0", "Pipeline"),
    3: ("SYNC_ALL_BARRIER", "Barrier"),
    4: ("CUBE_GEMM_REDUCE", "Cube GEMM"),
    5: ("EPILOGUE_OUTPUT_FWHT", "AIV"),
}


@dataclass(frozen=True)
class Boundary:
    block: int
    role: str
    phase: int
    cycle: int


def read_boundaries(path: Path) -> list[Boundary]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"block_idx", "role", "phase_id", "cycle_timestamp"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        boundaries = []
        for line, row in enumerate(reader, start=2):
            try:
                item = Boundary(int(row["block_idx"]), row["role"],
                                int(row["phase_id"]), int(row["cycle_timestamp"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid record at CSV line {line}") from exc
            if item.block < 0 or item.phase not in (*PHASES, 6) or item.cycle < 0:
                raise ValueError(f"out-of-range record at CSV line {line}")
            boundaries.append(item)
    if not boundaries:
        raise ValueError("cycle log contains no records")
    return boundaries


def convert(boundaries: list[Boundary], clock_hz: float) -> tuple[dict, list[dict]]:
    groups: dict[tuple[int, str], list[Boundary]] = defaultdict(list)
    for boundary in boundaries:
        groups[(boundary.block, boundary.role)].append(boundary)

    # Different core types can have unrelated counter epochs. Each block/role is
    # normalized independently; the output is a phase-duration view, not a claim
    # that timestamps from separate cores share a synchronized clock.
    events = []
    phase_cycles: dict[int, int] = defaultdict(int)
    for (block, role), records in sorted(groups.items()):
        records.sort(key=lambda r: (r.cycle, r.phase))
        phases = [r.phase for r in records]
        if phases != [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"block {block} role {role} needs ordered phase boundaries 1..6; got {phases}"
            )
        origin = records[0].cycle
        for start, end in zip(records, records[1:]):
            if end.cycle <= start.cycle:
                raise ValueError(f"non-positive phase duration for block {block}, phase {start.phase}")
            name, lane = PHASES[start.phase]
            duration_cycles = end.cycle - start.cycle
            phase_cycles[start.phase] += duration_cycles
            events.append({
                "name": name,
                "cat": "tq",
                "ph": "X",
                "ts": (start.cycle - origin) * 1_000_000.0 / clock_hz,
                "dur": duration_cycles * 1_000_000.0 / clock_hz,
                "pid": f"Block {block}",
                "tid": lane,
                "args": {"block_idx": block, "role": role,
                         "phase_id": start.phase, "cycles": duration_cycles},
            })
    total = sum(phase_cycles.values())
    summary = []
    for phase, (name, _) in PHASES.items():
        cycles = phase_cycles[phase]
        summary.append({"phase": name, "cycles": cycles,
                        "share_percent": (100.0 * cycles / total) if total else 0.0})
    trace = {"traceEvents": events, "displayTimeUnit": "us", "metadata": {
        "clock_hz": clock_hz,
        "cycle_counts_sum_per_phase_across_block_role_records": dict(
            (PHASES[p][0], phase_cycles[p]) for p in PHASES),
        "percentage_denominator": "sum of phase durations across all input block/role records",
    }}
    return trace, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV cycle-boundary log dumped from device memory")
    parser.add_argument("--clock-hz", type=float, required=True,
                        help="core clock in cycles per second")
    parser.add_argument("--output", type=Path, default=Path("reports/reduction_network_trace.json"))
    args = parser.parse_args()
    if args.clock_hz <= 0:
        parser.error("--clock-hz must be positive")
    trace, summary = convert(read_boundaries(args.input), args.clock_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print("| Phase | Cycles | Share |")
    print("|---|---:|---:|")
    for row in summary:
        print(f"| {row['phase']} | {row['cycles']} | {row['share_percent']:.2f}% |")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
