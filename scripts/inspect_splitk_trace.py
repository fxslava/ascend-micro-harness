#!/usr/bin/env python3
"""Summarize kernel and explicitly labeled phase durations from an msprof JSON trace.

Chrome/Perfetto trace timestamps are microseconds by default. If a vendor timeline
export uses another unit, select it with --unit ns|us|ms. Phase values are reported
as the union of matching event intervals, so overlapping workers do not inflate the
wall-time duration. This helper does not infer an unlabeled SyncAll or GEMM phase.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _walk_events(node: Any):
    if isinstance(node, dict):
        if _first(node, "name", "eventName", "event_name") is not None and _first(
            node, "ts", "timestamp", "start", "startTime", "start_time"
        ) is not None:
            yield node
            return
        for value in node.values():
            yield from _walk_events(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_events(value)


def _event_label(event: dict[str, Any]) -> str:
    pieces = [str(_first(event, "name", "eventName", "event_name") or "")]
    pieces.append(str(event.get("cat", event.get("category", ""))))
    for key in ("args", "argsMap", "metadata", "data"):
        value = event.get(key)
        if isinstance(value, dict):
            pieces.extend(str(v) for v in value.values() if isinstance(v, (str, int, float)))
    for key in ("kernelName", "kernel_name", "taskName", "task_name", "type"):
        if key in event:
            pieces.append(str(event[key]))
    return " ".join(pieces).lower()


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _intervals(events: list[dict[str, Any]], unit: str) -> list[tuple[float, float, str]]:
    factor = {"ns": 0.001, "us": 1.0, "ms": 1000.0}[unit]
    result: list[tuple[float, float, str]] = []
    open_events: dict[tuple[str, str, str], float] = {}
    for event in events:
        label = _event_label(event)
        name = str(_first(event, "name", "eventName", "event_name") or "event")
        start = _number(_first(event, "ts", "timestamp", "start", "startTime", "start_time"))
        if start is None:
            continue
        start *= factor
        duration = _number(_first(event, "dur", "duration", "durUs", "duration_us"))
        phase = str(event.get("ph", event.get("phase", "X")))
        if duration is not None:
            end = start + max(0.0, duration * factor)
            result.append((start, end, label))
        elif phase.lower() in {"b", "begin", "start"}:
            key = (str(event.get("pid", "")), str(event.get("tid", "")), name)
            open_events[key] = start
        elif phase.lower() in {"e", "end", "finish"}:
            key = (str(event.get("pid", "")), str(event.get("tid", "")), name)
            begin = open_events.pop(key, None)
            if begin is not None:
                result.append((begin, max(begin, start), label))
        else:
            end = _number(_first(event, "end", "endTime", "end_time"))
            if end is not None:
                result.append((start, max(start, end * factor), label))
    return result


def _union_duration(intervals: list[tuple[float, float, str]]) -> float:
    spans = sorted((start, end) for start, end, _ in intervals if end >= start)
    if not spans:
        return 0.0
    total = 0.0
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _display(value: float | None) -> str:
    return "N/A (not labeled in trace)" if value is None else f"{value:.3f}"


def summarize(path: Path, unit: str) -> int:
    trace_path = path
    if path.is_dir():
        candidates = [path / "msprof_trace.json", *sorted(path.rglob("msprof_trace.json"))]
        trace_path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not trace_path.is_file():
        print(f"error: no msprof_trace.json or JSON trace file at {path}", file=sys.stderr)
        return 2
    try:
        with trace_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read JSON trace {trace_path}: {exc}", file=sys.stderr)
        return 2

    events = list(_walk_events(document))
    intervals = _intervals(events, unit)
    kernel_re = re.compile(r"turboquant.*(decode|split.?k)|decode.*split.?k|fused.?decode")
    kernel_events = [interval for interval in intervals if kernel_re.search(interval[2])]
    if not kernel_events:
        kernel_events = [interval for interval in intervals if "kernel" in interval[2] and "memcpy" not in interval[2]]
    total_kernel = _union_duration(kernel_events) if kernel_events else None

    phase_patterns = {
        "Attention steady tile pipeline": re.compile(r"attention|attn|tile.?pipeline|steady.?phase"),
        "SyncAll barrier": re.compile(r"sync.?all|barrier"),
        "Transposed Cube GEMM reduction": re.compile(r"reduce.?gemm|gemm.?reduce|transposed.?cube|cube.?reduce"),
    }
    print(f"Trace: {trace_path}")
    print(f"Time unit: {unit} (reported durations are microseconds)")
    print("\n| Measurement | Duration (us) |")
    print("|---|---:|")
    print(f"| Kernel wall duration | {_display(total_kernel)} |")
    for label, pattern in phase_patterns.items():
        matching = [interval for interval in intervals if pattern.search(interval[2])]
        value = _union_duration(matching) if matching else None
        print(f"| {label} | {_display(value)} |")
    print("\nPhase durations require trace events or markers carrying phase names; missing phases are left unreported.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="msprof_trace.json, timeline JSON, or a report directory")
    parser.add_argument("--unit", choices=("ns", "us", "ms"), default="us",
                        help="timestamp unit in the input JSON (default: us for Chrome/Perfetto JSON)")
    args = parser.parse_args()
    return summarize(args.trace, args.unit)


if __name__ == "__main__":
    raise SystemExit(main())
