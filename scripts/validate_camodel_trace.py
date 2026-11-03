#!/usr/bin/env python3
"""Validate an official aggregate CAModel Chrome trace without parsing dumps.

CANN 9.2 uses string pid/tid values; pipe identity is in tid, while cat is
optional and is primarily used for cross-core flows. Preserve this native schema.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def validate(path):
    raw = path.read_bytes()
    trace = json.loads(raw)
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list) or not events:
        raise ValueError("traceEvents must be a non-empty array")
    phases = Counter()
    metadata = set()
    lanes = defaultdict(Counter)
    categories = Counter()
    timestamps = []
    durations = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("each event must be an object")
        phase = event.get("ph")
        if phase not in {"M", "X", "B", "E", "i", "I", "C", "s", "t", "f", "b", "n", "e", "R", "N", "O", "D"}:
            raise ValueError(f"invalid or empty phase: {phase!r}")
        phases[phase] += 1
        for key in ("pid", "tid"):
            if not isinstance(event.get(key), (str, int)) or isinstance(event[key], bool):
                raise ValueError(f"invalid {key}: {event.get(key)!r}")
        if phase == "M":
            if not event.get("name") or not isinstance(event.get("args"), dict) or not event["args"]:
                raise ValueError("metadata must have name and non-empty args")
            metadata.add((event["pid"], event["tid"]))
            continue
        ts = event.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 0:
            raise ValueError(f"invalid timestamp: {ts!r}")
        timestamps.append(ts)
        if phase == "X":
            duration = event.get("dur")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                raise ValueError(f"invalid duration: {duration!r}")
            durations.append(duration)
            if duration > 0:
                lanes[event["pid"]][event["tid"]] += 1
        if "cat" in event:
            if not isinstance(event["cat"], str) or not event["cat"]:
                raise ValueError("category must be non-empty when present")
            categories[event["cat"]] += 1
    if not phases["M"] or not durations or max(durations) <= 0 or not timestamps or max(timestamps) <= min(timestamps):
        raise ValueError("missing metadata or positive execution timeline")
    cores = set(lanes)
    if not any("cubecore" in str(core) for core in cores) or sum("veccore" in str(core) for core in cores) < 2:
        raise ValueError("expected an AIC and its two AIV cores with execution events")
    for core, pipes in lanes.items():
        for pipe in pipes:
            if (core, pipe) not in metadata:
                raise ValueError(f"missing lane metadata: {core}/{pipe}")
    pipes = {pipe for core in lanes.values() for pipe in core}
    for alternatives in ({"MTE1"}, {"MTE2"}, {"MTE3"}, {"CUBE", "MMAD"}, {"FIXP", "FIXPIPE"}, {"VECTOR", "RVECEX"}):
        if not pipes & alternatives:
            raise ValueError(f"missing execution pipe: {sorted(alternatives)}")
    return {
        "status": "PASS", "sha256": hashlib.sha256(raw).hexdigest(),
        "events": len(events), "phases": dict(phases),
        "timestamp_range_us": [min(timestamps), max(timestamps)],
        "positive_duration_events": sum(d > 0 for d in durations),
        "core_pipe_event_counts": {core: dict(counts) for core, counts in lanes.items()},
        "flow_categories": dict(categories),
        "note": "Native CANN pipe categories are tid lanes; cat is optional.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.trace)
    except (ValueError, OSError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    output = json.dumps(result, indent=2) + "\n"
    if args.summary:
        args.summary.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
