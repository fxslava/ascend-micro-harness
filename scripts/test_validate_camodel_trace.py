#!/usr/bin/env python3
"""Failure-path checks for the native JSON validator; synthetic schema only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from validate_camodel_trace import validate


class TraceValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.json"
        self.events = []
        for core, pipes in {
            "core0.cubecore0": ["MTE1", "MTE2", "CUBE", "FIXP"],
            "core0.veccore0": ["MTE2", "MTE3", "RVECEX"],
            "core0.veccore1": ["MTE2", "MTE3", "RVECEX"],
        }.items():
            for i, pipe in enumerate(pipes):
                self.events.extend([
                    {"ph": "M", "pid": core, "tid": pipe, "name": "thread_sort_index", "args": {"sort_index": i}},
                    {"ph": "X", "pid": core, "tid": pipe, "name": "instruction", "ts": i + 0.25, "dur": 0.125},
                ])

    def check_events(self, events):
        self.path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return validate(self.path)

    def test_native_string_lanes_without_cat(self):
        result = self.check_events(self.events)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["core_pipe_event_counts"]), 3)

    def test_rejects_invalid_timestamps_phases_and_durations(self):
        for key, value in (("ts", -1), ("ts", float("nan")), ("ts", "1"), ("ph", ""), ("dur", -1)):
            with self.subTest(key=key, value=value):
                events = copy.deepcopy(self.events)
                events[1][key] = value
                with self.assertRaises(ValueError):
                    self.check_events(events)

    def test_rejects_empty_or_metadata_only(self):
        for events in ([], [e for e in self.events if e["ph"] == "M"]):
            with self.assertRaises(ValueError):
                self.check_events(events)

    def test_rejects_missing_core_pipe_and_metadata(self):
        for events in (
            [e for e in self.events if e["pid"] != "core0.veccore1"],
            [e for e in self.events if e["tid"] != "MTE1"],
            [e for e in self.events if not (e["ph"] == "M" and e["tid"] == "FIXP")],
        ):
            with self.assertRaises(ValueError):
                self.check_events(events)


if __name__ == "__main__":
    unittest.main()
