#!/usr/bin/env python3
"""Negative trace fixtures: aggregate balance must never imply 1-safety."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_decode_ring_trace import places


def row(op, index, flag=2, tick=None):
    return dict(mnemonic=op, id=index, tick=index if tick is None else tick,
                tail=f"PIPE:VEC, TRIGGER PIPE:MTE2, FLAG ID:{hex(flag)}")


class TraceAudit(unittest.TestCase):
    def test_alternation_and_hex_ids(self):
        rows = [row("SET_FLAG", 0), row("WAIT_FLAG", 1),
                row("SET_FLAG", 2), row("WAIT_FLAG", 3)]
        self.assertEqual(places(rows, lambda r: r["id"])[("VEC", "MTE2", 2)], 2)

    def test_balanced_but_overfull(self):
        rows = [row("SET_FLAG", 0), row("SET_FLAG", 1),
                row("WAIT_FLAG", 2), row("WAIT_FLAG", 3)]
        with self.assertRaisesRegex(ValueError, "double SET"):
            places(rows, lambda r: r["id"])

    def test_unprimed_wait(self):
        with self.assertRaisesRegex(ValueError, "WAIT without token"):
            places([row("WAIT_FLAG", 0), row("SET_FLAG", 1)], lambda r: r["id"])

    def test_leaked_credit(self):
        with self.assertRaisesRegex(ValueError, "nonempty final marking"):
            places([row("SET_FLAG", 0)], lambda r: r["id"])

    def test_reserved_id(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            places([row("SET_FLAG", 0, flag=6)], lambda r: r["id"])

    def test_completion_order_is_independent(self):
        rows = [row("SET_FLAG", 0, tick=0), row("WAIT_FLAG", 1, tick=4),
                row("SET_FLAG", 2, tick=2), row("WAIT_FLAG", 3, tick=6)]
        places(rows, lambda r: r["id"])
        with self.assertRaisesRegex(ValueError, "double SET"):
            places(rows, lambda r: (r["tick"], r["id"]))


if __name__ == "__main__":
    unittest.main()
