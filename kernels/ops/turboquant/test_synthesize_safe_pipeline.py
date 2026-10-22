"""Checker regressions; run: python -m unittest discover -s kernels/ops/turboquant."""

import unittest

from synthesize_safe_pipeline import (
    Net, Policy, VerificationError, build, explore, header, partitions,
    synthesize_placements,
)


class ExplorerTests(unittest.TestCase):
    def check(self, net, clears=None, **kwargs):
        return explore(net, clears or {}, progress=False, max_states=200_000, **kwargs)

    def test_full_set_is_error_not_blocking(self):
        n = Net()
        n.seed("pc", "flag")
        n.add("SetFlag", ("pc",), ("flag",))
        with self.assertRaisesRegex(VerificationError, "overwrite"):
            self.check(n)

    def test_raw_assertion_is_not_a_hidden_wait(self):
        n = Net()
        n.seed("pc")
        n.add("read", ("pc",), ("done",), ("valid",), "RAW")
        with self.assertRaisesRegex(VerificationError, "RAW"):
            self.check(n)

    def test_deadlock_and_dirty_termination(self):
        n = Net()
        n.seed("pc")
        n.add("wait", ("pc", "absent"), ("done",))
        n.final = n.bits(["done"])
        with self.assertRaisesRegex(VerificationError, "deadlock"):
            self.check(n)

    def test_finite_clean_drain(self):
        n = Net()
        n.seed("pc", "flag")
        n.add("WaitFlag", ("pc", "flag"), ("done",))
        n.final = n.bits(["done"])
        result = self.check(n)
        self.assertEqual(result["states"], 2)
        self.assertEqual(result["terminal_states"], 1)

    def test_cycle_without_recurrence_is_rejected(self):
        n = Net()
        n.seed("start")
        n.add("leave", ("start",), ("trap",))
        n.add("spin", ("trap",), ("trap",))
        with self.assertRaisesRegex(VerificationError, "cannot drain / return"):
            self.check(n, cyclic=True)

    def test_index_handles_transition_without_precondition(self):
        n = Net()
        n.add("unguarded Set", (), ("flag",))
        with self.assertRaisesRegex(VerificationError, "overwrite"):
            self.check(n)

    def test_state_limit_is_inconclusive(self):
        n = Net()
        n.seed("start")
        n.add("step", ("start",), ("done",))
        n.final = n.bits(["done"])
        with self.assertRaisesRegex(VerificationError, "INCONCLUSIVE"):
            explore(n, {}, max_states=1, progress=False)

    def test_short_and_odd_tail_workloads(self):
        for chunks in (1, 2, 3):
            with self.subTest(chunks=chunks):
                n, clears = build(chunks=chunks)
                result = self.check(n, clears)
                self.assertEqual(result["terminal_states"], 1)
                self.assertEqual(result["safe_reachable_percent"], 100)

    def test_mutation_really_has_trace(self):
        n, clears = build(Policy(fix_drain=False), chunks=1)
        with self.assertRaises(VerificationError) as caught:
            self.check(n, clears)
        self.assertIn("drain", str(caught.exception))
        self.assertTrue(caught.exception.trace)

    def test_indexed_bfs_matches_naive_all_transition_scan(self):
        # Independent, deliberately simple enumerator verifies that the input
        # place index does not omit an interleaving or double-count an edge.
        n, clears = build(chunks=2)
        todo, seen, edges = [n.initial], {n.initial}, 0
        for marking in todo:
            for i, t in enumerate(n.transitions):
                if marking & t.pre != t.pre:
                    continue
                self.assertEqual(marking & t.require, t.require)
                remaining = marking & ~(t.pre | clears.get(i, 0))
                self.assertFalse(remaining & t.post)
                nxt = remaining | t.post
                self.assertTrue(all((nxt & g).bit_count() == 1 for g in n.groups))
                edges += 1
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        optimized = self.check(n, clears)
        self.assertEqual((optimized["states"], optimized["edges"]), (len(seen), edges))

    def test_queue_resource_p_invariants(self):
        # Linear place invariants prove exactly one idle/work/done token per
        # queue for every firing sequence, independently of BFS enumeration.
        n, clears = build(cyclic=True)
        groups = []
        for pipe, prefix in (("MTE2", "MTE2"), ("M", "M"), ("FIX", "FIX")):
            groups.append([f"AIC.PIPE_{pipe}.IDLE"] +
                          [f"{prefix}.{phase}.{c}" for c in range(4) for phase in ("work", "done")])
        for owner in range(2):
            groups.append([f"AIV{owner}.PIPE_V.IDLE"] +
                          [f"V.{phase}.{c}" for c in range(owner, 4, 2)
                           for phase in ("read", "read_done", "pack", "pack_done")])
            groups.append([f"AIV{owner}.PIPE_MTE3.IDLE"] +
                          [f"MTE3.{phase}.{c}" for c in range(owner, 4, 2)
                           for phase in ("work", "done")])
        for names in groups:
            self.assertTrue(all(name in n.places for name in names))
            group = n.bits(names)
            self.assertEqual((n.initial & group).bit_count(), 1)
            for i, t in enumerate(n.transitions):
                consumed = (t.pre | clears.get(i, 0)) & group
                produced = t.post & group
                self.assertEqual(consumed.bit_count(), produced.bit_count(), t.name)

    def test_no_execution_cycle_can_omit_a_transition(self):
        n, clears = build(cyclic=True)
        rows = []
        for bit in n.places.values():
            row = [int(bool(t.post & bit)) - int(bool((t.pre | clears.get(i, 0)) & bit))
                   for i, t in enumerate(n.transitions)]
            self.assertEqual(sum(row), 0)  # all-ones is an exact integer T-invariant
            if any(row):
                rows.append([x % 101 for x in row])
        rank = 0
        for col in range(len(n.transitions)):
            pivot = next((r for r in range(rank, len(rows)) if rows[r][col]), None)
            if pivot is None:
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            scale = pow(rows[rank][col], -1, 101)
            rows[rank] = [(x * scale) % 101 for x in rows[rank]]
            for r in range(rank + 1, len(rows)):
                scale = rows[r][col]
                if scale:
                    rows[r] = [(a - scale * b) % 101 for a, b in zip(rows[r], rows[rank])]
            rank += 1
        # Rank over GF(101) lower-bounds rational rank; the integer null vector
        # upper-bounds it by T-1. Equality proves a one-dimensional nullspace.
        # Any nonempty execution cycle must therefore fire EVERY transition
        # equally often: there is no cycle that starves a core/engine/flag.
        self.assertEqual(rank, len(n.transitions) - 1)


class SynthesisTests(unittest.TestCase):
    def test_complete_partition_enumeration(self):
        choices = list(partitions(4))
        self.assertEqual(len(choices), 15)
        self.assertEqual(len(set(choices)), 15)
        self.assertEqual(sum(max(c) == 2 for c in choices), 6)

    def test_latest_legal_wait_search(self):
        selected, choices = synthesize_placements(Policy())
        self.assertEqual(len(choices), 9)
        self.assertEqual(sum(c["legal"] for c in choices), 4)
        self.assertEqual((selected.free_position, selected.ready_position), ("late", "late"))

    def test_endpoint_ids_and_quotas(self):
        policy = Policy()
        for owner in range(2):
            ids = {policy.flag(owner, slot, ready) for slot in range(2) for ready in (True, False)}
            self.assertEqual(ids, set(range(owner * 16, owner * 16 + 4)))
        self.assertIn("AivLocalId(kFlagProductReady<C>)", header(policy))


if __name__ == "__main__":
    unittest.main()
