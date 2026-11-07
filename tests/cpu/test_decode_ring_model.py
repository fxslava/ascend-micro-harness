#!/usr/bin/env python3
"""Explore every engine interleaving of the fused decode's VECTOR REDUCTION ring.

REFINEMENT_OF_COMBAT_KERNELS.md sections 1 and 1a, for the reduction phase of
`kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp`.

WHICH PART OF THE OPERATOR THIS IS, AND WHICH PARTS IT IS NOT

The fused decode has three bounded token/queue structures and one model each:

  * the attention ring -- three ingest slots, two score windows, five queues, carried
    across consecutive tasks:        tests/cpu/test_fused_decode_opt_model.py
  * the Cube GEMM reduction, which `TQ_GEMM_REDUCE=ON` (the default) selects:
                                     tests/cpu/test_gemm_reduce_model.py
  * the VECTOR partial reducer, which `-DTQ_GEMM_REDUCE=OFF` selects instead and which
    `ASCEND_TQ_FUSED_DECODE_REDUCE_PHASE` reaches through `op.Reduce()`:
                                     THIS FILE.

It used to model the two deleted AIV single-token decode kernels -- that operator's
two-slot strip ring with its MTE3 archive edge, and its own Split-K record ring. Those
kernels are gone and the fused operator is the only decode left, so what this file
carries forward is the gate rather than the schedule: the one reduction leg a
`TQ_GEMM_REDUCE` build can select that no other model covers.

THE SCHEDULE, AS `TurboQuantPartialReducer::Reduce` ISSUES IT

One AIV subcore, three queues (V, MTE2, MTE3), from
`op_kernel/vector/turboquant_vector_service.h`:

    V       zero the accumulator and the running (max, sum), then -inf into runMax
    per split s = 0 .. S-1:
      V        SetFlag<V_MTE2>     SyncVectorToMte2(): the WAR fence that stops the next
      MTE2     WaitFlag<V_MTE2>    burst landing on a record still being merged
      MTE2     DataCopy(partAcc, workspace[token, head, s], headSize + kPartialTail)
      MTE2     SetFlag<MTE2_V>     SyncMte2ToVector(): the RAW fence in front of the
      V        WaitFlag<MTE2_V>    merge
      V        Max/Sub/Exp/Mul/Add/BroadcastMul/Add/Adds -- the online merge of record s
    V       NormalizeHeads, the finisher, then Cast into outBuf_
    V       SetFlag<V_MTE3>        WriteHeads()'s SyncVectorToMte3()
    MTE3    WaitFlag<V_MTE3>
    MTE3    DataCopy(outputGm[token, head], outBuf_, headSize)
    MTE3    SetFlag<MTE3_V>        WriteHeads()'s SyncMte3ToVector(): the WAR fence that
    V       WaitFlag<MTE3_V>       makes outBuf_ reusable by the NEXT reduction task

There is ONE landing (`partAcc`/`partState` inside `accBuf_`) and ONE output row
(`outBuf_`), not a ring of two: every split reuses the same bytes, so each of the four
fences above is load-bearing and none of them is a convenience barrier.

SECTION 1a: THE OUTER LOOP IS THE REDUCE-TASK LOOP

`Reduce()` runs once per (token, head) the block owns, so the reused bytes have a WAR
edge at the TASK boundary as well as inside the split loop -- `outBuf_` is the one that
exists only there, because its reader is MTE3 and its next writer is the next task's
Cast on V. A model that stopped at one call could not see it. Every run below therefore
carries the marking and every owner across consecutive tasks, and `no_mte3_v` is the
negative control for exactly that edge.

Flags are one-bit places, never counting semaphores: a SET into an occupied place is an
error here, because hardware would drop that token. A WAIT at zero blocks its
destination engine. Every SET is routed to the queue of the pipe that ISSUES it and
every WAIT to the queue of the pipe that CONSUMES it (section 1a), which for
`SyncEvent<V_MTE2>` puts the SET on V and the WAIT on MTE2.

WHAT THIS BOUNDED ABSTRACTION DOES NOT ESTABLISH

`Init()`'s `PipeBarrier<PIPE_ALL>` is not modelled as a transition; the model instead
STARTS from the empty marking with no live owner, which is the state that barrier
establishes, so what is left unproven is the barrier itself rather than anything after
it. Scalar issue, fence lowering, UB addresses and arithmetic are not here either:
those are the compile-time checks of section 2 and the CAModel gates of sections 3-5.
"""
from collections import Counter, deque

QUEUES = ("V", "MTE2", "MTE3")
# SyncEvent<ROUTE>() issues its SetFlag into the producer's pipe and its WaitFlag into
# the consumer's. turboquant_common.h: SyncVectorToMte2, SyncMte2ToVector,
# SyncVectorToMte3, SyncMte3ToVector.
PRODUCER = {"V_MTE2": "V", "MTE2_V": "MTE2", "V_MTE3": "V", "MTE3_V": "MTE3"}
CONSUMER = {"V_MTE2": "MTE2", "MTE2_V": "V", "V_MTE3": "MTE3", "MTE3_V": "V"}


def schedule(tasks, drop_v_mte2=False, drop_mte2_v=False, drop_v_mte3=False,
             drop_mte3_v=False, duplicate_v_mte2=False):
    """`tasks` holds the split count of each consecutive reduce task on one block."""
    queues = {pipe: [] for pipe in QUEUES}

    def emit(pipe, op, *args):
        queues[pipe].append((op,) + args)

    def fence(route, dropped=False, twice=False):
        if dropped:
            return
        repeats = 2 if twice else 1
        for _ in range(repeats):
            emit(PRODUCER[route], "set", route)
        for _ in range(repeats):
            emit(CONSUMER[route], "wait", route)

    for task, splits in enumerate(tasks):
        emit("V", "zero_state", task, splits)
        for split in range(splits):
            fence("V_MTE2", drop_v_mte2, duplicate_v_mte2)
            emit("MTE2", "load", task, split)
            fence("MTE2_V", drop_mte2_v)
            emit("V", "merge", task, split)
        emit("V", "normalize", task, splits)
        emit("V", "cast", task, splits)
        fence("V_MTE3", drop_v_mte3)
        emit("MTE3", "store", task, splits)
        fence("MTE3_V", drop_mte3_v)
    return tuple(tuple(queues[pipe]) for pipe in QUEUES)


def explore(queues):
    keys = sorted({cmd[1] for q in queues for cmd in q if cmd[0] in ("set", "wait")})
    key_index = {key: i for i, key in enumerate(keys)}
    sets = Counter(cmd[1] for q in queues for cmd in q if cmd[0] == "set")
    waits = Counter(cmd[1] for q in queues for cmd in q if cmd[0] == "wait")
    assert sets == waits, "unbalanced token census"
    # State: the three queue PCs, the marking, the owner of the single workspace
    # landing, the owner of the single output row, how many records the running state
    # has absorbed, which task that state belongs to, and whether it is normalized.
    start = ((0, 0, 0), (0,) * len(keys), None, None, 0, None, False)
    pending = deque([start])
    seen = {start}
    terminals = 0
    while pending:
        pcs, marking, landing, outrow, merged, owner, normalized = pending.popleft()
        progress = False
        for pipe, queue in enumerate(queues):
            if pcs[pipe] == len(queue):
                continue
            op = queue[pcs[pipe]]
            tokens = list(marking)
            next_landing, next_outrow = landing, outrow
            next_merged, next_owner, next_normalized = merged, owner, normalized
            if op[0] in ("set", "wait"):
                index = key_index[op[1]]
                if op[0] == "wait":
                    if tokens[index] == 0:
                        continue
                    tokens[index] = 0
                else:
                    assert tokens[index] == 0, "double SET loses token"
                    tokens[index] = 1
            elif op[0] == "zero_state":
                # The running (max, sum, Acc) belong to one task at a time. Re-seeding
                # them while the previous task's output row is still owed to MTE3 is
                # legal; overwriting the ROW is not, and `cast` is what checks that.
                next_owner, next_merged, next_normalized = op[1], 0, False
            elif op[0] == "load":
                assert next_landing is None, (
                    "workspace landing overwritten before its merge consumed it")
                next_landing = (op[1], op[2])
            elif op[0] == "merge":
                assert next_landing == (op[1], op[2]), (
                    "merge read an unwritten or stale workspace landing")
                assert next_owner == op[1] and not next_normalized, (
                    "merged into another task's running state")
                next_landing = None
                next_merged += 1
            elif op[0] == "normalize":
                assert next_owner == op[1] and next_merged == op[2], (
                    "normalized before every split was merged")
                next_normalized = True
            elif op[0] == "cast":
                assert next_normalized and next_owner == op[1], (
                    "cast an unnormalized accumulator")
                assert next_outrow is None, (
                    "output row overwritten before MTE3 read the previous task's")
                next_outrow = op[1]
            elif op[0] == "store":
                assert next_outrow == op[1], "stored an unwritten output row"
                next_outrow = None
            else:
                raise AssertionError(op[0])
            progress = True
            next_pcs = list(pcs)
            next_pcs[pipe] += 1
            state = (tuple(next_pcs), tuple(tokens), next_landing, next_outrow,
                     next_merged, next_owner, next_normalized)
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not progress:
            assert all(pc == len(q) for pc, q in zip(pcs, queues)), (
                "deadlock / unprimed cycle")
            assert not any(marking), "nonempty final marking: a hardware flag left set"
            assert landing is None and outrow is None, "buffer leak"
            terminals += 1
    assert terminals, "no finite drain"
    return len(seen)


def main():
    total = 0
    # Every split count the host tiling can emit for this reducer (the fused operator's
    # K is 1..8), at one task and at consecutive tasks on the same block, plus the empty
    # reduce task a block gets when its share of (token, head) pairs runs out.
    for tasks in ((0,), (1,), (2,), (3,), (4,), (5,), (8,), (1, 1), (2, 3), (8, 8),
                  (0, 4), (4, 0), (3, 1, 2)):
        states = explore(schedule(tasks))
        total += states
        print(f"PASS reduction tasks={tasks}: {states} reachable states, "
              f"empty final marking")
    controls = ("drop_v_mte2", "drop_mte2_v", "drop_v_mte3", "drop_mte3_v",
                "duplicate_v_mte2")
    for mutation in controls:
        try:
            explore(schedule((3, 2), **{mutation: True}))
        except AssertionError as error:
            print(f"PASS rejected negative control: {mutation}: {error}")
        else:
            raise AssertionError(f"accepted unsafe reduction schedule: {mutation}")
    print(f"PASS {total} states: 1-safe, ordered ownership across the reduce-task "
          f"boundary, liveness, finite drain; {len(controls)} negative controls")


if __name__ == "__main__":
    main()
