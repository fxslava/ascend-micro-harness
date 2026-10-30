#!/usr/bin/env python3
"""Explore every engine interleaving of the decode ring's bounded marked graph.

Flags are one-bit places, never counting semaphores. State includes queue PCs,
the marking and the owner of every live landing/page. SET into an occupied
place is an error, not a disabled transition (hardware would lose that token).
WAIT at zero blocks its destination engine. Scalar emission order constructs
three FIFO queues, matching the device's prefill/prepare/retire/refill loop.
"""
from collections import Counter, deque


def schedule(count, archive=True, unprime=False, duplicate=False, early_scale=False):
    queues = {p: [] for p in ("MTE2", "V", "MTE3")}

    def emit(pipe, op, *args):
        queues[pipe].append((op, *args))

    def flag(op, route, slot):
        source, target = route.split("_")[:2] if route not in ("SCALE", "PACKED") else ("V", "MTE2")
        # Packed and scale feedback have different IDs (0/1 vs 2/3).
        if route == "PACKED":
            source, target = "V", "MTE2"
        emit(source if op == "set" else target, op, route, slot)

    def load(i):
        slot = i % 2
        flag("wait", "PACKED", slot)
        emit("MTE2", "load_packed", slot, i)
        flag("wait", "SCALE", slot)
        emit("MTE2", "load_scale", slot, i)
        flag("set", "MTE2_V", slot)

    for slot in range(2):
        if not unprime:
            flag("set", "PACKED", slot)
            flag("set", "SCALE", slot)
        if duplicate:
            flag("set", "PACKED", slot)
        if archive:
            flag("set", "MTE3_V", slot)
    for i in range(min(count, 2)):
        load(i)
    for tick in range(count + 1):
        if tick < count:
            slot = tick % 2
            flag("wait", "MTE2_V", slot)
            emit("V", "dequant", slot, tick)
            flag("set", "PACKED", slot)
            if early_scale:
                flag("set", "SCALE", slot)
            if archive:
                flag("wait", "MTE3_V", slot)
            emit("V", "gemv", slot, tick)
            if archive:
                flag("set", "V_MTE3", slot)
                flag("wait", "V_MTE3", slot)
                emit("MTE3", "archive", slot, tick)
                flag("set", "MTE3_V", slot)
        if tick:
            slot = (tick - 1) % 2
            emit("V", "recurrence", slot, tick - 1)
            if not early_scale:
                flag("set", "SCALE", slot)
            if tick + 1 < count:
                load(tick + 1)
    for slot in range(2):
        if not unprime:
            flag("wait", "PACKED", slot)
            flag("wait", "SCALE", slot)
        if duplicate:
            flag("wait", "PACKED", slot)
        if archive:
            flag("wait", "MTE3_V", slot)
    return tuple(tuple(q) for q in queues.values())


def explore(queues):
    keys = sorted({cmd[1:] for q in queues for cmd in q if cmd[0] in ("set", "wait")})
    key_index = {key: i for i, key in enumerate(keys)}
    sets = Counter(cmd[1:] for q in queues for cmd in q if cmd[0] == "set")
    waits = Counter(cmd[1:] for q in queues for cmd in q if cmd[0] == "wait")
    assert sets == waits, "unbalanced token census"
    # Owners: packed, scales, intermediate pages, archived score panels.
    start = ((0, 0, 0), (0,) * len(keys), (-1,) * 8, 0)
    pending = deque([start])
    seen = {start}
    terminals = 0
    while pending:
        pcs, marking, owners, retired = pending.popleft()
        progress = False
        for pipe, queue in enumerate(queues):
            if pcs[pipe] == len(queue):
                continue
            op, slot_or_route, value = queue[pcs[pipe]]
            tokens, live = list(marking), list(owners)
            next_retired = retired
            if op in ("set", "wait"):
                index = key_index[(slot_or_route, value)]
                if op == "wait" and tokens[index] == 0:
                    continue
                assert op != "set" or tokens[index] == 0, "double SET loses token"
                tokens[index] = int(op == "set")
            else:
                slot, iteration = slot_or_route, value
                packed, scale, page, panel = slot, 2 + slot, 4 + slot, 6 + slot
                if op == "load_packed":
                    assert live[packed] == -1, "packed overwrite"
                    live[packed] = iteration
                elif op == "load_scale":
                    assert live[scale] == -1, "scale overwritten before recurrence"
                    live[scale] = iteration
                elif op == "dequant":
                    assert live[packed] == iteration and live[page] == -1, "dequant ownership"
                    live[packed], live[page] = -1, iteration
                elif op == "gemv":
                    assert live[page] == iteration and live[panel] == -1, "score overwrite"
                    live[panel] = iteration
                elif op == "archive":
                    assert live[panel] == iteration, "archive read race"
                    live[panel] = -1
                elif op == "recurrence":
                    assert retired == iteration, "online state reordered"
                    assert live[page] == live[scale] == iteration, "recurrence ownership"
                    live[page] = live[scale] = -1
                    # The non-archiving Split-K panel has only a V reader.
                    if not any(k[0] == "MTE3_V" for k in keys):
                        live[panel] = -1
                    next_retired += 1
                else:
                    raise AssertionError(op)
            progress = True
            next_pcs = list(pcs)
            next_pcs[pipe] += 1
            state = (tuple(next_pcs), tuple(tokens), tuple(live), next_retired)
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not progress:
            assert all(pc == len(q) for pc, q in zip(pcs, queues)), "deadlock / unprimed cycle"
            assert not any(marking) and all(x == -1 for x in owners), "token or buffer leak"
            terminals += 1
    assert terminals, "no finite drain"
    return len(seen)


# --------------------------------------------------------------------------
# The Split-K reduction's record ring (ReduceHead in
# kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp).
#
# A second, independent bounded marked graph in the same block, entered only
# after the grid barrier. It has the same shape as the strip ring -- two slots,
# capacity-one credits on both routes, primed before the loop and drained
# after it, including the slot a K below the ring depth never fills -- over a
# different payload: one packed partial-state record per split instead of a
# packed strip, and no archive edge.
#
# Its routes carry IDs of their own (V_MTE2 4/5, MTE2_V 2/3) because the strip
# ring's are audited per ID by section 5; the model names the routes, so what
# it proves is the structure, and the ID disjointness is the kernel's own
# static constants.
def reduce_schedule(splits, unprime=False, no_refill_credit=False, leak=False,
                    merge_without_ready=False):
    queues = {p: [] for p in ("MTE2", "V", "MTE3")}

    def emit(pipe, op, *args):
        queues[pipe].append((op, *args))

    def set_flag(route, slot):
        emit({"FREE": "V", "READY": "MTE2", "EGRESS": "V"}[route], "set", route, slot)

    def wait_flag(route, slot):
        emit({"FREE": "MTE2", "READY": "V", "EGRESS": "MTE3"}[route], "wait", route, slot)

    def fetch(split, slot):
        wait_flag("FREE", slot)
        emit("MTE2", "load_record", slot, split)
        if not merge_without_ready:
            set_flag("READY", slot)

    # The prologue: the reducing block decoded another head, so the gate is
    # re-read and the running state re-seeded before the first record lands.
    emit("MTE2", "load_gate", 0, 0)
    emit("V", "seed", 0, 0)
    # The gate's own place: a third slot of the READY route, which the kernel
    # spells as a distinct MTE2_V id for the same reason.
    set_flag("READY", 2)
    wait_flag("READY", 2)

    for slot in range(2):
        if not unprime:
            set_flag("FREE", slot)
    for split in range(min(splits, 2)):
        fetch(split, split)
    for split in range(splits):
        slot = split % 2
        if not merge_without_ready:
            wait_flag("READY", slot)
        emit("V", "merge", slot, split)
        if not leak:
            set_flag("FREE", slot)
        if split + 2 < splits:
            if no_refill_credit:
                emit("MTE2", "load_record", slot, split + 2)
                set_flag("READY", slot)
            else:
                fetch(split + 2, slot)
    for slot in range(2):
        if not unprime and not leak:
            wait_flag("FREE", slot)
    emit("V", "egress", 0, 0)
    set_flag("EGRESS", 0)
    wait_flag("EGRESS", 0)
    emit("MTE3", "store_context", 0, 0)
    return tuple(tuple(q) for q in queues.values())


def explore_reduce(queues, splits):
    keys = sorted({cmd[1:] for q in queues for cmd in q if cmd[0] in ("set", "wait")})
    key_index = {key: i for i, key in enumerate(keys)}
    sets = Counter(cmd[1:] for q in queues for cmd in q if cmd[0] == "set")
    waits = Counter(cmd[1:] for q in queues for cmd in q if cmd[0] == "wait")
    assert sets == waits, "unbalanced token census"
    # Per-slot record owner, how many splits have been merged, whether the
    # state is seeded and whether the context panel holds the egress result.
    start = ((0, 0, 0), (0,) * len(keys), (-1, -1), 0, False, False)
    pending = deque([start])
    seen = {start}
    terminals = 0
    while pending:
        pcs, marking, owners, merged, seeded, context = pending.popleft()
        progress = False
        for pipe, queue in enumerate(queues):
            if pcs[pipe] == len(queue):
                continue
            op, slot, value = queue[pcs[pipe]]
            tokens, live = list(marking), list(owners)
            next_merged, next_seeded, next_context = merged, seeded, context
            if op in ("set", "wait"):
                index = key_index[(slot, value)]
                if op == "wait" and tokens[index] == 0:
                    continue
                assert op != "set" or tokens[index] == 0, "double SET loses token"
                tokens[index] = int(op == "set")
            elif op == "load_gate":
                pass
            elif op == "seed":
                next_seeded = True
            elif op == "load_record":
                assert live[slot] == -1, "record slot overwritten before its merge"
                live[slot] = value
            elif op == "merge":
                assert seeded, "merged into an unseeded running state"
                assert live[slot] == value, "merge read the wrong record"
                # Splits may merge in any order and still be correct -- the
                # recurrence is associative and commutative -- but the record
                # a merge consumes must be the one its slot holds.
                live[slot] = -1
                next_merged += 1
            elif op == "egress":
                assert merged == splits, "egress before every split merged"
                next_context = True
            elif op == "store_context":
                assert context, "context stored before the egress produced it"
                next_context = False
            else:
                raise AssertionError(op)
            progress = True
            next_pcs = list(pcs)
            next_pcs[pipe] += 1
            state = (tuple(next_pcs), tuple(tokens), tuple(live), next_merged,
                     next_seeded, next_context)
            if state not in seen:
                seen.add(state)
                pending.append(state)
        if not progress:
            assert all(pc == len(q) for pc, q in zip(pcs, queues)), "deadlock / unprimed cycle"
            assert not any(marking) and all(x == -1 for x in owners), "token or buffer leak"
            assert merged == splits and not context, "reduction did not drain"
            terminals += 1
    assert terminals, "no finite drain"
    return len(seen)


def main():
    total = 0
    for archive in (False, True):
        for count in (0, 1, 2, 3, 4, 5, 8, 16, 64):
            states = explore(schedule(count, archive))
            total += states
            print(f"PASS {'AIV' if archive else 'Split-K'} N={count}: {states} reachable states, empty final marking")
    for mutation in ("unprime", "duplicate", "early_scale"):
        try:
            explore(schedule(4, **{mutation: True}))
        except AssertionError as error:
            print(f"PASS rejected negative control: {mutation}: {error}")
        else:
            raise AssertionError(f"accepted unsafe schedule: {mutation}")
    # Every K the kernel accepts, plus the two below the ring depth.
    for splits in (1, 2, 3, 4, 5, 8, 12, 16):
        states = explore_reduce(reduce_schedule(splits), splits)
        total += states
        print(f"PASS reduction K={splits}: {states} reachable states, empty final marking")
    # `merge_without_ready` drops the forward edge AND its SET, so the census
    # still balances: it is the control that has to be caught by ownership
    # rather than by counting, and it is the shape a reduction that trusted
    # program order across two queues would have.
    for mutation in ("unprime", "no_refill_credit", "leak", "merge_without_ready"):
        try:
            explore_reduce(reduce_schedule(5, **{mutation: True}), 5)
        except AssertionError as error:
            print(f"PASS rejected negative control: reduction {mutation}: {error}")
        else:
            raise AssertionError(f"accepted unsafe reduction schedule: {mutation}")
    print(f"PASS {total} states: 1-safe, ordered ownership, liveness, finite drain; 7 negative controls")


if __name__ == "__main__":
    main()
