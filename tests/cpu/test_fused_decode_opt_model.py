"""Capacity-one completion model for the imported 3-slot/2-window schedule.

Five FIFO queues: Cube, two Vector consumers and their two ingress engines.
Stages include their fenced MTE3 completion; GEMMs include fenced Fixpipe
completion. This bounded abstraction does not prove scalar issue/fence lowering;
that remains a device-trace gate. Every reachable completion interleaving is
explored, carrying operand generations and final tokens across consecutive tasks.
"""
from collections import deque


def schedule(counts, early_refill=False, wrong_window=False, duplicate=False):
    queues = [[] for _ in range(5)]
    def emit(q, *op): queues[q].append(op)
    def flag(q, op, kind, role, slot=0): emit(q, op, (kind, role, slot))
    for epoch, count in enumerate(counts):
        def read(role, i):
            slot = i % 3
            if i >= 3 and not early_refill:
                flag(3 + role, 'wait', 'refill', role, slot)
            emit(3 + role, 'read', role, slot, epoch * 1000 + i)
            flag(3 + role, 'set', 'read', role, slot)
        def stage(role, i):
            slot = i % 3
            if i >= 3: flag(1 + role, 'wait', 'free', role, slot)
            flag(1 + role, 'wait', 'read', role, slot)
            emit(1 + role, 'stage', role, slot, epoch * 1000 + i)
            flag(1 + role, 'set', 'ready', role, slot)
            if duplicate: flag(1 + role, 'set', 'ready', role, slot)
        def score(i):
            for role in range(2): flag(0, 'wait', 'ready', role, i % 3)
            emit(0, 'score', i % 3, 0 if wrong_window else i % 2, epoch * 1000 + i)
            for role in range(2): flag(0, 'set', 'score', role, i % 2)
        for role in range(2):
            for i in range(min(count, 3)): read(role, i)
            for i in range(min(count, 2)): stage(role, i)
            for i in range(count):
                flag(1 + role, 'wait', 'score', role, i % 2)
                emit(1 + role, 'softmax', role, i % 3, i % 2, epoch * 1000 + i)
                flag(1 + role, 'set', 'probs', role)
                if i + 3 < count:
                    if not early_refill: flag(1 + role, 'set', 'refill', role, i % 3)
                    read(role, i + 3)
                emit(1 + role, 'decay', role, epoch * 1000 + i)
                if i + 2 < count: stage(role, i + 2)
                flag(1 + role, 'wait', 'context', role)
                emit(1 + role, 'accumulate', role, epoch * 1000 + i)
        if count: score(0)
        for i in range(count):
            if i + 1 < count: score(i + 1)
            for role in range(2): flag(0, 'wait', 'probs', role)
            emit(0, 'context', i % 3, epoch * 1000 + i)
            for role in range(2): flag(0, 'set', 'context', role)
            if i + 3 < count:
                for role in range(2): flag(0, 'set', 'free', role, i % 3)
        # Completion join: next task may not prime a queue before all five
        # queues from this task have retired, including empty workers.
        for q in range(1, 5):
            flag(q, 'set', 'join', q)
            flag(0, 'wait', 'join', q)
        for q in range(1, 5):
            flag(0, 'set', 'resume', q)
            flag(q, 'wait', 'resume', q)
    return tuple(tuple(q) for q in queues)


def explore(queues):
    keys = sorted({op[1] for q in queues for op in q if op[0] in ('set', 'wait')})
    key_index = {key: i for i, key in enumerate(keys)}
    # Ingest and L1: 3 slots per role, scores: 2 per role, probs/context: 1.
    start = ((0,) * 5, (0,) * len(keys), (-1,) * 20, (-1, -1))
    seen, todo, terminals = {start}, deque([start]), 0
    while todo:
        pcs, flags, owners, retired = todo.popleft()
        progress = False
        for q, commands in enumerate(queues):
            if pcs[q] == len(commands): continue
            op, *args = commands[pcs[q]]
            f, o, r = list(flags), list(owners), list(retired)
            if op in ('set', 'wait'):
                k = key_index[args[0]]
                if op == 'wait' and not f[k]: continue
                assert op != 'set' or not f[k], 'double SET loses token'
                f[k] = int(op == 'set')
            elif op in ('read', 'stage'):
                role, slot, generation = args
                ingress, l1 = role * 3 + slot, 6 + role * 3 + slot
                if op == 'read':
                    assert o[ingress] == -1, 'ingress overwrite before scale consumer'
                    o[ingress] = generation
                else:
                    assert o[ingress] == generation and o[l1] == -1, 'L1 overwrite / wrong packed generation'
                    o[l1] = generation
            elif op == 'score':
                slot, window, generation = args
                assert o[6 + slot] == generation, 'wrong K generation'
                for role in range(2):
                    panel = 12 + role * 2 + window
                    assert o[panel] == -1, 'score window overwrite'
                    o[panel] = generation
            elif op == 'softmax':
                role, slot, window, generation = args
                ingress, panel, prob = role * 3 + slot, 12 + role * 2 + window, 16 + role
                assert o[ingress] == o[panel] == generation, 'wrong scale or score generation'
                assert o[prob] == -1, 'probability operand overwrite'
                o[ingress] = o[panel] = -1
                o[prob] = generation
            elif op == 'context':
                slot, generation = args
                assert o[9 + slot] == generation, 'wrong V generation'
                for role in range(2):
                    assert o[16 + role] == generation and o[18 + role] == -1, 'context/probability overwrite'
                    o[16 + role] = o[6 + role * 3 + slot] = -1
                    o[18 + role] = generation
            elif op == 'decay':
                role, generation = args
                assert r[role] % 1000 == generation % 1000 - 1 or generation % 1000 == 0, 'recurrence reordered'
            elif op == 'accumulate':
                role, generation = args
                assert o[18 + role] == generation, 'wrong context generation'
                o[18 + role] = -1
                r[role] = generation
            else: raise AssertionError(op)
            progress = True
            next_pcs = list(pcs); next_pcs[q] += 1
            state = tuple(next_pcs), tuple(f), tuple(o), tuple(r)
            if state not in seen: seen.add(state); todo.append(state)
        if not progress:
            assert all(pc == len(q) for pc, q in zip(pcs, queues)), 'deadlock'
            assert not any(flags) and all(x == -1 for x in owners), 'nonempty drain'
            terminals += 1
    assert terminals, 'no terminal state'
    return len(seen)


if __name__ == '__main__':
    for n in (0, 1, 2, 3, 4, 6, 7, 16):
        print(f'PASS N={n} two_tasks states={explore(schedule((n, n)))}')
    for option in ('early_refill', 'wrong_window', 'duplicate'):
        try: explore(schedule((7,), **{option: True}))
        except AssertionError as e: print(f'PASS negative {option}: {e}')
        else: raise AssertionError(f'negative control not rejected: {option}')
