"""Capacity-one token/queue model of the Cube GEMV Split-K reduction stage.

§1 and §1a of REFINEMENT_OF_COMBAT_KERNELS.md for the new stage, in the shape
tests/cpu/test_fused_decode_opt_model.py uses for the attention ring: in-order engine queues,
a capacity-one place per (route, id), generation ownership on every reused buffer, and every
reachable completion interleaving explored.

THE STAGE

Reduction task r combines S split partials for one (token, head). Slot s = r % 2 owns an L1
weight row, an L1 partial cube and a UB product landing.

    AIC MTE2      fetch(r)    DN2NZ GM -> L1 partial slot s
    AIV0          stats(r)    the S (max, sum) lanes, m* = max m_s, w_s = exp(m_s - m*),
                              l* = sum w_s l_s; MTE3 the weight row into L1 slot s
    AIC           gemm(r)     Load2DV2 A and B, Mmad(m=1, k=S, n=D), Fixpipe -> UB slot s
    AIV0          finish(r)   normalise UB slot s by 1/l*, the shared finisher, MTE3 out

Five queues, because a route belongs to the queue that ISSUES it and not to the one it is
named after (§1a): AIV0 vector, AIV0 MTE3, AIC MTE2, AIC (MTE1/M/FIX), and AIV1.

WHY AIV1 IS A QUEUE AND NOT A FOOTNOTE

Subcore 0 owns this prototype's arithmetic: M = 1 is one Fixpipe row, so there is nothing for
a dual-destination split to divide and the second subcore does no combine. It is still a
queue, for two reasons this model found rather than assumed:

  * CrossCoreSetFlag<2> releases BOTH vector subcores, so the Cube's one announcement posts a
    token to each. An AIV1 that never waits leaves a hardware flag set at the end of the phase.
  * Draining those tokens passively is NOT enough. With AIV1 unconstrained, the Cube is paced
    only by subcore 0, so it can reach gemm(r + 2) while AIV1 still owes its wait for gemm(r),
    and the one-bit flag drops the second SET. AIV1 therefore mirrors subcore 0's flag
    schedule -- it posts the same weights_ready credit, carrying no data -- and the Cube waits
    once per subcore per task, which is exactly the `for role in range(2)` idiom the attention
    ring already uses. Both failures are kept as negative controls below.

Cross-core places, per slot s and per subcore c:

    weights_ready[c][s]   AIV_c MTE3 -> AIC   initially 0, primed by the prologue
    product_ready[c][s]   AIC FIX    -> AIV_c initially 0

and one intra-core place per slot on the AIC, partial_free[s] (MTE1 -> MTE2), which stops the
fetch of r + 2 from overwriting the cube the Mmad of r is still loading. That credit is posted
only when a later fetch will consume it: posting it unconditionally balances nothing, because
a short worker then leaves the id set for the next worker's prologue to find.

WHAT IS AND IS NOT PROVEN HERE

Explored exhaustively for bounded instances: R = 0..10 over two consecutive workers, carrying
the marking across the boundary, because the stage is primed and drained once per Split-K
worker and an outer loop's boundary places are invisible to a model that stops at one pass.
A host model is not CAModel evidence. This one says nothing about scalar issue or fence
lowering; the recycled-id fence of §6.6 is a source audit plus a device trace gate, and tick
order per id is the trace's job, not this model's.
"""
from collections import deque

SLOTS = 2
SUBCORES = 2

Q_AIV0, Q_MTE3, Q_AIC_MTE2, Q_AIC, Q_AIV1 = range(5)
QUEUES = 5


def schedule(workers, *, no_prologue=False, early_fetch=False, duplicate_ready=False,
             shared_slot_id=False, skip_drain=False, idle_subcore_silent=False,
             idle_subcore_unpaced=False, free_unconditional=False):
    """Each queue's program. Keyword flags are negative controls."""
    queues = [[] for _ in range(QUEUES)]

    def emit(q, *op):
        queues[q].append(op)

    # One cross-core id per slot, so neither core can be released by a token another task's
    # handshake posted. Collapsing them is a negative control.
    def sid(slot):
        return 0 if shared_slot_id else slot

    # The two broken variants of AIV1: silent (never waits) and unpaced (waits but posts no
    # credit, so nothing holds the Cube behind it).
    aiv1_posts = not (idle_subcore_silent or idle_subcore_unpaced)
    aiv1_waits = not idle_subcore_silent
    # The Cube's announcement is a mode-2 SET: it posts to both subcores whatever AIV1 does.
    aic_posts_sub1 = True

    for worker, tasks in enumerate(workers):
        def gen(r):
            # Distinct per worker, so a stale generation cannot pass as a fresh one.
            return worker * 1000 + r

        def fetch(r):
            s = r % SLOTS
            if r >= SLOTS and not early_fetch:
                emit(Q_AIC_MTE2, 'wait', ('partial_free', s))
            emit(Q_AIC_MTE2, 'fetch', s, gen(r))
            emit(Q_AIC_MTE2, 'set', ('partial_ready', s))

        def stats(r):
            s = r % SLOTS
            emit(Q_AIV0, 'stats', s, gen(r))
            emit(Q_AIV0, 'set', ('weights_staged', s))
            emit(Q_MTE3, 'wait', ('weights_staged', s))
            emit(Q_MTE3, 'store_weights', s, gen(r))
            emit(Q_MTE3, 'set', ('weights_ready', 0, sid(s)))
            if aiv1_posts:
                emit(Q_AIV1, 'set', ('weights_ready', 1, sid(s)))

        def gemm(r, tasks):
            s = r % SLOTS
            emit(Q_AIC, 'wait', ('weights_ready', 0, sid(s)))
            if aiv1_posts:
                emit(Q_AIC, 'wait', ('weights_ready', 1, sid(s)))
            emit(Q_AIC, 'wait', ('partial_ready', s))
            emit(Q_AIC, 'gemm', s, gen(r))
            # Freeing the cube before announcing the product keeps MTE2 ahead of the Mmad.
            # Posting it after instead is slower, not wrong, so that is not a control here.
            if r + SLOTS < tasks or free_unconditional:
                emit(Q_AIC, 'set', ('partial_free', s))
            emit(Q_AIC, 'set', ('product_ready', 0, sid(s)))
            if duplicate_ready:
                emit(Q_AIC, 'set', ('product_ready', 0, sid(s)))
            if aic_posts_sub1:
                emit(Q_AIC, 'set', ('product_ready', 1, sid(s)))

        def finish(r):
            s = r % SLOTS
            emit(Q_AIV0, 'wait', ('product_ready', 0, sid(s)))
            emit(Q_AIV0, 'finish', s, gen(r))
            emit(Q_AIV0, 'set', ('output_staged', s))
            emit(Q_MTE3, 'wait', ('output_staged', s))
            emit(Q_MTE3, 'store_out', s, gen(r))
            if aiv1_waits:
                emit(Q_AIV1, 'wait', ('product_ready', 1, sid(s)))

        # Prologue: prime both feedback cycles for the first min(R, 2) tasks. These sets are
        # the only ones carrying no preceding wait, which is also the one fence exemption
        # §6.6 allows on a recycled id.
        prologue = 0 if no_prologue else min(tasks, SLOTS)
        for r in range(prologue):
            fetch(r)
            stats(r)

        for r in range(tasks):
            gemm(r, tasks)
            finish(r)
            # Only now may slot r % 2 be refilled: finish(r) has read UB product slot r % 2,
            # so the weight store for r + 2 is what releases it.
            if r + SLOTS < tasks:
                fetch(r + SLOTS)
                stats(r + SLOTS)

        # Worker boundary. A per-queue SetFlag/WaitFlag<x_S> pair stalls its own scalar stream
        # but orders NOTHING between two engines, so modelled as a self-pair the exploration
        # walks MTE2 straight into the next worker while this one's Mmad is outstanding. The
        # real boundary is the rendezvous DrainFusedDecodeTask performs: every queue
        # announces, one stream collects all of them, then releases them. It crosses cores
        # because the Cube is the only core writing the vector cores' landings. An empty
        # worker drains too -- "this queue is empty" is exactly what the boundary needs
        # established.
        if not skip_drain:
            for q in range(QUEUES):
                if q == Q_AIV0:
                    continue
                emit(q, 'set', ('join', q))
                emit(Q_AIV0, 'wait', ('join', q))
            for q in range(QUEUES):
                if q == Q_AIV0:
                    continue
                emit(Q_AIV0, 'set', ('resume', q))
                emit(q, 'wait', ('resume', q))

    return tuple(tuple(q) for q in queues)


def explore(queues):
    keys = sorted({op[1] for q in queues for op in q if op[0] in ('set', 'wait')})
    index = {key: i for i, key in enumerate(keys)}
    PARTIAL, WEIGHT, PRODUCT = 0, SLOTS, 2 * SLOTS
    start = ((0,) * QUEUES, (0,) * len(keys), (-1,) * (3 * SLOTS), (-1,))
    seen, todo, terminals = {start}, deque([start]), 0

    while todo:
        pcs, flags, owners, retired = todo.popleft()
        progress = False
        for q, program in enumerate(queues):
            if pcs[q] == len(program):
                continue
            op, *args = program[pcs[q]]
            f, o, r = list(flags), list(owners), list(retired)
            if op in ('set', 'wait'):
                k = index[args[0]]
                if op == 'wait':
                    if not f[k]:
                        continue
                    f[k] = 0
                else:
                    assert not f[k], f'double SET loses token: {args[0]}'
                    f[k] = 1
            elif op == 'fetch':
                slot, generation = args
                assert o[PARTIAL + slot] == -1, 'partial cube overwritten before its Mmad'
                o[PARTIAL + slot] = generation
            elif op == 'stats':
                slot, generation = args
                assert o[WEIGHT + slot] == -1, 'weight row overwritten before its Mmad'
                o[WEIGHT + slot] = generation
            elif op == 'store_weights':
                slot, generation = args
                assert o[WEIGHT + slot] == generation, 'wrong weight generation staged'
            elif op == 'gemm':
                slot, generation = args
                assert o[PARTIAL + slot] == generation, 'wrong partial generation in L0B'
                assert o[WEIGHT + slot] == generation, 'wrong weight generation in L0A'
                assert o[PRODUCT + slot] == -1, 'product landing overwritten before its read'
                o[PARTIAL + slot] = o[WEIGHT + slot] = -1
                o[PRODUCT + slot] = generation
            elif op == 'finish':
                slot, generation = args
                assert o[PRODUCT + slot] == generation, 'wrong product generation'
                # The combines are independent, but the model still pins the order the
                # schedule issues them in, so a reordered retire is caught rather than
                # excused as "they commute".
                assert r[0] < generation, 'reduction task retired out of order'
                o[PRODUCT + slot] = -1
                r[0] = generation
            elif op == 'store_out':
                slot, generation = args
                assert r[0] >= generation, 'output burst ahead of its normalise'
            else:
                raise AssertionError(op)
            progress = True
            nxt = list(pcs)
            nxt[q] += 1
            state = (tuple(nxt), tuple(f), tuple(o), tuple(r))
            if state not in seen:
                seen.add(state)
                todo.append(state)
        if not progress:
            assert all(pc == len(q) for pc, q in zip(pcs, queues)), 'deadlock'
            assert not any(flags), 'nonempty final marking: a hardware flag is left set'
            assert all(x == -1 for x in owners), 'a buffer is still owned at the end'
            terminals += 1
    assert terminals, 'no terminal state'
    return len(seen)


def balanced(queues):
    """Every place is consumed as often as it is produced: the per-id balance half of §6.2.
    Tick order per id is the device trace's gate, not this one's."""
    posts, takes = {}, {}
    for program in queues:
        for op, *args in program:
            if op == 'set':
                posts[args[0]] = posts.get(args[0], 0) + 1
            elif op == 'wait':
                takes[args[0]] = takes.get(args[0], 0) + 1
    bad = {k: (posts.get(k, 0), takes.get(k, 0))
           for k in set(posts) | set(takes) if posts.get(k, 0) != takes.get(k, 0)}
    assert not bad, f'unbalanced places (set, wait): {bad}'
    return len(posts)


def main():
    for tasks in range(11):
        queues = schedule((tasks, tasks))
        places = balanced(queues)
        print(f'PASS R={tasks} two_workers places={places} states={explore(queues)}')
    # §4 asks for real Split-K workers including empty ones, in both orders.
    for workers in ((4, 0), (0, 4), (1, 0, 5), (7, 1)):
        print(f'PASS workers={workers} states={explore(schedule(workers))}')

    negatives = {
        'no_prologue': 'the feedback cycles are never primed',
        'early_fetch': 'MTE2 refills a cube the Mmad has not loaded',
        'duplicate_ready': 'a second SET on a one-bit flag',
        'shared_slot_id': 'one cross-core id serves both slots',
        'skip_drain': 'the next worker primes an id this one left set',
        'idle_subcore_silent': 'the idle subcore never consumes its token',
        'idle_subcore_unpaced': 'the idle subcore drains but does not pace the Cube',
        'free_unconditional': 'a short worker leaves its cube credit set',
    }
    for option, why in negatives.items():
        try:
            queues = schedule((5, 5), **{option: True})
            explore(queues)
            balanced(queues)
        except AssertionError as e:
            print(f'PASS negative {option} ({why}): {e}')
        else:
            raise AssertionError(f'negative control not rejected: {option}')


if __name__ == '__main__':
    main()
