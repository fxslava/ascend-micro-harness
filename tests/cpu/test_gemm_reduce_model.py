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

# AIV0's MTE2 is its own queue, not part of its vector stream. That distinction is the whole
# content of the landing hazard: a burst issued into MTE2 completes independently of the vector
# work, so modelling the two on one in-order queue makes the hazard unreachable and its
# negative control unrejectable -- which is exactly what the first version of this model did,
# and why the two-task CAModel run found the missing fence instead.
Q_AIV0, Q_AIV0_MTE2, Q_MTE3, Q_AIC_MTE2, Q_AIC, Q_AIV1 = range(6)
QUEUES = 6


def schedule(workers, *, no_prologue=False, early_fetch=False, duplicate_ready=False,
             shared_slot_id=False, skip_drain=False, idle_subcore_silent=False,
             idle_subcore_unpaced=False, free_unconditional=False,
             no_landing_fence=False):
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
            # The statistics landing is ONE shared buffer, not a per-slot one: task r+1's MTE2
            # burst into it has to land behind task r's vector reduction of it. On the device
            # that fence is SyncVectorToMte2; here it is this place. Modelling stats as one
            # atomic step is what hid the hazard, and the two-task CAModel run returned inf.
            if not no_landing_fence:
                emit(Q_AIV0_MTE2, 'wait', ('landing_free', 0))
            emit(Q_AIV0_MTE2, 'fetch_stats', gen(r))
            emit(Q_AIV0_MTE2, 'set', ('landing_ready', 0))
            emit(Q_AIV0, 'wait', ('landing_ready', 0))
            emit(Q_AIV0, 'reduce_stats', s, gen(r))
            if not no_landing_fence:
                emit(Q_AIV0, 'set', ('landing_free', 0))
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
        # The landing credit is primed once per worker and retired at its end: it is a
        # feedback cycle like any other, and an unprimed one deadlocks.
        if tasks and not no_landing_fence:
            emit(Q_AIV0, 'set', ('landing_free', 0))
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
        if tasks and not no_landing_fence:
            emit(Q_AIV0_MTE2, 'wait', ('landing_free', 0))
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
    PARTIAL, WEIGHT, PRODUCT, LANDING = 0, SLOTS, 2 * SLOTS, 3 * SLOTS
    start = ((0,) * QUEUES, (0,) * len(keys), (-1,) * (3 * SLOTS + 1), (-1,))
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
            elif op == 'fetch_stats':
                generation, = args
                assert o[LANDING] == -1, 'statistics landing overwritten before its reduction'
                o[LANDING] = generation
            elif op == 'reduce_stats':
                slot, generation = args
                assert o[LANDING] == generation, 'wrong statistics generation reduced'
                assert o[WEIGHT + slot] == -1, 'weight row overwritten before its Mmad'
                o[LANDING] = -1
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


# =================================================================================================
# The dense batched form: M = 16 packed query rows against 16-channel slices
# =================================================================================================
#
# A different schedule, so a different model, sharing this file's explorer and balance check.
#
# THE STAGE
#
# A batch is a run of up to 16 consecutive live tasks. Batch slot s = b % 2 owns an L1
# block-diagonal weight cube and, in each vector subcore's own UB, a product panel. Inside a
# batch the Cube sweeps SLICES -- 16-channel column bands of D -- over its own two-slot ring.
#
#   AIV_c  stats(b)   ONE strided burst for every (row, split) statistics tail of subcore c's
#                     half, then per row m*/w_s/l*, then a 32-byte MTE3 store of w into the
#                     cube's row. BOTH subcores, each its own half, each announcing.
#   AIC    gemm(b)    the cube into L0A once; then per slice: DN2NZ ingest into L1, Load2DV2
#                     into L0B, Mmad into L0C, Fixpipe into BOTH panels (dualDstCtl splits M).
#   AIV_c  finish(b)  its own panel: NormalizeHeads, the finisher, one WriteHeads burst.
#
# WHY BOTH SUBCORES ARE REAL QUEUES HERE, WHICH IS THE POINT OF THE FORM
#
# In the M = 1 model above, AIV1 was a queue that mirrored a flag schedule and carried no data,
# because one Fixpipe row cannot be split. With M = 16 the dual-destination Fixpipe gives each
# subcore eight rows, so both run the same program over different data. The two failures the
# M = 1 model kept as controls -- an AIV1 that never waits, and one that drains without pacing
# the Cube -- are kept here too, because an idle subcore is still reachable at live = 1.
#
# THE FOUR AIC FEEDBACK CYCLES, AND TWO FAULTS THIS MODEL FOUND IN THEM
#
# slice_free (MTE1 -> MTE2), l0b_free (M -> MTE1), l0c_free (FIX -> M) and l0a_free (M -> MTE1)
# are all primed ONCE PER WORKER and drained once per worker. The first version of this schedule
# primed and drained the three inner ones inside each batch, on the theory that a credit which
# never crosses a batch boundary cannot leak across one. That is wrong, and it is SS1a's second
# rule verbatim: each of those cycles is primed on one queue and drained on ANOTHER -- l0c_free
# is primed on FIX and drained on M -- so nothing orders the next batch's prime against this
# batch's drain, FIX runs ahead into batch b + 1 and sets a flag batch b has not yet consumed.
# `prime_ring_per_batch` is that control. Priming per worker removes the boundary entirely: the
# slice ring simply continues across batches, which is 1-safe by the same induction, and
# kGemmSlices is even so the slot sequence stays continuous.
#
# The fifth thing on the M -> MTE1 route is not the kernel's at all: TPipe primes M_MTE1 ids
# 0..2 for its matmul macro at operator entry and waits them at exit, so a kernel that primes
# its own ids there double-SETs a one-bit flag. `runtime_owned_event` is that control, and it is
# here because the first CAModel run of this form hit it while printing worst_fp16_ulp = 0.
SLICES = 2

(B_AIV0, B_AIV0_MTE2, B_AIV0_MTE3, B_AIV1, B_AIV1_MTE2, B_AIV1_MTE3,
 B_AIC_MTE2, B_AIC_MTE1, B_AIC_M, B_AIC_FIX) = range(10)
B_QUEUES = 10
B_VEC = (B_AIV0, B_AIV1)
B_VEC_MTE2 = (B_AIV0_MTE2, B_AIV1_MTE2)
B_VEC_MTE3 = (B_AIV0_MTE3, B_AIV1_MTE3)


def batched_schedule(workers, *, slices=SLICES, no_prologue=False, shared_slot_id=False,
                     skip_drain=False, idle_subcore_silent=False, idle_subcore_unpaced=False,
                     no_landing_fence=False, no_slice_fence=False, no_l0b_fence=False,
                     no_l0c_fence=False, no_l0a_fence=False, runtime_owned_event=False,
                     prime_ring_per_batch=False):
    """Each queue's program for the batched stage. Keyword flags are negative controls."""
    queues = [[] for _ in range(B_QUEUES)]

    def emit(q, *op):
        queues[q].append(op)

    def sid(slot):
        return 0 if shared_slot_id else slot

    aiv1_posts = not (idle_subcore_silent or idle_subcore_unpaced)
    aiv1_waits = not idle_subcore_silent

    # The runtime's own M_MTE1 priming, on the id this schedule would otherwise use for
    # l0b_free. Modelled as a set with no matching wait in the kernel, which is what it is.
    if runtime_owned_event:
        for ss in range(slices):
            emit(B_AIC_M, 'set', ('l0b_free', ss))

    for worker, batches in enumerate(workers):
        def gen(b):
            return worker * 1000 + b

        def stats(b):
            s = b % SLOTS
            for c in (0, 1):
                if c == 1 and not aiv1_posts:
                    continue
                # The statistics landing is ONE shared buffer per subcore, not a per-slot one:
                # batch b+1's burst into it has to land behind batch b's vector reduction of
                # it. On the device that fence is SyncVectorToMte2.
                if not no_landing_fence:
                    emit(B_VEC_MTE2[c], 'wait', ('landing_free', c))
                emit(B_VEC_MTE2[c], 'fetch_stats', c, gen(b))
                emit(B_VEC_MTE2[c], 'set', ('landing_ready', c))
                emit(B_VEC[c], 'wait', ('landing_ready', c))
                emit(B_VEC[c], 'reduce_stats', c, s, gen(b))
                if not no_landing_fence:
                    emit(B_VEC[c], 'set', ('landing_free', c))
                emit(B_VEC[c], 'set', ('weights_staged', c, s))
                emit(B_VEC_MTE3[c], 'wait', ('weights_staged', c, s))
                emit(B_VEC_MTE3[c], 'store_weights', c, s, gen(b))
                emit(B_VEC_MTE3[c], 'set', ('weights_ready', c, sid(s)))

        def gemm(b, batches):
            s = b % SLOTS
            # The mode-2 quorum: one wait per announcing subcore. Both really staged a half.
            emit(B_AIC_MTE1, 'wait', ('weights_ready', 0, sid(s)))
            if aiv1_posts:
                emit(B_AIC_MTE1, 'wait', ('weights_ready', 1, sid(s)))
            # The A cube, into L0A once for the whole batch.
            if not no_l0a_fence:
                emit(B_AIC_MTE1, 'wait', ('l0a_free',))
            emit(B_AIC_MTE1, 'load_a', s, gen(b))
            emit(B_AIC_MTE1, 'set', ('l0a_ready',))
            emit(B_AIC_M, 'wait', ('l0a_ready',))

            # The control: priming the inner ring per batch instead of per worker. Primed on
            # MTE1/M/FIX and drained on MTE2/MTE1/M, so the next batch's prime races this
            # batch's drain.
            if prime_ring_per_batch:
                prime_ring()
            for j in range(slices * 2):
                ss = j % slices
                if not no_slice_fence:
                    emit(B_AIC_MTE2, 'wait', ('slice_free', ss))
                emit(B_AIC_MTE2, 'fetch_slice', ss, gen(b), j)
                emit(B_AIC_MTE2, 'set', ('slice_ready', ss))

                emit(B_AIC_MTE1, 'wait', ('slice_ready', ss))
                if not no_l0b_fence:
                    emit(B_AIC_MTE1, 'wait', ('l0b_free', ss))
                emit(B_AIC_MTE1, 'load_slice', ss, gen(b), j)
                if not no_slice_fence or slice_free_unconditional:
                    emit(B_AIC_MTE1, 'set', ('slice_free', ss))
                emit(B_AIC_MTE1, 'set', ('l0b_ready', ss))

                emit(B_AIC_M, 'wait', ('l0b_ready', ss))
                if not no_l0c_fence:
                    emit(B_AIC_M, 'wait', ('l0c_free', ss))
                emit(B_AIC_M, 'mmad', ss, s, gen(b), j)
                if not no_l0b_fence:
                    emit(B_AIC_M, 'set', ('l0b_free', ss))
                emit(B_AIC_M, 'set', ('l0c_ready', ss))

                emit(B_AIC_FIX, 'wait', ('l0c_ready', ss))
                emit(B_AIC_FIX, 'fix', ss, s, gen(b), j)
                if not no_l0c_fence:
                    emit(B_AIC_FIX, 'set', ('l0c_free', ss))
            if prime_ring_per_batch:
                drain_ring()
            if not no_l0a_fence:
                emit(B_AIC_M, 'set', ('l0a_free',))
            # Both panels are complete only after the last Fixpipe of the batch.
            emit(B_AIC_FIX, 'set', ('product_ready', 0, sid(s)))
            emit(B_AIC_FIX, 'set', ('product_ready', 1, sid(s)))
            static_cast = batches  # the signature mirrors the M = 1 model's
            del static_cast

        def finish(b):
            s = b % SLOTS
            for c in (0, 1):
                if c == 1 and not aiv1_waits:
                    continue
                emit(B_VEC[c], 'wait', ('product_ready', c, sid(s)))
                emit(B_VEC[c], 'finish', c, s, gen(b))
                emit(B_VEC[c], 'set', ('output_staged', c, s))
                emit(B_VEC_MTE3[c], 'wait', ('output_staged', c, s))
                emit(B_VEC_MTE3[c], 'store_out', c, s, gen(b))

        def prime_ring():
            for ss in range(slices):
                if not no_slice_fence:
                    emit(B_AIC_MTE1, 'set', ('slice_free', ss))
                if not no_l0b_fence and not runtime_owned_event:
                    emit(B_AIC_M, 'set', ('l0b_free', ss))
                if not no_l0c_fence:
                    emit(B_AIC_FIX, 'set', ('l0c_free', ss))

        def drain_ring():
            for ss in range(slices):
                if not no_slice_fence:
                    emit(B_AIC_MTE2, 'wait', ('slice_free', ss))
                if not no_l0b_fence:
                    emit(B_AIC_MTE1, 'wait', ('l0b_free', ss))
                if not no_l0c_fence:
                    emit(B_AIC_M, 'wait', ('l0c_free', ss))

        if batches:
            if not no_landing_fence:
                for c in (0, 1):
                    emit(B_VEC[c], 'set', ('landing_free', c))
            if not no_l0a_fence:
                emit(B_AIC_M, 'set', ('l0a_free',))
            if not prime_ring_per_batch:
                prime_ring()
        prologue = 0 if no_prologue else min(batches, SLOTS)
        for b in range(prologue):
            stats(b)
        for b in range(batches):
            gemm(b, batches)
            finish(b)
            if b + SLOTS < batches:
                stats(b + SLOTS)

        if batches:
            if not no_landing_fence:
                for c in (0, 1):
                    emit(B_VEC_MTE2[c], 'wait', ('landing_free', c))
            if not prime_ring_per_batch:
                drain_ring()
            if not no_l0a_fence:
                emit(B_AIC_MTE1, 'wait', ('l0a_free',))
        # The worker boundary rendezvous, as in the M = 1 model: every queue announces, one
        # stream collects all of them, then releases them. An empty worker drains too.
        if not skip_drain:
            for q in range(B_QUEUES):
                if q == B_AIV0:
                    continue
                emit(q, 'set', ('join', q))
                emit(B_AIV0, 'wait', ('join', q))
            for q in range(B_QUEUES):
                if q == B_AIV0:
                    continue
                emit(B_AIV0, 'set', ('resume', q))
                emit(q, 'wait', ('resume', q))

    return tuple(tuple(q) for q in queues)


def batched_explore(queues, slices=SLICES):
    """Generation ownership on every reused buffer of the batched stage."""
    keys = sorted({op[1] for q in queues for op in q if op[0] in ('set', 'wait')})
    index = {key: i for i, key in enumerate(keys)}
    start = ((0,) * B_QUEUES, (0,) * len(keys),
             # weights[slot], landing[core], panel[core][slot], l0a, slice[ss], l0b[ss], l0c[ss]
             (-1,) * (SLOTS + 2 + 2 * SLOTS + 1 + 3 * slices))
    W, LAND, PANEL = 0, SLOTS, SLOTS + 2
    L0A = PANEL + 2 * SLOTS
    SLICE, L0B, L0C = L0A + 1, L0A + 1 + slices, L0A + 1 + 2 * slices
    seen, todo, terminals = {start}, deque([start]), 0

    while todo:
        pcs, flags, owners = todo.popleft()
        progress = False
        for q, program in enumerate(queues):
            if pcs[q] == len(program):
                continue
            op, *args = program[pcs[q]]
            f, o = list(flags), list(owners)
            if op in ('set', 'wait'):
                k = index[args[0]]
                if op == 'wait':
                    if not f[k]:
                        continue
                    f[k] = 0
                else:
                    assert not f[k], f'double SET loses token: {args[0]}'
                    f[k] = 1
            elif op == 'fetch_stats':
                core, generation = args
                assert o[LAND + core] == -1, 'statistics landing overwritten before its reduction'
                o[LAND + core] = generation
            elif op == 'reduce_stats':
                core, slot, generation = args
                assert o[LAND + core] == generation, 'wrong statistics generation reduced'
                o[LAND + core] = -1
                # Both subcores write the SAME cube, disjoint halves, so the slot's generation
                # is set by whichever gets there first and must agree for the other.
                assert o[W + slot] in (-1, generation), 'weight cube half from another batch'
                o[W + slot] = generation
            elif op == 'store_weights':
                core, slot, generation = args
                assert o[W + slot] == generation, 'wrong weight generation staged'
                del core
            elif op == 'load_a':
                slot, generation = args
                assert o[W + slot] == generation, 'wrong weight generation in L0A'
                assert o[L0A] == -1, 'the A cube overwritten before its last Mmad'
                o[W + slot] = -1
                o[L0A] = generation
            elif op == 'fetch_slice':
                ss, generation, j = args
                assert o[SLICE + ss] == -1, 'L1 slice overwritten before its Load2DV2'
                o[SLICE + ss] = generation * 100 + j
            elif op == 'load_slice':
                ss, generation, j = args
                assert o[SLICE + ss] == generation * 100 + j, 'wrong slice generation loaded'
                assert o[L0B + ss] == -1, 'L0B overwritten before its Mmad'
                o[SLICE + ss] = -1
                o[L0B + ss] = generation * 100 + j
            elif op == 'mmad':
                ss, slot, generation, j = args
                assert o[L0B + ss] == generation * 100 + j, 'wrong slice generation in L0B'
                assert o[L0A] == generation, 'wrong A cube generation in L0A'
                assert o[L0C + ss] == -1, 'L0C overwritten before its Fixpipe'
                o[L0B + ss] = -1
                o[L0C + ss] = generation * 100 + j
                # The A cube is read by the Mmad and by nothing else, so its live range ends at
                # the batch's LAST Mmad -- which is where the kernel posts l0a_free, on M. An
                # earlier version of this model released it at the last Fixpipe instead and
                # reported a fault the schedule does not have; the lifetime is the Mmad's.
                if j == slices * 2 - 1:
                    o[L0A] = -1
                del slot
            elif op == 'fix':
                ss, slot, generation, j = args
                assert o[L0C + ss] == generation * 100 + j, 'wrong product generation Fixpiped'
                o[L0C + ss] = -1
                # The last slice of the batch completes both panels. Modelled per slice, so a
                # panel still owned by an unread batch is caught.
                for core in (0, 1):
                    assert o[PANEL + core * SLOTS + slot] in (-1, generation), \
                        'product panel overwritten before its normalise'
                    o[PANEL + core * SLOTS + slot] = generation
                del j
            elif op == 'finish':
                core, slot, generation = args
                assert o[PANEL + core * SLOTS + slot] == generation, 'wrong product generation'
                o[PANEL + core * SLOTS + slot] = -1
            elif op == 'store_out':
                core, slot, generation = args
                del core, slot, generation
            else:
                raise AssertionError(op)
            progress = True
            nxt = list(pcs)
            nxt[q] += 1
            state = (tuple(nxt), tuple(f), tuple(o))
            if state not in seen:
                seen.add(state)
                todo.append(state)
        if not progress:
            assert all(pc == len(p) for pc, p in zip(pcs, queues)), 'deadlock'
            assert not any(flags), 'nonempty final marking: a hardware flag is left set'
            assert all(x == -1 for x in owners), 'a buffer is still owned at the end'
            terminals += 1
    assert terminals, 'no terminal state'
    return len(seen)


def batched_main():
    # Exhaustive over every reachable completion interleaving, for BOUNDED instances: two
    # consecutive workers with the marking carried across, because the stage is primed and
    # drained once per Split-K worker and an outer loop's boundary places are invisible to a
    # model that stops at one pass. The slice sweep is modelled at `slices` slots over
    # 2 * slices iterations -- prime, steady state and drain -- and not at the device's
    # sixteen, which adds generations rather than places.
    #
    # The bound is smaller than the M = 1 model's R = 0..10, and the reason is the ten queues:
    # this stage's reachable state count is 4.1e4 at B = 0 and 1.1e6 at B = 1 over two workers,
    # a factor of 26 per batch, so a two-worker B = 2 is ~3e7 states and does not fit in Python.
    # The coverage is therefore split by what each shape is FOR, and said rather than implied:
    #
    #   two workers, B = 0..1   the worker BOUNDARY, which is what SS1a exists for: the marking
    #                           is carried across and an id left set by one worker is caught by
    #                           the next worker's prime.
    #   one worker, B = 2..4    ring DEPTH and WRAP: B = 2 is the first full steady state of the
    #                           two batch slots, 3 the odd tail, 4 a complete reuse cycle.
    #
    # Neither shape is a claim about every batch count. The device ladder covers depth and wrap
    # at the real kGemmSlices = 16, which this model abstracts to `slices` slots over
    # 2 * slices iterations -- prime, steady state and drain.
    for batches in range(2):
        queues = batched_schedule((batches, batches))
        places = balanced(queues)
        print(f'PASS batched B={batches} two_workers places={places} '
              f'states={batched_explore(queues)}')
    for batches in (2, 3, 4):
        queues = batched_schedule((batches,))
        places = balanced(queues)
        print(f'PASS batched B={batches} one_worker places={places} '
              f'states={batched_explore(queues)}')
    # SS4 asks for real Split-K workers including empty ones, in both orders. An empty worker
    # must still drain: nothing of its own was primed, and the boundary rendezvous still runs.
    for workers in ((1, 0), (0, 1), (1, 0, 1)):
        print(f'PASS batched workers={workers} states={batched_explore(batched_schedule(workers))}')

    negatives = {
        'no_prologue': 'the feedback cycles are never primed',
        'shared_slot_id': 'one cross-core id serves both batch slots',
        'skip_drain': 'the next worker primes an id this one left set',
        'idle_subcore_silent': 'a subcore never consumes its product token',
        'idle_subcore_unpaced': 'a subcore drains but does not pace the Cube',
        'no_landing_fence': 'the next batch overwrites the landing being reduced',
        'no_slice_fence': 'MTE2 refills an L1 slice Load2DV2 has not drained',
        'no_l0b_fence': 'MTE1 refills an L0B region the Mmad has not read',
        'no_l0c_fence': 'the Mmad overwrites an L0C region the Fixpipe has not drained',
        'no_l0a_fence': "the next batch's A cube lands mid-Mmad",
        'runtime_owned_event': "the kernel primes M_MTE1 on an id TPipe's matmul macro owns",
        'prime_ring_per_batch': 'a cycle primed on one queue and drained on another, per batch',
    }
    for option, why in negatives.items():
        try:
            queues = batched_schedule((3, 3), **{option: True})
            batched_explore(queues)
            balanced(queues)
        except AssertionError as e:
            print(f'PASS batched negative {option} ({why}): {e}')
        else:
            raise AssertionError(f'negative control not rejected: {option}')


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
        'no_landing_fence': 'the next task overwrites the landing being reduced',
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
    print('--- the dense batched form ---')
    batched_main()
