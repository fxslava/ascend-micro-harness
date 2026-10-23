# Safe pipeline synthesis

Run from the repository root with Python 3.10 or newer:

```powershell
python kernels/ops/turboquant/synthesize_safe_pipeline.py
```

The script uses only the standard library. It writes `safe_pipeline_flags.hpp`
and `safe_pipeline_proof.json` beside itself after all proofs and negative
controls pass. `--output-dir`, `--chunks`, and `--max-states` override the output
directory, finite workload size, and per-exploration resource limit. Limit
exhaustion returns nonzero and reports **INCONCLUSIVE**, never a partial PASS.
The JSON includes shortest BFS counterexamples for rejected assignments and
mutations, reachable-state/edge counts, and the script's SHA-256.

The full explicit exploration is substantial: the modulo graph contains
41,844,109 states and 250,476,791 edges. The reference Windows/Python run used
about 9.4 GiB peak working memory and about 18 minutes for the full proof
(947.427 seconds modulo, 140.106 seconds finite). The separate regression
suite below runs in about a second.

## What is proved

The generated protocol uses two READY and two FREE channels per AIV, shared
across iterations through a complete ownership handshake:

```text
initial FREE credit
  -> AIC Wait FREE -> Fixpipe -> FIX_S Set/Wait -> Set READY
  -> AIV Wait READY -> last landing read -> V_S Set/Wait -> Set FREE
```

READY and FREE are separate logical channels. Two initial FREE credits permit
both landing slots to be occupied concurrently. A FREE notification can only
occur after the corresponding READY was consumed and the read completed; the
next READY can only occur after that FREE was consumed and the write completed.
Consequently the previous notification is consumed before the same channel is
set again. There is no additional global barrier between iterations.

The finite net contains one program instance per chunk, with independent AIC,
AIV0, AIV1, ingress, and egress control streams, plus asynchronous engine
completion transitions. Exhaustive BFS explores all their enabled interleavings.
It checks every buffer's one-hot state invariant on every edge, rejects any
attempted set on an occupied register, verifies flag provenance, checks RAW and
WAR ownership, and verifies queue-to-scalar drain certificates. Invalid memory
accesses and full-register sets are **assertion failures**, not hidden wait
conditions. This distinction prevents a circular proof that merely disables
every unsafe action.

The finite graph must have exactly the prescribed drained terminal marking.
Reverse graph traversal must reach every explored state from that marking.
Finite control is acyclic, so a maximal execution terminates; no infinite
internal execution is possible. Empty flags/events/queues are required at exit.
FREE buffer tokens, idle resource tokens, and the terminal control token remain:
requiring literally every Petri place to be empty would contradict resource
conservation.

The modulo net contains four chunk *positions*, not four iterations. Its
control streams loop independently and BFS runs until the complete graph closes.
Producer and consumer may cross their iteration boundaries at different times.
No bounded unroll, time horizon, random schedule, or partial-order reduction is
used. Every reachable marking must be able to return to the initial marking,
and every transition must occur somewhere in the reachable graph. Thus there
are no deadlocks, non-drainable traps, or dead transitions in the recurrent net.
Scheduler progress assumes weak fairness; a scheduler allowed to ignore an
enabled core forever cannot have an unconditional starvation-freedom guarantee.

For shutdown at a quiescent period boundary, consume the restored FREE, L1,
and packed-output credits exactly once using the finite epilogue. At that
boundary all queues are idle and READY is empty, so these waits cannot block
and leave no hardware register set. Reverse reachability proves that such a
boundary is reachable from every cyclic marking. Arbitrary partial-period
tails are checked by the finite `--chunks` model; the default checks eight
chunks and the regression suite also checks one, two, and three.

Generation numbers are absent from the cyclic state because slot ownership
prevents ABA: a slot cannot begin another write until the previous read has
released its FREE credit. The per-slot cycle above gives an inductive lift from
each finite cyclic trace to arbitrarily many distinct chunk generations.
This lift depends on consuming waits. The sticky-wait mutation deliberately
breaks that premise and must produce a counterexample.

## Search space and minimality

The four logical channels per owner are READY[0], READY[1], FREE[0], FREE[1].
The synthesizer enumerates all 15 set partitions, modulo physical-ID renaming,
in increasing channel-count order. Symmetric endpoints use the same partition;
the physical register spaces of the two owners are disjoint. A candidate is
rejected if its prologue aliases credits, a wait consumes the wrong channel,
or any reachable execution violates safety/liveness. The surviving four-color
assignment uses canonical AIC IDs 0, 1, 2, 3 and 16, 17, 18, 19.

The counterexamples also give a small lower-bound certificate: the two FREE
channels conflict because both are primed; the two READY channels conflict
because both may be pending; and any READY/FREE alias lets a consumer mistake
a primed FREE bit for READY. All six pairs therefore conflict in this grammar.

Minimality is **within this protocol grammar**, preserving two outstanding
landing slots and separate READY/FREE channel identities without new throttle
messages. It is not a lower bound over every possible synchronization algorithm.
A deliberately serialized protocol, directional register aliasing on different
hardware, or a different channel multiplexing scheme changes that problem.
HardEvents are keyed by core, route, and numeric ID; the generated local IDs are
0..3 for L1, 0..1 for packed output, and 0 for serial drains. Numeric IDs 6/7
and mode-4 IDs 11..14/27..30 are excluded.

Wait placement enumerates nine pairs of cuts in the two dependency chains,
rejects five that follow a protected access, and selects the latest of the four
legal pairs. Earlier pairs are dominated in the independent-work slack metric.
Wait placement maximizes the amount of independent work before the wait:
compute precedes the FREE wait, and address preparation precedes the READY wait.
Moving either wait one operation further would put it after its protected
access. This is a partial-order slack objective, not a measured cycle-time or
throughput optimum. Queue drain placement is mandatory for both notifications.

## Abstraction and integration contract

The model covers synchronization and ownership, not quantisation arithmetic,
cache coherence, memory alignment, bus burst splitting, or compiler lowering.
It is a new generated protocol, **not an extracted proof of the reference C++**.
The header describes the insertion points and supplies finite template constants
and runtime modulo mappings; the reference kernel is not automatically patched.

MTE2 ingress and MTE3 egress have independent, in-order streams. Ingress obeys
the column dispatcher's program-order bound: StagePage(0), then
StagePage(c+1)/Gemm(c), with StagePage(c+2) permitted only after Gemm(c)'s scalar
notification. Program-order permits in the net are not hardware flag IDs.
Within that bound this permits more issuance interleavings than the reference.
Each engine has one operation in service; deeper hardware issue queues require
a refinement argument or a richer model. MTE1 and Cube compute are one ordered
macro-operation with L1 held through compute and L0C held through FIX_S. This
is conservative about buffer lifetime and does not certify fine-grained
MTE1/M/L0 scheduling in the existing kernel. The packed bytes and scale slice
share ownership; DMA completion represents completion of **both** stores.
Shared Vector scratch is drained before the next chunk reuses it.

Engine completion and its subsequent HardEvent signal are separate transitions.
The two local completion signals for Cube and quantisation are emitted as one
ordered macro-transition. They have independent consumers; neither consumer can
cause the next signal on either route before the other signal has been emitted.
The terminal epilogue is an atomic join of otherwise independent waits, after
all streams have issued their final work. There are no downstream users of the
consumed credits. MTE3 must also be drained to scalar before a real kernel return;
the header explicitly calls out an MTE3_S drain for that integration step.

The supplied requirement treats mode-4 flags as consuming one-bit latches.
Huawei's [CrossCoreWaitFlag documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)
describes counters and distinguishes AIC IDs from AIV-local IDs. A protocol
that never has more than one outstanding notification also fits consuming
counter semantics. It cannot prove correctness for a non-consuming level test.
The repository's `MICROROUTINE_DESIGN_GUIDELINES.md` sections 6.2 and 6.2c record
different endpoint observations and an unresolved stale-wait failure.
Therefore the header exposes the endpoint conversion explicitly and hardware
or CAModel validation against the installed CANN version remains necessary.
The proof report must not be described as that hardware validation.

### CAModel integration counterexample (Ascend950PR_9599, CANN 9.2.0-beta.2)

An attempted integration of the generated four-color mapping into
`turboquant_reshape_and_cache_a8.cpp` built in Release mode and completed the
16-token CAModel fixture with bit-exact caches and intact guards. Its
instruction trace nevertheless refuted the required consuming-wait premise:
on AIV1, READY channel 16 was set by AIC at cycle 4906, waited at 4916,
**waited again at 7589**, and only set for the second chunk at 8621. The
second wait therefore passed 1032 cycles before its matching notification.
The old `analyze_reshape_trace.py` correctly rejected repeated IDs after the
apparently successful numerical run. The attempted kernel integration was
reverted; the existing one-ID-per-event schedule remains in place.

The abstract proof in `safe_pipeline_proof.json` remains valid under its
explicit consuming-latch semantics, but it is **not a hardware refinement
proof** for this CAModel configuration. Reusing READY/FREE IDs in a production
kernel requires an independently verified clear operation or another
generation-aware handshake. Passing numerical output on one simulator run
does not establish that property.

## Negative controls

Every normal run checks missing FIX_S, missing V_S, missing FREE_ACK waits,
non-consuming waits, missing packed-output reuse waits, and aliased READY slots.
All must fail with a concrete counterexample. The JSON retains the full traces,
including the asynchronous completion ordering that makes each failure possible.

The separate checker regressions run quickly:

```powershell
python -m unittest discover -s kernels/ops/turboquant -p test_synthesize_safe_pipeline.py -v
```

They include a naive all-transition BFS cross-check, short and odd workloads,
invalid-set detection, assertion-versus-wait behavior, deadlock/dirty termination,
non-recurrent cycles, state limits, partition enumeration, and endpoint mappings.
They also check the linear place invariant that every queue retains exactly one
idle/work/done token across every transition.

An additional structural liveness certificate checks the cyclic incidence
matrix: all 130 transition counts equal to one form an integer T-invariant,
and elimination modulo the prime 101 gives rank 129. A nonzero modular minor
lower-bounds the rational rank, while the integer invariant upper-bounds it by
129, so the exact nullspace is one-dimensional. Every nonempty execution cycle
must therefore fire every transition equally often. Together with the finite
reachable graph and absence of deadlocks, this excludes an infinite firing
execution that starves a particular transition. Hardware must still make
progress rather than stop servicing all enabled work.

The delivered run passed all 15 regression tests. The generated declarations
also compiled as C++17 with MSVC `/W4 /WX`; compile-time assertions checked the
chunk-7 endpoints and 10,000 modulo positions. These are host declaration checks,
not a CANN device compilation or CAModel execution.
