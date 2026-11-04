# Fused decode: three-zone schedule (2026-11-03)

## Change and invariant

`TurboQuantFusedDecodeOpt` selects the deep path once for `numTiles >= 4`.
The Vector fill is exactly `read(0)`, `read(1)/stage(0)`,
`read(2)/stage(1)`; Cube scores tile 0 in the final fill tick. These are
logical scheduling ticks, not measured simultaneous hardware intervals.

The deep loop ends at `steady_limit = numTiles - 3`, unrolls six unconditional
phases and keeps all slot/window indices as template constants. The bounded
remainder selects a statically indexed three-step drain; it does not reset
the phase at `N-3`. Drain steps successively remove read, stage, then score.
`StageTaskOutput` and the existing per-task engine/cross-core drain remain.
Short tasks (1..3) have a separate bounded path; empty workers bypass both.

The first unconditional stage wait needs a **new initial free credit for L1
slot 2**. Cube publishes it before score 0, behind the same producer fence
used by recycled credits. For every deep task:

| Channel | Production and consumption |
|---|---|
| SlotReady[s] | stage tile k -> score k, s=k%3 |
| ScoresReady[w] | score tile k -> softmax k, w=k%2 |
| ProbsReady | softmax k -> context k |
| ContextReady | context k -> accumulate k |
| SlotFree[s] | initial s=2 -> stage 2; context k -> stage k+3 |

There are N score, probability and context handshakes, N ready handshakes,
and N-2 free handshakes (one initial plus N-3 recycled). No final free
credits are emitted by the three drain steps. Per subcore, each place has
capacity one. The task join prevents the next task from priming before the
previous task's queues have drained.

This removes **schedule boundary branches**. Page-table traversal, partial
row masking and service-level head/geometry checks remain. It is not a claim
that every transitive arithmetic/DMA helper emits branch-free machine code,
or that the five engines physically overlap; those require separate evidence.

## Isolation and oracle

Simulator entry, launcher, buffers, runner and recording services are under
`tests/sim/kernel/`. The simulator translation unit includes the production
kernel and calls its `ComputeTask` path. The physical launcher no longer
contains the probe dispatch. No physical NPU is executed.

The oracle uses the existing 256-channel x 64-token KV4/FP8 geometry,
nonidentity logical-to-physical mapping `p -> 2*p+1`, equal scores for exact
boundary checks, and independent FP16 rounding. A partial last tile is
weighted by its valid row count in the CPU oracle. The separately labelled
varied-score check retains the existing one-FP16-ULP tolerance.

All allocation guards, unused cache pages, scale padding, inactive output
rows and inactive workspace bytes begin as `0x5A`. Every input allocation
is snapshotted and compared byte-for-byte after each launch. Active metadata
must contain valid indices or the unmapped `-1` sentinel rather than a
`0x5A5A5A5A` address. Full input comparisons include those metadata bytes.

The four named cases are T0 (isolated logical page 0), T1 (N=4, first deep
swap), Tmid (N=9, one complete Period-6 body), and Tlast (N=16 with only 17
valid rows in the final tile). Tmid/Tlast execute all intermediate tiles:
four independently rebased single-tile probes would not test a steady loop
or a ring wrap. These are final-output prefix/boundary oracles plus full
traces, not instrumentation of four intermediate accumulator snapshots.
Additional isolated pages 4/7 and strip 1 of page 7 precede reuse tests.

## Verification environment and commands

- WSL Ubuntu 22.04 -> Docker container `6b07e573f9cf`.
- Image `ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12`, digest
  `sha256:b82c08f530f956afe7c445361b1454cbcd97aaf60c4093749fe872a9657edcd7`.
- CANN `/usr/local/Ascend/cann-9.2.0-beta.2`, CAModel `Ascend950PR_9599`.
- Evidence root in container: `/work/fused-three-zone-20261103/evidence`.
- Corresponding WSL path:
  `/home/ladislav/work/fused-three-zone-20261103/evidence`.
- Reproduction entry: `scripts/run_fused_decode_three_zone_camodel.sh`.
- Build continuation command: `reports/fused_decode_three_zone_20261103/build.sh`.

The first build attempt passed the host gates, then stopped because this
image has no `rsync`. The normal runner now falls back to `tar`; this run
copied the source using WSL rsync and continued with the recorded build
command. Build paths are on ext4, not the Windows mount.

## Gate results

- PASS: strict host contract, including UB budget/alignment/bank assertions.
- PASS: capacity-one completion exploration, N=0..16, two consecutive tasks;
  N=16 explores 1,214,429 reachable states. Early refill, wrong window,
  duplicate SET and missing initial slot-2 credit are rejected.
- PASS: extracted **actual production C++ schedule methods** compiled with
  recording services at `-Wall -Wextra -Werror`, N=1..130. Checks operation
  order, counts, literal slot/window selection, all six drain residues, and
  absence of boundary conditions in both steady method bodies.
- PASS: five source-fence/trace-auditor positive and negative controls.
- Device build and CAModel ladder: pending completion below.

## Limits of the event audit

User HardEvents must be 0..5. Cross-core channels are a different namespace:
the existing protocol uses 0..10 including task join 10. CANN's existing
`ArithProgression` lowering emits reserved `VEC -> SCALAR` HardEvent 7;
the auditor permits only that exact compiler-owned route and audits its
lifetime. Thus a literal claim that **every generated event** is 0..5 would
be false. Toolkit global reduction barriers also use their reserved mode-0
channel 11. No new user ID is introduced by this refactor.

The completion model abstracts fenced GEMM/Fixpipe and stage/MTE3 operations;
it is not CAModel evidence. The existing cross-core trace audit checks
producer census/fences and available completion order, but lacks a direct
observation of every scalar wait release. No silicon performance or measured
engine-overlap claim is made. No push or hardware tag is performed.
