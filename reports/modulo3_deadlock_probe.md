# Three-stage modulo deadlock probe

Kernel skeleton: `kernels/skeleton/turboquant_reshape_and_cache.cpp`.

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599` CAModel. The
boundary run dispatches exactly three 128-byte diagnostic tiles. Each tile is
an independent synthetic head, so each output is stored once. The production
online-softmax recurrence is stated in the kernel source, but arithmetic and
FP16/BF16 normalization are intentionally absent from this concurrency probe.
There is no Split-K partial-sum buffer or GM workspace.

## Event lifecycle

Each route owns three separate one-bit event IDs, 0, 1, and 2. Prologue sets
`V_MTE2` and `MTE3_V` once for each slot (six free-slot tokens). The ramp-up
dispatches ingress 0 and 1 and Vector 0. The only complete three-stage column
is `StepColumn<2,1,0>`; the other two template columns are compile-time drain
steps. Epilogue consumes the six final feedback sets. No runtime `% 3`,
hot-loop branch, or `PipeBarrier<PIPE_ALL>()` is present.

| Route | Sets | Waits | Meaning |
| --- | ---: | ---: | --- |
| `MTE2_V` | 3 | 3 | Ingress payload ready |
| `V_MTE2` | 6 | 6 | Three primed + three Vector releases |
| `V_MTE3` | 3 | 3 | Vector output ready |
| `MTE3_V` | 6 | 6 | Three primed + three MTE3 retirements |

Each feedback dependency cycle starts with a free-slot token, so the initial
marked-graph token sum is at least one. The CAModel instruction trace contains
18 `SET_FLAG` and 18 `WAIT_FLAG` operations across 12 route-ID pairs, all in
the range 0..2. Matching counts plus the explicit epilogue waits establish
zero residual tokens in this run.

## Execution and overlap evidence

The host queue model passed with zero deadlocks, zero canary corruptions,
18/18 event pairs, zero residual tokens, and two ticks with three work queues
active. The native Ascend C source compiled with `ccec`, and the CAModel
runner exited zero with all 96 output words equal to `0xDEADBEEF` and both
64-byte GM guard bands intact. The CAModel reported 3,394 model ticks and
8,745.91 ms model wall time for the saved trace run. These are simulator
figures, not silicon latency or steady-state throughput.

The core-0 queue trace provides actual overlap evidence. In its latest run,
MTE2 copy instruction 136 was popped at cycle 1,650 and completed at 2,198;
Vector add instruction 254 was popped at 1,779 and completed at 1,786; MTE3
copy instruction 164 was popped at 1,772 and completed at 2,307. Their
in-flight intervals intersect at cycles 1,779..1,786. The Scalar core had
queued the final event waits by cycle 1,686, while the last MTE3 copy did not
complete until cycle 2,798. This confirms Scalar dispatch ran ahead of the
execution queues in this boundary run.

The evidence is in [popped queue instructions](camodel_core0_popped.dump),
[completed instructions](camodel_core0_completed.dump), and the
[trace summary](camodel_trace_summary.txt). The overlap and event counts can
be recalculated with `scripts/analyze_modulo3_trace.py`.

## Scope of the canary check

The Vector operation reads every ingress word and adds a constant so an
uncorrupted `0x55AA55AA` becomes `0xDEADBEEF`. The CAModel runner checks the
entire GM output and GM guard bands. The host queue model also poisons and
checks the 64-byte UB guard bands around each slot. The native kernel reserves
those UB guard regions but does not read them back; native UB guard integrity
is therefore not directly measured by this run. The 288-byte slot stride is
32-byte aligned and shifts adjacent slot bases by one 32-byte bank relative
to a 256-byte multiple.

`N=3` contains one complete three-stage column. It establishes liveness at
the depth boundary and one short three-pipe overlap interval. It does not
establish sustained overlap, bank-conflict freedom for larger workloads, or
the correctness of online-softmax arithmetic.

## Reproduce

In WSL Ubuntu 22.04, from the repository root:

```sh
bash scripts/run_modulo3_camodel.sh
```

The script builds in a fresh `/tmp` directory, runs the host queue model,
launches the native kernel under the direct 950PR CAModel runtime, and parses
core-0 instruction traces. Set `MODULO3_BUILD_DIR` to reuse a completed build
with `bash scripts/run_modulo3_camodel.sh profile-only`.
