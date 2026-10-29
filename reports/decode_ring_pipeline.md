# Two-slot AIV decode pipeline validation

2026-10-29. CANN 9.2.0-beta.2, `Ascend950PR_9599` CAModel (`dav_3510` model).
This report covers `turboquant_single_token_decode_aiv.cpp` and
`turboquant_single_token_decode_splitk_aiv.cpp`. No physical NPU was used.

## Implementation

Both operators use a depth-two modulo ring for packed keys/values, key/value
scales, dequantized key/value pages and raw score panels. Packed feedback uses
`V_MTE2` IDs 0/1; scale feedback uses IDs 2/3. Ready events use `MTE2_V` IDs 0/1.
The AIV archive has its own `V_MTE3`/`MTE3_V` pair per slot. Every feedback credit
is primed once and explicitly drained, including an unused slot or empty worker.

The schedule prefills two slots, prepares strip i, recurs strip i-1, and refills
the retired slot. Carried m/l/acc are updated in sequence order. Vector stages
share one ALU and execute serially; the concurrent engines are MTE2, Vector and
MTE3. SIMD scratch remains shared in Vector FIFO order. Split-K's vector batch
merge and atomic protocol are unchanged. The simulator benchmark now skips ACL
timing-event APIs, which request unsupported resource type 5 from the stub
driver; simulator timings were already reported as unavailable. Stream
synchronization and all numerical/protocol validation still execute, and the
hardware timing branch remains intact and was compiled with strict warnings. Production launcher ABIs are unchanged;
diagnostic launchers select a logical-page window and call the same processing
functions. Ordinary zero-offset varied-score tests call the production launcher.

Each scale landing reserves 256 bytes despite its 64-byte payload, preventing
unmasked register reads from touching a neighboring DMA landing. Ring allocations
have a 32-byte skew and compile-time alignment/disjointness/bank checks.

| Arena | Payload end | Including final 256-byte read pad | Limit |
| --- | ---: | ---: | ---: |
| AIV | 190,880 | 191,136 | 196,608 bytes |
| Split-K | 189,824 | 190,080 | 196,608 bytes |

## Host and compilation evidence

- The queue/marked-graph model explored 21,738 reachable states over both
  variants at N=0,1,2,3,4,5,8,16,64. All reachable terminal states drained every
  flag and buffer owner. All interleavings preserved ordered recurrence and
  capacity-one places. Negative controls rejected an unprimed cycle, a double
  SET with balanced totals, and a premature scale release with balanced totals.
- Six trace-auditor tests passed, including hexadecimal IDs, reserved IDs,
  nonempty final markings, unprimed waits and independent completion ordering.
- Both actual kernel libraries and runners built in a fresh ext4 source/build
  tree with `-Wall -Werror` on device sources and `-Wall -Wextra -Werror` on
  runners. CANN headers are system includes. The SDK packager prints its
  advisory about multiple exported kernels in a translation unit because the
  diagnostic entries share production code; this is not a compiler warning.
- The existing single-token, Split-K and rotation/oracle host checks were run
  as recorded in the evidence logs. Skill validation and `git diff --check`
  passed.

Host model exploration is bounded evidence supplemented by the slot-generation
induction in `REFINEMENT_OF_COMBAT_KERNELS.md`; it is not a proof of all possible
hardware executions or numerical inputs.

## CAModel evidence

Final source/build/trace root: `/tmp/ascend-decode-ring.9MUIFm` in WSL
`Ubuntu-22.04`. Subsequent benchmark build: `/tmp/ascend-ring-regression-build`.
Each case has its own `run.log`, dispatch/completion/exception dumps,
`audit.json`, and either `census.txt` or `atomic.json`. Small evidence files are
retained beside this report; raw traces remain in WSL.

The boundary fixture uses the actual 256-channel × 16-token geometry, one query
row of head 2, logical K=8, and a non-identity block table. Probes select
head `(page=0, strip=0)`, middle `(4,4)`, and tail `(7,7)` without rebasing the
table. Only those slices are real. All GM payloads start as `0x7FC0BEEF`; exterior
guards and every read-only input are checked bytewise too. FP32 qNaNs are byte
patterns when viewed as packed int4 or BF16.

An exact fixture (query=1/256, key=3, scales=1, gate=0) makes every score 3 and
every exp weight 1. Scalar token sums and a dense matrix rotation independently
define expected outputs. Positive even values plus one odd channel keep division
numerators and rotation outputs nonzero. This matters because an earlier exact
fixture produced correct zero outputs but triggered CANN's precise-Div internal
Inf/NaN diagnostic; it was rejected by the strict model-error gate. No diagnostic
was suppressed and no arithmetic tolerance was widened. General zero-numerator
behavior of that SDK implementation is not established by the final fixture.

The AIV checks raw scores, all m/l replicas, accumulator, rotation and BF16
context bitwise. Split-K exports context and protocol metadata, so its isolated
probe checks exact BF16 context, protocol fields and every unassigned byte.
Its unused state rows, metadata padding, inactive heads, inputs and guards must
be unchanged. No native UB guard-canary claim is made; UB safety is checked by
layout assertions, queue ownership and simulator execution diagnostics.

Final baseline and regression totals are recorded in `decode_ring_pipeline/summary.json`.
The manifest requires all ten baseline cases and five production regressions
to pass, rechecks the raw traces and runner success markers, and verifies that
the tested kernel-source SHA-256 values match the workspace.

## Reproduction

```bash
python3 tests/cpu/test_decode_ring_model.py
python3 tests/cpu/test_decode_ring_trace_audit.py
bash scripts/run_decode_pipeline_camodel.sh
```

The baseline script creates a fresh `/tmp/ascend-decode-ring.*` directory,
builds both strict targets, and runs six isolated boundary probes plus depth=2
and wrap=4 for each kernel. It fails on timeout, model errors, nonempty exception
dumps, numerical/canary failures, absent traces, flag leaks/over-capacity or
missing measured overlap on the wrap case. `--strips N` on the trace auditor
also requires exactly `1+floor((N+1-s)/2)` feedback pairs for each slot s.

After ten baseline passes, build `bench_decode_splitk_sim` in a fresh CMake tree
with the same toolkit/SoC, then run:

```bash
# Build the simulator benchmark from current sources without launching it.
decode_bench_dir=$(mktemp -d /tmp/ascend-decode-bench.XXXXXX)
SPLITK_SRC_DIR="$decode_bench_dir/src" SPLITK_BUILD_DIR="$decode_bench_dir/build" \
  bash scripts/run_decode_splitk_bench.sh sim --help
bash scripts/run_decode_pipeline_regressions.sh BASELINE_DIR "$decode_bench_dir/build"
python3 scripts/verify_decode_pipeline_evidence.py BASELINE_DIR
```

This adds the six-row production path, a non-identity two-page sequence,
an odd three-strip drain, K=2 with five strips per worker and real merging,
and K=4 over three pages with an empty worker. The varied-score tests use their
existing Exp/reduction/BF16 numerical budgets, separately from exact boundaries.
Simulator intervals describe observed overlap, not silicon latency or throughput.

The new primary procedure is `REFINEMENT_OF_COMBAT_KERNELS.md`; `AGENTS.md`,
the repository guides and the project skills route future pipeline refinements
through the same boundary-first baseline before pushing or hardware tagging.
