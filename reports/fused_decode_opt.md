# KV4/FP8 fused decode integration

Date: 2026-11-03. Target: Ascend950PR_9599, CANN 9.2.0-beta.2, WSL.

The operator is a self-contained integration of
`C:/Ascend/vllm-ascend/csrc/attention/turboquant/op_kernel/turboquant_fused_decode_opt.cpp`.
The dependency snapshot and byte hashes are under
`kernels/ops/turboquant/fused_decode/`. It preserves 64-token tiles, D=256,
the three-slot packed/scale/L1 ring, two score windows, online max/sum state,
FP8 query/probability operands, and the upstream affine KV4 codebook.
Existing 16-token BF16 and AIV decode operators are unchanged.

## Microroutine boundary

`routine_turboquant_dequant4.hpp` supplies signed nibble widening before the
source's channel-half deinterleave and +0.5 midpoint. The source's FP8 rounding
and scale/gain convention remain. `routine_vector_online_softmax.hpp` now exposes
the three probability phases separately from accumulator update; the original
four-phase function delegates to those phases and retains its existing ABI.
`routine_vector_cube_softmax.hpp` implements the 64-key native engine and
Vector decay/accumulation. These leaves issue Vector arithmetic only, with no
scalar UB readback or event ownership. DMA and cross-core events stay in the
operator/services. TPipe is retained for the source codec/basis/reducer and
launch event allocation; this integration is not a full static-allocator port.

The imported Cube service retains its FP8 LoadData/Mmad/dual-destination
Fixpipe path, which cannot be replaced by the existing fixed 16x16 FP16
Cube engine without changing arithmetic or geometry. Raw-query basis and
Split-K reduction services are retained as local dependencies.

Producer notifications are preceded by the appropriate engine-to-Scalar
fence. L1-slot release uses the completed context Fixpipe, which transitively
proves MTE1/Matrix completion. A block-local join on channel 10 drains task
output and remote Fixpipe landings before task/launch reuse. The source uses
FFTS mode 2, not the older harness operator's mode 4.

## Reproduce

```sh
bash scripts/run_fused_decode_opt_camodel.sh build
bash scripts/run_fused_decode_opt_camodel.sh ladder
# Reuse the printed build, without recompiling:
FUSED_DECODE_OPT_ROOT=/tmp/ascend-fused-decode-opt.XXXXXX \
  bash scripts/run_fused_decode_opt_camodel.sh reuse --canary 4
```

The script builds on fresh ext4 storage. Every run has its own log, core
dispatch/completion dumps and audit JSON. Importing again is intentionally
blocked by `scripts/import_fused_decode_opt.py` when destination files exist.

## Verification gates

| Gate | Status / evidence |
| --- | --- |
| Host launch/shape contract and NZ cache bijection | PASS, strict g++ C++17, 1..4 KV heads; zero divisors, misalignment and incomplete task coverage rejected |
| Shared online-softmax regression | PASS, existing four-tile/16-channel host oracle, byte-identical recurrence and guards; `fused_decode_opt_softmax_regression.log` |
| Capacity-one model | PASS, all completion interleavings for N=0,1,2,3,4,6,7,16 over two consecutive tasks; `fused_decode_opt_host_model.log` |
| Negative model controls | PASS: early ingress overwrite, score-window aliasing and double SET rejected |
| Compile-time UB | PASS host assertions, 142304 bytes decode + raw-basis/reducer budgets + 256-byte read pad <=192 KiB; production allocations share the checked list |
| Strict CANN simulator + NPU builds | PASS, both standalone libraries and both Split-K benchmark executables; benchmark build root `/tmp/ascend-fused-opt-bench.yOuivK` |
| Isolated logical pages 0,4,7 and final tile | PASS, four independent runs, 4096 bit-exact FP16 outputs each, input bytes and inactive outputs preserved, exterior guards intact |
| Depth=3, wrap=6, odd=7 | PASS, exact FP16 oracle and each run's own trace audit |
| Repeated tasks/launches | PASS, 17 query heads (two tasks), one tile per task, two launches; exact context and empty event drains |
| Real Split-K workers, including empty workers | PASS, K=4/grid=4 over three pages: three nonempty workers and one empty worker; exact final context |
| Dispatch/completion/cross-core census | PASS on every case; per-core opcode/engine census, balanced capacity-one hard events and empty final marking |
| Varied tile maxima | PASS, seven tiles, scalar FP64 softmax reference within one FP16 ULP; producer/consumer trace checks pass |
| Raw-query basis and within-tile varied probability quantization | Pending; compilation alone does not validate those paths |
| Multi-wave grids >32 | Unavailable in admitted launch contract; requires a separate two-wave model and dispatch ladder |
| Physical hardware | Unavailable on this host; no silicon performance or overlap claim |

The host model is a bounded completion abstraction of five FIFO queues:
Cube, two Vector consumers and their MTE2 ingress queues. A stage includes
its fenced MTE3 completion; a GEMM includes its fenced Fixpipe completion.
It tracks generations of packed/scales, L1 K/V, score windows, probabilities,
and context, then checks ordered recurrence and an empty final marking.
It does not establish compiler fence lowering or simulator flag semantics.

The isolated runner keeps eight logical pages, a non-identity fragmented
block table, full cache allocations, and poisoned inactive bytes. A diagnostic
cursor selects one tile at its original logical offset and calls the same
`ComputeTask` and pipeline methods as production. Every input byte is checked
after execution, every inactive output row is checked, and both exterior
guards are checked. The exact fixture has constant scores, per-tile/channel
values, uniform probabilities and an independently rounded FP16 context
oracle. It intentionally does not replace a varied-score/alpha<1 regression.

Inputs use `0x7FC0BEEF` poison outside assigned slices. The shared runner's
exterior guards and inactive output rows use `0xDEADBEEF`; all are compared
byte-for-byte. Packed int4 poison is a byte sentinel, not necessarily NaN.

Initial build attempts exposed a dropped host-launch define in the second
`ascendc_compile_options` call and a missing host-side GM_ADDR definition.
Both were corrected; those attempts are not successful build evidence.

The first numerically correct canary then exposed a real imported event fault:
the context loader set `MTE1->CUBE id 0` twice before its first consumer retired
(second SET at tick 12021). Independent L0B bands now use one final handoff,
after all bands, rather than reusing that capacity-one event inside the band
loop. The counterexample is archived under
`fused_decode_opt_20261103_pre_band_fix/`, including `audit-failure.txt`.

Passing operator evidence is under `fused_decode_opt_20261103/`; final standalone
build/empty-context evidence is under `fused_decode_opt_20261103_build/`.
Every case has its run log, audit JSON and compressed original core dumps.
The initial long ladder was stopped after depth/wrap in favor of individual
cases; those individual case logs and audits are the authoritative results.
During development, editing a live shell script interrupted wrapper completion
after CMake had built its libraries. The final separate benchmark build exited
successfully and the final empty-context wrapper exited successfully.

The trace audit distinguishes two toolkit-owned channels from user events:
BiSheng emits `VEC->SCALAR id 7` around CANN's annotated `ArithProgression`
calls, and `SyncAll<false>()` announces the global barrier on mode 0/id 11.
The former still gets full dispatch/completion capacity-one checks. The latter
gets a separate per-AIC announcement census, not a fictitious local wait pair.
The K=4 case has exactly one global-barrier announcement on each of cores
0..3. User FFTS mode-2 channels remain restricted to 0..10.

| Run | MMAD / Fixpipe count | Result |
| --- | ---: | --- |
| Each isolated boundary | 2 / 2 | Exact |
| Depth 3 | 6 / 6 | Exact |
| Wrap 6 | 12 / 12 | Exact |
| Odd 7 | 14 / 14 | Exact |
| Varied maxima, 7 tiles | 14 / 14 | <=1 FP16 ULP |
| 17 heads, two tasks, two launches | 8 / 8 | Exact |
| K=4 over three pages | 4 / 4 on each of cores 0..2; 0 / 0 on empty core 3 | Exact |

No measured engine-overlap or hardware-throughput claim is made. Multi-wave
grids, raw-query basis output stages, arbitrary within-tile score patterns and
Split-K ring wrap inside each nonempty worker remain additional qualification
work before hardware promotion.

## Split-K benchmark

`tests/npu/bench/bench_turboquant_decode_splitk.cpp` dispatches this kernel with
`--impl fused-opt`; `--impl all` includes it alongside the existing two legs.
`both` retains its original meaning. The new leg uses FP32 pre-rotated queries,
the source KV4/FP8 cache, and FP16 rotated-basis output. Its event interval
includes the in-launch Split-K reduction. It does not include query rotation
or inverse rotation/gating; it is not a direct end-to-end speedup comparison
against the other fixtures.

`--heads` counts KV heads and `--rows` counts GQA query heads per KV head for
this leg (1..16 for `fused-opt`, 1..8 when also selecting the legacy AIV leg).
The fused-opt K sweep defaults to 1,2,4,8. Explicit K>8 is rejected for this
leg; `all` prints NA in its columns at unsupported K. Grid size is capped at
32 physical blocks and excess logical tasks run in the kernel's task loop.

The added CSV columns are `opt_grid,opt_us,opt_speedup,opt_kv_cache_gbs,opt_launches`.
The existing columns remain in their original order. OPT speedup uses its
own K=1 baseline, and GB/s uses the live cache footprint including padded scale
slots, excluding fragmented canary pages. The source has no worker timestamp
ABI; no tail/export timestamps are fabricated for it. CAModel event timings
and speedups remain NA.

```sh
bash scripts/build_fused_decode_opt_bench.sh
# Simulator smoke against the printed build root:
bash scripts/run_fused_decode_opt_bench_smoke.sh /tmp/ascend-fused-opt-bench.yOuivK
# On a host with a physical 950PR:
./build/lib/bench_turboquant_decode_splitk --impl fused-opt --rows 16 \
  --context 1024,4096 --splitk 1,2,4,8
```

The benchmark's CAModel smoke passed at N=192, 64-token pages, two requests,
two KV heads, GQA group=3 and K=2: eight physical MIX blocks, one launch,
3072 FP16 output elements within one ULP of a scalar FP64 reference, unchanged
input bytes, intact guards, finite fully written partial records, passing
per-core trace audit, and the expected CSV row with unavailable timings.
The original AIV benchmark path also passed at N=16/K=1, with OPT absent.
CLI help and invalid fused-opt split/page guards passed. The host fixture
probe checks the packed NZ image through a separate inverse map, query/head/
request variation, a one-tile exact oracle, and full 16-head task geometry with
the 32-block cap. No physical NPU timing was run on this host.

Benchmark evidence, final host refresh/link logs, CMake compile/link settings,
CSV outputs and original traces are archived under
`fused_decode_opt_bench_20261103/`. The final eight-block rerun used the
mantissa-carry-safe FP16 oracle; its trace ended at model tick 28444 (not a
hardware duration). CLI row-limit guards also passed. The original shared
conversion helper was left unchanged.
