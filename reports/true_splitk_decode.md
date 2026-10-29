# True in-kernel Split-K decode (dav-3510)

This is the validation record for commit `f8bbb24`. Later AIC schema and
benchmark presentation changes are recorded in
[the M=16 comparison report](decode_splitk_aic_comparison.md).

The new operator partitions physical pages across `(request, kv_head, split)`
blocks. Each worker runs the existing AIV GEMV and online recurrence only over
its range. A bounded SIMT TAS acquire and atomic arrival increment protect one
shared head slot. The last arrival normalizes, runs the register-resident egress
butterfly, applies the sigmoid gate and writes BF16 `out_context` in that launch.

The three frozen production files are unchanged. Existing unrelated edits in
the single-token CAModel script/runner and `probe/prof_output` were preserved.

## State and critical section

`include/turboquant_splitk_contract.hpp` defines the shared ABI. A 128-byte
control header precedes eight 1088-byte row records. The usual six GQA rows
are independent softmax states under one head lock and arrival counter; one
256-element accumulator cannot represent all six queries.

Local state packing runs before acquiring the lock. After acquisition, later
arrivals fetch all six row records in one contiguous 6528-byte MTE2 transfer.
`MergeHeadState` computes all 48 replicated max/sum lanes together in registers,
then streams 64-channel register groups across the accumulator panel. There are
no per-row DMA calls, HardEvent pairs or Scalar reads of numeric UB state.
Intermediate workers write one contiguous 6528-byte MTE3 transfer and drain
once before releasing the lock. The tail retains the merged state in UB.

Egress expands denominators in a register VF and performs one continuous
`Div` over the complete row panel, then uses the existing butterfly and gated
BF16 store. No frozen egress routine was edited.

CANN 9.2's `asc_vf_call` returns `void` and dispatches asynchronously. Protocol
control therefore uses one `V_S` completion handoff, invalidates the scalar
cache line and reads the SIMT-published arrival directly from GM into a scalar
register. The kernel contains no `GetValue()` call or `MTE2_S` fence. The
shared-state MTE2 read still has the required single `MTE2_V` dependency, and
the writeback has a single publication/drain/release dependency chain.

First arrivals seed directly, so the host only needs to reset lock/counter
words; the benchmark clears the entire slot. Empty splits carry a finite
neutral `(-3e38, 0, 0)` state, increment the arrival count and can be the tail.
A timed-out worker never fetches, modifies or releases the shared state; the
benchmark rejects its launch. Slots must be reset before every launch and
must not be shared by overlapping streams.

## Benchmark

Defaults are genuine contexts `1024,2048,4096,8192,16384`, 128 tokens per page,
four KV heads, six query rows and `K=1,2,4,8,12,16`. Physical pages are distinct
across requests and traversed through reversed block-table permutations.
There are no cyclic page waves or synthetic partial-state launches.

Each sample resets/poisons buffers before the start event, times exactly one
decode launch, then validates every output against a full-context FP64 CPU
softmax and independent dense rotation. The output tolerance is
`0.005 * row_max_abs_reference + 1e-6`, including BF16 rounding. Arrival
permutations, exactly one tail per head, page counts, unlocked completed slots
and replicated metric lanes are checked after every launch. K=1 is always
measured first, even if omitted from the requested sweep.

CSV reports device p50/p95, speedup versus K=1, effective KV GB/s (packed K+V
plus FP32 scales), first-completion-to-tail deltas and lock-region p50/p95,
failed exchanges and watchdog timeouts. Counter metrics use system-counter
ticks; CAModel does not emit hardware event latency, GB/s or speedup numbers.
`lock_region` includes acquire contention, control transfer and completed
release: it is a conservative hold-time bound, not an isolated merge timer.
The 950PR documentation gives 1000 system-counter ticks per microsecond on
silicon; simulator measurements are retained as model ticks.

## Validation

- CANN 9.2.0-beta.2 NPU and simulator builds, targeting `Ascend950PR_9599` /
  `dav-3510`, passed with `-Wall -Werror` on device code and
  `-Wall -Wextra -Werror` on the benchmark. No compiler warnings/errors.
  CANN's nested CMake emits its existing unused-configuration-variable warnings.
- `test_splitk_decode_probe`: all 2048 `(P=1..128, K=1..16)` partitions pass;
  shuffled real partial merges including empty splits pass; the two-request
  4096-token fixture maps every physical page once. Reintroducing cyclic
  eight-page reads changes context values by up to `0.00832015`.
- Existing `test_single_token_decode_probe` passes.
- CAModel: six rows, three distinct 16-token pages, one head, K=1/2/4.
  Every launch passed the FP64 reference, arrival permutation, page-count,
  tail-selection and completion checks, including empty split ranges.
- Completed-trace audit: seven critical sections across the three launches;
  at most one full-head state fetch and one full-head state writeback per
  section. Every HardEvent route balances and no reserved ID 6/7 is used.

### Before/after CAModel comparison

Both implementations use the same fixture, six rows, 48 genuine tokens in
three 16-token pages, one timed sample per K, and the same added lock-region
instrumentation. The prior row-wise kernel was retained in a separate `/tmp`
source/build tree for this comparison; production sources were not reverted.
These are functional model runs, not statistical NPU throughput measurements.

| K | Row-wise lock-region p50 | Batched lock-region p50 | Row-wise tail delta | Batched tail delta | Failed exchanges, old/new |
|---|---:|---:|---:|---:|---:|
| 1 | 4519 | 4646 | 0 | 0 | 0 / 0 |
| 2 | 8388 | 5180 | 35531 | 30756 | 0 / 0 |
| 4 | 17192 | 7570 | 59414 | 40239 | 52 / 23 |

All intervals are model ticks; every run had zero watchdog timeouts. At K=2
the median lock region decreases 38.25% and the tail delta 13.44%; at K=4
they decrease 55.97% and 32.27%. The K=1 protocol region increases 2.81%.

Atomic completion-to-completion trace intervals range from 3400 to 10953
ticks in the row-wise implementation and 3396 to 4938 ticks in the batched
implementation. The old critical sections contain up to 13 MTE2 reads and
18 MTE3 writes, versus at most one of each in the batched implementation.
The trace interval begins when the successful acquire exchange completes and
ends when the release exchange completes; it must not be confused with the
broader scalar-bracketed `lock_region` metric in the CSV.

Evidence: [batched CSV](true_splitk_camodel.csv),
[row-wise CSV](true_splitk_rowwise_camodel.csv),
[batched trace census](true_splitk_trace.json),
[row-wise trace census](true_splitk_rowwise_trace.json).
The complete raw model dumps are in `/tmp/vector-splitk-sim-build` and
`/tmp/row-splitk-baseline-build` in WSL Ubuntu-22.04.

## Commands and limits

```bash
bash scripts/run_decode_splitk_bench.sh build
bash scripts/run_decode_splitk_bench.sh smoke
bash scripts/run_decode_splitk_bench.sh hw --context 4096 --splitk "1,2,4,8,16"
# On the NPU host, from the built executable directory:
./bench_turboquant_decode_splitk --context 4096 --splitk "1,2,4,8,16"
python3 scripts/analyze_splitk_decode_trace.py <completed-sim-build> --assert-batched
# This stricter check fails on the observed model intervals:
python3 scripts/analyze_splitk_decode_trace.py <completed-sim-build> --max-hold-ticks 300
```

The hardware command was attempted, but this host has neither `npu-smi` nor
`/dev/davinci*`; the script exits with status 3. Actual 4096-token NPU
validation, silicon speedup curves and full AIV utilization remain unverified.
Increasing launch grid size alone does not establish 100% utilization.

The requested strict lock hold below 300 cycles is **not achieved by the
observed CAModel intervals**. A successful functional test or batched DMA
census must not be represented as meeting that performance bound.
