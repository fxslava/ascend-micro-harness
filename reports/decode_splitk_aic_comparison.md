# Split-K presentation, M=16 AIC model and single-launch profiling

The benchmark now prints a fixed-width ASCII comparison table and writes CSV
only when `--csv-out PATH` is supplied. Columns include AIV latency, speedup
and physical KV bandwidth, raw AIC batch latency, AIC modeled latency per row
(`T_batch / 16`), modeled reuse bandwidth, tail/lock intervals and failed spins.
CSV additionally records physical AIC KV bandwidth and per-implementation launch
counts. Model bandwidth is `16 * KV_bytes / T_batch`; it is not physical HBM
bandwidth. Unavailable device event timing prints `NA`, never infinity or zero
latency as a performance claim.

## Profiling behavior

`--impl aiv|aic|both` selects the execution paths. `--single` forces warmup=0
and iters=1 after parsing all other flags, disables the automatic K=1 baseline,
and avoids percentile sorting/multi-sample accumulation. Each selected
implementation launches once per explicitly selected `(N,K)` case. Thus
`--single --context 2048 --splitk 4 --impl both` launches **one AIV and one AIC
kernel**, not one kernel total. Stream synchronization fences surround every
individual launch; end events are queued before the post-launch stream fence
so host waiting time is not added to the event interval.

The required profiling banner is printed. Table headings use single-sample
labels in this mode; tail/lock show maxima across the launch's workers/heads.
Normal mode reports medians, includes warmups, and adds a K=1 baseline when
needed for AIV speedup. Failed spins sum over measured samples.

## AIC execution and output contract

The existing `camodel_paged_decode_attention` kernel and launcher retain their
fixed-grid validation ABI and standalone egress behavior. The same source now
also exports `turboquant_paged_decode_splitk_aic` and
`launch_turboquant_paged_decode_splitk_aic` with runtime geometry:

```text
requests, heads, block_size, cache_pages, seq_pages,
splits, page_begin, page_count
```

The selected page window is `[page_begin, page_begin + page_count)`. Every
request has a `seq_pages`-entry block table. Workers partition the selected
window with `ceil(page_count / splits)`, including empty workers. The logical
grid is `requests * heads * splits`; MIX still uses one AIC and two AIV roles
per logical block. All 16 query rows contain valid distinct fixtures.

For `item = ((request * heads + head) * splits + split)`:

```text
out_state[item * 256] : [m: 16 rows x 8 replicas][l: 16 rows x 8 replicas]
out_acc[item * 4096]  : [16 rows x 256 FP32 channels]
out_qrot[item * 4096] : [16 rows x 256 BF16 channels]
```

Query rotation runs once per nonempty worker. Each page executes the existing
eight-strip modulo schedule with the original per-strip mode-4 handshakes.
After draining the Cube's score-free waits, full-page completion flags 10 and
26 let AIV0/AIV1 reuse the page's state/rings/L1 key slots safely. AIV0 seeds
state once, carries it across pages and publishes only its final partial.
Empty workers write exact neutral `m=-3e38, l=0, acc=0` records and issue no
query rotation or page handshakes.

Partial outputs are checked against FP64 references, including every active
query-rotation BF16 value and every replicated state lane. They can be consumed
by an external log-sum-exp reduction. The measured AIC path does **not** include
that external combine or final egress, while AIV includes in-place combination
and tail egress. The table explicitly labels AIC as a model and reports only
the first tested context where its amortized time is lower, not a proven
production end-to-end crossover. Its rows share one KV sequence per head,
as packed GQA queries do; they are not 16 independent KV cache sets.

## Verification

- Strict CANN 9.2.0-beta.2 builds target `dav-3510` / `Ascend950PR_9599`.
  Device compilation uses `-Wall -Werror`; benchmark compilation uses
  `-Wall -Wextra -Werror`. No compiler warnings/errors. CANN's nested CMake
  retains its pre-existing unused-variable configuration warnings.
- Host M=16 probe passes all nine requested combinations
  `N={1024,4096,16384}`, `K={1,4,16}`, plus uneven/empty partitions, two requests,
  multiple heads, distinct query rows and nonzero page windows. Independently
  combined partials match the full-context dense-rotation reference.
- A frontend unit check using supplied input numbers verifies single-mode
  override precedence, `/16` normalization, physical versus 16x logical
  bandwidth, and CSV field alignment. Those inputs are not device measurements.
- CAModel comparison smoke: N=48, 16-token pages, one KV head, AIV M=1 and
  AIC M=16, K=2 and K=4. Both implementations match references. K=2 carries
  state across two pages; K=4 includes an empty worker. Exactly AIV=2/AIC=2
  kernel launches, with no implicit K=1 or warmup launches.
- Completed smoke trace: 171 balanced HardEvent pairs, no reserved IDs 6/7,
  mode-4 sets/waits balance per physical block. Six page-completion handshakes
  reach each AIV role. Exactly 368 Cube Mmads establish five query prologues
  (`5*64`) and six score strips (`6*8`), proving once-per-worker rotation.
  See [trace census](decode_aic_comparison_trace.json).
- CSV/window test: N=48, page_begin=1, page_count=1, two requests, K=2,
  AIC M=16, with `--single --warmup 9 --iters 7`. The run prints warmup=0,
  iters=1, validates all partials, and emits one CSV row with AIC=1/AIV=0.
- All-empty window: N=32, page_count=0, K=4, AIC M=16. Exact neutral records
  validate and a one-row CSV is emitted with AIC=1/AIV=0.

- Legacy AIC isolated canary: all numeric, rotation, context and inactive-slot
  guard checks pass through the unchanged legacy launcher ABI.
- Full eight-strip page: N=128, one head, K=1, AIC M=16, one launch. All partial
  states and query rotations match references. The trace contains 128 Mmads
  (`64` prologue + `8*8` score), 76 balanced HardEvent pairs, both original
  modulo-2 slot handshakes and one full-page completion per AIV role.

Evidence: [formatted comparison smoke output](decode_aic_comparison_smoke.txt),
[nonzero-window CSV](decode_aic_window_smoke.csv),
[empty-window CSV](decode_aic_empty_smoke.csv),
[eight-strip CSV](decode_aic_eight_strip_smoke.csv).

## Commands

```bash
bash scripts/run_decode_splitk_bench.sh build
bash scripts/run_decode_splitk_bench.sh smoke

# Hardware single-shot command (one launch per selected implementation):
bash scripts/run_decode_splitk_bench.sh hw \
  --single --context 2048 --splitk "4" --impl both --csv-out single.csv

# Hardware comparative sweep:
bash scripts/run_decode_splitk_bench.sh hw \
  --context "1024,4096,16384" --splitk "1,4,16" --impl both --csv-out sweep.csv

# Completed short smoke trace:
python3 scripts/analyze_aic_splitk_trace.py /tmp/compare-splitk-sim-build \
  --workers 5 --pages 6 --strips 1
```

The hardware single-shot command was attempted. This host has no `npu-smi` or
`/dev/davinci*`, so the hardware runner exits with status 3. Native NPU timing,
`msprof op` capture, and the large-context device crossover sweep remain
unverified. CAModel event timing is unavailable, so its performance cells
correctly remain `NA`. A large-context CAModel run is not an immediate operation
even when its host launch count is one.

The previously frozen AIC decode source was changed as explicitly requested
for this task. `turboquant_reshape_and_cache_bf16.cpp` and
`routine_vector_egress_rotate.hpp` remain untouched. Pre-existing edits to the
single-token CAModel runner/script and profiler output were preserved.
