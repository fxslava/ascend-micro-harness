# Wave-v13 conditional CPM candidates

The standalone `solve_cpm.py` replays the retained Wave-v11 and Wave-v12 traces at
34,935 and 35,271 ticks. It searches a local, source-supported final-pair reorder;
it is not a full pipeline simulator or an exhaustive schedule search. No kernel
was implemented, compiled or simulated for these Phase 1 candidates. PipeSim's
generic frontier was not used because its matmul lowering omits this decode's
CCU dependencies and scalar issue costs.

## Ranked candidates

| Rank | Mutation / crossover | Conditional makespan | Saving vs 34,935 | Review |
|---:|---|---:|---:|---|
| 1 | K: final pair only, prepare slot-1 key scales before waiting for score 1 | 34,739 | 196 ticks (0.56%) | Narrowest candidate; recommended for later implementation approval |
| 2 | KV: also hoist the final value-scale preparation before score 1 | 34,739 | 196 ticks | Same critical path; extra code movement is dominated by K |
| — | K + KV | 34,739 | 196 ticks | KV already includes K; savings do not add |
| — | K with fully serialized scalar prefix and prior Vector work | 34,850 | 85 ticks | Conservative model sensitivity, not a separate implementation |
| — | Parent | 34,935 | 0 | Measured reference |

Predictions are conditional on unchanged instruction costs, cache behavior and
downstream task-tail costs. Adding a 457-tick pre-VF delay and 191 extra terminal
ticks to K gives 35,387 ticks, a regression of 452 ticks versus the parent. This
observed-scale sensitivity is not a statistical confidence interval or a bound.

## Legality and weight evidence

`PreparePairLogits<1>` gathers from the resident slot-1 scale tile and scales its
dedicated key-scale field. It does not consume the score panel. Both scale tiles
have joined their MTE2 ingress before the final pair, and the prior pair's
consumers are complete. Keep both score waits before the paired softmax VF and
retain all P/context replay ordering. The K candidate changes only this order:

```text
parent:  wait score0 → key0 preparation → wait score1 → key1 preparation → paired VF
K:       wait score0 → key0 preparation → key1 preparation → wait score1 → paired VF
```

No new buffers, events, data transfers or arithmetic are introduced. L1 reuse
still requires the previous PV's SLOT_FREE; packed-UB reuse has its separate
lifetime. `slot = t & 1` is retained. The bounded ownership graph checks
1/2/4/5/8/16 tiles and rejects a deliberately introduced packed/L1-credit cycle.
This graph is not an exhaustive hardware event verifier.

The final score-1 wait windows are 24,761–24,974 on AIV0 and
24,778–24,974 on AIV1. Previous key-0 Vector work ends at 24,872/24,889.
Instruction traces assign 115 ticks to scalar descriptor preparation, 92 to
key-1 Vector work, and 185 to remaining value preparation/VF admission. Their
sum is 392 ticks. The checker retains single-queue Vector ordering and permits
scalar issue overlap with prior Vector service. K advances the slower AIV1
paired-VF start from 25,366 to 25,170: 196 ticks. Downstream two-AIV quorums,
PV2/PV3, output DMA, both AIV terminal sinks and AIC joins/END propagate that
change through a Max-Plus DAG. A separate event-readiness solver checks every
node against the longest-path result.

## Prologue, loop and epilogue constraints

- Prologue `read(0)` remains monolithic packed+scales. Earlier ingress is not
  ranked as a win: the measured split prologue regressed by 336 ticks, and the
  joined early-ingress Slot01 control regressed by 1,559 ticks (36,494 total).
- The elite loop already hoists packed `a+2,b+2` before the paired VF. Scale
  refills remain after their final Vector consumer. Imposing pack+scale atomicity
  on all steady-state refills would incorrectly forbid the elite schedule.
- The first pair has no score wait to hide this prefix behind. All-pair hoisting
  has no demonstrated extra saving, so the ranked K change is final-pair only.
- Wave-v12's 457-tick pre-VF excess and 194-tick branch-to-END gap are not
  removable Wave-v11 costs. Wave-v11's branch-to-END span is only 3 ticks.
- Cold prologue metadata misses and issue bubbles remain inside fixed upstream
  measured boundaries. No cold-miss saving is claimed.

## Resource and calibration limits

The model enforces the 216 KiB ceiling against the retained 192,384-byte
operator allocation claim and adds zero bytes. Every observed MTE2 span is
represented as 200 ticks initiation plus its measured service/contention
residual. Ordered issue is not full-transfer serialization: observed Q,
packed T0 and scale T0 transfer envelopes overlap on the same AIV.

Shared HBM contention is retained through the measured multi-core transfer
envelopes. The search does not advance ingress or change its competing AIC/AIV
traffic. Output transfers shift together, retaining their overlap; any newly
overlapping ingress/output windows are rejected. This is a constrained
contention-preserving model, not a fitted HBM arbitration model. It cannot
predict mutations that change upstream traffic or cache misses.

Exact replay of the two totals checks calibration consistency; it does not
independently validate a new order. Slot01 is a negative control, not a third
predicted run. Source/input hashes and all graph edges are in `results.json`.
K was subsequently measured at 37,203 ticks and 0 BF16 ULP. The 2,268-tick
regression leaves Wave-v11 as the native reference. See the
[four-tile CAModel result](../wave-v13-k/README.md). The predicted plots retain
the original model inputs and assumptions.

## Reproduce

```bash
python3 /mnt/d/Projects/ascend-micro-harness/reports/sync_schedule_ablation/wave-v13-cpm/solve_cpm.py
python3 /mnt/d/Projects/ascend-micro-harness/reports/sync_schedule_ablation/wave-v13-cpm/render_overview.py
```

`predicted-gantt.png`/`.svg` show final-pair packing and downstream sinks.
`predicted-overview.png`/`.svg` show selected engine spans over the full run.
