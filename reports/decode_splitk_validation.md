# Split-K decode validation: AIC (M=16) and AIV (M≤8)

Validation suites for the two split-K decode kernels, and an explicit record of
which gates have been **run** versus merely **built**. Directive 3.

| kernel | entry point | launch type |
| --- | --- | --- |
| AIC partial decode | `turboquant_paged_decode_splitk_aic` | `KERNEL_TYPE_MIX_AIC_1_2`, M=16, d=256 |
| AIV single-token decode | `turboquant_single_token_decode_splitk_aiv` | AIV-only, M≤8, d=256 |

Sources:

| file | role |
| --- | --- |
| `tests/npu/turboquant_splitk_rigs.hpp` | both device rigs: buffers, launch, full census |
| `tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp` | AIC ladder |
| `tests/npu/validation/test_turboquant_decode_splitk_aiv_validation.cpp` | AIV matrix |
| `tests/common/aic_splitk_decode_fixture.hpp` | `ReferenceAicPartial`, `AicRawQuery` (FP64 partial oracle) |
| `tests/common/splitk_decode_fixture.hpp` | `Fixture`, `Fixture::expected` (FP64 full-context oracle) |

## 1. Where the checks come from

The two rigs were **not written for this report**. They already existed inside
`bench_turboquant_decode_splitk.cpp`'s anonymous namespace as `Device` and
`AicDevice`, and they already carried the full correctness census for both
kernels. This change lifted them into `tests/npu/turboquant_splitk_rigs.hpp` as
`AivRig` and `AicRig`, and the benchmark now includes that header and keeps its
old names through two aliases.

That is deliberate and it is the main structural claim here: the benchmark and
the validation tests run the **same** checks, not two copies. A tolerance, a
neutral element or a torn-payload rule that drifted between a bench copy and a
validation copy would make one of them pass and the other fail while saying
nothing about the kernel. The bench's copy is the one with the passing history,
so it is the one that moved.

What is new is the **ladders** — which geometries get launched — plus the build
targets and this record.

## 2. AIC: what each rung checks

Per rung, for all 16 query rows of every `(request, head, split)` item, against
`ReferenceAicPartial` (independent FP64 softmax over the uploaded bytes, read
through the uploaded block table, no online merge):

- emitted partial state `(m, l)`, and that its 8 replicated lanes are not torn;
- the partial accumulator `Acc` over all 256 channels;
- the rotated query `Q_rot`, **bit-exact** against the BF16 reference;
- for a split whose page range is empty, the exact neutral element
  (`m == -3e38f`, `l == 0`, `Acc == 0`) and an untouched `Q_rot` canary
  (`0xAAAA`).

Geometry: 128-token physical pages, 16 packed query rows, 256 channels,
2 requests × 2 heads, so the item index the kernel derives from its block id is
exercised in all three dimensions rather than only in the split.

| pages | why | K |
| --- | --- | --- |
| 1 | boundary canary | 1, 2, 4, 8 |
| 4 | plain multi-page range | 1, 2, 4, 8 |
| 8 | multi-temporal split | 1, 2, 4, 8 |
| 3, 5, 7 | non-power-of-2: unequal and empty `Partition` ranges | 1, 2, 4, 8 |

Each rung with more than one page also runs at `page_begin = 1`. Without that,
a kernel that ignored the window would still pass every check above, because
the oracle is handed the same window and would simply agree on the full range.
The CPU probe pins that the window is numerically observable; this pins that
the kernel honours it.

Where `K` exceeds the page count, `Partition` hands out empty ranges, so the
neutral-element path is reached by the geometry rather than by a special case.

## 3. AIV: what each rung checks

Per rung, through `AivRig::Validate`:

- the L2 `AtomicHeadSlot` protocol — every worker acquired the lock, none hit
  the watchdog, the arrival tickets over the K workers form a permutation of
  `1..K`, exactly one worker is the tail, and each head slot ends drained
  (`lock == 0`, `arrived_count == K`);
- the per-worker metric payload is not torn across its 8 replicated lanes, and
  each worker's page count equals `Partition`'s for its split;
- the final context — normalized, butterfly-rotated, gated, BF16 — against
  `Fixture::expected`, which applies `DenseRotate` rather than the kernel's
  factorization, so the egress **ordering** is checked rather than assumed.

Geometry: 4 pages of 128 tokens, 4 KV heads, 2 requests, 256 channels.

| M | K |
| --- | --- |
| 1, 2, 4, 6, 8 | 2, 4, 8 |

`M = 1` is the single-token boundary, `M = 8` the kernel's maximum, `M = 6` the
default the rest of the harness uses. `K = 1` is deliberately absent: with one
worker there is no contention and the tail is the first arrival, so it would not
exercise the protocol this test exists for. The benchmark covers `K = 1`.

Contention is concurrent rather than nominal: `requests * heads` = 8 slots are
contended by `requests * heads * K` workers in a single launch — 64 workers at
`K = 8`. `failed_exchanges` is reported, not asserted on: a nonzero count is
the compare-and-swap retrying, which is what contention looks like from inside,
and is the number that says the race actually happened.

## 4. Build targets

| target | file | leg |
| --- | --- | --- |
| `test_npu_decode_splitk_aic` | `cmake/NpuTargets.cmake` | physical 950PR |
| `test_npu_decode_splitk_aiv` | `cmake/NpuTargets.cmake` | physical 950PR |
| `test_decode_splitk_aic_sim` | `cmake/SimTargets.cmake` | CAModel |
| `test_decode_splitk_aiv_sim` | `cmake/SimTargets.cmake` | CAModel |

All four at `-Wall -Wextra -Werror`. The simulator twins matter more than most
here: the NPU legs need a 950PR this harness's development host does not have,
so for anyone without one the sim legs are the only place these ladders run
rather than merely compile.

## 5. Verification matrix — what has actually been run

This host has **no CANN toolchain and no 950PR**. Recorded per
`AGENTS.md`: unavailable gates are named, not implied.

| gate | status | evidence |
| --- | --- | --- |
| AIC/AIV rigs extracted, bench unchanged in behaviour | **PASS** | `bench_turboquant_decode_splitk.cpp` compiles clean at `-Wall -Wextra -Werror` after the extraction |
| AIC validation harness compiles | **PASS** | host stubs, `-Wall -Wextra -Werror` |
| AIV validation harness compiles | **PASS** | host stubs, `-Wall -Wextra -Werror` |
| both added to host gate stage 4 | **PASS** | `scripts/run_reshape_wave_host_gate.sh` |
| `NpuTargets.cmake` / `SimTargets.cmake` parse | **PASS** | isolated `cmake` include; parse only — the function bodies are not executed, so this is **not** a build of the targets |
| AIC ladder executed on CAModel | **NOT RUN** | needs CANN 9.2 + Ascend950PR_9599 |
| AIC ladder executed on silicon | **NOT RUN** | needs a 950PR |
| AIV matrix executed on CAModel | **NOT RUN** | needs CANN 9.2 + Ascend950PR_9599 |
| AIV matrix executed on silicon | **NOT RUN** | needs a 950PR |
| device compile/link of either kernel | **NOT RUN** | needs BiSheng |
| "zero memory/sync hazards" | **NOT ESTABLISHED** | no run, and see §6 |

**The headline: not one rung of either ladder has been executed.** Everything
green above is a compile or a parse. A harness that compiles says nothing about
a kernel, and the directive's "run the tests, verify zero memory/sync hazards"
is the half that is still owed in full.

## 6. What these suites still would not establish, after they run

Worth stating before anyone reads a green run as more than it is — this is the
same failure mode that produced the wrong conclusion recorded in
`MICROROUTINE_DESIGN_GUIDELINES.md` §6.2c.

- **"Zero memory/sync hazards" is not a thing either test observes.** Both
  check emitted values and the metric/slot census. A missing drain can balance
  every census and still produce bit-exact output while a buffer is read a
  generation early. For the reshape kernels that gap is covered by
  `scripts/audit_cross_core_fences.py`; **neither decode kernel has been
  through that audit**, and `turboquant_paged_decode_attention.cpp` recycles
  mode-4 ids over four announcements each. Running these ladders green would
  not close it. That audit over the decode kernels is the obvious next gate and
  it is not done.
- **Timing-dependent races need more than one schedule.** Each rung is one
  launch. The AIV arrival permutation is checked, but a lock that is correct
  only at the schedule the model happens to produce would pass. Repeated runs
  and a trace audit are what separate those, as §6.6's promotion criterion
  requires for the reshape operator.
- **The AIC rig compares against an oracle handed the same window**, so
  `page_begin` coverage rests on the windowed rungs differing from the
  whole-range ones — which the CPU probe establishes numerically, not these
  tests.
- **No trace audit or per-id event table** exists for either decode kernel, so
  §6.6's promotion criterion is unmet for both.

## 7. Reproducing

```bash
# host: compile-only, what this machine can establish
bash scripts/run_reshape_wave_host_gate.sh        # stages 3 and 4 cover these

# with a CANN toolchain, CAModel leg (the only leg that runs without a 950PR)
cmake --build build_sim --target test_decode_splitk_aic_sim test_decode_splitk_aiv_sim
./build_sim/test_decode_splitk_aic_sim
./build_sim/test_decode_splitk_aiv_sim

# with a 950PR
cmake --build build_npu --target test_npu_decode_splitk_aic test_npu_decode_splitk_aiv
./build_npu/test_npu_decode_splitk_aic
./build_npu/test_npu_decode_splitk_aiv
```

Update §5 with the rungs each leg actually executed, and do not mark a row PASS
from a clean exit alone where §6 says the exit is not the evidence.
