# Context tail masking in single-token decode, at arbitrary S

Decode kernels walked whole 16-token strips of whole pages and derived their
trip count from `seq_pages * strips`. In generative decoding `S` grows by one
per step, so that count rounds the context **up** and admits cache slots this
sequence never wrote. This change separates allocated capacity from live
tokens, masks the one partial strip, and records which gates ran.

| piece | file |
| --- | --- |
| the arithmetic, host and device | `kernels/ops/turboquant/turboquant_decode_tail.hpp` |
| the masking phase, shared by all three decode kernels | `routines/routine_vector_decode_recurrence.hpp` |
| plain AIV decode, wired | `kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp` |
| split-K AIV decode, wired | `kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp` |
| live tokens, beyond-S canaries, masked oracle | `tests/common/splitk_decode_fixture.hpp` |
| exhaustive geometry check | `tests/cpu/test_decode_tail_geometry.cpp` |
| fixture and oracle probe | `tests/cpu/test_decode_tail_probe.cpp` |
| the unaligned ladder | `tests/npu/test_npu_decode_single_token_unaligned.cpp` |

## 1. The mask is a write, not a multiply

This is the one design decision in the change, and it is forced rather than
chosen.

The obvious masking is multiplicative: keep lanes `[0, active)` and multiply
the rest by zero. That is wrong here, because the per-token key **scale** of a
never-written cache slot is an arbitrary FP32 bit pattern, and under the
isolated-canary protocol this harness already uses
(`tests/camodel/test_camodel_single_token_decode_aiv.cpp`) it is deliberately
`0x7FC0BEEF`, a quiet NaN. `0 * NaN` is NaN. A multiplicative mask would
propagate the canary straight through `m`, `l`, `Acc`, the butterfly and the
gate — and fail the very test meant to prove the mask works.

So `mask_tail_scores` **overwrites** the dead lanes with `kMaskedScore`
(`-3.0e38f`) using a bit-masked `Duplicate`, discarding whatever the unmasked
`Mul` computed there. `exp(kMaskedScore - m)` then underflows to `+0`, so a
masked lane adds exactly `0` to the denominator and exactly `0` to every
channel of the accumulator. There is deliberately no second place where a lane
count has to be right: phases 2 to 5 are the aligned ones, untouched.

The same NaN hazard exists on the host, and the oracle handles it the same way
— it skips the dead payload rather than weighting it by zero. The comparison is
therefore between two independently masked computations.

**The one assumption, and the first thing a device run must check:** that in
bit-mask mode a vector instruction leaves the *deselected* destination lanes
unmodified. If this part does not honour that, the fallback is `Compare` into a
select mask plus `Select` in TENSOR_SCALAR mode — one more scratch operand and
a 16-lane ramp, but no assumption about untouched lanes. The fallback is named
in the code at the point it would be needed.

## 2. Dead strips are dropped, not masked

`kMaskedScore` has exactly one degenerate input: a strip with **zero** live
lanes. With every lane masked, `m_new` is `kMaskedScore`, the centred row is
exactly `0`, `exp(0)` is `1`, and the strip contributes `16` to `l` instead of
`0`. That is the single way this mask produces a wrong answer rather than a
no-op.

`LiveStripsInWindow` is what prevents it: dead strips never enter the trip
count. It relies on the walk's token base rising with the index, which holds
exactly when `strips <= block_size / 16` — the window rule both kernels already
enforced. `WalkIsMonotone` names it so the launcher and the device check the
same predicate.

## 3. What was verified, and how

Everything in this section ran on this machine.

| check | scale | result |
| --- | --- | --- |
| `ActiveInStrip` against a per-token count | 4,198,401 `(base, seq_len)` pairs | **PASS** |
| every strip is exactly one of dead / full / partial | same sweep | **PASS** |
| the walk partitions its live tokens exactly once, and the live strips are the walk's **prefix** | 639,360 windows over `block_size` 16/32/128, every in-page window, `page_begin` 0..2, `seq_pages` 1..8, `seq_len` 0..capacity+16 | **PASS** |
| the aligned default keeps the pre-tail trip count, and never enters the masking phase | 11,264 window geometries | **PASS** |
| negative controls rejected | 4 | **PASS** |
| beyond-S slots carry `0x7FC0BEEF` in both scale planes, and live slots do not | 11 rungs | **PASS** |
| the masked FP64 reference is finite — the oracle never read a dead slot | 11 rungs | **PASS** |
| masking changes the gated context by more than the device tolerance | 11 rungs | **PASS** |
| aligned length is a bit-identical no-op (control for the line above) | S = 512 | **PASS** |

The partition check is the one that matters most: a trip count alone cannot see
a token visited twice (counted twice in the denominator) or skipped (missing
from the softmax). A bitmap comparison sees both.

The four negative controls are each a plausible wrong implementation — every
strip full (the pre-fix behaviour), the partial strip rounded up to 16 lanes, a
truncating trip count that drops the partial strip, and the `(v + d - 1) / d`
round-up that overflows. The last one returns **0** strips for `2^32 - 1`
tokens, which is why `CeilDiv` is written the other way.

### Observability, and an honest margin

`test_decode_tail_probe` measures how far masking moves the final gated
context, against the 0.005·row_max + 1e-6 tolerance the device comparison uses:

| S | dead slots | max abs diff | tolerance | margin |
| --- | --- | --- | --- | --- |
| 1 | 127 | 1.05 | 5.3e-3 | ×199 |
| 7 | 121 | 0.187 | 9.7e-4 | ×194 |
| 15 | 113 | 0.179 | 9.0e-4 | ×198 |
| 17 | 111 | 0.159 | 8.2e-4 | ×193 |
| 31 | 97 | 0.0885 | 5.5e-4 | ×161 |
| 33 | 95 | 0.0820 | 5.2e-4 | ×158 |
| 127 | 1 | 3.95e-4 | 1.5e-4 | **×2.6** |
| 129 | 127 | 1.53e-2 | 1.5e-4 | ×105 |
| 255 | 1 | 2.10e-4 | 6.8e-5 | **×3.1** |
| 257 | 127 | 3.58e-3 | 5.8e-5 | ×61 |
| 1023 | 1 | 5.52e-5 | 2.8e-5 | **×1.9** |

At the rungs where only **one** token is dead (127, 255, 1023) the numeric
margin is under a factor of 3, and at S = 1023 it is 1.9. So on those rungs the
arithmetic comparison is close to being unable to tell a masked kernel from an
unmasked one, and the real discriminator is the NaN canary, which fails hard
rather than by a margin. Both checks are in the harness; neither is redundant.

## 4. What has NOT run

This host has **no CANN toolchain and no 950PR**. Recorded per `AGENTS.md`.

| gate | status | why |
| --- | --- | --- |
| the masking phase compiles | **NOT RUN** | needs BiSheng; `routines/routine_vector_decode_recurrence.hpp` only compiles under `__CCE_AICORE__` |
| the bit-mask assumption in §1 | **NOT ESTABLISHED** | the whole change rests on it and nothing here can test it |
| either kernel compiles or links | **NOT RUN** | needs CANN 9.2 |
| the unaligned ladder on CAModel | **NOT RUN** | needs Ascend950PR_9599 |
| the unaligned ladder on silicon | **NOT RUN** | needs a 950PR |
| `NpuTargets.cmake` / `SimTargets.cmake` parse | **PASS** | isolated `cmake` include; a **parse**, not a build — the targets live inside `function(...)` bodies that this never executes |
| the new harness compiles strictly | **PASS** | host stubs, `-Wall -Wextra -Werror`, host gate stage 4 |
| the three already-committed split-K harnesses still compile | **PASS** | the fixture change is source-compatible |

**No rung of the unaligned ladder has executed.** The device half of this
change — the two instructions that do the masking — has not been compiled, let
alone run. What has been established is which tokens are live, that the oracle
masks them correctly, and that masking is observable. That is the half this
machine can settle, and it is not the half that proves the kernel.

The frozen Cube path is unchanged by reading rather than by compiling:
splitting `update_max_and_alpha` into `scale_row` + `reduce_max_and_alpha`
leaves the same instructions in the same order (`Mul`, barrier, `ReduceRepeat`,
barrier, `Brcb`, barrier, `Max`, `Sub`, `Exp`), and both halves are
`__aicore__ inline` and called back to back.

## 5. Deliberately not done

- **The Cube decode operator is not wired.**
  `turboquant_paged_decode_attention.cpp` still derives its trip count from
  capacity and still rounds the context up. `vector_decode_recur_strip_masked`
  exists in the shared routine so the fix is reachable from there, but its
  strip loop, its launcher and its verified eight-strip ladder have not been
  touched. The AIC leg of the directive's §1 is therefore **open**.
- **`out_scores` and `out_state` are not compared** in the unaligned harness,
  only the end-to-end gated context plus the output guard band. There is a
  specific trap here worth recording: the score archive is written from the
  panel **after the GEMV and before the recurrence**, so on the partial strip
  it holds the *unmasked* row — the dead lanes carry whatever the canary
  nibbles and NaN scale produced. That is correct behaviour (the archive is a
  record of the GEMV, not of the softmax), but it means any future bit-exact
  check of `out_scores` at an unaligned length has to mask the oracle's panel
  itself. The aligned CAModel runner gates those panels bit-exactly today and
  is unaffected.
- **One request per rung.** Found while writing the harness, and worth
  recording as a limitation of the kernel rather than of the test: the plain
  AIV operator indexes its block table as `block_table[page_begin + ...]` with
  **no request stride**, so it is only defined for a single request. The
  split-K operator uses `request * table_stride + ...` and has no such limit.
  The already-committed aligned AIV suite covers two requests.
- **`LiveStripsInWindow` is a scalar loop, not a closed form.** It runs once
  per launch and stops at the first dead strip, so at the geometries here it is
  at most 64 iterations. At production context lengths — hundreds of pages — it
  is hundreds of scalar iterations per block, still once per launch but no
  longer free. A closed form is available (the live set is a prefix, so the
  count is `page_index * strips + live_in_that_page`); the loop was kept
  because it is `constexpr`-evaluable and therefore checkable exhaustively,
  which an algebraic rearrangement is not. If a profile ever shows it, replace
  it and keep the loop as the test oracle.
- **`WorkerMetrics.pages`** still reports the split's *capacity* page count, so
  the rig's existing census (`w.pages[0] == Partition(...).count`) holds
  unchanged at an unaligned length. A live-page count would be a new field and
  a new census rule; neither was added.

## 6. Reproducing

```bash
# host: everything this machine can establish, stages 4 and 6
bash scripts/run_reshape_wave_host_gate.sh

# the two tail checks on their own
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. tests/cpu/test_decode_tail_geometry.cpp -o /tmp/g && /tmp/g
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I. tests/cpu/test_decode_tail_probe.cpp -o /tmp/p && /tmp/p

# with a CANN toolkit, the only leg that RUNS the ladder without a 950PR
cmake --build build_sim --target test_decode_single_token_unaligned_sim
./build_sim/test_decode_single_token_unaligned_sim

# with a 950PR
cmake --build build_npu --target test_npu_decode_single_token_unaligned
./build_npu/test_npu_decode_single_token_unaligned
```

Check §1's bit-mask assumption first. If it does not hold, every rung fails the
same way and the fallback named in `mask_tail_scores` is the fix — not the
geometry, which §3 has already settled.
