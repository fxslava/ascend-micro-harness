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
| Cube (MIX_AIC_1_2) decode, wired differently — see §5 | `kernels/ops/turboquant/turboquant_paged_decode_attention.cpp` |
| the AIC partial oracle, masked | `tests/common/aic_splitk_decode_fixture.hpp` |
| unaligned AIC rungs | `tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp` |
| live tokens, beyond-S canaries, masked oracle | `tests/common/splitk_decode_fixture.hpp` |
| exhaustive geometry check | `tests/cpu/test_decode_tail_geometry.cpp` |
| fixture and oracle probe | `tests/cpu/test_decode_tail_probe.cpp` |
| the unaligned ladder | `tests/npu/test_npu_decode_single_token_unaligned.cpp` |

## 1. The mask is a selection, and there were two NaN paths

The first revision of this work masked the score lanes with a bit-masked
`Duplicate`, and recorded one unverified assumption: that a vector instruction
leaves its *deselected* destination lanes unmodified. A physical 950PR run
**disproved it**:

```
CANARY LEAK S=1 item=0 row=0 c=0: context is not finite,
a cache slot past seq_len reached the softmax
```

The CANN 9.2 `dav_3510` implementation shows why it was never a safe bet. The
bit-mask form of `Duplicate` goes through `VecDupLevel0Template`, which
allocates a temporary UB buffer and issues its own `V_S` fence — a great deal
of hidden machinery to rest "the other lanes keep their old value" on, and a
property no header in the toolkit states.

It is now an explicit selection, and that is a different mechanism rather than
a different spelling: a `Select` **writes every lane**, choosing per lane
between two sources from a predicate, so there is no preserved lane to depend
on and the canary cannot survive by omission.

```
select_tail(active)   Compares(tail_sel, lane_index, active, CMPMODE::LT)
mask_tail_scales()    Select(key_scale,   tail_sel, key_scale,   zero_fill)
                      Select(value_scale, tail_sel, value_scale, zero_fill)
mask_tail_scores()    Select(scaled_row,  tail_sel, scaled_row,  masked_fill)
clamp_tail_probs()    Select(p_scaled,    tail_sel, p_scaled,    zero_fill)
```

### The second NaN path, which masking the scores did not close

Equally important, and a mistake in the first revision's own reasoning. It
claimed "phases 2 to 5 need no lane count at all". That is false:

```
fold_value_scale:  p_scaled = p * value_scale
```

reads the per-token **value** scale, and a dead slot holds the canary there
too. The score mask makes `p` exactly `0` on a dead lane, but `0 * NaN` is
NaN, so the accumulator was poisoned through a path the score mask never
touched — while the denominator `l` stayed clean, because it sums `p`, not
`p_scaled`. That is precisely the observed failure: a non-finite **context**
with a plausible `l`. It would have leaked at S = 1 even if the `Duplicate`
had behaved.

### Why the scales are masked before the multiplies, not after

Masking only the *results* left both multiplies reading a dead lane's canary
and relying on a later `Select` to discard the answer. That passes, and the
CAModel still logged **306** `vec_err_idata_inf_nan_t0` errors — NaN reaching
a vector instruction's input. Masking the two scale vectors first takes that to
18 (§5). A kernel that hands NaN to the vector unit and overwrites the result
is one assertion away from failing, and it fills the model's error log so a
real hazard could not be seen in it.

`mask_tail_scales` runs once per strip, in the strip entry point, because the
two scales are strip inputs shared by all 16 queries. The probability clamp is
now redundant by construction — a zeroed value scale already gives
`p_scaled = 0` — and is kept anyway: one instruction on one strip, on a path
that has leaked once.

The host oracle masks the same tokens and skips the dead payload rather than
weighting it by zero, for the same NaN reason. The comparison is between two
independently masked computations.

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

## 4. The device gates: what has run, and what has not

This table changed substantially once the CANN 9.2 container was used (§7).
The device half of this work is now compiled and **run**.

| gate | status | evidence |
| --- | --- | --- |
| the bit-mask `Duplicate` assumption | **DISPROVED** | a physical 950PR run leaked the canary at S = 1; see §1 |
| all three decode kernels compile for `dav-3510` | **PASS** | CANN 9.2.0-beta.2 container, `ccec` + `ld.lld -m aicorelinux`, zero errors |
| the masking phase compiles | **PASS** | same build; `CreateVecIndex`, `Compares` and `Select` all resolve for `dav_3510` |
| the UB arena still fits | **PASS** | the four new operands are inside `kArenaEnd`, which every kernel `static_assert`s |
| S = 1 on CAModel (`Ascend950PR_9589`, `dav_3510`) | **PASS** | 1 live token, 15 canary lanes; context finite and within tolerance of the FP64 oracle |
| S = 17 on CAModel | **PASS** | one full strip then a strip with 1 live lane |
| canaries beyond S untouched | **PASS** | the runner asserts it per rung; `0x7FC0BEEF` in both scale planes |
| the zero-active bypass | **PASS** | 2 split ranges per rung landed entirely past the live tail and published the exact neutral element |
| NaN never reaches a vector instruction's input | **NOT CLEAN** | 306 → 18 `vec_err_idata_inf_nan_t0`, still 18 at 2 PCs; see §5 |
| the full ladder (S up to 1023) on CAModel | **NOT RUN** | deliberately: hours of model time, and longer rungs add pages, not tail cases |
| anything on silicon | **NOT RUN** | the leak that started this was reported from a 950PR; the fix has not been back-tested there |
| `NpuTargets.cmake` / `SimTargets.cmake` parse | **PASS** | and now a real build, not just a parse |
| every harness compiles strictly on the host | **PASS** | host stubs, `-Wall -Wextra -Werror`, host gate stage 4 |

Run detail, from `scripts/run_decode_tail_container.sh`: 2 rungs, 2 plain-AIV
launches and 4 split-K launches, 16 AICs × 2 vector cores, 427,102 ticks,
545,790 ms of model time, `run status: 0`.

**What a pass here does and does not mean.** It means the mask works at the two
tail shapes, on the real `dav_3510` model, against an independently masked FP64
oracle, with a quiet-NaN canary in every dead slot. It does not mean the
operator is clean: §5 records 18 remaining NaN-input errors, and nothing here
has run on silicon since the fix.

The frozen Cube path is unchanged by reading rather than by compiling:
splitting `update_max_and_alpha` into `scale_row` + `reduce_max_and_alpha`
leaves the same instructions in the same order (`Mul`, barrier, `ReduceRepeat`,
barrier, `Brcb`, barrier, `Max`, `Sub`, `Exp`), and both halves are
`__aicore__ inline` and called back to back.

## 5. The Cube operator, and why it is wired differently

`turboquant_paged_decode_attention.cpp` is now wired too, but not the way the
two AIV kernels are, and the difference is forced by the cross-core protocol
rather than chosen.

In `KERNEL_TYPE_MIX_AIC_1_2` the Cube announces a score panel per strip
(`kFlagScoreReady`) and waits for the Vector core to release the slot
(`kFlagScoreFree`). The Cube's per-page strip count is **one value for the
whole page range** — `run_cube_core(..., strips, range.count, true)` — while
the live strip count varies per page, since only the last live page is short.
Shortening the Vector core's trip count the way the AIV kernels do would leave
the Cube's announcements unconsumed and its waits unsatisfied: a hang, not a
wrong number. Giving the Cube a per-page count would mean restructuring the
frozen, verified strip pipeline.

So here the strip count stays at **capacity**, every flag edge is issued
exactly as before, and only the arithmetic inside `RecurStrip` varies — three
ways, on a runtime `active`:

| `active` | what runs |
| --- | --- |
| 16 | `vector_decode_recur_strip`, the unchanged aligned path |
| 1..15 | `vector_decode_recur_strip_masked` |
| 0 | **no recurrence at all**; the flags are still exchanged |

Skipping a fully dead strip rather than masking it is what makes this correct
without the trip-count change. Masking all sixteen lanes has the degenerate
failure of §2 — the centred row is exactly 0, `exp(0)` is 1, and the strip adds
16 to the denominator — and in this kernel that case is reachable, because a
split whose entire page range lies past the live tail has no live strip to
precede the dead ones. Carrying `(m, l, Acc)` untouched is exactly a zero
contribution and is right regardless of order.

That also makes an all-dead split publish the same exact neutral element as an
empty split, by a different path through the kernel. The AIC ladder now
exercises both: `tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp`
gained seven unaligned rungs, including two with whole dead pages, and counts
the split ranges that have pages to walk but no live token in any of them.
`ReferenceAicPartial` masks the same tokens, so its untouched neutral state is
what the comparison expects.

The legacy validation entry (`camodel_paged_decode_attention`) keeps its ABI
and declares its whole strip window live, so its ladder is bit-identical.

## 5a. The 18 that are left

Masking the two scales took `vec_err_idata_inf_nan_t0` from 306 occurrences at
6 PCs down to **18 at 2 PCs** — 6 at one site and 12 at another, across 6
launches, so once and twice per launch respectively. The run passes; these are
the model reporting NaN or Inf reaching a vector instruction's *input*, which a
kernel can do and still emit a correct answer if it overwrites the result.

It is not diagnosed. The two sites are not the four that masking the scales
removed, and the PCs shifted with the code change, so they cannot be mapped
back from the earlier log. The obvious candidates are ruled out: the GEMV
multiplies dequantised int4 by a BF16 query, both finite by construction, and
`p_scaled`, `p_x4` and the value page are all clean once the scales are. A
per-launch count points at the prologue or the epilogue rather than the strip
loop.

Resolving it needs a trace with symbols — `ccec -g` and the instruction dump
the trace audit already parses for the reshape operator — not another guess.
Until then this is an open hazard with a known bound, recorded rather than
implied, and it is the reason the matrix above says NOT CLEAN rather than PASS.

## 5b. A hole the live-aware trip count opened

Worth recording because it did not exist before this work and is not a tail
bug at all. With a live-aware trip count, a window that starts past the last
live token walks no strip, so the plain single-token operator carries `l = 0`
into an epilogue that divides by it: 0/0, a NaN context from an entirely
different cause, easily mistaken for a canary leak.

`WindowCoversSeqLen` does not catch it — a window can cover every live token by
ending after them while also starting after them. `WindowHasLiveToken` does,
with its own static_asserts, and it is refused at both the device guard and the
launcher.

Split-K deliberately does **not** use it: there a worker with no live page is
required to emit the neutral element, and only the tail worker normalizes. As
long as `seq_len >= 1`, page 0 is live and the split owning it is not neutral,
so the sum the tail divides by is nonzero.

## 6. Deliberately not done

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

## 7. The toolchain: a CANN 9.2 container, and two corrections

**The correction that matters most: there is a CANN 9.2 container on this
machine, and it builds and runs this project.**

```
ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12   (image b82c08f530f9)
```

It is CANN 9.2.0-beta.2 — the version the project targets — and it has
everything the earlier reports said was missing: `dav-3510` in BiSheng's target
list, a `dav_3510` per-architecture API tree, and the `Ascend950PR_9599`
simulator. `CMakeLists.txt` already defaults `ASCEND_HOME_PATH` to
`/usr/local/Ascend/cann-9.2.0-beta.2` and `SOC_VERSION` to
`Ascend950PR_9599`, so the build needs no arguments at all inside it;
`scripts/run_decode_tail_container.sh` is the whole recipe. Every "needs CANN
9.2" and "needs Ascend950PR_9599" row in the earlier reports was therefore
answerable all along, and several are now answered.

The image is tagged `910b`, which is a red herring: the toolkit it carries is
not limited to that part.

### The second correction, about the WSL instance

Separately, earlier reports and the header of
`scripts/run_reshape_wave_host_gate.sh` state that this machine has **no CANN
toolkit** at all. That is also wrong. What is installed directly in the WSL
Ubuntu-22.04 instance — distinct from the container above — is:

| | |
| --- | --- |
| root | `/usr/local/Ascend` — `cann-8.5.0`, `ascend-toolkit`, `driver` |
| compiler | `cann-8.5.0/tools/bisheng_compiler/bin/bisheng`, plus `bishengir-compile` |
| simulators | ~38 of them, from `Ascend310` through `Ascend910_939x` and `Ascend920A` |
| env | `ASCEND_HOME_PATH` and friends **unset**; `set_env.sh` present but not sourced |

That 8.5.0 install cannot build this project, for three specific reasons — but
note that the conclusion the earlier reports drew from it, that the device legs
cannot run *at all on this machine*, was wrong, because of the container above.
The three reasons 8.5.0 itself is unusable here:

1. **No `dav-3510`.** The project targets arch35 / dav-3510. The installed
   BiSheng knows `dav-2002 2201 3101 310r6 c100 c310 l100 l210 m100 m201 n350
   t100`, and a search for `dav-3510` in the binary returns **zero** hits. The
   per-architecture API implementations are `dav_c100`, `dav_c220`, `dav_m200`,
   `dav_m300`, `dav_m310` — again no arch35.
2. **No `Ascend950PR_9599` simulator.** The set stops well short of it, so the
   CAModel leg is unavailable regardless of the compiler.
3. **No `ReduceRepeat`.** The shared decode routines use
   `AscendC::ReduceRepeat` throughout, and it appears in **zero** files of the
   8.5.0 headers. So even retargeted at a supported architecture, these
   routines would not compile. 8.5.0 and 9.2 are different API generations.

### What reading 8.5.0's headers settled

This was done before the container was found, and it narrowed §1's assumption
without closing it. The container then closed it the other way: the
assumption was **false**, and the physical run had already shown it. Kept here
because the mechanism is still the explanation.
`asc/include/basic_api/kernel_operator_vec_duplicate_intf.h` declares, as a
Level 0 API:

> `Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[], const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)`

which is exactly the parameter shape `mask_tail_scores` passes, in that order
and with those widths. And the `dav_c220` implementation of it is

> `SetMask<T, isSetMask>(mask[1], mask[0]);` then `vector_dup(dstLocal, scalarValue, repeatTime, dstBlockStride, 1, dstRepeatStride, 0);`

— it programs the hardware vector mask registers and then issues the
instruction, which is precisely the mechanism the tail mask relies on, and the
same mechanism every other masked call in this codebase uses.

What it does **not** settle: no header in the install says what happens to the
destination lanes the mask excludes. The doc comment is `dst[i] = scalar` and
nothing more, and a search of the whole implementation and interface tree for
any statement about unselected lanes returns nothing. So the assumption is
narrowed from "does this overload exist and is the call well-formed" — yes, on
this API generation — to "are deselected lanes preserved", which still needs a
device run on 9.2.

### One sharp edge found while reading

That implementation is wrapped in `if ASCEND_IS_AIV`, so the masked
`Duplicate` is a **no-op on a Cube core**. It is harmless here because the
recurrence runs on the Vector core in all three kernels, the MIX_AIC_1_2 one
included — `RecurStrip` is reached from `run_value_core`, not `run_cube_core`.
But anything that later moves softmax arithmetic onto the Cube would lose the
mask silently, which is the failure mode this whole report is about.

## 8. Reproducing

```bash
# the device leg: build for dav-3510 and run the two tail-tile rungs
docker run --rm -v "$(pwd)":/workspace -w /workspace \
    ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 \
    bash scripts/run_decode_tail_container.sh
# then grep the log for vec_err_idata_inf_nan -- a pass with those in it is
# not a clean run (section 5a)

# host: the arithmetic and the oracle, stages 4 and 6
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

# the Cube leg's unaligned rungs ride on the existing AIC suite
./build_sim/test_decode_splitk_aic_sim      # or build_npu/test_npu_decode_splitk_aic
```

Check §1's bit-mask assumption first. If it does not hold, every rung fails the
same way and the fallback named in `mask_tail_scores` is the fix — not the
geometry, which §3 has already settled.
