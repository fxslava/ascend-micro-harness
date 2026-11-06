# Fused decode: a BF16 instance alongside the FP16 one

Date: 2026-11-06
Operator: `kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp`
Target: Ascend 950PR / `dav-3510` / CANN 9.2.0-beta.2 / `Ascend950PR_9599`, `KERNEL_TYPE_MIX_AIC_1_2`
Protocol: `REFINEMENT_OF_COMBAT_KERNELS.md` §1–§5
Baseline: 2fd5d8e, audited in `reports/fused_decode_host_tiling_audit_20261106.md`
Environment: WSL2 Ubuntu-22.04, BiSheng, CAModel. **No physical NPU on this host** (`npu-smi` and
`/dev/davinci*` absent), so every device claim below is CAModel and every hardware gate is recorded
as pending in §7, not as passed.

---

## 1. What changed, in one paragraph

`TurboQuantFusedDecodeOpt<MODE, scalar_t, PRE_ROTATED, UNPACK>` is now instantiated at
`bfloat16_t` as well as `half`, and both pre-rotated and raw-query entries dispatch to either
instance on a new `Launch::element_type` field. `scalar_t` is the **query/output** element type and
not the cache format: the cache stays KV4/FP8, so `MODE`, the codec, the three-slot ring ingest,
the Cube's score and context GEMMs, the online-softmax recurrence, the Split-K partials and every
tiling descriptor are the same objects and the same instructions in both instances. The BF16
instance differs from the FP16 one in exactly three places — the output cast, the raw query and
gate the basis service reads, and the basis service's Cube chunk operand — and in nothing else. The
UB map does not move, because `sizeof(half) == sizeof(bfloat16_t) == 2`. The FP16 path's generated
code is unchanged (§3.4).

---

## 2. Where `scalar_t` actually reaches

Surveyed site by site before editing, because the answer decides whether this is a type
substitution or a second kernel. It is a type substitution.

| site | file | `scalar_t`-specific? |
|---|---|---|
| output GM tensor + the UB row cast into it | `vector/turboquant_vector_service.h` `WriteHeads` | **yes** — the operator's only output rounding |
| `TurboQuantPartialReducer` output | `vector/turboquant_vector_service.h` | **yes** — same pair, reached only on a Split-K launch |
| `TurboQuantGemmReducer` / `TurboQuantBatchedGemmReducer` output | `cube/turboquant_gemm_reduce_service.h` | **yes** — same pair |
| raw query + gate GM tensors, staging buffer | `cube/turboquant_query_basis.h` | **yes** — 16-bit inputs |
| basis Cube chunk Mmad operand | `cube/turboquant_query_basis.h` | **yes** — see §3.2 |
| `OperandT` (score/context GEMM operands, dequant expand, NZ staging) | `vector/turboquant_vector_service.h` | no — `OperandT = Mm::OperandT`, derived from `MODE` |
| scores, online softmax, probabilities, accumulator, Split-K partials and merge | all | no — fp32 in both |
| tile cursor, dispatch records, ring indices, cross-core flags | `turboquant_fused_decode.cpp` | no — integer words |
| host tiling descriptors | `include/turboquant_fused_decode.hpp` | no — counts, page indices, row bases |

The two reducers turned out to be entirely fp32 internally (their reduction GEMM takes `float`
operands), so they needed no arithmetic change at all — only the output pair.

---

## 3. The four places that needed a decision

### 3.1 The dequant path — checked, and not on this operator's route

`op_kernel/common/routine_turboquant_dequant4.hpp` documents that `int16_t -> bfloat16_t` is **not**
in the 950PR `Cast` table and emits no instructions at all on dav-3510. Confirmed against the local
API mirror: the 950PR table (`type_conversion/Cast.md` lines 104–174) lists `int16_t` sources to
`int4b_t`, `uint8_t`, `half`, `int32_t`, `uint32_t` and `float` — and no `bfloat16_t`. That makes
`NativeDequant4Engine::cast_and_scale`, the **unsigned**-nibble path, unusable at a `bfloat16_t`
output.

It is not on this operator's route. The fused decode expands its KV pages through the codec's
`UnpackAffine` / `UnpackAffineGather` at `OperandT`, which comes from `MODE` (KV4/FP8) and never
from `scalar_t`, so the expand is byte-for-byte the same in both instances. No change was made to
the dequant routine, and no BF16 caller was added to its unsigned path.

### 3.2 The basis service's Cube chunk operand — the one real behavioural change

The raw-query prologue rotates whole chunks of its share on the Cube: each vector subcore sign-flips
its half, casts to a 16-bit operand, stages it to L1, and the AIC multiplies by H16. That operand
was `half` unconditionally, and for an fp16 query it is exact, since the signs and H16 are ±1 and
`q -> fp32 -> ±q -> fp16` round-trips bitwise.

**It is not exact for a bf16 query.** bf16's 8 significand bits fit inside fp16's 11, so the
mantissa always survives — but the exponent range does not: |q| above 65504 overflows fp16 to
infinity and |q| below 2^-24 flushes to zero, and a bf16 query carries values in both ranges. A
bf16 query staged through an fp16 operand would therefore rotate a different vector than `ApplyPi`
rotates from the same input, so the chunked and unchunked halves of one launch's share would
disagree — silently, and only for the chunked half.

So the operand now follows `scalar_t`. Verified against the API mirror before writing it:

* `bfloat16_t × bfloat16_t -> float` is a listed Mmad combination on 950PR
  (`mmad_compute/Mmad.md` table 5, and table 6 with bias).
* The arch-3510 fractal is `16 × (32B / sizeof(T))`, so C0 is 16 elements for fp16 and bf16 alike.
  `op_kernel/cube/routine_stage_page_zn_bf16.hpp` derives that for bf16 and the two agree, so the
  `LoadData2DParamsV2` descriptors are unchanged. The constant was renamed `kHalfOperandC0` ->
  `kOperandC0` with the derivation written at its definition.
* `half -> bfloat16_t` is a listed `CAST_RINT` pair, so the H16 table stays **fp16 in GM for both
  instances** — it is a shared input and its entries are ±1, exact in both — and is converted once
  on ingest where the operand is not fp16. The converted copy goes behind the staged one inside the
  existing scratch buffer (2048 elements against this tile's 2×256), so no allocation was added and
  `kBasisUbBytes` does not move.
* `float -> bfloat16_t` (`CAST_RINT`) and `bfloat16_t -> float` (`CAST_NONE`) are both listed, which
  covers the staging cast and the input widening.

The fp16 branch is selected by `if constexpr (AscendC::IsSameType<OperandT, half>::value)` and keeps
the original instruction sequence verbatim.

### 3.3 The UB map — pinned, not re-derived

`include/turboquant_fused_decode_opt.hpp`'s `kUbEnd`, `kBasisUbBytes` and `kReducerUbBytes` are
unchanged, and that is load-bearing rather than incidental: the Cube's dual-destination Fixpipe
names the vector cores' landings by **offset** on three cores with nothing in the toolchain checking
that the three agree. Every element-type-dependent term in those figures is a 16-bit extent, and
both instantiated types are 16-bit, so both allocate the identical list at identical offsets.

To stop a third element type reaching that map quietly, `sizeof(scalar_t) == 2` static_asserts were
added to `TurboQuantPartialReducer`, `TurboQuantVectorDecodeService`, `TurboQuantGemmReducer` and
`TurboQuantBatchedGemmReducer`, and `sizeof(OperandT) == 2` to `TurboQuantQueryBasis`, each naming
the header a change would have to go through first. The reason is written down at
`kBasisUbBytes`/`kReducerUbBytes` where the numbers are.

### 3.4 The FP16 path is unchanged

* The device instruction sequence for `scalar_t = half` is textually unchanged everywhere. The only
  conditional added on that path is `if constexpr (IsSameType<OperandT, half>::value)` in the H16
  ingest, whose `true` branch is the original two `DataCopy`s and the original `MTE2_MTE3` fence.
* `Launch::element_type` defaults to `0` (= `AscendType::FP16`), so a caller that predates the field
  keeps the behaviour it had.
* `Fixture`'s element parameter defaults to `Element::kFp16`, so every existing fixture call site is
  unchanged. `tests/cpu/test_fused_decode_opt_fixture.cpp` pins bitwise that the FP16 and BF16
  fixtures produce identical fp32 `query`, `scales` and `expected` and identical `key`, `value`,
  `table` and `lengths`.

---

## 4. The launch path

`Launch` gained `std::uint32_t element_type = 0`, matching `vllm_ascend::AscendType` (0 = FP16,
1 = BF16). It names the query/output type only: not the cache format, which is the launch's `MODE`,
and not the tiling, whose descriptors are counts and indices and so serve both instances from one
buffer.

Both `turboquant_mm_fused_decode_opt_impl` and `..._raw_query_impl` now dispatch on `type` after the
shared geometry guards instead of returning silently on anything but FP16. The convenience ABI
(`launch_fused_decode_opt`, `launch_fused_decode_opt_raw_query`) maps the field through
`ascend_type_of`, which **throws** on an element type with no kernel instance rather than returning a
value the dispatch would drop on the floor — a launch that quietly computes nothing is the failure
mode the previous audit's SPU work was about. The simulator launcher applies the same rule.

Four device entry points now exist where there were two:

```
turboquant_mm_fused_decode_opt_kv4fp8_half            turboquant_mm_fused_decode_opt_kv4fp8_bf16
turboquant_mm_fused_decode_opt_raw_query_kv4fp8_half  turboquant_mm_fused_decode_opt_raw_query_kv4fp8_bf16
```

No new `ascendc_library` was added. The BF16 instances are additional instantiations inside the
existing translation unit, so the simulator leg, the physical leg and the staged integration legs
each pick them up without a second nested ExternalProject — which on this operator would have cost
three more full BiSheng passes per leg.

The simulator probes were likewise folded into one macro over `scalar_t`
(`ASCEND_TQ_DECLARE_FUSED_DECODE_OPT_PROBES`), so the BF16 canary cannot drift from the FP16 one in
geometry, tiling or the methods it calls. That is the only reason the two ladder legs are
comparable.

---

## 5. The gold model carries the element type

Rounding an fp32 oracle at the very end is the right model only where the device's own rounding is
at the very end. On this operator that is true of one value and false of two others, so the host
gold model was made format-aware rather than format-rounded:

| value | device | gold model |
|---|---|---|
| output | accumulates in fp32, casts **once** at the final GM burst | fp64 -> fp32 -> `reference_bits(element, ·)`; one rounding, same place |
| raw query | **reads** 16-bit from GM, widens to fp32 | built in the element format, and the oracle consumes the **widened** value, not the fp32 ideal it rounded from |
| gate logit | **reads** 16-bit from GM, widens to fp32 | `expected_row`'s `kGated` stage divides by `1 + exp(-gate_widened(i))`, never by `1 + exp(-gate_value(i))` |
| everything between | fp32 in both instances | fp64-then-fp32 in both; carrying it in 16 bits would model a kernel this is not |

`raw_query` and `gate_logits` **refuse** a value the format cannot hold exactly instead of rounding
it. That is not defensive dressing: the whole reason the raw-query entry can be held to the
pre-rotated entry's oracle at output stage 0 is that `Pi(raw)` reproduces the pre-rotated query bit
for bit, and an input the fixture had to round would make `expected` an approximation of a different
kernel's answer while the rungs still reported 0 ULP.

Comparisons are made on the **stored 16-bit word at its original position**, in all three of the
CAModel runner, the NPU rig and the NPU validation test. Nothing widens the device's word before
comparing, which is what §3 of the protocol asks for on BF16: a check that widened both sides, or
that accepted either half of an fp32 word, would pass on a halfword-swapped output.

`ascend_micro::sim::bfloat16_to_float` was added to both hand-synced copies of
`include/sim_micro_testbed.hpp`; `float_to_bfloat16_rne` already existed.

---

## 6. Numerical evidence

### 6.1 Every exact-arithmetic claim survives BF16 at the FP16 budget

The ladder's exact rungs demand **0 ULP** of the output format and that was **not relaxed** for
BF16. `scripts/check_fused_decode_bf16_margins.py` derives why, per rung: the distance from each
rung's exact rational to the nearest rounding **midpoint** of each format's grid, in units of the
value's own fp32 ULP. A device fp32 result must be off by more than that margin before the
comparison can flip.

| rung | fp16 | fp16 margin | bf16 | bf16 margin |
|---|---|---|---|---|
| isolated page canaries (4 of them) | exact | 4096.0 | exact | 32768.0 |
| rung 1 — tiles=1, tail=64 | exact | 4096.0 | exact | 32768.0 |
| rung 2 — tiles=2, tail=15 | inexact | 570.3 | inexact | 2073.9 |
| rung 3 — tiles=5, tail=1 | inexact | 79.7 | inexact | 4207.6 |
| rung 4 — tiles=3 (ring depth) | inexact | 1365.3 | inexact | 10922.7 |
| rung 5 — tiles=6 (ring wrap) | inexact | 1365.3 | inexact | 10922.7 |

Worst fp16 margin 79.7 fp32 ULP (rung 3, 2061/257); worst bf16 margin **2073.9** fp32 ULP (rung 2,
975/79). Every BF16 margin is **wider** than the FP16 margin of the same rung — a coarser grid puts
its midpoints further apart, which is the right way round — and the FP16 leg of the same comparison
already passes at 0 ULP on CAModel. BF16 is the easier of the two here, so a failure on the BF16
leg would be a real defect and not a quantisation artefact.

The device's accumulation error is nowhere near any of these margins: in the exact fixture every key
nibble is 0, so every score is equal and every softmax weight is exactly 1, the numerator is a small
exact integer in fp32, and the only inexact steps are one reciprocal and one multiply — a handful of
fp32 ULPs.

### 6.2 The raw query is BF16-exact, with zero margin

The fixture's pre-rotated query row is the constant vector `(coefficient / 256) · 1` with
`coefficient ∈ {1,2,3}`, and `Pi` of a constant vector is `±coefficient · k / 2048` for integer
`k = (H s)_i / 2` — small integers over a power of two. Over the fixture's own sign vector the
widest significand that actually occurs is:

| GQA coefficient | query value | significand bits used |
|---|---|---|
| 1 | 0.00390625 | 5 |
| 2 | 0.0078125 | 5 |
| 3 | 0.01171875 | **7** |

fp16 stores 10 (margin 3); bf16 stores 7 (**margin 0**). So the fixture's raw query is exactly
representable in BF16 and needed no change — but with nothing to spare. 7 is the **measured** width,
not a bound on it: a different sign function, a different coefficient set or a non-constant query
could push it to 9 and break BF16 while leaving FP16 intact. That is why
`tests/cpu/test_fused_decode_opt_fixture.cpp` **asserts** the full round trip — widen the BF16 raw
query, rotate it, land on the pre-rotated query bitwise, over all 16 rows — and prints the measured
width, rather than leaving it as a comment.

The gate logits are quarter-steps in [-2, 2], at most 3 significand bits, exact in both formats;
`gate_widened(i) == gate_value(i)` is asserted rather than assumed.

### 6.3 The two BF16 rounders agree

`reference_bfloat16` (splits the fp32 word, decides round-half-to-even explicitly, reassembles) and
`ascend_micro::sim::float_to_bfloat16_rne` (adds a bias and shifts) are different code. They are
pinned against each other over all 1793 grid points of [1,15] **and over all 1792 halfway points
between them** — where a wrong half-to-even rule shows and a right one does not — plus the
carry-into-exponent case and both tie parities.

---

## 7. Gates

### Front-end, strict (`-Wall -Werror`, both core archs, `-fsyntax-only`)

| translation unit | `dav-c310-cube` | `dav-c310-vec` |
|---|---|---|
| `tests/sim/kernel/fused_decode_opt_kernel.cpp` (simulator leg; includes the production source) | ok | ok |
| `op_kernel/turboquant_fused_decode.cpp` (what `turboquant_fused_decode_opt_npu` builds) | ok | ok |

Both instances are instantiated in each, so the BF16 template body — including the basis service's
BF16 Mmad path — is front-end checked on both archs. Run via
`bash scripts/check_fused_decode_device_syntax.sh [source]`.

### Tier-1 host gates

| gate | verdict |
|---|---|
| `tests/cpu/test_fused_decode_opt_contract.cpp` | **PASS** — UB = 142304 + 256 bytes, unchanged |
| `tests/cpu/test_fused_decode_opt_fixture.cpp` | **PASS** — including the whole BF16 gold-model block of §5/§6 |
| `tests/cpu/test_fused_decode_opt_model.py` | **PASS** |
| `tests/cpu/test_fused_decode_opt_trace.py` | **PASS** |
| `tests/sim/kernel/test_three_zone_schedule.py` | **PASS** |
| `vllm_integration_layer/tests/host/test_integration_host_gate.cpp` | **PASS** — 211 checks, no device |
| `scripts/check_fused_decode_bf16_margins.py` | **PASS** |

`test_fused_decode_opt_fixture.cpp`'s `require` now reports the failing line and expression; it
previously reported only that something failed, which cost a bisect per failure. Its first run after
the element-type change caught a genuine bug in a new assertion of mine — `a.scales == b.scales`
compared `std::vector<float>` containing the `0x7FC0BEEF` NaN flood, so `==` was false between two
byte-identical copies. The comparison is now bitwise (`same_bits`), which is also the stronger claim
the assertion wanted.

### Compile-only (no NPU on this host)

| target | verdict |
|---|---|
| `tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp` + `turboquant_fused_decode_opt_rig.hpp` | compiles `-Wall -Wextra -Werror` |

### CAModel canary ladder

**PASS, both element types.** `bash scripts/run_fused_decode_canary_ladder.sh`, exit 0, 23 cases,
23 trace audits, no `[error]`, no `[aicore error]`, no timeout, no mismatch anywhere in the run.
Evidence root `/tmp/fused-decode-bf16-ladder` (ext4; a case directory per rung holding
`command.txt`, `run.log` and `audit.log`).

| case | FP16 | BF16 |
|---|---|---|
| `isolated-first` — logical page 0, non-identity table | PASS | PASS |
| `isolated-middle` — logical page K/2 = 4 | PASS | PASS |
| `isolated-last` — logical page K-1 = 7 | PASS | PASS |
| `isolated-last-tile` — final 64-row tile of the final page | PASS | PASS |
| `rung1-single-tile` — tiles 1, tail 64 | PASS | PASS |
| `rung2-ragged-tail` — tiles 2, tail 15 | PASS | PASS |
| `rung3-single-row-tail` — tiles 5, tail 1 | PASS | PASS |
| `rung4-ring-depth` — tiles 3 = kSlots | PASS | PASS |
| `rung5-ring-wrap` — tiles 6 = 2·kSlots | PASS | PASS |
| `splitk-workers` — tiles 6, 4 real Split-K workers incl. an empty one | PASS | PASS |
| `unmapped-degrades` — tiles 12, planted interior hole | PASS | — (§7 note) |
| `empty-worker` | PASS | — |
| `repeat-tasks` — tiles 4, 17 heads, 2 launches | PASS | — |

Every case: active rows compared at **0 ULP of the output format against the stored 16-bit word at
its original position**, every GM payload flooded with `0x7FC0BEEF`, both exterior guard bands
asserted intact, every input byte snapshotted before upload and compared **bitwise** after
synchronisation, inactive output slots and (at S=1) the whole workspace still carrying the flood
word. 23/23 reported `inputs=unchanged` and 23/23 `guards=0x7FC0BEEF-intact`. The per-case trace
audit reported `status: PASS` with `hardevents: capacity-one, balanced dispatch/completion, empty
drain` in all 23, with real Vector work on every one (4 `veccore` routes on the degenerate cases,
42 on the Split-K rungs).

Three cases ran at FP16 only, deliberately rather than for time: `unmapped-degrades`,
`empty-worker` and `repeat-tasks` assert properties of the tile cursor's density guard, the
empty-worker bypass and slot reuse across launches, all of which are compiled from descriptor words
and ring indices with no `scalar_t` in them. What *is* element-type specific — every GM write of a
result — is covered at both types by the five rungs and the four canaries, and the Split-K
reducer's own output cast and burst by `splitk-workers` at both types. The script's header records
this choice where the cases are.

**The BF16 leg is not a vacuous pass.** The exact rungs oracle every channel to an integer
`2n+1`, `n ∈ 0..7`, and the two formats encode those completely differently — 3.0 is `0x4200` in
fp16 and `0x4040` in bf16, 15.0 is `0x4B80` against `0x4170`, and no value shares a word. Both legs
compared at 0 ULP against **their own** format's reference at the stored word position, so had the
dispatch ignored `element_type` and run the FP16 kernel for the BF16 leg, every channel of every
BF16 case would have failed rather than passed. Each case's `command.txt` was checked to carry the
`--dtype` its directory name claims.

No long-context case was run; see the pending list below.

### Pending or unavailable

* **Every hardware gate.** There is no 950PR on this host (`npu-smi` and `/dev/davinci*` absent), so
  `test_npu_fused_decode_opt`, the benchmark and the msprof runs are **pending**, for BF16 and for
  FP16 alike. The hardware validation ladder now runs its whole rung table at **both** element types
  and compares in each leg's own format, so it is ready to execute when silicon is available; it has
  been compiled, never run.
* **`turboquant_fused_decode_opt_npu` as a linked `ascendc_library`.** Its device translation unit is
  front-end checked at both archs with the real strict flag set, and its host branch is compiled by
  the simulator leg's `ascendc_library` host leg (the probe TU includes the production source). A
  full `ascendc_library` pass over the physical leg itself was **not** run: it adds nothing the two
  checks above do not already cover for a change that introduces no new source file, and it is tens
  of minutes per leg.
* **The raw-query entry on CAModel at either element type.** The ladder drives the pre-rotated entry;
  the raw-query entry's numerical evidence is the fixture's host-side involution gate (§6.2) and the
  hardware rig, which is pending. This is unchanged from the FP16 baseline — the previous audit
  records the same gap — and is not something the BF16 instance made worse.
* **`FusedOptRig::kUlpBudgetRestaged` at BF16.** Output stages 1 and 2 put a 256-term fp32 rotation
  (and at 2 a division) between the oracle and the output. The FP16 budget is 8 ULP and was always a
  first-run placeholder; the BF16 budget is derived from it as **1** ULP, because one bf16 ULP spans
  eight fp16 ULPs and the bound is on an absolute fp32 residual — carrying 8 across would have
  widened the BF16 gate eightfold for no stated reason. Both numbers remain unmeasured, on hardware
  or on CAModel, and the rig prints the worst ULP it actually saw in the format it saw it in.
* **`unmapped-degrades`, `empty-worker` and `repeat-tasks` at BF16.** Not run, and not pending
  either: each asserts a property of the tile cursor, the cross-core flag protocol or slot reuse
  across launches, none of which contains `scalar_t`. Running them a second time would re-execute
  the same instructions. Recorded here so the asymmetry in §7's ladder table is a stated decision
  rather than an omission.
* **No long-context case was run**, deliberately. §4's inductive argument is the depth/wrap pair plus
  the boundaries; a 16k-token context on a cycle-accurate model is hours with nothing in it the wrap
  rung does not already exercise.

---

## 8. Files touched

Device:

* `op_kernel/turboquant_fused_decode.cpp` — BF16 instantiations; both `_impl`s dispatch on `type`;
  the convenience ABI carries `element_type`.
* `op_kernel/cube/turboquant_query_basis.h` — the Cube chunk operand follows `scalar_t`; H16
  converted on ingest where it is not fp16; `kOperandC0`.
* `op_kernel/vector/turboquant_vector_service.h` — `sizeof(scalar_t) == 2` pins; `WriteHeads`
  documents the operator's single output rounding.
* `op_kernel/cube/turboquant_gemm_reduce_service.h` — the same pins on both reducers.
* `include/turboquant_fused_decode_opt.hpp` (+ the repo-root `include/` copy) — `element_type`,
  `ascend_type_of`, `element_bytes`; the UB figures' element-independence written down.

Host and tests:

* `include/sim_micro_testbed.hpp` (both copies) — `bfloat16_to_float`.
* `tests/common/fused_decode_opt_fixture.hpp` (both copies) — `Element`, `reference_bfloat16`,
  `reference_bits`, `widen_bits`, `exact_input_bits`, `gate_widened`; the fixture carries its
  element type into every 16-bit encoding and into `Launch`.
* `tests/npu/turboquant_fused_decode_opt_rig.hpp` (both copies) — compares in the fixture's format;
  the derived BF16 restaged budget.
* `tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp` — the rung table runs at both
  element types.
* `tests/sim/kernel/fused_decode_opt_kernel.cpp` — the probes are one macro over `scalar_t`; the
  simulator launcher dispatches and refuses an unknown element type.
* `tests/sim/kernel/test_camodel_fused_decode_opt.cpp` — `--dtype fp16|bf16`; `OutputFormat`;
  comparison at the stored word.
* `tests/cpu/test_fused_decode_opt_fixture.cpp` — the BF16 gold-model gate; `require` reports its
  line.
* `scripts/run_fused_decode_canary_ladder.sh` — both legs; `FUSED_DECODE_OPT_DTYPES`.
* `scripts/check_fused_decode_bf16_margins.py` — new; the derivation behind §6.1 and §6.2.

`git status` lists exactly the files above and nothing else. The untracked
`reports/reduction_probe_*` and `reports/single_tile_msprof_*` evidence directories live in the
main checkout, not in this worktree, so nothing in this change could reach them; no unrelated
working-tree state was modified here either.

### The two hand-synced copies

`include/sim_micro_testbed.hpp`, `tests/common/fused_decode_opt_fixture.hpp` and
`tests/npu/turboquant_fused_decode_opt_rig.hpp` each exist twice — once under `tests/`/`include/` and
once under `vllm_integration_layer/` — and are kept in step by hand. All three pairs are
byte-identical after this change, asserted by the edit scripts rather than eyeballed.
`include/turboquant_fused_decode_opt.hpp` is the exception: its two copies differ, and only, in
three relative include paths (the root copy reaches the `fused_decode/` snapshot, the staged copy its
own `op_kernel/common/`). That difference predates this change and was preserved; the root copy was
re-derived from the staged one and the remaining diff checked to be exactly those three lines.
