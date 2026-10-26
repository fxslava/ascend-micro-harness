# Cube WTA / Fixpipe probe — Ascend 950PR, Arch3510

The implementation computes the requested recurrent arithmetic entirely on one
AIC: `V <- ReLU(V W)`, W the lateral-inhibition matrix below, with FP16
feedback through L1 and a final scalar-quantized INT8 export. **Universal
one-hot argmax convergence in T=1..8 is not a valid property of this
recurrence.** At the specified α/β the probe measures an empirical convergence
rate of zero within 8 steps for every tested margin, and reports the rate at
which the recurrence does converge instead of altering the algorithm.

## Contract and assumptions

`routines/routine_cube_wta_fixpipe.hpp` exposes synchronization-free engine
leaves and a separate `cube_wta_run<Config>(engine, sync)` scheduling driver.
The native engine accepts caller-owned `LocalTensor` views. `WtaConfig<T,
ScaleBits, Output>` chooses the static iteration count, FP32 scale bit pattern,
and `int8_t`/`uint8_t`. C++17 does not support floating-point non-type template
parameters, hence the bit-pattern interface. Invalid nonpositive, subnormal,
infinite and NaN scales are rejected at compile time. The scale's low 13 bits
are cleared and bit 46 is set only for signed output; the offset is zero.

The inhibition matrix is fixed by the task contract,
`W = (1 + α) I − (β/16) 1 1ᵀ` with `α = 0.25`, `β = 0.4`. L0B stores W as
FP16, so the device multiplies by the FP16-rounded entries: diagonal
`1.224609375` (= 1254/1024) and off-diagonal `-0.024993896484375`
(= −819/2¹⁵). Both are exact dyadic rationals, which keeps every oracle in
this report exact; the real-valued 0.025 is never representable and never
compared against. The probe asserts the rounded entries at runtime.

Input is a direct random FP16 tile: sixteen independent rows of sixteen
standard-normal competitors (Box–Muller), the winner raised to second + Δ with
Δ swept over [0.01, 1.0], rounded once to FP16 on storage. **No host CDF
transform, no Lloyd-Max quantizer, and no corpus-calibrated scale table are
used.** An earlier iteration of this probe routed inputs through a normal-CDF
map with Lloyd-Max centroids and per-T saturation scales; that machinery was
removed when the direct-input contract was adopted, and the fixed probe scale
is 128 (`0x43000000`) for every T, sign and margin. The stored margin can
differ from the requested real margin by up to one FP16 ulp per side.

Each step computes `raw = V W` in FP32. Steps before the last apply Fixpipe
ReLU and FP16 round-to-nearest-even back into the C1 state in L1. The last
step applies ReLU and `QF322B8_PRE` directly to the FP32 result; it does
**not** round through FP16 first. No argmax indices or normalization are
produced.

```cpp
using Config = ascend_micro::WtaConfig<8, 0x43000000u, std::int8_t>;
ascend_micro::NativeCubeWtaEngine<Config> engine{
    a2, b2, co1, c1_state, c1_victory};
ascend_micro::NativeCubeWtaSync<0> sync;
ascend_micro::cube_wta_run<Config>(engine, sync);
```

The caller initializes static mode with `InitSocState`, launches AIC-only,
orders ingress before MTE1, and reserves event ID 0 on the driver's four
routes. Weights may be resident or loaded by earlier instructions in MTE1's
queue; the first `MTE1_M` covers those loads. Buffers must not overlap within
an address space:

| Buffer | Position | Capacity | Base alignment |
|---|---|---:|---:|
| activation operand | A2 | 512 B | 512 B (LoadData requirement) |
| resident W, Zn | B2 | 512 B | 512 B |
| accumulator | CO1 | 1024 B | at least 64 B |
| initial/feedback state, Nz | C1 | 512 B | at least 32 B |
| final B8 victory tile, Nz | C1 | 256 B | at least 32 B |

All Fixpipe moves use `CFG_NZ`, `FixpipeParamsArch3510<CO2Layout::NZ>`,
`mSize=nSize=srcStride=16`, **`dstStride=16 elements`**, and `reluEn=true`.
There is only one N fractal, so the inter-fractal destination stride is not
exercised as a multi-fractal layout in this specialization.

### N=16 B8 layout correction

The general NZ2NZ page describes 16x32 INT8 channel merging. The more specific
`Int8-Channel-Merge.md` states that the last group of 16 channels is **not
merged when N/16 is odd**. N=16 therefore produces a packed 16x16 byte tile,
not sixteen rows with a 32-byte pitch. CAModel readback confirms this. The
initial diagnostic observer incorrectly used a 32-byte pitch and failed at
row 1; correcting the observer to read the packed bytes resolved it.

The observer reserves 512 bytes at the C1 victory address because its test-only
LoadData reads a complete 512-byte fractal. That extra read pad is not part of
the routine's 256-byte output. No vector 256-byte read-quantum rule applies to
the routine, which never accesses UB.

## Event schedule and native observation

The serial recurrence is:

`FIX_MTE1 wait -> LoadData -> MTE1_M pair -> Mmad -> M_FIX pair -> Fixpipe -> FIX_MTE1 set`.

`FIX_MTE1` is primed once and drained once: T+1 sets and T+1 waits. The
forward routes each have T sets and T waits. This dependence chain also
protects L0A and L0C reuse: the preceding Fixpipe completes before the next
LoadData/Mmad, and the preceding Mmad completes before C1 is overwritten.
The driver ends with a `FIX_S` pair. There is no dynamic event allocation,
tensor scalar read, runtime recurrence loop, AIV operation, mode-4 event,
UB bounce or intermediate GM spill. The leaves contain no HardEvents; only
the driver and wrapper own them.

`kernels/camodel/wta_fixpipe.cpp` declares `KERNEL_TYPE_AIC_ONLY` and launches
one block. The wrapper stages inputs through MTE2, then invokes the routine.
Arch3510 does not support the continuous L1-to-GM copy API. After the routine
returns, an observation-only INT8 identity Mmad reads the actual 256 C1 bytes
as an 8x32 matrix. Its INT32 output is copied to GM through Fixpipe. Unsigned
bytes are recovered from the sign-extended INT32 readback. This extra Mmad,
its loads, and the GM Fixpipe are excluded from recurrent cycle measurements.

The trace auditor checks every emitted event route/ID, matched completion,
one-bit token ownership at retirement, no other active core, no cross-core
event, exact instruction counts, no UB path, quantization register bits,
and forward/back-edge retirement order. CANN emits terminal BAR/DSB
instructions and internal Fixpipe barriers; the source introduces no coarse
`PIPE_ALL` stage barrier.

## Tier 1 and convergence

The host test executes the same unrolled driver against three independently
scheduled engine queues, rejects overwrite/read hazards and double sets,
and checks all buffer canaries. The mock engine multiplies the actual FP16 W
entries and rounds the exact double product-sum once to FP32 — the value Mmad
leaves in L0C. A separate closed-form double oracle computes
`raw_j = d·x_j + off·(S − x_j)` per row; both evaluations are exact on FP16
inputs (every intermediate fits in double), so per-step FP16 feedback and
final B8 comparisons are **bit-exact** while the two arithmetic paths remain
independent. The device's own FP32 accumulation order is not assumed at this
tier.

400 randomized direct-input tiles cover 100 requested margins from 0.01 to
1.00 and four seeds each. Every tile runs T=1..8 in signed and unsigned modes:
6,400 executions, plus scale-truncation, saturation, tie, zero, negative and
known one-hot fixed-point cases (the identity tile stays one-hot at every T).
Event tokens balance and canaries remain intact; runtime ≈ 1 s.

Because random FP16 inputs are not dyadic-aligned, an alternate Mmad
accumulation order can legitimately move results in the last bits. The probe
measures this directly with a sequential-FP32-accumulation oracle: against
the exact oracle at T=8 it agrees on 102,398/102,400 bytes, maximum |B8
difference| 1, and **zero** rows change survivor support. That measurement,
not an assumption, sets the Tier-2 tolerance below.

At scale 128 the signed convergence census (6,400 rows) is:

| T | Rows | Unique survivor at input argmax | Multiple survivors | All-zero | Wrong single |
|---:|---:|---:|---:|---:|---:|
| 1..8 | 6400 each | 0 | 6400 | 0 | 0 |

**The empirical convergence rate within T ≤ 8 is zero at every tested
margin.** With β = 0.4 a competitor is only eliminated once the leader
exceeds it by a factor of d/c = 49, and near-parity ratios grow only ~4% per
step; margins of 1.0 on unit-scale competitors cannot cross that threshold in
8 steps. The recurrence does converge eventually, and the probe reports that
rate on the FP16 feedback chain (independent of quantization): of 6,400 rows,
the first unique survivor appears at step 10, the mode is step 31, the median
is step 38, 5,293 rows (82.7%) are unique by step 64, and 1,107 are not. The
split is a function of the margin (measured, 4 seeds per margin):
Δ=0.01 → 0/64 rows converge by 64; Δ=0.10 → 18/64; Δ=0.30 → 56/64;
Δ≥0.50 → 64/64. The explicit near-tie counterexample is seed 13, requested
margin 0.01: all sixteen rows still have multiple survivors at T=8, including
on CAModel.

`--require-one-hot` makes the host probe fail explicitly on this unsatisfied
specification (exit 1); the ordinary mode passes the implementation checks
and prints the convergence failure.

## CAModel measurements

Built with CANN 9.2.0-beta.2 for `SOC_VERSION=Ascend950PR_9599` in WSL
Ubuntu 22.04 on 2026-10-23. This installation resolves that simulator directory
to `x86_64-linux/simulator/dav_3510/lib`; its driver stub reports
`Ascend950PR_9589`. The report identifies both rather than claiming distinct
9599 silicon validation. These are CAModel ticks, not hardware measurements.

The full sweep is T=1..8 × signed/unsigned × Δ ∈ {0.01, 0.1, 0.5, 1.0},
seed 13: **64/64 launches pass** with zero `su_ccu_mpu_err`, zero
`invalid ldst addr` and zero `[error]`/`[fatal]` lines; all 64 trace audits
pass (balanced one-bit tokens, no cross-core events, no UB path, quantization
register written once with the expected bits). Every launch is **bit-exact
against the exact oracle: 256/256 bytes, maximum |difference| 0** — the ±1
tolerance admitted by the Tier-1 sensitivity measurement was not consumed,
and oracle and device survivor censuses agree (sixteen multi-survivor rows
in every run, consistent with the Tier-1 census at β = 0.4). Total model
ticks range 2,738 (T=1) to 3,623 (T=8) per launch.

The T=8 signed, seed-13, margin-0.01 trace measures:

| Step | MTE1 | Mmad | Fixpipe to C1 | Sum of engine latencies | Load to Fixpipe completion |
|---|---:|---:|---:|---:|---:|
| first, FP16 feedback | 32 | 28 | 44 | 104 | 108 |
| recurrent 2..7, FP16 feedback | 31 | 28 | 44 | 103 | 107 |
| final, INT8 | 31 | 28 | 43 | 102 | 111 |

T=1 signed measures 32/28/43 for its single B8 step. Latency is
`completion_tick - popped_tick`. Event and configuration gaps are excluded
from the engine sum and included in the last column. Successive LoadData
start intervals are 110 cycles initially, then 109 cycles. The quantization
move itself is one cycle shorter than the preceding FP16 move; the complete
final step is four cycles longer, with the scalar quantization register write
and associated dispatch gap included. These engine latencies are unchanged
from the earlier CDF/calibrated iteration — the revision altered host data
preparation and the scale constant only, not any recurrent instruction.

At T=8 the routine has 26 event pairs (9 feedback, 8 load-to-Cube, 8
Cube-to-Fixpipe, 1 final drain); the full wrapper has 30. T=1 has 9
full-wrapper pairs. All residual token counts are zero in every audited
trace. GM guards cover 512 bytes per launch and are intact; native L1/L0
guard integrity is not inferred from those GM guards.

## Reproduction

Run from the repository in WSL with the specified CANN installation:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/cpu/test_wta_convergence_probe.cpp \
    -o /tmp/test_wta_convergence_probe
/tmp/test_wta_convergence_probe
/tmp/test_wta_convergence_probe --require-one-hot  # expected exit 1
bash scripts/run_wta_camodel.sh
```

The script uses a fresh `/tmp` build because CANN preprocessing mutates its
intermediate object files. To rerun an existing build, set `WTA_BUILD_DIR`
and pass `run-only`. `WTA_STEPS`, `WTA_SIGNS`, and `WTA_MARGINS` select subsets;
the defaults are 1..8, both formats, and 0.01/0.1/0.5/1.0. Every launch has
a 60-second timeout and its own raw trace directory. `analyze_wta_trace.py`
is mandatory in this path, rather than relying on the general MIX analyzer.

## Local references

- `MICROROUTINE_DESIGN_GUIDELINES.md`, sections 1–3: scalar purity and event ownership.
- `C:/PipeSim/docs/ascendc_api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToL1.md`: Arch3510 fields, alignment, ReLU and quantization.
- `.../cube_store_key_features/on_the_fly_quantization.md`: scale mantissa truncation and sign bit 46.
- `.../cube_store_key_features/Int8-Channel-Merge.md`: unmerged final odd group of 16 channels.
- `.../cube_compute_load/Load2D.md`: 512-byte fractals and alignment.
- `.../mmad_compute/Mmad.md`: FP16/FP32 and INT8/INT32 combinations, layouts and unit flags.
- `.../cube_compute_store/DataCopy_L1ToGM_continuous.md`: unsupported on 950PR.
- Installed `asc/impl/basic_api/dav_3510/kernel_operator_fixpipe_impl.h`: actual scalar quantization dispatch and L0C-to-C1 intrinsic.
