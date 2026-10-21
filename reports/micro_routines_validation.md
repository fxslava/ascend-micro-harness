# Ascend 950PR micro-routine validation

## Documentation checked before implementation

All paths below are under `C:\PipeSim\docs\ascendc_api\SIMD-API`.

- `general_description_and_constraints.md`: UB/L1 addresses require 32-byte
  alignment, L0A/L0B require 512-byte alignment, and L0C requires 64-byte
  alignment. The ISASI `Mmad` page strengthens L0C to 1024 bytes.
- `basic_api/memory_vector_compute/SIMD_compute/high_dimension_slicing.md`:
  one repeat spans eight 32-byte data blocks, repeat count is at most 255,
  and repeat strides are measured in data blocks.
- `basic_api/memory_vector_compute/type_conversion/Cast.md`,
  `basic_arithmetic/Mul.md`, `basic_arithmetic/Muls.md`,
  `logical_compute/And.md`, `scatter_gather/Gather.md`, and
  `data_structures/LocalTensor/ReinterpretCast.md`: verified 950PR nibble
  conversion and scale path. `int4b_t` is signed; masking its cast `int16_t`
  representation with 15 recovers uniform unsigned nibble values.
  The probe defines packed order as low nibble first; native packing order
  still needs a CAModel/device check. No `ArithProgression` entry was present
  in the verified `basic_api` tree; the dequant path does not depend on it.
- `basic_api/cube_compute_TensorAPI/cube_compute_load/Copy_L1ToL0A.md`,
  `Copy_L1ToL0B.md`, `mmad_compute/Mmad.md`, and
  `cube_compute_store/Copy_L0CToUB.md`: verified L1→L0, Mmad, and Fixpipe
  signatures. `unitFlag=3` pairs one Mmad with one Fixpipe dump.
- `basic_api/memory_vector_compute/reduction_compute/ReduceMax.md`,
  `ReduceSum.md`, `ReduceRepeat.md`, and `basic_arithmetic/Exp.md`: verified
  the 16-element online state update and channel reductions. The requested
  `WholeReduceSum` memory-vector page is absent; `ReduceSum` and
  `ReduceRepeat<SUM>` are documented for 950PR. `Exp` permits up to 1 ulp
  error, so bit-identical native output versus `std::exp` is not guaranteed.
- `basic_api/memory_vector_compute/basic_arithmetic/Div.md`,
  `type_conversion/Cast.md`, and
  `data_move/DataCopy_GMAndUB_continuous.md`: verified precise FP32 divide,
  FP16/BF16 round-to-even cast, and the 32-byte MTE3 length requirement.

## Host probe results

Compiled with WSL Ubuntu 22.04 `g++ -std=c++17 -O2 -Wall -Wextra -Werror`.
Each probe compares its host mock against a scalar golden oracle and checks
all 64-byte canary bands. Timings include staging, oracle, and checks, so they
are not NPU kernel latency.

| Probe | Cases | Last host time |
| --- | --- | ---: |
| dequant4 | 4 rows × 2 types × 128 elements | 12.174 µs |
| cube matmul | logits/context at 16×16 and 32×16 | 269.371 µs |
| online softmax | 4 streaming tiles × 16 channels | 19.299 µs |
| normalize/store | 12 FP16/BF16 cases | 21.193 µs |

The two Cube phases require a caller-owned completion boundary after
L1→L0 loading. Normalize and MTE3 store likewise require a caller-owned
V→MTE3 completion boundary. No explicit synchronization primitive or scalar
elementwise compute appears in the routine headers. Callers must provide
32-byte bank skew between contiguous UB slots; the isolated mock buffers do
not validate a production UB allocation layout.

## CAModel (Tier 2) results

All four micro-routines now execute on the `Ascend950PR_9599` CAModel through
the ACL runner (`bash scripts/run_modulo3_camodel.sh micro <name>`), with
CANN 9.2.0-beta.2, `dav-3510`, `CMAKE_BUILD_TYPE=Release`. Every run exits 0
with zero `invalid ldst addr` / `unrecognize ldst addr` / `su_ccu_mpu_err`
lines and all GM guard bands intact.

| Probe | Result | Scalar addr errors | Dispatch span | Complete span | Model ticks |
| --- | --- | ---: | --- | --- | ---: |
| dequant4 | 64/64 exact, 128/128 guard bytes | 0 | 714..2513 (1799 cyc, 190 instr) | 718..2533 (1815 cyc, 187 instr) | 2839 |
| cube matmul | 16×16 256/256 and 32×16 512/512 exact, 384/384 guard bytes | 0 | 660..5173 (4513 cyc, 463 instr) | 664..5193 (4529 cyc, 477 instr) | 5507 |
| normalize/store | FP16 and BF16 16/16 bit-exact, 384/384 guard bytes | 0 | 705..6645 (5940 cyc, 844 instr) | 709..6665 (5956 cyc, 802 instr) | 6971 |
| online softmax | 4 tiles, max abs error 2.38418579e-07, max 8.000 ULP (`acc`), 512/512 guard bytes | 0 | 713..7671 (6958 cyc, 1882 instr) | 717..7691 (6974 cyc, 1765 instr) | 8081 |

Cycle spans come from `scripts/analyze_micro_trace.py` over the per-core
`instr_log` / `instr_popped_log` dumps. These are simulator figures for a
reduced inductive grid, not silicon latency or steady-state throughput.

### Online softmax: faults fixed this cycle

The pre-fix baseline emitted 16 CAModel errors per launch
(`ldst_addr: 64`, `accessUb:0 accessDdr:1`) while still printing a numeric
PASS. The simulator error lines, the decoded `core0.veccore0` instruction
window and the producing library code are preserved in
[`online_softmax_scalar_fault_trace.txt`](online_softmax_scalar_fault_trace.txt).
Root cause and both corrections to the original diagnosis are recorded in
[`../MICROROUTINE_DESIGN_GUIDELINES.md`](../MICROROUTINE_DESIGN_GUIDELINES.md)
§0. In summary:

1. `AscendC::ReduceSum` on FP32/FP16 ends with a library-internal `V_S`
   drain plus `accValFloat = dstLocal[0]`, allocating the event from
   `GetTPipePtr()`. These kernels place their own `LocalTensor`s and have no
   `TPipe`, so the event-bitmap accesses dereference null at GM `0x40`.
   Replaced with `ReduceRepeat<ReduceType::SUM>`, which has no accumulator
   side channel and never leaves the vector pipe. 16 errors → 0.
2. `Duplicate(tensor, tensor)` was **not** the scalar trip — it lowers to a
   pure-vector `Reg::LoadAlign` + `Reg::Duplicate`. It was replaced by `Brcb`
   in this routine anyway, because its unmasked 256-byte source read is a
   224-byte over-read of a 32-byte state block, and the replicated-state
   layout makes `Brcb` exact.
3. Removing `ReduceSum` removed an accidental per-tile vector-pipe drain,
   which exposed a **latent V→MTE2 WAR hazard**: single-buffered `scores` /
   `v_by_channel` were being refilled by MTE2 while the vector pipe was still
   reading them, so every tile computed on a later tile's scores. Fixed with
   a primed, drained `V_MTE2` back edge. `m` had masked this because every
   tile in the fixture shares the same maximum.

Numeric accuracy is unchanged from the pre-fix baseline (2.38418579e-07 max
absolute error in both). The `<= 1 ULP` acceptance criterion is not attainable
for `acc`: `Exp` is documented to permit 1 ULP per call, `acc` is a cancelling
signed sum (10.2× cancellation at column 0), and the device reduces with a
hardware tree against a sequential host oracle. See guidelines §5.

## Native build status

CANN 9.2 BiSheng compiles and links all four CAModel kernel libraries plus
the ACL runners. No 950PR silicon execution was performed; CAModel is a
cycle-accurate simulator, so it does not validate silicon timing, power, or
multi-core scale-out.
