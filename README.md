# Ascend 950PR 16×16 Cube Hadamard probe

Micro-routine development rules — the zero-scalar-compute invariant, UB
alignment and bank invariants (AKA3003), RAW/WAR hazard management inside the
AIV, and the two-tier verification methodology — are in
[`MICROROUTINE_DESIGN_GUIDELINES.md`](MICROROUTINE_DESIGN_GUIDELINES.md).
Read it before adding a routine.

The three-stage modulo synchronization probe is documented in
[`reports/modulo3_deadlock_probe.md`](reports/modulo3_deadlock_probe.md). Run it
under WSL/CAModel with `bash scripts/run_modulo3_camodel.sh`.

The micro-routine CAModel probes run with
`bash scripts/run_modulo3_camodel.sh micro <dequant4|cube_matmul|normalize_store|online_softmax>`.
Latest results are in
[`reports/micro_routines_validation.md`](reports/micro_routines_validation.md).

`bash scripts/run_modulo3_camodel.sh micro fused_attention_3tiles` runs the
production fused operator in
[`kernels/ops/turboquant/turboquant_paged_attention_opt.cpp`](kernels/ops/turboquant/turboquant_paged_attention_opt.cpp):
five leaves in one `MIX_AIC_1_2` three-stage modulo pipeline over three key
pages, driving one Cube core and **both** Vector cores, with mode-4
double-handshake flags and zero Global Memory workspace — the dequantised key
page goes straight from Vector UB into Cube L1 over the dav-3510 MTE3 channel
(`routines/routine_cv_transfer_ub_to_l1.hpp`). See
[`reports/fused_attention_integration.md`](reports/fused_attention_integration.md).

That report also retracts an earlier claim, repeated in §6.1 of the
guidelines, that no UB → L1 transfer exists on this part.

`routines/routine_cube_hadamard16.hpp` dispatches one FP16×FP16→FP32
`Mmad` followed by one L0C→GM `Fixpipe` copy. The caller precomputes the
canonical Sylvester matrix as four Kronecker `H2` factors and stages the
resulting 16×16 FP16 tile in L0B/ZN. One dense Cube instruction computes
`X × H16`; there are no scalar elementwise operations in the AIC leaf.

## Native caller contract

- Product/API: Ascend 950PR, arch35, CANN 9.2 Tensor API.
- L0A: 16×16 FP16 NZ, 512-byte aligned; L0B: 16×16 FP16 ZN, 512-byte
  aligned; L0C: 16×16 FP32 NZ, 64-byte aligned.
- Output: 16×16 ND in GM, 32-byte aligned. Select `Output=float`, `half`, or
  `bfloat16_t` with `OutflowMode::Fp32`, `Fp16`, or `Bf16` respectively.
- The caller completes GM→L1 conversion and L1→L0 loads before invoking the
  leaf. It also waits for GM output visibility after the leaf returns. These
  lifecycle barriers belong outside the leaf.
- The recommended L1 layout has X at byte 0 and H at byte 544: 512 bytes
  for X plus a 32-byte bank skew pad. Do not alias the operands or output.
- The single `Mmad` and `Fixpipe` use matching `unitFlag=3` to enforce the
  L0C producer/consumer dependency without explicit synchronization calls.

The matrix is mathematically factorized when prepared; the runtime performs
one dense 16×16 Cube multiply. Four separate Cube butterfly stages would
need intermediate casts and inter-stage dependencies, so they are not the
operation modeled by this leaf.

## Host probe

```sh
mkdir -p bin
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/test_hadamard16_probe.cpp -o bin/test_hadamard16_probe
bin/test_hadamard16_probe
```

The C++17 testbed has guarded aligned mock GM/L1/L0/UB buffers and models the
same one-Mmad/one-Fixpipe dispatch. The CPU scalar oracle independently
computes `X × H16` and checks all 256 results bit for bit in FP32, FP16 and
BF16 for five input patterns. Input values are dyadic FP16 numbers so their
signed 16-term FP32 sums are exact and insensitive to reduction ordering.
FP16 and BF16 cast checks use round-to-nearest-even.

The host timing includes staging, golden reference, execution and checks; it
is not AIC latency. The structural cycle model is 2 GM→L1 transfers,
2 L1→L0 transfers, 1 `Mmad` (4096 multiply-accumulates, 8192 FLOPs), and 1
Fixpipe drain. With stage rates in bytes/cycle for each path and FLOPs/cycle
for Cube, a serial single-tile model is
`1024/B_GM_L1 + 1024/B_L1_L0 + 8192/P_Cube + O/B_Fixpipe` cycles.
For a steady stream of independently buffered tiles, the throughput interval
is bounded below by the maximum of those four terms. `O=1024` bytes for FP32
and `O=512` bytes for FP16/BF16. Startup, drain, bank effects and caller
synchronization add to the serial model. A numeric
hardware cycle estimate requires 950PR issue/bandwidth data or a device
trace. The native branch also requires a CANN 9.2 BiSheng toolchain and
950PR hardware for compilation and execution validation.
