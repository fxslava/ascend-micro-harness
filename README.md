# Ascend 950PR 16×16 Cube Hadamard probe

## Test domains and NPU TurboQuant harness

`tests/cpu/` contains host-only oracle probes; `tests/camodel/` contains CANN
simulator runners; `tests/npu/validation/` and `tests/npu/bench/` contain physical
NPU tests. One CANN 9.2 build includes both CAModel and NPU targets for
Ascend950PR_9599, using BiSheng through `ascendc_library`.

The BF16 comparison kernel is vendored under
[`kernels/ops/turboquant/baseline/`](kernels/ops/turboquant/baseline/); building
does not require a separate `vllm-ascend` checkout.

From WSL with the CANN environment loaded, a default build compiles every
CAModel and NPU target together:

```sh
cmake -S . -B /tmp/ascend-micro-all -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ascend-micro-all -j4
```

Executables and their kernel shared libraries are placed together in
`/tmp/ascend-micro-all/lib`.

From PowerShell, compile both NPU targets:

```powershell
& .\scripts\build_turboquant_npu.ps1
```

The script creates a fresh WSL ext4 build directory and prints its path.
Compilation does not require an attached NPU. On a machine with a 950PR and
working AscendCL device access, run the printed validation binary first, then
the benchmark binary:

```powershell
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/test_turboquant_a8_validation
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/test_turboquant_bf16_validation
wsl -- /tmp/ascend-turboquant-npu-XXXXXX/lib/bench_turboquant_a8_vs_bf16
```

A8 validation checks one 16-token strip and 512/1024-token contexts. Each strip
uses one physical AI Core; all packed key and value nibbles and FP32 scale bits
are compared with an independent integer Hadamard oracle.

BF16 validation shares that fixture, oracle and per-page verifier — only the
input encoding differs, and with integer activations the two instances must
produce byte-identical pages — and drives its context as several launches:

```powershell
wsl -- .../test_turboquant_bf16_validation --strips 33 --window 32
wsl -- .../test_turboquant_bf16_validation --tokens 1024
```

`--strips` and `--tokens` are repeatable and build a custom ladder; with
neither, the ladder is 1, 32, 33 and 64 strips. `--window W` caps the strips per
launch at W (default 32), because the kernel allows one strip per grid block per
launch and a grid wider than the AI core count does not dispatch. The rungs are
chosen for what they can fail on: 33 is one strip in the **second** wave, a
ragged tail launch straight after a full one, where a core that has already used
its mode-4 flag ids must re-arm them; 64 is two full waves, which needs a
complete reuse cycle instead. A token leaked by the first launch shows up at 33
and at no smaller rung.

The benchmark measures three kernels — A8 Cube, BF16 Cube and the vendored BF16
Vector baseline — with 50 warmups and 500 device-event samples each, in both
warm and 128 MiB SDMA cache-pressure modes. The dummy copy does not guarantee
full L2 invalidation. Only the two BF16 figures share an input precision and a
quantizer, so only that pair isolates Cube against Vector rotation; either
against A8 also changes the input precision and is a wall-time comparison for
the same 1024-token geometry and nothing narrower.

These three binaries need an attached 950PR. On a host without one they abort
during static initialization of the NPU-mode runtime, before `main`, so they
cannot even print their usage — the CAModel probes above are the verification
gate here.

Micro-routine development rules — the zero-scalar-compute invariant, UB
alignment and bank invariants (AKA3003), RAW/WAR hazard management inside the
AIV, and the two-tier verification methodology — are in
[`MICROROUTINE_DESIGN_GUIDELINES.md`](MICROROUTINE_DESIGN_GUIDELINES.md).
Read it before adding a routine.

[`SKELETON_KERNEL_WORKFLOW.md`](SKELETON_KERNEL_WORKFLOW.md) is the procedure
that enforces those rules: where a new kernel goes, how to wire its build and
launch ABI, and the five verification stages with the gate each one must pass
— compile-time invariants, Tier-1 host model, CAModel at the ring depth,
CAModel past the ring depth for steady state, and the trace audit. Each stage
catches a class the previous one is blind to, which is why none is optional.

The three-stage modulo synchronization probe is documented in
[`reports/modulo3_deadlock_probe.md`](reports/modulo3_deadlock_probe.md). Run it
under WSL/CAModel with `bash scripts/run_modulo3_camodel.sh`.

The micro-routine CAModel probes run with
`bash scripts/run_modulo3_camodel.sh micro <dequant4|cube_matmul|normalize_store|online_softmax>`.
Latest results are in
[`reports/micro_routines_validation.md`](reports/micro_routines_validation.md).

The production fused operator in
[`kernels/ops/turboquant/turboquant_paged_attention_opt.cpp`](kernels/ops/turboquant/turboquant_paged_attention_opt.cpp)
puts five leaves in one `MIX_AIC_1_2` three-stage modulo pipeline over a ring
of key pages, driving one Cube core and **both** Vector cores, with mode-4
double-handshake flags and zero Global Memory workspace — the dequantised key
page goes straight from Vector UB into Cube L1 over the dav-3510 MTE3 channel
(`routines/routine_cv_transfer_ub_to_l1.hpp`). Its page count is a runtime
option of one probe over one kernel source:

```sh
bash scripts/run_modulo3_camodel.sh micro fused_attention 8   # steady state
bash scripts/run_modulo3_camodel.sh micro fused_attention 3   # depth boundary
```

Three pages never wrap the depth-3 ring, so that run measures a pipeline fill
and its buffer-reuse back edges are only primed and drained. Eight pages wrap
it twice, which contends every back edge and gives a per-page steady-state
interval — and which is how AIV1's missing packed-page WAR edge was found. See
[`reports/fused_attention_integration.md`](reports/fused_attention_integration.md).

That report also retracts an earlier claim, repeated in §6.1 of the
guidelines, that no UB → L1 transfer exists on this part.

The prefill counterpart in
[`kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp`](kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp)
is a **four-stage** `MIX_AIC_1_2` ring over the eight chunks of one 16-token ×
1024-channel strip: MTE2 stages each raw INT8 page from Global Memory into the
L1 ring with no Unified Buffer hop, the Cube applies the **whole** randomised
Hadamard rotation as 64 INT8×INT8→INT32 Mmads per chunk and hands the finished
panel over in one Fixpipe, both Vector cores only reduce, scale and pack to
INT4, and MTE3 writes a 2 KiB packed page plus a 64-byte scale slice per chunk.
Eight chunks over a depth-4 L1 ring and depth-2 per-core UB rings wrap every
ring twice, so one run contends every buffer-reuse back edge.

The rotation sits entirely on the Cube because the first revision split it —
Cube inner factor, Vector sign diagonal and butterfly cascade — and measured
the result: the Cube waited on a Vector core for 33.6% of its attributed cycles
against 2.8% spent multiplying, while the Vector cost was register traffic
rather than arithmetic. Folding the outer factor's signs into 16 resident
32×32 INT8 blocks costs eight times the multiply-adds and buys back four
full-panel Vector passes and 65 KiB of Unified Buffer; the trade and what it
measured are in the report.

```sh
bash scripts/run_modulo3_camodel.sh micro reshape_and_cache_a8
```

Its quantiser uses a power-of-two scale extracted by integer ops on the FP32
bit pattern, so the whole path from INT8 input to packed nibble carries exactly
one rounding step and the KV cache is checked **bit-exactly against an integer
oracle for an arbitrary input**, not against a tolerance or a crafted fixture.
Its Tier-1 probe checks what that comparison cannot — that the dense
256×256 rotation satisfies `M Mᵀ = 256 I`, that the oracle's factored form
reproduces it exactly, and that the 16 resident blocks the Cube multiplies
compose back to all 65536 entries of it:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    tests/cpu/test_reshape_and_cache_a8_probe.cpp \
    -o bin/test_reshape_and_cache_a8_probe
bin/test_reshape_and_cache_a8_probe
```

Results, including what the trace does and does not establish, are in
[`reports/reshape_and_cache_a8.md`](reports/reshape_and_cache_a8.md).

The BF16 instance of that operator raised a data-movement question the INT8
one could not: the arch-3510 fractal is 16 × (32B/sizeof(T)), so C0 drops from
32 to 16 and a 32-wide axis stops being one fractal. The single-tile probe in
[`tests/camodel/test_micro_layout_bf16.cpp`](tests/camodel/test_micro_layout_bf16.cpp)
answers it on one AIC, sixteen Mmads and no synchronization at all:

```sh
bash scripts/run_micro_layout_bf16_camodel.sh
```

The 16 × 256 activation page's Nz/Zn duality **survives** the widening, so the
Value chunks need no new load path. The 32 × 32 resident rotation block's does
**not**: a row-major blob stops being its own Nz image, and the probe
reproduces the resulting misread bit-exactly from the index map before showing
that an ND→Nz ingress removes it. MTE1 cannot repair this — its copy unit is
the 512-byte fractal and its only rearrangement is an intra-fractal transpose —
so the fix is at ingress: `stage_blocks_nz` in
[`routines/routine_stage_page_zn_bf16.hpp`](routines/routine_stage_page_zn_bf16.hpp),
with the four byte-identity L1→L0 loads in
[`routines/routine_fractal_load_zn_bf16.hpp`](routines/routine_fractal_load_zn_bf16.hpp).

That ingress is now the operator's prologue, and the operator is bit-exact on
CAModel at 1, 2 and 3 strips:

```sh
bash scripts/run_reshape_bf16_camodel.sh
```

With the flat copy put back it fails on the same runner, which is what makes
the fix measured rather than argued. The report also records why the failure
looked Value-specific — the runner caps its comparison at three chunks and the
value heads come first, so the equally broken key chunks were never printed.
Details and the gates in
[`reports/micro_layout_bf16.md`](reports/micro_layout_bf16.md).

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
    tests/cpu/test_hadamard16_probe.cpp -o bin/test_hadamard16_probe
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
