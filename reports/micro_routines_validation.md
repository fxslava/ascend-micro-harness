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

## Native build status

CANN 9.2 BiSheng compiled and linked the temporary syntax-check library under
`build/native_check/out4` using `CMAKE_BUILD_TYPE=Release` and
`SOC_VERSION=Ascend950PR_9599`. No 950PR device/CAModel execution or native
bitwise comparison was completed. The host probes do not validate hardware
scheduling, native `Exp` rounding, or DMA visibility.
