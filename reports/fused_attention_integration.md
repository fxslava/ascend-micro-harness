# Fused TurboQuant paged attention — MIX_AIC_1_2, N = 3 pages

Kernel: [`kernels/ops/turboquant/turboquant_paged_attention_opt.cpp`](../kernels/ops/turboquant/turboquant_paged_attention_opt.cpp).
Runner: [`tests/camodel/test_camodel_fused_attention_3tiles.cpp`](../tests/camodel/test_camodel_fused_attention_3tiles.cpp).
CV leaf: [`routines/routine_cv_transfer_ub_to_l1.hpp`](../routines/routine_cv_transfer_ub_to_l1.hpp).
Trace audit: [`fused_attention_cv_event_summary.txt`](fused_attention_cv_event_summary.txt).

```sh
bash scripts/run_modulo3_camodel.sh micro fused_attention_3tiles
```

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599` CAModel,
`KERNEL_TYPE_MIX_AIC_1_2` (one Cube core and **both** Vector cores),
`CMAKE_BUILD_TYPE=Release`, `blockDim = 1`.

## What changed from the previous revision

| | previous | this revision |
| --- | --- | --- |
| Key page path | UB → GM workspace → L1(Zn) | UB → L1, on chip |
| GM workspace bytes | `ws_k`, 3 × 512 B | **0**, argument removed |
| Vector cores used | AIV0 only, AIV1 idle | AIV0 **and** AIV1 |
| Per-page GM stores | 1 score-panel dump per page | **0** |
| Cube path barriers | 2 × `PipeBarrier<PIPE_ALL>` per page | **0** |
| Steady-state control flow | runtime `for` with `% 2` ping-pong | 3 unrolled compile-time columns |
| Model ticks | 9203 | **6404** |
| Dispatch span | 8169 cycles | **5381 cycles** |

Numerics are unchanged: the same 768/768 bit-exact score entries and the same
`m = 0.125`, `l = 34.0713806`, `context[0] = 1.85358536`,
`context[15] = 1.82630539`. That bit-for-bit agreement with the already
validated GM-workspace revision, against an oracle that did not change, is the
"establish numerics before concurrency" evidence section 6.3 of the
[guidelines](../MICROROUTINE_DESIGN_GUIDELINES.md) asks for.

## The hardware CV path

`routines/routine_cv_transfer_ub_to_l1.hpp` moves the dequantised FP16 key
page from Vector UB straight into Cube L1. The public entry point is

```cpp
DataCopy(LocalTensor<T> dst /*A1|B1*/, LocalTensor<T> src /*VECCALC*/,
         DataCopyParams)
```

which on `dav-3510` reaches `DataCopyUB2L1Impl`
(`impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:619`). That function
is `__in_pipe__(MTE3) __out_pipe__(MTE3)`, and for any AIC:AIV ratio other than
1:1 its body is a single `CopyUbufToCbuf`, i.e. the `copy_ubuf_to_cbuf`
hardware move in
`impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h`.
The kernel `#error`s if `__MIX_CORE_AIC_RATION__ == 1`, because at 1:1 the same
call silently becomes a KFC relay through GM.

The move shows up in the AIV1 trace as one instruction per page:

```
[00001827] MTE3 : (ID: 000664) MOV_UB_TO_L1  Src:UB, Dst:L1, XD:X18=0x220, XN:X14=0x1080
[00001876] MTE3 : (ID: 000727) MOV_UB_TO_L1  Src:UB, Dst:L1, XD:X19=0x440, XN:X14=0x12a0
[00001925] MTE3 : (ID: 000801) MOV_UB_TO_L1  Src:UB, Dst:L1, XD:X4=0x660,  XN:X2=0x14c0
```

`XD` is the L1 destination: 0x220, 0x440, 0x660 = 544, 1088, 1632, exactly the
three `kL1KeySlot` bases. `XN` is the UB source: 0x1080, 0x12a0, 0x14c0 = 4224,
4768, 5312, exactly the three `kUbKeyRing` slots. The analyzer asserts both the
count and that the staging core issues **zero** `Dst:OUT` stores, so the key
page provably never reaches Global Memory.

### Why not `DataCopy(dst, src, Nd2NzParams)`

The `Nd2NzParams` overload exists for UB → L1 and is the API the task
description names, but on `dav-3510` it is `DataCopyUB2L1ND2NZImpl`: it calls
`PopStackBuffer<T, TPosition::LCM>()` and `GetTPipePtr()->FetchEventID()`,
rearranges the cube inside UB with a vector `TransND2NZ`, and only then issues
the same `CopyUbufToCbuf`. A kernel that places its own `LocalTensor`s never
instantiates a `TPipe`, so `GetTPipePtr()` is null and both accesses land at GM
`0x40` — the exact fault mechanism dissected in section 0 of the guidelines.
The arch-3510 variant with none of that, `DataCopyUB2L1ND2NZImplV2`, is
library-internal and unreachable from the public interface, so the leaf
reimplements its burst schedule on the `DataCopyParams` overload: one
`(NValue × 1 block)` strided burst per C0 column strip, emitted by a fold over
a compile-time index sequence. Same instruction stream, no `TPipe`.

### Fractal placement, and the one contract change

ND → NZ at 32-byte block granularity is a strided *placement*, not an element
permutation. For a 16-wide FP16 panel the row is exactly one C0 block, so the
schedule collapses to a single contiguous 16-block burst — which is why the
trace shows one instruction per page rather than a loop.

The Cube B operand wants Zn. For one 16×16 half fractal, Nz is
`(M1,N1,M0,N0)` row major and Zn is `(K1,N1,N0,K0)`, its transpose, so
`Nz(X)` and `Zn(X^T)` are the same bytes. Staging the ND page `K[key][d]` as Nz
therefore lands exactly the Zn image of `K^T` that `CopyL12L0B` expects, with
no element permutation anywhere on chip.

That costs one change to the host contract: `k_packed` is now packed key-major,
nibble index `key * kDim + d`, instead of `d * kKeys + key`. This is a KV-cache
layout decision, which is where it belongs — a production cache stores pages in
the fractal order the Cube reads. The alternative would be an element transpose
on the Vector side, which the 32-byte burst engine cannot express. `v_packed` is
unchanged at `ch * kKeys + key`.

## Work split across the two Vector cores

| Core | Per page |
| --- | --- |
| **AIV1** | MTE2 ingress of packed K, Vector dequant to FP16, MTE3 UB → L1 staging |
| **AIV0** | MTE2 ingress of packed V, Vector dequant to FP32, the online-softmax recurrence, and the single outflow |
| **AIC** | quorum wait, MTE1 L1 → L0A/L0B, `Mmad`, Fixpipe L0C → AIV0 UB |

The recurrence sits on AIV0 because `Te::FixpipeParams` defaults
`subBlockId = false`, so `routine_cube_matmul.hpp`'s Fixpipe lands in sub block
0's UB without the leaf being touched. Every leaf is used unmodified.

The split is not balanced — AIV0 runs 2053 instructions against AIV1's 317 —
but both cores do real work on the critical path, and they are concurrently in
flight for 1551 of the 5381 dispatch cycles. A balanced split would have to cut
the recurrence across cores, which needs 8 copies of the carried state or a
reworked accumulator leaf; neither is what an N = 3 depth probe establishes.

## Three-stage modulo schedule

Column `j`, slots taken modulo `kStageDepth = 3` at compile time:

| stage | work | tile | slot |
| --- | --- | --- | --- |
| 0 | MTE2 ingress + dequant + CV stage | `j + 2` | `(j + 2) % 3` |
| 1 | MTE1 → L0, `Mmad`, Fixpipe | `j + 1` | `(j + 1) % 3` |
| 2 | online-softmax recurrence | `j` | `j % 3` |

Nothing in a column depends on anything else in that column. The three columns
are instantiated as `StepColumn<2,1,0>`, `StepColumn<0,2,1>`,
`StepColumn<1,0,2>` on each of the three cores, with `if constexpr` ramp flags,
so the steady state has no loop, no `%`, no predicate and no scalar read of UB.
Prologue is stage 0 for tiles 0 and 1 plus stage 1 for tile 0; ramp-down falls
out of the compile-time flags.

### Mode-4 double handshake

`set_intra_block` / `wait_intra_block` ids form one space per AI Core block, of
which the library gives the upper half to sub block 1 — `GetIntraFlagId` adds
`subBlockId * INTRA_NUM`, and `SyncAllImpl` on a non-1:1 ratio waits both
`SYNC_AIV_FLAG` and `SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX`. The trace confirms the
convention directly: an AIV prints `sync_id` relative to its own half while the
AIC prints the absolute id, so AIV1's raw id 16 appears as `sync_id:0` on AIV1
and `sync_id:16` on the AIC.

| Flag | ids | Direction | Sets | Waits |
| --- | --- | --- | ---: | ---: |
| slot ready, value page | 0..2 | AIV0 → AIC | 3 | 3 |
| product ready | 4..6 | AIC → AIV0 | 3 | 3 |
| panel free | 8..10 | AIV0 → AIC | 3 primed + 3 | 3 + 3 drained |
| slot ready, key page | 16..18 | AIV1 → AIC | 3 | 3 |
| L1 slot free | 20..22 | AIC → AIV1 | 3 primed + 3 | 3 + 3 drained |

Before each GEMM the Cube waits one `PIPE_MTE1` flag from each Vector core:

```
WAIT_INTRA_BLOCK  PIPE:MTE1, XT:X5=0x10, sync_id:16     <- AIV1, key staged
WAIT_INTRA_BLOCK  PIPE:MTE1, XT:X2=0,    sync_id:0      <- AIV0, value dequantised
```

so a page is multiplied only once both Vector cores have finished their half of
the staging. Intra-core ordering is 28 `(route, ID)` HardEvent pairs —
`MTE3_V`, `MTE2_V`, `V_MTE3` on AIV1, `V_MTE2`, `MTE2_V`, `V_MTE3`, `MTE3_S` on
AIV0, `M_MTE1`, `MTE1_M`, `FIX_M`, `MTE2_MTE1` on the AIC — with the `Mmad`
and `Fixpipe` paired under matching `unitFlag = 3`. There is no
`PipeBarrier<PIPE_ALL>` anywhere in the kernel.

## Measured results

```
PASS fused attention: 3 pages, 768/768 score entries bit-exact,
  max 2.000 ULP (worst operand context), 16/16 FP16 context within 1 half ULP,
  1024/1024 GM guard bytes intact
  final m=0.125 l=34.0713806  context[0]=1.85358536 context[15]=1.82630539
```

| Metric | Value |
| --- | --- |
| Device exit code | 0 |
| `invalid ldst addr` / `unrecognize ldst addr` / `su_ccu_mpu_err` | 0 |
| `[error]` / `[fatal]` simulator lines | 0 |
| `core0.excp_log.dump` / `core1.excp_log.dump` | 0 bytes |
| Score panels | 768/768 bit-exact FP32 |
| Max error, `m` / `l` / FP32 context | 2.000 ULP (budget 8.0) |
| FP16 context | 16/16 within one half ULP |
| GM guard bands (`0xDEADBEEF`) | 1024/1024 bytes intact, 8 buffers |
| GM workspace bytes for key staging | 0 |
| Replicated-state invariant | all 8 lanes of the `m` and `l` blocks identical |
| Model ticks | 6404 |
| Dispatch span | cycles 692..6073 (5381 cycles) |
| Completion span | cycles 696..6093 (5397 cycles) |

Three runs of the same build gave model ticks 6404, 6432 and 6438 and an
absolute span start of 691 or 692. The span *lengths* (5381 and 5397), every
engine occupancy and every overlap figure were identical across all three, so
the ±1 cycle and ±34 tick spread is launch jitter, not measurement noise in
the numbers this report draws conclusions from.

### Engine occupancy and overlap

```
engine busy cycles: MTE2 1160, Cube 1700, Vector 4094, MTE3 675
  overlap Cube   + Vector:  1693 cycles (widest 1273..2966)
  overlap Cube   + MTE2  :  1023 cycles (widest 1273..2296)
  overlap Cube   + MTE3  :   100 cycles (widest 1925..1959)
  overlap Vector + MTE2  :  1156 cycles (widest 1140..2296)
  overlap Vector + MTE3  :   100 cycles (widest 1925..1959)
  overlap MTE2   + MTE3  :   100 cycles (widest 1925..1959)
four-engine overlap: 100 cycles, widest CAModel cycles 1925..1959
AIV0 + AIV1 concurrent: 1551 cycles, widest CAModel cycles 703..1961
```

All six pairwise overlaps are non-empty and MTE2, Cube, Vector and MTE3 are
concurrently in flight for 100 cycles. The four-engine window is *narrower*
than the previous revision's 438 cycles, and that is the expected consequence
of the change rather than a regression: removing the GM round trip cut MTE3
occupancy from 3891 to 675 cycles and MTE2 from 2073 to 1160, so there is far
less DMA left to overlap with, while the whole run got 30% shorter. Reporting
the overlap figure without that context would be misleading.

### Where the time goes, honestly

Vector is the critical engine at 4094 of 5381 dispatch cycles, and the Cube
retires its last Fixpipe at cycle 1800 — a third of the way through. With three
pages, three-deep buffering and a primed panel-free back edge, the Cube drains
the whole sequence before the recurrence has caught up, so the measured span is
recurrence-bound and the tail from cycle ~1961 to 6073 is AIV0 finishing the
last two recurrences plus the outflow. The back edges that would throttle the
Cube to the recurrence rate exist and are primed and drained, but at N = 3 the
ring never wraps, so they are structural rather than load-bearing in this run.
This figure is a depth-boundary measurement, not a steady-state interval.

## Verification checklist

| Requirement | Evidence |
| --- | --- |
| Zero GM workspace bytes for key staging | `ws_k` argument and buffer removed; analyzer asserts 0 `Dst:OUT` stores on the staging core |
| Both AIV0 and AIV1 active | 2053 and 317 instructions, 1551 cycles concurrently in flight |
| Zero scalar load/store errors | 0 matches for `invalid ldst addr`, `unrecognize ldst addr`, `su_ccu_mpu_err`; both `excp_log` dumps empty |
| Canary guard bands intact | 1024/1024 GM bytes of `0xDEADBEEF` over 8 buffers |
| Context within 2.0 ULP | max 2.000 ULP, worst operand `context` |
| All 768 score entries bit-exact | 768/768, `bits(have) == bits(want)` |
| No per-page MTE3 dump | 5 GM stores, all after the last Fixpipe at cycle 1800 |
| No `PipeBarrier<PIPE_ALL>` | none in the kernel source; Cube path uses `M_MTE1`, `MTE1_M`, `FIX_M` and `unitFlag = 3` |

## What remains untested

- Silicon. CAModel is cycle accurate but is not hardware; it does not validate
  timing closure, power, or multi-core scale out.
- `N = 3` is a depth boundary, not sustained throughput. The ring never wraps,
  so the reuse back edges are exercised for token balance but not for
  contention, and the overlap figures are for this depth only.
- One AI Core (`blockDim = 1`). No inter-AI-Core (mode 0) synchronization and
  no multi-block key-page distribution.
- The recurrence and outflow cover query row 0. The Cube produces all 16 rows
  and every one of the 768 score entries is checked, but the (m, l, Acc)
  recurrence is instantiated once, not 16 times.
- The Vector work split is deliberately asymmetric; a balanced split needs a
  per-core accumulator contract that these leaves do not have.
- Dequantisation has no zero point, so the quantised range is unsigned 0..15
  times a scale. A signed or per-group quantiser is a different leaf contract.
- `DataCopyUB2L1ND2NZImplV2`'s multi-strip branch is reimplemented and
  compile-time asserted but, at `DValue == C0`, only the single-strip case
  actually executes here. A panel wider than one C0 block would exercise the
  fold.
