# Micro-routine design guidelines — Ascend 950PR / dav-3510 / CANN 9.2

Pipeline and buffer-lifetime refactors of real operators follow the primary
[combat kernel refinement baseline](REFINEMENT_OF_COMBAT_KERNELS.md): token model,
UB checks, isolated logical-page head/middle/tail canaries, ring wrap, trace audit.


Target: `Ascend950PR_9599` CAModel, `__NPU_ARCH__ == 3510`, CANN
9.2.0-beta.2 (`ccec` / BiSheng). Every rule below is traced either to a
CANN 9.2 implementation header or to a CAModel instruction trace from this
repository. Where a rule is inferred rather than observed, it says so.

Evidence paths are relative to
`/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/asc`.

Machine constants used throughout:

| Constant | Value | Source |
| --- | ---: | --- |
| UB DataBlock | 32 B (8 × FP32) | `general_description_and_constraints.md` |
| `VECTOR_REG_WIDTH` | 256 B | `impl/basic_api/utils/kernel_utils_constants.h:118` |
| `GetVecLen()` | `VECTOR_REG_WIDTH` for arch 3510 | `include/basic_api/kernel_operator_sys_var_intf.h:75` |
| UB window | 256 KB (192 KB used as the harness budget) | `kernel_utils_constants.h` |

---

## 0. What this cycle actually found

The online-softmax leaf was emitting 16 CAModel errors per launch:

```
[ERROR] pem_lsu.cc:477 mem_map_calc_49bit_op0 invalid ldst addr.
ldst_addr: 64
[ERROR] pem_lsu.cc:438 mem_map_decode unrecognize ldst addr.
[error] [core0.veccore0] [su_ccu_mpu_err_t0] (PC: 0x10d0d4c8) SCALAR :
  (ID: 000294) LD_XD_XN_IMM dtype:B64, XD:X19=0, XN:X15=0, IMM:0x40,
  accessUb:0, accessDdr:1,
```

Four distinct PCs, each hit once per tile, four tiles → 16 errors. Decoding
the surrounding window of `core0.veccore0.instr_popped_log.dump` gives the
whole mechanism:

```
ID 293: LD   X15 <- [X13+0]        ; X15 = 0          <-- null pointer loaded
ID 294: LD   X19 <- [X15+0x40]     ; addr 64          <-- FAULT (alloc)
ID 295: SFF  X12 <- scan-first-free(X19)
ID 296: SBITSET X19 <- set bit X12
ID 297: ST   [X15+0x40] <- X19     ; addr 64          <-- FAULT (alloc)
ID 301: SET_FLAG  PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0
ID 302: WAIT_FLAG PIPE:VEC, TRIGGER PIPE:SCALAR, FLAG ID:0
ID 303: SBITSET X15 <- clear bit X12
ID 304: ST   [X13+0x40] <- X15     ; addr 64          <-- FAULT (release)
```

`SFF`/`SBITSET` around a `SET_FLAG`/`WAIT_FLAG` pair is a **dynamic event-ID
allocator**, and its free-flag bitmap lives behind a pointer that is null.
The producer is a single library call:

```cpp
// impl/basic_api/dav_3510/kernel_operator_vec_reduce_impl.h:736-758
template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T* dstLocal, ...)
{
    ...
    if constexpr (is_same<T, float> || is_same<T, half>) {
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(eventID);
        accValFloat = dstLocal[0];      // scalar load of UB
    }
}
```

`AscendC::ReduceSum` on FP32/FP16 **always** drains the vector pipe and
scalar-reads lane 0, purely so that `GetAccVal()` can observe the result.
These kernels place their own `LocalTensor`s and never instantiate a
`TPipe`, so `GetTPipePtr()` is null and both bitmap accesses land at GM
`0x40`. That is the entire fault.

### Two corrections to the original diagnosis

Both were worth getting right, because they point at different fixes.

1. **`Duplicate(dst, srcTensor, count)` does not perform a scalar
   dereference.** It lowers to `Reg::LoadAlign` + `Reg::Duplicate`
   (`kernel_operator_vec_duplicate_impl.h:164-179`) — pure vector pipe, no
   `GetValue`, no `GetTPipePtr`, no `SET_FLAG`. It was never the source of
   the scalar trap. It *does* carry a separate, real hazard (§2.3).
2. **The faults were not an alignment problem.** Every operand was already
   32-byte aligned. Padding state to 32 bytes is required for other reasons
   (§2.1) but would not have removed a single error.

### The bug the fix uncovered

Removing `ReduceSum` removed an *accidental* full vector-pipe drain once per
tile. That drain had been masking a latent **V→MTE2 WAR hazard**: `scores`
and `v_by_channel` are single-buffered, and nothing stopped MTE2 from
refilling them while the vector pipe was still reading the current tile. A
per-tile GM trace showed it plainly:

| tile | device `p_sum` | reference `p_sum` | device actually used |
| ---: | ---: | ---: | --- |
| 0 | 7.270860 | 6.715189 | tile 1 scores |
| 1 | 7.340635 | 7.270860 | tile 3 scores |
| 2 | 7.340635 | 6.958729 | tile 3 scores |
| 3 | 7.340635 | 7.340635 | tile 3 scores |

MTE2 had run to completion before the vector pipe consumed anything. `m`
stayed correct only because every tile in this fixture has the same maximum
(1.0), which is exactly why the bug survived the original validation.

**A performance fix that removes an incidental barrier can expose a
correctness bug that barrier was hiding. Re-validate numerics, not just the
error count, after deleting any synchronization.**

---

## 1. The Zero Scalar Compute Invariant

### 1.1 Why the Scalar core must stay a dispatcher

On DaVinci v3 the Scalar unit runs **asynchronously and far ahead** of the
Vector/MTE pipes — it is the instruction dispatcher, not a co-processor. The
modulo3 trace shows Scalar queuing its final event waits at cycle 1,686
while the last MTE3 copy retires at cycle 2,798. Any scalar read of a value
the Vector pipe produces must therefore be fenced by a `V_S` event, and that
fence:

- costs a **full vector-pipe drain** at that point, destroying software
  pipelining;
- requires a **valid `TPipe`** to allocate the event ID, which a kernel with
  hand-placed `LocalTensor`s does not have.

So the invariant is not stylistic. In this codebase a scalar trip is both a
performance cliff and a hard fault.

### 1.2 Prohibited inside a leaf routine

- `LocalTensor::GetValue()` / `SetValue()` and any `tensor[i]` scalar
  subscript on UB.
- Any host-style control flow whose predicate is a UB value.
- `GetAccVal()` and the `ReduceSum` accumulator side channel.
- Deriving an index or stride from computed UB data. Index vectors are
  supplied by the caller (the online-softmax leaf takes
  `p_repeat_offsets` as a GM-resident input for exactly this reason).

### 1.3 API audit — which calls leave the vector pipe

Verified by reading the dav-3510 implementations and by error-count
attribution in CAModel (one `ReduceSum` call site × 4 tiles × 4 faulting
accesses = the 16 observed errors; no other call site contributed).

| API | Leaves vector pipe? | Needs `TPipe`? | Evidence |
| --- | --- | --- | --- |
| `ReduceSum` (FP32/FP16) | **Yes** — `V_S` drain + `dstLocal[0]` scalar read | **Yes** | `vec_reduce_impl.h:736-758` |
| `ReduceMax` / `ReduceMin` | No | No | `vec_reduce_impl.h:1079`, no `GetTPipePtr` |
| `ReduceRepeat<SUM/MAX/MIN>` | No | No | `ReduceAlignCall`/`ReduceUnalignCall`, no scalar path |
| `Brcb` | No | No | `vec_brcb_impl.h` `BrcbCommonImpl` |
| `Duplicate` (scalar src) | No | No | `DuplicateFromScalarImpl` |
| `Duplicate` (tensor src) | No | No | `DuplicateFromTensorImpl` |
| `Add`/`Sub`/`Mul`/`Max`/`Div`/`Exp` (level 2, continuous) | No | No | `vec_binary_continuous_impl.h` |
| `Adds`/`Muls` level-0 **bit-mask mode, 1-byte types** | **Yes** — `V_S` + `S_V` | **Yes** | `vec_binary_scalar_impl.h:114,122` |
| `Gather` | No | No | no `GetTPipePtr` in impl |

The last row is narrow: the trip is guarded by
`isMaskBitMode && SupportBytes<T,1>()`. The level-2 continuous
`Adds(dst, src, imm, count)` used in these kernels does not take it.

**Rule Z1 — never call `ReduceSum`/`ReduceMin`-with-accumulator from a leaf
routine.** Use `ReduceRepeat<ReduceType::SUM>`, which produces the same
value, needs no workspace, and never leaves the vector pipe:

```cpp
// sum kOnlineTile FP32 (== 2 DataBlocks) into dst[0], no scalar trip
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, float>(
    dst, src, /*mask*/ 16, /*repeatTime*/ 1, /*dstRepStride*/ 1,
    /*srcBlkStride*/ 1, /*srcRepStride*/ 2,
    AscendC::ReduceOrder::ORDER_ONLY_VALUE);
```

**Rule Z2 — if you must keep a level-2 API that needs a `TPipe`, give the
kernel a real `TPipe`.** That is the alternative fix, and it is the wrong
trade here: it restores correctness of the event allocation but keeps the
per-tile pipe drain, and a `TPipe` also wants to own UB allocation, which
conflicts with the explicit `{VECCALC, offset, len}` layout this harness
depends on.

### 1.4 Broadcasting without a scalar trip

`Brcb` reads 8 source elements and splats each into its own 32-byte
DataBlock. For `repeatTime == 1` and `BrcbRepeatParams{1, 8}` it reads
exactly one DataBlock and writes 8 (256 B):

```cpp
AscendC::Brcb(dst64, src8, 1, {1, 8});   // dst[0..7] = src[0], dst[8..15] = src[1], ...
```

This gives a clean composition rule with §2.2: **if the source block is
lane-replicated, one `Brcb` repeat fills all 64 destination lanes with the
same value**, so a single instruction broadcasts a state scalar to any tile
width up to 64 FP32. A reduction destination is *not* replicated (a
reduction writes lane 0 only), so feed it straight to `Brcb` and consume
only the first output block — `Brcb` re-establishes replication there.

`Duplicate(dst, srcTensor, n)` remains the right API when the source is not
replicated or `n > 64` (`routine_vector_normalize_store.hpp` keeps it for
precisely that reason), subject to Rule A3 below.

### 1.5 The `Reg::` level, and the seven traps in it

Everything above is about the level-2 APIs, whose operands live in Unified
Buffer. Inside a `__simd_vf__` body the `AscendC::Reg::` APIs operate on
vector registers directly, and a leaf that keeps its working set resident
there is a different performance regime entirely — measured, on the same
256-channel randomised-Hadamard cascade over 16 rows
(`reports/rotate_butterfly.md`):

| form | vector instructions | `RVECEX` | `RV_VLD` | `RV_VST` | vector span |
| --- | --- | --- | --- | --- | --- |
| level-2 over UB operands | 11250 | 2512 | 3072 | 1728 | 30215 |
| `Reg::` register-resident | 1203 | 1057 | 64 | 64 | 1850 |

Same arithmetic, same answer bit for bit, 16.3x the Vector-engine span. Three
things cost the level-2 form that are not arithmetic: nine stage results per
row written back to UB and read out again (`RV_VLD`/`RV_VST` are 70% of its
vector instructions against 22% for `RVECEX`), **288 `VF` entries** — one per
level-2 API call, each opening and closing its own vector function — and
1,696 `RV_SMEM_BAR` plus 1,440 `RV_SEND` of intra-VF bookkeeping. Its
`RVECEX` is also 2.4x the register form's for identical work, because a
repeat-form instruction covers 64 lanes and a 256-lane stage therefore needs
four `RV_VLOOP`-driven repeats where the register form has four *named*
registers and no loop.

**Rule R1 — a leaf whose working set fits in the register file should stay
there, and should prove it by instruction count.** Four register loads and
four register stores per 256-channel FP32 row is what no-spill looks like;
anything more is a spill, and the count is in the trace.

The seven API facts that cost a build cycle each, all verified against the
dav-3510 implementation headers and then on CAModel:

| Fact | Consequence |
| --- | --- |
| `Reg::Arange` has **no unsigned form**. Its `vci` intrinsic is declared for s8/s16/s32/f16/f32 only. | Build the ramp as `int32_t`. |
| `Reg::Gather` requires an index `RegTensor` of `uint8_t`/`uint16_t`/`uint32_t` **of the same width as the data**. | The `int32_t` ramp has to become `uint32_t` somehow — see the next row. |
| `Cast<uint32_t, float>` and `Cast<uint32_t, int32_t>` **do not exist**; the FP32 narrowing pairs are `int32_t <- float` (`f322s32`) and `int16_t <- float`, and the only b32 unsigned pairs go *down* to 16 or 8 bits. | A conversion cannot bridge it. |
| A **reference reinterpret between two `RegTensor`s of one register width** is the vendor's own idiom: `kernel_reg_compute_vec_vconv_impl.h` writes `(RegTensor<uint16_t>&)srcReg` in the bf16 -> fp8e8m0 path. | That is how the ramp, the sign bit patterns and the FP32/uint32 views of a quantiser's absmax are shared, at no instruction cost. |
| `ShiftLefts` / `ShiftRights` take their shift count as an **`int16_t` and nothing else** (`static_assert(SupportType<U, int16_t>())`), while `Muls`/`Mins`/`Maxs` accept every width. | Cast the literal, or the error arrives as a template instantiation trace. |
| `Reg::Gather` lowers to **one `RV_VSELR`** and `Reg::MulAddDst` to **one `RV_VMULA`**. | A radix-2 XOR butterfly stage is two instructions per register: gather the partner, fold in the signed self term. |
| `Reg::DeInterleave` lowers to **one `RV_VDINTLV`** and `Reg::Select` to one `RV_VSEL`, and both take a lane-invariant source, so a constant mask pattern costs one `Interleave` of two `Duplicate`s. | A table-free butterfly is four instructions per register instead of two, and measured 6% slower — a fallback, not a rewrite, if `Gather` ever becomes multi-cycle. |

Two composition rules follow and are worth stating separately.

**Rule R2 — build per-stage constants in-register, not in Unified Buffer.** A
sign vector of `±1.0f` keyed on bit `k` of the lane index is

```cpp
//  ((lane & s) << (31 - k)) | 0x3F800000     -- the stride's own bit, moved
//                                               to the FP32 sign position
Reg::And<uint32_t>(bit, lane, stride, mask);
Reg::ShiftLefts<uint32_t>(shifted, bit, int16_t(31 - k), mask);
Reg::Or<uint32_t>(sign_bits, shifted, one_bits, mask);
```

three instructions with no comparison, no mask register and no conversion,
against the 6 KiB of UB offset and sign vectors the level-2 form needs.

**Rule R3 — a cross-lane reduction is a butterfly all-reduce, not a `Reduce`
plus a broadcast.** Six rounds of (gather the XOR partner, take the max) over
the index registers an XOR butterfly already carries leave the result in
**every** lane, which is the form a following per-lane multiply wants anyway:
fifteen instructions, no reduction API and no new constant. `Reg::Reduce`
writes lane 0 and then needs a broadcast to be usable.

---

## 2. Memory alignment and bank invariants (rule AKA3003)

### 2.1 Rule A1 — every UB operand base is 32-byte aligned

A `LocalTensor` base that is not 32-byte aligned is not addressable by a
vector operand at all; `Brcb` and the reduction APIs restate the requirement
explicitly. The practical consequence: **a logically scalar value occupies a
whole 32-byte DataBlock**, because that is the allocation granularity, not
because a 4-byte store would corrupt its neighbours (stores are lane-masked,
§2.3, so they are exact).

Encode it, do not assume it:

```cpp
constexpr std::uint32_t RoundUpBlock(std::uint32_t b)
{ return (b + 31u) / 32u * 32u; }
constexpr std::uint32_t NextSlot(std::uint32_t base, std::uint32_t bytes)
{ return RoundUpBlock(base + bytes) + kBankSkew; }

static_assert(kAlpha % 32u == 0, "every UB operand base must be 32-byte aligned");
```

Both CAModel kernels in this repo now derive every offset through
`NextSlot` and `static_assert` the result, so a layout edit cannot silently
break alignment.

### 2.2 Rule A2 — 32-byte bank skew between live slots

Consecutive slots are separated by one spare DataBlock so two live operands
do not land on the same UB bank column and serialize their load ports. The
modulo3 skeleton states the same invariant structurally
(`static_assert(kSlotStrideBytes % 256u != 0u)`): a slot stride that is a
multiple of 256 B puts every slot on the same bank phase.

This is a throughput rule, not a correctness rule, and this harness has not
measured the penalty on 950PR. It is applied because it is free.

### 2.3 Rule A3 — the source read quantum is 256 B, not 32 B

This is the rule that was missing, and it is the one with teeth. A level-2
continuous vector op on dav-3510 reads **a whole vector register, unmasked**,
from each source base, and only masks the result:

```cpp
// impl/basic_api/dav_3510/kernel_operator_vec_binary_continuous_impl.h:45-50
for (uint16_t i = 0; i < repeatTime; ++i) {
    mask = Reg::UpdateMask<T, RegType::trait>(sreg);
    Reg::LoadAlign(src0Reg, src0 + i * repeatStride);   // UNMASKED, 256 B
    Reg::LoadAlign(src1Reg, src1 + i * repeatStride);   // UNMASKED, 256 B
    func(dstReg, src0Reg, src1Reg, mask);
    Reg::StoreAlign(dst + i * repeatStride, dstReg, mask);   // masked, exact
}
```

So `Max(m_new, m_prev, tile_max, /*count=*/8)` reads **256 bytes** from each
of `m_prev` and `tile_max`, not 32. Consequences:

- **Writes are exact.** A masked `StoreAlign` touches only `count` lanes, so
  packing distinct variables into one block would not corrupt them. A1 is an
  addressability rule, not an anti-corruption rule.
- **Reads over-run.** Every source base needs `VECTOR_REG_WIDTH` bytes of
  *addressable* UB behind it. Inside a padded arena the surplus lanes are
  discarded by the mask and are harmless; a source placed within 256 B of the
  end of the UB window faults.

```cpp
static_assert(kArenaEnd + kVecReg <= kUbWindow, "UB arena leaves no read pad");
```

Not every API has this footprint. Measured read widths:

| API | Source read footprint |
| --- | --- |
| level-2 continuous binary/unary (`Add`, `Mul`, `Max`, `Div`, `Exp`, …) | 256 B per source, **unmasked** |
| `Duplicate(dst, srcTensor, n)` | 256 B, **unmasked**, independent of `n` |
| `Brcb` | exactly 32 B per repeat (`Reg::LoadDist::DIST_E2B_B32`) |
| `ReduceRepeat` / `ReduceMax` | **masked** `LoadAlign`, so `mask` lanes only |

`Duplicate`'s tensor-source form is therefore a 224-byte over-read when
pointed at a 32-byte state block. That is survivable inside a padded arena
and fatal at the arena edge — which is why the online-softmax leaf switched
to `Brcb` (its state is replicated anyway, so `Brcb` is strictly better) and
the normalize leaf kept `Duplicate` behind an explicit
`static_assert(kLDenominator + kVecReg <= kUbWindow)`.

### 2.4 Rule A4 — reduction destinations and MTE3 lengths

- `ReduceRepeat<SUM, float>` needs 4-byte destination alignment;
  `MAX`/`MIN` with `ORDER_VALUE_INDEX` needs 8-byte (`ReduceRepeat.md`
  Table 3). Allocating a full DataBlock satisfies every case.
- A `Brcb` destination must hold `8 × 32 B = 256 B` per repeat; the store
  uses a full mask and writes all of it.
- FP16/BF16 MTE3 stores must be a whole number of 32-byte bursts, so a
  normalize tile length must be a multiple of 16 elements.

---

## 3. RAW / WAR hazard management inside AIV

Three distinct mechanisms, routinely conflated. Picking the wrong one either
deadlocks or silently corrupts.

### 3.1 Inter-pipe hazards — `SetFlag`/`WaitFlag`, caller-owned

Different engines (MTE2, Vector, MTE3, Scalar, Cube) are independent queues.
Ordering between them needs a `HardEvent` token. Model the kernel as a
**marked graph**: one token per free buffer slot, every feedback cycle primed
with at least one token, sets and waits balanced per `(route, ID)`.

The forward edges are the obvious ones and were already present:

```cpp
DataCopy(ub, gm, n);
SetFlag<HardEvent::MTE2_V>(0);  WaitFlag<HardEvent::MTE2_V>(0);   // MTE2 -> V
compute();
SetFlag<HardEvent::V_MTE3>(0);  WaitFlag<HardEvent::V_MTE3>(0);   // V -> MTE3
```

**The back edge is the one that gets forgotten.** With a single-buffered
input, MTE2 must not refill the buffer until the Vector pipe has finished
reading it:

```cpp
SetFlag<HardEvent::V_MTE2>(0);              // prime: input slot free
for (tile = 0; tile < N; ++tile) {
    WaitFlag<HardEvent::V_MTE2>(0);         // MTE2 waits for the reader
    DataCopy(scores, gm_scores[...], kTile);
    SetFlag<HardEvent::MTE2_V>(0);  WaitFlag<HardEvent::MTE2_V>(0);
    compute();
    SetFlag<HardEvent::V_MTE2>(0);          // reader done, slot free again
}
WaitFlag<HardEvent::V_MTE2>(0);             // drain, no token leaks
```

Token balance: 1 prime + N sets = N+1 sets; N loop waits + 1 drain = N+1
waits. **Always drain the primed token after the loop** — a leaked event bit
persists across launches and will hang or mis-trigger the next one.

Omitting this back edge is what produced the table in §0. It is invisible to
any test whose tiles happen to agree on the value being checked.

**A `HardEvent` names the route, and the wait stalls the *consumer* queue.**
`HardEvent::A_B` is producer `A`, consumer `B`: `SetFlag<A_B>` is issued on
`A`'s queue and `WaitFlag<A_B>` stalls **`B`'s** queue, nothing else. The
CAModel trace states both halves — the `PIPE:`/`TRIGGER PIPE:` pair is the
route, and the pipe column at the left of the line is the queue the
instruction was issued on:

```
MTE3  : SET_FLAG   PIPE:MTE3, TRIGGER PIPE:VEC   <- issued on MTE3, frees V
VEC   : WAIT_FLAG  PIPE:MTE3, TRIGGER PIPE:VEC   <- stalls V
MTE2  : WAIT_FLAG  PIPE:VEC,  TRIGGER PIPE:MTE2  <- stalls MTE2
```

So the route has to be chosen by **which queue must stall**, not by which
engines are involved in the hazard. A ring slot that MTE2 refills and the
Vector pipe reads needs `V_MTE2`; `MTE3_V` placed in front of the `DataCopy`
looks like it guards the refill and reads like it in source order, but it
stalls `V` and leaves MTE2 ungated. The fused paged-attention kernel shipped
exactly that mistake on its staging core for one revision.

Two consequences, and the second is why it survived validation:

1. **A queue with no incoming wait free runs.** Per core, per queue, count the
   waits that can stall it. A queue stalled by nothing will run as far ahead
   as its instruction stream allows, however carefully the *other* queues are
   synchronised.
2. **A flag is one bit per `(route, ID)`, not a counter.** A producer that
   runs ahead sets the same id twice before the first wait consumes it; the
   second token is silently lost and the next wait on that id hangs. Source
   level token balance still looks perfect, so balance accounting is necessary
   and not sufficient.

Neither is observable at the ring depth. With `kStageDepth` slots and one
event id per slot, a run of exactly `kStageDepth` tiles reuses no slot and
repeats no id, so there is no WAR hazard to hit and no id to double set.
**Run the probe past the ring depth** — at least `2 * kStageDepth` tiles, so
every slot is refilled at least once — or the back edges are structural rather
than load bearing and the run proves only that the token arithmetic adds up.

### 3.2 Intra-pipe hazards — `PipeBarrier<PIPE_V>`, leaf-owned

A `PipeBarrier<PIPE_V>` orders the vector pipe **against itself**. It
allocates no event ID and creates no token the caller must balance, so it
does **not** breach leaf-routine purity the way `SetFlag`/`WaitFlag` would —
a leaf may issue one; a leaf may not issue a `HardEvent`.

It also **cannot** express a cross-engine edge. A `PipeBarrier<PIPE_V>` at
the top of a tile loop does nothing whatsoever to stop MTE2 running ahead;
that was the second trap in §0.

Where the online-softmax leaf places them: at the two
reduction → broadcast boundaries, where a 4-byte lane-0 write is consumed as
a full 32-byte block read.

```cpp
ReduceRepeat<SUM, float>(p_sum_raw, p, 16, 1, 1, 1, 2, ORDER_ONLY_VALUE);
PipeBarrier<PIPE_V>();            // lane-0 write -> whole-block read
Brcb(p_sum_bc, p_sum_raw, 1, {1, 8});
PipeBarrier<PIPE_V>();
Mul(l_scaled, l_prev, alpha, 8);
```

Scope note: these two barriers are belt-and-braces. The compiler demonstrably
does order most same-pipe UB dependencies across separate VF blocks — the
routine's other producer/consumer pairs carry no barrier and are bit-correct.
This harness has not run the ablation that would prove the two barriers
unnecessary, so they stay, at a cost of two instructions per tile.

### 3.3 Cross-engine RAW that no barrier covers

Fixpipe → MTE3 and Cube → Vector dependencies are not vector-pipe
dependencies. The Hadamard leaf pairs one `Mmad` with one `Fixpipe` under
matching `unitFlag=3` so the L0C producer/consumer dependency is carried by
the instruction encoding rather than by an event.

### 3.4 Caller-owned boundaries versus leaf purity

The contract this harness uses:

| Concern | Owner |
| --- | --- |
| Input readiness (`MTE2_V`), output visibility (`V_MTE3`, `MTE3_S`) | Caller |
| Buffer-reuse back edges (`V_MTE2`, `MTE3_V`) | Caller |
| Event-ID namespace and token balance | Caller |
| Intra-vector-pipe ordering (`PipeBarrier<PIPE_V>`) | Leaf |
| Operand alignment, bank skew, 256-byte read pad | Caller (layout) |
| Zero scalar compute | Leaf |

A leaf that issues a `HardEvent` forces its event ID onto every caller and
breaks composition. A leaf that issues a `PipeBarrier<PIPE_V>` does not.

---

## 4. Two-tier verification methodology

Neither tier subsumes the other. This cycle is the proof: Tier 1 passed
throughout, including while the kernel was computing on the wrong tile's
scores.

### Tier 1 — host discrete-event queue model (sub-millisecond)

`tests/cpu/test_pipeline_deadlock_probe.cpp` runs a `QueueModel` that schedules
MTE2/Vector/MTE3 as independent queues over one-bit event tokens. It proves
*structural* properties and nothing else:

- deadlock freedom (`require(progress, "deadlock: all three execution queues stalled")`),
- token-balance liveness (`require(!tokens[r][slot], "event token leaked after epilogue")`),
- no double-`SetFlag` on a live event bit,
- guard-band integrity of the mock buffers.

The per-routine host probes (`tests/cpu/test_*_probe.cpp`) add a bit-exact scalar
oracle over `include/sim_micro_testbed.hpp` guarded buffers. Latest run:

| Tier-1 probe | Result | Host time |
| --- | --- | ---: |
| pipeline deadlock | 0 deadlocks, 18/18 event pairs, 0 residual tokens | — |
| hadamard16 | 5 cases × 3 modes × 256 elements bit-identical | 326.247 µs |
| dequant4 | 4 rows × 2 types × 128 outputs bit-identical | 9.353 µs |
| cube matmul | 16×16 and 32×16 bit-identical FP32 | 338.341 µs |
| online softmax | 4 tiles × 16 channels, bit-identical recurrence | 12.377 µs |
| normalize/store | 12 FP16/BF16 cases bit-identical | 51.587 µs |

**Tier 1 cannot see:** scalar addressing traps, null `TPipe` dereferences,
the 256-byte unmasked read footprint, engine run-ahead, or any WAR hazard —
its mock engines are sequential, so MTE2 can never outrun Vector. Every
defect found in this cycle was invisible to Tier 1.

### Tier 2 — boundary cycle-accurate CAModel

`bash scripts/run_modulo3_camodel.sh micro <name>` builds the kernel with
`ccec` and runs it against the `Ascend950PR_9599` CAModel through the ACL
runner. Grid is reduced to the smallest inductive size that still exercises
every pipeline phase (4 tiles for softmax, 3 for the modulo3 skeleton), which
keeps a run at tens of seconds instead of hours.

The runner fails the build on any of
`[error]`, `[fatal]`, `invalid ldst addr`, `unrecognize ldst addr`, so a
scalar trap cannot pass silently even when the numeric result is correct —
which is exactly what happened here: the faulting baseline still printed
`PASS ... max absolute error 2.38418579e-07`.

**Tier 2 catches:** scalar addressing traps, packing-order mismatches,
engine run-ahead and execution races, real `Exp`/`Div` rounding, DMA
visibility, GM guard-band violations.

### What to check, in order

1. Device exit code 0 **and** zero `invalid ldst addr` / `su_ccu_mpu_err`
   lines. A numeric PASS alongside scalar faults means the faults are
   side-effect-free *today*, not that the kernel is correct.
2. GM canaries (`0xDEADBEEF`) intact on every buffer, head and tail.
3. Numeric agreement against the oracle, **per tile**, not only at the end.
   A whole-run check hides run-ahead whenever the fixture's tiles agree on
   the checked quantity.
4. `SET_FLAG`/`WAIT_FLAG` counts balanced per `(route, ID)` in
   `core0.veccore0.instr_popped_log.dump`.

### Instrumenting a Tier-2 numeric divergence

Dumping per-tile intermediates to spare GM lanes resolved §0 in one build
cycle, after static reasoning had failed on three hypotheses. Widen the
output buffer, store the state blocks you suspect behind `#ifdef`, print them
next to the oracle's per-tile values, and read off which tile diverges first.
It is far cheaper than bisecting a kernel at ~5 minutes per rebuild.

---

## 5. Numeric tolerance — what is actually achievable

`test_camodel_online_softmax` reports max absolute error and max ULP:

```
PASS native online softmax: 4 tiles, max absolute error 2.38418579e-07,
max 8.000 ULP (worst operand acc), 512/512 GM guard bytes intact
```

Per-operand, against a `std::exp` host oracle:

| Operand | Magnitude | Bound |
| --- | ---: | --- |
| `m` | 1.0 | exact by construction — `MAX` selects, it does not compute, and the fixture's scores are dyadic |
| `l` | 28.285 | ≤ 0.125 ULP (worst case if the whole 2⁻²² lands on `l`) |
| `acc[col]` | 0.085 … 1.32 | 8 ULP, measured — this is the reported worst operand |

Only the `acc` figure is a direct measurement; the `m` and `l` rows are
bounds, since the test reports one max-ULP figure rather than a per-operand
breakdown.

A **≤ 1 ULP** bound on `acc` is not attainable, and the reason is structural
rather than a defect:

- `Exp` is documented to permit up to 1 ULP of error per call
  (`basic_arithmetic/Exp.md`), and each `acc[col]` consumes 16 `Exp` results
  per tile across 4 tiles.
- `V` is signed over `[-11/32, +11/32]`, so `acc[col]` is a cancelling sum.
  At column 0 the largest intermediate is 0.907 against a final value of
  0.089 — a 10.2× cancellation that amplifies relative error by the same
  factor.
- The device reduces with a hardware tree while the oracle sums
  sequentially, so the two disagree in the last bits by construction.

The correct framing is absolute error against the *working* magnitude:
2⁻²² absolute on intermediates of order 1 is ≈ 0.5 ULP of the computation,
which is at the documented floor. This figure is **unchanged from the
pre-fix baseline** — the fixes altered scheduling and addressing, not
arithmetic.

**Specify ULP budgets against the largest intermediate, not the final
result, for any cancelling reduction.**

---

## 6. Integrating leaves into a MIX pipeline

Verified by building the fused paged-attention kernel
([`reports/fused_attention_integration.md`](reports/fused_attention_integration.md)).

### 6.1 The UB → L1 transfer exists; use the MTE3 burst form

**This section previously claimed that no UB → L1 path exists and that a
Vector-produced Cube operand must round-trip through a GM workspace. That was
wrong for CANN 9.2 / dav-3510, and the claim has been retracted.** It was
inferred from one incomplete read of the Tensor API copy tags; the correction
is traced below to the implementation headers and to a CAModel instruction
trace.

CANN 9.2 offers the transfer twice:

| API | Position pair | What it lowers to on dav-3510 |
| --- | --- | --- |
| `Te` `CopyUB2L1` | ND→ND, DN→DN, NZ→NZ, ZN→ZN only (`impl/tensor_api/arch/vector/ub_to_l1/routing.h`) | `CopyUbufToCbuf` |
| `DataCopy(LocalTensor<A1\|B1>, LocalTensor<VEC>, DataCopyParams)` | any, caller-strided | `DataCopyUB2L1Impl` → `CopyUbufToCbuf` |
| `DataCopy(..., Nd2NzParams)` | ND→NZ | `DataCopyUB2L1ND2NZImpl` → **needs a `TPipe`** |

`DataCopyUB2L1Impl`
(`impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h:619`) is declared
`__in_pipe__(MTE3) __out_pipe__(MTE3)`, and for any AIC:AIV ratio other than
1:1 its whole body is one `CopyUbufToCbuf`, that is the `copy_ubuf_to_cbuf`
hardware move in
`impl/c_api/instr_impl/npu_arch_3510/vector_datamove_impl/asc_copy_ub2l1_impl.h`.
No `TPipe`, no stack buffer, no Global Memory. So the path is

```
int4 GM -> UB -> dequant (Vector) -> UB -> L1(Zn) -> L0B -> Mmad
```

and the fused kernel's trace shows it as one `MOV_UB_TO_L1 Src:UB, Dst:L1`
MTE3 instruction per page.

Three constraints come with it, and all three have teeth.

1. **Declare the topology.** The direct body is guarded by
   `#if KFC_C310_SSBUF == 1 || __MIX_CORE_AIC_RATION__ != 1`. At a 1:1 ratio
   the identical call becomes a KFC relay that copies UB → GM and messages the
   AIC to copy GM → L1, silently reintroducing the workspace. Declare
   `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` and `#error` on
   `__MIX_CORE_AIC_RATION__ == 1` rather than trusting the default.
2. **Do not use the `Nd2NzParams` overload from a kernel with no `TPipe`.**
   `DataCopyUB2L1ND2NZImpl` calls `PopStackBuffer<T, TPosition::LCM>()` and
   `GetTPipePtr()->FetchEventID()`, so in this harness's hand-placed-tensor
   style it is the §0 null-`TPipe` fault, not a transfer. The arch-3510 variant
   that avoids all of it, `DataCopyUB2L1ND2NZImplV2`, is library-internal.
   Reimplement its burst schedule on the `DataCopyParams` overload instead:
   one `(nValue × 1 block)` strided burst per C0 column strip, with the strip
   count a template parameter so the emitter unrolls at compile time.
   `routines/routine_cv_transfer_ub_to_l1.hpp` is that leaf.
3. **The burst engine places blocks; it does not permute elements.** ND → NZ at
   32-byte granularity is a strided placement. A conversion that needs an
   element transpose — ND → Zn for a single FP16 fractal, for instance — is not
   expressible, because Zn is the within-fractal transpose of Nz. Fix that in
   the cache layout, not in the transfer: for one 16×16 half fractal, `Nz(X)`
   and `Zn(X^T)` are the same bytes, so packing the KV page key-major makes the
   dequantised UB buffer already the Zn image of `K^T`. A production paged-KV
   cache stores pages in the fractal order the Cube reads anyway.

A GM workspace is still the right answer when the conversion genuinely needs
the GM→L1 converter's element-level rearrangement, or on a 1:1 part. It is not
the only answer, and on 950PR in MIX_AIC_1_2 it is the slower one: removing it
from the fused kernel cut MTE3 occupancy from 3891 to 675 cycles and the
dispatch span from 8169 to 5381, with bit-identical results.

### 6.1b Reaching a chosen Vector core from the Cube

`Te::FixpipeParams{unitFlag, subBlockId}` selects which Vector core's UB a
`CopyL0C2UB` lands in, and `FixpipeParamsArch3510::dualDstCtl` can instead
split the M×N cube in half and write both UBs at once (M split needs even M,
N split needs N divisible by 32). `subBlockId` defaults to `false`, so a leaf
that does not expose it always writes sub block 0 — put the consumer of the
Fixpipe output on AIV0 and no leaf needs changing.

### 6.2 Cross-core edges are mode 4, and they need priming and draining too

`CrossCoreSetFlag<4, pipe>` / `CrossCoreWaitFlag<4, pipe>` lower to
`set_intra_block` / `wait_intra_block`
(`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:790`), which synchronise
the Cube core with the Vector cores inside one AI Core block, in **both**
directions. Modes 0..3 instead emit `ffts_cross_core_sync` with the mode
masked to two bits; mode 4 is the intra-block one and the one to use here.

The flag id space is **per sub block**: the library's own `GetIntraFlagId`
adds `subBlockId * INTRA_NUM` with `INTRA_NUM == 16`, and `SyncAllImpl` on a
non-1:1 ratio has the AIC wait both `SYNC_AIV_FLAG` and
`SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX`. So ids 0..15 are AIV0's channel and 16..31
are AIV1's, the Cube participates on both, and a "both Vector cores are ready"
barrier is two waits, one id from each half. Keep clear of the library's
reserved ids (`SYNC_AIC_AIV_FLAG` 13, `SYNC_AIV_ONLY_ALL` 14, and their +16
images).

This is load bearing in **both** directions, and §6.2d records the run that
showed it: renumbering AIV1's channel into 0..15 — on the mistaken reading
that the id is a four-bit field in one pool shared by the block — deadlocks the
kernel, with the Cube stuck on its own "panel free" wait and AIV1 stuck on its
"panel ready" wait. The `+ INTRA_NUM` offset is how the peer's sub block is
named, not a convention.

In the trace an AIV prints `sync_id` **relative to its own half** while the AIC
prints the absolute id, so AIV1's raw id 16 appears as `sync_id:0` on
`core0.veccore1` and `sync_id:16` on `core0.cubecore0`. Rebase before counting,
or every AIV1 flag will look unpaired.

The `pipe` template parameter on a wait is the pipe whose issuance blocks, so a
wait guarding Vector work takes `PIPE_V`, one guarding an L1→L0 load takes
`PIPE_MTE1`, and one guarding a Fixpipe takes `PIPE_FIX`.

### 6.2a A mode-4 set is a scalar instruction; the `pipe` argument does not order it

**This section previously claimed that "a set is ordered after the preceding
work on that same pipe, which is how 'the L1 slot is free once MTE1 has read
it' is expressed: `CrossCoreSetFlag<4, PIPE_MTE1>`". That is wrong, and the
claim is retracted.** It was inferred from the API's shape and it survived the
N = 3 fused probe because at the ring depth every producer is ahead of its
consumer anyway. The eight-page run disproved it in the most direct way
available: wrong numbers.

`CrossCoreSetFlag<4, PIPE_x>` lowers to `set_intra_block(PIPE_x, id)`, and in
the CAModel trace that instruction issues on the **FLOWCTRL unit**, which runs
the scalar stream:

```
[00001279] FLOWCTRL : SET_INTRA_BLOCK  PIPE:MTE1, sync_id:20   <- "L1 slot 0 free"
[00002862] MTE1     : LOAD_2Dv2  Src:L1, Dst:L0A, XN:X1=0
[00002863] MTE1     : LOAD_2Dv2  Src:L1, Dst:L0B, XN:X18=0x220 <- reads L1 slot 0
```

The set is 1583 cycles in front of the load it is supposed to follow. In the
fused kernel the Cube's scalar stream ran through six columns of flags in 130
cycles and released six L1-slot-free tokens by cycle 1402; the staging core
refilled L1 slot 0 with page 3 at cycle 2344; the Cube's first `Mmad` then
multiplied page 3's keys into page 0's score panel, and the probe caught it as
`panel 0 holds page 3's panel`.

The library agrees, in the only way library code can: `SyncAllImpl`
(`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:161`) puts a
`PipeBarrier<PIPE_ALL>` in front of its `set_intra_block` /
`wait_intra_block` handshake. It drains every pipe rather than trusting the
pipe argument.

So a cross-core set that is supposed to mean *"the work on pipe x has
finished"* needs the scalar stream drained behind pipe x first:

```cpp
compute_or_move_on_pipe_x();
SetFlag<HardEvent::x_S>(id);      // pipe x -> scalar
WaitFlag<HardEvent::x_S>(id);     // the scalar stream waits for pipe x
CrossCoreSetFlag<4, PIPE_x>(flag);
```

Two instructions per edge, against a whole-pipeline stall for
`PipeBarrier<PIPE_ALL>`. The available routes are `V_S`, `MTE2_S`, `MTE3_S`,
`FIX_S` and `M_S`; **there is no `MTE1_S`**, so an MTE1 drain has to ride a
later pipe that provably depends on it — in the fused kernel the Fixpipe, which
`unitFlag = 3` pairs to the `Mmad`, which waited `MTE1_M`.

Cross-core **waits** need none of this. `wait_intra_block` also executes on the
scalar stream, so everything the consumer issues after it is behind the flag by
construction. Only sets are affected, and only the ones whose meaning is "my
earlier pipe work is done" — a priming set, whose meaning is "this buffer
starts empty", has nothing to drain.

The marked-graph discipline from §3.1 applies across cores unchanged: a
buffer-reuse back edge, such as "the shared UB score panel is free again",
is primed once before the loop and drained once after it. In the trace these
appear as `SET_INTRA_BLOCK` / `WAIT_INTRA_BLOCK` with a `sync_id`, and a set
on one core must be matched by a wait on the *other* — so balance has to be
checked per sync id across both core dumps, not per core.

Note that `CrossCoreSetFlag` calls issued back to back by one core have **no
guaranteed order** between different flagIds. Separate them with a wait if the
order matters.

### 6.2c A recycled mode-4 flag id is a token — once the announcement is drained

**This section previously claimed that a mode-4 flag "is not a counting
semaphore you can cycle an id through", and that recycling an id was refuted on
this part. That diagnosis is wrong and is retracted. The measurements below are
unchanged; their cause was not the flag mechanism.** Mode-4 flags do implement
consuming-latch behaviour on Ascend 950PR and in the CANN 9.2 CAModel. What
produced the stale releases was §6.2a's hazard — a `CrossCoreSetFlag` issued on
the scalar unit, overtaking the pipeline work it reports, because the `pipe`
template argument does not order it. Recycling an id turns that reordering into
a *stale release*; single-shot allocation turns it into a harmless duplicate.
That is why it showed under recycling and why recycling looked like the cause.

So the rule is now a precondition rather than a prohibition:

> A cyclic two-slot `READY`/`FREE` ring over recycled mode-4 ids is valid for
> production pipelines, provided **every** announcement is preceded by its
> `SetFlag<HardEvent::x_S>` / `WaitFlag<HardEvent::x_S>` pair on the pipe whose
> work it reports, and provided the epilogue drains every queue before the
> kernel body returns (§6.5, rule 3). A priming set, whose meaning is "this
> buffer starts empty", reports no earlier work and needs no drain.

`scripts/audit_cross_core_fences.py` is the gate for that precondition over
both reshape-and-cache kernels and the probe below, with negative controls. It
exists because the property is invisible to every other check here: set and
wait counts balance without it, the id census passes, and the output can be
bit-exact while a panel is read a generation early.

#### What settled it

`tests/research/kernel_cluster_sync_probe.cpp` (targets `test_cluster_sync_probe`
on hardware and `test_cluster_sync_probe_sim` on CAModel) isolates the
question: one physical AI Core block, `MIX_AIC_1_2`, 1 Cube + 2 Vector cores, no
data at all, and a multi-iteration cyclic bilateral handshake over **recycled**
ids 0, 1, 16 and 17 — AIV *k* announces on `k*16`, the Cube waits both, drains
`MTE1_M` then `M_S`, and releases both on `k*16 + 1`; each Vector core drains
`V_S` before its own announcement. Both iterations executed cleanly on physical
silicon. CAModel completed cleanly too (176,061 ticks), with no early wait
retirement in the trace.

Read the trace, not the exit status. The probe moves no data, so a *sticky*
latch would retire every wait early and the run would still finish and still
print nothing wrong; what distinguishes the two semantics is the tick order of
`SET_INTRA_BLOCK` against `WAIT_INTRA_BLOCK` on one recycled id. The probe's
own output says so, and the binaries name their leg so a pasted log cannot be
mistaken for the other.

#### The measurements that were misread

With `kFlagProductReady = Owner * INTRA_NUM + Slot` — two slots, so each id
carries *two* announcements over the run — AIV1's second wait on its slot-1 id
did not block.

| event | CAModel cycle |
| --- | ---: |
| Cube's chunk-3 `SET_INTRA_BLOCK sync_id:17` (panel ready, slot 1) | 6840 |
| AIV1 consumes it, reads slot 1 for chunk 3 | 7282 |
| AIV1 reaches its chunk-7 wait on the same id | 8988 |
| AIV1 starts reading slot 1 for chunk 7 | 10502 |
| Cube's chunk-7 Fixpipe **completes** | 11165 |
| Cube's chunk-7 `SET_INTRA_BLOCK sync_id:17` | 11167 |

The chunk-7 wait was released 2179 cycles before the announcement it was
waiting for, and the resulting page was 87% identical to the page the slot had
held for chunk 3. `safe_pipeline.md` records the same shape with a 1032-cycle
margin on READY id 16 of the synthesised ring.

At the time **the mechanism was not isolated**, and the reading taken — that
the flag is a level bit a wait tests without clearing — was the conservative
one. It was also wrong: the probe above exercises exactly the recycled
handshake those runs were doing, with drains added, and finds no early
retirement on silicon or in simulation. What the revisions above lacked was a
drain on the *producer's* pipe before the announcement, so the Cube's "panel
ready" for chunk 7 was visible on the scalar stream before its Fixpipe had
written the panel. With one id per slot, the wait that observed it was the
chunk-7 wait, and it read the panel early.

The first fix adopted was to **label every cross-core event uniquely** — one
flag id per announcement and one per release, each set and waited exactly once.
It works, and it remains available as the regression fallback, but it is not
the production policy: eleven ids per sub block saturates the legal 0..10 at an
eight-chunk schedule, so a ninth chunk or a second temporal segment does not
fit. The recycled ring costs five ids at any schedule length, and §6.5's rule 2
is now written around it.

**Why it had never shown.** Set and wait counts balance under a reused id, so
no token-leak audit and no per-id pairing check fails. And a pipeline in which
one engine runs far ahead never consumes a stale token: the factored revision
of that operator had its Cube finish all eight chunks by cycle 6121 of a
19331-cycle run, so every panel was ready long before a Vector core looked at
it. **Balancing the pipeline is what exposed it** — the kernel had been passing
twice, reproducibly, with every gate green. Three consequences:

* a cross-core edge audit should check **tick order**, not just balance: that
  every wait was released by an announcement already issued, and that no id
  ever carried two outstanding ones. `scripts/analyze_reshape_trace.py` does
  this per id across all three core dumps, and it is the check that caught
  both stale releases; a per-id count check passes vacuously;
* the fence in front of every announcement is a correctness precondition of
  the recycled policy, so it gets its own audit
  (`scripts/audit_cross_core_fences.py`) rather than a comment;
* `kernels/ops/turboquant/turboquant_paged_attention_opt.cpp` cycles three
  slots through each of five ids over eight pages, so each of its cross-core
  ids carries up to three announcements. Under the corrected diagnosis that is
  legitimate — but only if each of its announcements is drained first, which
  has not been audited. Run `scripts/audit_cross_core_fences.py` over it before
  relying on that.

### 6.2d The id space really is per sub block

The run that produced §6.2c's table was first misdiagnosed as the two Vector
cores sharing one four-bit flag pool, on the strength of
`NotifyEventImpl` / `WaitEventImpl`
(`impl/basic_api/dav_3510/kernel_operator_sync_impl.h:789`, `:816`) passing the
id straight to `set_intra_block` / `wait_intra_block` with no sub-block offset
of their own, plus `GetffstMsg`'s `flagId & 0xf` on the sibling ffts path and
the AIV trace printing an id of 16 as `sync_id:0`. **That reading is wrong**,
and renumbering the two cores into 0..7 to "fix" it deadlocked the kernel:
AIV1 stalled on `WAIT_INTRA_BLOCK sync_id:2` (its "panel ready") and the Cube
on `WAIT_INTRA_BLOCK sync_id:6` (its "panel free" from AIV1), each waiting on a
flag the peer had addressed into the other sub block. §6.2's `+ INTRA_NUM` is
how a participant names the *other* sub block's channel, in both directions,
and the AIV-side trace value is that id with the sub-block bits already
resolved. Recorded because the wrong reading survived a library read and a
trace read, and only a deadlock settled it.

### 6.3 Establish numerics before concurrency

Build the integration as a strict ping-pong first, prove the numbers, then add
look-ahead and double buffering and re-prove the numbers. The fused kernel
went 11509 → 9203 ticks this way with bit-identical results. The reverse order
cannot distinguish an integration bug from a pipelining bug, and §0 is the
standing reminder that adding or removing synchronization can change
correctness, not just speed.

### 6.4 Make the pages distinguishable, or the hazard checks are theatre

An integration fixture must make its tiles **numerically distinct**:
per-tile reduction maxima strictly increasing (so the rescale path runs every
tile), per-tile sums distinct, and every output channel distinct. A fixture
whose pages are interchangeable cannot detect a run-ahead, double-buffering or
wrong-slot fault, because the wrong operand produces the right answer. This is
not hypothetical: the online-softmax V→MTE2 WAR hazard in §0 survived its
original validation precisely because every tile shared the same maximum.

Where the arithmetic permits, choose operands that make the exact result
representable — the fused fixture keeps every score a multiple of 1/128 so the
whole Cube output panel is checked **bit-exactly** rather than against a
tolerance.

**A modular page term does not survive a longer run.** The three-page fused
fixture varied the page through `(d + 1) * 14 * t mod 16`, which gives maxima
0, 1/16, 2/16 — strictly increasing, exactly as required. Extended to eight
pages the same expression folds: 0, 1/16, 2/16, 7/16, 0, 2/16, 0, 0. Four of
the eight pages never exceed the running maximum, `alpha` is exactly 1 on
those pages and the rescale path stops being exercised at all. Modular
arithmetic is the natural way to generate distinct tiles and the wrong way to
generate monotone ones.

What does work is a non-wrapping slope: a page-invariant base, reduced so that
it leaves `tiles - 1` of headroom in the quantised range, plus `t` times a
per-element slope chosen from the reduction's own weights. Give the slope a
guaranteed positive floor so the maximum has to rise, and let the rest of it
vary per element so the tile's *shape* changes with the page and not just its
height — a uniform shift leaves every `p_sum` identical, which fails the
distinct-sums property even though the maxima look perfect.

**Assert the properties, do not describe them.** Every one of them is a few
lines over the oracle the runner already computes: nibbles in range, maxima
strictly increasing, `alpha < 1` after the first tile, `p_sum` pairwise
distinct, output channels pairwise distinct, every reference value exactly
representable. A fixture property stated only in a comment is one edit away
from being false, and when it goes false the probe does not fail — it silently
stops testing what it was built to test.

### 6.5 The Mode-4 inter-wave drain protocol

A `KERNEL_TYPE_MIX_AIC_1_2` launch dispatches its grid across physical AI Core
blocks of one Cube and two Vector cores. Above the physical block count the
scheduler runs the grid in **waves**: with 32 blocks of silicon, grid blocks
0..31 are wave 0 and 32..63 wave 1, and each physical block executes its wave-1
block after its wave-0 one. Two things survive that boundary:

* the **mode-4 intra-block flag registers**, which are local to the physical
  block and are not cleared by the hardware scheduler between waves;
* the block's **engine queues**. An MTE2 burst, a Fixpipe transfer or an MTE3
  store still in flight when the wave-0 block returns is still in flight when
  the wave-1 block starts — and every address a kernel like this uses is a
  compile-time constant, so wave 1 allocates the *same* L1 ring slot, the same
  landing panel and the same packed egress slot the in-flight transfer is
  writing.

Both reshape-and-cache instances now carry the protocol below, and both
dispatch monolithic multi-wave grids as a result. The host-side consequence
came first and is worth stating plainly: **a launch window is not a fix, it is
a way of not testing this.** Driving 64 strips as two 32-strip launches re-enters
the kernel at the boundary, which re-arms everything; the bug the window hides
only appears when one dispatch spans two waves. Every `--window` parameter,
cyclic slice and staggered sub-launch in the reshape harnesses was removed for
that reason, and the validation ladder is now 1, 32, 33, 48, 64, 128 strips,
each one dispatch.

#### Rule 1 — boundary and guard isolation

Every core role computes the true task boundary from the same two host scalars:

```cpp
const uint32_t total_strips = numTokens / tokensPerCore;   // TotalStrips()
if (block >= total_strips) return;                         // before anything
AscendC::InitSocState();
```

The return is taken **before `InitSocState`, before any tensor or SOC state
exists, before a feedback cycle is primed and before any Global Memory address
is formed.** The failure it prevents is not a wrong answer: a block past the
end has no `slotMapping` entry, and reading `slotMapping[block]` there is an
invalid load that kills that AIV, after which the Cube sits on its next
`CrossCoreWaitFlag` for the rest of the launch. All three roles derive `block`
from the same two hardware registers, so they reach the same decision, and a
guarded group announces nothing its peers would wait for. Divide-by-zero is
part of the guard, not an assumption: `TotalStrips` returns 0 for a zero
`tokensPerCore`, which guards every block rather than faulting the scalar unit.

#### Rule 2 — deterministic intra-block channel allocation

Single-shot id allocation — one id per announcement and one per release — does
not scale: eleven ids per sub block covers an eight-chunk schedule and the
legal space is 0..10, so a ninth chunk or a second temporal segment collides
with the library's reserved 11..14. The answer is a bounded bilateral ring:
two READY and two FREE channels per sub block, each alternating strictly
`SET, WAIT, SET, WAIT`, the alternation enforced by the paired channel rather
than asserted. **Five ids per sub block, whatever the schedule length, and that
is the shipped default.**

It was synthesised and exhaustively checked 1-safe, deadlock-free and
co-reachable over 41,844,109 modulo states
(`kernels/ops/turboquant/safe_pipeline_proof.json`) *under an explicit
consuming one-bit latch semantics*, and that premise is now measured rather
than assumed: §6.2c's probe ran the cyclic handshake over recycled ids on
physical silicon and on CAModel with no early wait retirement.

**The precondition is the fence, and it is not optional.** Every announcement
must be behind its `SetFlag`/`WaitFlag<HardEvent::x_S>` pair on the pipe whose
work it reports (§6.2a), because a mode-4 set issues on the scalar unit and the
`pipe` argument does not order it. Two earlier revisions of this operator
recycled ids *without* that drain and measured a second wait retiring 1032 and
2179 cycles before its announcement; §6.2c has the tables and the corrected
diagnosis. `scripts/audit_cross_core_fences.py` is the gate, and a priming set
is the only exemption.

The budget is a compile-time object rather than a comment in both policies:

```cpp
template <std::uint32_t Chunks, std::uint32_t LandDepth>
struct RecycledChannels {
    static constexpr std::uint32_t kMaxLocalId = 2u * LandDepth;
    static_assert(kMaxLocalId <= kMaxSafeLocalId, "...");
};
```

A schedule that outgrows the legal space fails to compile instead of silently
renumbering a flag into id 13.

Both policies are compiled, selected by a non-type template parameter, and
both are dispatchable:

```cpp
template <bool Reuse> using Flags =
    turboquant_wave::ChannelPolicy<Reuse, kChunks, kLandDepth>;

extern "C" __global__ void op(...)                     { KernelBody<true>(...);  }
extern "C" __global__ void op_mode4_single_shot(...)   { KernelBody<false>(...); }
```

They differ in **nothing but the id allocation** — same schedule, same wait
placement, same priming, same drain — so a difference in behaviour between the
two entry points isolates recycling and nothing else. The harnesses expose this
as `--mode4-single-shot` (and `-DTQ_TEST_MODE4_SINGLE_SHOT=1` for a whole CI
job). Keep the fallback: it is the bisection control for any future
cross-core fault, and the one policy under which no reading of the flag
mechanism can make a second wait pass early. **Read the trace either way.**
Both revisions that were wrong produced bit-exact caches and intact guards; a
green run has never been the gate here.

#### Rule 3 — the mandatory inter-wave epilogue drain

Draining the marked graph's credits retires this block's *event bits*. It does
not retire its *transfers*. Before leaving the kernel body — or before
advancing to a next temporal or page segment — every role moves every
asynchronous queue it owns into the scalar stream, and the Cube then tells both
Vector cores that it has:

```cpp
// Cube, after draining every ring credit and every panel release:
SetFlag<HardEvent::MTE2_S>(0); WaitFlag<HardEvent::MTE2_S>(0);
SetFlag<HardEvent::M_S>(0);    WaitFlag<HardEvent::M_S>(0);
SetFlag<HardEvent::FIX_S>(0);  WaitFlag<HardEvent::FIX_S>(0);
CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrained<0>);
CrossCoreSetFlag<4, PIPE_FIX>(kFlagPageDrained<1>);

// AIV0:                                   // AIV1:
CrossCoreWaitFlag<4, PIPE_V>(drained0);    CrossCoreWaitFlag<4, PIPE_MTE3>(drained1);
V_S, then MTE2_S, then MTE3_S              MTE3_S, then V_S, then MTE2_S
```

Four points, each of which cost something to learn:

* **The Vector cores await the Cube's drain first.** The Cube is the only core
  that writes their landing rings, so until it reports every queue retired, a
  Fixpipe of *this* wave can still be in flight against a panel the next wave
  is about to allocate at the same UB address.
* **MTE1 has no scalar route.** §6.2a: the available routes are `V_S`,
  `MTE2_S`, `MTE3_S`, `FIX_S` and `M_S`, and there is no `MTE1_S`. MTE1 is
  covered transitively by the `M_MTE1` wait, which sits behind the Mmads that
  consumed its loads.
* **Drain an idle queue anyway.** Neither Vector core issues MTE2 work in these
  kernels. The pair is issued regardless, because "this queue is empty" is
  exactly what the wave boundary needs established, and an empty queue retires
  it immediately. Two instructions, against a whole-pipeline stall for
  `PipeBarrier<PIPE_ALL>`.
* **The two Vector cores drain in different orders**, each leading with the
  pipe it is most likely to still have work on — AIV0 the Vector queue it has
  just finished casting on, AIV1 the MTE3 queue carrying its last cache bursts.

#### What the model establishes, and what it does not

`tests/cpu/test_reshape_wave_model.py` is the capacity-one token/queue model of
this protocol: three cores, in-order engine queues, a scalar stream that
dispatches into them in program order and stalls on cross-core waits, HardEvent
places of capacity one, mode-4 places shared between the Cube and one Vector
core, and generation ownership on every L1 slot, landing panel, packed slot,
L0 page region and L0C. It runs **two** grid blocks on the same physical block,
carrying the marking across, which is what a multi-wave grid does.

Two hardware properties were unestablished when the model was written, and it
treats both as knobs rather than assumptions: whether `wait_intra_block`
**consumes** the flag it tested, and whether kernel entry **clears** the
block's mode-4 registers. The four combinations do not agree, which is the
useful part — the model says what each world costs without picking one:

| wait consumes | entry clears | single-shot | recycled |
| --- | --- | --- | --- |
| yes | yes | safe | safe |
| yes | no | safe | safe |
| no (sticky) | yes | safe | fails in wave 0 |
| no (sticky) | no | fails in wave 1 | fails in wave 0 |

**§6.2c's probe puts the hardware in the top half of that table**: mode-4 waits
consume, on silicon and in the CANN 9.2 CAModel, provided the announcement is
drained first. Rows 1 and 2 are the live ones and both policies are safe there,
which is why the recycled default is sound and why the fallback stays
dispatchable rather than being deleted. The bottom half is kept because it is
what the two stale releases *looked* like, and because it says what to expect
if a toolkit revision ever reintroduces them: the model reproduces both
failures as buffer-ownership violations rather than as assertions about flags —
the recycled policy's second wait on a reused id passes early and the Vector
core casts a landing panel the Fixpipe has not written yet, which is the shape
of both traces. That is the signature to look for, and
`scripts/analyze_reshape_trace.py` is what looks for it.

The model is exhaustive on bounded instances (two chunks over a depth-2 L1
ring: 57,348 reachable states per wave) and randomised on the four- and
eight-chunk schedules, at ingress lookaheads 1 and 3. Three negative controls
must fail and do: unprimed landing credits deadlock, a missing epilogue drain
leaks mode-4 flags past the block, and a doubled announcement is rejected. It
models synchronisation and buffer lifetime only — not numerics, alignment, bus
burst splitting or lowering — and **a pass is not a CAModel result**.

#### Checklist for a MIX kernel that may span waves

- [ ] Every role returns on `block >= numTokens / tokensPerCore`, before
      `InitSocState` and before any GM address is formed.
- [ ] The strip-count divide happens once; the guard and any sampling
      predicate share it.
- [ ] Every cross-core id is `static_assert`ed inside 0..10 per sub block, by
      a budget expression and not by inspection.
- [ ] Every announcement is behind its `x_S` drain on the pipe whose work it
      reports; only priming sets are exempt. Audited, not inspected.
- [ ] The trace audit checks tick order per id — every wait released by an
      announcement already issued, never two outstanding — not just balance.
- [ ] Every role drains every queue it owns into the scalar stream before
      returning, and the Vector cores wait for the Cube's drain first.
- [ ] The host drives multi-wave grids as ONE dispatch. No window, no cyclic
      slice, no staggered sub-launch.
- [ ] The ladder includes a rung with a *ragged* second wave (33, 48), not
      only full ones (32, 64).

## 7. Checklist for a new micro-routine

- [ ] No `GetValue`/`SetValue`/UB subscript; no `ReduceSum` on FP32/FP16.
- [ ] Broadcasts via `Brcb` from a replicated block, or `Duplicate` with a
      256-byte read pad asserted.
- [ ] Logically scalar state carried as a lane-replicated 32-byte DataBlock.
- [ ] Every operand base derived through `NextSlot` and `static_assert`ed
      32-byte aligned; 32-byte bank skew between slots.
- [ ] `static_assert(kArenaEnd + 256 <= kUbWindow)`.
- [ ] Forward **and** back event edges present; tokens primed before the
      loop and drained after it; sets == waits per `(route, ID)`.
- [ ] Every back edge on the route whose **consumer queue** is the one that
      must stall, and no queue left with nothing that can stall it (§3.1).
- [ ] `PipeBarrier<PIPE_V>` only for intra-vector ordering; never as a
      substitute for a cross-engine edge.
- [ ] Tier 1 green, then Tier 2 green with **zero** `invalid ldst addr`.
- [ ] Per-tile numeric check, not just an end-of-run check.
- [ ] ULP budget stated against the largest intermediate.

And for a `Reg::` register-resident leaf (§1.5):

- [ ] Working set resident across the whole cascade, with the **count** to
      prove it: no more register loads and stores than the data arriving and
      leaving (Rule R1).
- [ ] Per-stage constants built in-register from `Arange` and bit arithmetic,
      not read from a Unified Buffer table (Rule R2).
- [ ] Cross-lane reductions as a butterfly all-reduce over the index
      registers the leaf already carries, not `Reduce` plus a broadcast
      (Rule R3).
- [ ] Index registers unsigned and of the data's width; `ShiftLefts` counts
      cast to `int16_t`; no `Cast` to an unsigned b32 type expected to exist.

And when integrating leaves into a pipeline (§6):

- [ ] Vector-produced Cube operands staged UB → L1 with the MTE3 burst form,
      one ring slot per pipeline stage; `KERNEL_TYPE_MIX_AIC_1_2` declared and
      `__MIX_CORE_AIC_RATION__ == 1` rejected at compile time (§6.1). A GM
      workspace only where the conversion needs the GM→L1 converter.
- [ ] Both Vector cores given work, with per-sub-block mode-4 flag ids
      (0..15 and 16..31) and a Cube-side wait from each core where the Cube
      must not outrun either (§6.2).
- [ ] Cross-core back edges primed and drained, and balance checked per
      `sync_id` **across all three** core dumps, rebasing the AIV1 ids by
      `INTRA_NUM` because an AIV reports `sync_id` inside its own half.
- [ ] Every cross-core set that means "my pipe work is done" preceded by a
      `HardEvent::x_S` drain of that pipe, because a mode-4 set runs on the
      scalar stream and the `pipe` argument does not order it (§6.2a).
- [ ] Probe run **past** the ring depth, at least `2 * kStageDepth` tiles, so
      every reuse back edge is contended; the reported figure is the per-column
      steady-state interval, not the whole-run span (§3.1).
- [ ] Ping-pong proven numerically before look-ahead or double buffering is
      added, then re-proven.
- [ ] Fixture tiles numerically distinct: increasing maxima, distinct sums,
      distinct output channels.
- [ ] Engine overlap demonstrated from the trace, not inferred from source
      order.
