# Micro-routine design guidelines — Ascend 950PR / dav-3510 / CANN 9.2

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

`tests/test_pipeline_deadlock_probe.cpp` runs a `QueueModel` that schedules
MTE2/Vector/MTE3 as independent queues over one-bit event tokens. It proves
*structural* properties and nothing else:

- deadlock freedom (`require(progress, "deadlock: all three execution queues stalled")`),
- token-balance liveness (`require(!tokens[r][slot], "event token leaked after epilogue")`),
- no double-`SetFlag` on a live event bit,
- guard-band integrity of the mock buffers.

The per-routine host probes (`tests/test_*_probe.cpp`) add a bit-exact scalar
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

## 6. Checklist for a new micro-routine

- [ ] No `GetValue`/`SetValue`/UB subscript; no `ReduceSum` on FP32/FP16.
- [ ] Broadcasts via `Brcb` from a replicated block, or `Duplicate` with a
      256-byte read pad asserted.
- [ ] Logically scalar state carried as a lane-replicated 32-byte DataBlock.
- [ ] Every operand base derived through `NextSlot` and `static_assert`ed
      32-byte aligned; 32-byte bank skew between slots.
- [ ] `static_assert(kArenaEnd + 256 <= kUbWindow)`.
- [ ] Forward **and** back event edges present; tokens primed before the
      loop and drained after it; sets == waits per `(route, ID)`.
- [ ] `PipeBarrier<PIPE_V>` only for intra-vector ordering; never as a
      substitute for a cross-engine edge.
- [ ] Tier 1 green, then Tier 2 green with **zero** `invalid ldst addr`.
- [ ] Per-tile numeric check, not just an end-of-run check.
- [ ] ULP budget stated against the largest intermediate.
