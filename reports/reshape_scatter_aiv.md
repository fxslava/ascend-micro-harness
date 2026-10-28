# `turboquant_reshape_scatter_aiv` — the AIV-only sub-strip KV ingress

Ascend 950PR / arch35 / `dav-3510` / CANN 9.2.0-beta.2, `Ascend950PR_9599`
CAModel. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`, no Cube, no L1,
`gridDim = 1` for every run below.

**No NPU on this host.** Every number here is a cycle-accurate model tick for
one Vector core. None of it is a latency, throughput or power figure for the
part, and no run has ever had two blocks of this kernel computing at once.

| Field | Value |
| --- | --- |
| What it proves | the eight-stage `Pi . x` ingress, the power-of-two quantiser with its `absmax == 0` guard, the token-major key burst and the strip-blocked channel-major value read-modify-write are bit-exact and touch nothing they do not own |
| What it deliberately omits | a grid wider than one block, concurrent heads, `blockSize > 16`, non-dyadic activations, and any accuracy claim about real activations |
| Kernel | [`kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp`](../kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp) |
| Leaf | [`routines/routine_vector_rotate_butterfly.hpp`](../routines/routine_vector_rotate_butterfly.hpp), `RotateImpl::RegGather` |
| Runner | [`tests/camodel/test_camodel_reshape_scatter_aiv.cpp`](../tests/camodel/test_camodel_reshape_scatter_aiv.cpp) |
| Oracle | [`tests/common/turboquant_scatter_oracle.hpp`](../tests/common/turboquant_scatter_oracle.hpp) |
| Tier-1 probes | [`test_rotate_butterfly_probe.cpp`](../tests/cpu/test_rotate_butterfly_probe.cpp), [`test_reshape_scatter_probe.cpp`](../tests/cpu/test_reshape_scatter_probe.cpp) |
| Trace audit | [`scripts/analyze_scatter_trace.py`](../scripts/analyze_scatter_trace.py) |

```sh
bash scripts/run_reshape_scatter_camodel.sh            # the spatial sweep, H = 0, 2, 3
bash scripts/run_reshape_scatter_camodel.sh ladder     # every rung
bash scripts/run_reshape_scatter_camodel.sh --mode sequential --head 2
```

---

## 1. What the operator is for, and why it has no Cube

Every decode step writes `B x (1 + k)` tokens into `B` **different** pages at
arbitrary non-consecutive slots, and every request's prefill tail writes
`n < 16`. The aligned prefill writer cannot serve either: its contract is
`num_tokens % 16 == 0` with sixteen consecutive slots, and a decode step has no
slot run to offer.

The unit of work is **(request, kv_head)**, not (token, kv_head). All `1 + k`
tokens of one request share a strip in the MTP case, so folding them into one
work item makes the value-plane read-modify-write a single transaction and
removes the "no two work items collide on a strip" assertion the per-token form
would need and that MTP would violate. One KV head of `d = 256` is owned
exclusively by one physical AIV; there is no intra-head splitting, because the
head is the butterfly's operand and the butterfly does not decompose across
cores.

This is the one place the family's Cube-centric verdict inverts. At `M = 1` the
Cube prologue would stage 32 KiB of rotation tables to transform 512 bytes of
payload, and `GEMV.md` records that an `M = 1` Mmad leaves only 32–64 bytes
valid in each 1024-byte L0C sub-vector — a 16:1 waste.

### 1.1 Which butterfly, and how that was decided

The leaf carries three bit-identical implementations of the eight-stage cascade
and the production kernel selects one by measurement, not preference
([`rotate_butterfly.md`](rotate_butterfly.md)): over 16 rows of 256 FP32 on one
AIV,

| variant | vector instr | `RVECEX` | `RV_VLD` | `RV_VST` | vector span |
| --- | --- | --- | --- | --- | --- |
| Blocked — UB operands, `BinaryRepeatParams` + UB `Gather` tables | 11250 | 2512 | 3072 | 1728 | 30215 |
| RegShuffle — register-resident, `DeInterleave`/`Select`, no tables | 1981 | 1834 | 64 | 64 | 1965 |
| **RegGather** — register-resident, `Reg::Gather` + `MulAddDst` | **1203** | 1057 | 64 | 64 | **1850** |

Both register-resident variants issue exactly four register loads and four
register stores per row — the row arriving and leaving, and not one register
more. Approach A spills 1,664 extra registers over the same work and runs 16.3x
longer on the Vector engine.

---

## 2. The layout contract, and why the two planes are written differently

```
key   nibble = ((block*H + head)*S + token)*d + channel
value nibble = ((block*H + head)*S)*d + (token/R)*(R*d) + channel*R + (token%R)
scale index  = ((kv*P + block)*H + head)*S + token          kv 0 key, 1 value
nibbles       two per byte, low nibble first, signed INT4 in [-8, 7]
```

with `R = 16` the strip height. Neither order is a choice — RFC §3.1a records
why both are hardware mandates — and the consequence for this kernel is that the
planes share no code path at all:

**KEY needs no read-modify-write.** Token-major puts one token's 256 nibbles at
`token * d/2 = token * 128` bytes: 128 **contiguous** bytes at a 32-byte aligned
offset, so one token is four aligned 32-byte bursts and a run of consecutive
tokens is one burst of `n * 128`.

**VALUE is a read-modify-write of one 2048-byte strip.** Channel-major inside
the strip puts one token's 256 nibbles at stride 16 — 128 bytes spread over 2048
at half-byte granularity — and there is **no aligned-burst expression of that**.
So the strip arrives in one MTE2 burst of 64 DataBlocks, the owned nibbles are
spliced in by register bitwise masks, and the whole strip goes back in one MTE3
burst.

**The read-modify-write always preserves.** Lanes the work item does not own go
back unchanged. That is not defensive: in decode a strip accumulates over 16
successive steps, so zeroing the unowned lanes would destroy the previous
fifteen tokens. It also makes the canary protocol *stronger*, because an
untouched lane must still read back as canary.

### 2.1 The merge arithmetic, settled on the host first

Inside the strip a channel owns 16 consecutive nibbles, one per token, so a
channel is 8 bytes and the strip is 256 of them. As uint32 words, word `2c`
holds channel `c`'s tokens 0..7 and word `2c + 1` its tokens 8..15, with token
`t` at bit `4 * (t % 8)`. One token therefore owns

```
words  { 2c + t/8 : c = 0..255 }      -- every other word
bits   [4*(t%8) + 3 : 4*(t%8)]        -- the same nibble in every one
```

`Reg::Interleave(e0, e1, codes, codes)` duplicates each code into **both** words
of its channel and the nibble mask then deletes the copy in the word this token
does not own, so there is no second interleave and no lane-parity constant.
`t / 8` is a template parameter, which keeps the only branch outside the VF body.

The Tier-1 probe checks that derivation against the nibble map for all
`16 x 256` positions, and then **simulates the whole masked merge at word
granularity** and compares it against the nibble-level reference over five
tokens spanning both halves — including that every nibble of every unwritten
token is preserved. That is the one check in this cycle that could have caught
an endianness or half-selection error before a ten-minute CAModel run.

---

## 3. The quantiser, and the guard the integer fixtures cannot reach

For `2^e <= absmax < 2^(e+1)` the scale is `2^(e-2)` and the code is
`min(rint(Y * 2^(2-e)), 7)`, clamped from above only. The power of two is
extracted by integer arithmetic on the FP32 bit pattern, because a float `2^k`
carries exponent field `k + 127`:

```
exp_bits = bits(absmax) & 0x7F800000      -> 2^e,     field e + 127
pow_bits = exp_bits - (2 << 23)           -> 2^(e-2), field e + 125
inv_bits = (254 << 23) - pow_bits         -> 2^(2-e), field 129 - e
```

The last line is the exact reciprocal of a power of two: the two fields sum to
254. No division, no logarithm and no hardware reciprocal enters the contract,
which is what lets the host oracle reproduce the device in integer arithmetic
and the gate be bit equality rather than a tolerance. The Tier-1 probe checks it
against `ldexp` for every exponent from −20 to 20.

`invSqrtLen` folds into the **stored scale** and never into the code: the
quantised panel is scale-invariant, because scaling `Y` and `absmax` together
leaves every code unchanged.

**`absmax == 0` takes an explicit branch** (`scale = 1.0f`, codes 0). Without it
`exp_bits` is 0, `pow_bits` borrows an exponent, and the stored scale is
`-1.70141e+38` — the probe prints that number rather than asserting the
description of it. The **codes** are unaffected either way, because every `y` is
zero and `0 * inv` is zero for any finite reciprocal, which is exactly why the
fault is silent: a zero K or V row never occurs with integer fixtures and occurs
immediately in production. The guard is a `Reg::Compare` against zero plus a
`Reg::Select`, both on the broadcast absmax register, so it costs two
instructions per row and no branch.

### 3.1 The absmax, as a butterfly all-reduce

The row maximum is not a `Reduce` plus a broadcast. Four `Abs`, three `Max`, and
then six rounds of (gather the XOR partner, take the max) over the index
registers the rotation already built leave the maximum in **every** lane — which
is the form the quantise multiply wants anyway. Fifteen instructions, no
reduction API, no broadcast step and no new constant.

---

## 4. Synchronisation: five intra-core routes and nothing else

No Cube, no L1, no cross-core flag, no mode-4 id, no modulo ring. Straight line
per work item over one buffer set, with the WAR back edges a single buffer set
needs:

| Route | What it bounds |
| --- | --- |
| `MTE2_V` id 0 | the raw activation rows are in Unified Buffer |
| `MTE2_V` id 1 | the 2048-byte strip is in Unified Buffer |
| `V_MTE2` id 0 | Vector finished reading the raw rows → MTE2 may refill |
| `V_MTE3` id 0/1 | the packed key page / the merged strip is ready |
| `MTE3_V` id 0 | MTE3 drained the packed page and the scale slots |
| `MTE3_MTE2` id 0 | MTE3 drained the strip → MTE2 may read the next one |

`MTE3_MTE2` is the one that is easy to get wrong. The strip is written by MTE2,
mutated by Vector and read by MTE3, so the reuse edge runs from MTE3 **straight
back to MTE2**; routing it through Vector would not bound the MTE2 queue at all,
and MTE2 is the queue that runs furthest ahead.

### 4.1 The Unified Buffer budget

Every base derived through `NextSlot` and `static_assert`ed 32-byte aligned,
one DataBlock of bank skew between live slots, 256 bytes of read pad behind the
arena (Rule A3):

| Slot | Bytes |
| --- | --- |
| BF16 activation landing, 16 x 256 | 8,192 |
| FP32 activations | 16,384 |
| FP32 codes, post-clamp | 16,384 |
| INT32 codes, the single rounding | 16,384 |
| FP16 staging for the INT4 cast | 8,192 |
| packed key nibbles, 16 x 128 | 2,048 |
| the value strip | 2,048 |
| `pi_signs` | 1,024 |
| scale slots, one vector register per row | 4,096 |
| the launch probe | 128 |
| **total, with skews** | **~74 KiB of 192 KiB** |

Nowhere near the budget, which is the point of an AIV-only kernel.

---

## 5. The single-head canary protocol

Every Global Memory payload — both cache planes, the scale plane, the launch
probe — is flooded with `0x7FC0BEEF` before the launch. It is a quiet NaN read
as FP32, so a stray scale read propagates rather than perturbs, and it is not
`0x00`, which a half-issued burst also leaves behind.
`camodel::GuardedDeviceBuffer` additionally surrounds each payload with 64 bytes
of `0xDEADBEEF`: **`0xDEADBEEF` catches a burst that runs off the end of a
buffer, `0x7FC0BEEF` catches one that lands in the wrong place inside one.**

Request 0 lands on page 3 — an **odd** page in the middle of an eight-page pool,
so every other page must still hold its poison afterwards and a kernel that
derived its page from the block index rather than from `slotMapping` fails.

**The expected image is the whole cache.** The oracle is handed the pre-launch
cache images and returns them with the owned lanes replaced, so the comparison
is byte for byte over every byte of every plane. That one comparison is
simultaneously the correctness gate, the read-modify-write preservation gate,
the unassigned-head gate and the unassigned-page gate: a lane the kernel should
not have touched is *expected* to still read `0x7FC0BEEF`.

Because a byte comparison alone reports a wrong page and a page packed the wrong
way round identically, the mismatches are then classified: owned nibbles that
disagree with the reference, unowned nibbles **inside** a written strip that
moved (the read-modify-write failed to preserve), and nibbles **outside** every
written strip that moved (a burst landed in the wrong place).

### 5.1 The rungs, and what each one can see

| Rung | What only it can see |
| --- | --- |
| `--mode canary --head {0,2,3}` | that one isolated head writes its own slice and nothing else; the boundary heads catch an off-by-one in the head stride that a middle head would not |
| `--mode sequential` | **the read-modify-write across launches.** Three launches into one strip — token 0, then token 1, then tokens 2..4 — each starting from what the **previous launch actually left in Global Memory**, not from what it was supposed to leave. A single launch into a poisoned cache cannot distinguish a kernel that preserves from one that rewrites the whole strip, because its reference starts from the same poison |
| `--mode zero` | the `absmax == 0` branch, which no integer fixture reaches |
| `--mode sweep` | every token count 1..16, so the multi-token key burst, the run grouping and both halves of the strip word are all exercised |

---

## 6. Results

All six rungs. One AIV, `gridDim = 1`, bit equality throughout.

```
canary_H0              PASS
canary_H2              PASS
canary_H3              PASS
sequential_H2          PASS
zero_H0                PASS
sweep_H3               PASS
```

Every rung reported, for every launch:

```
key bytes wrong 0, value bytes wrong 0, scale words wrong 0
nibble census: owned wrong 0, preserved-in-strip moved 0,
               foreign moved 0, owned unwritten 0
```

and every head and tail guard band intact on all nine Global Memory buffers.

That is **22 launches** in total — three isolated heads, a three-step ladder, the
zero row and sixteen token counts — and no launch moved a byte it did not own.

The single-token canary launch at `H = 0` is 7,168 model ticks end to end with a
6,122-tick dispatch span; the fixture's rotated panel peaks at
`|Pi . x| = 367`, giving `absmax = 367`, `e = 8` and a stored scale of
`2^6 / 16 = 4`.

The sweep is the largest single dump: 136 rotated rows per plane over sixteen
launches, and its audit is the strongest form of every count in §6.2 because
every one of them is exactly sixteen times the single-launch figure.

```
routes      (MTE2,VEC,0) 48   (MTE2,VEC,1) 16   (VEC,MTE2,0) 48
            (VEC,MTE3,0) 32   (VEC,MTE3,1) 16   (MTE3,VEC,0) 48
            (MTE3,MTE2,0) 32  (MTE3,SCALAR,0) 32        balanced, 16x each
butterfly   RV_VMULA 6528 == 6 x 4 x 2 x 136
            RV_VSELR 8160 == 6528 butterfly + 1632 absmax
movers      64 loads  == 16 pi_signs + 3 x 16
            320 stores == 16 probe + 2 x 16 + 2 x 136
            16 addresses both read and written, each read before written
```

### 6.1 The launch probe: the roadmap's open item, settled

The kernel publishes `GetBlockIdx()`, `GetBlockNum()` and `GetSubBlockNum()`
from block 0 of every launch rather than relying on either convention, because
the documented MIX behaviour is
`AIV: get_block_idx() * get_subblockdim() + get_subblockid()` and the safe MIX
idiom is `GetBlockIdx() / GetSubBlockNum()`. Observed, identically on all three
canary coordinates:

```
launch probe: GetBlockIdx 0, GetBlockNum 1, GetSubBlockNum 1, itemCount 1
```

**Every AIV is its own block under `KERNEL_TYPE_AIV_ONLY`, and the MIX idiom is
not needed here.** Measured at `gridDim = 1` only; the scaling of
`GetBlockNum()` under this kernel type is unobserved, so roadmap risk R5 closes
for the one-block case and stays open for a wide grid.

The scalar reaches Global Memory through `Duplicate` into four Unified Buffer
DataBlocks and one MTE3 burst, not through a `LocalTensor::SetValue` — which
would need a `V_S` fence and a `TPipe` this kernel does not have (guidelines
§1.1).

### 6.2 The trace audit

Six checks, each fatal, over each run's own archived dumps
([`analyze_scatter_trace.py`](../scripts/analyze_scatter_trace.py)). All green
on every rung. Three of them are worth quoting.

**Zero cross-core flags, as an absence.** `SET_INTRA_BLOCK`,
`WAIT_INTRA_BLOCK` and every other mode-4 instruction must be **absent**, not
balanced: a balanced pair would still mean the kernel is paying for a handshake
it has no partner for, and a stale mode-4 set is the mechanism behind two of the
faults the decode operator had to fix. Count: zero.

**The HardEvent census, keyed correctly.** Eight `(pipe, trigger pipe, flag id)`
routes on the single-token rung, 17 sets and 17 waits, balanced one for one:

```
(MTE2, VEC,    0) x3   pi_signs, key rows, value rows
(MTE2, VEC,    1) x1   the strip read
(VEC,  MTE2,   0) x3   priming + one per plane
(VEC,  MTE3,   0) x2   the launch probe + the packed key page
(VEC,  MTE3,   1) x1   the merged strip
(MTE3, VEC,    0) x3   priming + one per plane
(MTE3, MTE2,   0) x2   priming + the strip's reuse edge
(MTE3, SCALAR, 0) x2   the probe's drain + the launch drain
```

no reserved id 6 or 7, and every `SET_FLAG` precedes its `WAIT_FLAG` in
instruction order on every route. Each count is exactly what the kernel's own
event-id assignment predicts, which is what makes this a check and not an
observation.

On the three-launch sequential ladder every one of those eight routes shows
**exactly three times** the single-launch count — 9, 3, 9, 6, 3, 9, 6, 6 — so
no token leaked from one launch into the next. That is the property the
prime-before / drain-after pattern exists for, and a leaked token is the one
fault that a single-launch rung cannot see at all.

Two faults in the keying, both in the audit and not in the kernel, are worth
recording because each made a real check weaker without failing:

* The route key comes from the instruction's **own operand fields** —
  `PIPE:x, TRIGGER PIPE:y, FLAG ID:z` — and not from the queue column it was
  dispatched on. A `SET_FLAG` is dispatched on its pipe and the matching
  `WAIT_FLAG` on the trigger pipe, so keying on the column pairs nothing with
  anything: the first version reported nine phantom leaks on a correct kernel.
* The model prints a non-zero flag id in **hex** (`FLAG ID:0x2`) and zero in
  decimal (`FLAG ID:0`). A `\d+` pattern matches the leading `0` of `0x2` and
  stops, so **every id collapsed into bucket 0** — which turned the balance
  check into a comparison of two routes' sums against two others', and made the
  reserved-id check vacuous. This one *passed*, and it would have kept passing
  through an imbalance that cancelled between two ids of one route. It was
  caught by comparing the census against the kernel's own event-id assignment
  and finding a route with four sets where the source has three, then confirmed
  against an archived dump from an unrelated probe
  ([`reports/camodel_core0_popped.dump`](camodel_core0_popped.dump)) whose
  twelve routes resolve to ids 0, 1 and 2 only after the fix.

**The read-modify-write, proved from the data movers.** Nine `MOV_SRC_TO_DST`
bursts on the single-token rung — four `OUT->UB` loads and five `UB->OUT`
stores, exactly the straight-line body — and the strip's load and store are at
the **same Global Memory address**, the load first:

```
id    217 tick    1824 MTE3  UB->OUT gm 0x164b90c40   the launch probe
id    240 tick    2398 MTE2  OUT->UB gm 0x164b90440   pi_signs
id    659 tick    3564 MTE2  OUT->UB gm 0x164b6d840   key rows
id   1056 tick    4991 MTE3  UB->OUT gm 0x164b74c40   the packed key page
id   1107 tick    4992 MTE3  UB->OUT gm 0x164b8f340   the key scale, DataCopyPad
id   1179 tick    4908 MTE2  OUT->UB gm 0x164b6e240   value rows
id   1190 tick    4910 MTE2  OUT->UB gm 0x164b84e40   the strip, READ
id   1576 tick    5989 MTE3  UB->OUT gm 0x164b84e40   the strip, WRITTEN
id   1607 tick    5990 MTE3  UB->OUT gm 0x164b8fb40   the value scale, DataCopyPad
```

A kernel that wrote a strip it never read would still pass a comparison whose
reference started from the same poison; it does not pass this.

The burst census scales with **both** the transaction count and the token count,
and getting that wrong was a fault in the verification rather than in the kernel.
A run of `n` tokens is *one* strided key-row burst, *one* value-row burst and
*one* strip read — so the loads scale with runs — but the scale plane is four
bytes per token, sub-burst, one `DataCopyPad` each, so the stores scale with
tokens as well:

```
loads  = launches + 3 x runs
stores = launches + 2 x runs + 2 x tokens
```

On the three-launch sequential ladder (1 + 1 + 3 tokens) that is 12 loads and 19
stores, and the dump has exactly 12 and 19. The first version of this check had
`4 x runs` stores and reported a numerically correct kernel as wrong on the first
multi-token rung.

A second assertion was removed as unsound: that the number of **distinct**
re-written Global Memory addresses equals the run count. Each launch allocates its
own Global Memory and a freed allocation is handed straight back, so two launches
of a multi-launch rung routinely see the same address for their strip — the
sequential ladder's three launches show two distinct addresses. The property that
matters is the one above, *every re-written address was read first*, and it does
not depend on how the allocator recycled.

**The butterfly did not spill, as an equality.** Each lane-bit stage of the
RegGather leaf is one `RV_VSELR` (the register lane permutation) plus one
`RV_VMULA` (the fused partner + sign·self), and the absmax all-reduce adds six
more `RV_VSELR`. For one row over two planes that predicts 48 and 60; for the
sequential ladder's five rows, 240 and 300. The dumps have exactly:

```
one token:   RV_VMULA  48 == 6 stages x 4 registers x 2 planes x 1 row
             RV_VSELR  60 ==  48 butterfly +  12 absmax
five tokens: RV_VMULA 240 == 6 stages x 4 registers x 2 planes x 5 rows
             RV_VSELR 300 == 240 butterfly +  60 absmax
```

An equality rather than a bound, because a spill of the four resident registers
would multiply the surrounding loads and stores without changing these, and a
recompiled leaf that lost a stage would change them exactly.

**The companion Vector core did nothing** — an empty instruction census — which
is what `KERNEL_TYPE_AIV_ONLY` with a one-block grid is supposed to mean and
what the protocol's "all unassigned head regions remain unmutated" conclusion
rests on.

### 6.3 One limitation of the multi-launch rungs

The model writes its instruction dumps when the **process** stops, so a runner
mode that performs several launches leaves one dump covering all of them. The
audit's counts are therefore totals (`--rows`, `--runs`, `--launches` scale
every assertion), and it cannot attribute a flag imbalance to a particular
launch within `sequential` or `sweep`. What it can still say, and does, is that
the totals balance — which is the property that rules out a token leaking from
one launch into the next.

---

## 7. What this does and does not establish

**Established**, on this host, by the runs above:

* The eight-stage `Pi . x` ingress is bit-exact against a dense `256 x 256`
  right-multiplication (host) and against the stage cascade (device).
* The quantiser is bit-exact: codes and scales, every rung, every token count
  from 1 to 16.
* The value plane's read-modify-write preserves every lane it does not own,
  **across launches** — the second launch of the sequential ladder starts from
  what the first physically left.
* No byte outside the owned `(block, head, strip)` moves, on any of the three
  planes, with every guard band intact.
* The `absmax == 0` branch is reached and stores `scale = 1.0f` with all-zero
  codes.
* Zero cross-core flags, zero `wait_intra_block`, balanced HardEvents with
  set-before-wait on every route, and a spill-free register-resident butterfly.
* `GetBlockIdx()`, `GetBlockNum()` and `GetSubBlockNum()` under
  `KERNEL_TYPE_AIV_ONLY` at `gridDim = 1`.

**Not established**, and not claimed:

* **Silicon.** Model ticks for one AIV with no other core running. The model's
  issue rules are not a timing guarantee, and there is no NPU on this host.
* **A wide grid.** Every run is `gridDim = 1` with `itemCount = 1`. The
  grid-stride loop over `(request, kv_head)` is compiled and exercised for a
  single item; two work items have never run concurrently, so nothing here says
  that two cores writing two heads of one page do not tear — although they
  cannot, because the planes' head strides are disjoint and the strip windows
  are per `(block, head, strip)`, which §2 of the Tier-1 probe proves as an
  index property rather than as a run.
* **`blockSize > 16`.** The fixture uses `S == R == 16`, one strip per page,
  where the value plane is byte-identical to plain channel-major. The kernel's
  `token / R` strip index and `strip_byte_offset` are general and the Tier-1
  probe checks the window property for every `(block, head, strip)` of the
  shape it is given, but no CAModel run has used `S = 128`, which is vLLM's
  page size.
* **Non-dyadic activations.** The fixture is integer-valued in BF16 so that bit
  equality is demandable. Real activations are not, and then the rotation's FP32
  adds are still exact while the `rint` boundary becomes input-dependent — the
  arithmetic does not change, but a bit-exact gate stops being available.
  No tolerance-based accuracy rung has been run.
* **A very small `absmax`.** The guard covers `absmax == 0`. For
  `absmax < 2^-124` the exponent arithmetic `2^(e-2)` underflows the FP32
  normal range and the stored scale becomes a denormal or zero; `||y|| = 16
  ||x||` for this rotation bounds that away for any activation a BF16 tensor can
  hold with a normal exponent, but the kernel does not check it and no run has
  probed it.
* **The scale outflow's efficiency.** The scale plane is one FP32 per
  `(kv, token, head)` — four bytes, sub-burst — so it goes out as one
  `DataCopyPad` per token, which is up to sixteen tiny MTE3 transfers per plane
  per run. The RFC allows gathering a run's tokens into one burst instead, and
  that is not done here; at `n = 1`, the decode case, it is one transfer either
  way.
* **Any claim about Approach A as shipped.** The Blocked butterfly is in the
  leaf as the measured baseline. Nothing here retracts the decode egress's use
  of the same shape *outside* a hot loop — that trade was measured separately
  ([`paged_decode_attention.md`](paged_decode_attention.md)) and the rotation
  there runs once per sequence, not once per cached token.

### 7.1 What was deliberately not done

`routines/routine_vector_egress_rotate.hpp` is **not** refactored onto the new
leaf, although roadmap §1.0 lists it. The decode operator's egress is validated
against nine canary coordinates and a three-rung strip ladder, and re-pointing it
at a new header is a change to verified device code with no gate of its own in
this cycle. The new leaf carries the egress direction, the Tier-1 probe checks it
against the dense right-multiplication, and `RotateImpl::RegGather`'s egress path
compiles — but it has not been run on CAModel, and the refactor stays a separate
item with its own ladder.

`tests/common/turboquant_oracle.hpp` is **not** extended in place, although
roadmap §1.1 lists it. That header is the oracle for the aligned INT8 writer and
the BF16 one, and both device validations rest on it; putting a second contract
behind one set of names in that file is the kind of overlap a bug can live in.
The scatter oracle is a separate header that **reuses** `tq_oracle`'s rounding
rule and nibble conventions and `rot_oracle`'s rotation by inclusion, so nothing
in this family is defined twice.
