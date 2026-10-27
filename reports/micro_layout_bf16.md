# BF16 Cube operand layouts on arch35 — single-tile CAModel probe

Target `Ascend950PR_9599` CAModel, `__NPU_ARCH__ == 3510`, CANN 9.2.0-beta.2.
Probe: [`tests/camodel/test_micro_layout_bf16.cpp`](../tests/camodel/test_micro_layout_bf16.cpp)
over [`kernels/camodel/micro_layout_bf16.cpp`](../kernels/camodel/micro_layout_bf16.cpp).

```sh
bash scripts/run_micro_layout_bf16_camodel.sh
```

There is no NPU on this host. A pass means "executes on the cycle-accurate
model and matches the audited golden", not a hardware claim.

---

## 0. The question, and the answer the run gives

`turboquant_reshape_and_cache_bf16.cpp` is the INT8 prefill operator with one
element widened from 1 byte to 2. The INT8 version rests on two "the same bytes
read two ways" identities, both of which the widening puts in doubt: the
arch-3510 fractal is 16 × (32B / sizeof(T)), so C0 drops from 32 to 16 and a
32-wide axis stops being one fractal and becomes two.

The widening breaks exactly one of the two, and it is **not** the one the
original diagnosis named.

| identity | at C0 = 32 | at C0 = 16 | measured |
| --- | --- | --- | --- |
| 16 × 256 activation page: Nz image == Zn image of its transpose | holds | **holds** | case 4, and cases 2–3 over all eight groups |
| 32 × 32 resident rotation block: row-major blob == its own Nz image | holds | **fails** | case 1 against a predicted misread; cases 2–3 against the true oracle |

So the Value chunks do not fail because the page cannot be consumed as ZN. The
page is fine. Both the Key and the Value chunks are fed a **scrambled resident
operand**, because the prologue stages the 16 rotation blocks with a flat
ND → ND copy on the strength of an identity that only holds at 8 bits.

§4 carries the operator-level control: with the flat ingress restored the
operator fails on its own runner, and with the per-block ND → Nz ingress it is
bit-exact at 1, 2 and 3 strips.

### Why the page survives

Writing the two formats out (`key_fractal_format_detailed_explanation.md`):

```
Nz(M, N):  (M, N) -> (N1, M1, M0, N0)   offset = n1*M1*M0*N0 + m1*M0*N0 + m0*N0 + n0
Zn(K, N):  (K, N) -> (K1, N1, N0, K0)   offset = k1*N1*N0*K0 + n1*N0*K0 + n0*K0 + k0
```

with M0 = N0 = 16 and N0(Nz) = K0(Zn) = 32B/sizeof(T) = 16 at BF16.

The page is Nz(16, 256), so M1 = 1 and its image is `n1*256 + t*16 + n0` with
channel `n1*16 + n0`. The group-G slice read as Zn(32, 16) has N1 = 1 and asks
for `k1*256 + t*16 + k0` with channel `G*32 + k1*16 + k0`. In both readings the
only fractal-grid axis is the channel axis and the group's two sub-fractals are
adjacent along it, so the bytes coincide. The read as Nz(16, 32) for the A role
coincides for the same reason. Nothing has to change for the page.

### Why the resident block does not

A 32 × 32 block has M1 = K1 = 2 at BF16, so both grid axes are live. The image
the Cube reads is

```
offset = (c/16)*512 + (r/16)*256 + (r%16)*16 + (c%16)        holds W[r][c]
```

— four 16 × 16 sub-blocks, each row-major, ordered **column-block major** —
against row-major's `r*32 + c`. At C0 = 32 the two are the same map, because
the fractal is a whole 32-wide row and the grid is one fractal along K. At
C0 = 16 they disagree everywhere outside the first sub-block's first row.

The same image serves both operand roles, exactly as at 8 bits: the A-role
Nz(32, 32) of W and the B-role Zn(32, 32) of `B[k][n] = W[n][k]` are the same
map with the second index renamed. One ingress conversion, two byte-identity
L1 → L0 loads.

---

## 1. The probe

AIC only. One 16-token × 256-channel page, eight 32 × 32 resident blocks,
sixteen accumulating Mmads, two Fixpipes straight to Global Memory. No Vector
core, no Unified Buffer, no cross-core flag, no pipeline ring — so a mismatch
cannot be a synchronisation artefact, which is what separates this from running
the full operator and bisecting.

It computes one (chunk, output-group) cell of the production GEMM in both
operand orientations over the same operands:

```
value orientation   C_v[c0][t] = sum_G sum_i W_G[c0][i] . x[t][G*32 + i]
                    A = block Nz(32, 32),     B = page group Zn(32, 16)
key orientation     C_k[t][c0] = the same sum
                    A = page group Nz(16, 32), B = block Zn(32, 32)
```

`C_k` is `C_v` transposed, so one oracle checks both and a fault in either
operand role shows up twice.

Operands come from the shared fixture in
[`tests/common/turboquant_oracle.hpp`](../tests/common/turboquant_oracle.hpp):
the value plane of head 0, strip 0 (production chunk 0) and the eight blocks
output group `c1 = 0` selects. With integer activations and ±1 blocks every
product and partial sum is an integer below 2²⁴, so FP32 accumulation is exact
whatever order the Cube reduces in and every comparison is **bit-exact**, not
toleranced.

### The four cases

| # | ingress | uploaded blob | must reproduce |
| --- | --- | --- | --- |
| 1 | flat ND → ND (the 8-bit prologue) | row-major | the **predicted misread** |
| 2 | per-block ND → Nz (`stage_blocks_nz`) | row-major | the true oracle |
| 3 | flat ND → ND | host-side Nz image | the true oracle |
| 4 | flat ND → ND | Nz identity in block 0, zeros elsewhere | `C_v[c0][t] == x[t][c0]` |

Case 1 does not merely have to fail. The runner predicts, from the Nz index map
alone, which entries the Cube will read out of a row-major blob, recomputes the
oracle with that permuted operand, and requires the device to match **that**
bit-exactly. The two predictions disagree on **508 of 512** cell entries, so the
case is not passing by coincidence. A diagnosis that predicts the wrong answer
exactly is a mechanism; one that only predicts "wrong" is a guess.

Case 3 attributes the fault from the other side: everything downstream of the
ingress is identical between cases 1 and 3, so fixing the blob instead of the
ingress isolates the ingress as the only thing wrong. Case 4 isolates the page's
Nz/Zn duality with no dependence on block content at all.

---

## 2. Results

```
fixture: the two operand images disagree on 508 of 512 cell entries
PASS flat ingress, row-major blob     512/512 exact in both orientations
PASS nd2nz ingress, row-major blob    512/512 exact in both orientations
PASS flat ingress, host Nz blob       512/512 exact in both orientations
PASS flat ingress, Nz identity blob   512/512 exact in both orientations
Model RUN TIME: 37935.8 ms
[INFO] Total tick: 14190
```

Gates, in the order
[`MICROROUTINE_DESIGN_GUIDELINES.md`](../MICROROUTINE_DESIGN_GUIDELINES.md) §4
puts them:

1. device exit code 0, and **zero** `[error]` / `[fatal]` /
   `invalid ldst addr` / `su_ccu_mpu_err` lines in a 67-line log;
2. head and tail `0xDEADBEEF` canaries intact on all four Global Memory
   buffers, after every launch;
3. bit-exact numeric agreement per entry, in both orientations, for all four
   cases.

Cost: 14 190 device ticks for the four launches, inside ~38 s of simulator wall
clock. The wall clock is dominated by model start-up — the log shows 32 AI cores
being attached before anything runs — not by the kernel; the device work is
~3 500 ticks per launch. The ~5 s target in the brief is met by the execution,
not by the end-to-end wall clock of the model.

---

## 3. What this settles about the two proposed approaches

### Approach 1 — MTE1 sub-fractal load: not expressible, and that is a property of the path

[`routines/routine_fractal_load_zn_bf16.hpp`](../routines/routine_fractal_load_zn_bf16.hpp)

`Copy_L1ToL0B.md` is explicit: the L1 → L0 path "uses a 512-byte data fractal as
the basic copy unit", the b16 fractal cube is 16 × 16, and exactly two layout
combinations exist per destination —

```
L0A:  Nz -> Nz (non-transpose)    Zn -> Nz (transposes each fractal)
L0B:  Zn -> Zn (non-transpose)    Nz -> Zn (transposes each fractal)
```

Both transposing forms transpose **inside** a fractal. Neither reorders
fractals, and no form gathers a sub-fractal out of a source that is not already
fractal-organised. The low-level ISASI `LoadData` is the same instruction
underneath — `LOAD_2Dv2` addresses L1 by fractal index — so it does not widen
the envelope. MTE1 therefore cannot repair a row-major L1 image, at any
parameter setting.

What it *does* supply is the page duality, and the routine is the four loads
that use it: two byte-identity reads of one L1 page image for the A and B roles,
and two byte-identity reads of one resident-block image for the A and B roles.
No transposing form is needed anywhere on this operator.

### Approach 2 — ingress formatting: one call, and it is the fix

[`routines/routine_stage_page_zn_bf16.hpp`](../routines/routine_stage_page_zn_bf16.hpp)

`stage_blocks_nz<Blocks, Dim>` converts each block on the way in with
ND → Nz, landing block *b* at `l1_byte_base + b * Dim * Dim * 2` — where a flat
blob would have put it — so only the bytes inside a block move and **every
downstream base address, layout descriptor and Mmad is unchanged**. It is a
prologue cost paid once per launch.

`stage_page_nz` is the production page ingress factored out unchanged, because
the page needs no repair; it exists so a probe can stage the page exactly the
way the operator does. `stage_blocks_flat` is the 8-bit behaviour, kept so the
difference is something the probe measures rather than something a comment
claims.

One Copy per block rather than one batched Copy over all sixteen: the batched
form describes the blob as `(B, (Dim, Dim))` and derives the destination spacing
as "one cube after alignment", which happens to equal `Dim*Dim*2` here but is a
derived quantity rather than a stated one. The blocks are staged once, so the
extra MTE2 descriptors cost nothing worth that coupling.

---

## 4. Integration, and the before/after control

`stage_blocks_nz<kTableBlocks, kGroup>` replaced the flat prologue copy in
[`turboquant_reshape_and_cache_bf16.cpp`](../kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp).
Nothing else moved: block *b* still lands at `b * kBlockElems`, the two
L1 → L0 copies are still byte identities, every Mmad and every layout
descriptor is untouched, and the GM contract for `tables` — 16 row-major
32 × 32 blocks — is unchanged, so no host, runner or oracle code had to move.

```sh
bash scripts/run_reshape_bf16_camodel.sh          # 1, 2 and 3 strips
```

| ingress | 1 strip | result |
| --- | --- | --- |
| `stage_blocks_flat` (the 8-bit prologue) | **FAIL** | value chunk 0: 2013/2048 bytes differ, 7/16 scales differ; chunk 1: 1972/2048; chunk 2: 1993/2048 |
| `stage_blocks_nz` | **PASS** | bit-exact at 1, 2 and 3 strips, 8/16/24 chunks |

The control is the point: the operator was run with the flat ingress restored
and it failed, so "the fix was necessary" is measured rather than inferred.
Ticks were 15 183 (failing) against 16 134 / 16 189 / 16 106 for the three
passing strip counts — the conversion is a prologue cost and does not move the
steady state.

`turboquant_bf16_npu`, which compiles the same source for a physical device
under `-std=c++17 -Wall -Werror`, also builds clean, so the new include carries
no diagnostics into the strict build:

```sh
cmake --build <dir> --target turboquant_bf16_npu
```

### Why the breakage looked Value-specific

The original diagnosis read the failure as "Value chunks deterministically
fail", which pointed at the page's ZN consumption. It is an artefact of the
runner's reporting, not of the kernel. Both operand roles take the same
scrambled blob, so the Key chunks are equally wrong — but
`test_camodel_turboquant_reshape_and_cache_bf16.cpp:247` caps the comparison at
`failing < 3`, and chunks 0..3 are the value heads. The runner therefore stops
after three value chunks and never prints a key chunk, whatever the key chunks
did.

A failure report truncated by the harness is evidence about the harness as much
as about the kernel. This one sent a cycle of work at the wrong operand.

## 5. What this does **not** establish

* **One cell, not the operator.** The probe covers one (chunk, output-group)
  cell of one strip. It says nothing about the four-stage ring, the cross-core
  flags, the quantiser or the cache layouts — all of which the operator's own
  runner covers and none of which this one touches.
* **Case 4 tests group 0's page read only**, since blocks 1..7 are zero there.
  The per-group L1 offset arithmetic is covered by cases 2 and 3, which run all
  eight groups against the true oracle.
* **No claim about FP16 or other 16-bit types** beyond the shared C0: the
  arithmetic exactness argument here is specific to integer-valued BF16
  activations with ±1 blocks.
