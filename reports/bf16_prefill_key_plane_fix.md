# The BF16 prefill key-plane defect: a recycled mode-4 id does not consume

`turboquant_reshape_and_cache_bf16` failed its own single-strip CAModel gate.
Key heads 1 and 3 disagreed with the host oracle in ~2000 of 2048 packed bytes
and in 15 of 16 scale lanes; the other two key heads and all four value heads
were bit-exact. This report has the diagnosis, the fix, every gate run with its
result, and every gate not run with the reason.

No NPU on this host. Everything below is `Ascend950PR_9599` CAModel under CANN
9.2.0-beta.2, and nothing here is a hardware claim.

---

## 1. Root cause, in one paragraph

A mode-4 `wait_intra_block` on an id that already carried an announcement does
not reliably consume it on this target, so the **second** wait on a recycled id
is released by the **first** announcement's residue. The operator's production
default was a recycled two-slot mode-4 ring, which gives each Vector core two
`productReady` ids over four landing generations. AIV1's second and third waits
on those ids retired 1774 and 2600 CAModel cycles before the announcements that
should have released them, so AIV1 copied two landing panels out of the ring
before the Cube's Fixpipe had written them, and the two chunks it owns in those
generations — chunks 5 and 7 — were quantised from stale panels. AIV0's
corresponding waits retired 10 cycles *after* each matching announcement, which
is correct, and that ten-cycle margin is the whole reason four of the eight
chunks were right. This is **not** guidelines §6.2a's hazard: every
announcement in the kernel is behind its scalar drain and
`scripts/audit_cross_core_fences.py` was already green on the failing
revision — the announcement had not issued *at all* when the wait passed. The
fix replaces the mode-4 ring with the mode-2 readiness quorum and release
broadcast, whose ids are counters that bank what is not yet consumed, so
recycling is sound by construction rather than by a latch reading.

## 2. The signature, and what was wrong about it

Reproduced exactly as reported:

```
BF16_BUILD_DIR=/work/bf16keyfix/b2 BF16_SLICES=1 bash scripts/run_reshape_bf16_camodel.sh

strip 0 page 1 chunk 5 (key head 1): 2006/2048 bytes differ (first byte 0: device 0x1f oracle 0xe2), 15/16 scales differ
strip 0 page 1 chunk 7 (key head 3): 2011/2048 bytes differ (first byte 0: device 0x0e oracle 0xd1), 15/16 scales differ
2 of 8 chunks mismatch (2 printed)
FAIL turboquant_reshape_and_cache_bf16 (1 slices, recycled mode-4 ring)
```

The runner's three-chunk break was lifted first, because `README.md` records
that an earlier revision of it hid broken key chunks behind that cap. It did
not hide anything this time — **exactly two of eight** chunks mismatch — but
the count is printed with the failure now, so "how many chunks are wrong" and
"which ones were shown" are separate facts rather than one number.

Two things about the signature were read wrongly at first, and both mattered:

**It is not "the odd key heads".** The harness's `chunk_plane(chunk)` is
`chunk >= kHeads ? key : value`, so chunks 0–3 are the four *value* heads and
4–7 the four *key* heads — value plane first. With `kOwner<C> = C & 1`, the key
chunks 4, 5, 6, 7 belong to Vector cores 0, 1, 0, 1. The failing set is
therefore exactly **the key chunks owned by AIV1**, and key heads 0 and 2, which
AIV0 owns, passed. Core ownership explains the whole failure set; "odd head"
does not.

**"Key plane" and "second trip round the ring" are the same statement here.**
`kLandSlot<C> = (C / 2) % kLandDepth` with `kLandDepth = 2`, so the value plane
is landing generations 0 and 1 — the *first* use of each id — and the key plane
is generations 2 and 3, the *recycled* use. A defect that only bites the second
announcement on an id can only appear in the key plane of this schedule. That
is what made a synchronisation fault look like a key-plane layout fault, and it
is why the 32x32 rotation-block Nz question was a plausible but wrong lead.

## 3. Kernel defect or environmental: the discriminators

The cheap bisection the finding session suggested — rebuild at `bdb479b` — was
**not run**. A commit bisect localises *when*; it cannot distinguish a kernel
defect from a target property, which was the open question. Two measurements
answer that question directly and were run instead.

### 3.1 The policy A/B

The two `__global__` entry points were byte-identical schedules differing only
in which physical id each event got.

| 1 strip, same binary | result |
| --- | --- |
| recycled two-slot mode-4 ring (the default) | **FAIL**, chunks 5 and 7 |
| mode-4 single-shot channels, one id per event | **PASS**, all 8 chunks |

So the fault is id recycling and nothing else.

### 3.2 The trace, in retirement order

`scripts/analyze_reshape_trace.py`'s order check could not be used: it compares
the two **issue** columns, and a wait that blocked is issued before the
announcement that releases it by definition, so it reported "a stale set
released it" on *every* run — including the bit-exact single-shot one. §6 has
what was done about that.

Reading the completion log instead, for AIV1's `productReady` channel
(`sync_id:16`, which is AIV1's local id 0):

| event | CAModel tick |
| --- | ---: |
| Cube `SET_INTRA_BLOCK sync_id:16`, generation 0 (chunk 1) | 6960 |
| AIV1's first wait on it retires | 6970 |
| AIV1's **second** wait retires, issued at 9534 | **9535** |
| Cube's second `SET_INTRA_BLOCK sync_id:16`, generation 2 (chunk 5) | 11309 |

1774 cycles early, with nothing outstanding. The same shape on `sync_id:17`:
wait retires 10985, announcement 13585 — 2600 cycles early. Those two waits gate
chunks 5 and 7. AIV0's channel in the same run:

| id | announcements | wait retirements |
| --- | --- | --- |
| `m4:0` | 5898, 10199 | 5908, 10209 |
| `m4:1` | 8060, 12370 | 8070, 12380 |

Ten cycles after each, both times. The consequence is visible downstream too:
AIV1's generation-2 release on `m4:18` was issued at 10745, before the Cube's
wait at 10847 consumed the generation-0 release still outstanding — two
announcements live on one id, which is a *lost* token. That is a consequence of
the early read, not a second fault.

### 3.3 The repo's own probe, with no data in flight

```
scripts/run_cluster_sync_probe_camodel.sh 4 4

core0.cubecore0 waits id m4:0  on MTE1  ok               [34085, 32959, 33028, 32902]
core0.cubecore0 waits id m4:16 on MTE1  ok               [16624, 15410, 15207, 15254]
core0.veccore0  waits id m4:1  on VEC   ok               [36492, 37437, 47184, None]
core0.veccore1  waits id m4:17 on VEC   early retirement [36213, 35673, 34155, None]
VIOLATION: core0.veccore1 waits id m4:17 on VEC: wait 1 was released at tick
158173 but the announcement that should have released it is at tick 159423.
```

This settles it independently of the operator: the fault is in the mechanism,
it is in the AIC → AIV direction, and it is **not symmetric between the sub
blocks** — AIV1's channel collapses in the same run in which AIV0's is clean.
Guidelines §6.2c had retracted an earlier "mode 4 does not consume" finding on
the strength of this probe at **two** iterations, which cannot tell a blocking
wait followed by a legitimately satisfied one from a latch that stopped
blocking; the runner's own header says so and defaults to four for that reason.
Four shows it.

`tests/cpu/test_reshape_wave_model.py` had also already said so: the recycled
policy fails its `sticky latch, never cleared` column while the single-shot
policy survives it. §6.2c chose the `consuming wait` column. The column this
target is in is the other one.

### 3.4 The leads that were ruled out

| lead | verdict |
| --- | --- |
| the 32x32 rotation block's Nz duality (`stage_blocks_nz`) | not it — all four value chunks and two of four key chunks are bit-exact through the same resident blocks, and the layout probe is unaffected by the policy A/B |
| `GetBlockIdx() / GetSubBlockNum()` per-core indexing | not it — at one strip the quotient is 0 either way, and the failure is per *chunk*, not per strip |
| the `Ascend950PR_9589` vs `9599` SOC-version discrepancy | not it — the single-shot leg passes on the same simulator in the same container |
| a wrong host oracle | not it — the oracle is shared with the INT8 instance, which passes, and the same oracle passes six of eight chunks here |

## 4. The fix

### 4.1 Mode 2: one readiness quorum, one release broadcast

Mode 2 is a different instruction, not a variant: `NotifyEventImpl` lowers mode
4 to `set_intra_block` and every other mode to `ffts_cross_core_sync`. It is the
fused "AIC and all AIVs" mode — the AIC's counter advances only once **both**
Vector cores have announced, one AIC announcement releases both, and each id is
a counter in 0..15 that banks unconsumed announcements. Recycling one is sound
by construction. `turboquant_paged_decode_attention.cpp` has shipped on it since
commit `719a390e`; this is the same construction.

The rendezvous is per landing **generation**, not per chunk, because mode 2
cannot address one Vector core. Generation `g` owns chunks `2g` and `2g + 1` —
one per core — and one landing slot `g % kLandDepth` in each core's UB:

| channel | id | direction | meaning |
| --- | ---: | --- | --- |
| `kSync2PanelReady` | 8 | AIC → both AIVs | both landing panels of this generation hold their products; issued after the Fixpipe of the generation's odd chunk, so it reports both |
| `kSync2PanelFree` | 9 | both AIVs → AIC | one quorum wait per generation: both cores have copied their panel out of the slot the generation about to be written last held |
| `kSync2PageDrained` | 10 | AIC → both AIVs | the inter-wave epilogue join, §6.5 rule 3 |

Ids 8, 9 and 10 are clear of SyncAll's 11..14 and of the Matmul API's worst-case
0..7, asserted rather than commented. Readiness and release never share an id.
Worst-case outstanding is `kLandDepth + 1 = 3`, against a counter maximum of
15, and it does not grow with the strip count.

Every pipe is `static_assert`ed, because **a mode-2 call naming a pipe its own
core may not use compiles to NOTHING on arch 3510, silently**: the AIC waits and
announces on `PIPE_FIX`, the pipe that writes the landing panel; both AIVs wait
and announce on `PIPE_V`, the pipe that reads it. The trace census below checks
the same thing from the other end — a dropped call shows up as a zero count.

The mode-4 recycled ring is no longer reachable from this kernel. The mode-4
**single-shot** leg is kept as the bisection control, on the second entry point
and `--mode4-single-shot`; because its eleven ids per sub block are exactly the
eight-chunk schedule, the launcher refuses to dispatch it with more than one
strip per block rather than letting it silently recycle an id.

### 4.2 The strip stream: trip count decoupled from channel capacity

One flat runtime loop over the block's strips, on both the Cube and the Vector
cores. The feedback cycles are primed **once** before the loop and drained
**once** after it, and the mode-2 counters carry across strip boundaries, so a
strip boundary is invisible inside the loop and there is no per-strip
prime/drain cascade. Nothing in the pipeline grows with the trip count.

What makes that work is an arithmetic property, now asserted:
`kChunks` is a whole multiple of both `kStageDepth` and `2 * kLandDepth`, so
every ring slot index stays a function of the **chunk** alone and a strip
boundary does not shift which slot a chunk owns. Only the Global Memory bases
advance — the activation pointers by one strip of rows, the cache bases from
each strip's own `slotMapping` entry, so a fragmented slot mapping still works.

The live mask: `strips = min(tokensPerCore / blockSize, (numTokens - strip) /
blockSize)`. All three roles compute it from the same two host scalars and reach
the same number, so a short block announces and consumes a matched census and
leaks no token. The launcher still refuses a context that is not a whole number
of `tokensPerCore`, so this is the guard that makes lifting that refusal a
host-side change.

### 4.3 The shared header

`kReshapeDefaultReuse` is `false` now, so the INT8 instance — which was **not**
ported to mode 2 — defaults to its verified single-shot channels instead of the
refuted ring. `kReshapeBf16DefaultMode2` is new and is what the BF16 kernel and
its harness select on, because the two instances no longer use the same
mechanism and one constant for both would make one harness silently dispatch
the other's discipline. `RecycledChannels` stays compiled, and the host gate
keeps checking its algebra, because that is what keeps the refutation
reproducible.

## 5. Gates

Build: `cmake -DSOC_VERSION=Ascend950PR_9599`, CANN 9.2.0-beta.2 in
`ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12`, clean under the project's
`-Wall -Werror` device build.

### 5.1 Run and passed

| gate | command | result |
| --- | --- | --- |
| cross-core fence audit, all three kernels | `python3 scripts/audit_cross_core_fences.py` | PASS — every bf16 announcement `ok` with an adjacent drain, one priming set exempt by name |
| host protocol gate, header leg | `bash scripts/run_reshape_wave_host_gate.sh` | PASS — both policies legal, balanced, alternating, within budget; boundary guard exact on 1/32/33/48/64/128 strips |
| host protocol gate, as a whole | same | **FAIL**, one finding, and it is the pre-existing `tests/research/kernel_cluster_sync_probe.cpp:120` fence finding named in §5.4. That file is untouched here. Every other leg of the script passes, the wave model included |
| isolated single-strip canary | `BF16_SLICES=1` | PASS — 8/8 chunks bit-exact, chunks 5 and 7 included |
| depth, 2 strips | `BF16_SLICES=2` | PASS |
| depth, 3 strips (interior step, core reuse) | `BF16_SLICES=3` | PASS |
| in-block stream, 2 strips x 2 blocks | `BF16_SLICES=4 BF16_STRIPS_PER_BLOCK=2` | PASS |
| in-block stream, 4 strips x 2 blocks | `BF16_SLICES=8 BF16_STRIPS_PER_BLOCK=4` | PASS |
| in-block stream, 8 strips x 1 block | `BF16_SLICES=8 BF16_STRIPS_PER_BLOCK=8` | PASS |
| mode-4 single-shot fallback, 1 strip | `BF16_SYNC=mode4-single-shot BF16_SLICES=1` | PASS |
| mode-4 single-shot fallback, 3 strips | `BF16_SYNC=mode4-single-shot BF16_SLICES=3` | PASS |
| INT8 instance under the new shared default | `A8_SLICES="1 2 3" bash scripts/run_reshape_a8_camodel.sh` | PASS at 1, 2 and 3 strips — this is what verifies flipping `kReshapeDefaultReuse`, since the INT8 kernel itself is unchanged and only its default policy moved |
| one full wave, 32 strips, 32 blocks | `BF16_SLICES=32` | PASS — 256/256 chunks bit-exact |
| trace census and channel shape, 1 strip | `analyze_reshape_trace.py` | PASS |
| trace census and channel shape, 8-strip stream | `analyze_reshape_trace.py --strips-per-block 8` | PASS |
| trace census, mode-4 fallback | `analyze_reshape_trace.py --mode4-single-shot` | PASS |
| release-tick / stall audit, 1 strip | `audit_mode4_tick_order.py --block 0 --cross` | PASS on every data channel; one caveat in §5.2 |
| release-tick / stall audit, 8-strip stream | same | PASS on every data channel; same caveat |
| exception dumps | `core*.excp_log.dump` | all zero bytes on every rung |
| CAModel errors | `invalid ldst addr`, `unrecognize ldst addr`, `su_ccu_mpu_err` | zero occurrences on every rung |

What the trace audits establish, on the 8-strip stream:

* intra-core HardEvents: 725 sets / 725 waits across 34 (route, id) pairs on
  three cores, **zero residual**, every place alternating `SWSWSW...`;
* `m2:8` 32 AIC announcements → 32 waits on each AIV; `m2:9` 34 announcements
  from each AIV (2 priming + 32) → 34 AIC waits (32 in the schedule + 2
  epilogue drains); `m2:10` one → one each. Every counter ends at zero;
* the pipes are the asserted ones, from the trace: `AIC SET m2:8 on FIXP`,
  `AIC WAIT m2:9 on FIXP`, `AIV0/AIV1 SET m2:9 on VEC`, `AIV0/AIV1 WAIT m2:8 on
  VEC`. No call was silently dropped;
* 80 GM→L1 ND→Nz bursts = 64 pages plus **16** resident rotation blocks staged
  once for the whole eight-strip stream. That the table is not re-staged per
  strip is direct evidence the stream crosses strip boundaries;
* 4096 Mmads and 64 Fixpipes, one per chunk; 0 UB→L1 moves on the Cube; 64
  Global Memory stores from each Vector core, its own half;
* every AIV wait on `m2:8` and `m2:10` blocked a real interval on every use —
  1182 to 4722 ticks, 32 of 32 — which is the property the fix rests on, in the
  direction that was broken.

### 5.2 Run, with a caveat recorded

`audit_mode4_tick_order.py` reports `early retirement` for one channel on both
mode-2 runs: the Cube's quorum wait `m2:9` on `FIXP`. The flagged wait is the
**last epilogue drain** in both runs — wait 5 of 6 at one strip, wait 33 of 34
at eight — and never a schedule wait.

That reading is not usable, for the reason the script itself documents: a
cross-core wait gates the issuance of the pipe it names, so its release can only
be read from the scalar drain behind it, and by the epilogue the Cube's FIX
queue is **empty** — a `FIX_S` pair on an empty queue retires immediately
whether or not the gate has opened. The two epilogue drains also share that one
`FIX_S` pair, so the script cannot attribute a separate release tick to each.
Its stall column says the same thing: `[2, 2, ... 3, 2]`, which the script
defines as `no-fence: stall not measurable on this channel`.

What would make this conclusive is the two-wave rung, where a leaked credit
would let wave 1's Cube overwrite a panel wave 1's AIVs were still reading. That
rung fails for a different reason — §5.3 — so **this channel's epilogue drain is
not verified**, and it is listed here rather than in §5.1.

### 5.3 Run and FAILED: the wave boundary, 33 strips

```
BF16_SLICES=33 BF16_STRIPS_PER_BLOCK=1 bash scripts/run_reshape_bf16_camodel.sh

strip 32 page 65 chunk 0 (value head 0): 1995/2048 bytes differ
... all eight chunks of strip 32 ...
8 of 264 chunks mismatch
FAIL turboquant_reshape_and_cache_bf16 (33 slices, 1 strip(s) per block, mode-2 quorum and broadcast)
```

Strips 0–31 are bit-exact; 32 blocks on 32 AI Cores — one full wave — passes
outright, 256 of 256 chunks. Strip 32, the single block of wave 1, is wrong in
**all eight** of its chunks, both planes, both Vector cores.

**This is a role-placement fault, not a synchronisation one, and it is not in
anything this change touched.** Counting mode-2 calls per physical sub-core
across the whole 33-block run — the Cube issues 5 per grid block it runs, each
Vector core 6 — gives:

| role | grid blocks executed | expected |
| --- | ---: | ---: |
| Cube | 33 | 33 |
| AIV0 | **32** | 33 |
| AIV1 | **34** | 33 |

and the two outlying physical cores are:

| physical core | Cube | AIV0 | AIV1 |
| --- | ---: | ---: | ---: |
| `core9` | 2 blocks | **1** | 2 |
| `core30` | 1 block | 1 | **2** |

So wave 1's grid block ran with its Cube on `core9`, **two sub-block-1 Vector
cores** — `core9.veccore1` and `core30.veccore1` — and **no sub-block-0 Vector
core at all**. Both of those took the `Core == 1` arm of the kernel's role
dispatch, so both computed the odd chunks and nothing wrote the even ones; and
`core30.veccore1` had no Cube of its own announcing panels for that block, so
whatever it quantised came out of its own landing ring. That is sufficient to
explain all eight chunks: the four even heads were never written for strip 32,
and the four odd heads were written twice from different panels.

The role dispatch is `if ASCEND_IS_AIC / else if (GetSubBlockIdx() == 0) / else`
and is byte-identical before and after this change, as is the
`GetBlockIdx() / GetSubBlockNum()` derivation above it. The census is also
independent of the cross-core discipline: it counts how many times each physical
sub-core entered the kernel body, which no channel policy can affect. So this
is a **second, independent defect, at the wave boundary, and it is open**.

Nothing in this repository claimed this rung passed. `README.md` claimed 1, 2
and 3 strips, and that claim is corrected there — it was written of the recycled
ring, which failed at one.

### 5.4 Not run, with the reason

| gate | why not |
| --- | --- |
| physical NPU, any rung | no NPU on this host. Every result here is CAModel and none is a hardware claim. The flag semantics in §3 are the first thing to re-measure on silicon, in both sub-block directions |
| commit bisect to `bdb479b` or earlier | would localise *when*, not *what*. The policy A/B (§3.1) and the data-free probe (§3.3) localise the mechanism, which was the open question. The defect is at least as old as the recycled default |
| full-count soak, `-DASCEND_MICRO_SPARSE_SAMPLING=OFF` | the bf16 probe has no sampling predicate at all — every block of every rung above ran |
| 48, 64, 128 strips | blocked behind §5.3: a wider grid is more waves, and wave 1 is already wrong |
| 33 strips on the mode-4 single-shot leg | **stopped by the operator** after ~35 minutes against a ~3-minute mode-2 reference for the same rung. No verdict: it had not reached the comparison. It was being run to show the wave defect predates this change, and §5.3's role census shows that discipline-independently instead, so it was not restarted |
| `scripts/run_micro_layout_bf16_camodel.sh` | the 32x32 Nz question was ruled out by §3.4 without it. Worth running before touching the table ingress for §5.3 |
| `tests/research/kernel_cluster_sync_probe.cpp:120` fence finding | pre-existing and out of scope for this change; the file is untouched. It is the one remaining `FAIL` in `run_reshape_wave_host_gate.sh` |
| mode-2 leg of the cluster sync probe | commit `719a390e` measured it at four iterations in both directions and the decode operator has shipped on it since. Not re-measured here |

## 6. Measurement defects found on the way

Three, each of which made a gate pass vacuously or fail unconditionally.

**`analyze_reshape_trace.py`'s order check could not pass.** It built its
timeline from the **popped** log, so a wait's tick was its ISSUE tick. A wait
that blocks is issued before the announcement that releases it, so the check
reported "WAIT released with no outstanding announcement" on every run,
including the bit-exact single-shot one. A check no correct run can pass is not
a gate. It is replaced by the census and the channel-shape check, and the script
now prints the `audit_mode4_tick_order.py` command and says in as many words
that order is not checked there. That sibling script derives the release tick
from the scalar drain behind the wait, has the quorum rule for mode 2, and
reports an unmeasurable stall as `no-fence` rather than scoring it.

**`analyze_reshape_trace.py` was mode-4 only**, so against the mode-2 kernel it
printed `0 ids in the trace` and then failed check 3 for the wrong reason. It
reads `SET_CROSS_CORE` / `WAIT_FLAG_DEV` now, unpacks `GetffstMsg`'s
`1 + (mode << 4) + (flagId << 8)`, and never pools the two id spaces.

**Its check 5 had never passed on the BF16 instance.** It expected `CHUNKS`
GM→L1 bursts and `CHUNKS + 1` MTE2 data moves — the INT8 instance's flat table
copy. The BF16 ingress stages the 16 resident blocks one at a time, because at
C0 = 16 a row-major 32x32 block is not its own Nz image, so the real count is
`CHUNKS * strips + 16`. Nobody had seen it, because the order check aborted the
script first.

The harness's three-chunk break is gone (§2), and the runner now names the
cross-core discipline explicitly in both directions, so the shipped default can
change without the script quietly dispatching the other one.

## 7. Commands

```sh
# container with the persistent ext4 work root bind-mounted at /work
docker run -d --name bf16keyfix -v /home/ladislav/.cache/ascend-micro:/work \
    -w /work/bf16keyfix/src \
    ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 sleep infinity
docker exec bf16keyfix bash -lc 'apt-get update -qq && apt-get install -y -qq rsync'

# the gate, and the ladder
BF16_BUILD_DIR=/work/bf16keyfix/b5 BF16_SLICES=1 bash scripts/run_reshape_bf16_camodel.sh
BF16_BUILD_DIR=/work/bf16keyfix/b5 BF16_SLICES="2 3" bash scripts/run_reshape_bf16_camodel.sh run-only
BF16_BUILD_DIR=/work/bf16keyfix/b5 BF16_SLICES=8 BF16_STRIPS_PER_BLOCK=8 \
    bash scripts/run_reshape_bf16_camodel.sh run-only
BF16_BUILD_DIR=/work/bf16keyfix/b5 BF16_SYNC=mode4-single-shot BF16_SLICES="1 3" \
    bash scripts/run_reshape_bf16_camodel.sh run-only

# the audits
python3 scripts/audit_cross_core_fences.py
bash scripts/run_reshape_wave_host_gate.sh
python3 scripts/audit_mode4_tick_order.py /work/bf16keyfix/b5/s1_x1_mode2 --block 0 --cross
D=/work/bf16keyfix/b5/s8_x8_mode2
python3 scripts/analyze_reshape_trace.py --strips-per-block 8 \
    $D/core0.veccore0.instr_popped_log.dump $D/core0.veccore0.instr_log.dump \
    $D/core0.veccore1.instr_popped_log.dump $D/core0.veccore1.instr_log.dump \
    $D/core0.cubecore0.instr_popped_log.dump $D/core0.cubecore0.instr_log.dump

# the refutation, with no data in flight
bash scripts/run_cluster_sync_probe_camodel.sh 4 4
```

## 8. What is left

1. **The wave-boundary defect, §5.3.** Open. A grid whose block count is not a
   multiple of the physical AI Core count places one grid block's Vector work
   on two sub-block-1 cores and no sub-block-0 core, so four of the eight heads
   of that strip are never written. Start from `GetSubBlockIdx()` in wave 1 and
   from how the runtime places a MIX_AIC_1_2 block's three sub-cores when the
   last wave is partial; a kernel-side workaround would have to derive the role
   from something other than the sub-block index, which the whole UB arena and
   the Fixpipe's `subBlockId` also depend on. Until it is fixed, dispatch grids
   that are a whole number of waves, or use `tokensPerCore` to make them so —
   which is now possible, because a block can stream several strips (§4.2).
2. **The Cube's epilogue quorum drain, §5.2.** Not verified, because the rung
   that would verify it is §5.3. It needs either §5.3 fixed or a different
   construction whose release is measurable on a non-empty queue.
3. **The INT8 instance.** Still on mode 4. Its default is the single-shot
   allocation now, which cannot take this fault, but the same per-chunk
   per-core channel shape means it cannot stream strips either. Porting it to
   mode 2 is the same change as §4.1.
4. **The vendored copy.**
   `kernels/ops/turboquant/vllm_integration_layer/op_kernel/common/turboquant_wave_protocol.hpp`
   is a separate file with its own `kReshapeDefaultReuse = true`, and
   `turboquant_reshape_batch.cpp` selects on it. It is untouched here — another
   session is working in that tree — and it still defaults to the refuted ring.
