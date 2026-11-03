# Fused decode three-zone schedule: CAModel verification (2026-11-04)

Independent re-verification of commit `6b38819` "[test new approach]" (merged as
`d936d46`), the three-zone schedule refactor of
`TurboQuantFusedDecodeOpt`, against `REFINEMENT_OF_COMBAT_KERNELS.md`.

This run supersedes the "Device build and CAModel ladder: pending completion"
line in `reports/fused_decode_three_zone_20261103.md`: the device build and the
boundary/depth/wrap CAModel gates were executed here and are recorded below.

One gate-tooling defect was found and **fixed** in this session
(`scripts/analyze_fused_decode_opt_trace.py`, see "Gate tooling fix"); the
stored traces were then re-audited with the fixed auditor and all eleven
completed cases pass. The schedule itself needed no change.

Scope was reduced mid-run on instruction: the comprehensive edge-case sweep
(repeated launches, Split-K workers, empty workers, unmapped pages, varied
scores) is **deferred**, not passed. Nothing was re-run that already passed.

## Verification environment

No physical NPU is present on this host; every device result below is CAModel.

- Native WSL2 `Ubuntu 22.04.5 LTS`, kernel `6.18.33.2-microsoft-standard-WSL2`,
  12 logical CPUs. No Docker container was used.
- CANN `9.2.0-beta.2`, `Ascend-cann-toolkit`, inner version
  `V100R001C12B086`, arch `x86_64`, at `/usr/local/Ascend/cann-9.2.0-beta.2`.
- BiSheng `clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)`,
  build stamp `2026-08-18T19:12:48+08:00`.
- `SOC_VERSION=Ascend950PR_9599`; the model loaded
  `simulator/dav_3510/lib/config.json`. Note the driver stub reports
  `halGetSocVersion soc_version:Ascend950PR_9589` — recorded as observed, not
  reconciled.
- Source/build/evidence root on **ext4**, not the Windows mount:
  `/home/dmin/work/three-zone-check-20261103` (`/dev/sdd`, ext4).
- Source staged with `rsync` from `/mnt/c/Ascend/ascend-micro-harness` at
  working-tree state `d936d46` plus the three pre-existing untracked
  `reports/camodel_trace_*` directories. The Windows working tree was
  not modified by this run (verified with `git status --porcelain`: only
  those three untracked directories, as at session start).

### Commands

```bash
# host gates
g++ -std=c++17 -O2 -Wall -Wextra -Werror tests/cpu/test_fused_decode_opt_contract.cpp -o host-contract && ./host-contract
python3 -u tests/cpu/test_fused_decode_opt_model.py
python3 -u tests/sim/kernel/test_three_zone_schedule.py
python3 -u tests/cpu/test_fused_decode_opt_trace.py
python3 -u scripts/audit_cross_core_fences.py

# device build
cmake -S src -B build -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2 -DSOC_VERSION=Ascend950PR_9599
cmake --build build --target test_camodel_fused_decode_opt turboquant_fused_decode_opt_npu -j8

# per case (trace audit included)
bash scripts/run_fused_decode_opt_case.sh "$root" <tag> [--canary P [--tile T] | --tiles N [--tail-rows R]]
```

Per-case `command.txt`, `result.txt` and `audit.json`, plus
`ladder-summary.txt` (run-time result, pre-fix auditor),
`reaudit-summary.txt` (post-fix audit of the same traces) and `reproduce.sh`
are committed under `reports/fused_decode_three_zone_camodel_20261104/`.

Two classes of evidence are deliberately **not** committed: the gate `*.log`
files, because `.gitignore:10` excludes `*.log` (their counts and verdicts are
quoted inline in this report instead), and the raw `*.dump` traces at
~64 MB/case. Both remain in the ext4 evidence root above, which is outside the
repository and not preserved by this commit.

## Gate results

### §1 Token/queue model — PASS

`tests/cpu/test_fused_decode_opt_model.py`: capacity-one exploration, N=0..16,
two consecutive tasks, 221 → 1,214,429 reachable states. Negative controls all
rejected: `early_refill` (ingress overwrite before scale consumer),
`wrong_window` (score window overwrite), `duplicate` (double SET loses token),
`unprimed` (deadlock).

`tests/sim/kernel/test_three_zone_schedule.py`: PASS — the actual production
C++ scheduling methods, compiled at `-Wall -Wextra -Werror`, N=1..130, all six
residues, no boundary branches in either steady phase body.

### §2 Compile-time memory checks — PASS

Host contract: `PASS launch guards, NZ bijection (1..4 heads), shared
recurrence phases; UB=142304+256 bytes` — within the
`kArenaEnd + 256 <= kUbWindow <= 192*1024` assertion.

Device compile of the real kernel: clean. Both `test_camodel_fused_decode_opt`
and the production `turboquant_fused_decode_opt_npu` device object built under
bisheng `-Wall -Werror` with **zero** compiler diagnostics (the only `warning:`
hits in `build.log` are CMake's own policy warnings).

### §3 Isolated logical-page boundary canaries — PASS (4/4)

One tile of the real 256-channel geometry, full cache/table allocations,
non-identity block table `p -> 2p+1`, nonzero logical pages. Every case reports
`comparison=exact inputs=unchanged guards=0x5A-intact`.

| Case | Selector | Ticks |
| --- | --- | ---: |
| `iso-page0` | `--canary 0` | 17,705 |
| `iso-page-mid` | `--canary 4` (K/2, K=8) | 17,835 |
| `iso-page-last` | `--canary 7` (K-1) | 17,880 |
| `iso-page-last-strip` | `--canary 7 --tile 1` | 17,876 |

### §4 Depth and wrap — PASS (7/7 run), remainder deferred

| Case | Tiles | Ticks | Zone coverage |
| --- | ---: | ---: | --- |
| `depth-n3` | 3 | 32,762 | short bounded path (`numTiles < kSlots+1`) |
| `deep-first-n4` | 4 | 41,248 | **minimal deep path**, first slot swap |
| `wrap-n6` | 6 | 56,858 | full ring wrap |
| `odd-n7` | 7 | 69,350 | odd residue |
| `period-exact-n9` | 9 | 71,329 | prologue + exactly one Period-6 + drain |
| `fulldepth-n12` | 12 | 95,671 | prologue + full Period-6 + residue-3 steady tail + drain |
| `tail-rows-n16` | 16 | 120,461 | two full periods + residue-1 tail, partial last tile (17 of 64 rows) |

`deep-first-n4` is the minimal tile count that saturates the deep path: the
kernel selects it at `numTiles >= kSlots + 1 = 4`. It was already verified
before the scope reduction and was not re-run.

**Zone arithmetic for the headline case.** With `kSlots=3`, `kPhases=6`,
`steady_limit = numTiles - 3`:

- N=9 → `steady_limit=6`: one full unrolled Period-6 body, residue 0, then the
  3-step drain. This is "prologue + exactly one period + drain".
- N=12 → `steady_limit=9`: one full unrolled Period-6 body, **plus** a
  statically indexed residue-3 steady tail (phases 0,1,2 re-entered for tiles
  6,7,8), then `RunCubeDrain<3>` / `RunVectorDrain<3>` for tiles 9,10,11.

N=12 is therefore a strict superset of N=9: it exercises a complete period and
a ragged partial re-entry after it, which is the analogue of §6.5's rung 33.
Both were run and both pass.

### §5 Trace audit and instruction census — PASS (11/11 completed cases)

Per case the auditor requires real traces, empty exception dumps, equal
dispatch/completion flag counts, capacity-one markings in completion order,
zero final marking, per-id tick order, and non-vacuous Vector work.
96 `instr_popped_log.dump` files per case (32 blocks × cube + 2 vector);
**zero nonempty exception dumps in every case**.

#### Handshake census at N=12 matches the design exactly

Mode-2 channels, `fulldepth-n12`. AIC→AIV sets reach both AIVs and AIV→AIC
needs a two-AIV quorum, so cube-side set counts are 2× the handshake count.

| Channel | Role | Handshakes | Expected |
| --- | --- | ---: | --- |
| `m2:0/1/2` | SlotReady[s] | 4+4+4 = **12** | N = 12 |
| `m2:6/7` | ScoresReady[w] | 6+6 = **12** | N = 12 |
| `m2:9` | ProbsReady | **12** | N = 12 |
| `m2:8` | ContextReady | **12** | N = 12 |
| `m2:3/4/5` | SlotFree[s] | 3+3+**4** = **10** | N−2 = 10 |
| `m2:10` | task join | 2 | 1 task, both directions |

The asymmetry in `SlotFree` is the load-bearing observation: slots 0 and 1
carry 3 credits each while **slot 2 carries 4**. That extra credit is exactly
the new initial free credit for L1 slot 2 that this refactor introduces,
published by Cube before score 0 — consumed once, and never re-emitted by the
three drain steps. Recycled credits are `context k -> stage k+3` for
`k = 0..N-4`, i.e. 9, giving 9 + 1 = 10 = N−2.

#### The same census at N=16 confirms the credit accounting scales

`tail-rows-n16`, after the gate fix below: SlotReady `m2:0/1/2` = 6+5+5 = **16**,
ScoresReady `m2:6/7` = 8+8 = **16**, ProbsReady `m2:9` = **16**,
ContextReady `m2:8` = **16**, SlotFree `m2:3/4/5` = 5+4+**5** = **14** = N−2.

Predicted independently: recycled credits are `k = 0..N-4` = 13, distributing
as slot 0 ×5, slot 1 ×4, slot 2 ×4, plus the one initial slot-2 credit → 5/4/5.
Observed exactly. Slot 2 again carries the extra credit.

All channel ids are within the legal mode-4/mode-2 space (0..10, task join 10);
no new user id is introduced. Kernel-owned HardEvent ids observed are 0..3,
inside the legal 0..5, at both N=12 and N=16.

Per-id tick order was checked by the auditor (`audit_stalls`) and passes: on
every vector-core channel each producer SET precedes its consumer WAIT. No
`cann_global_barrier_announcements` were emitted in this configuration.

## Gate tooling fix

### 1. Trace auditor rejected a legal compiler-owned event direction (FIXED)

As first observed, `tail-rows-n16` (`--tiles 16 --tail-rows 17`) — **the kernel
passed, the auditor crashed.** The run itself reported:

```
PASS fused_decode_opt tiles=16 tail_rows=17 ... comparison=exact inputs=unchanged guards=0x5A-intact
[INFO] Total tick: 120461    [INFO] Model stopped successfully.
```

then `scripts/analyze_fused_decode_opt_trace.py` aborted:

```
ValueError: invalid/reserved user flag route ('SCALAR', 'VEC', 7)
```

`analyze_fused_decode_opt_trace.py:23` allow-lists the reserved BiSheng event
around `ArithProgression` VF calls as exactly one route,
`compiler_event = key == ('VEC', 'SCALAR', 7)`. At 16 tiles BiSheng also emits
the **opposite** direction of that same reserved pair. Measured on the N=16
trace, per vector core:

| Route | SET | WAIT | Capacity-one | Final marking |
| --- | ---: | ---: | --- | ---: |
| `('VEC','SCALAR',7)` | 3 | 3 | yes | 0 |
| `('SCALAR','VEC',7)` | 1 | 1 | yes | 0 |

At N=12 only `('VEC','SCALAR',7)` ×2 appears, which is why shorter cases pass.
Both id-7 routes are balanced, capacity-one and drained; no kernel event uses
id 7; no exception dumps. So this was an **auditor false negative, not a kernel
defect**.

**Fix applied.** `scripts/analyze_fused_decode_opt_trace.py` now allows the
reserved id on either direction of that one VEC/SCALAR pair:

```python
compiler_event = key in (('VEC', 'SCALAR', 7), ('SCALAR', 'VEC', 7))
```

This is deliberately the narrowest change that admits the observed compiler
behaviour. It does **not** relax the gate in any other respect:

- the id-7 routes stay inside `places()`, so each is still held to
  capacity-one, SET/WAIT balance and zero final marking;
- no other pipe pair may carry id 7;
- no other reserved id is admitted on any route;
- the dispatch-vs-completion equality check is untouched.

Four negative/positive controls were added to
`tests/cpu/test_fused_decode_opt_trace.py` to pin that boundary:
`test_compiler_reserved_reverse_route` (the new direction is accepted),
`test_compiler_reserved_reverse_route_other_pipe_rejected` (`SCALAR->MTE2` id 7
rejected), `test_other_reserved_id_still_rejected` (id 11 on the same pipe pair
rejected), and `test_compiler_reserved_route_leak_rejected` (on both id-7
routes, a dangling SET, a double SET and an unmatched WAIT are each still
rejected). The suite runs 9 tests, all passing.

**Re-audit of the stored traces with the fixed auditor** — no new simulation
was run; the existing `*.instr_popped_log.dump` sets were re-audited in place:

| Case | Audit |
| --- | --- |
| `iso-page0`, `iso-page-mid`, `iso-page-last`, `iso-page-last-strip` | PASS |
| `depth-n3`, `deep-first-n4`, `wrap-n6`, `odd-n7` | PASS |
| `period-exact-n9`, `fulldepth-n12` | PASS |
| `tail-rows-n16` | **PASS** (was the failure) |

No previously passing case regressed, and the N=16 handshake census recorded
above was obtained from that now-passing audit.

## Open finding

### The 2026-11-03 report claims an N=16 case it cannot have audited

`reports/fused_decode_three_zone_20261103.md` names `Tlast (N=16 with only 17
valid rows in the final tile)` as one of its four verified cases, and states
"Tmid/Tlast execute all intermediate tiles". With the auditor **as committed in
`6b38819`**, the N=16 trace audit aborts as above. That case's trace audit
therefore cannot have passed in the state the commit records, and that report's
own "Device build and CAModel ladder: pending completion" line is the accurate
one. The case does pass now, but only with the auditor fixed in this session —
so the earlier report's claim was not supported by its own tooling at the time
it was written. Flagged rather than edited; `6b38819`'s report is left as the
historical record.

## Deferred, not passed

Stopped mid-flight or never started, by instruction to conserve simulation
time. None of these is evidence either way:

- `repeat-launch-n4x2` (`--tiles 4 --heads 17 --repeats 2`) — interrupted
  in flight when the ladder was stopped. Its `run.log` has no
  `Model stopped successfully` and no runner `PASS` line, and its traces are
  truncated mid-stream, so re-auditing it reports `nonempty final marking`.
  That is an artifact of the SIGTERM, **not** a finding about the kernel — and
  it is incidental evidence that the lifetime audit still rejects a dangling
  token after the allow-list change. The case must simply be re-run.
- `empty-task` (`--empty 1`), `workers-n16-k4` (`--tiles 16 --splits 4`),
  `empty-worker-n6-k4` (`--tiles 6 --splits 4`),
  `unmapped-hole-n12` (`--tiles 12 --hole 2`),
  `varied-n9` / `varied-fulldepth-n12` (`--varied 1`) — never started.

Consequently **§4's repeated-launch and real/empty Split-K worker clauses are
unmet**, and no varied-score arithmetic regression was run in this session.
Per §4, "Do not infer the tail-worker merge path from a single-worker probe":
every case above ran `splits=1`.

## Limitations

- No physical NPU on this host. Every device number here is CAModel over
  `dav_3510`; none of it is a silicon claim.
- Simulator ticks are not silicon throughput. The tick counts above are
  recorded for scaling and comparison only.
- No overlap claim is made. Engine census per core is recorded, but
  intersecting MTE2/Vector intervals were not extracted, so §5's overlap
  evidence is absent by choice.
- §6 (inter-wave MIX_AIC_1_2 gates) was not exercised: every case here ran a
  single grid block (`grid=1`, `splits=1`), so no second wave occurs and the
  wave-boundary gates are untested in this session.
- The host completion model abstracts fenced GEMM/Fixpipe and stage/MTE3
  operations; it is not CAModel evidence, and the CAModel audit lacks a direct
  observation of every scalar wait release.
- Nothing was pushed, committed or tagged for hardware by this run.
