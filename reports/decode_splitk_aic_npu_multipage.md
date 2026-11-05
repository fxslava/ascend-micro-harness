# Physical-NPU coverage for the multi-page L1 reuse hazard

Target: CANN 9.2.0-beta.2, `dav-3510`, `Ascend950PR_9599`.

**THERE IS NO NPU ON THIS HOST.** `npu-smi` and `/dev/davinci*` are both
absent. The physical-device leg of everything below was **built and not run**,
and no statement in this report is a hardware claim. What was executed here ran
either on the host CPU or on CAModel over a collapsed grid.

This adds device-side regression coverage for the hazard
`reports/decode_splitk_aic_page_race.md` diagnosed and
`reports/decode_splitk_aic_strip_pipeline.md` closed: at four or more carried
pages AIV1 restaged the L1 key page for sequence page `p + 1` over page `p`'s
before the Cube's MTE1 had read it.

## 1. What the existing ladder already covered

Worth stating first, because the task that prompted this assumed the device
tests were single-tile canary isolations. **They are not.**
`tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp` (CMake
target `test_npu_decode_splitk_aic`) already walked:

* page counts 1, 3, 4, 5, 7 and 8 against splits 1, 2, 4 and 8, each also with
  `page_begin = 1`, so `pages >= 4` at `K = 1` was already reached;
* 2 requests x 2 heads x 16 query rows x 256 channels;
* a seven-rung unaligned-context tail ladder;
* `m`, `l`, `acc` and bit-exact `qrot` against the independent FP64 oracle
  `splitk_decode_test::ReferenceAicPartial`, through the shared
  `tq_splitk_rig::AicRig::Validate`;
* a **non-identity** block table: the fixture's reverse permutation,
  `physical = request * pages + pages - 1 - page`.

So the gap was never "no multi-page coverage". It was five narrower things.

## 2. The five gaps, and what closed each

| # | gap | closed by |
| --- | --- | --- |
| 1 | `strips = 1` (`block_size = 16`) was unreachable — the ladder hardcodes `kPageTokens = 128`. This is the geometry the CAModel sweep used to expose the defect, and only the benchmark CLI reached it; a benchmark is not a regression test. | two new rungs at `block_size = 16` |
| 2 | No ULP metric anywhere. The bounds are relative: 2e-6 on `m`, 1e-4 on `l`, `2e-4*scale + 1e-5*l + 1e-6` on `acc`. | `UlpOf` / `UlpDistance` in the rig, with opt-in caps. The accumulator needed a second, scale-referenced form -- see 2b. |
| 3 | The block table permutation is MONOTONE, so a kernel that walks its pages backwards reads a correct-looking sequence out of it. | `PageMap::kScattered` in the fixture |
| 4 | No canary scan on the non-empty path. The 0xAA fill spells the float -3.0316488e-13, which is finite and passes the 1e-6 additive floor, so a canary word in a near-zero channel was invisible. | per-word canary and nonfinite census over the whole record |
| 5 | `pages >= 4` at `K = 1` existed only as two coordinates of a cross product, named nowhere, so a future narrowing of the ladder would drop the hazard's geometry silently. | a named `l1-reuse` section |

### 2a. The new section

Four launches, at `requests = 1, heads = 1, splits = 1` — so the **entire** page
window is carried by one physical core and every page transition contends the
same L1 key ring. At 2 requests and 2 heads the work spreads over four blocks
and any one core carries fewer consecutive pages, which is the opposite of the
pressure the defect needed.

| rung | block_size | strips/page | pages | total strips |
| --- | ---: | ---: | ---: | ---: |
| one strip per page, four carried | 16 | 1 | 4 | 4 |
| one strip per page, eight carried | 16 | 1 | 8 | 8 |
| full pages, four carried | 128 | 8 | 4 | 32 |
| full pages, eight carried | 128 | 8 | 8 | 64 |

The `block_size = 16` rungs matter for a specific reason: at one strip per page
every strip lands in ring slot 0, so **a page boundary and a slot wrap coincide
on every page** rather than every eighth one. Each rung goes through a scattered
block table, and the section asserts that table is a genuine permutation and is
neither the identity nor the reverse before it launches — otherwise the rung is
the old ladder under a new name.

### 2b. The ULP budgets are taken from measurement

`worst_state_ulp <= 64` and an accumulator budget of 256. On values of order 1,
where one FP32 ULP is 1.1920929e-07,
`reports/decode_splitk_aic_strip_pipeline.md` section 5 measured for this
revision:

| quantity | measured | in ULP |
| --- | ---: | ---: |
| `acc`, mean \|deviation\|, 2/4/8-page rungs | 4.35e-08 … 6.21e-08 | ~0.4 … 0.5 |
| `m` | bit-exact | 0 |
| `l`, worst rung | ~1.2e-05 at l ~ 90 | ~1.6 |
| **the defect's own signature** | **1.022830e-03** | **~8600** |

So 64 and 256 sit two to three orders of magnitude **above** the measured noise
— enough that another device's rounding cannot false-alarm — and one to two
orders **below** the defect's signature. A budget that did not separate those
two numbers would not be worth asserting.

**These are budgets for hardware that has not run them.** If a device run trips
one, the failure names the row, channel, value and reference.

#### Which accumulator ULP the 256 holds, and why the obvious one does not work

This is a correction to the first version of this change, and the CAModel run
in section 5 is what forced it.

The obvious reading of `worst_acc_ulp <= 256` — and the one originally
implemented — is the deviation over the FP32 spacing **at that channel's own
value**. Running the rungs produced:

| rung | total strips | worst per-channel ULP | \|got − ref\| there | \|ref\| there |
| --- | ---: | ---: | ---: | ---: |
| one strip per page, four carried | 4 | 48 | 1.78e-07 | 0.0577 |
| one strip per page, eight carried | 8 | 98 | 4.54e-08 | 0.00624 |
| full pages, four carried | 32 | 247 | 1.15e-07 | 0.00397 |
| full pages, eight carried | 64 | **1327** | 7.73e-08 | 0.000642 |

The deviations do not grow — 1.78e-07, 4.54e-08, 1.15e-07, 7.73e-08, all around
the 4.35e-08…6.21e-08 the table above predicts. What grows is the ULP *number*,
because the unluckiest of 4096 channels keeps landing closer to zero. A row is
a 256-channel sum **with cancellation**, so some channel always nearly cancels,
and nothing bounds how nearly. The metric was measuring channel magnitude, not
kernel error: at 32 strips it had already reached 247 of its 256 budget, and at
64 strips a **correct** kernel measured 1327, which would have failed that
budget by 5.2x.

The host oracle confirms this directly and without a device. For each rung,
the row accumulation scale against the smallest \|acc\| any channel reaches,
and the resulting amplification `ulp(scale) / ulp(min |acc|)` — the factor by
which a per-channel ULP exceeds a scale-referenced one for the *same* absolute
deviation:

| rung | row scale | min \|acc\| | amplification | measured per-channel ULP |
| --- | ---: | ---: | ---: | ---: |
| block=16 pages=4 | 1.1496 | 0.0586743 | 32x | 48 |
| block=16 pages=8 | 1.20752 | 0.0286948 | 64x | 98 |
| block=128 pages=4 | 1.03564 | 0.00397417 | 256x | 247 |
| block=128 pages=8 | 3.03663 | 0.000642073 | **4096x** | **1327** |

The amplification column tracks the measured ULP, and on each line the
`min |acc|` channel the host oracle names **is the very channel** the device
reported: `ref=-0.00397416547` for row three, `ref=0.000642073414` for row
four. The host measurement was made before the fourth rung had ever executed
and predicted it at 4096x; it then measured 1327, failing the requested 256
budget by 5.2x **while computing correctly.**

So the raw per-channel number is reported (`margins.acc_ulp`, printed on every
launch) and **not capped** — it is unbounded above for a bit-correct kernel, so
no value of it is both safe and tight. The 256 is held against
`margins.acc_scaled_ulp`: the same deviation over the spacing at the **row's
accumulation scale**, the magnitude the arithmetic actually worked at, which is
what the summation error is bounded by.

This keeps the gate's sensitivity to the defect intact. The defect's
1.022830e-03 against a scale of order 1 is still about 8600 scaled ULP, so it
trips a 256 budget by ~34x. Measured, the four rungs read 1.77, 2.67, 4.86 and
5.04 scaled ULP, so the budget holds about fifty times headroom over a correct
kernel and still fires an order of magnitude below the defect. What changed is
only the false-failure mode.

That the amplification *stays* large is now itself a host assertion, so the
rationale cannot go stale silently — see section 4.

### 2c. The caps are opt-in, deliberately

`state_ulp_cap` and `acc_ulp_cap` default to 0, meaning "report, do not
enforce". The existing ladder and `bench_turboquant_decode_splitk` quote their
margins as fractions of the RELATIVE bounds; holding them to a ULP cap they
were never written for would be a change of gate rather than an addition. Only
a rung that sets a cap is held to one. The ULP numbers are printed either way.

`acc_ulp_cap` exists and works, so the assertion as originally requested is
available to a caller who wants it; no rung sets it, for the reason in 2b.

The canary and nonfinite censuses are NOT opt-in: a published partition holding
a word no worker wrote is a defect under any tolerance.

That does mean the pre-existing ladder is now held to a check it was not held
to before, on a device nothing here can run. The risk of a FALSE positive is
that a legitimately computed channel lands bit-identically on -3.0316488e-13,
which for an FP32 `sum_t p_t * vs_t * V[c][t]` is not a value the arithmetic
produces unless the channel was never written — so a fire would be a true
positive. It is called out because strengthening a gate one cannot run is a
decision, not a detail.

## 3. Gate status

| gate | state |
| --- | --- |
| `test_npu_decode_splitk_aic` builds, `-Wall -Wextra -Werror` | **PASS** — zero errors, zero warnings |
| `test_npu_decode_splitk_aiv` builds (shared rig, other leg) | **PASS** — zero errors, zero warnings |
| host probe `test_aic_splitk_decode_probe`, existing claims | **PASS** — unchanged by the fixture edit |
| host probe, new page-map section | **PASS** — section 4 |
| host probe, new ULP-amplification section | **PASS** — section 4 |
| CAModel twin, `--only l1-reuse`, all four rungs | **PASS** — section 5 |
| **the ladder on physical hardware** | **NOT RUN — no NPU on this host** |
| full `test_npu_decode_splitk_aic` ladder on CAModel | **NOT RUN** — about ninety launches on the model; the `--only` selector exists because of that |

## 4. The host gates: the scattered mapping, and the ULP amplification

`PageMap` moves WHERE a logical token is stored without changing WHAT it holds:
the table entry and the data offsets both derive from `physical`, while the
content is keyed by the LOGICAL page. So the FP64 oracle's output must be
bit-identical under both mappings, and that equality is the self-consistency
check — a permutation that broke the table/data correspondence would move the
numbers.

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I include tests/cpu/test_aic_splitk_decode_probe.cpp -o /tmp/probe && /tmp/probe
```

```text
PASS: 16 distinct M rows; uneven/empty temporal splits; multiple requests/heads;
nonzero page window; N=1024,4096,16384 K=1,4,16 full-context combines.
PASS page maps: 2 geometries where the scattered block table is a permutation,
is neither the identity nor the reverse, and leaves the FP64 oracle
bit-identical to the default mapping
PASS acc ULP amplification: 4 L1-reuse geometries where the worst row amplifies
a per-channel ULP distance at least 16x over a scale-referenced one, so only
the scaled metric can carry a budget
```

This is the only property of the scattered mapping establishable without a
device. What the mapping is FOR — making a kernel that ignores the block table
read unrelated data rather than the neighbouring page — needs the kernel.

The third line is the amplification gate from 2b. It asserts the worst row of
each of the four L1-reuse geometries amplifies a per-channel ULP distance at
least 16x over a scale-referenced one (measured: 32x, 64x, 256x, 4096x), which
pins the premise the rigs' scaled-ULP choice rests on. If a future fixture
change made these accumulators well conditioned, this fires and says the raw
metric is usable again.

## 5. CAModel execution of the new rungs

`test_decode_splitk_aic_sim` over a collapsed grid. **A CAModel result is not a
hardware claim** — it is a cycle-accurate model of one, and it is the only leg
on this host that executes a rung at all.

Two runs. The first used the per-channel accumulator ULP as the enforced
metric; it is the evidence that forced the correction in 2b, so it is recorded
rather than discarded.

### Run 1 — per-channel metric, three of four rungs

Started 2026-11-02 20:33:15, killed by the script's 5400 s `timeout`. Wall
clock 5753 s, the excess being the model flushing ~800 MB of per-core
instruction dumps on shutdown. Verbatim, with the driver-stub and
`AicWrapper` noise stripped:

```text
AIC partial profile: acc over tolerance 0 of 4096 channels in 0 of 16 rows; m over 0, l over 0; replicas torn 0; qrot wrong 0; mean signed deviation -6.688996e-10, mean |deviation| 6.208090e-08
AIC partial ULP: worst_state_ulp=1.00 (l, row 1) worst_acc_ulp=48.00 (row 15, c 4, got=-0.0577439666 ref=-0.0577441446); canary words 0, nonfinite words 0; budgets enforced
  ok  L1 reuse: block_size=16  strips=1 pages=4 K=1 table=[1,0,3,2]           one strip per page, four carried
AIC partial profile: acc over tolerance 0 of 4096 channels in 0 of 16 rows; m over 0, l over 0; replicas torn 0; qrot wrong 0; mean signed deviation 7.450344e-12, mean |deviation| 4.350898e-08
AIC partial ULP: worst_state_ulp=2.00 (l, row 0) worst_acc_ulp=98.00 (row 8, c 8, got=-0.00624020398 ref=-0.0062402494); canary words 0, nonfinite words 0; budgets enforced
  ok  L1 reuse: block_size=16  strips=1 pages=8 K=1 table=[3,5,2,7,1,0,4,6]   one strip per page, eight carried
AIC partial profile: acc over tolerance 0 of 4096 channels in 0 of 16 rows; m over 0, l over 0; replicas torn 0; qrot wrong 0; mean signed deviation 2.981686e-10, mean |deviation| 7.936511e-08
AIC partial ULP: worst_state_ulp=2.00 (l, row 6) worst_acc_ulp=247.00 (row 3, c 8, got=-0.00397405028 ref=-0.00397416547); canary words 0, nonfinite words 0; budgets enforced
  ok  L1 reuse: block_size=128 strips=8 pages=4 K=1 table=[1,0,3,2]           full pages, four carried
```

What this establishes, for the three rungs that ran:

* the scattered block tables are what the rungs need — `[1,0,3,2]` and
  `[3,5,2,7,1,0,4,6]`, neither identity nor reverse, asserted before launch;
* `acc over tolerance 0 of 4096`, `m over 0`, `l over 0`, `replicas torn 0`,
  `qrot wrong 0` against the FP64 oracle;
* **`canary words 0, nonfinite words 0`** — the new census, on the real
  multi-page path, which is the check the 0xAA-fill blind spot (gap 4) needed;
* `worst_state_ulp` 1.00, 2.00, 2.00 against the 64 budget, so the state ULP
  budget has ~32x headroom and `budgets enforced` confirms the caps were live;
* all 32 `*excp_log*.dump` files zero-length.

It did **not** establish the fourth rung, and it is the run that showed
`worst_acc_ulp` to be the wrong thing to cap.

### Run 2 — scale-referenced metric, all four rungs

Re-run after the 2b correction with a 21600 s timeout. Started 2026-11-03
09:42:26, finished 11:52:43, 7816 s. **All four rungs completed**, and the
exception dumps are again all zero-length.

```text
AIC partial ULP: worst_state_ulp=1.00 (l, row 1) worst_acc_ulp=48.00 (row 15, c 4, got=-0.0577439666 ref=-0.0577441446); canary words 0, nonfinite words 0; budgets enforced
AIC partial scaled ULP: worst_acc_scaled_ulp=1.77 of 256 (row 3, c 11, got=0.630742371 ref=0.63074216, row scale=1.14729651); min |ref| over 4096 channels = 0.0481908891
  ok  L1 reuse: block_size=16  strips=1 pages=4 K=1 table=[1,0,3,2]           one strip per page, four carried
AIC partial ULP: worst_state_ulp=2.00 (l, row 0) worst_acc_ulp=98.00 (row 8, c 8, got=-0.00624020398 ref=-0.0062402494); canary words 0, nonfinite words 0; budgets enforced
AIC partial scaled ULP: worst_acc_scaled_ulp=2.67 of 256 (row 1, c 2, got=0.101167709 ref=0.101167868, row scale=0.808601838); min |ref| over 4096 channels = 0.0062402494
  ok  L1 reuse: block_size=16  strips=1 pages=8 K=1 table=[3,5,2,7,1,0,4,6]   one strip per page, eight carried
AIC partial ULP: worst_state_ulp=2.00 (l, row 6) worst_acc_ulp=247.00 (row 3, c 8, got=-0.00397405028 ref=-0.00397416547); canary words 0, nonfinite words 0; budgets enforced
AIC partial scaled ULP: worst_acc_scaled_ulp=4.86 of 256 (row 8, c 10, got=-0.0119167864 ref=-0.0119166415, row scale=0.385875697); min |ref| over 4096 channels = 0.00397416547
  ok  L1 reuse: block_size=128 strips=8 pages=4 K=1 table=[1,0,3,2]           full pages, four carried
AIC partial ULP: worst_state_ulp=5.00 (l, row 7) worst_acc_ulp=1327.00 (row 6, c 7, got=0.00064215064 ref=0.000642073414); canary words 0, nonfinite words 0; budgets enforced
AIC partial scaled ULP: worst_acc_scaled_ulp=5.04 of 256 (row 1, c 6, got=1.90900445 ref=1.90900505, row scale=1.90900505); min |ref| over 4096 channels = 0.000642073414
  ok  L1 reuse: block_size=128 strips=8 pages=8 K=1 table=[3,5,2,7,1,0,4,6]   full pages, eight carried
SECTIONS RUN: ladder=NO tail=NO l1-reuse=yes  -- NARROWED, this is not the full gate
PASS AIC multi-page L1 reuse regression: 4 launches at requests=1 heads=1 K=1 -- one core carrying every page -- over block_size 128 and 16 at 4 and 8 carried pages, each through a scattered block table; worst_state_ulp 5.00 of a 64 budget, worst_acc_scaled_ulp 5.04 of a 256 budget, zero canary words and zero nonfinite words in any published partition (worst per-channel acc ULP 1327.00, reported not capped)
```

So, over all four rungs, on the model:

| quantity | worst over the four rungs | budget |
| --- | ---: | ---: |
| `worst_state_ulp` | 5.00 | 64 |
| `worst_acc_scaled_ulp` | 5.04 | 256 |
| `worst_acc_ulp` (reported, uncapped) | 1327 | — |
| canary words | 0 | 0 |
| nonfinite words | 0 | 0 |
| `acc over tolerance` | 0 of 4096 | 0 |
| `m over` / `l over` / replicas torn / qrot wrong | 0 | 0 |

Rung 4 is the confirmation that mattered: at 1327 per-channel ULP it would have
failed the originally requested 256 cap, and at 5.04 scaled ULP it passes with
fifty times headroom. Both numbers come from the same launch of the same
correct kernel.

### A wart this run exposed, now fixed

The narrowed run printed `PASS AIC split-K decode validation: 0 launches …` and
the same for the tail ladder — a PASS line over a section that did not run,
which is the silent coverage loss gap 5 exists to prevent. A section that ran
nothing now prints `SKIPPED … not selected, 0 launches` instead. The output
above predates that fix, which is why those two PASS lines are not quoted.

### What no run here establishes

Nothing has executed on hardware. CAModel is a cycle-accurate model of a 950PR,
not a 950PR.

## 6. Commands

Everything below is copy-pasteable. `ASCEND_HOME_PATH` defaults to
`/usr/local/Ascend/cann-9.2.0-beta.2` and `SOC_VERSION` to
`Ascend950PR_9599`; override either in the environment.

One naming trap: this harness keys off **`ASCEND_HOME_PATH`**, not
`ASCEND_TOOLKIT_HOME`. CANN's own `set_env.sh` exports `ASCEND_TOOLKIT_HOME`,
and setting only that leaves the build pointed at the default path. If the
toolkit is installed somewhere else, set `ASCEND_HOME_PATH`.

### On a physical 950PR host — the deliverable

```bash
bash scripts/run_decode_splitk_npu_validation.sh all
```

That builds both validation binaries and runs them. It **refuses** to run
without a device rather than failing inside `aclrtSetDevice`, and says what it
checked. To build only:

```bash
bash scripts/run_decode_splitk_npu_validation.sh build
```

To run an already-built tree, or just the new section:

```bash
bash scripts/run_decode_splitk_npu_validation.sh run --only l1-reuse
```

The script sources `$ASCEND_HOME_PATH/set_env.sh` and sets `LD_LIBRARY_PATH` to
`<build>/lib:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib` itself, so no
environment setup is needed beyond `ASCEND_HOME_PATH` if the toolkit is not at
the default path.

By hand, without the script:

```bash
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}/set_env.sh"
```

```bash
cmake -S . -B build-npu -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend950PR_9599 -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && cmake --build build-npu --target test_npu_decode_splitk_aic -j 4
```

```bash
LD_LIBRARY_PATH="$PWD/build-npu/lib:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/devlib:${LD_LIBRARY_PATH}" ./build-npu/lib/test_npu_decode_splitk_aic
```

Two build traps apply on the WSL development host and **not** on a native Linux
NPU host; the script handles both, and by-hand builds there should too. Build
out of the WSL ext4 filesystem rather than `/mnt/c`, because the drvfs page
cache can hand BiSheng a stale copy of a source a Windows-side editor has just
rewritten. And always use a FRESH build directory, because CANN's preprocess
stage links each device object over itself and a second pass in the same
directory dies with `unknown file type`.

### On a host with no NPU

```bash
bash scripts/run_decode_splitk_npu_validation.sh sim --only l1-reuse
```

That is the CAModel twin, `test_decode_splitk_aic_sim`, which is the only way
to execute a rung here. Drop `--only l1-reuse` for the whole ladder and expect
hours. The host probe:

```bash
bash scripts/run_splitk_workspace_build.sh
```

## 7. Follow-ups

* **Nothing has run on hardware.** That is the whole of the remaining risk for
  this change, and the first command in section 6 is what closes it.
* **The full ladder has never run on CAModel either**, on this or any earlier
  revision — about ninety launches on the model. `reports/decode_splitk_validation.md`
  records which rungs have been executed on which leg.
* The `--only` selector narrows by section, not by rung, so re-running one rung
  means paying for the others again — the full four-rung section is about
  2 h 10 min on the model. A `--only l1-reuse:N` form would make the expensive
  rung reachable on its own.
* A narrowed run now prints `SKIPPED` rather than `PASS` for the sections it
  did not run, but nothing yet *prevents* a CI job from narrowing. If this
  ladder is wired into CI, the job should assert the unnarrowed form.
* `ASCEND_MICRO_DECODE_SPLITK_SIM` is defined for both simulator twins by
  `cmake/SimTargets.cmake` and read by nothing. It predates `--only` and now has
  no job; worth removing or wiring up.
