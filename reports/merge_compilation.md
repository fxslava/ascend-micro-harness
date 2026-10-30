# Merge and compilation repair — 2026-10-30

The merge of `135dcb2506afed23d8c354b1d98e9192432823b4` into
`d440a7864be46e0eabdc9c28437a8f46383100b9` registered the cluster-sync probe
twice. The explicit conflicts were in `CMakeLists.txt` and
`cmake/NpuTargets.cmake`; the automatic merge also duplicated the simulator
library and executable in `cmake/SimTargets.cmake`.

Each mode now registers one probe library and one executable. The simulator
runner retains `ASCEND_MICRO_PROBE_SIM`, simulator linkage and RPATH, and the
NPU aggregate retains the incoming split-K and unaligned-context validation
targets. The consolidated probes retain the existing strict compiler options,
including the CANN missing-braces exception. No kernel arithmetic or pipeline
code was changed as part of this repair.

Windows CRLF line endings also prevented the shell host gate from starting in
WSL (`set: pipefail\r: invalid option name`). Shell scripts now have an LF Git
attribute; the build helper and the incoming reshape gate/runner scripts were
normalized to LF.

A rebuild of the completed tree exposed a CANN 9.2 legacy-toolkit issue:
`merge_obj_text.sh` links the preprocessing AIC/AIV objects in place, turning
ELF relocatable objects (`ET_REL`) into executables (`ET_EXEC`). On the next
build its always-run link step rejects those files with `ld.lld: error: ...
unknown file type`. `cmake/CannRebuild.cmake` now adds a preparation step to
each legacy preprocessing project. `scripts/prepare_cann_preprocess.py`
removes only those consumed core objects so CANN recompiles them. It preserves
relocatable objects, host objects, preprocessing text and source files, and
rejects paths resolving outside the preprocessing build directory. The
toolkit installation is unchanged; source/header dependency checks remain
enabled. Rebuilds incur recompilation of those preprocessing core objects.

## Reproduce

From PowerShell in the repository root:

```powershell
wsl -- bash scripts/run_merge_build_checks.sh
wsl -- bash -lc 'set -o pipefail; bash scripts/run_reshape_wave_host_gate.sh 2>&1 | tee reports/merge_compilation/host_gate.log'
wsl -- python3 tests/cpu/test_cann_preprocess_rebuild.py -v
```

The script loads `/usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh`, creates a
fresh ext4 build directory, configures `Release` for `Ascend950PR_9599`, builds
all default simulator and NPU targets with four jobs. The second command runs
the existing host gate. Pass a build directory as the first argument to reuse it; set
`ASCEND_HOME_PATH` or `BUILD_JOBS` to override the toolkit or parallelism.

Logs are in `reports/merge_compilation/` (`*.log` are Git-ignored).

## Validation

Toolkit: CANN 9.2.0 beta.2 in the default Ubuntu 22.04 WSL distribution;
host compiler: GNU 11.4.0. Build directory:
`/tmp/ascend-merge-build-RXDBmA`, binaries under its `lib/` directory.

| Gate | Result |
| --- | --- |
| Resolve and stage both conflicted paths | PASS; no unmerged index entries |
| Configure and build all simulator and NPU targets from a fresh tree | PASS; `clean_configure.log`, `clean_build.log` |
| Rebuild the existing tree before the CANN workaround | FAIL, reproduced `ld.lld: unknown file type` as described above |
| Rebuild the existing tree with the CANN workaround | PASS, exit 0, all default targets; preparation steps ran for all 28 preprocessing projects; `build.log` |
| Rebuild helper controls | PASS, 3 tests: selective removal/preservation and idempotence, fresh tree, outside-tree symlink rejection; `rebuild_helper.log` |
| Existing host gate | PASS; `host_gate.log` |
| Final AIC/AIV device symbol audit | PASS, 39 device objects, no unavailable objects and no remaining `turboquant_wave` symbols; `device_symbols.log` |

The host gate includes ten strict harness syntax checks, protocol/header and
two-translation-unit linkage checks, cross-core fence audit, the A8 arithmetic
probe, 4,198,401 tail-geometry pairs, the eleven-rung masked oracle, the
two-wave protocol model and the 21,738-state decode ring model with negative
controls. These are host checks, not CAModel execution.

The final symbol audit used the completed device-output directories only,
excluding preprocessing files named `.o` that actually contain text:

```bash
python3 scripts/audit_device_symbols.py \
    /tmp/ascend-merge-build-RXDBmA/*_aic_device_dir \
    /tmp/ascend-merge-build-RXDBmA/*_aiv_device_dir
```

Use the new build's printed directory when reproducing this check.

## Runtime limits

This repair checks build integration and host contracts. It does not execute
the CAModel boundary canaries, depth/wrap ladders, multi-worker/wave ladders,
or trace census, and it does not launch kernels on physical hardware. Building
simulator and NPU binaries is not evidence that those runtime gates pass.
The incoming operator pipeline changes therefore still require the baseline
in `REFINEMENT_OF_COMBAT_KERNELS.md` before pushing them or tagging hardware.
