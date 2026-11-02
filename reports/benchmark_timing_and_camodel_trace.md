# Benchmark timing and native CAModel trace export

Environment: WSL Ubuntu 22.04, CANN 9.2.0-beta.2, Ascend950PR_9599.
This change affects host measurement and profiling tools. Operator arithmetic,
pipeline synchronization and device buffer layouts are unchanged.

## Measurement changes

`bench_turboquant_decode_splitk.cpp` validates exactly once per implementation,
context and K, after the final sample is synchronized. Warmups and measured
iterations perform no downloads or CPU reference checks. Single-shot mode still
forces warmup=0, iters=1, and never inserts an implicit K=1 launch. Its launch is
followed by event recording/synchronization before validation.

Latency percentiles use all measured samples. Tail/export counters now describe
workers in the final validated launch: median across those workers normally,
maximum in single-shot mode. They no longer pool worker counters across samples.

Normal sweeps establish K=1 for each selected implementation. The added
`aic_speedup` CSV field is appended, preserving the positions of existing fields.
AIC speedup is its K=1 raw batch time divided by its current raw batch time;
the 16-row normalization cancels. AIV and AIC keep independent baselines.
Unavailable/zero device timing and absent implementations remain `NA` deliberately;
division is guarded by finite, positive measured latency. CAModel does not
provide ACL event latency, so its speedup fields remain unavailable.

Each context owns its fixtures and each implementation/K scope owns its device
rig. Their existing vector/Buffer destructors release host and device storage
before the next context. Added constructor error cleanup releases the first
event when the second creation fails, finalizes ACL/resets the device when
stream creation fails, and frees an unexpectedly unaligned device allocation.

## Supported export mechanism

The installed CLI documents `msprof op simulator`, including `--application`,
`--export`, `--output`, `--soc-version`, `--kernel-name`, `--launch-count` and
mandatory `PipeUtilization`. Its timeout unit is **minutes**. This is the
toolkit's official collection-and-conversion entry point; a separate STARS
JSON converter is unnecessary. The installation also has
`tools/msopprof/bin/msopprof`. `msprof-analyze` is not the instruction timeline
export path used here.

The [official simulator guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)
describes native instruction timeline export and Chrome viewing. The installed
9.2 CLI help, captured alongside the result, determines the flags used by this
repository; newer documents may use `msopprof simulator` as the command name.

Inspection of the installed `config_stars.json` found `stars.ffts_mode`,
`model_top.{sim_type,num_aic,num_aiv}` and `pem.{parsim,parsim_thd_limit}`.
`config.json` contains LOG, L2CACHE and WRAPPER settings. Neither shipped file
contains an enable/format=perfetto option. No such speculative key is added.
The earlier `/tmp/a8-prof/profile_a8.sh` instead forced internal switches with
an LD_PRELOAD shim and observed empty standalone STARS profiling TOMLs.
That standalone flusher is unnecessary for the official msprof export flow.

`scripts/profile_camodel_trace.sh` creates a private fresh Linux workspace,
copies source without stale simulator logs/dumps, builds both benchmark targets,
copies the shipped configs (or a supplied CAMODEL_CONFIG_PATH directory), and
uses one simulator library family without an inherited LD_PRELOAD shim. It
invokes msprof for one AIC launch using the shared production rig. The default
is one supported 16-token diagnostic page/strip, M=16, d=256, one request/head,
K=1. A full production-size page is available through explicit overrides.

The script checks the profiler's exit status, application CPU-reference PASS
and exact launch count, clean model completion (`Total tick`), and one aggregate
`OPPROF_*/simulator/trace.json`. It validates that JSON and copies it byte-for-byte
to the output directory. CANN's own parser may consume internal simulator dumps;
the repository implements no dump-to-timeline regex parser or timestamp synthesis.
An empty/unmatched export cannot pass, and existing output directories are
refused so an old trace cannot satisfy a new run.

The validator requires finite nonnegative timestamps/durations, execution with
positive duration, non-empty phase/metadata, per-lane metadata, a Cube core and
two Vector cores, and execution in MTE1/MTE2/MTE3/CUBE/FIXP/Vector-family lanes.
CANN uses string `pid`/`tid`; pipe categories are **tid lanes**. `cat` is optional
and is used primarily for cross-core flows. SIMD/SIMT Vector work may appear as
RVECEX/RVECLD/RVECST and related lanes rather than a single VECTOR lane.
The exporter preserves these native names.

## Reproduce

From the repository in WSL:

```bash
bash scripts/run_decode_benchmark_host_checks.sh
python3 scripts/test_validate_camodel_trace.py
bash scripts/profile_camodel_trace.sh reports/my_camodel_trace
python3 scripts/check_camodel_trace_perfetto.py reports/my_camodel_trace/trace.json --download
```

The final command fetches Google's official trace_processor launcher if
`--download` is supplied. To use an existing installation instead, pass
`--trace-processor /path/to/trace_processor`. It imports the native JSON through
Perfetto's engine and checks the resulting core/pipe tracks, rather than only
checking JSON syntax. See [Perfetto's trace processor documentation](https://perfetto.dev/docs/analysis/trace-processor).
Open the resulting `trace.json` using “Open trace file” in https://ui.perfetto.dev.

Optional full-page collection (slower):

```bash
CAMODEL_TRACE_PAGE_TOKENS=128 CAMODEL_TRACE_CONTEXT=128 \
CAMODEL_TRACE_TIMEOUT_MINUTES=15 \
bash scripts/profile_camodel_trace.sh reports/my_full_page_trace
```

Outputs include source/binary/config hashes, actual command, CMake compiler
flags, build/collection logs, native trace path, trace checksum, and JSON
validation summary. `CAMODEL_TRACE_KEEP_WORK=1` retains native profiler output
and the build for investigation; failed runs always retain their workspace.
Successful default runs remove the temporary build and simulator dumps.

## Verification and limits

Both benchmark targets compiled with `-Wall -Wextra -Werror`. Host regression
checks passed for actual benchmark measurement ordering/count, single-shot
synchronization, zero timing, AIC-only speedup/CSV arithmetic, scoped buffers
across context sizes, and event/session constructor failure cleanup. Both the
hardware-event and simulator preprocessor branches were exercised with ACL test
doubles. Four JSON validator test methods cover native string lanes and reject
invalid timestamps/phases/durations, empty traces, and missing cores/pipes/metadata.
These host checks are not device timing evidence.

The initial 128-token profiling attempt reached the five-minute application
timeout and was interrupted. Editing the script while that first shell run was
active also disrupted its final status handling. Its exploratory logs are in
`camodel_trace_verification/`; it is not an accepted trace result. The finalized
script was then run in a fresh workspace with the minimal default geometry.

The finalized minimal workflow exited **0**. The application printed
`PASS: CPU references matched. Kernel launches: AIV=0 AIC=1.`, followed by
`Total tick: 28249` and `Model stopped successfully.` Both benchmark targets
built successfully from the final source; their saved compiler flag files
confirm `-O3 -DNDEBUG -Wall -Wextra -Werror -std=gnu++17`.

The [native trace](camodel_trace_minimal/trace.json) contains **25,871 events**:
75 metadata (`M`) and 25,796 complete (`X`) events, of which 24,592 have positive
duration. Event start timestamps span approximately 0.386667–15.456667 us.
The [validation census](camodel_trace_minimal/trace-validation.json) records
execution on `core0.cubecore0`, `core0.veccore0` and `core0.veccore1`, including
MTE1, MTE2, MTE3, CUBE, FIXP and Vector RVEC lanes. This short trace contains
no `cat` flow events; it has all required execution lanes and phase metadata.
SHA-256: `76876b2172d0bd735b8f40f398a0fd7a4e3d7b56cf7d55083582274a8ad47a83`.

Google's actual Perfetto trace processor imported that JSON successfully;
the [SQL result](camodel_trace_minimal/perfetto-import.txt) confirms named
core/pipe tracks and positive-duration slices without any JSON rewriting.
The successful run also removed its temporary build/dump workspace.
Perfetto reports 17,120 `slice_spill_overlapping_complete_event` diagnostics:
native instruction intervals overlap on their pipe tracks, so it places those
slices on overflow tracks merged into the thread at display time. Its diagnostic
explicitly states that no data is lost. See the saved
[import diagnostics](camodel_trace_minimal/perfetto-import.stderr.txt); the native
JSON is retained rather than reshaping instruction intervals to suppress them.

The native profiler warns that source-line/call-stack information is absent
from the release kernel build. Instruction lanes and timestamps were exported
and imported successfully; adding debug line information is separate from
this timing/trace workflow. The installed CLI also warns that `--application`
will be deprecated in a future release; it is supported by this CANN version.

No physical NPU performance measurement is claimed. The minimal collection is
one worker across its AIC and two AIV cores; it does not establish multi-worker
scaling, page-depth or ring-wrap correctness. No production pipeline refactor
was made, and the pipeline-refinement verification protocol is not replaced by
this profiling smoke check.
