# Ingestion audit table — 2026-10-30

`tests/npu/bench/bench_turboquant_reshape_and_cache.cpp` now aggregates its
measurements with the shared host-only `tests/npu/turboquant_audit_report.hpp`
renderer (`PrintTableA`). It emits the requested nine-column Markdown table
after the existing per-leg timing output. Each warm/pressure measurement is
its own row; the note identifies the mode and recycled/single-shot policy.

As confirmed for this task, model names are explicit **labels for the existing
4-KV-head × 256-channel fixture**, not model-specific network geometries.
Repeated model, context, batch and chunk options form a Cartesian sweep.
Explicit horizons allocate per-request caches and place each request's chunk
at its final aligned pages. Only `B * chunk_tokens` activation rows are timed.
The default preserves the previous effective 2,048-token run and fragmented
mapping. The kernel arithmetic and synchronization are unchanged.

## Running on hardware

Load the CANN environment, then build:

```bash
cmake -S . -B build -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
cmake --build build --target bench_turboquant_reshape_and_cache -j4
```

For example, with a working Ascend950PR device:

```bash
export ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV=reports/ingest_prefill.csv
./build/lib/bench_turboquant_reshape_and_cache \
    --model Qwen3.5-9B --model DeepSeek-V4-Flash --model GLM-5.2-744B \
    --context 8192,65536,262144,1048576 --batch 1,2 \
    --chunk-tokens 1024 --warmups 50 --samples 500
```

Use `--skip-native` to omit the native leg. `--single` selects zero warmups and
one sample per leg, mode and sweep case. Existing `--tokens`, `--strips` and
mode-4 policy options remain available; repeated token/strip arguments now
produce rows for every supplied chunk instead of selecting only the last.
Contexts and chunks must be 16-aligned, `chunk_tokens <= S`, and the total
ingested chunk must remain within the existing 65,536-token fixture bound.
Slot addresses must fit INT32. Large `B*S` sweeps allocate full horizon caches,
including uncompressed BF16 native caches when that leg is enabled.

## Metrics

* Tiers use the inclusive boundaries 8,192, 65,536 and 262,144.
* The Cube cell is `median_cube (median_vector / min_cube)` in microseconds.
  Even-size sample sets use the average of their two middle values.
* Speedups are `native_median / cube_median (native_median / vector_median)`.
  Missing, nonpositive or nonfinite timing values produce `N/A`; no speedup
  is inferred when the native measurement is absent.
* Write bandwidth is `(packed_K_bytes + packed_V_bytes + scale_bytes) /
  (cube_median_us * 1000)`, in decimal GB/s. It excludes input reads and
  unwritten cache capacity. Warm measurements may be served by L2, so the note
  calls it effective cache-write bandwidth. The original per-leg console
  report still uses input-plus-output traffic.
* The Vector throughput gain is `(vector_median / cube_median - 1) * 100%`.
  "Crosses native V5 speed" is printed only when native/Cube exceeds one.

The optional native leg calls CANN's
[`aclnnScatterPaKvCache`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnScatterPaKvCache.md)
with BF16 `[N_chunk, 4, 256]` inputs, ND caches and the same slot mapping as
TurboQuant. Tensor/workspace preparation is outside timing; a repeatable
executor times only the stage-2 device launch. CMake discovers its header and
operator/base libraries. They are loaded lazily for the native leg and kept
for the process lifetime. Missing symbols or native preparation/measurement
failures produce `N/A` with a reason; a failed stream synchronization remains
a benchmark failure. `-DASCEND_MICRO_NATIVE_INGEST=OFF` builds without this leg.
These are different cache representations: native writes BF16, while TQ also
rotates/quantizes and writes INT4 plus scales.

## CSV

`ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV` names the append destination. The first nine
headers and formatted values match the Markdown columns exactly. Additional
columns retain the mode, measurement source, chunk size, numeric token count,
channel policy, A8/Cube/Vector/native medians, Cube minimum, both numeric
speedups, output and total traffic bytes, write GB/s and Vector throughput
gain. Commas, quotes and line breaks are CSV-escaped. A header is written once;
an incompatible existing header is rejected rather than mixing schemas.

The generated `reports/ingest_audit/report_fixture.csv` and
`environment_fixture.csv` contain **synthetic reporting-test data only**,
explicitly marked `measurement_source=synthetic_fixture`. They are not NPU
performance results. Their matching rendered table is in
[`ingest_audit/example_table.md`](ingest_audit/example_table.md).

## Verification

```powershell
wsl -- bash scripts/run_ingest_audit_checks.sh
```

This builds the requested target and `test_ingest_audit_report`, compiles the
native-unavailable path against the host ACL declarations, runs the reporting
test binary and checks both direct and environment-driven CSV output using
Python's CSV reader. Logs are under `reports/ingest_audit/`.

| Gate | Result |
| --- | --- |
| Requested CMake benchmark target | PASS, CANN 9.2.0 beta.2 / GNU 11.4.0, `-Wall -Wextra -Werror`, native API code enabled; `build.log` |
| Native-unavailable host compilation | PASS with `-Wall -Wextra -Werror`; `native_unavailable_compile.log` |
| Shared table runtime | PASS: four tiers, comma formatting, triplet order, ratio direction, write-byte units, missing/zero/negative/NaN/infinite timing guards; `report_fixture.log` |
| Sweep/geometry checks | PASS: labels, contexts, batches/chunks, default/legacy flags, overflow/alignment rejection, final chunk slot placement at a 1M horizon |
| CSV append and escaping | PASS: one header, same nine formatted columns, raw metrics, quote/comma escaping, missing final newline and incompatible-header rejection |
| CSV environment export | PASS: `ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV` and Python CSV-reader checks; `csv_check.log` |
| Physical CLI / timing run | UNAVAILABLE as described below; `help.log`, `startup_debug.log` |

The mounted Windows build directory emitted Make clock-skew warnings (about
0.2 seconds) for generated dependency/object timestamps. The build exited 0;
the final benchmark source was compiled and linked with the required strict
compiler flags. No compiler warning was suppressed for this reporting change.

No `/dev/davinci0` is available on this host. A physical-binary `--help` attempt
aborted before `main` with `basic_string::_S_construct null not valid`; the
loader trace's final library initialization was `libruntime_v100.so`. It also
failed with optional operator libraries unloaded. The physical CLI/sampling
gate is therefore unavailable here; the verification script records it and
executes the shared renderer in the independent host test instead. No actual
BF16-Cube, Vector or native V5 latency/speedup claims are made. CAModel kernel
execution, physical correctness and physical performance were not run.
