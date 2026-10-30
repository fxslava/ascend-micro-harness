#!/usr/bin/env bash
# Build the physical benchmark; exercise its shared reporting code on the host.
# No physical kernels are launched by this script.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${1:-${repo}/build}"
logs="${repo}/reports/ingest_audit"
out="$(mktemp -d /tmp/ascend-ingest-audit-XXXXXX)"
mkdir -p "${logs}"
cd "${repo}"
set +u
source "${toolkit}/set_env.sh"
set -u
cmake -S . -B "${build}" -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION=Ascend950PR_9599 -DASCEND_HOME_PATH="${toolkit}" \
    2>&1 | tee "${logs}/configure.log"
cmake --build "${build}" --target bench_turboquant_reshape_and_cache \
    test_ingest_audit_report -j "${BUILD_JOBS:-4}" 2>&1 | tee "${logs}/build.log"
g++ -std=c++17 -Wall -Wextra -Werror -Itests/host_stubs \
    -Ikernels/ops/turboquant/baseline -fsyntax-only \
    tests/npu/bench/bench_turboquant_reshape_and_cache.cpp \
    2>&1 | tee "${logs}/native_unavailable_compile.log"
if [[ -e /dev/davinci0 ]]; then
    "${build}/lib/bench_turboquant_reshape_and_cache" --help \
        2>&1 | tee "${logs}/help.log"
    "${build}/lib/bench_turboquant_reshape_and_cache" --skip-native --help \
        > "${logs}/skip_native_help.log" 2>&1
else
    # This CANN physical runtime aborts in its library initializer before main
    # on this device-less host. Do not mistake a host formatter test for a run
    # of the hardware benchmark, or repeatedly launch the failing initializer.
    printf '%s\n' 'UNAVAILABLE physical benchmark CLI: no /dev/davinci0; shared formatter runtime is tested below' \
        | tee "${logs}/help.log"
fi
ASCEND_BENCH_TQ_AUDIT_PREFILL_CSV="${out}/from-env.csv" \
    "${build}/lib/test_ingest_audit_report" "${out}/prefill.csv" --check-env \
    2>&1 | tee "${logs}/report_fixture.log"
cp "${out}/prefill.csv" "${logs}/report_fixture.csv"
cp "${out}/from-env.csv" "${logs}/environment_fixture.csv"
cp "${logs}/report_fixture.log" "${logs}/example_table.md"
python3 - "${out}" <<'PY' | tee "${logs}/csv_check.log"
import csv
from pathlib import Path
import sys

expected = ["Tier", "Model", "Context (S)", "Batch (B)", "Tokens Ingested",
            "TQ Ingest BF16-Cube (μs)", "Native V5 Ingest (μs)",
            "Speedup vs V5", "Throughput / Note"]
for name, count in (("prefill.csv", 5), ("from-env.csv", 4)):
    with (Path(sys.argv[1]) / name).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[:9] == expected
        rows = list(reader)
    assert len(rows) == count
    assert rows[0]["Tokens Ingested"] == "2,048"
    assert rows[0]["tokens_ingested"] == "2048"
    assert rows[0]["measurement_source"] == "synthetic_fixture"
    assert rows[0]["TQ Ingest BF16-Cube (μs)"] == "10.000 (20.000 / 8.000)"
    assert rows[0]["speedup_cube_vs_v5"] == "4.000"
    assert rows[0]["speedup_vector_vs_v5"] == "2.000"
    assert rows[0]["write_gbps"] == "20.480"
    assert rows[1]["Model"] == 'Model, "A"'
    assert rows[1]["Native V5 Ingest (μs)"] == "N/A"
    assert rows[1]["Speedup vs V5"] == "N/A"
    assert rows[3]["Tier"] == "Extreme"
    assert rows[3]["Context (S)"] == "1,048,576"
    assert all(None not in row for row in rows), "CSV field count mismatch"
    print("PASS", name, "exact headings, numeric metrics, missing native, escaping, append")
PY
