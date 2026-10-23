#!/usr/bin/env bash
# Build and run the inverse-CDF int4x2 quantization probes on CAModel:
# Tier 1 host probe, then the 4-tile (ring boundary) and 8-tile (V3, wraps
# the depth-2 ring twice) CAModel runs, each with the error-pattern gate and
# the trace audit (engine census, vector utilization, per-instruction cycle
# costs, SET/WAIT_FLAG balance).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${ICDF_BUILD_DIR:-$(mktemp -d /tmp/ascend-icdf-camodel-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"
set +u
source "${toolkit}/set_env.sh"
set -u
export LD_LIBRARY_PATH="${build}/lib:${sim}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

if [[ "${1:-all}" != "run-only" ]]; then
    if [[ -f "${build}/CMakeCache.txt" ]]; then
        echo "Use a fresh ICDF_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    cmake -S "${repo}" -B "${build}" -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build}" --target test_icdf_quantize_probe \
        test_camodel_icdf_quantize test_camodel_icdf_quantize_8tiles -j4
    "${build}/test_icdf_quantize_probe" | tee "${build}/tier1.log"
fi

for case in icdf_quantize:4 icdf_quantize_8tiles:8; do
    name="${case%%:*}"
    tiles="${case##*:}"
    dir="${build}/t${tiles}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        if ! timeout "${ICDF_TIMEOUT:-180s}" "${build}/test_camodel_${name}" > run.log 2>&1; then
            cat run.log >&2
            exit 1
        fi
        grep -E 'PASS|FAIL|Total tick|Model RUN TIME' run.log
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|su_ccu_mpu_err' run.log; then
            echo "CAModel errors in ${dir}" >&2; exit 1
        fi
        python3 "${repo}/scripts/analyze_icdf_trace.py" --tiles "${tiles}" . | tee audit.txt
    )
done
echo "ICDF build and trace directory: ${build}"
