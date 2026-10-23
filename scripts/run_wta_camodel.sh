#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${WTA_BUILD_DIR:-$(mktemp -d /tmp/ascend-wta-camodel-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"
set +u
source "${toolkit}/set_env.sh"
set -u
export LD_LIBRARY_PATH="${build}/lib:${sim}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
if [[ "${1:-all}" != "run-only" ]]; then
    if [[ -f "${build}/CMakeCache.txt" ]]; then
        echo "Use a fresh WTA_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    cmake -S "${repo}" -B "${build}" -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build}" --target test_camodel_wta_fixpipe test_wta_convergence_probe -j4
    "${build}/test_wta_convergence_probe" | tee "${build}/tier1.log"
fi
for sign in ${WTA_SIGNS:-1 0}; do
    for steps in ${WTA_STEPS:-1 2 3 4 5 6 7 8}; do
        for margin in ${WTA_MARGINS:-0.01 0.1 0.5 1.0}; do
            dir="${build}/t${steps}_sign${sign}_margin${margin}"
            mkdir -p "${dir}"
            (
                cd "${dir}"
                if ! timeout "${WTA_TIMEOUT:-60s}" "${build}/test_camodel_wta_fixpipe" \
                    "${steps}" "${sign}" 13 "${margin}" > run.log 2>&1; then
                    cat run.log >&2
                    exit 1
                fi
                grep -E 'PASS|FAIL|Total tick|Model RUN TIME' run.log
                if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|su_ccu_mpu_err' run.log; then
                    echo "CAModel errors in ${dir}" >&2; exit 1
                fi
                python3 "${repo}/scripts/analyze_wta_trace.py" --steps "${steps}" . | tee audit.txt
            )
        done
    done
done
echo "WTA build and trace directory: ${build}"
