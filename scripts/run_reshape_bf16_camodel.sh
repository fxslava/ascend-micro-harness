#!/usr/bin/env bash
# Build and run the TurboQuant BF16 prefill reshape-and-cache probe on CAModel
# over the inductive strip set, the same ladder the INT8 instance uses:
# 1 strip (cold start), 2 strips (two grid blocks, each addressing its own
# strip and its own physical page) and 3 strips (one interior step, which on a
# model with fewer AI Cores than strips also reuses a physical core and so
# catches a flag or token leaked by strip 0).
#
# There is no NPU on this host, so this is the verification gate. A pass means
# "executes on the cycle-accurate model over S = {0, 1, 2} and matches the
# audited golden", not a hardware claim.
#
# Wider contexts need --window, because a grid wider than the AI Core count
# does not dispatch and one strip per block per launch is what the mode-4 flag
# id budget allows:
#
#   BF16_SLICES="1 2 3 32" BF16_WINDOW=32 scripts/run_reshape_bf16_camodel.sh
#
# Unlike the INT8 probe this kernel has no sparse boundary sampling: its grids
# are small enough that every block runs.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${BF16_BUILD_DIR:-$(mktemp -d /tmp/ascend-bf16-camodel-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"
slices="${BF16_SLICES:-1 2 3}"
window="${BF16_WINDOW:-0}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN 9.2 toolkit not found: ${toolkit}" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${build}/lib:${sim}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

if [[ "${1:-all}" != "run-only" ]]; then
    if [[ -f "${build}/CMakeCache.txt" ]]; then
        echo "Use a fresh BF16_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    cmake -S "${repo}" -B "${build}" -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build}" \
        --target test_camodel_turboquant_reshape_and_cache_bf16 -j4
fi

for n in ${slices}; do
    dir="${build}/s${n}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        args=(--slices "${n}")
        if [[ "${window}" != "0" ]]; then args+=(--window "${window}"); fi
        if ! timeout "${BF16_TIMEOUT:-1800s}" \
                "${build}/lib/test_camodel_turboquant_reshape_and_cache_bf16" \
                "${args[@]}" > run.log 2>&1; then
            cat run.log >&2
            echo "FAILED at ${n} slice(s)" >&2
            exit 1
        fi
        grep -E 'PASS|FAIL|Total tick|Model RUN TIME' run.log || true
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|su_ccu_mpu_err' run.log; then
            echo "CAModel errors in ${dir}" >&2
            exit 1
        fi
    )
done
echo "BF16 CAModel build and run directory: ${build}"
