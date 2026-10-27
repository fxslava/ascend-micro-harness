#!/usr/bin/env bash
# Build and run the single-tile BF16 operand-layout probe on CAModel.
#
# One AIC, one 16-token x 256-channel page, eight 32 x 32 resident blocks,
# sixteen Mmads, four launches. No Vector core and no ring, so the run is a
# few seconds of model time on top of the model's fixed start-up cost, and a
# mismatch cannot be a synchronisation artefact.
#
# There is no NPU on this host. A pass means "executes on the cycle-accurate
# model and matches the audited golden", not a hardware claim.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${MICRO_BUILD_DIR:-$(mktemp -d /tmp/ascend-micro-layout-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"

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
        echo "Use a fresh MICRO_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    cmake -S "${repo}" -B "${build}" -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build}" --target test_micro_layout_bf16 -j4
fi

dir="${build}/run"
mkdir -p "${dir}"
(
    cd "${dir}"
    if ! timeout "${MICRO_TIMEOUT:-600s}" "${build}/lib/test_micro_layout_bf16" \
            > run.log 2>&1; then
        cat run.log >&2
        echo "FAILED" >&2
        exit 1
    fi
    grep -E 'PASS|FAIL|Total tick|Model RUN TIME' run.log || true
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|su_ccu_mpu_err' run.log; then
        echo "CAModel errors in ${dir}" >&2
        exit 1
    fi
)
echo "micro layout bf16 build and run directory: ${build}"
