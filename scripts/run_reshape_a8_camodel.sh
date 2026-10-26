#!/usr/bin/env bash
# Build and run the TurboQuant A8 prefill reshape-and-cache probe on CAModel
# over the inductive strip set: 1 strip (cold start, the only grid this kernel
# had ever run at), 2 strips (the induction step -- two grid blocks, each one's
# Vector cores must address its own strip and its own physical page) and 3
# strips (one interior step, which on a model with fewer than three AI Cores
# also reuses a physical core and so catches a flag or token leaked by strip 0).
#
# There is no NPU on this host, so this is the verification gate for multi-block
# addressing. A pass means "executes on the cycle-accurate model over
# S = {0, 1, 2} and matches the audited golden", not a hardware claim.
#
# The full device shapes are affordable here and are worth running before any
# claim, not just the collapsed set: measured cost on this model is a ~45 s
# fixed cost plus ~8 s of wall clock per *executed* strip, so
#
#   A8_SLICES="1 2 3 32 64" scripts/run_reshape_a8_camodel.sh
#
# covers 512 and 1024 tokens as well. 64 strips is the one grid that exceeds
# the model's 32 AI Cores and so runs a second execution wave, re-setting every
# mode-4 flag id on a physical core that has already used them.
#
# The build defaults to compile-time sparse boundary sampling
# (tests/common/turboquant_sampling.hpp): above a wave's worth of blocks, only
# the first four, the four straddling the midpoint and the last four of each
# 32-block wave execute, and the runner requires every skipped block's page to
# still hold its canary. That is 24 of 64 blocks, so the 1024-token grid costs
# about what 24 strips cost. For the full-count soak, which is the stronger
# gate and the one to run before any claim that leaves this workstation:
#
#   A8_CMAKE_ARGS=-DASCEND_MICRO_SPARSE_SAMPLING=OFF \
#   A8_SLICES="32 64" scripts/run_reshape_a8_camodel.sh
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${A8_BUILD_DIR:-$(mktemp -d /tmp/ascend-a8-camodel-XXXXXX)}"
sim="${toolkit}/tools/simulator/Ascend950PR_9599/lib"
slices="${A8_SLICES:-1 2 3}"

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
        echo "Use a fresh A8_BUILD_DIR: CANN preprocessing mutates its object files; use run-only for an existing build." >&2
        exit 2
    fi
    # shellcheck disable=SC2086  # A8_CMAKE_ARGS is a deliberate word list
    cmake -S "${repo}" -B "${build}" -DSOC_VERSION=Ascend950PR_9599 \
        -DASCEND_HOME_PATH="${toolkit}" ${A8_CMAKE_ARGS:-}
    cmake --build "${build}" --target test_camodel_turboquant_reshape_and_cache_a8 -j4
fi

for n in ${slices}; do
    dir="${build}/s${n}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        if ! timeout "${A8_TIMEOUT:-900s}" \
                "${build}/lib/test_camodel_turboquant_reshape_and_cache_a8" \
                --slices "${n}" > run.log 2>&1; then
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
echo "A8 CAModel build and run directory: ${build}"
