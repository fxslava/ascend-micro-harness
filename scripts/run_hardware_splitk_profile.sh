#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

TOOLKIT="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
if [[ -f "${TOOLKIT}/set_env.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${TOOLKIT}/set_env.sh"
    set -u
fi

if [[ ! -x "./build/lib/bench_turboquant_decode_splitk" ]]; then
    echo "missing hardware benchmark: ./build/lib/bench_turboquant_decode_splitk" >&2
    echo "build it first with: cmake --build build --target bench_turboquant_decode_splitk -j" >&2
    exit 2
fi
if ! command -v msprof >/dev/null 2>&1; then
    echo "msprof is unavailable; source the CANN 9.2 environment on the NPU host" >&2
    exit 2
fi

OUTPUT_DIR="reports/prof_hardware_splitk8_$(date +%Y%m%d_%H%M%S)"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "refusing to overwrite ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Starting msprof hardware trace on Ascend 950PR..."
msprof --output="${OUTPUT_DIR}" \
    --application="./build/lib/bench_turboquant_decode_splitk --context 16384 --splitk 8 --heads 4 --rows 16 --single" \
    --aic-mode=task-based \
    --aic-metrics=PipeUtilization \
    --task-time=l1 \
    --task-block=all \
    --export=on

echo "[INFO] Profiling complete. Output written to ${OUTPUT_DIR}"
