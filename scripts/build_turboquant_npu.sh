#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
vllm="${VLLM_ASCEND_ROOT:-/mnt/c/Ascend/vllm-ascend}"
build="${1:-$(mktemp -d /tmp/ascend-turboquant-npu-XXXXXX)}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN 9.2 toolkit not found: ${toolkit}" >&2
    exit 2
fi
if [[ ! -f "${vllm}/csrc/attention/turboquant/op_kernel/turboquant_paged_attention.cpp" ]]; then
    echo "BF16 baseline source not found under ${vllm}" >&2
    exit 2
fi

set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
cmake -S "${repo}" -B "${build}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
    -DASCEND_HOME_PATH="${toolkit}" -DVLLM_ASCEND_ROOT="${vllm}"
cmake --build "${build}" --target turboquant_npu_all -j 4
echo "NPU binaries built at: ${build}"
echo "Hardware only: ${build}/test_turboquant_a8_validation"
echo "Hardware only: ${build}/bench_turboquant_a8_vs_bf16"
