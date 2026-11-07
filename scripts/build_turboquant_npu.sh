#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${1:-$(mktemp -d /tmp/ascend-turboquant-npu-XXXXXX)}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN 9.2 toolkit not found: ${toolkit}" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
cmake -S "${repo}" -B "${build}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend950PR_9599 \
    -DASCEND_HOME_PATH="${toolkit}"
cmake --build "${build}" --target turboquant_npu_all -j"${JOBS:-$(nproc)}"
echo "NPU binaries built at: ${build}"
echo "Hardware only: ${build}/lib/test_turboquant_a8_validation"
echo "Hardware only: ${build}/lib/bench_turboquant_reshape_and_cache"
# The fused decode operator's own pair. The benchmark holds one kernel unless it was
# configured with -DTQ_BENCH_LEGACY_DECODERS=ON, and the validation ladder is 16 launches
# against an FP64 oracle with 0x5A canaries on every allocation, every unmapped KV page and
# every inter-head scale padding lane. Both need a 950PR.
echo "Hardware only: ${build}/lib/test_npu_fused_decode_opt"
echo "Hardware only: ${build}/lib/bench_turboquant_decode_splitk"
