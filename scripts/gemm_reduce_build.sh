#!/usr/bin/env bash
# Configure and build the gemm-reduce probe AND the production fused-decode operator inside the
# CANN 9.2 container.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_build.sh [BUILD_DIR]
#
# The production operator is built too, and that is the point of wiring the reduction into it:
# the default (vector-reducer) path has to keep compiling strict, and the Cube GEMM path has to
# be reachable from the same translation unit.
#
# Always a FRESH build directory by default, because CANN's preprocess stage links each device
# object over ITSELF and a second pass in the same directory dies with "unknown file type".
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
soc="${SOC_VERSION:-Ascend950PR_9599}"
build="${1:-/work/build}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rm -rf "$build"
echo "=== configure ($soc) ==="
cmake -S "$src" -B "$build" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSOC_VERSION="$soc" \
    >"$build.configure.log" 2>&1 || { tail -40 "$build.configure.log"; exit 1; }

echo "=== build: gemm reduce probe + host gate + production operator ==="
cmake --build "$build" -j8 --target camodel_gemm_reduce_kernel test_camodel_gemm_reduce \
    test_gemm_reduce_geometry camodel_fused_decode_opt_kernel \
    >"$build.build.log" 2>&1 || {
    grep -E 'error|Error' "$build.build.log" | head -40
    tail -20 "$build.build.log"
    exit 1
}
grep -E 'Built target' "$build.build.log" | tail -6
echo "PASS strict device build (-Wall -Werror, $soc), default reduction path"
