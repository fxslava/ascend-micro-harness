#!/usr/bin/env bash
# Recompile only host benchmark sources after changing a fixture/CLI. The
# existing simulator/NPU kernel DSOs are read without rebuilding their objects.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(realpath "$1")"
leg="${2:-sim}"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
set +u
source "$toolkit/set_env.sh"
set -u
case "$leg" in sim) target=bench_decode_splitk_sim;; npu) target=bench_turboquant_decode_splitk;; *) exit 2;; esac
for path in tests/npu/bench/bench_turboquant_decode_splitk.cpp \
    tests/npu/turboquant_fused_decode_opt_rig.hpp tests/common/fused_decode_opt_fixture.hpp; do
    cp "$repo/$path" "$root/src/$path"
done
cd "$root/build"
object="CMakeFiles/$target.dir/tests/npu/bench/bench_turboquant_decode_splitk.cpp.o"
gmake -f "CMakeFiles/$target.dir/build.make" "$object"
cmake -E cmake_link_script "CMakeFiles/$target.dir/link.txt" --verbose=1
