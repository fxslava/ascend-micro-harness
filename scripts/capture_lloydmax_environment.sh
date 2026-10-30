#!/usr/bin/env bash
set -euo pipefail
build="$1" report="$2"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
compiler="$toolkit/x86_64-linux/ccec_compiler/bin/bisheng"
{
    date -u '+Measured UTC: %Y-%m-%dT%H:%M:%SZ'
    uname -a
    "$compiler" --version
    grep -E 'SOC_VERSION:|ASCEND_HOME_PATH:' "$build/CMakeCache.txt"
    readlink -f "$toolkit/tools/simulator/Ascend950PR_9599/lib"
    "$compiler" --cce-aicore-arch=dav-c310-vec --cce-aicore-only -x cce -dM -E /dev/null \
        | grep -E '^#define (__NPU_ARCH__|__CCE_AICORE__|__DAV_C310_VEC__) '
    cat "$build/camodel_lloydmax_quantize_kernel_preprocess-prefix/src/camodel_lloydmax_quantize_kernel_preprocess-build/CMakeFiles/aiv_obj.dir/flags.make"
} > "$report/environment.txt"
(
    cd "$repo"
    sha256sum include/routines/routine_vector_lloydmax_quantize.hpp \
        include/routines/routine_vector_gaussian_cdf_quantize.hpp \
        routines/routine_vector_icdf_quantize.hpp kernels/camodel/lloydmax_quantize.cpp \
        tests/cpu/test_lloydmax_probe.cpp tests/common/lloydmax_oracle.hpp \
        tests/camodel/test_camodel_lloydmax_quantize.cpp tests/camodel/camodel_common.hpp \
        tests/cpu/icdf_fixture.hpp cmake/SimTargets.cmake CMakeLists.txt
) > "$report/sources.sha256"
sha256sum "$build/lib/libcamodel_lloydmax_quantize_kernel.so" \
    "$build/lib/test_camodel_lloydmax_quantize" > "$report/binaries.sha256"
snapshot="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build/CMakeCache.txt")"
(
    cd "$snapshot"
    sha256sum -c "$report/sources.sha256"
) > "$report/source_snapshot_verification.txt"
