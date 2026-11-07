#!/usr/bin/env bash
# Compile all simulator and NPU targets.
# Logs are retained under reports/merge_compilation; no device is launched.
#
# THIS IS THE LONGEST BUILD IN THE REPOSITORY, AND IT IS SUPPOSED TO BE.
#
# A target-less `cmake --build` builds every ascendc_library the two target modules
# declare -- 27 of them at the time of writing -- and CANN compiles each kernel's
# translation unit several times per library through BiSheng (the measured breakdown is
# in the top-level CMakeLists.txt; the fused decode operator alone is about 2500 s and
# seven compiles of one TU). That is the point of a merge gate: everything compiles
# strictly, on both legs, before a merge.
#
# What makes it bearable:
#   * -j now defaults to the host's core count rather than 4. The aic and aiv device
#     legs of each library are independent ExternalProjects and overlap, as do separate
#     libraries.
#   * install ccache and the top-level CMakeLists.txt routes every one of those
#     compiles through it, which is what turns the second and later runs of this gate
#     from "most of an hour" into minutes. CANN forwards the launcher into all five
#     nested project kinds; nothing else reaches those compiles.
#   * BUILD_JOBS still overrides -j, and a prior build directory can be passed as $1 to
#     reuse it.
#
# If all that is wanted is "does the decode operator still compile", build the two
# decode targets by name instead of running this:
#   cmake --build <dir> --target test_camodel_fused_decode_opt turboquant_fused_decode_opt_npu -j$(nproc)
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
build="${1:-$(mktemp -d /tmp/ascend-merge-build-XXXXXX)}"
logs="${repo}/reports/merge_compilation"
mkdir -p "${logs}"
printf '%s\n' "${build}" > "${logs}/build_directory.log"
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
cmake -S "${repo}" -B "${build}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend950PR_9599 \
    -DASCEND_HOME_PATH="${toolkit}" 2>&1 | tee "${logs}/configure.log"
start_seconds=${SECONDS}
cmake --build "${build}" -j "${BUILD_JOBS:-$(nproc)}" \
    2>&1 | tee "${logs}/build.log"
printf 'merge compile gate: %s s with -j%s in %s\n' \
    "$((SECONDS - start_seconds))" "${BUILD_JOBS:-$(nproc)}" "${build}" |
    tee "${logs}/build_duration.log"
