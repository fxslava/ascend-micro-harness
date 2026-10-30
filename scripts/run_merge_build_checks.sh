#!/usr/bin/env bash
# Compile all simulator and NPU targets.
# Logs are retained under reports/merge_compilation; no device is launched.
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
cmake --build "${build}" -j "${BUILD_JOBS:-4}" \
    2>&1 | tee "${logs}/build.log"
