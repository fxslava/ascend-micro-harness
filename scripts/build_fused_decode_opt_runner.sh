#!/usr/bin/env bash
# Rebuild only the ordinary C++ oracle/runner against an existing kernel DSO.
# A separate output name preserves any runner used by an ongoing evidence run.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(realpath "$1")"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
sim="$toolkit/tools/simulator/${SOC_VERSION:-Ascend950PR_9599}/lib"
output="${2:-$root/build/lib/test_camodel_fused_decode_opt_updated}"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"$toolkit/include" \
    "$repo/tests/camodel/test_camodel_fused_decode_opt.cpp" \
    -L"$root/build/lib" -lcamodel_fused_decode_opt_kernel \
    -L"$toolkit/lib64" -lascendcl -L"$sim" -lnpu_drv_camodel -lmodel_top \
    -lruntime_camodel -lnpu_drv -lascendcl -ltiling_api -lregister -lplatform \
    -lerror_manager -lprofapi -lge_common_base -lunified_dlog -lmmpa -ldl -lascend_dump -lc_sec \
    "-Wl,-rpath-link,$sim" "-Wl,-rpath-link,$toolkit/devlib" -Wl,--disable-new-dtags \
    "-Wl,-rpath,\$ORIGIN:$sim:$toolkit/lib64" \
    -o "$output"
sha256sum "$repo/tests/camodel/test_camodel_fused_decode_opt.cpp" > "$output-source.sha256"
printf '%s\n' "$output"
