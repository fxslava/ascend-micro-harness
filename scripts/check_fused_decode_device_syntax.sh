#!/usr/bin/env bash
# A fast BiSheng front-end check of the fused decode's device translation unit.
#
# WHY IT EXISTS. `ascendc_library` expands into a nested ExternalProject per leg -- precompile,
# preprocess, an AIC object, an AIV object, a merge -- and on this operator one full pass is tens
# of minutes. That is the right gate before a push and the wrong one for the edit/compile loop: a
# misspelled member costs the same forty minutes as a real pipeline bug. This runs the same
# compiler, the same -Wall -Werror, the same CANN include set and the same two core archs over
# the same translation unit with -fsyntax-only, and finishes in about a minute.
#
# IT IS NOT A SUBSTITUTE FOR THE REAL BUILD. It checks the front end only: no code generation, no
# stack-size accounting, no auto-sync insertion, no linking, and nothing at all about behaviour.
# REFINEMENT_OF_COMBAT_KERNELS.md section 2 means the full strict compile, and section 3 onward
# mean CAModel.
#
#   bash scripts/check_fused_decode_device_syntax.sh
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
host="$toolkit/x86_64-linux"
bisheng="$host/ccec_compiler/bin/bisheng"
[[ -x "$bisheng" ]] || { echo "BiSheng not found at $bisheng" >&2; exit 2; }

stage="$repo/kernels/ops/turboquant/vllm_integration_layer"
source="${1:-$repo/tests/sim/kernel/fused_decode_opt_kernel.cpp}"

includes=()
for dir in asc/impl/adv_api asc/impl/basic_api asc/impl/c_api asc/impl/basic_api/reg_compute \
           asc/impl/simt_api asc/impl/utils asc asc/include asc/include/adv_api \
           asc/include/basic_api asc/include/aicpu_api asc/include/c_api \
           asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils \
           tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl; do
    includes+=("-isystem$host/$dir")
done

status=0
for arch in dav-c310-cube dav-c310-vec; do
    printf 'front-end check, %s ... ' "$arch"
    if "$bisheng" -fsyntax-only -std=c++17 -O3 -DNDEBUG -Wall -Werror \
        -DASCEND_MICRO_CANN_LAUNCH -DTILING_KEY_VAR=0 -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 \
        -D__DAV_C310__ "--cce-aicore-arch=$arch" --cce-aicore-only --cce-aicore-lang \
        --cce-disable-kernel-global-attr-check \
        -include "$host/../include/version/asc_devkit_version.h" \
        "${includes[@]}" \
        "-I$stage/op_kernel" "-I$stage/include" "-I$stage/routines" \
        "$source" 2>"/tmp/fused-decode-syntax-$arch.log"; then
        echo 'ok'
    else
        echo 'FAILED'
        sed -n '1,60p' "/tmp/fused-decode-syntax-$arch.log"
        status=1
    fi
done
exit "$status"
