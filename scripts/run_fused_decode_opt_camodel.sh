#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
root="${FUSED_DECODE_OPT_ROOT:-$(mktemp -d /tmp/ascend-fused-decode-opt.XXXXXX)}"
mkdir -p "$root"
printf 'Evidence/build root: %s\n' "$root"
set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"
if [[ "${1:-}" != reuse ]]; then
    g++ -std=c++17 -O2 -Wall -Wextra -Werror "$repo/tests/cpu/test_fused_decode_opt_contract.cpp" -o "$root/host-contract"
    "$root/host-contract" | tee "$root/host-contract.log"
    python3 -u "$repo/tests/cpu/test_fused_decode_opt_model.py" >"$root/host-model.log"
    python3 "$repo/tests/cpu/test_fused_decode_opt_trace.py" >"$root/host-audit-controls.log" 2>&1
    mkdir -p "$root/src"
    if command -v rsync >/dev/null; then
        rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' "$repo/" "$root/src/"
    else
        tar -C "$repo" --exclude='.git' --exclude='reports' --exclude='bin' --exclude='build*' -cf - . |
            tar -C "$root/src" -xf -
    fi
    cmake -S "$root/src" -B "$root/build" -DASCEND_HOME_PATH="$toolkit" -DSOC_VERSION="$soc" >"$root/configure.log" 2>&1
    cmake --build "$root/build" --target test_camodel_fused_decode_opt turboquant_fused_decode_opt_npu -j4 >"$root/build.log" 2>&1 || { tail -100 "$root/build.log"; exit 1; }
else
    shift
fi
if [[ "${1:-}" == build ]]; then exit 0; fi
sim="$toolkit/tools/simulator/$soc/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
run_one() {
    local tag="$1"; shift
    mkdir -p "$root/$tag"
    local status=0
    (cd "$root/$tag"; timeout "${FUSED_DECODE_OPT_TIMEOUT:-1800s}" "$root/build/lib/test_camodel_fused_decode_opt" "$@" >run.log 2>&1) || status=$?
    cat "$root/$tag/run.log"
    if ((status)); then return "$status"; fi
    if grep -Ei '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/$tag/run.log"; then return 1; fi
    python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" "$root/$tag" >"$root/$tag/audit.log"
    printf 'PASS trace audit: %s/audit.log\n' "$root/$tag"
}
if [[ "${1:-}" == ladder ]]; then
    for page in 0 4 7; do run_one "canary-$page" --canary "$page"; done
    run_one canary-tail --canary 7 --tile 1
    for tiles in 3 6 7; do run_one "tiles-$tiles" --tiles "$tiles"; done
    run_one repeat --tiles 1 --repeats 2 --heads 17
    run_one empty --empty 1
    run_one split-empty --tiles 6 --splits 4
else
    run_one "${FUSED_DECODE_OPT_TAG:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}" "$@"
fi
