#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
# The work root. Persistent and on a local filesystem, not /tmp, which a WSL restart
# or systemd-tmpfiles deletes along with however many hours went into it.
# ASCEND_MICRO_WORK_ROOT overrides it; scripts/lib/work_root.sh has the reasoning.
. "$(dirname "${BASH_SOURCE[0]}")/lib/work_root.sh"
root="${FUSED_DECODE_OPT_ROOT:-$(ascend_micro_work_dir ascend-fused-decode-opt)}"
mkdir -p "$root"
printf 'Evidence/build root: %s\n' "$root"
set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"
if [[ "${1:-}" != reuse ]]; then
    g++ -std=c++17 -O2 -Wall -Wextra -Werror "$repo/tests/cpu/test_fused_decode_opt_contract.cpp" -o "$root/host-contract"
    "$root/host-contract" | tee "$root/host-contract.log"
    # Two of the operator's three capacity-one models: the attention ring, and the vector
    # partial reducer that -DTQ_GEMM_REDUCE=OFF selects (section 1 of
    # REFINEMENT_OF_COMBAT_KERNELS.md lists all three). Running only the first used to be
    # enough when the reduction legs had no model; both have one now, and at splits = 1 the
    # ladder below never enters a reduction at all, so the reducer's ring is a host gate or
    # it is nothing.
    #
    # The THIRD model, tests/cpu/test_gemm_reduce_model.py for the Cube GEMM reduction that
    # TQ_GEMM_REDUCE=ON selects by default, is deliberately NOT run here. Its batched legs
    # explore 15 388 101 states on the deepest rung alone and overran a 1800 s cap when
    # this ladder's host leg was timed with it in, and it already has its own driver,
    # scripts/run_gemm_reduce_camodel.sh, which is where that operator's evidence lives. Run
    # that ladder when the reduction phase changes; putting it inline here would add half an
    # hour to every canary run for a gate nothing in this ladder can invalidate.
    python3 -u "$repo/tests/cpu/test_fused_decode_opt_model.py" >"$root/host-model.log"
    python3 -u "$repo/tests/cpu/test_decode_ring_model.py" >"$root/host-vector-reducer-model.log"
    python3 "$repo/tests/cpu/test_fused_decode_opt_trace.py" >"$root/host-audit-controls.log" 2>&1
    mkdir -p "$root/src"
    if command -v rsync >/dev/null; then
        rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' "$repo/" "$root/src/"
    else
        tar -C "$repo" --exclude='.git' --exclude='reports' --exclude='bin' --exclude='build*' -cf - . |
            tar -C "$root/src" -xf -
    fi
    cmake -S "$root/src" -B "$root/build" -DASCEND_HOME_PATH="$toolkit" -DSOC_VERSION="$soc" >"$root/configure.log" 2>&1
    # ONE cmake --build invocation for both targets, and -j from the host's core count.
    #
    # Each ascendc_library in the chain is recompiled by BUILD_ALWAYS on every
    # invocation -- the top-level CMakeLists.txt has the measured figures and why CANN
    # needs it -- so two invocations cost that twice. The aic and aiv device legs are
    # independent ExternalProjects, so the jobs are used.
    #
    # turboquant_fused_decode_opt_npu is the section 2 strict-compile gate on the
    # PHYSICAL leg: the same operator source at RUN_MODE=npu, which is a second full
    # pass through BiSheng and about half this script's build time. It belongs in the
    # pre-push ladder, so it is built by default. Set
    # FUSED_DECODE_OPT_SKIP_NPU_GATE=1 to drop it while iterating on the CAModel
    # rungs, and run the ladder without it once before pushing.
    build_targets=(test_camodel_fused_decode_opt)
    if [[ -z "${FUSED_DECODE_OPT_SKIP_NPU_GATE:-}" ]]; then
        build_targets+=(turboquant_fused_decode_opt_npu)
    else
        printf 'NOTE: the physical-leg compile gate (turboquant_fused_decode_opt_npu) is SKIPPED\n'
    fi
    cmake --build "$root/build" --target "${build_targets[@]}" -j"${JOBS:-$(nproc)}" >"$root/build.log" 2>&1 || { tail -100 "$root/build.log"; exit 1; }
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
