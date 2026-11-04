#!/usr/bin/env bash
# Run the fused decode operator on CAModel with TQ_GEMM_REDUCE=ON, so the Cube GEMM reduction
# drives the reduction phase out of the overlaid UB pool.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_overlay_camodel.sh
#
# This is the gate the switch gate cannot be: that one proves the overlay COMPILES, and a pool
# aliased onto the wrong bytes compiles just as well. What this checks is that the operator
# still produces the right context with the reduction reading and writing inside the attention
# phase's dead transient pool, that its canaries are intact, and that its own traces audit.
#
# The rungs are the operator's own, not the probe's: only --splits > 1 reaches the reduction at
# all, so the split rung is the one that matters and the single-split rungs are the control that
# the overlay has not disturbed the attention phase.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
soc="${SOC_VERSION:-Ascend950PR_9599}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${GEMM_OVERLAY_ROOT:-/work/overlay}"
repo="$src"
fails=0

rm -rf "$root"
mkdir -p "$root"
printf 'Evidence/build root: %s\n' "$root"

echo "=== configure + build with TQ_GEMM_REDUCE=ON ==="
cmake -S "$repo" -B "$root/build" -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" -DSOC_VERSION="$soc" \
    -DTQ_GEMM_REDUCE=ON >"$root/configure.log" 2>&1 ||
    { tail -30 "$root/configure.log"; exit 1; }
cmake --build "$root/build" --target test_camodel_fused_decode_opt -j8 \
    >"$root/build.log" 2>&1 || { grep -E 'error' "$root/build.log" | head -20; exit 1; }
echo "PASS build (-Wall -Werror, $soc) with the Cube GEMM reduction wired in"

sim="$ASCEND_HOME_PATH/tools/simulator/$soc/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:${LD_LIBRARY_PATH:-}"

run_one() {
    local tag="$1"; shift
    mkdir -p "$root/$tag"
    local status=0
    (cd "$root/$tag"; rm -f ./*.dump
     timeout "${GEMM_OVERLAY_TIMEOUT:-2400s}" "$root/build/lib/test_camodel_fused_decode_opt" "$@" \
        >run.log 2>&1) || status=$?
    grep -E '^(PASS|FAIL|worst|max_)' "$root/$tag/run.log" | head -6 || true
    if ((status)); then
        printf 'FAIL %s: exit %d\n' "$tag" "$status"; tail -20 "$root/$tag/run.log"
        fails=$((fails + 1)); return 1
    fi
    if grep -qEi '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/$tag/run.log"; then
        printf 'FAIL %s: the model reported errors\n' "$tag"
        grep -m6 -Ei '\[error\]|FAIL' "$root/$tag/run.log"
        fails=$((fails + 1)); return 1
    fi
    if ! PYTHONPATH="$repo/scripts" python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" \
            "$root/$tag" >"$root/$tag/audit.json" 2>"$root/$tag/audit.err"; then
        printf 'FAIL %s: trace audit\n' "$tag"; cat "$root/$tag/audit.err"
        fails=$((fails + 1)); return 1
    fi
    printf 'PASS %s\n' "$tag"
}

# The rung that actually reaches the reduction: more than one split, so NeedsReduction() is true
# and the Cube GEMM runs out of the overlaid pool.
run_one overlay-splits-4   --tiles 6 --splits 4
# A second split count, so the batcher sees a different live-task run length.
run_one overlay-splits-2   --tiles 6 --splits 2
# Controls: single-split geometries never enter the reduction, so these say whether moving three
# attention buffers into a TBufPool disturbed the attention phase itself.
run_one overlay-canary-0   --canary 0
run_one overlay-canary-7   --canary 7
run_one overlay-tiles-7    --tiles 7

if ((fails)); then
    printf 'FAIL overlay CAModel: %d rung(s) failed (%s)\n' "$fails" "$root"
    exit 1
fi
printf 'PASS overlay CAModel: the Cube GEMM reduction runs out of the aliased pool (%s)\n' "$root"
