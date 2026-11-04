#!/usr/bin/env bash
# Run gemm-reduce probe rungs on the Ascend950PR_9599 CAModel, inside the CANN 9.2 container.
#
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_run.sh TAG -- ARGS...
#   bash scripts/gemm_reduce_container.sh run scripts/gemm_reduce_run.sh ladder
#
# A case is evidence only if its own exception dumps are empty and its own traces audit. The
# printed arithmetic is checked LAST on purpose: every device fault this probe family has hit
# produced numerically correct or structurally plausible output while faulting, so a run scored
# on its result alone scores those as passes.
set -uo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
soc="${SOC_VERSION:-Ascend950PR_9599}"
build="${GEMM_BUILD:-/work/build}"
root="${GEMM_EVIDENCE:-/work/evidence}"
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim="$ASCEND_HOME_PATH/tools/simulator/$soc/lib"
export LD_LIBRARY_PATH="$build/lib:$sim:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:${LD_LIBRARY_PATH:-}"
mkdir -p "$root"

fails=0

run_one() {
    local tag="$1"; shift
    mkdir -p "$root/$tag"
    local status=0
    (cd "$root/$tag"; rm -f ./*.dump ./*.json
     timeout "${GEMM_TIMEOUT:-2400s}" "$build/lib/test_camodel_gemm_reduce" "$@" \
        >run.log 2>&1) || status=$?
    grep -E '^(leg=|worst_|PASS|FAIL|first dis|  task=)' "$root/$tag/run.log" | head -14 || true
    if ((status)); then
        printf 'FAIL %s: exit %d\n' "$tag" "$status"
        tail -25 "$root/$tag/run.log"
        fails=$((fails + 1)); return 1
    fi
    if grep -qEi '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/$tag/run.log"; then
        printf 'FAIL %s: the model reported errors\n' "$tag"
        grep -m8 -Ei '\[error\]|\[aicore error\]|FAIL' "$root/$tag/run.log"
        fails=$((fails + 1)); return 1
    fi
    # The shared auditor: capacity-one HardEvent places per route, dispatch == completion, an
    # empty final marking, legal mode-2 channels, cross-core tick order per id, and a
    # non-vacuous census. It is not specific to the attention schedule's ids.
    if ! PYTHONPATH="$src/scripts" python3 "$src/scripts/analyze_fused_decode_opt_trace.py" \
            "$root/$tag" >"$root/$tag/audit.json" 2>"$root/$tag/audit.err"; then
        printf 'FAIL %s: trace audit\n' "$tag"
        cat "$root/$tag/audit.err"
        fails=$((fails + 1)); return 1
    fi
    printf 'PASS %s\n' "$tag"
}

if [ "${1:-}" != ladder ]; then
    tag="$1"; shift
    [ "${1:-}" = -- ] && shift
    run_one "$tag" "$@"
    exit $?
fi

# ---------------------------------------------------------------------------------------------
# The ladder for the dense batched form
# ---------------------------------------------------------------------------------------------
# The reference leg first, so a later batched failure cannot be blamed on the fixture.
run_one b-vector-ref        --leg vector  --splits 8 --tokens 4 --heads 4 --grid 1
run_one b-gemv-ref          --leg gemm    --splits 8 --tokens 4 --heads 4 --grid 1

# The design point: exactly one full batch, both subcores reducing eight rows each.
run_one b-full-16           --leg batched --splits 8 --tokens 4 --heads 4 --grid 1

# Ragged batches, which is where the subcore split and the dead-fractal containment are
# exercised. live = 1 is the one geometry with nothing for subcore 1 to reduce; it must still
# wait, drain and announce.
for live in 1 2 3 5 7 9 15; do
    run_one "b-live-$live"  --leg batched --splits 8 --tokens "$live" --heads 1 --grid 1
done

# Batch-ring depth and wrap, in units of kGemmBatchRows tasks: 16 is one batch (prologue only),
# 32 is the first full steady state, 48 is a ragged third and 64 is repeated reuse a single
# wrap cannot show.
run_one b-batches-2         --leg batched --splits 8 --tokens 8  --heads 4 --grid 1
run_one b-batches-3-ragged  --leg batched --splits 8 --tokens 11 --heads 4 --grid 1
run_one b-batches-4         --leg batched --splits 8 --tokens 16 --heads 4 --grid 1
run_one b-batches-ragged-17 --leg batched --splits 8 --tokens 17 --heads 1 --grid 1

# Split-count sweep. 1, 4 and 5 leave K padding lanes live in the hardware's eyes, which is
# what the zeroed weight lanes and DN2NZ's zero fill both have to cover; 8 is the exact one.
for s in 1 4 5; do
    run_one "b-splits-$s"   --leg batched --splits "$s" --tokens 4 --heads 4 --grid 1
done

# Real Split-K workers, and an empty one: a block whose share starts past the end must still
# drain. Boundary isolation does not prove this and a one-worker probe does not either.
run_one b-workers-2         --leg batched --splits 8 --tokens 8 --heads 4 --grid 2
run_one b-empty-worker      --leg batched --splits 8 --tokens 8 --heads 4 --grid 2 --empty-worker

# Fused tokens interleaved: every other token is skipped, so the batcher sees task-id GAPS and
# must break its runs there -- dnNum walks one uniform stride, so a batch that spanned a gap
# would read the wrong rows. Every skipped token's output slice is a canary for that.
run_one b-mixed             --leg batched --splits 8 --tokens 9 --heads 2 --mixed
run_one b-mixed-workers     --leg batched --splits 8 --tokens 9 --heads 2 --mixed --grid 2

# The arithmetic regression, on its own documented budget. The exact fixture's bitwise gate
# above is never relaxed to make this one pass.
run_one b-varied            --leg batched --splits 8 --tokens 4 --heads 4 --fixture varied
run_one b-varied-wrap       --leg batched --splits 8 --tokens 16 --heads 4 --fixture varied

# The GEMV knob. With M = 16 there is no GEMV case for disableGemv to affect, so these should
# be indistinguishable from the default leg -- which is the measurement, because in the M = 1
# form they were not.
run_one b-gemvknob-16       --leg batched-gemv --splits 8 --tokens 4  --heads 4
run_one b-gemvknob-wrap     --leg batched-gemv --splits 8 --tokens 16 --heads 4
run_one b-gemvknob-varied   --leg batched-gemv --splits 8 --tokens 4  --heads 4 --fixture varied

if ((fails)); then
    printf 'FAIL gemm reduce batched ladder: %d rung(s) failed (%s)\n' "$fails" "$root"
    exit 1
fi
printf 'PASS gemm reduce batched ladder: %s\n' "$root"
