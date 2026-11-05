#!/usr/bin/env bash
# The Cube GEMV Split-K reduction prototype, on the Ascend950PR_9599 CAModel.
#
#   bash scripts/run_gemm_reduce_camodel.sh              # host gates, build, full ladder
#   bash scripts/run_gemm_reduce_camodel.sh build        # host gates and build only
#   bash scripts/run_gemm_reduce_camodel.sh reuse        # ladder against an existing root
#   bash scripts/run_gemm_reduce_camodel.sh reuse one TAG --leg gemm --splits 4 ...
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, GEMM_REDUCE_ROOT, GEMM_REDUCE_TIMEOUT.
#
# Simulator libraries only. There is no NPU on this host and nothing here launches on one.
#
# Two things this does the way the other probe runners here do, both for reasons already paid
# for once: it copies the tree into the WSL ext4 filesystem rather than building out of /mnt/c,
# because the drvfs page cache can serve BiSheng a stale copy of a source a Windows-side editor
# has just rewritten; and it always configures a fresh build directory, because CANN's
# preprocess stage links each device object over ITSELF and a second pass in the same directory
# dies with "unknown file type".
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
root="${GEMM_REDUCE_ROOT:-$(mktemp -d /tmp/gemm-reduce.XXXXXX)}"
mkdir -p "$root"
printf 'Evidence/build root: %s\n' "$root"
set +u
source "$toolkit/set_env.sh"
set -u
export ASCEND_HOME_PATH="$toolkit"

if [[ "${1:-}" != reuse ]]; then
    # Gate 1 first, before any simulator time is spent: the capacity-one token/queue model and
    # the host compile of the shared geometry contract. A host model is not CAModel evidence,
    # but a schedule that fails here cannot be worth simulating.
    python3 -u "$repo/tests/cpu/test_gemm_reduce_model.py" | tee "$root/host-token-model.log"
    g++ -std=c++17 -O2 -Wall -Wextra -Werror \
        "$repo/tests/cpu/test_gemm_reduce_geometry.cpp" -o "$root/host-geometry"
    "$root/host-geometry" | tee "$root/host-geometry.log"

    mkdir -p "$root/src"
    if command -v rsync >/dev/null; then
        rsync -a --exclude '.git' --exclude 'reports' --exclude 'bin' --exclude 'build*' \
            "$repo/" "$root/src/"
    else
        tar -C "$repo" --exclude='.git' --exclude='reports' --exclude='bin' --exclude='build*' \
            -cf - . | tar -C "$root/src" -xf -
    fi
    cmake -S "$root/src" -B "$root/build" -DASCEND_HOME_PATH="$toolkit" -DSOC_VERSION="$soc" \
        >"$root/configure.log" 2>&1 || { tail -60 "$root/configure.log"; exit 1; }
    cmake --build "$root/build" --target camodel_gemm_reduce_kernel test_camodel_gemm_reduce \
        -j4 >"$root/build.log" 2>&1 || { tail -120 "$root/build.log"; exit 1; }
    printf 'PASS strict device build (-Wall -Werror, %s)\n' "$soc"
else
    shift
fi
if [[ "${1:-}" == build ]]; then exit 0; fi

sim="$toolkit/tools/simulator/$soc/lib"
export LD_LIBRARY_PATH="$root/build/lib:$sim:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"

# A case is evidence only if its own exception dumps are empty and its own traces audit. The
# printed arithmetic is checked LAST on purpose: all three device faults this prototype hit
# produced numerically correct output while faulting, so a run that is scored on its result
# alone scores them as passes.
run_one() {
    local tag="$1"; shift
    mkdir -p "$root/$tag"
    local status=0
    (cd "$root/$tag"; rm -f ./*.dump
     timeout "${GEMM_REDUCE_TIMEOUT:-1800s}" "$root/build/lib/test_camodel_gemm_reduce" "$@" \
        >run.log 2>&1) || status=$?
    grep -E '^(leg=|worst_|PASS|FAIL)' "$root/$tag/run.log" || true
    if ((status)); then
        printf 'FAIL %s: exit %d\n' "$tag" "$status"; tail -25 "$root/$tag/run.log"; return 1
    fi
    if grep -qEi '\[error\]|\[aicore error\]|timeout|mismatch|FAIL' "$root/$tag/run.log"; then
        printf 'FAIL %s: the model reported errors\n' "$tag"
        grep -m6 -Ei '\[error\]|FAIL' "$root/$tag/run.log"; return 1
    fi
    # The shared auditor: capacity-one HardEvent places per route, dispatch == completion,
    # empty final marking, legal mode-2 channels, cross-core tick order per id, and a
    # non-vacuous census. It is not specific to the attention schedule's ids.
    python3 "$repo/scripts/analyze_fused_decode_opt_trace.py" "$root/$tag" \
        >"$root/$tag/audit.json" 2>"$root/$tag/audit.err" || {
            printf 'FAIL %s: trace audit\n' "$tag"; cat "$root/$tag/audit.err"; return 1; }
    printf 'PASS %s (audit: %s/audit.json)\n' "$tag" "$root/$tag"
}

if [[ "${1:-}" == one ]]; then
    shift; tag="$1"; shift
    run_one "$tag" "$@"
    exit 0
fi

# ---------------------------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------------------------
# Every rung runs with numSplits > 1 unless it is explicitly the S = 1 edge, because an
# unexercised reduction is the state this operator was already in.

# The reference leg first, so a later GEMV failure cannot be blamed on the fixture.
run_one vector-exact        --leg vector --splits 8 --tokens 1 --heads 4
run_one vector-varied       --leg vector --splits 8 --tokens 1 --heads 4 --fixture varied

# Split-count sweep. 1, 4 and 5 leave K padding lanes live in the hardware's eyes, which is the
# case the zeroed weight row and DN2NZ's zero fill both have to cover; 8 is the exact one.
for s in 1 4 5 8; do
    run_one "gemm-splits-$s" --leg gemm --splits "$s" --tokens 1 --heads 4
done

# Ring depth. kReduceSlots is 2, so 2 is the prologue-only case, 4 is the first full steady
# state and 8 is the repeated reuse a single wrap cannot show. 5 is the odd tail.
run_one gemm-depth-2        --leg gemm --splits 8 --tokens 1 --heads 2
run_one gemm-depth-4        --leg gemm --splits 8 --tokens 1 --heads 4
run_one gemm-odd-5          --leg gemm --splits 8 --tokens 5 --heads 1
run_one gemm-wrap-8         --leg gemm --splits 8 --tokens 2 --heads 4

# Real Split-K workers, and an empty one: a block whose share starts past the end must still
# drain. Boundary isolation does not prove this and a one-worker probe does not either.
run_one gemm-workers-2      --leg gemm --splits 8 --tokens 2 --heads 4 --grid 2
run_one gemm-empty-worker   --leg gemm --splits 8 --tokens 2 --heads 4 --grid 2 --empty-worker

# Combined and fused tokens interleaved: tokens 0, 2, 4 are combined and 1, 3 are left alone,
# so the stage is exercised at the head, the middle and the tail of the workspace at once and
# every skipped token's output slice is a canary for a task-slicing error.
run_one gemm-mixed          --leg gemm --splits 8 --tokens 5 --heads 2 --mixed
run_one gemm-mixed-workers  --leg gemm --splits 8 --tokens 5 --heads 2 --mixed --grid 2

# The arithmetic regression, on its own documented budget. The exact fixture's bitwise gate
# above is never relaxed to make this one pass.
run_one gemm-varied         --leg gemm --splits 8 --tokens 1 --heads 4 --fixture varied
run_one gemm-varied-wrap    --leg gemm --splits 8 --tokens 2 --heads 4 --fixture varied

# The GEMV knob: same body, same schedule, same ids, one MmadParams field different, so a
# behavioural difference isolates Mmad's M = 1 GEMV mode and nothing else.
run_one gemv-splits-8       --leg gemv --splits 8 --tokens 1 --heads 4
run_one gemv-splits-5       --leg gemv --splits 5 --tokens 1 --heads 4
run_one gemv-varied         --leg gemv --splits 8 --tokens 1 --heads 4 --fixture varied

printf 'PASS gemm reduce CAModel ladder: %s\n' "$root"
