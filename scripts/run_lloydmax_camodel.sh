#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
report="${LLOYDMAX_REPORT_DIR:-${repo}/reports/lloydmax_quantize}"
mkdir -p "$report"
report="$(cd "$report" && pwd)"
build="${LLOYDMAX_BUILD_DIR:-$(mktemp -d /tmp/ascend-lloydmax-build-XXXXXX)}"
mkdir -p "$build"
build="$(cd "$build" && pwd)"
if [[ "${1:-all}" != run-only ]]; then
    src="$(mktemp -d /tmp/ascend-lloydmax-src-XXXXXX)"
    # Fresh ext4 copy avoids drvfs compiler-cache issues; no worktree mutation.
    rsync -a --exclude .git --exclude build --exclude reports --exclude sim_log --exclude bin "$repo/" "$src/"
    set +u
    source "$toolkit/set_env.sh"
    set -u
    cmake -S "$src" -B "$build" -DSOC_VERSION=Ascend950PR_9599 -DASCEND_HOME_PATH="$toolkit" > "$report/configure.log" 2>&1
    cmake --build "$build" --target test_lloydmax_probe test_icdf_quantize_probe test_camodel_lloydmax_quantize -j4 > "$report/build.log" 2>&1
    "$build/lib/test_lloydmax_probe" | tee "$report/tier1.log"
    "$build/lib/test_icdf_quantize_probe" | tee "$report/original_tier1.log"
fi
printf '%s\n' "$build" > "$report/build_dir.txt"
bash "$repo/scripts/capture_lloydmax_environment.sh" "$build" "$report"
export LD_LIBRARY_PATH="$build/lib:$toolkit/tools/simulator/Ascend950PR_9599/lib:$toolkit/lib64:$toolkit/devlib:${LD_LIBRARY_PATH:-}"
run_case() {
    local name="$1" bits="$2" mode="$3" n="$4" rows="$5" boundary="$6"
    local work="$build/runs/$name" dir="$report/$name"
    mkdir -p "$work" "$dir"
    # Run on ext4 too: simulator trace creation on drvfs dominates tiny VFs.
    if [[ "${LLOYDMAX_REUSE_RESULTS:-0}" != 1 || ! -f "$work/audit.json" ]]; then
        (cd "$work"; timeout "${LLOYDMAX_TIMEOUT:-180s}" "$build/lib/test_camodel_lloydmax_quantize" "$bits" "$mode" "$n" "$rows" "$boundary" > run.log 2>&1)
    fi
    grep 'PASS\|FAIL' "$work/run.log"
    python3 "$repo/scripts/analyze_lloydmax_trace.py" "$work" --bits "$bits" --mode "$mode" --n "$n" --rows "$rows" > "$work/audit.txt"
    # CAModel emits hundreds of empty files for inactive cores. Retain every
    # nonempty log/trace as evidence and avoid copying those empty placeholders.
    find "$work" -type f -empty -delete
    rsync -a "$work/" "$dir/"
}
for n in 128 256; do
    for rows in 1 8; do
        for bits in 2 4; do
            modes=(0 1 2 4 5)
            if [[ "$bits" == 4 ]]; then modes+=(3); fi
            for mode in "${modes[@]}"; do
                run_case "b${bits}_m${mode}_n${n}_r${rows}" "$bits" "$mode" "$n" "$rows" 0
            done
        done
    done
    for bits in 2 4; do
        for mode in 0 1; do
            run_case "boundary_b${bits}_m${mode}_n${n}" "$bits" "$mode" "$n" 1 1
        done
    done
done
python3 "$repo/scripts/summarize_lloydmax_benchmark.py" "$report"
echo "Lloyd-Max CAModel report: $report"
