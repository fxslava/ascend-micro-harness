#!/usr/bin/env bash
# Reproduce the four-page cross-core defect in `turboquant_paged_decode_splitk_aic`
# and the evidence that closes it. See reports/decode_splitk_aic_page_race.md.
#
#   bash scripts/run_decode_splitk_aic_page_race.sh models   # host only, no CANN
#   bash scripts/run_decode_splitk_aic_page_race.sh sweep    # the page sweep on CAModel
#   bash scripts/run_decode_splitk_aic_page_race.sh audit DIR # trace gates on one rung
#   bash scripts/run_decode_splitk_aic_page_race.sh all      # models, sweep, audit
#
# Why the sweep is the shape it is
# -------------------------------
# The defect is a threshold at FOUR carried pages, not a property of any page's
# data and not drift with depth, so the sweep holds the carried count fixed and
# slides the window: three pages from 0, from 1 and from 2 all pass, and four
# and five pages fail wherever they start. A ladder that only grows the context
# cannot tell those two readings apart.
#
# Each rung keeps its own evidence directory because the trace gates need that
# rung's own dumps; the previous ladder overwrote them between rungs and that is
# why its section 5 was never run.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, RACE_DIR.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
work="${RACE_DIR:-/tmp/ascend-decode-page-race}"
mode="${1:-all}"

host_models() {
    echo "=== host gate: the page-loop token/queue model (section 1a) ==="
    python3 "${repo_dir}/tests/cpu/test_decode_page_loop_model.py"
    echo
    echo "=== host gate: the single-ring model, unchanged by this operator ==="
    python3 "${repo_dir}/tests/cpu/test_decode_ring_model.py"
}

build_sim() {
    SPLITK_SRC_DIR="${work}/src" SPLITK_BUILD_DIR="${work}/build" \
        ASCEND_HOME_PATH="${toolkit}" SOC_VERSION="${soc}" \
        bash "${repo_dir}/scripts/run_decode_splitk_bench.sh" sim --help >"${work}/build.log" 2>&1
    local status=$?
    if (( status != 0 )); then
        echo "sim build failed; see ${work}/build.log" >&2
        return "${status}"
    fi
    echo "built: ${work}/build/lib/bench_decode_splitk_sim"
}

# One rung, in its own evidence directory, with its dumps preserved.
rung() {
    local tag="$1"; shift
    local evidence="${work}/ev_${tag}"
    mkdir -p "${evidence}"
    (
        set +u
        source "${toolkit}/set_env.sh"
        set -u
        export ASCEND_HOME_PATH="${toolkit}"
        export LD_LIBRARY_PATH="${work}/build/lib:${toolkit}/tools/simulator/${soc}/lib:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
        cd "${evidence}"
        rm -f core*.dump excp_log*.dump
        timeout "${SPLITK_TIMEOUT:-3600s}" \
            "${work}/build/lib/bench_decode_splitk_sim" "$@" >"${work}/${tag}.log" 2>&1
    )
    printf '%-14s exit=%s  %s\n' "${tag}" "$?" "$*"
    grep -E "AIC partial (margins|profile)|^PASS|^FAIL|Total tick" "${work}/${tag}.log" \
        | grep -v DRVSTUB | sed 's/^/    /'
}

sweep() {
    mkdir -p "${work}"
    build_sim || return 1
    local base="--page-tokens 16 --heads 1 --rows 1 --splitk 1 --single --impl aic"
    # Three pages, slid across the window: all three must pass.
    rung p3_from0 --context 48 ${base} --page-begin 0 --page-count 3
    rung p3_from1 --context 64 ${base} --page-begin 1 --page-count 3
    rung p3_from2 --context 80 ${base} --page-begin 2 --page-count 3
    # Four and five carried pages: the rungs the defect lives in.
    rung p4_from0 --context 64 ${base} --page-begin 0 --page-count 4
    rung p4_from1 --context 80 ${base} --page-begin 1 --page-count 4
    rung p5_from0 --context 80 ${base} --page-begin 0 --page-count 5
    # One page per worker at the same depth: isolates the carry from the data.
    rung p4_k4 --context 64 --page-tokens 16 --heads 1 --rows 1 --splitk 4 --single --impl aic
}

audit() {
    local evidence="${1:-${work}/ev_p4_from0}"
    echo "=== section 5: HardEvent balance, mode-4 census, Mmad census ==="
    python3 "${repo_dir}/scripts/analyze_aic_splitk_trace.py" "${evidence}" \
        --workers 1 --pages 4 --strips 1
    echo
    echo "=== section 5: capacity-one ORDER and cross-core STALL ==="
    # kFlagPageDrainedV1 (absolute id 26) must stall on EVERY page, not just
    # the first. 1000 ticks is well under the ~20600-tick page cadence of this
    # geometry and well over the 3-tick floor the collapsed channel showed.
    # Scoped to that id: a stall floor is per channel, and the Cube's
    # per-strip score-free wait legitimately reads ~1000 on one slot.
    python3 "${repo_dir}/scripts/audit_mode4_tick_order.py" "${evidence}" \
        --block 0 --cross --expect-stall 1000 --expect-id 26
    echo
    echo "=== nonempty exception dumps (must be none) ==="
    find "${evidence}" -name "*excp*.dump" -size +0 -print | sed 's/^/    /' || true
}

case "${mode}" in
models) host_models ;;
sweep)  sweep ;;
audit)  audit "${2:-}" ;;
all)    host_models && sweep && audit ;;
*)      echo "usage: $0 {models|sweep|audit [DIR]|all}" >&2; exit 2 ;;
esac
