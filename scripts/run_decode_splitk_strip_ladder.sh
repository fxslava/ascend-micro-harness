#!/usr/bin/env bash
# The functional ladder for the continuous strip pipeline in
# `turboquant_paged_decode_splitk_aic`, on CAModel.
#
#   bash scripts/run_decode_splitk_strip_ladder.sh cheap   # 16-token pages
#   bash scripts/run_decode_splitk_strip_ladder.sh tails   # ragged contexts
#   bash scripts/run_decode_splitk_strip_ladder.sh wide    # 128-token pages
#   bash scripts/run_decode_splitk_strip_ladder.sh all
#
# Why this is not `test_decode_splitk_aic_sim`
# -------------------------------------------
# That target is the CAModel twin of the NPU validation driver and it IS the
# right instrument on silicon: one binary, the whole page-count x K ladder plus
# the tail ladder, every rung against the FP64 partial oracle. Under CAModel it
# is about ninety launches of up to sixty-four strips each, which is hours of
# model time on this host -- the eight-strip legacy rung alone is about fifteen
# minutes. So that target is BUILT here and not run, and this script runs the
# specific geometries the pipeline's correctness turns on through
# bench_decode_splitk_sim, which takes them as flags.
#
# What each group is for
# ----------------------
# cheap  16-token pages: one strip per page, so EVERY strip crosses a page
#        boundary. This is the harshest rung for StripCursor's page-transition
#        path and for the L1 key ring's WAR edge, and it is cheap because the
#        strip count equals the page count.
# tails  unaligned contexts at 128-token pages. The pipeline's tail mask rides
#        a running token counter rather than a division, so an unaligned
#        context is what exercises that change; S = 1 is also the only rung
#        where a single live lane carries the whole result.
# wide   128-token pages: eight strips per page, so the WITHIN-page increment
#        path runs and the page transition is rare. This is the production
#        geometry, and the 32-strip rung is the expensive one.
#
# Rungs are independent and each keeps its own evidence directory, so they run
# concurrently. Overrides: ASCEND_HOME_PATH, SOC_VERSION, LADDER_DIR.
set -uo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
work="${LADDER_DIR:-/tmp/ascend-decode-strip-ladder}"
mode="${1:-all}"

build_sim() {
    mkdir -p "${work}"
    SPLITK_SRC_DIR="${work}/src" SPLITK_BUILD_DIR="${work}/build" \
        ASCEND_HOME_PATH="${toolkit}" SOC_VERSION="${soc}" \
        SPLITK_LEGACY_DECODERS=ON \
            bash "${repo_dir}/scripts/run_decode_splitk_bench.sh" sim --help \
        >"${work}/build.log" 2>&1
    local status=$?
    if (( status != 0 )); then
        echo "sim build failed; see ${work}/build.log" >&2
        return "${status}"
    fi
    echo "built: ${work}/build/lib/bench_decode_splitk_sim"
}

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
        timeout "${LADDER_TIMEOUT:-5400s}" \
            "${work}/build/lib/bench_decode_splitk_sim" "$@" >"${work}/${tag}.log" 2>&1
    )
    printf '%-14s exit=%s  %s\n' "${tag}" "$?" "$*"
    grep -E "AIC partial (margins|profile)|^PASS|^FAIL|Total tick" "${work}/${tag}.log" \
        | grep -v DRVSTUB | sed 's/^/    /'
}

base="--impl aic --single --requests 1 --heads 1 --rows 1"

cheap() {
    # One strip per page: every strip is a page transition.
    rung p1_k1_t16 --context 16  --page-tokens 16 ${base} --page-count 1 --splitk 1 &
    rung p4_k1_t16 --context 64  --page-tokens 16 ${base} --page-count 4 --splitk 1 &
    rung p8_k1_t16 --context 128 --page-tokens 16 ${base} --page-count 8 --splitk 1 &
    rung p4_k2_t16 --context 64  --page-tokens 16 ${base} --page-count 4 --splitk 2 &
    rung p4_k4_t16 --context 64  --page-tokens 16 ${base} --page-count 4 --splitk 4 &
    wait
}

tails() {
    # Capacity stays one whole 128-token page; only the live count moves.
    rung s1_t128   --context 128 --page-tokens 128 ${base} --splitk 1 --seq-len 1 &
    rung s113_t128 --context 128 --page-tokens 128 ${base} --splitk 1 --seq-len 113 &
    rung s127_t128 --context 128 --page-tokens 128 ${base} --splitk 1 --seq-len 127 &
    wait
}

wide() {
    # Eight strips per page: the within-page increment path.
    rung p1_k1_t128 --context 128 --page-tokens 128 ${base} --page-count 1 --splitk 1 &
    wait
    # Thirty-two strips. The expensive rung, on its own.
    rung p4_k1_t128 --context 512 --page-tokens 128 ${base} --page-count 4 --splitk 1
}

case "${mode}" in
cheap) build_sim && cheap ;;
tails) build_sim && tails ;;
wide)  build_sim && wide ;;
all)   build_sim && cheap && tails && wide ;;
*)     echo "usage: $0 {cheap|tails|wide|all}" >&2; exit 2 ;;
esac
