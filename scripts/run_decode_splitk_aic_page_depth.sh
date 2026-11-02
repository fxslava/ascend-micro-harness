#!/usr/bin/env bash
# Page-depth gate for turboquant_paged_decode_splitk_aic.
#
#   bash scripts/run_decode_splitk_aic_page_depth.sh models   # host gates only, no CANN
#   bash scripts/run_decode_splitk_aic_page_depth.sh build     # + build the sim twin
#   bash scripts/run_decode_splitk_aic_page_depth.sh ladder    # + the CAModel page ladder
#   bash scripts/run_decode_splitk_aic_page_depth.sh audit     # + the trace census
#   bash scripts/run_decode_splitk_aic_page_depth.sh all
#
# Why this exists separately from run_decode_splitk_bench.sh: that script's
# `smoke` covers K=2 and K=4 over three 16-token pages, where the DEEPEST
# worker carries two pages. Nothing in it carries three or four pages in one
# worker, which is the geometry the inter-page (m, l, Acc) carry actually
# needs, and `--splitk 1` is the only way to get it -- a single worker owns
# every page of the window.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, AIC_DEPTH_BUILD_DIR,
# AIC_DEPTH_SRC_DIR, AIC_DEPTH_OUT, AIC_DEPTH_TIMEOUT.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${AIC_DEPTH_SRC_DIR:-/tmp/ascend-aic-depth-src}"
build_dir="${AIC_DEPTH_BUILD_DIR:-/tmp/ascend-aic-depth-build}"
out_dir="${AIC_DEPTH_OUT:-${repo_dir}/reports/decode_splitk_aic_page_depth}"
run_timeout="${AIC_DEPTH_TIMEOUT:-3600s}"
mode="${1:-all}"

mkdir -p "${out_dir}"

# --------------------------------------------------------------- host gates
run_models() {
    echo "== gate 1: capacity-one token models =="
    python3 "${repo_dir}/tests/cpu/test_decode_ring_model.py" \
        | tee "${out_dir}/ring_model.txt" | tail -2
    # The multi-page MIX loop the single-page ring model does not reach.
    python3 "${repo_dir}/tests/cpu/test_decode_page_loop_model.py" \
        | tee "${out_dir}/page_loop_model.txt" | tail -4
}

# --------------------------------------------------------------- sim build
configure_and_build() {
    [[ -f "${toolkit}/set_env.sh" ]] || {
        echo "CANN environment not found: ${toolkit}/set_env.sh" >&2; exit 2; }
    set +u; source "${toolkit}/set_env.sh"; set -u
    export ASCEND_HOME_PATH="${toolkit}"
    mkdir -p "${src_dir}"
    # Out of /mnt/c and into ext4, and always a FRESH build directory: the same
    # two reasons run_decode_splitk_bench.sh gives.
    rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
        "${repo_dir}/" "${src_dir}/"
    rm -rf "${build_dir}"
    cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build_dir}" --target bench_decode_splitk_sim -j 4
}

sim_env() {
    set +u; source "${toolkit}/set_env.sh"; set -u
    export LD_LIBRARY_PATH="${build_dir}/lib:${toolkit}/tools/simulator/${soc}/lib:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
}

# --------------------------------------------------------------- the ladder
# Each rung is CONTEXT:PAGE_TOKENS:SPLITS. The K=1 rungs are the page-carry
# depth ladder; the K=4/K=2 rungs at the same context isolate whether a
# deviation belongs to one page or to the carry across several.
LADDER=(
    "16:16:1"    # one page: the single-page canary
    "32:16:1"    # two pages: the first boundary
    "48:16:1"    # three pages
    "64:16:1"    # four pages
    "64:16:4"    # the same four pages, one per worker: no carry at all
    "64:16:2"    # two workers of two pages each
    "256:128:2"  # two EIGHT-strip pages: ring wrap and a page boundary together
)

run_ladder() {
    sim_env
    cd "${build_dir}"
    local csv="${out_dir}/page_depth.csv"
    echo "context,page_tokens,splits,pages,result,margins,detail" > "${csv}"
    for rung in "${LADDER[@]}"; do
        IFS=: read -r context page_tokens splits <<< "${rung}"
        local pages=$((context / page_tokens))
        local log="${out_dir}/ladder_n${context}_p${page_tokens}_k${splits}.txt"
        echo "== ladder context=${context} page-tokens=${page_tokens} K=${splits} (pages=${pages}) =="
        rm -f core*.dump excp_log*.dump
        local status=PASS detail=
        if ! timeout "${run_timeout}" ./lib/bench_decode_splitk_sim \
                --context "${context}" --page-tokens "${page_tokens}" \
                --heads 1 --rows 1 --splitk "${splits}" --single --impl aic \
                > "${log}" 2>&1; then
            status=RUN_ERROR
        fi
        if grep -q '^PASS' "${log}"; then
            status=PASS
        elif grep -q '^FAIL' "${log}"; then
            status=FAIL
            detail="$(grep -m1 -E 'AIC partial mismatch|^FAIL' "${log}" | tr -d ',')"
        fi
        # A simulator-level fault is a failure even when the comparison passed.
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' "${log}"; then
            status=SIM_FAULT
            detail="$(grep -m1 -Ei '\[(error|fatal)\]|invalid ldst addr' "${log}" | tr -d ',')"
        fi
        shopt -s nullglob
        for dump in core*.excp_log.dump excp_log*.dump; do
            [[ -s "${dump}" ]] && { status=EXCEPTION; detail="${dump} not empty"; }
        done
        shopt -u nullglob
        # The margins matter as much as the verdict: a rung that passes at
        # 0.98x of its bound is not a pass to build on, and the whole point of
        # a depth ladder is to see the trend rather than the last line crossed.
        local margins
        margins="$(sed -n 's/^AIC partial margins[^:]*: *//p' "${log}" | head -1 | tr -d ',')"
        echo "${context},${page_tokens},${splits},${pages},${status},${margins},${detail}" >> "${csv}"
        echo "   -> ${status} ${margins} ${detail}"
        # Keep the deepest K=1 rung's dumps for the trace audit.
        if [[ "${rung}" == "64:16:1" ]]; then
            mkdir -p "${out_dir}/trace_n64_k1"
            cp core*.instr_log.dump "${out_dir}/trace_n64_k1/" 2>/dev/null || true
        fi
    done
    echo
    column -s, -t < "${csv}"
}

# ---------------------------------------------------------- the trace audit
run_audit() {
    local dir="${out_dir}/trace_n64_k1"
    [[ -d "${dir}" ]] || { echo "no trace dumps; run the ladder first" >&2; exit 2; }
    echo "== trace census: 1 worker, 4 pages, 1 strip per page =="
    python3 "${repo_dir}/scripts/analyze_aic_splitk_trace.py" "${dir}" \
        --workers 1 --pages 4 --strips 1 | tee "${out_dir}/trace_census.json"
}

case "${mode}" in
models) run_models ;;
build)  run_models; configure_and_build ;;
ladder) run_models; configure_and_build; run_ladder ;;
audit)  run_audit ;;
all)    run_models; configure_and_build; run_ladder; run_audit ;;
*) echo "usage: $0 {models|build|ladder|audit|all}" >&2; exit 2 ;;
esac

echo
echo "artifacts: ${out_dir}"
