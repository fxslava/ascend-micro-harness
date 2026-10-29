#!/usr/bin/env bash
# Mandatory pipeline baseline: model -> strict build -> isolated boundaries
# -> N>=2*depth -> complete trace audit. All outputs live in a fresh /tmp tree.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
mode="${1:-all}"
[[ "${mode}" == all || "${mode}" == build ]] || { echo "usage: $0 [all|build]" >&2; exit 2; }
work_dir="$(mktemp -d /tmp/ascend-decode-ring.XXXXXX)"
echo "artifact directory: ${work_dir}"
python3 "${repo_dir}/tests/cpu/test_decode_ring_model.py" | tee "${work_dir}/model.log"
python3 "${repo_dir}/tests/cpu/test_decode_ring_trace_audit.py" > "${work_dir}/trace_tests.log" 2>&1
for probe in single_token_decode splitk_decode; do
    g++ -std=c++17 -O2 -Wall -Wextra -Werror \
        "${repo_dir}/tests/cpu/test_${probe}_probe.cpp" -o "${work_dir}/${probe}"
    "${work_dir}/${probe}" > "${work_dir}/${probe}.log"
done
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${work_dir}/build/lib:${toolkit}/tools/simulator/${soc}/lib:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
mkdir -p "${work_dir}/src"
rsync -a --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
    "${repo_dir}/" "${work_dir}/src/"
cmake -S "${work_dir}/src" -B "${work_dir}/build" -G "Unix Makefiles" \
    -DSOC_VERSION="${soc}" -DASCEND_HOME_PATH="${toolkit}" > "${work_dir}/configure.log" 2>&1
cmake --build "${work_dir}/build" --target test_camodel_single_token_decode_aiv \
    test_camodel_decode_splitk_boundary -j4 > "${work_dir}/build.log" 2>&1 || {
    tail -60 "${work_dir}/build.log" >&2; exit 1;
}
if [[ "${mode}" == build ]]; then exit 0; fi
run_one() {
    local impl="$1" tag="$2" page="$3" strips="$4" strip_begin="$5"
    local dir="${work_dir}/${impl}_${tag}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        trap 'status=$?; if [[ ${status} -ne 0 && -f run.log ]]; then tail -40 run.log >&2; fi' EXIT
        if [[ "${impl}" == aiv ]]; then
            timeout "${DECODE_RING_TIMEOUT:-1800s}" "${work_dir}/build/lib/test_camodel_single_token_decode_aiv" \
                --head 2 --rows 1 --page-begin "${page}" --strips "${strips}" \
                --strip-begin "${strip_begin}" --exact-boundary --no-fixture-check > run.log 2>&1
        else
            timeout "${DECODE_RING_TIMEOUT:-1800s}" "${work_dir}/build/lib/test_camodel_decode_splitk_boundary" \
                --head 2 --rows 1 --page-begin "${page}" --strips "${strips}" \
                --strip-begin "${strip_begin}" > run.log 2>&1
        fi
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' run.log; then
            cat run.log >&2; exit 1
        fi
        shopt -s nullglob
        for dump in core*.excp_log.dump excp_log*.dump; do
            if [[ -s "${dump}" ]]; then echo "nonempty exception dump: ${dump}" >&2; exit 1; fi
        done
        local audit_args=()
        if [[ "${tag}" == wrap ]]; then audit_args+=(--require-overlap); fi
        python3 "${repo_dir}/scripts/analyze_decode_ring_trace.py" . --strips "${strips}" "${audit_args[@]}" > audit.json
        if [[ "${impl}" == aiv ]]; then
            python3 "${repo_dir}/scripts/analyze_single_token_decode_trace.py" \
                core0.veccore0.instr_popped_log.dump core0.veccore0.instr_log.dump \
                --rows 1 --strips "${strips}" --items 1 > census.txt
        else
            python3 "${repo_dir}/scripts/analyze_splitk_decode_trace.py" . --assert-batched > atomic.json
        fi
    )
    echo "PASS ${impl} ${tag}: ${dir}" | tee -a "${work_dir}/summary.txt"
}
for impl in aiv splitk; do
    run_one "${impl}" head 0 1 0
    run_one "${impl}" middle 4 1 4
    run_one "${impl}" tail 7 1 7
    # Two iterations exercise the depth boundary; four traverse both slots
    # twice. One-tile canaries alone cannot prove steady-state reuse.
    run_one "${impl}" depth 0 2 0
    run_one "${impl}" wrap 4 4 4
done
python3 "${repo_dir}/scripts/verify_decode_pipeline_evidence.py" "${work_dir}" --baseline-only
echo "PASS pipeline baseline; evidence: ${work_dir}"
