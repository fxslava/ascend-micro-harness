#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_rel="kernels/ops/turboquant/vllm_integration_layer"
stage_dir="${repo_dir}/${stage_rel}"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
work_dir="${TQ_STAGE_CAMODEL_DIR:-$(mktemp -d /tmp/turboquant-staged-camodel.XXXXXX)}"
src_dir="${work_dir}/src"
build_dir="${work_dir}/build"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
mkdir -p "${work_dir}"
printf 'staged CAModel evidence directory: %s\n' "${work_dir}"

if [[ ! -f "${toolkit}/set_env.sh" || ! -f "${sim_lib_dir}/libruntime_camodel.so" ]]; then
    echo "CANN CAModel runtime not found under ${toolkit} for ${soc}" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

# Build from an ext4 copy because BiSheng's CANN preprocessing can observe stale
# source pages when the checkout is on /mnt/c.
mkdir -p "${src_dir}"
rsync -a --delete "${stage_dir}/" "${src_dir}/"
cmake -S "${src_dir}/standalone" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
    -DASCEND_HOME_PATH="${toolkit}" -DTURBOQUANT_RUN_MODE=sim \
    >"${work_dir}/configure.log" 2>&1 || {
        tail -80 "${work_dir}/configure.log" >&2
        exit 1
    }
cmake --build "${build_dir}" --target test_integration_host_gate \
    test_integration_decode test_integration_batch test_integration_scatter \
    bench_integration_batch -j4 \
    >"${work_dir}/build.log" 2>&1 || {
        tail -100 "${work_dir}/build.log" >&2
        exit 1
    }

# The Tier-1 host gate first, before any device launch. It needs no simulator and finishes in
# milliseconds, and every CAModel case below compares the device against the oracles it checks:
# a failure here means the rest of this run could not tell a wrong kernel from a wrong oracle,
# so there is no point spending half an hour of model time to find out.
"${build_dir}/bin/test_integration_host_gate" >"${work_dir}/host-gate.log" 2>&1 || {
    cat "${work_dir}/host-gate.log" >&2
    exit 1
}
cat "${work_dir}/host-gate.log"

run_case() {
    local name="$1"; shift
    local exe="$1"; shift
    local dir="${work_dir}/${name}"
    mkdir -p "${dir}"
    (
        cd "${dir}"
        timeout "${TQ_STAGE_CAMODEL_TIMEOUT:-1800s}" "${build_dir}/bin/${exe}" "$@" \
            >run.log 2>&1 || { cat run.log >&2; exit 1; }
        if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' run.log; then
            cat run.log >&2
            exit 1
        fi
        shopt -s nullglob
        for dump in core*.excp_log.dump excp_log*.dump; do
            [[ ! -s "${dump}" ]] || { echo "nonempty exception dump: ${dump}" >&2; exit 1; }
        done
        shopt -u nullglob
    )
}

audit_scatter() {
    local name="$1" rows="$2" runs="$3" launches="$4"
    local d="${work_dir}/${name}"
    python3 "${repo_dir}/scripts/analyze_scatter_trace.py" \
        "${d}/core0.veccore0.instr_popped_log.dump" \
        "${d}/core0.veccore0.instr_log.dump" \
        --aiv1 "${d}/core0.veccore1.instr_popped_log.dump" \
        --rows "${rows}" --runs "${runs}" --launches "${launches}" \
        >"${d}/trace-audit.txt"
}

audit_batch() {
    local name="$1" d="${work_dir}/${name}"
    python3 "${repo_dir}/scripts/analyze_reshape_trace.py" \
        "${d}/core0.veccore0.instr_popped_log.dump" "${d}/core0.veccore0.instr_log.dump" \
        "${d}/core0.veccore1.instr_popped_log.dump" "${d}/core0.veccore1.instr_log.dump" \
        "${d}/core0.cubecore0.instr_popped_log.dump" "${d}/core0.cubecore0.instr_log.dump" \
        >"${d}/trace-audit.txt"
}

run_case decode test_integration_decode --camodel-smoke

# Exercise cold start, two and three strips, then the ragged and complete
# second-wave rungs required by the project's CAModel refinement guide.
for strips in 1 2 3 32 33 48 64 128; do
    name="batch-${strips}"
    run_case "${name}" test_integration_batch --slices "${strips}"
    audit_batch "${name}"
done

for head in 0 2 3; do
    name="scatter-canary-${head}"
    run_case "${name}" test_integration_scatter --mode canary --head "${head}"
    audit_scatter "${name}" 1 1 1
done
run_case scatter-sequential test_integration_scatter --mode sequential --head 2
audit_scatter scatter-sequential 5 3 3
run_case scatter-zero test_integration_scatter --mode zero --head 0
audit_scatter scatter-zero 1 1 1
run_case scatter-sweep test_integration_scatter --mode sweep --head 3
audit_scatter scatter-sweep 136 16 16

# The prefill batch benchmark, on the narrowest rung and with one timed sample. Built with
# ASCEND_MICRO_BENCH_SIM, so it reports NA for latency -- the CAModel's stub driver rejects ACL
# timing events -- and what this case establishes is that the benchmark's own fixture setup,
# verification pass, traffic arithmetic and CSV path execute against a real launch rather than
# only compiling. Its hardware figures have to come from a 950PR.
run_case bench-batch bench_integration_batch --strips 1 --warmup 0 --iters 1 \
    --csv-out bench-batch.csv

echo "PASS staged CAModel validation; evidence: ${work_dir}"
