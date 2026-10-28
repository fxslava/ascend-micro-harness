#!/usr/bin/env bash
# Build and run the AIV-only scatter ingress operator on the Ascend950PR_9599
# CAModel, then audit its traces.
#
#   bash scripts/run_reshape_scatter_camodel.sh                  # the spatial sweep
#   bash scripts/run_reshape_scatter_camodel.sh ladder            # every rung
#   bash scripts/run_reshape_scatter_camodel.sh --mode sequential --head 2
#
# With no arguments it runs the task's spatial sweep: one isolated head per
# launch at H = 0 (boundary), H = 2 (middle) and H = 3 (boundary). `ladder`
# adds the read-modify-write ladder, the absmax == 0 branch and the 1..16 token
# sweep. Anything else is passed to the runner unchanged.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, SCATTER_BUILD_DIR, SCATTER_SRC_DIR,
# SCATTER_TIMEOUT.
#
# Two things this script does the way run_paged_decode_camodel.sh does, both
# for reasons already paid for once: it copies the tree into the WSL ext4
# filesystem rather than building out of /mnt/c, because the drvfs page cache
# can serve BiSheng a stale copy of a source a Windows-side editor has just
# rewritten; and it always builds from scratch, because CANN's preprocess stage
# links each device object over itself and a second pass in the same build
# directory dies with "unknown file type".
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${SCATTER_SRC_DIR:-/tmp/ascend-scatter-src}"
build_dir="${SCATTER_BUILD_DIR:-/tmp/ascend-scatter-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
run_timeout="${SCATTER_TIMEOUT:-2400s}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
if [[ ! -f "${sim_lib_dir}/libruntime_camodel.so" ]]; then
    echo "CAModel runtime not found: ${sim_lib_dir}" >&2
    exit 2
fi

set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"

# ---------------------------------------------------------------- Tier 1
# Both host oracles first. A CAModel run compares the device against them, so a
# wrong oracle would be indistinguishable from a wrong kernel -- and it is
# these probes, not the simulator, that establish the butterfly stage tables,
# the layout bijections and the (word, bit) merge arithmetic. Sub-second each.
mkdir -p "${repo_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_rotate_butterfly_probe.cpp" \
    -o "${repo_dir}/bin/test_rotate_butterfly_probe"
"${repo_dir}/bin/test_rotate_butterfly_probe"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_reshape_scatter_probe.cpp" \
    -o "${repo_dir}/bin/test_reshape_scatter_probe"
"${repo_dir}/bin/test_reshape_scatter_probe"

# ---------------------------------------------------------------- build
mkdir -p "${src_dir}"
rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
    "${repo_dir}/" "${src_dir}/"
rm -rf "${build_dir}"
cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
    -DASCEND_HOME_PATH="${toolkit}"
cmake --build "${build_dir}" --target test_camodel_reshape_scatter_aiv -j 4

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
cd "${build_dir}"

# ---------------------------------------------------------------- run
# The directly linked 950PR CAModel drops its per-core instruction dumps in the
# current directory, so each run archives them under its own tag before the
# next run overwrites them.
# run_one <tag> <rows> <runs> <launches> -- <runner arguments>
#
# rows/runs/launches are the TOTALS the dump will cover, because the model
# writes its instruction dumps when the process stops: a runner mode that
# performs several launches leaves one dump covering all of them. They are
# what makes the audit's counts assertions rather than observations.
run_one() {
    local tag="$1"; shift
    local rows="$1"; shift
    local runs="$1"; shift
    local launches="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    local log="camodel_scatter_${tag}.log"
    rm -f core*.dump excp_log*.dump
    local status=0
    timeout "${run_timeout}" ./lib/test_camodel_reshape_scatter_aiv "$@" \
        2>&1 | tee "${log}" || status=$?
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' \
            "${log}"; then
        echo "CAModel reported execution errors in ${tag}" >&2
        status=1
    fi
    shopt -s nullglob
    for dump in core*.excp_log.dump excp_log*.dump; do
        [[ -s "${dump}" ]] && { echo "non-empty exception dump: ${dump}" >&2; status=1; }
    done
    shopt -u nullglob

    local dumps="dumps_scatter_${tag}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true

    # The trace audit, over this run's own dumps, unconditionally -- including
    # after a failed run, because a CAModel flag diagnostic is exactly when the
    # audit has something to say. The grid is one block, so the work is on
    # core0's first Vector core; the audit asserts that rather than assuming
    # it, and asserts that the SECOND Vector core did nothing, which is what
    # KERNEL_TYPE_AIV_ONLY with a one-block grid is supposed to mean.
    if [[ -f "${dumps}/core0.veccore0.instr_popped_log.dump" ]]; then
        python3 "${repo_dir}/scripts/analyze_scatter_trace.py" \
            "${dumps}/core0.veccore0.instr_popped_log.dump" \
            "${dumps}/core0.veccore0.instr_log.dump" \
            --aiv1 "${dumps}/core0.veccore1.instr_popped_log.dump" \
            --rows "${rows}" --runs "${runs}" --launches "${launches}" \
            | tee "audit_scatter_${tag}.txt" || status=$?
    else
        echo "no veccore0 dump for ${tag}; the audit cannot run" >&2
        ls "${dumps}" >&2 || true
        status=1
    fi
    printf '%-22s %s\n' "${tag}" \
        "$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)" >> SCATTER_SUMMARY.txt
    return ${status}
}

: > SCATTER_SUMMARY.txt
overall=0
if [[ "${1:-}" == "ladder" ]]; then
    # The spatial sweep first -- boundary, middle, boundary -- then the three
    # rungs that test what an isolated single-token launch cannot: the
    # read-modify-write across launches, the zero row, and every token count.
    # The sequential rung is 1 + 1 + 3 tokens over three launches, and the
    # sweep is n = 1..16, which is 136 rows over sixteen launches.
    for h in 0 2 3; do
        run_one "canary_H${h}" 1 1 1 -- --mode canary --head "${h}" || overall=1
    done
    run_one "sequential_H2" 5 3 3 -- --mode sequential --head 2 || overall=1
    run_one "zero_H0" 1 1 1 -- --mode zero --head 0 || overall=1
    run_one "sweep_H3" 136 16 16 -- --mode sweep --head 3 || overall=1
elif [[ $# -gt 0 ]]; then
    # An ad-hoc run: the audit reports rather than asserts, because only the
    # caller knows how many rows the arguments imply.
    run_one run 0 0 0 -- "$@" || overall=1
else
    for h in 0 2 3; do
        run_one "canary_H${h}" 1 1 1 -- --mode canary --head "${h}" || overall=1
    done
fi

echo "---- run summary ----"
cat SCATTER_SUMMARY.txt
echo "build and trace directory: ${build_dir}"
exit ${overall}
