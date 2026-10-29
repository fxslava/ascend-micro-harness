#!/usr/bin/env bash
# Build and run the AIV-only single-token decode operator (roadmap No.3) on the
# Ascend950PR_9599 CAModel, then audit its traces.
#
#   bash scripts/run_single_token_decode_camodel.sh              # the spatial sweep
#   bash scripts/run_single_token_decode_camodel.sh ladder       # every rung
#   bash scripts/run_single_token_decode_camodel.sh rest         # every rung but the sweep
#   bash scripts/run_single_token_decode_camodel.sh smoke        # one strip, head 0
#   bash scripts/run_single_token_decode_camodel.sh --head 2 --pages 4
#
# With no arguments it runs the task's spatial sweep: one isolated
# (request, kv_head) item per launch at H = 0 (boundary), H = 2 (middle) and
# H = 3 (boundary), eight strips of one page each. `ladder` adds the multi-page
# rung -- which is the only rung that can tell a kernel that walks the block
# table from one that ignores it -- the four-item concurrent grid and the row
# sweep. `smoke` is one strip and exists to measure the per-run cost before
# committing to a long rung. Anything else is passed to the runner unchanged.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, STDECODE_BUILD_DIR, STDECODE_SRC_DIR,
# STDECODE_TIMEOUT, STDECODE_SKIP_BUILD.
#
# Two things this does the way run_reshape_scatter_camodel.sh does, both for
# reasons already paid for once: it copies the tree into the WSL ext4
# filesystem rather than building out of /mnt/c, because the drvfs page cache
# can serve BiSheng a stale copy of a source a Windows-side editor has just
# rewritten; and it always builds from scratch, because CANN's preprocess stage
# links each device object over itself and a second pass in the same build
# directory dies with "unknown file type".
set -euo pipefail

# The primary logical-page boundary baseline for pipeline refactors.
if [[ "${1:-}" == boundary ]]; then
    exec bash "$(dirname "${BASH_SOURCE[0]}")/run_decode_pipeline_camodel.sh"
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${STDECODE_SRC_DIR:-/tmp/ascend-stdecode-src}"
build_dir="${STDECODE_BUILD_DIR:-/tmp/ascend-stdecode-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
run_timeout="${STDECODE_TIMEOUT:-5400s}"

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
# The host oracles first, and both of them. A CAModel run compares the device
# against this operator's oracle, so a wrong oracle is indistinguishable from a
# wrong kernel there -- and it is these probes, not the simulator, that
# establish the GEMV's repeat-stride arithmetic, the order-independence the
# bit-exact panel gate rests on, and the butterfly stage tables. Seconds each.
mkdir -p "${repo_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_rotate_butterfly_probe.cpp" \
    -o "${repo_dir}/bin/test_rotate_butterfly_probe"
"${repo_dir}/bin/test_rotate_butterfly_probe"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_single_token_decode_probe.cpp" \
    -o "${repo_dir}/bin/test_single_token_decode_probe"
"${repo_dir}/bin/test_single_token_decode_probe"

# ---------------------------------------------------------------- build
if [[ -z "${STDECODE_SKIP_BUILD:-}" ]]; then
    mkdir -p "${src_dir}"
    rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
        "${repo_dir}/" "${src_dir}/"
    rm -rf "${build_dir}"
    cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build_dir}" --target test_camodel_single_token_decode_aiv -j 4
fi

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
cd "${build_dir}"

# ---------------------------------------------------------------- run
# The directly linked 950PR CAModel drops its per-core instruction dumps in the
# current directory, so each run archives them under its own tag before the
# next run overwrites them.
#
# run_one <tag> <rows> <strips> <items> -- <runner arguments>
#
# rows/strips/items are the TOTALS the dump will cover, because the model
# writes its instruction dumps when the process stops: a runner mode that
# performs several launches leaves one dump covering all of them. They are what
# makes the audit's counts assertions rather than observations.
run_one() {
    local tag="$1"; shift
    local rows="$1"; shift
    local strips="$1"; shift
    local items="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    local log="camodel_stdecode_${tag}.log"
    rm -f core*.dump excp_log*.dump
    local status=0
    local started
    started=$(date +%s)
    timeout "${run_timeout}" ./lib/test_camodel_single_token_decode_aiv "$@" \
        2>&1 | tee "${log}" || status=$?
    local elapsed=$(( $(date +%s) - started ))
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

    local dumps="dumps_stdecode_${tag}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true

    # The trace audit, over this run's own dumps, unconditionally -- including
    # after a failed run, because a CAModel flag diagnostic is exactly when the
    # audit has something to say.
    #
    # Which core to read is NOT assumed. In the isolated runs the grid is one
    # block, so the work is on core0's first Vector core; the audit asserts
    # that the core it reads did the work that core is defined by, because an
    # idle core in a CAModel dump is not an empty one -- it issues a scalar
    # prologue and an early return, and every balance check is vacuously true
    # on it. That trap cost the Cube decode operator six audited rungs.
    if [[ -f "${dumps}/core0.veccore0.instr_popped_log.dump" ]]; then
        python3 "${repo_dir}/scripts/analyze_single_token_decode_trace.py" \
            "${dumps}/core0.veccore0.instr_popped_log.dump" \
            "${dumps}/core0.veccore0.instr_log.dump" \
            --aiv1 "${dumps}/core0.veccore1.instr_popped_log.dump" \
            --rows "${rows}" --strips "${strips}" --items "${items}" \
            | tee "audit_stdecode_${tag}.txt" || status=$?
    else
        echo "no veccore0 dump for ${tag}; the audit cannot run" >&2
        ls "${dumps}" >&2 || true
        status=1
    fi
    printf '%-24s %-5s %6ds\n' "${tag}" \
        "$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)" "${elapsed}" \
        >> STDECODE_SUMMARY.txt
    return ${status}
}

: > STDECODE_SUMMARY.txt
overall=0
mode="${1:-}"
if [[ "${mode}" == "smoke" ]]; then
    # One strip, one item: the cheapest run that exercises every stage, and the
    # one to time before committing to a longer rung.
    run_one "smoke_H0" 6 1 1 -- --head 0 --strips 1 || overall=1
elif [[ "${mode}" == "ladder" || "${mode}" == "rest" ]]; then
    # `ladder` is every rung; `rest` is every rung EXCEPT the spatial sweep, for
    # continuing after a sweep that has already passed. A CAModel run is tens of
    # minutes, so re-running a rung that passed is not free.
    if [[ "${mode}" == "ladder" ]]; then
        for h in 0 2 3; do
            run_one "canary_H${h}" 6 8 1 -- --head "${h}" || overall=1
        done
    fi
    # The multi-page rung (roadmap gate 3.1.1 asks for one page AND four): four
    # pages through a non-identity block table, which is the only rung where
    # reading the wrong page is observable, and the only one that exercises the
    # branch where the CARRIED maximum wins.
    run_one "pages4_H1" 6 32 1 -- --head 1 --pages 4 || overall=1
    # The row sweep. `rows` is a runtime bound in the kernel, so these rungs
    # check that the bound is really runtime -- and together with the 6-row
    # canary rungs they give three points on the audit's RV_VMULA derivation,
    # which is linear in the row count, so 48 / 144 / 288 and nothing else.
    #
    # 1 and 3, NOT 1 and 8. The kernel's kMaxRows = 8 is a Unified Buffer
    # bound; the ORACLE models 6 rows, because that is H_q / H_kv. A rung at
    # --rows 8 drove the device correctly and then compared it against host
    # memory past the end of the oracle's panel and state vectors, reporting
    # 256 "wrong" entries and a largest |acc| of 1.7e35 that came from outside
    # the allocation. The runner now refuses --rows above 6 for that reason.
    run_one "rows1_H0" 1 8 1 -- --head 0 --rows 1 || overall=1
    run_one "rows3_H0" 3 8 1 -- --head 0 --rows 3 || overall=1
    # The concurrent grid: four items in one launch, one per physical AIV. Last,
    # because it is the only rung where the companion Vector core is expected to
    # work and so the only one that relaxes an audit check.
    run_one "grid_all" 6 8 4 -- --all || overall=1
elif [[ $# -gt 0 ]]; then
    # An ad-hoc run: the audit reports rather than asserts, because only the
    # caller knows how many strips the arguments imply.
    run_one run 0 0 0 -- "$@" || overall=1
else
    for h in 0 2 3; do
        run_one "canary_H${h}" 6 8 1 -- --head "${h}" || overall=1
    done
fi

echo "---- run summary ----"
cat STDECODE_SUMMARY.txt
echo "build and trace directory: ${build_dir}"
exit ${overall}
