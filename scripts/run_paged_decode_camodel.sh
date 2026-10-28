#!/usr/bin/env bash
# Build and run the paged decode attention operator on the Ascend950PR_9599
# CAModel, then audit its traces.
#
#   bash scripts/run_paged_decode_camodel.sh                 # 8 strips, block 0
#   bash scripts/run_paged_decode_camodel.sh --strips 2       # the ring depth
#   bash scripts/run_paged_decode_camodel.sh --canary 4 2     # one tile, T=4 H=2
#   bash scripts/run_paged_decode_camodel.sh ladder           # every rung
#
# Everything after the script name is passed to the runner unchanged, except
# the single word `ladder`, which runs the whole verification ladder: the strip
# ladder on one block and then the nine single-tile canary coordinates.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, PAGED_DECODE_BUILD_DIR,
# PAGED_DECODE_SRC_DIR, PAGED_DECODE_TIMEOUT.
#
# Two things this script does differently from the other probe runners here,
# both for reasons that were paid for once
# ----------------------------------------------------------------------------
# 1. It copies the tree into the WSL ext4 filesystem and builds there instead
#    of building out of /mnt/c. Under WSL the drvfs page cache can serve a
#    stale copy of a source file that a Windows-side editor has just rewritten,
#    and BiSheng then compiles code that no longer exists: the symptom was
#    template errors quoting text already deleted from the file, with the stale
#    text present only inside CANN's own preprocess object files. Reading the
#    sources through WSL first did not help and neither did a fresh build
#    directory. It is also several times faster.
# 2. It always builds from scratch. CANN's preprocess stage links each device
#    object over ITSELF (`ld.lld ... -o <the same .o>`), so a second pass in
#    the same build directory hands ld.lld a file it has already rewritten and
#    the build dies with "unknown file type". SKELETON_KERNEL_WORKFLOW.md says
#    to use a fresh build directory per full build; this is what goes wrong if
#    you do not.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${PAGED_DECODE_SRC_DIR:-/tmp/ascend-paged-decode-src}"
build_dir="${PAGED_DECODE_BUILD_DIR:-/tmp/ascend-paged-decode-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
# One strip is about seven minutes of cycle-accurate model time on this host
# and eight strips about fifteen, so the timeout is generous by default.
run_timeout="${PAGED_DECODE_TIMEOUT:-2400s}"

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
# The host oracle first: a CAModel run compares the device against it, so a
# wrong oracle would be indistinguishable from a wrong kernel. Sub-second.
mkdir -p "${repo_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_paged_decode_probe.cpp" \
    -o "${repo_dir}/bin/test_paged_decode_probe"
"${repo_dir}/bin/test_paged_decode_probe"

# ---------------------------------------------------------------- build
mkdir -p "${src_dir}"
rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
    "${repo_dir}/" "${src_dir}/"
rm -rf "${build_dir}"
cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
    -DASCEND_HOME_PATH="${toolkit}"
cmake --build "${build_dir}" --target test_camodel_paged_decode_attention -j 4

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
cd "${build_dir}"

# ---------------------------------------------------------------- run
# The directly linked 950PR CAModel drops its per-core instruction dumps in the
# current directory, so each run archives them under its own tag before the
# next run overwrites them.
run_one() {
    local tag="$1"; shift
    local log="camodel_paged_decode_${tag}.log"
    rm -f core*.dump excp_log*.dump
    local status=0
    timeout "${run_timeout}" ./lib/test_camodel_paged_decode_attention "$@" \
        2>&1 | tee "${log}" || status=$?
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' "${log}"; then
        echo "CAModel reported execution errors in ${tag}" >&2
        status=1
    fi
    # The model writes core<N>.excp_log.dump, one per core in the grid, so
    # the bare excp_log*.dump glob this loop used to carry matched nothing and
    # the census never fired at all. nullglob keeps an unmatched pattern from
    # arriving as a literal filename.
    shopt -s nullglob
    for dump in core*.excp_log.dump excp_log*.dump; do
        [[ -s "${dump}" ]] && { echo "non-empty exception dump: ${dump}" >&2; status=1; }
    done
    shopt -u nullglob
    # Archive every core's instruction logs, not core 0's. The grid is 32
    # blocks, the model dumps all 32 cores, and block N runs on core N: a run
    # that isolates block 16 leaves core 0 holding nothing but the early
    # return. Keeping all of them makes auditing the wrong core a decision
    # rather than an accident. The ccu and dcache logs stay behind -- nothing
    # here reads them, and the active core's ccu log alone is 30 MB.
    local dumps="dumps_${tag}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true

    # The trace audit, over this run's own dumps. The strip count scales every
    # per-strip assertion in it, so it has to match the launch.
    local strips=8
    local i
    for ((i = 1; i <= $#; ++i)); do
        if [[ "${!i}" == "--strips" ]]; then strips="${@:i+1:1}"; fi
        if [[ "${!i}" == "--canary" ]]; then strips=1; fi
    done
    # Unconditionally, including after a failed run: a CAModel flag
    # diagnostic is exactly when the trace audit has something to say.
    # Skipping it on failure would have hidden sections 4.6 and 4.8.
    # Which core to read: the one the runner says it isolated. Taking that
    # from the runner's own report rather than re-deriving it from the
    # arguments here keeps the block -> core mapping under test instead of
    # assumed. If the work ever lands on another core, the audit's first check
    # fails and names the reason, which is how this defect surfaced.
    local block core
    block=$(sed -n 's/.*block \([0-9][0-9]*\) isolated.*/\1/p' "${log}" | head -1)
    core="core${block:-0}"
    if true; then
        python3 "${repo_dir}/scripts/analyze_paged_decode_trace.py" \
            --strips "${strips}" \
            "${dumps}/${core}.veccore0.instr_popped_log.dump" \
            "${dumps}/${core}.veccore0.instr_log.dump" \
            "${dumps}/${core}.veccore1.instr_popped_log.dump" \
            "${dumps}/${core}.veccore1.instr_log.dump" \
            "${dumps}/${core}.cubecore0.instr_popped_log.dump" \
            "${dumps}/${core}.cubecore0.instr_log.dump" \
            | tee "camodel_paged_decode_${tag}_trace.txt" || status=$?
    fi
    printf '%-18s %-5s audited %s\n' "${tag}" \
        "$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)" "${core}" \
        >> LADDER_SUMMARY.txt
    return ${status}
}

: > LADDER_SUMMARY.txt
if [[ "${1:-}" == "ladder" ]]; then
    # The UB rings are two slots deep, so: 2 strips is the depth boundary
    # and
    # reuses no slot, 4 is kSlots + 2 and is the smallest grid that fires
    # every structural branch once, 8 wraps every ring four times and is the
    # only rung that yields a steady-state interval. Then the nine specified
    # canary coordinates.
    #
    # Run the whole ladder, not just the canary coordinates. All nine of
    # them passed while two separate ring-reuse faults were live, because
    # one strip reuses no slot: it took --strips 4 and --strips 8 to
    # expose them. reports/paged_decode_attention.md sections 4.5, 4.6.
    overall=0
    for strips in 2 4 8; do
        run_one "strips${strips}" --strips "${strips}" --block 0 \
            --no-fixture-check || overall=1
    done
    for t in 0 4 7; do
        for h in 0 2 3; do
            run_one "canary_T${t}_H${h}" --canary "${t}" "${h}" \
                --no-fixture-check || overall=1
        done
    done
    echo "---- ladder summary ----"
    cat LADDER_SUMMARY.txt
    echo "build and trace directory: ${build_dir}"
    exit ${overall}
fi

run_one run "$@"
echo "build and trace directory: ${build_dir}"
