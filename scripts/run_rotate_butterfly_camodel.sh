#!/usr/bin/env bash
# Benchmark the three randomised-Hadamard butterfly implementations on the
# Ascend950PR_9599 CAModel and report their per-engine censuses side by side.
#
#   bash scripts/run_rotate_butterfly_camodel.sh                  # all three, 16 rows
#   bash scripts/run_rotate_butterfly_camodel.sh --rows 1          # the decode shape
#   bash scripts/run_rotate_butterfly_camodel.sh --variant gather   # one of them
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, BUTTERFLY_BUILD_DIR,
# BUTTERFLY_SRC_DIR, BUTTERFLY_TIMEOUT.
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
src_dir="${BUTTERFLY_SRC_DIR:-/tmp/ascend-butterfly-src}"
build_dir="${BUTTERFLY_BUILD_DIR:-/tmp/ascend-butterfly-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
run_timeout="${BUTTERFLY_TIMEOUT:-1800s}"

rows=16
reps=1
variants=(blocked gather shuffle)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rows) rows="$2"; shift 2 ;;
        --reps) reps="$2"; shift 2 ;;
        --variant) variants=("$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

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
# The host probe first. A CAModel run compares the device against this oracle,
# so a wrong oracle would be indistinguishable from a wrong kernel -- and it is
# this probe, not the simulator, that establishes the (P.B_0)^6 == B_5...B_0
# identity the shuffle leaf is built on. Sub-second.
mkdir -p "${repo_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_rotate_butterfly_probe.cpp" \
    -o "${repo_dir}/bin/test_rotate_butterfly_probe"
"${repo_dir}/bin/test_rotate_butterfly_probe"

# ---------------------------------------------------------------- build
mkdir -p "${src_dir}"
rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
    "${repo_dir}/" "${src_dir}/"
rm -rf "${build_dir}"
cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
    -DASCEND_HOME_PATH="${toolkit}"
cmake --build "${build_dir}" --target test_camodel_rotate_butterfly -j 4

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
cd "${build_dir}"

status=0
summaries=()
for variant in "${variants[@]}"; do
    log="camodel_butterfly_${variant}.log"
    rm -f core*.dump excp_log*.dump
    run_status=0
    timeout "${run_timeout}" ./lib/test_camodel_rotate_butterfly \
        --variant "${variant}" --rows "${rows}" --reps "${reps}" \
        2>&1 | tee "${log}" || run_status=$?
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr|su_ccu_mpu_err' \
            "${log}"; then
        echo "CAModel reported execution errors in ${variant}" >&2
        run_status=1
    fi
    shopt -s nullglob
    for dump in core*.excp_log.dump excp_log*.dump; do
        [[ -s "${dump}" ]] && { echo "non-empty exception dump: ${dump}" >&2; run_status=1; }
    done
    shopt -u nullglob

    dumps="dumps_butterfly_${variant}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true

    # The grid is one block, so the work is on core0. Unconditionally, including
    # after a failed run: a CAModel diagnostic is exactly when the census has
    # something to say.
    summary="summary_butterfly_${variant}.json"
    if [[ -f "${dumps}/core0.veccore0.instr_popped_log.dump" ]]; then
        python3 "${repo_dir}/scripts/analyze_butterfly_trace.py" \
            --label "${variant}" --rows "${rows}" --reps "${reps}" \
            --json "${summary}" \
            "${dumps}/core0.veccore0.instr_popped_log.dump" \
            "${dumps}/core0.veccore0.instr_log.dump" \
            | tee "census_butterfly_${variant}.txt" || run_status=$?
        summaries+=("${summary}")
    else
        echo "no veccore0 dump for ${variant}; the census cannot run" >&2
        ls "${dumps}" >&2 || true
        run_status=1
    fi
    printf '%-10s %s\n' "${variant}" \
        "$([[ ${run_status} -eq 0 ]] && echo PASS || echo FAIL)" >> BUTTERFLY_SUMMARY.txt
    [[ ${run_status} -eq 0 ]] || status=1
done

echo "---- variant summary ----"
cat BUTTERFLY_SUMMARY.txt
if [[ ${#summaries[@]} -gt 1 ]]; then
    echo "---- engine census comparison ----"
    python3 "${repo_dir}/scripts/analyze_butterfly_trace.py" --compare "${summaries[@]}" \
        | tee butterfly_comparison.txt
fi
echo "build and trace directory: ${build_dir}"
exit ${status}
