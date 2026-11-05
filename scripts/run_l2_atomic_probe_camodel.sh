#!/usr/bin/env bash
# Build and run the Split-K synchronisation probe and the single-tile Cube
# baseline on the Ascend950PR_9599 CAModel, then audit their traces.
#
#   bash scripts/run_l2_atomic_probe_camodel.sh            # the K sweep, modes A/B/C
#   bash scripts/run_l2_atomic_probe_camodel.sh sweep      # the same, explicitly
#   bash scripts/run_l2_atomic_probe_camodel.sh baseline   # the Cube baseline only
#   bash scripts/run_l2_atomic_probe_camodel.sh all        # both, then the crossover
#   bash scripts/run_l2_atomic_probe_camodel.sh smoke      # one cheap rung
#   bash scripts/run_l2_atomic_probe_camodel.sh --mode b --splits 8 --reps 4
#
# Anything that is not one of those words is passed to the probe runner
# unchanged, as a single ad-hoc launch.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, L2ATOMIC_BUILD_DIR,
# L2ATOMIC_SRC_DIR, L2ATOMIC_TIMEOUT, L2ATOMIC_SKIP_BUILD.
#
# Two things this does the way the other probe runners here do, both for
# reasons already paid for once: it copies the tree into the WSL ext4
# filesystem rather than building out of /mnt/c, because the drvfs page cache
# can serve BiSheng a stale copy of a source a Windows-side editor has just
# rewritten; and it always builds from scratch, because CANN's preprocess
# stage links each device object over ITSELF and a second pass in the same
# build directory dies with "unknown file type".
#
# What the sweep is for
# ---------------------
# The probe measures T_sync(K) and the Cube baseline measures
# T_compute_strip. Neither is the answer: the answer is the tile size at which
# the first drops below 3% of the second, and
# `scripts/l2_atomic_crossover.py` computes that from the two runs' outputs.
# K=1 is in the sweep although the task asks for K in {2, 4, 8}, because an
# uncontended lock is the only baseline against which the contended ones mean
# anything -- without it, every cycle of a K=8 acquisition could be dispatch
# overhead rather than contention, and nothing in the K>=2 runs could tell.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${L2ATOMIC_SRC_DIR:-/tmp/ascend-l2atomic-src}"
build_dir="${L2ATOMIC_BUILD_DIR:-/tmp/ascend-l2atomic-build}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
# The synthetic probe is tens of seconds; the single-tile decode baseline is
# minutes, and the eight-strip rung about fifteen on this host.
run_timeout="${L2ATOMIC_TIMEOUT:-2400s}"

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

mode_word="${1:-sweep}"
case "${mode_word}" in
    sweep|baseline|all|smoke) shift || true ;;
    *) mode_word="adhoc" ;;
esac

# ---------------------------------------------------------------- Tier 1
# The host oracles first. A CAModel run compares the device against them, so a
# wrong oracle is indistinguishable from a wrong kernel -- and it is these
# probes, not the simulator, that establish that the uniform fixture's combine
# really is order-independent and bit-exact, which is the only reason the
# device gate can be `==` rather than a tolerance. Milliseconds each.
mkdir -p "${repo_dir}/bin"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
    "${repo_dir}/tests/cpu/test_l2_atomic_probe.cpp" \
    -o "${repo_dir}/bin/test_l2_atomic_probe"
"${repo_dir}/bin/test_l2_atomic_probe"
if [[ "${mode_word}" == "baseline" || "${mode_word}" == "all" ]]; then
    # The decode operator's own oracle, for the baseline's bit-exact gates.
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
        "${repo_dir}/tests/cpu/test_paged_decode_probe.cpp" \
        -o "${repo_dir}/bin/test_paged_decode_probe"
    "${repo_dir}/bin/test_paged_decode_probe"
fi

# ---------------------------------------------------------------- build
if [[ -z "${L2ATOMIC_SKIP_BUILD:-}" ]]; then
    mkdir -p "${src_dir}"
    rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
        "${repo_dir}/" "${src_dir}/"
    rm -rf "${build_dir}"
    cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build_dir}" --target test_camodel_l2_atomic_synthetic -j 4
    if [[ "${mode_word}" == "baseline" || "${mode_word}" == "all" ]]; then
        cmake --build "${build_dir}" \
            --target test_camodel_cube_single_tile_baseline -j 4
    fi
fi

export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
cd "${build_dir}"
: > L2ATOMIC_SUMMARY.txt

# The directly linked 950PR CAModel drops its per-core instruction dumps in
# the current directory and overwrites them on the next run, so each run
# archives them under its own tag before the next one starts.
#
# run_probe <tag> <mode> <splits> <reps> <fixture>
run_probe() {
    local tag="$1" mode="$2" splits="$3" reps="$4" fixture="$5"
    local log="camodel_l2atomic_${tag}.log"
    rm -f core*.dump excp_log*.dump
    local status=0 started
    started=$(date +%s)
    timeout "${run_timeout}" ./lib/test_camodel_l2_atomic_synthetic \
        --mode "${mode}" --splits "${splits}" --reps "${reps}" \
        --fixture "${fixture}" 2>&1 | tee "${log}" || status=$?
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

    local dumps="dumps_l2atomic_${tag}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true
    # The CCU logs too, for THIS probe only. They carry one line per cycle in
    # which a core could not issue, which is the only stall record these dumps
    # contain, and the audit attributes those cycles to the spin and the
    # critical section. They are about a megabyte each here. The decode
    # baseline's are tens of megabytes, which is why run_baseline below does
    # not copy them.
    cp core*.veccore*.ccu_log.dump "${dumps}/" 2>/dev/null || true

    # The trace audit, unconditionally -- including after a failed run,
    # because a CAModel flag diagnostic is exactly when it has something to
    # say. It reads core<k> for every split k, and its first check is that
    # those cores ran the probe body: an idle core in this grid issues a
    # scalar prologue and an early return, on which every balance check is
    # vacuously true.
    python3 "${repo_dir}/scripts/analyze_l2_atomic_trace.py" \
        --mode "${mode}" --splits "${splits}" --reps "${reps}" "${dumps}" \
        2>&1 | tee "audit_l2atomic_${tag}.txt" || status=$?

    printf '%-28s %-5s %5ds\n' "${tag}" \
        "$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)" "${elapsed}" \
        >> L2ATOMIC_SUMMARY.txt
    return ${status}
}

# run_baseline <tag> <strips> <block> <strip>
run_baseline() {
    local tag="$1" strips="$2" block="$3" strip="$4"
    local log="camodel_cube_baseline_${tag}.log"
    rm -f core*.dump excp_log*.dump
    local status=0 started
    started=$(date +%s)
    timeout "${run_timeout}" ./lib/test_camodel_cube_single_tile_baseline \
        --strips "${strips}" --block "${block}" --strip "${strip}" \
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

    local dumps="dumps_cube_baseline_${tag}"
    rm -rf "${dumps}"; mkdir -p "${dumps}"
    cp core*.instr_log.dump core*.instr_popped_log.dump core*.excp_log.dump \
        "${dumps}/" 2>/dev/null || true

    # Which core to read is taken from the runner's own report rather than
    # re-derived here, which keeps the block -> core mapping under test instead
    # of assumed.
    local core
    core=$(sed -n 's/.*block \([0-9][0-9]*\) isolated on core \([0-9][0-9]*\).*/\2/p' \
        "${log}" | head -1)
    python3 "${repo_dir}/scripts/analyze_cube_tile_baseline.py" \
        --strips "${strips}" --core "${core:-0}" "${dumps}" \
        2>&1 | tee "audit_cube_baseline_${tag}.txt" || status=$?

    printf '%-28s %-5s %5ds\n' "baseline_${tag}" \
        "$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)" "${elapsed}" \
        >> L2ATOMIC_SUMMARY.txt
    return ${status}
}

overall=0
case "${mode_word}" in
smoke)
    # The cheapest rung that exercises the whole Mode B protocol: two splits,
    # two acquisitions each. Run this first to time a rung before committing
    # to the sweep.
    run_probe "smoke_b_K2" b 2 2 uniform || overall=1
    ;;
sweep|all)
    # Mode A at every K. Nothing contends, so this is the data-movement floor
    # that Mode B's cost has to be measured against: same two bursts, same
    # record, no lock.
    for k in 1 2 4 8; do
        run_probe "mode_a_K${k}" a "${k}" 4 uniform || overall=1
    done
    # Mode B at every K, uniform fixture: the bit-exact lost-update gate.
    for k in 1 2 4 8; do
        run_probe "mode_b_K${k}" b "${k}" 4 uniform || overall=1
    done
    # Mode B staggered: the only fixture that exercises the rescale, and the
    # one that makes the atomic maximum's handling of negative floats
    # observable. At both ends of the K sweep.
    run_probe "mode_b_stag_K2" b 2 4 staggered || overall=1
    run_probe "mode_b_stag_K8" b 8 4 staggered || overall=1
    # Mode C, the tail split's egress. K is the launch width; only the last
    # split works, which is also what makes "no other slice was written"
    # meaningful.
    run_probe "mode_c_K8" c 8 4 uniform || overall=1
    ;;
esac

case "${mode_word}" in
baseline|all)
    # One isolated strip, for the span, at both ends of the strip axis so a
    # strip-dependent cost would show. Then the eight-strip rung, for the
    # steady-state interval, which is the crossover's denominator.
    run_baseline "strip0" 1 0 0 || overall=1
    run_baseline "strip7" 1 0 7 || overall=1
    run_baseline "strips8" 8 0 0 || overall=1
    ;;
adhoc)
    log="camodel_l2atomic_adhoc.log"
    rm -f core*.dump excp_log*.dump
    timeout "${run_timeout}" ./lib/test_camodel_l2_atomic_synthetic "$@" \
        2>&1 | tee "${log}" || overall=$?
    ;;
esac

if [[ "${mode_word}" == "all" ]]; then
    echo "---- crossover ----"
    python3 "${repo_dir}/scripts/l2_atomic_crossover.py" "${build_dir}" \
        | tee "crossover.txt" || overall=1
fi

echo "---- run summary ----"
cat L2ATOMIC_SUMMARY.txt
echo "build and trace directory: ${build_dir}"
exit ${overall}
