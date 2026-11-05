#!/usr/bin/env bash
# Build, and on a physical Ascend 950PR also run, the Split-K decode validation
# ladders -- including the multi-page L1 reuse regression added for the hazard
# reports/decode_splitk_aic_page_race.md diagnosed.
#
#   bash scripts/run_decode_splitk_npu_validation.sh build        # no device needed
#   bash scripts/run_decode_splitk_npu_validation.sh run [ARGS]   # needs a 950PR
#   bash scripts/run_decode_splitk_npu_validation.sh all [ARGS]   # build then run
#   bash scripts/run_decode_splitk_npu_validation.sh sim [ARGS]   # the CAModel twin
#
# `run` REFUSES to proceed without a device rather than failing obscurely
# inside aclrtSetDevice, and says what it checked. On a host with no NPU
# `build` establishes that the ladder compiles and links against the
# physical-device kernel libraries, and nothing about what the device computes
# -- and `sim` is then the only mode that actually executes a rung, on CAModel
# over a collapsed grid. A CAModel result is not a hardware claim.
#
# ARGS are passed to the driver. `--only l1-reuse` runs just the multi-page L1
# reuse section, which is what makes the CAModel twin usable at all: every rung
# there is a real launch on the model and the full ladder is about ninety of
# them. A narrowed run prints `SECTIONS RUN: ... NARROWED` and is not the gate.
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, NPU_SRC_DIR, NPU_BUILD_DIR,
# NPU_TIMEOUT.
#
# Build discipline, as everywhere in this harness
# -----------------------------------------------
# Copy into the WSL ext4 filesystem rather than building out of /mnt/c: the
# drvfs page cache can serve BiSheng a stale copy of a source a Windows-side
# editor has just rewritten, and BiSheng then compiles code that no longer
# exists. And always build into a FRESH directory, because CANN's preprocess
# stage links each device object over ITSELF and a second pass in the same
# directory dies with "unknown file type".
#
# On a native Linux NPU host neither applies and `NPU_SRC_DIR` can simply be
# the checkout; the rsync is then a no-op copy and costs only disk.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
src_dir="${NPU_SRC_DIR:-/tmp/ascend-npu-validation-src}"
build_dir="${NPU_BUILD_DIR:-/tmp/ascend-npu-validation-build}"
run_timeout="${NPU_TIMEOUT:-1800s}"
mode="${1:-all}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"

# Both validation binaries: the AIC (Cube) Split-K path, which carries the
# multi-page ladder, and the AIV path beside it, so a change to the shared rig
# cannot pass one leg and break the other unnoticed.
targets=(test_npu_decode_splitk_aic test_npu_decode_splitk_aiv)
# The CAModel twins of the same two drivers.
sim_targets=(test_decode_splitk_aic_sim test_decode_splitk_aiv_sim)
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"

have_device() {
    command -v npu-smi >/dev/null 2>&1 && compgen -G "/dev/davinci*" >/dev/null 2>&1
}

configure_and_build() {
    mkdir -p "${src_dir}"
    rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
        "${repo_dir}/" "${src_dir}/"
    rm -rf "${build_dir}"
    cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build_dir}" --target "$@" -j 4
}

do_build() {
    configure_and_build "${targets[@]}"
    echo
    for target in "${targets[@]}"; do
        echo "built: ${build_dir}/lib/${target}"
    done
    echo "These binaries need a 950PR. Nothing was executed."
}

# The CAModel twin of the AIC driver: the only mode that executes a rung on a
# host with no NPU. The model drops its per-core instruction dumps in the
# current directory, so the run gets its own evidence directory.
do_sim() {
    if [[ ! -f "${sim_lib_dir}/libnpu_drv_camodel.so" ]]; then
        echo "CAModel runtime not found: ${sim_lib_dir}" >&2
        exit 2
    fi
    configure_and_build "${sim_targets[0]}"
    export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
    local evidence="${build_dir}/ev_sim"
    mkdir -p "${evidence}"
    cd "${evidence}"
    rm -f core*.dump excp_log*.dump
    local status=0
    timeout "${run_timeout}" "${build_dir}/lib/${sim_targets[0]}" "$@" 2>&1 \
        | tee "${evidence}/sim_run.log" || status=$?
    echo
    echo "=== nonempty exception dumps (must be none) ==="
    find "${evidence}" -name '*excp*.dump' -size +0 -print | sed 's/^/    /' || true
    echo "evidence: ${evidence}"
    return "${status}"
}

do_run() {
    if ! have_device; then
        echo "No NPU on this host: npu-smi and /dev/davinci* are both absent." >&2
        echo "Nothing was executed. Use the 'build' mode here and run this on a" >&2
        echo "950PR host." >&2
        exit 3
    fi
    export LD_LIBRARY_PATH="${build_dir}/lib:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
    local overall=0
    for target in "${targets[@]}"; do
        echo "=== ${target} ==="
        # The ladders are compiled in, so a rung cannot be skipped by a
        # missing flag; `--only` can narrow deliberately and says so.
        timeout "${run_timeout}" "${build_dir}/lib/${target}" "$@" || overall=1
        echo
    done
    return "${overall}"
}

shift || true
case "${mode}" in
build) do_build ;;
run)   do_run "$@" ;;
sim)   do_sim "$@" ;;
all)   do_build && do_run "$@" ;;
*)     echo "usage: $0 {build|run|sim|all} [driver args]" >&2; exit 2 ;;
esac
