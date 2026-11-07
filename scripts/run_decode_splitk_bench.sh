#!/usr/bin/env bash
# Build and run the TurboQuant Split-K decode benchmark.
#
#   bash scripts/run_decode_splitk_bench.sh build       # NPU build only (no device needed)
#   bash scripts/run_decode_splitk_bench.sh hw [args]   # build and run on a 950PR
#   bash scripts/run_decode_splitk_bench.sh sim [args]  # build and run the simulator twin
#   bash scripts/run_decode_splitk_bench.sh smoke       # the cheap functional smoke, on sim
#
# Overrides: ASCEND_HOME_PATH, SOC_VERSION, SPLITK_BUILD_DIR, SPLITK_SRC_DIR,
#            SPLITK_LEGACY_DECODERS=ON to build the AIC comparison columns in,
# SPLITK_TIMEOUT.
#
# Two binaries, and the difference matters
# ----------------------------------------
#   bench_turboquant_decode_splitk   the deliverable. Links the physical-device
#                                    kernel libraries and needs a 950PR. Its
#                                    numbers are measurements.
#   bench_decode_splitk_sim          the same source against the CAModel
#                                    libraries. Its numbers are model ticks and
#                                    the binary says so at startup. It exists
#                                    because the harness's CLI, statistics,
#                                    tail arithmetic and watchdog census would
#                                    otherwise never execute anywhere.
#
# `hw` refuses to run without a device rather than failing obscurely inside
# aclrtSetDevice, and says what it checked.
#
# Build discipline, as everywhere in this harness: copy into the WSL ext4
# filesystem rather than building out of /mnt/c, because the drvfs page cache
# can serve BiSheng a stale copy of a source a Windows-side editor has just
# rewritten; and always build into a FRESH directory, because CANN's preprocess
# stage links each device object over itself and a second pass in the same
# directory dies with "unknown file type".
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
# The work root. Persistent and on a local filesystem, not /tmp, which a WSL restart
# or systemd-tmpfiles deletes along with however many hours went into it.
# ASCEND_MICRO_WORK_ROOT overrides it; scripts/lib/work_root.sh has the reasoning.
. "$(dirname "${BASH_SOURCE[0]}")/lib/work_root.sh"
src_dir="${SPLITK_SRC_DIR:-$(ascend_micro_work_dir ascend-splitk-src)}"
build_dir="${SPLITK_BUILD_DIR:-$(ascend_micro_work_dir ascend-splitk-build)}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
run_timeout="${SPLITK_TIMEOUT:-3600s}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"

mode="${1:-build}"
shift || true

have_device() {
    command -v npu-smi >/dev/null 2>&1 && compgen -G "/dev/davinci*" >/dev/null 2>&1
}

configure_and_build() {
    local target="$1"
    mkdir -p "${src_dir}"
    rsync -a --delete --exclude '.git' --exclude 'bin' --exclude 'probe/prof_output' \
        "${repo_dir}/" "${src_dir}/"
    rm -rf "${build_dir}"
    # TQ_BENCH_LEGACY_DECODERS adds the AIC paged-decode comparison columns to the
    # benchmark. OFF, the default, puts only the fused decode operator in the binary, so
    # --impl aic needs this script run with SPLITK_LEGACY_DECODERS=ON. There is no AIV
    # setting: that decoder is deleted.
    cmake -S "${src_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}" \
        -DTQ_BENCH_LEGACY_DECODERS="${SPLITK_LEGACY_DECODERS:-OFF}"
    # -j from the host's core count: CANN's aic and aiv device legs are independent
    # ExternalProjects and overlap. The top-level CMakeLists.txt has the build-time
    # figures and the compiler-cache hook that makes a rebuild cheap.
    cmake --build "${build_dir}" --target "${target}" -j"${JOBS:-$(nproc)}"
}

case "${mode}" in
build)
    configure_and_build bench_turboquant_decode_splitk
    echo
    echo "built: ${build_dir}/lib/bench_turboquant_decode_splitk"
    echo "This binary needs a 950PR. Nothing was executed."
    ;;
hw)
    if ! have_device; then
        echo "No NPU on this host: npu-smi and /dev/davinci* are both absent." >&2
        echo "The hardware benchmark cannot run here. Options:" >&2
        echo "  bash $0 build   -- build it, which is what CI can check" >&2
        echo "  bash $0 sim     -- run the same harness on the simulator twin" >&2
        exit 3
    fi
    configure_and_build bench_turboquant_decode_splitk
    export LD_LIBRARY_PATH="${build_dir}/lib:${toolkit}/lib64:${LD_LIBRARY_PATH:-}"
    cd "${build_dir}"
    timeout "${run_timeout}" ./lib/bench_turboquant_decode_splitk "$@"
    ;;
sim|smoke)
    configure_and_build bench_decode_splitk_sim
    export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"
    cd "${build_dir}"
    rm -f core*.dump excp_log*.dump
    if [[ "${mode}" == "smoke" ]]; then
        # K=2 carries state across pages; K=4 includes an empty worker.
        # --single launches exactly once per selected implementation/case.
        #
        # The smoke case used to be `--page-tokens 16 ... --impl both`, which was the AIV
        # and AIC pair. The AIV decoder is deleted and the fused operator does not take
        # 16-token pages, so this is the same two properties on the implementation that is
        # left: three 64-token pages, so K=2 carries state across a page boundary and K=4
        # leaves one worker with no pages at all.
        set -- --context 192 --page-tokens 64 --heads 1 --rows 16 \
            --splitk "2,4" --single --impl fused-opt
    fi
    timeout "${run_timeout}" ./lib/bench_decode_splitk_sim "$@"
    ;;
*)
    echo "usage: $0 {build|hw|sim|smoke} [benchmark arguments]" >&2
    exit 2
    ;;
esac
