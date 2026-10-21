#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
build_dir="${MODULO3_BUILD_DIR:-$(mktemp -d /tmp/ascend-modulo3-camodel-XXXXXX)}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
mode="${1:-all}"

if [[ ! -f "${toolkit}/set_env.sh" ]]; then
    echo "CANN environment not found: ${toolkit}/set_env.sh" >&2
    exit 2
fi
if [[ ! -f "${sim_lib_dir}/libruntime_camodel.so" ]]; then
    echo "CAModel runtime not found: ${sim_lib_dir}" >&2
    exit 2
fi

# CANN's nested ExternalProject mutates intermediate .o files during its
# preprocess phase. Use a fresh ext4 build directory for each full build.
set +u
source "${toolkit}/set_env.sh"
set -u
export ASCEND_HOME_PATH="${toolkit}"
export LD_LIBRARY_PATH="${build_dir}/lib:${sim_lib_dir}:${toolkit}/lib64:${toolkit}/devlib:${LD_LIBRARY_PATH:-}"

if [[ "${mode}" != "run-only" && "${mode}" != "profile-only" ]]; then
    mkdir -p "${repo_dir}/bin"
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
        "${repo_dir}/tests/test_pipeline_deadlock_probe.cpp" \
        -o "${repo_dir}/bin/test_pipeline_deadlock_probe"
    "${repo_dir}/bin/test_pipeline_deadlock_probe" \
        "${repo_dir}/bin/pipeline_queue_trace.csv"

    cmake -S "${repo_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=sim -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    cmake --build "${build_dir}" --target test_pipeline_deadlock_probe_camodel -j 4
fi

# The directly linked 950PR CAModel emits per-core instruction and popped
# queue logs in its current directory. npusim's generic Ascend950 wrapper
# selects a different simulator library set on this installation.
cd "${build_dir}"
timeout 120s ./test_pipeline_deadlock_probe_camodel | tee camodel_run.log
python3 "${repo_dir}/scripts/analyze_modulo3_trace.py" \
    core0.veccore0.instr_popped_log.dump \
    core0.veccore0.instr_log.dump | tee camodel_trace_summary.txt

echo "CAModel build and trace directory: ${build_dir}"
