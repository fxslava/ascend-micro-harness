#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
toolkit="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0-beta.2}"
soc="${SOC_VERSION:-Ascend950PR_9599}"
build_dir="${MODULO3_BUILD_DIR:-$(mktemp -d /tmp/ascend-modulo3-camodel-XXXXXX)}"
sim_lib_dir="${toolkit}/tools/simulator/${soc}/lib"
# A probe that issues more device instructions needs more wall clock in the
# cycle-accurate model; the reshape-and-cache operator runs 64 Mmads and eight
# chunks of Vector work against the fused probe's eight Mmads.
run_timeout="${MODULO3_TIMEOUT:-120s}"
mode="${1:-all}"
micro_name="${2:-dequant4}"

# The reshape-and-cache operator is addressable by its short name too, so the
# probe name does not have to repeat the operator family.
if [[ "${micro_name}" == "reshape_and_cache_a8" ]]; then
    micro_name="turboquant_reshape_and_cache_a8"
fi

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
    if [[ "${mode}" != "micro" ]]; then
        mkdir -p "${repo_dir}/bin"
        g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${repo_dir}/include" \
            "${repo_dir}/tests/cpu/test_pipeline_deadlock_probe.cpp" \
            -o "${repo_dir}/bin/test_pipeline_deadlock_probe"
        "${repo_dir}/bin/test_pipeline_deadlock_probe" \
            "${repo_dir}/bin/pipeline_queue_trace.csv"
    fi

    cmake -S "${repo_dir}" -B "${build_dir}" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=sim -DSOC_VERSION="${soc}" \
        -DASCEND_HOME_PATH="${toolkit}"
    if [[ "${mode}" == "micro" ]]; then
        cmake --build "${build_dir}" --target "test_camodel_${micro_name}" -j 4
    else
        cmake --build "${build_dir}" --target test_pipeline_deadlock_probe_camodel -j 4
    fi
fi

# The directly linked 950PR CAModel emits per-core instruction and popped
# queue logs in its current directory. npusim's generic Ascend950 wrapper
# selects a different simulator library set on this installation.
cd "${build_dir}"
if [[ "${mode}" == "micro" ]]; then
    timeout "${run_timeout}" "./test_camodel_${micro_name}" 2>&1 | tee "camodel_${micro_name}_run.log"
    if grep -Eiq '\[(error|fatal)\]|invalid ldst addr|unrecognize ldst addr' "camodel_${micro_name}_run.log"; then
        echo "CAModel reported execution errors" >&2
        exit 1
    fi
    python3 "${repo_dir}/scripts/analyze_micro_trace.py" \
        core0.veccore0.instr_popped_log.dump \
        core0.veccore0.instr_log.dump \
        core0.veccore1.instr_popped_log.dump \
        core0.veccore1.instr_log.dump \
        core0.cubecore0.instr_popped_log.dump \
        core0.cubecore0.instr_log.dump | tee "camodel_${micro_name}_trace_summary.txt"
    # The fused probe is the only one that drives the Cube core and both Vector
    # cores together, so it also gets the event-balance, CV-path, steady-state
    # interval and engine-overlap audit over all three per-core dumps. The page
    # count is part of the target name and every per-page assertion in the
    # analyzer is scaled by it.
    if [[ "${micro_name}" == fused_attention_*tiles ]]; then
        fused_tiles="${micro_name#fused_attention_}"
        fused_tiles="${fused_tiles%tiles}"
        python3 "${repo_dir}/scripts/analyze_fused_trace.py" --tiles "${fused_tiles}" \
            core0.veccore0.instr_popped_log.dump \
            core0.veccore0.instr_log.dump \
            core0.veccore1.instr_popped_log.dump \
            core0.veccore1.instr_log.dump \
            core0.cubecore0.instr_popped_log.dump \
            core0.cubecore0.instr_log.dump \
            | tee "camodel_${micro_name}_event_summary.txt"
    fi
    # The reshape-and-cache operator drives the same three cores but a
    # four-stage ring with a different flag graph, so it has its own audit
    # rather than reusing the fused attention one. Without this branch the
    # V4 checks would silently not run, which is the trap the workflow warns
    # about.
    if [[ "${micro_name}" == "turboquant_reshape_and_cache_a8" ]]; then
        python3 "${repo_dir}/scripts/analyze_reshape_trace.py" \
            core0.veccore0.instr_popped_log.dump \
            core0.veccore0.instr_log.dump \
            core0.veccore1.instr_popped_log.dump \
            core0.veccore1.instr_log.dump \
            core0.cubecore0.instr_popped_log.dump \
            core0.cubecore0.instr_log.dump \
            | tee "camodel_${micro_name}_event_summary.txt"
    fi
else
    timeout "${run_timeout}" ./test_pipeline_deadlock_probe_camodel | tee camodel_run.log
    python3 "${repo_dir}/scripts/analyze_modulo3_trace.py" \
        core0.veccore0.instr_popped_log.dump \
        core0.veccore0.instr_log.dump | tee camodel_trace_summary.txt
fi

echo "CAModel build and trace directory: ${build_dir}"
