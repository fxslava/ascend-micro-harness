cmake_minimum_required(VERSION 3.16)
project(ascend_modulo3_deadlock_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "CAModel SoC")
set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.2.0-beta.2"
    CACHE PATH "CANN toolkit root")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")

# CANN 9.2 adds its own headers with -I and emits hundreds of warnings under
# -Wall. Reclassifying those same directories as system headers for BiSheng is
# what lets -Werror stay strict for our kernels and the baseline source.
#
# Defined once, here, because both target modules need the identical list and
# a divergence between them is silent: the simulator leg and the physical leg
# compile the same kernel sources, so a path added to one list and not the
# other turns -Werror on in one leg only, and the leg that starts failing is
# whichever one a given host can actually build. SimTargets.cmake's copy
# carried the comment "matching the physical-device target", which is the
# property this makes structural instead of aspirational.
set(ASCEND_MICRO_CANN_SYSTEM_OPTIONS "")
foreach(path IN ITEMS
    asc/impl/adv_api asc/impl/basic_api asc/impl/c_api
    asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils
    asc asc/include asc/include/adv_api asc/include/basic_api
    asc/include/aicpu_api asc/include/c_api
    asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils
    tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
  list(APPEND ASCEND_MICRO_CANN_SYSTEM_OPTIONS
      "-isystem${ASCEND_HOME_PATH}/x86_64-linux/${path}")
endforeach()

# The Cube GEMM reduction for the fused decode operator's reduction phase
# (TurboQuantBatchedGemmReducer, kernels/ops/turboquant/fused_decode/cube/). ON by
# default: the reducer's 27264 UB bytes come from a TBufPool aliased onto the
# attention phase's dead 32768-byte transient pool, so the 19648-byte deficit that
# used to make this switch a static_assert failure is paid without spending
# headroom. reports/gemm_batched_reduction_camodel_20261104.md carries the
# arithmetic, the overlay's CAModel rungs and what is NOT established; the ladder
# that gates a change to this default is scripts/gemm_reduce_switch_gate.sh.
#
# Defined here, once, rather than in SimTargets.cmake, for the same reason
# ASCEND_MICRO_CANN_SYSTEM_OPTIONS is: both target modules compile the identical
# kernel source, so an option visible to one leg only would build the Cube GEMM
# reduction on the simulator and the vector reducer on the physical device -- two
# different operators behind one flag, and whichever leg a given host can build
# decides which one it gets. Turn it off with -DTQ_GEMM_REDUCE=OFF and both legs
# fall back to TurboQuantPartialReducer together.
option(TQ_GEMM_REDUCE "Drive TurboQuantFusedDecodeOpt's reduction with the Cube GEMM" ON)
set(TQ_FUSED_DECODE_REDUCE_DEFINES "")
if(TQ_GEMM_REDUCE)
  set(TQ_FUSED_DECODE_REDUCE_DEFINES -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1)
endif()

# ------------------------------------------------------------------ build time
# WHAT ONE ascendc_library COSTS, MEASURED
#
# It is not one compile of the kernel. CANN's legacy pipeline
# (tools/tikcpp/ascendc_kernel_cmake/legacy_modules/function.cmake) expands each
# ascendc_library into nested ExternalProjects, and its device_preprocess_project
# builds THREE libraries out of the same source -- preprocess_obj with -E, aiv_obj
# and aic_obj -- before the separate aic_device and aiv_device projects compile the
# generated translation unit and the host project compiles the kernel again. A clean
# build of test_camodel_fused_decode_opt on this harness's development host measures
# 2479 s, over seven compiles that each pull the whole kernel in: five of the kernel
# source itself (precompile_obj, preprocess_obj, aic_obj, aiv_obj, host_bisheng_obj)
# and two of the generated translation unit that includes it (the aic and aiv
# device_obj). The fused decode operator is the expensive one because it pulls the
# cube, vector and reduction services in through templates.
#
# Every one of those ExternalProjects carries BUILD_ALWAYS TRUE, and
# cmake/CannRebuild.cmake must delete the preprocess core objects before each build:
# CANN links them in place, turning ET_REL into ET_EXEC, and its always-run merge
# step then cannot link them a second time -- `ld.lld -m aicorelinux -Ttext=0 x.o -o
# x.o` on an ET_EXEC answers "unknown file type", which is why that step exists and
# why it cannot simply be dropped. So aiv_obj and aic_obj are recompiled on EVERY
# build whether anything changed or not, and that is CANN's design rather than
# something this repository can switch off.
#
# WHAT CAN BE DONE ABOUT IT
#
# Those recompiles are the same command over the same unchanged input, which is
# exactly what a compiler cache is for, and CANN forwards
# CMAKE_C_COMPILER_LAUNCHER/CMAKE_CXX_COMPILER_LAUNCHER onto the configure line of
# every one of its nested projects -- it is the one hook that reaches the compiles
# that cost the time. With a cache present the duplicated compiles inside a single
# clean build and every rebuild after the first become hits.
#
# Found, never required: a host with no cache builds exactly as it did before, and
# -DASCEND_MICRO_COMPILER_CACHE=OFF disables it where one is installed but unwanted,
# which is what to pass when bisecting a suspected miscompile. An explicit
# -DCMAKE_CXX_COMPILER_LAUNCHER=... on the command line wins over this block.
#
# The other half of the cost is in how the build is INVOKED. `cmake --build` with one
# target per invocation pays the BUILD_ALWAYS recompile of every ascendc_library in
# the dependency chain once per invocation; naming every target in a single
# invocation pays it once. The scripts under scripts/ do that now, and they take -j
# from the host's core count because the aic and aiv device legs are independent
# ExternalProjects that can overlap.
option(ASCEND_MICRO_COMPILER_CACHE
    "Route the BiSheng and host compiles through ccache/sccache when one is installed" ON)
if(ASCEND_MICRO_COMPILER_CACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(ASCEND_MICRO_COMPILER_CACHE_PROGRAM NAMES ccache sccache)
  if(ASCEND_MICRO_COMPILER_CACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${ASCEND_MICRO_COMPILER_CACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${ASCEND_MICRO_COMPILER_CACHE_PROGRAM}")
    message(STATUS "Compiler cache: ${ASCEND_MICRO_COMPILER_CACHE_PROGRAM}, forwarded into "
                   "CANN's nested device projects")
  else()
    message(STATUS "Compiler cache: none found. CANN recompiles each kernel's preprocess "
                   "core objects on every build; install ccache to make those hits.")
  endif()
endif()

set(ASCENDC_CMAKE_DIR
    "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  message(FATAL_ERROR "CANN ascendc.cmake not found at ${ASCENDC_CMAKE_DIR}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

# CANN disables RPATH globally. Restore it for the runners, which load the
# adjacent kernel libraries from $ORIGIN without a build-directory search path.
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# The vLLM integration layer under kernels/ops/turboquant/vllm_integration_layer/.
# Its device legs are a SECOND compile of three operators this build already compiles -- for the
# fused decode because SimTargets/NpuTargets compile that very file (it is the single copy of
# that operator now), and for the two writers because their sources are staging copies -- and
# `ascendc_library` plus CannRebuild.cmake's retained BUILD_ALWAYS makes that cost recur on
# every build, so they are off unless asked for. Its
# device-free targets (the Tier-1 host gate, the adapter-header compile, the tiling
# descriptor) are three ordinary C++ compiles and are always built: the host gate is the only
# verification target of that subtree that runs without an NPU or the simulator, and it is
# where the UB restatement turboquant_gemm_reduce.hpp makes of
# turboquant_fused_decode_opt.hpp's figures is checked -- no production translation unit
# includes both headers, so nowhere else can check it.
#
# Declared here, next to TQ_GEMM_REDUCE and ASCEND_MICRO_CANN_SYSTEM_OPTIONS and for the same
# reason: the staged kernel sources are compiled by both target legs below, so an option
# visible to one leg only would stage one operator on the simulator and another on the
# physical device.
#   cmake -B <dir> -DTQ_VLLM_INTEGRATION_LAYER=ON
option(TQ_VLLM_INTEGRATION_LAYER
    "Build the staged vLLM integration layer's device legs as well as its host gate" OFF)

# The decode benchmark's kernel isolation, declared here for the third time for the same
# reason as the two options above: cmake/NpuTargets.cmake and cmake/SimTargets.cmake compile
# the IDENTICAL driver source (tests/npu/bench/bench_turboquant_decode_splitk.cpp), so an
# option visible to one leg only would benchmark one set of kernels on the simulator and
# another on the physical device, and whichever leg a given host can build would decide which
# comparison it gets. It used to be declared inside add_ascend_micro_npu_targets(), which runs
# AFTER add_ascend_micro_sim_targets(), so the simulator leg could not see it at all and
# hard-coded TQ_BENCH_LEGACY_DECODERS=1 instead.
#
# OFF, the default, puts exactly one kernel in both benchmark binaries: the fused decode
# operator. ON adds the AIC paged-decode comparison columns, which is what
# scripts/profile_camodel_trace.sh, scripts/run_decode_splitk_aic_page_depth.sh,
# scripts/run_decode_splitk_aic_page_race.sh and scripts/run_decode_splitk_strip_ladder.sh
# need. There is no AIV setting: that decoder is deleted.
option(TQ_BENCH_LEGACY_DECODERS
    "Benchmark the AIC split-K decoder alongside the fused decode operator" OFF)

include(cmake/SimTargets.cmake)
include(cmake/NpuTargets.cmake)
include(cmake/IntegrationTargets.cmake)

# The research probes under tests/research are registered inside these two
# functions rather than through add_subdirectory, because RUN_MODE is set per
# function: a subdirectory included here would pick up whichever mode happened
# to be global and would give one build of the probe instead of the hardware
# and simulator pair the evidence needs.
add_ascend_micro_sim_targets()
add_ascend_micro_npu_targets()

# The staged integration layer, for the same reason and in the same shape: its host targets
# carry no RUN_MODE at all, and each device leg sets its own, so the two legs are registered
# by two calls rather than by one subdirectory that would pick up whichever mode was global.
add_turboquant_integration_host_targets()
if(TQ_VLLM_INTEGRATION_LAYER)
  add_turboquant_integration_targets(sim)
  add_turboquant_integration_targets(npu)
endif()

include(cmake/CannRebuild.cmake)
add_ascend_micro_cann_rebuild_steps()
