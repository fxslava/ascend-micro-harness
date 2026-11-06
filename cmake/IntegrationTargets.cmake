# The staged vLLM integration layer, kernels/ops/turboquant/vllm_integration_layer/.
#
# WHAT THIS SUBTREE IS. Three shipping operators -- the fused KV4/FP8 decode
# (op_kernel/turboquant_fused_decode.cpp), the BF16 prefill reshape-and-cache
# (turboquant_reshape_and_cache_bf16.cpp) and the AIV scatter ingress
# (turboquant_reshape_scatter_aiv.cpp) -- together with the service headers, micro-routines,
# oracles and runners they need, arranged into the op_kernel/op_host/op_adapter layout the
# vLLM-Ascend integration expects. It carries its own self-contained CMake project under
# standalone/, which is how the Phase 1 build was verified
# (reports/turboquant_vllm_integration_staging_audit.md).
#
# THE DECODE IS NO LONGER A COPY. It is the single source of that operator; the original
# kernels/ops/turboquant/turboquant_fused_decode_opt.cpp is deleted and cmake/SimTargets.cmake
# and cmake/NpuTargets.cmake compile the file here
# (reports/fused_decode_host_tiling_audit_20261106.md). The batch and scatter sources are still
# staging copies of their originals, differing in include paths and cross-tree comment
# references only.
#
# WHY THE DEVICE LEGS ARE OFF BY DEFAULT. Building them here compiles the same three
# translation units a SECOND time per leg -- for the decode because the simulator and physical
# legs already compile this very file, and for batch and scatter because the sources are copies
# of what those legs compile -- and `ascendc_library`
# expands into a nested ExternalProject per leg (aic_device, aiv_device, host, plus precompile,
# preprocess and a merge step), with CannRebuild.cmake keeping CANN's BUILD_ALWAYS, so the cost
# is paid on EVERY build and not just the first. On the fused decode operator that is already
# the single most expensive thing in this build; SimTargets.cmake's own comments trace a
# duplicate ascendc_library that existed by accident and what it cost. Nothing is gained by
# paying it unasked: the decode has no original left to diverge from, the two staged writers
# cannot diverge from theirs without someone editing them, and when someone does, this switch
# is how the staged build is exercised:
#
#   cmake -B build -DTQ_VLLM_INTEGRATION_LAYER=ON
#
# WHAT IS ALWAYS ON. The device-free targets, which cost a single ordinary C++ compile each:
# the Tier-1 host gate, the adapter-header compile and the staged tiling descriptor. The host
# gate is the one verification target of this subtree that RUNS on a host with no NPU and no
# simulator -- every other one opens an ACL device session in main() -- and it is also where
# the cross-header UB restatement between turboquant_gemm_reduce.hpp and
# turboquant_fused_decode_opt.hpp is checked, which no production translation unit can do
# because none of them includes both.
#
# WHY THE TWO LEGS SHARE ONE FUNCTION BODY BELOW. The same reason ASCEND_MICRO_CANN_SYSTEM
# _OPTIONS and TQ_FUSED_DECODE_REDUCE_DEFINES live in the top-level CMakeLists.txt: the
# simulator leg and the physical leg compile the identical kernel sources, so an option or a
# define applied to one leg only would build two different operators behind one flag, and which
# one a given host gets would depend on which leg it can build. `add_turboquant_integration
# _targets(sim|npu)` is called once per leg with RUN_MODE set, and the per-leg difference is
# confined to the link line and the ASCEND_MICRO_BENCH_SIM define.

set(TQ_STAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/vllm_integration_layer")

set(TQ_STAGE_INCLUDE_DIRS
    "${TQ_STAGE_DIR}/include"
    "${TQ_STAGE_DIR}/op_kernel"
    "${TQ_STAGE_DIR}/op_kernel/common"
    "${TQ_STAGE_DIR}/op_kernel/cube"
    "${TQ_STAGE_DIR}/op_kernel/vector"
    "${TQ_STAGE_DIR}/routines"
    "${TQ_STAGE_DIR}/tests/common"
    "${TQ_STAGE_DIR}/tests/npu"
    "${TQ_STAGE_DIR}/tests/validation")

# The device-free targets. No RUN_MODE, no ascendc_library, no ACL: three ordinary C++
# compiles, registered once rather than per leg.
function(add_turboquant_integration_host_targets)
  add_executable(test_stage_host_gate
      "${TQ_STAGE_DIR}/tests/host/test_integration_host_gate.cpp")
  target_include_directories(test_stage_host_gate PRIVATE ${TQ_STAGE_INCLUDE_DIRS})
  target_compile_options(test_stage_host_gate PRIVATE -Wall -Wextra -Werror)

  add_executable(test_stage_adapter_headers
      "${TQ_STAGE_DIR}/tests/validation/test_adapter_headers.cpp")
  target_include_directories(test_stage_adapter_headers PRIVATE
      ${TQ_STAGE_INCLUDE_DIRS} "${TQ_STAGE_DIR}/op_adapter")
  target_compile_options(test_stage_adapter_headers PRIVATE -Wall -Wextra -Werror)

  add_library(turboquant_stage_tiling STATIC
      "${TQ_STAGE_DIR}/op_host/turboquant_tiling.cpp")
  target_include_directories(turboquant_stage_tiling PUBLIC ${TQ_STAGE_INCLUDE_DIRS})
  target_compile_options(turboquant_stage_tiling PRIVATE -Wall -Wextra -Werror)
endfunction()

# One leg of the staged device build. `leg` is sim or npu; it selects RUN_MODE, the runtime
# link line and whether the benchmarks are built as CAModel binaries.
function(add_turboquant_integration_targets leg)
  if(NOT leg STREQUAL "sim" AND NOT leg STREQUAL "npu")
    message(FATAL_ERROR "add_turboquant_integration_targets: leg must be sim or npu")
  endif()
  set(RUN_MODE "${leg}")

  # --------------------------------------------------------------- kernel libraries
  # TQ_FUSED_DECODE_REDUCE_DEFINES carries -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 while
  # TQ_GEMM_REDUCE is ON, which is the default. It reaches the staged decode kernel for the
  # same reason it reaches the production one on both legs: the staged source is a copy of
  # that file, so a staged build without the define would be the vector-reducer operator
  # wearing the shipping operator's name.
  # add_tq_stage_kernel(<target> <source> [<extra definitions>...])
  function(add_tq_stage_kernel target source)
    ascendc_library(${target} SHARED "${TQ_STAGE_DIR}/${source}")
    ascendc_compile_definitions(${target} PRIVATE ASCEND_MICRO_CANN_LAUNCH)
    # Same strictness and the same BiSheng system-header reclassification the production
    # kernels get, from the single list in the top-level CMakeLists.txt. The staged tree's own
    # include roots go on the command line because the kernel sources include their service
    # headers relative to op_kernel/ rather than to the repository root.
    ascendc_compile_options(${target} PRIVATE
        -std=c++17 -Wall -Werror ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS} ${ARGN}
        -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
        "-I${TQ_STAGE_DIR}/op_kernel"
        "-I${TQ_STAGE_DIR}/include"
        "-I${TQ_STAGE_DIR}/routines")
  endfunction()

  add_tq_stage_kernel(turboquant_stage_decode_${leg}
      "op_kernel/turboquant_fused_decode.cpp" ${TQ_FUSED_DECODE_REDUCE_DEFINES})
  add_tq_stage_kernel(turboquant_stage_batch_${leg}
      "op_kernel/turboquant_reshape_batch.cpp")
  add_tq_stage_kernel(turboquant_stage_scatter_${leg}
      "op_kernel/turboquant_reshape_scatter.cpp")

  # ------------------------------------------------------------------ the link line
  set(tq_stage_runtime ascendcl)
  set(tq_stage_bench_defines "")
  if(leg STREQUAL "sim")
    set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
    list(APPEND tq_stage_runtime
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
    # The benchmarks read ACL timing events, and the CAModel's stub driver rejects that
    # resource outright. Without this they would be sim-leg binaries that cannot complete a
    # single sample; with it they run the ladder functionally, print NA for latency and say at
    # startup that their figures are not silicon.
    set(tq_stage_bench_defines ASCEND_MICRO_BENCH_SIM)
  endif()

  # add_tq_stage_runner(<target> <source> <kernel target> [<extra definitions>...])
  function(add_tq_stage_runner target source kernel)
    add_executable(${target} "${TQ_STAGE_DIR}/${source}")
    target_include_directories(${target} PRIVATE
        ${TQ_STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
    target_link_libraries(${target} PRIVATE ${kernel} ${tq_stage_runtime})
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    if(ARGN)
      target_compile_definitions(${target} PRIVATE ${ARGN})
    endif()
    if(leg STREQUAL "sim")
      target_link_options(${target} PRIVATE
          "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
      set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
          INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")
    else()
      set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
          INSTALL_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64")
    endif()
  endfunction()

  # ------------------------------------------------------------------- verification
  add_tq_stage_runner(test_stage_decode_${leg}
      "tests/validation/test_integration_decode.cpp"
      turboquant_stage_decode_${leg})
  add_tq_stage_runner(test_stage_batch_${leg}
      "tests/validation/test_integration_batch.cpp"
      turboquant_stage_batch_${leg})
  add_tq_stage_runner(test_stage_scatter_${leg}
      "tests/validation/test_integration_scatter.cpp"
      turboquant_stage_scatter_${leg})

  # ---------------------------------------------------------------------- benchmarks
  # One per staged kernel. The batch benchmark verifies every rung bit-exactly against the
  # shared oracle before timing it, so it links the same kernel library its validation runner
  # does rather than a timing-only copy.
  add_tq_stage_runner(bench_stage_pipeline_${leg}
      "tests/benchmark/bench_integration_pipeline.cpp"
      turboquant_stage_decode_${leg} ${tq_stage_bench_defines})
  add_tq_stage_runner(bench_stage_batch_${leg}
      "tests/benchmark/bench_integration_batch.cpp"
      turboquant_stage_batch_${leg} ${tq_stage_bench_defines})
  add_tq_stage_runner(bench_stage_scatter_${leg}
      "tests/benchmark/bench_integration_scatter.cpp"
      turboquant_stage_scatter_${leg})

  # An aggregate per leg, so the staged subtree can be built without naming nine targets.
  add_custom_target(turboquant_stage_${leg} DEPENDS
      test_stage_decode_${leg} test_stage_batch_${leg} test_stage_scatter_${leg}
      bench_stage_pipeline_${leg} bench_stage_batch_${leg} bench_stage_scatter_${leg})
endfunction()
