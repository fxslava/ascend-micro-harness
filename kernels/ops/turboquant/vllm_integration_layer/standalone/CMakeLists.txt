cmake_minimum_required(VERSION 3.16)
project(turboquant_vllm_integration_layer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Ascend SoC")
set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.2.0-beta.2" CACHE PATH "CANN toolkit root")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(TURBOQUANT_RUN_MODE "npu" CACHE STRING "CANN target mode: npu or sim")
set_property(CACHE TURBOQUANT_RUN_MODE PROPERTY STRINGS npu sim)
if(NOT TURBOQUANT_RUN_MODE STREQUAL "npu" AND NOT TURBOQUANT_RUN_MODE STREQUAL "sim")
  message(FATAL_ERROR "TURBOQUANT_RUN_MODE must be npu or sim")
endif()
set(RUN_MODE "${TURBOQUANT_RUN_MODE}")

if(NOT EXISTS "${ASCEND_HOME_PATH}/set_env.sh")
  message(FATAL_ERROR "CANN environment not found under ${ASCEND_HOME_PATH}")
endif()
if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
  set(ASCENDC_CMAKE_DIR "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
  set(ASCENDC_CMAKE_DIR "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
else()
  message(FATAL_ERROR "CANN ascendc.cmake is missing under ${ASCEND_HOME_PATH}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

get_filename_component(STAGE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(STAGE_INCLUDE_DIRS
  "${STAGE_DIR}/include"
  "${STAGE_DIR}/op_kernel"
  "${STAGE_DIR}/op_kernel/common"
  "${STAGE_DIR}/op_kernel/cube"
  "${STAGE_DIR}/op_kernel/vector"
  "${STAGE_DIR}/routines"
  "${STAGE_DIR}/tests/common"
  "${STAGE_DIR}/tests/npu"
  "${STAGE_DIR}/tests/validation")

set(CANN_SYSTEM_OPTIONS)
foreach(path IN ITEMS
    asc/impl/adv_api asc/impl/basic_api asc/impl/c_api
    asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils
    asc asc/include asc/include/adv_api asc/include/basic_api
    asc/include/aicpu_api asc/include/c_api
    asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils
    tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
  list(APPEND CANN_SYSTEM_OPTIONS "-isystem${ASCEND_HOME_PATH}/x86_64-linux/${path}")
endforeach()

# The fused decode operator's reduction phase, exactly as the harness's own top-level
# CMakeLists.txt declares it: ON drives the reduction with the Cube GEMM reducer
# (TurboQuantBatchedGemmReducer, op_kernel/cube/turboquant_gemm_reduce_service.h), OFF falls
# back to TurboQuantPartialReducer. The default matches the harness's, because the staged
# kernel source IS the harness's: a staged build that defaulted the other way would compile a
# different operator than the one the harness verifies under the same file name.
option(TQ_GEMM_REDUCE "Drive the staged decode kernel's reduction with the Cube GEMM" ON)
set(TQ_STAGE_REDUCE_DEFINES)
if(TQ_GEMM_REDUCE)
  set(TQ_STAGE_REDUCE_DEFINES -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1)
endif()

function(add_turboquant_kernel target source)
  ascendc_library(${target} SHARED "${STAGE_DIR}/${source}")
  ascendc_compile_definitions(${target} PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(${target} PRIVATE
    -std=c++17 -Wall -Werror
    ${CANN_SYSTEM_OPTIONS}
    ${ARGN}
    -forward-options-to-host-compiler
    -DASCEND_MICRO_CANN_LAUNCH
    "-I${STAGE_DIR}/op_kernel"
    "-I${STAGE_DIR}/include"
    "-I${STAGE_DIR}/routines")
endfunction()

add_turboquant_kernel(turboquant_stage_decode op_kernel/turboquant_fused_decode.cpp
  ${TQ_STAGE_REDUCE_DEFINES})
add_turboquant_kernel(turboquant_stage_reshape_batch op_kernel/turboquant_reshape_batch.cpp)
add_turboquant_kernel(turboquant_stage_scatter op_kernel/turboquant_reshape_scatter.cpp)

find_library(ASCENDCL_LIBRARY ascendcl HINTS "${ASCEND_HOME_PATH}/lib64")
if(NOT ASCENDCL_LIBRARY)
  message(FATAL_ERROR "libascendcl was not found under ${ASCEND_HOME_PATH}/lib64")
endif()
set(TURBOQUANT_RUNTIME_LIBRARIES "${ASCENDCL_LIBRARY}")
if(TURBOQUANT_RUN_MODE STREQUAL "sim")
  set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
  foreach(sim_lib IN ITEMS libnpu_drv_camodel.so libmodel_top.so)
    if(NOT EXISTS "${SIM_LIB_DIR}/${sim_lib}")
      message(FATAL_ERROR "CAModel library missing: ${SIM_LIB_DIR}/${sim_lib}")
    endif()
  endforeach()
  list(APPEND TURBOQUANT_RUNTIME_LIBRARIES
    "${SIM_LIB_DIR}/libnpu_drv_camodel.so"
    "${SIM_LIB_DIR}/libmodel_top.so")
endif()

add_executable(test_integration_decode "${STAGE_DIR}/tests/validation/test_integration_decode.cpp")
target_include_directories(test_integration_decode PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(test_integration_decode PRIVATE turboquant_stage_decode ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(test_integration_decode PRIVATE -Wall -Wextra -Werror)

add_executable(test_integration_scatter "${STAGE_DIR}/tests/validation/test_integration_scatter.cpp")
target_include_directories(test_integration_scatter PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(test_integration_scatter PRIVATE turboquant_stage_scatter ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(test_integration_scatter PRIVATE -Wall -Wextra -Werror)

# ASCEND_MICRO_BENCH_SIM on the simulator leg, and only there. The benchmarks take their
# figures from ACL timing events, and the CAModel's stub driver rejects the event resource
# outright, so without this define a sim-mode build produces benchmark binaries that cannot
# complete a single sample -- a target that compiles and always fails. With it they run the
# ladder functionally and print NA for the latencies, which is what the sim leg can honestly
# report. The define also makes each binary say at startup that its figures are not silicon,
# so a pasted log cannot be mistaken for hardware.
set(TURBOQUANT_BENCH_DEFINES)
if(TURBOQUANT_RUN_MODE STREQUAL "sim")
  set(TURBOQUANT_BENCH_DEFINES ASCEND_MICRO_BENCH_SIM)
endif()

add_executable(bench_integration_pipeline "${STAGE_DIR}/tests/benchmark/bench_integration_pipeline.cpp")
target_include_directories(bench_integration_pipeline PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(bench_integration_pipeline PRIVATE turboquant_stage_decode ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(bench_integration_pipeline PRIVATE -Wall -Wextra -Werror)
if(TURBOQUANT_BENCH_DEFINES)
  target_compile_definitions(bench_integration_pipeline PRIVATE ${TURBOQUANT_BENCH_DEFINES})
endif()

add_executable(bench_integration_scatter "${STAGE_DIR}/tests/benchmark/bench_integration_scatter.cpp")
target_include_directories(bench_integration_scatter PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(bench_integration_scatter PRIVATE turboquant_stage_scatter ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(bench_integration_scatter PRIVATE -Wall -Wextra -Werror)

# The prefill batch writer's benchmark: the third staged kernel, which had a validation
# ladder and no timed runner. Every rung is verified against the shared oracle before it is
# timed, so this links the same kernel library test_integration_batch does.
add_executable(bench_integration_batch "${STAGE_DIR}/tests/benchmark/bench_integration_batch.cpp")
target_include_directories(bench_integration_batch PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(bench_integration_batch PRIVATE turboquant_stage_reshape_batch ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(bench_integration_batch PRIVATE -Wall -Wextra -Werror)
if(TURBOQUANT_BENCH_DEFINES)
  target_compile_definitions(bench_integration_batch PRIVATE ${TURBOQUANT_BENCH_DEFINES})
endif()

add_library(turboquant_stage_tiling STATIC "${STAGE_DIR}/op_host/turboquant_tiling.cpp")
target_include_directories(turboquant_stage_tiling PUBLIC ${STAGE_INCLUDE_DIRS})
target_compile_options(turboquant_stage_tiling PRIVATE -Wall -Wextra -Werror)

add_executable(test_integration_batch
  "${STAGE_DIR}/tests/validation/test_integration_batch.cpp")
target_include_directories(test_integration_batch PRIVATE ${STAGE_INCLUDE_DIRS} "${ASCEND_HOME_PATH}/include")
target_link_libraries(test_integration_batch PRIVATE turboquant_stage_reshape_batch ${TURBOQUANT_RUNTIME_LIBRARIES})
target_compile_options(test_integration_batch PRIVATE -Wall -Wextra -Werror)

add_executable(test_integration_adapter_headers
  "${STAGE_DIR}/tests/validation/test_adapter_headers.cpp")
target_include_directories(test_integration_adapter_headers PRIVATE
  ${STAGE_INCLUDE_DIRS} "${STAGE_DIR}/op_adapter")
target_compile_options(test_integration_adapter_headers PRIVATE -Wall -Wextra -Werror)

# The Tier-1 host gate. No ACL, no kernel library, no CANN runtime: it links the staged
# subtree's C++17 headers only, so it is the one verification target that RUNS on a host with
# neither a 950PR nor the simulator libraries. Everything else here opens a device session in
# main(). It is not kernel evidence; it establishes that the oracles the device runners compare
# against are sound. See the file header for the gates.
add_executable(test_integration_host_gate
  "${STAGE_DIR}/tests/host/test_integration_host_gate.cpp")
target_include_directories(test_integration_host_gate PRIVATE ${STAGE_INCLUDE_DIRS})
target_compile_options(test_integration_host_gate PRIVATE -Wall -Wextra -Werror)

message(STATUS "Building Phase 1 kernels with CANN mode=${TURBOQUANT_RUN_MODE}; no Torch or torch_npu dependency is used")
