cmake_minimum_required(VERSION 3.16)
project(ascend_modulo3_deadlock_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "CAModel SoC")
set(RUN_MODE "sim" CACHE STRING "CANN run mode")
set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.2.0-beta.2"
    CACHE PATH "CANN toolkit root")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")

if(NOT RUN_MODE STREQUAL "sim")
  message(FATAL_ERROR "This probe's CMake runner is for CAModel only")
endif()

set(ASCENDC_CMAKE_DIR
    "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
  message(FATAL_ERROR "CANN ascendc.cmake not found at ${ASCENDC_CMAKE_DIR}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

ascendc_library(turboquant_reshape_and_cache_skeleton SHARED
    kernels/skeleton/turboquant_reshape_and_cache.cpp)
ascendc_compile_definitions(turboquant_reshape_and_cache_skeleton PRIVATE ASCEND_MICRO_CANN_LAUNCH)
ascendc_compile_options(turboquant_reshape_and_cache_skeleton PRIVATE
    -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

add_executable(test_pipeline_deadlock_probe_camodel
    tests/test_pipeline_deadlock_probe_camodel.cpp)
target_include_directories(test_pipeline_deadlock_probe_camodel PRIVATE
    "${ASCEND_HOME_PATH}/include")
set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
target_link_libraries(test_pipeline_deadlock_probe_camodel PRIVATE
    turboquant_reshape_and_cache_skeleton ascendcl
    "${SIM_LIB_DIR}/libnpu_drv_camodel.so"
    "${SIM_LIB_DIR}/libmodel_top.so")
target_link_options(test_pipeline_deadlock_probe_camodel PRIVATE
    "-Wl,-rpath-link,${SIM_LIB_DIR}")
target_compile_options(test_pipeline_deadlock_probe_camodel PRIVATE
    -Wall -Wextra -Werror)

set_target_properties(test_pipeline_deadlock_probe_camodel PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

# add_micro_camodel_probe(<name> [<kernel_basename>|<kernel_path>] [<define>...])
# Builds target test_camodel_<name> from tests/camodel/test_camodel_<name>.cpp
# against kernels/camodel/<kernel_basename>.cpp, defaulting to <name>. A second
# argument containing a slash is taken as a repository-relative source path
# instead, which is how the production operator under kernels/ops is built.
# Any further arguments are NAME=VALUE compile definitions for the device
# kernel and the runner.
#
# `-forward-options-to-host-compiler` takes exactly one option here, and the
# definitions go through ascendc_compile_definitions instead. CANN's legacy
# ascendc_compile_options stores everything after that marker in the target's
# HOST_COMPILE_OPTIONS property and hands it to a nested ExternalProject as a
# single `-DASCENDC_HOST_COMPILE_OPTIONS=<property>` configure argument: a
# property with two elements loses one of them, and when the one it loses is
# ASCEND_MICRO_CANN_LAUNCH the host leg compiles an empty translation unit and
# `launch_camodel_fused_attention` silently disappears from the shared library.
# ascendc_compile_definitions reaches every device leg with the whole list,
# which is where the page count is needed.
function(add_micro_camodel_probe name)
  set(kernel_src "kernels/camodel/${name}.cpp")
  set(extra_defs "")
  if(ARGC GREATER 1)
    if(ARGV1 MATCHES "/")
      set(kernel_src "${ARGV1}")
    else()
      set(kernel_src "kernels/camodel/${ARGV1}.cpp")
    endif()
  endif()
  if(ARGC GREATER 2)
    list(SUBLIST ARGV 2 -1 extra_defs)
  endif()
  ascendc_library(camodel_${name}_kernel SHARED ${kernel_src})
  ascendc_compile_definitions(camodel_${name}_kernel PRIVATE
      ASCEND_MICRO_CANN_LAUNCH ${extra_defs})
  ascendc_compile_options(camodel_${name}_kernel PRIVATE
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
  add_executable(test_camodel_${name} tests/camodel/test_camodel_${name}.cpp)
  if(extra_defs)
    target_compile_definitions(test_camodel_${name} PRIVATE ${extra_defs})
  endif()
  target_include_directories(test_camodel_${name} PRIVATE "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_camodel_${name} PRIVATE camodel_${name}_kernel ascendcl
      "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(test_camodel_${name} PRIVATE "-Wl,-rpath-link,${SIM_LIB_DIR}")
  target_compile_options(test_camodel_${name} PRIVATE -Wall -Wextra -Werror)
  set_target_properties(test_camodel_${name} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}/lib;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")
endfunction()

add_micro_camodel_probe(dequant4)
add_micro_camodel_probe(cube_matmul)
add_micro_camodel_probe(wta_fixpipe)
add_executable(test_wta_convergence_probe tests/test_wta_convergence_probe.cpp)
target_compile_options(test_wta_convergence_probe PRIVATE -Wall -Wextra -Werror)
add_micro_camodel_probe(normalize_store)
add_micro_camodel_probe(online_softmax)

# AIV-only inverse-CDF int4x2 quantization. The tile count is a build
# parameter: 4 is the depth-2 ring boundary (V2), 8 wraps every slot twice
# (V3).
add_executable(test_icdf_quantize_probe tests/test_icdf_quantize_probe.cpp)
target_compile_options(test_icdf_quantize_probe PRIVATE -Wall -Wextra -Werror)
add_micro_camodel_probe(icdf_quantize)
add_micro_camodel_probe(icdf_quantize_8tiles
    kernels/camodel/icdf_quantize.cpp
    ICDF_QUANT_TILES=8)

# The fused operator's page count is a build parameter, so one kernel source
# serves both probes: the three-page depth boundary that established the
# numerics, and the eight-page run that wraps the depth-3 ring twice.
add_micro_camodel_probe(fused_attention_3tiles
    kernels/ops/turboquant/turboquant_paged_attention_opt.cpp
    FUSED_ATTENTION_TILES=3)

add_micro_camodel_probe(fused_attention_8tiles
    kernels/ops/turboquant/turboquant_paged_attention_opt.cpp
    FUSED_ATTENTION_TILES=8)

# The prefill reshape-and-cache operator. Its grid is fixed by the strip shape
# -- eight chunks of one 16-token page -- so there is no build parameter: the
# eight chunks already wrap the depth-4 L1 ring and the depth-2 per-core UB
# rings twice, which is what V3 asks a second probe for elsewhere.
add_micro_camodel_probe(turboquant_reshape_and_cache_a8
    kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp)
