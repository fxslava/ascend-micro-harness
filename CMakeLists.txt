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
