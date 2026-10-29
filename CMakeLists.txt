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

include(cmake/SimTargets.cmake)
include(cmake/NpuTargets.cmake)

# The research probes under tests/research are registered inside these two
# functions rather than through add_subdirectory, because RUN_MODE is set per
# function: a subdirectory included here would pick up whichever mode happened
# to be global and would give one build of the probe instead of the hardware
# and simulator pair the evidence needs.
add_ascend_micro_sim_targets()
add_ascend_micro_npu_targets()
