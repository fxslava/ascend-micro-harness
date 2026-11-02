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

include(cmake/CannRebuild.cmake)
add_ascend_micro_cann_rebuild_steps()
