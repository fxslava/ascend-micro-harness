# CANN selects the physical-device host stubs and runtime for these targets.
function(add_ascend_micro_npu_targets)
  set(RUN_MODE npu)
  # CANN 9.2 adds its own headers with -I and emits hundreds of warnings under
  # -Wall. Reclassify those same directories as system headers for BiSheng so
  # -Werror remains strict for our kernel and the baseline source.
  set(CANN_SYSTEM_OPTIONS "")
  foreach(path IN ITEMS
      asc/impl/adv_api asc/impl/basic_api asc/impl/c_api
      asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils
      asc asc/include asc/include/adv_api asc/include/basic_api
      asc/include/aicpu_api asc/include/c_api
      asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils
      tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
    list(APPEND CANN_SYSTEM_OPTIONS
        "-isystem${ASCEND_HOME_PATH}/x86_64-linux/${path}")
  endforeach()
  set(BASELINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/baseline")
  set(BASELINE_SOURCE "${BASELINE_DIR}/turboquant_paged_attention.cpp")
  ascendc_library(turboquant_a8_npu SHARED
      kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp)
  ascendc_compile_definitions(turboquant_a8_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_a8_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  ascendc_library(turboquant_bf16_baseline_npu SHARED "${BASELINE_SOURCE}")
  ascendc_compile_options(turboquant_bf16_baseline_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS})

  add_executable(test_turboquant_a8_validation
      tests/npu/validation/test_turboquant_a8_validation.cpp)
  target_include_directories(test_turboquant_a8_validation PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_turboquant_a8_validation PRIVATE turboquant_a8_npu ascendcl)
  target_compile_options(test_turboquant_a8_validation PRIVATE -Wall -Wextra -Werror)

  add_executable(bench_turboquant_a8_vs_bf16
      tests/npu/bench/bench_turboquant_a8_vs_bf16.cpp)
  target_include_directories(bench_turboquant_a8_vs_bf16 PRIVATE
      "${ASCEND_HOME_PATH}/include" "${BASELINE_DIR}")
  target_link_libraries(bench_turboquant_a8_vs_bf16 PRIVATE
      turboquant_a8_npu turboquant_bf16_baseline_npu ascendcl)
  target_compile_options(bench_turboquant_a8_vs_bf16 PRIVATE -Wall -Wextra -Werror)

  foreach(target test_turboquant_a8_validation bench_turboquant_a8_vs_bf16)
    set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64")
  endforeach()
  add_custom_target(turboquant_npu_all DEPENDS
      test_turboquant_a8_validation bench_turboquant_a8_vs_bf16)
endfunction()
