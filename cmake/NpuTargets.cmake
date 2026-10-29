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

  # The BF16 input instance of our own operator -- a dedicated kernel, not a
  # templated A8, so the verified INT8 path is untouched. ASCEND_MICRO_SPARSE
  # _SAMPLING is deliberately absent here as it is for the A8 library: sparse
  # sampling is a CAModel verification device and must not exist in a
  # physical-device build.
  ascendc_library(turboquant_bf16_npu SHARED
      kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp)
  ascendc_compile_definitions(turboquant_bf16_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_bf16_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  add_executable(test_turboquant_a8_validation
      tests/npu/validation/test_turboquant_a8_validation.cpp)
  target_include_directories(test_turboquant_a8_validation PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_turboquant_a8_validation PRIVATE turboquant_a8_npu ascendcl)
  target_compile_options(test_turboquant_a8_validation PRIVATE -Wall -Wextra -Werror)

  # The BF16 instance's validation harness. It shares the fixture, the golden
  # oracle and the per-page verifier with the A8 one through
  # tests/npu/turboquant_npu_common.hpp -- only the input encoding differs --
  # and drives its strip ladder as several launches, so its default run reaches
  # the second execution wave at 33 and 64 strips.
  add_executable(test_turboquant_bf16_validation
      tests/npu/validation/test_turboquant_bf16_validation.cpp)
  target_include_directories(test_turboquant_bf16_validation PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_turboquant_bf16_validation PRIVATE
      turboquant_bf16_npu ascendcl)
  target_compile_options(test_turboquant_bf16_validation PRIVATE -Wall -Wextra -Werror)

  # The benchmark now measures three kernels, so it links our BF16 Cube library
  # as well: A8 Cube, BF16 Cube and the vendored BF16 Vector baseline. Only the
  # two BF16 figures share an input precision and a quantiser, so only that pair
  # isolates Cube against Vector; the runner says so in its own output.
  add_executable(bench_turboquant_reshape_and_cache
      tests/npu/bench/bench_turboquant_reshape_and_cache.cpp)
  target_include_directories(bench_turboquant_reshape_and_cache PRIVATE
      "${ASCEND_HOME_PATH}/include" "${BASELINE_DIR}")
  target_link_libraries(bench_turboquant_reshape_and_cache PRIVATE
      turboquant_a8_npu turboquant_bf16_npu turboquant_bf16_baseline_npu ascendcl)
  target_compile_options(bench_turboquant_reshape_and_cache PRIVATE -Wall -Wextra -Werror)

  # ---------------------------------------------------------- Split-K decode
  # Physical-device decode libraries. The AIC source retains its legacy
  # validation ABI and also exports the dynamic M=16 temporal split entry.
  # CANN selects RUN_MODE per target.
  ascendc_library(turboquant_paged_decode_npu SHARED
      kernels/ops/turboquant/turboquant_paged_decode_attention.cpp)
  ascendc_compile_definitions(turboquant_paged_decode_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_paged_decode_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  ascendc_library(turboquant_single_token_decode_npu SHARED
      kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp)
  ascendc_compile_definitions(turboquant_single_token_decode_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_single_token_decode_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The Split-K reduction. A separate translation unit from the CAModel probe
  # in kernels/camodel/l2_atomic_synthetic.cpp on purpose: this one carries the
  # watchdog, K up to 16 and the absolute timestamps that hardware needs, and
  # the probe's published instruction budget was measured without them. See the
  # header of the source and §5 of reports/l2_atomic_contention_split_k.md.
  ascendc_library(turboquant_splitk_reduce_npu SHARED
      kernels/ops/turboquant/turboquant_splitk_reduce.cpp)
  ascendc_compile_definitions(turboquant_splitk_reduce_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_splitk_reduce_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  ascendc_library(turboquant_single_token_decode_splitk_npu SHARED
      kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp)
  ascendc_compile_definitions(turboquant_single_token_decode_splitk_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_single_token_decode_splitk_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The benchmark. HARDWARE ONLY: it needs a 950PR, and the development host
  # for this harness has none, so what CI can establish for it is a clean build
  # -- the same standing as bench_turboquant_reshape_and_cache, which
  # reports/reshape_and_cache_a8.md records as compile-only for the same
  # reason. `add_ascend_micro_sim_targets` registers a simulator twin that
  # exercises the CLI, the statistics and the tail arithmetic where no device
  # exists.
  add_executable(bench_turboquant_decode_splitk
      tests/npu/bench/bench_turboquant_decode_splitk.cpp)
  target_include_directories(bench_turboquant_decode_splitk PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(bench_turboquant_decode_splitk PRIVATE
      turboquant_single_token_decode_splitk_npu turboquant_paged_decode_npu ascendcl)
  target_compile_options(bench_turboquant_decode_splitk PRIVATE
      -Wall -Wextra -Werror)

  # ------------------------------------------------- mode-4 research probe
  # tests/research/kernel_cluster_sync_probe.cpp: one physical AI Core block,
  # 1 Cube + 2 Vector cores, no data, and a cyclic bilateral handshake over
  # RECYCLED mode-4 ids 0, 1, 16 and 17 with an x_S drain in front of every
  # announcement. It is the experiment that settled whether a mode-4 flag
  # consumes -- guidelines section 6.2c -- and the reason the production
  # kernels now default to the recycled two-slot ring, so it is a target rather
  # than a transcript: a toolkit regression that reintroduces the stale release
  # has to be reproducible here.
  #
  # Run it with more than the default two iterations and read the trace, not
  # the exit status. The probe moves no data, so a sticky latch would make
  # every wait retire early and the run would still finish and still print
  # nothing wrong; what distinguishes the two semantics is the tick order of
  # SET_INTRA_BLOCK against WAIT_INTRA_BLOCK on one id, which only the trace
  # shows. `add_ascend_micro_sim_targets` registers the CAModel twin that
  # produces that trace.
  ascendc_library(cluster_sync_probe_npu SHARED
      tests/research/kernel_cluster_sync_probe.cpp)
  ascendc_compile_definitions(cluster_sync_probe_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(cluster_sync_probe_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  add_executable(test_cluster_sync_probe
      tests/research/test_cluster_sync_probe.cpp)
  target_include_directories(test_cluster_sync_probe PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_cluster_sync_probe PRIVATE
      cluster_sync_probe_npu ascendcl)
  target_compile_options(test_cluster_sync_probe PRIVATE -Wall -Wextra -Werror)

  foreach(target test_turboquant_a8_validation test_turboquant_bf16_validation
      bench_turboquant_reshape_and_cache bench_turboquant_decode_splitk
      test_cluster_sync_probe)
    set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64")
  endforeach()
  add_custom_target(turboquant_npu_all DEPENDS
      test_turboquant_a8_validation test_turboquant_bf16_validation
      bench_turboquant_reshape_and_cache bench_turboquant_decode_splitk)
endfunction()
