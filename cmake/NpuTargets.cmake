# CANN selects the physical-device host stubs and runtime for these targets.
function(add_ascend_micro_npu_targets)
  set(RUN_MODE npu)
  # The BiSheng system-header reclassification, defined once in the top-level
  # CMakeLists.txt so this leg and the simulator leg cannot drift apart.
  set(CANN_SYSTEM_OPTIONS ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS})
  # TQ_FUSED_DECODE_REDUCE_DEFINES carries -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 when
  # TQ_GEMM_REDUCE is ON, which it is by default. The option lives in the top-level
  # CMakeLists.txt so this leg gets the identical define the simulator leg does: the two
  # legs compile the same kernel source, and a reduction phase selected on one leg only is
  # a different operator on hardware than the one the CAModel evidence is about.
  # THE ONE FUSED DECODE SOURCE. There used to be two copies of this kernel -- the original
  # under kernels/ops/turboquant/ and a staging copy under vllm_integration_layer/ -- and the
  # staged subtree's own targets compiled the second one. The optimizations now live once, in the
  # subtree that is exported to vllm-ascend, and this leg, the simulator leg and the staged legs
  # all compile that file. The kernel's own includes are relative to its directory, so the -I
  # flags below are only for the service headers that reach for the subtree's include/ and
  # routines/ roots by a shorter path.
  set(TQ_FUSED_DECODE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/vllm_integration_layer")
  ascendc_library(turboquant_fused_decode_opt_npu SHARED
      kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp)
  ascendc_compile_definitions(turboquant_fused_decode_opt_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_fused_decode_opt_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      ${TQ_FUSED_DECODE_REDUCE_DEFINES}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
      "-I${TQ_FUSED_DECODE_DIR}/op_kernel"
      "-I${TQ_FUSED_DECODE_DIR}/include"
      "-I${TQ_FUSED_DECODE_DIR}/routines")
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

  option(ASCEND_MICRO_NATIVE_INGEST "Enable the optional CANN native ingest comparator" ON)
  if(ASCEND_MICRO_NATIVE_INGEST)
    find_path(TQ_INGEST_ACLNN_INCLUDE aclnn_scatter_pa_kv_cache.h
        HINTS "${ASCEND_HOME_PATH}/include/aclnnop"
              "${ASCEND_HOME_PATH}/x86_64-linux/include/aclnnop" NO_DEFAULT_PATH)
    find_library(TQ_INGEST_OPAPI NAMES opapi_transformer opapi
        HINTS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    find_library(TQ_INGEST_NNOPBASE NAMES nnopbase
        HINTS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    if(TQ_INGEST_ACLNN_INCLUDE AND TQ_INGEST_OPAPI AND TQ_INGEST_NNOPBASE)
      target_compile_definitions(bench_turboquant_reshape_and_cache PRIVATE
          ASCEND_MICRO_HAVE_NATIVE_INGEST
          TQ_INGEST_OPAPI_PATH="${TQ_INGEST_OPAPI}"
          TQ_INGEST_NNOPBASE_PATH="${TQ_INGEST_NNOPBASE}")
      target_include_directories(bench_turboquant_reshape_and_cache PRIVATE
          "${TQ_INGEST_ACLNN_INCLUDE}")
      target_link_libraries(bench_turboquant_reshape_and_cache PRIVATE
          ${CMAKE_DL_LIBS})
    else()
      message(STATUS "Native ingest comparator unavailable; audit will report N/A")
    endif()
  endif()

  add_executable(test_ingest_audit_report tests/cpu/test_ingest_audit_report.cpp)
  target_compile_options(test_ingest_audit_report PRIVATE -Wall -Wextra -Werror)

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

  # turboquant_single_token_decode_npu and turboquant_single_token_decode_splitk_npu
  # are gone with their sources. The fused decode operator
  # (turboquant_fused_decode_opt_npu, above) is the only decode this harness ships, and
  # the gates those two libraries carried are listed in
  # scripts/run_fused_decode_canary_ladder.sh and
  # tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp.

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

  # The benchmark. HARDWARE ONLY: it needs a 950PR, and the development host
  # for this harness has none, so what CI can establish for it is a clean build
  # -- the same standing as bench_turboquant_reshape_and_cache, which
  # reports/reshape_and_cache_a8.md records as compile-only for the same
  # reason. `add_ascend_micro_sim_targets` registers a simulator twin that
  # exercises the CLI, the statistics and the tail arithmetic where no device
  # exists.
  # KERNEL ISOLATION. TQ_BENCH_LEGACY_DECODERS is declared in the top-level CMakeLists.txt,
  # next to TQ_GEMM_REDUCE, because the simulator twin compiles the identical driver source
  # and an option visible to one leg only would benchmark different kernels on the two.
  #
  # OFF, the default, builds a benchmark that contains exactly one kernel: the fused decode
  # operator. That follows the same rule the split-K validation driver below follows -- link
  # the one kernel library the rigs actually reference, because CANN keeps BUILD_ALWAYS on
  # every ascendc_library and a spare link costs a 54-second rebuild of a library nothing
  # calls. ON restores the AIC paged-decode comparison columns, and the source's
  # TQ_BENCH_FUSED_OPT_ONLY blocks are that switch. The modeled AIV-against-AIC crossover is
  # not among them: it was relative to the deleted decoder's K=1 time.
  #
  # The AIV columns are not a switch any more: the AIV split-K decoder is deleted, and
  # --impl aiv, both and all went out of the driver with it. --impl aic is the only legacy
  # implementation left.
  add_executable(bench_turboquant_decode_splitk
      tests/npu/bench/bench_turboquant_decode_splitk.cpp)
  target_include_directories(bench_turboquant_decode_splitk PRIVATE
      "${ASCEND_HOME_PATH}/include")
  if(TQ_BENCH_LEGACY_DECODERS)
    target_compile_definitions(bench_turboquant_decode_splitk PRIVATE
        TQ_BENCH_LEGACY_DECODERS=1)
    target_link_libraries(bench_turboquant_decode_splitk PRIVATE
        turboquant_paged_decode_npu turboquant_fused_decode_opt_npu ascendcl)
  else()
    target_link_libraries(bench_turboquant_decode_splitk PRIVATE
        turboquant_fused_decode_opt_npu ascendcl)
  endif()
  target_compile_options(bench_turboquant_decode_splitk PRIVATE
      -Wall -Wextra -Werror)

  # ------------------------------ split-K decode validation (directive 3)
  # The correctness ladder for the AIC split-K decode kernel. HARDWARE ONLY, on the
  # same footing as the benchmark above and for the same reason: each rung is a real
  # launch whose every emitted value is compared against the FP64 oracles in
  # tests/common, so a build on a host with no 950PR establishes that the harness
  # compiles and nothing about the kernel. `add_ascend_micro_sim_targets` registers the
  # simulator twin, which is where this can actually run without a device.
  #
  # test_npu_decode_splitk_aiv used to sit beside it, over the deleted AIV split-K
  # kernel. The fused operator's Split-K path is validated by test_npu_fused_decode_opt
  # on hardware and by the splitk-workers rung of
  # scripts/run_fused_decode_canary_ladder.sh on CAModel, at both element types and
  # including an empty worker.
  #
  # It links the ONE kernel library its driver calls into.
  # tests/npu/turboquant_splitk_rigs.hpp declares the launchers its rigs use, but a
  # declaration is not a reference: each rig's `Launch` is an implicitly inline member
  # emitted only where that rig is used, and this driver uses exactly one. `nm -u` on its
  # object shows test_npu_decode_splitk_aic needing only
  # launch_turboquant_paged_decode_splitk_aic -- with and without
  # ASCEND_MICRO_DECODE_SPLITK_SIM, so the simulator twin is the same.
  #
  # The cost of a spare link was a whole extra ascendc_library per driver, on every build
  # rather than on clean ones, because CannRebuild.cmake keeps CANN's BUILD_ALWAYS: a
  # no-op rebuild of the paged-decode library alone measures 54 s and 17 compile
  # invocations. Future-proofing that expensive is not worth it when the failure mode is a
  # link error that names the missing symbol and is repaired by putting the library back.
  add_executable(test_npu_decode_splitk_aic
      tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp)
  target_include_directories(test_npu_decode_splitk_aic PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_npu_decode_splitk_aic PRIVATE
      turboquant_paged_decode_npu ascendcl)
  target_compile_options(test_npu_decode_splitk_aic PRIVATE
      -Wall -Wextra -Werror)

  # ------------------- fused decode operator validation (hardware, GQA + canaries)
  # tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp: the hardware counterpart of
  # the operator's CAModel ladder. 16 query heads over one KV head at D = 256, 128-token pages,
  # S in {1, 2, 4, 8} over four contexts, every emitted channel against the fixture's own FP64
  # attention oracle at a budget of ZERO fp16 ULP, and 0x5A5A5A5A canaries on the guard bands of
  # every allocation, on every physical KV page the block table does not map, on the inter-head
  # scale padding of every page it does, on the output tail and on the workspace outside the
  # live partial extent.
  #
  # HARDWARE ONLY, like the benchmark and the two split-K ladders above and for the same
  # reason: every rung is a real launch compared against an FP64 oracle, so a build on a host
  # with no 950PR establishes that this harness compiles and nothing about the kernel. It links
  # the one library it calls into, turboquant_fused_decode_opt_npu, which carries
  # -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 whenever TQ_GEMM_REDUCE is ON -- so with the default
  # configuration this ladder validates the Cube GEMM reduction, and with -DTQ_GEMM_REDUCE=OFF
  # the same ladder validates the vector reducer. Neither is selected here.
  add_executable(test_npu_fused_decode_opt
      tests/npu/validation/test_turboquant_fused_decode_opt_npu.cpp)
  target_include_directories(test_npu_fused_decode_opt PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_npu_fused_decode_opt PRIVATE
      turboquant_fused_decode_opt_npu ascendcl)
  target_compile_options(test_npu_fused_decode_opt PRIVATE -Wall -Wextra -Werror)

  # ------------------------- the WRITE -> READ round trip (hardware, two operators)
  # THE NZ-TILED INGRESS INSTANCE of the BF16 prefill. A THIRD library out of the same
  # source, beside turboquant_bf16_npu above, and the only difference is
  # TURBOQUANT_INGRESS_NZ: it makes the writer target common/turboquant_layout.h's
  # NzTiled* contract -- what the kv4fp8 fused decode reads -- instead of RFC 3.1's strip
  # one. The two are not reachable from one another by any permutation of addresses
  # (tests/cpu/test_turboquant_kv_layout_bridge.cpp check 3), so this is a separate
  # instance and not a flag on the shipping one: turboquant_bf16_npu is untouched and is
  # still what every other harness here links, and no production launch gets this define.
  #
  # cmake/SimTargets.cmake registers camodel_turboquant_reshape_nz_kernel, the CAModel twin
  # of this library, for the same reason and with the same define.
  ascendc_library(turboquant_bf16_nz_npu SHARED
      kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp)
  ascendc_compile_definitions(turboquant_bf16_nz_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_INGRESS_NZ)
  ascendc_compile_options(turboquant_bf16_nz_npu PRIVATE
      -std=c++17 -Wall -Werror ${CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # tests/npu/validation/test_turboquant_e2e_npu.cpp: ONE process, TWO real operators, ONE
  # cache. The NZ prefill writes quantised K and V into physical 64-token pages through a
  # fragmented slot mapping, and the fused decode then reads those very pages through a
  # block table and computes attention over them -- nothing between the two launches but a
  # stream synchronise. It is the first target in this harness where a genuine
  # ingress -> decode round trip exists at all; before the NZ instance landed, the two
  # operators named different bytes and reports/turboquant_e2e_roundtrip.md records that
  # finding.
  #
  # It therefore links BOTH kernel libraries, which is the whole point and is the one place
  # in this file where the "link the one library the driver calls" rule does not apply.
  #
  # TURBOQUANT_INGRESS_NZ is set on the EXECUTABLE as well as on the kernel library. The
  # driver does not branch on it; it carries an #error on its absence, so a build that
  # linked the strip instance by mistake fails at compile time rather than launching, passing
  # no gate and being very hard to read.
  #
  # HARDWARE ONLY, like the four ladders above and for the same reason: every gate is a real
  # launch pair compared against a golden image and an FP64 oracle, so a build on a host with
  # no 950PR establishes that this harness compiles and nothing about either kernel. Unlike
  # those four it has no CAModel twin yet -- the simulator leg has the two halves in isolation
  # (scripts/run_turboquant_e2e_roundtrip.sh), which is a canary and not a baseline.
  #
  # acl_op_compiler is linked when the toolkit has it. This driver calls no aclnn or aclop
  # entry point, so the link is for parity with the runtime set the hardware runners are
  # expected to carry and is skipped rather than failing configure where the library is absent.
  add_executable(test_turboquant_e2e_npu
      tests/npu/validation/test_turboquant_e2e_npu.cpp)
  target_include_directories(test_turboquant_e2e_npu PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_compile_definitions(test_turboquant_e2e_npu PRIVATE TURBOQUANT_INGRESS_NZ)
  target_link_libraries(test_turboquant_e2e_npu PRIVATE
      turboquant_bf16_nz_npu turboquant_fused_decode_opt_npu ascendcl)
  find_library(TQ_E2E_ACL_OP_COMPILER NAMES acl_op_compiler
      HINTS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
  if(TQ_E2E_ACL_OP_COMPILER)
    target_link_libraries(test_turboquant_e2e_npu PRIVATE ${TQ_E2E_ACL_OP_COMPILER})
  else()
    message(STATUS "acl_op_compiler not found; test_turboquant_e2e_npu links ascendcl only")
  endif()
  target_compile_options(test_turboquant_e2e_npu PRIVATE -Wall -Wextra -Werror)

  # ------------------------- single-token decode at an unaligned context S
  # test_npu_decode_single_token_unaligned walked S = 1, 7, 15, 17, 31, 33, 127, 129,
  # 255, 257, 1023 -- not one of them a multiple of the 16-token strip and most not of
  # the page -- over the two deleted AIV decode kernels, with a quiet-NaN canary in
  # every cache slot past seq_len. Both kernels and the driver are gone.
  #
  # What replaced it: the fused decode runner takes the live row count of the final tile
  # as --tail-rows, so rungs 2, 3 and 8 of scripts/run_fused_decode_canary_ladder.sh end
  # a context inside a tile, rung 8 does it over the whole eight-page block table, and
  # every one of them holds the cache, the workspace outside the live extent and both
  # exterior guard bands to 0x7FC0BEEF. The hardware ladder is test_npu_fused_decode_opt.

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
      -std=c++17 -Wall -Werror -Wno-missing-braces ${CANN_SYSTEM_OPTIONS}
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
      test_npu_decode_splitk_aic
      test_npu_fused_decode_opt
      test_turboquant_e2e_npu
      test_cluster_sync_probe)
    set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64")
  endforeach()
  add_custom_target(turboquant_npu_all DEPENDS
      test_turboquant_a8_validation test_turboquant_bf16_validation
      bench_turboquant_reshape_and_cache bench_turboquant_decode_splitk
      test_npu_decode_splitk_aic
      test_npu_fused_decode_opt
      test_turboquant_e2e_npu)
endfunction()
