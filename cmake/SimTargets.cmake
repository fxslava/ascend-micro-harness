# CANN selects the simulator host stubs and runtime when each target is created.
function(add_ascend_micro_sim_targets)
  set(RUN_MODE sim)
  ascendc_library(turboquant_reshape_and_cache_skeleton SHARED
      kernels/skeleton/turboquant_reshape_and_cache.cpp)
  ascendc_compile_definitions(turboquant_reshape_and_cache_skeleton PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(turboquant_reshape_and_cache_skeleton PRIVATE
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  add_executable(test_pipeline_deadlock_probe_camodel
      tests/camodel/test_pipeline_deadlock_probe_camodel.cpp)
  target_include_directories(test_pipeline_deadlock_probe_camodel PRIVATE
      "${ASCEND_HOME_PATH}/include")
  set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
  # The model libraries depend on other simulator DSOs. DT_RPATH lets the
  # runner's simulator path resolve those indirect dependencies as well.
  target_link_libraries(test_pipeline_deadlock_probe_camodel PRIVATE
      turboquant_reshape_and_cache_skeleton ascendcl
      "${SIM_LIB_DIR}/libnpu_drv_camodel.so"
      "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(test_pipeline_deadlock_probe_camodel PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(test_pipeline_deadlock_probe_camodel PRIVATE
      -Wall -Wextra -Werror)

  set_target_properties(test_pipeline_deadlock_probe_camodel PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

  # add_micro_camodel_probe(<name> [<kernel_basename>|<kernel_path>])
  # Builds target test_camodel_<name> from tests/camodel/test_camodel_<name>.cpp
  # against kernels/camodel/<kernel_basename>.cpp, defaulting to <name>. A second
  # argument containing a slash is taken as a repository-relative source path
  # instead, which is how the production operator under kernels/ops is built.
  # `-forward-options-to-host-compiler` takes exactly one option here, and the
  # launch definition goes through ascendc_compile_definitions. CANN's legacy
  # ascendc_compile_options stores everything after that marker in the target's
  # HOST_COMPILE_OPTIONS property and hands it to a nested ExternalProject as a
  # single `-DASCENDC_HOST_COMPILE_OPTIONS=<property>` configure argument: a
  # property with two elements loses one of them, and when the one it loses is
  # ASCEND_MICRO_CANN_LAUNCH the host leg compiles an empty translation unit and
  # `launch_camodel_fused_attention` silently disappears from the shared library.
  # ascendc_compile_definitions reaches every device leg.
  # A third argument names extra compile definitions that must reach both the
  # kernel's device legs and the runner. They go through
  # ascendc_compile_definitions, which reaches every device leg, and through a
  # plain target_compile_definitions on the runner, which is an ordinary CMake
  # target with none of the forwarding limitation above. The host leg of the
  # kernel library does not get them, so anything they gate must live in the
  # device leg only -- which is the case for sparse sampling: the launcher and
  # the kernel signature are identical either way.
  # add_micro_camodel_runner(<name> <kernel_target> [<extra definitions>...])
  # The RUNNER half of add_micro_camodel_probe, on its own, so that a second
  # runner over an operator can link the kernel library that already exists
  # instead of provoking a duplicate build of it.
  #
  # That duplicate is not a stray object file. `ascendc_library` expands into a
  # nested ExternalProject per leg -- aic_device, aiv_device, host, plus
  # precompile, preprocess and a merge step -- so a second library over the
  # same source compiles the whole translation unit three more times through
  # BiSheng and runs CANN's preprocessing again. On this operator that is the
  # single most expensive thing in the build, and the second copy was linked by
  # exactly one runner that calls the same `launch_camodel_*` symbol as the
  # first. It also silently escaped the strict warning set: the per-target
  # `-std=c++17 -Wall -Werror ${splitk_system_options}` below names
  # camodel_paged_decode_attention_kernel, so the duplicate was the one
  # translation unit of this operator compiled without it.
  function(add_micro_camodel_runner name kernel_target)
    if(name STREQUAL "fused_decode_opt" OR name STREQUAL "gemm_reduce")
      add_executable(test_camodel_${name} tests/sim/kernel/test_camodel_${name}.cpp)
    else()
      add_executable(test_camodel_${name} tests/camodel/test_camodel_${name}.cpp)
    endif()
    if(ARGN)
      target_compile_definitions(test_camodel_${name} PRIVATE ${ARGN})
    endif()
    target_include_directories(test_camodel_${name} PRIVATE "${ASCEND_HOME_PATH}/include")
    target_link_libraries(test_camodel_${name} PRIVATE ${kernel_target} ascendcl
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
    target_link_options(test_camodel_${name} PRIVATE
        "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
    target_compile_options(test_camodel_${name} PRIVATE -Wall -Wextra -Werror)
    set_target_properties(test_camodel_${name} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")
  endfunction()

  function(add_micro_camodel_probe name)
    set(kernel_src "kernels/camodel/${name}.cpp")
    if(ARGC GREATER 1)
      if(ARGV1 MATCHES "/")
        set(kernel_src "${ARGV1}")
      else()
        set(kernel_src "kernels/camodel/${ARGV1}.cpp")
      endif()
    endif()
    set(extra_defs "")
    if(ARGC GREATER 2)
      set(extra_defs ${ARGV2})
    endif()
    ascendc_library(camodel_${name}_kernel SHARED ${kernel_src})
    ascendc_compile_definitions(camodel_${name}_kernel PRIVATE
        ASCEND_MICRO_CANN_LAUNCH ${extra_defs})
    ascendc_compile_options(camodel_${name}_kernel PRIVATE
        -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
    add_micro_camodel_runner(${name} camodel_${name}_kernel ${extra_defs})
  endfunction()

  add_micro_camodel_probe(dequant4)
  add_micro_camodel_probe(cube_matmul)
  add_micro_camodel_probe(wta_fixpipe)
  add_executable(test_wta_convergence_probe tests/cpu/test_wta_convergence_probe.cpp)
  target_compile_options(test_wta_convergence_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(normalize_store)
  add_micro_camodel_probe(online_softmax)
  add_micro_camodel_probe(fused_decode_opt
      tests/sim/kernel/fused_decode_opt_kernel.cpp)
  add_micro_camodel_runner(reduction_probe camodel_fused_decode_opt_kernel)
  add_executable(test_fused_decode_opt_contract tests/cpu/test_fused_decode_opt_contract.cpp)
  target_compile_options(test_fused_decode_opt_contract PRIVATE -Wall -Wextra -Werror)
  add_executable(test_fused_decode_opt_fixture tests/cpu/test_fused_decode_opt_fixture.cpp)
  target_compile_options(test_fused_decode_opt_fixture PRIVATE -Wall -Wextra -Werror)
  # TQ_GEMM_REDUCE drives the operator's reduction phase with the Cube GEMM reduction
  # (TurboQuantBatchedGemmReducer) instead of the shipping vector reducer. It is ON by
  # default and the option itself is declared in the top-level CMakeLists.txt, so this leg
  # and the physical leg cannot disagree about which reduction the operator ships.
  #
  # It used to be OFF, and turning it ON used to FAIL the build on a static_assert naming
  # the operator's 19648-byte UB deficit -- the batched form's two 8-row fp32 panels did not
  # fit the 4256 bytes left under the 192 KB window. That deficit is now paid by a TBufPool
  # aliased onto the attention phase's dead transient pool, so ON builds and runs; the
  # arithmetic is in include/turboquant_gemm_reduce.hpp and the overlay's own evidence in
  # reports/gemm_batched_reduction_camodel_20261104.md. scripts/gemm_reduce_switch_gate.sh
  # is the retained sanity check: ON must build, and a pool cut too short must still be
  # rejected by that static_assert rather than silently aliasing out of bounds.
  # tests/sim/kernel/fused_decode_opt_kernel.cpp includes the single production kernel source,
  # which now lives in the integration subtree; these are that subtree's include roots, the same
  # three cmake/IntegrationTargets.cmake and cmake/NpuTargets.cmake put on their own command
  # lines, so the simulator leg and the two physical legs compile the identical operator.
  ascendc_compile_options(camodel_fused_decode_opt_kernel PRIVATE
      -std=c++17 -Wall -Werror ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS}
      ${TQ_FUSED_DECODE_REDUCE_DEFINES}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
      "-I${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/vllm_integration_layer/op_kernel"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/vllm_integration_layer/include"
      "-I${CMAKE_CURRENT_SOURCE_DIR}/kernels/ops/turboquant/vllm_integration_layer/routines")

  # The Split-K final reduction on the Cube. Its KERNEL source stays under tests/, so the
  # probe's entry points cannot reach a production launcher, but its SERVICE header now lives
  # in kernels/ops/turboquant/fused_decode/cube/ because the dense batched reducer is wired
  # into the operator's reduction phase behind TQ_GEMM_REDUCE above. The probe emits five entry
  # points over one workspace -- the M = 1 GEMV stage, that stage with Mmad's GEMV mode left
  # on, the dense batched form, the batched form with the same knob, and the shipping
  # TurboQuantPartialReducer as the numerical reference -- so every comparison is against
  # production arithmetic compiled in the same build rather than a host reimplementation of it.
  add_micro_camodel_probe(gemm_reduce
      tests/sim/kernel/gemm_reduce_kernel.cpp)
  add_executable(test_gemm_reduce_geometry tests/cpu/test_gemm_reduce_geometry.cpp)
  target_compile_options(test_gemm_reduce_geometry PRIVATE -Wall -Wextra -Werror)
  # -Wall -Werror, the same strictness the operator's own device leg is built at, and NOT
  # -Wextra: this variant includes the production vector service, and -Wextra fails there on a
  # pre-existing unused parameter in TurboQuantModeName
  # (kernels/ops/turboquant/fused_decode/common/turboquant_mode.h:297), which is not this
  # prototype's to fix. The -Wall -Wextra -Werror compile §6.1 asks for is the host gate over
  # the shared contract header, test_gemm_reduce_geometry above.
  ascendc_compile_options(camodel_gemm_reduce_kernel PRIVATE
      -std=c++17 -Wall -Werror ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The randomised-Hadamard butterfly leaf (roadmap No.6) and its three
  # competing permutation strategies. One kernel source, three entry points,
  # one runner that takes --variant: the Unified Buffer censuses that decide
  # which leaf the AIV-only scatter ingress uses have to come from the same
  # build, the same fixture and the same surrounding DMA, or they are not a
  # comparison. Its Tier-1 probe checks the stage tables and both register
  # algorithms against the dense 256 x 256 matrix with no device at all.
  add_executable(test_rotate_butterfly_probe tests/cpu/test_rotate_butterfly_probe.cpp)
  target_compile_options(test_rotate_butterfly_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(rotate_butterfly)

  # AIV-only inverse-CDF int4x2 quantization; --tiles selects the runtime count.
  add_executable(test_icdf_quantize_probe tests/cpu/test_icdf_quantize_probe.cpp)
  target_compile_options(test_icdf_quantize_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(icdf_quantize)
  add_executable(test_lloydmax_probe tests/cpu/test_lloydmax_probe.cpp)
  target_compile_options(test_lloydmax_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(lloydmax_quantize)
  # The fused operator uses the same runtime tile selection for its page count.
  add_micro_camodel_probe(fused_attention
      kernels/ops/turboquant/turboquant_paged_attention_opt.cpp)

  # The prefill reshape-and-cache operator. Eight chunks of one 16-token strip
  # already wrap the depth-4 L1 ring and the depth-2 per-core UB rings twice,
  # which is what V3 asks a second probe for elsewhere; runtime --slices N adds
  # the grid dimension on top of that, so one binary covers the inductive set
  # 1, 2, 3 strips for multi-block addressing without a build parameter.
  #
  # ASCEND_MICRO_SPARSE_SAMPLING makes the device shapes reachable too: at a
  # grid wider than one sample set the kernel executes only the first, middle
  # and last four blocks of each wave and the runner requires every other
  # block's page to stay canary, so 64 strips cost about what 24 do. It is set
  # here and nowhere else -- NpuTargets.cmake does not set it, so the
  # physical-device library contains no sampling predicate at all. Turn it off
  # for the full-count soak:
  #   cmake -B <dir> -DASCEND_MICRO_SPARSE_SAMPLING=OFF
  option(ASCEND_MICRO_SPARSE_SAMPLING
      "CAModel probe runs only the boundary and midpoint blocks of each wave" ON)
  set(a8_defs "")
  if(ASCEND_MICRO_SPARSE_SAMPLING)
    set(a8_defs ASCEND_MICRO_SPARSE_SAMPLING)
  endif()
  # Unquoted on purpose: an empty a8_defs must vanish from the argument list
  # rather than arrive as an empty third argument, which would reach the
  # compiler as a bare -D.
  add_micro_camodel_probe(turboquant_reshape_and_cache_a8
      kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp ${a8_defs})

  # The BF16 input instance: a dedicated kernel rather than a templated A8, so
  # the verified INT8 path is not touched. Same output contract, same golden
  # oracle -- with integer-valued BF16 activations it must produce byte
  # identical nibbles and scales to the INT8 instance. No sparse sampling: its
  # grids are 1 and 32, where every block runs anyway.
  add_micro_camodel_probe(turboquant_reshape_and_cache_bf16
      kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp)

  # The paged decode attention operator: 16 query tokens, d = 256, an int4
  # paged KV cache, a 12-column eight-strip modulo schedule over two-slot UB
  # rings, and a 32-block grid. Its runner takes --strips/--strip-begin/--block
  # at runtime, so one binary covers both the single-tile canary protocol and
  # the eight-strip steady-state run; no build parameter is needed.
  #
  # Run the strip ladder, not just the canary coordinates. All nine canary
  # coordinates passed while two separate ring-reuse faults were live, because
  # a single strip reuses no slot; it took --strips 4 and --strips 8 to expose
  # them. reports/paged_decode_attention.md sections 4.5 and 4.6.
  add_micro_camodel_probe(paged_decode_attention
      kernels/ops/turboquant/turboquant_paged_decode_attention.cpp)
  # Its Tier-1 host oracle. The CAModel runner compares the device against
  # this oracle, so it cannot tell a wrong device from a wrong oracle: the
  # rotation's factorisation, the egress butterfly, the cache packing and every
  # fixture property are checked here, with no device and no simulator.
  add_executable(test_paged_decode_probe tests/cpu/test_paged_decode_probe.cpp)
  target_compile_options(test_paged_decode_probe PRIVATE -Wall -Wextra -Werror)

  # The AIV-only sub-strip scatter ingress operator (roadmap No.1). Its runner
  # takes --mode/--head/--tokens at runtime, so one binary covers the
  # single-head canary sweep, the 1..16 token sweep, the read-modify-write
  # ladder and the absmax == 0 branch; no build parameter is needed.
  #
  # Run the ladder, not just one coordinate. A single launch into a poisoned
  # cache cannot tell a read-modify-write that preserves from one that rewrites
  # the whole strip from the oracle it is compared against, because the oracle
  # starts from the same poison: only --mode sequential, whose second launch
  # starts from what the FIRST launch left, can.
  add_micro_camodel_probe(reshape_scatter_aiv
      kernels/ops/turboquant/turboquant_reshape_scatter_aiv.cpp)
  # Its Tier-1 host oracle. The layout bijections, the 2048-byte strip window,
  # the (word, bit) merge arithmetic and the power-of-two scale extraction are
  # all settled here, with no device and no simulator.
  add_executable(test_reshape_scatter_probe tests/cpu/test_reshape_scatter_probe.cpp)
  target_compile_options(test_reshape_scatter_probe PRIVATE -Wall -Wextra -Werror)

  # Whether an ingress store can be read back by the fused decode AT ALL. Maps every
  # (block, kv head, token, channel) through both address functions -- the ingress
  # writers STRIP contract (tests/common/turboquant_scatter_oracle.hpp) and the fused
  # decode NZ-TILED NzTiledPackedByte (common/turboquant_layout.h) -- and holds the
  # finding that they name different bytes, with co-byte matchings that share not one
  # pair. It PASSES on the layouts disagreeing: see reports/turboquant_e2e_roundtrip.md
  # for why that is the gate and not a defect, and what would have to change for a real
  # round trip to exist.
  add_executable(test_turboquant_kv_layout_bridge tests/cpu/test_turboquant_kv_layout_bridge.cpp)
  target_compile_options(test_turboquant_kv_layout_bridge PRIVATE -Wall -Wextra -Werror)

  # The host oracle that USED to be the AIV-only single-token decode operator's. That
  # operator is deleted -- kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp
  # and its split-K sibling -- and its CAModel probe, boundary runner and unaligned
  # ladder went with it. The fused decode carries those gates now
  # (tests/sim/kernel/test_camodel_fused_decode_opt.cpp,
  # scripts/run_fused_decode_canary_ladder.sh).
  #
  # This target stays because it is no longer about that kernel.
  # tests/common/single_token_decode_oracle.hpp survives as the shared decode geometry
  # and FP64 reference that tests/common/splitk_decode_fixture.hpp is built on, and
  # through it the AIC Split-K probes, the decode tail probe and the decode benchmark.
  # The GEMV repeat-stride arithmetic, the order-independence those bit-exact panel
  # gates rest on, the egress stage order against the dense rotation and every fixture
  # property are settled here, with no device and no simulator, and nothing else checks
  # them.
  add_executable(test_single_token_decode_probe
      tests/cpu/test_single_token_decode_probe.cpp)
  target_compile_options(test_single_token_decode_probe PRIVATE -Wall -Wextra -Werror)

  # The Split-K synchronisation probe (L2 atomic contention). Three entry
  # points in one kernel -- Mode A collision-free, Mode B shared slot under a
  # Global Memory spinlock, Mode C the tail split's egress flush -- and one
  # runner that takes --mode/--splits/--reps/--fixture at runtime, so one
  # binary covers the whole K sweep and both fixtures.
  #
  # This is the only kernel here that contains a SIMT VF. The atomics it needs
  # exist nowhere else: asc_atomic_exch/_add/_max are SIMT API interfaces
  # declared __simt_callee__, reachable only from a __simt_vf__ dispatched by
  # asc_vf_call -- and asc_vf_call expands to nothing unless SPLIT_CORE_VEC is
  # defined, which simt_api/common_functions.h does for __DAV_VEC__ at
  # __NPU_ARCH__ 3510, i.e. in the AIV leg only. Hence KERNEL_TYPE_AIV_ONLY
  # and hence no Cube path in the probe.
  add_micro_camodel_probe(l2_atomic_synthetic)
  # Its Tier-1 host oracle: the log-sum-exp combine, the order independence
  # the bit-exact device gate rests on, and the egress reference against the
  # dense 256 x 256 rotation. No device, no simulator, milliseconds.
  add_executable(test_l2_atomic_probe tests/cpu/test_l2_atomic_probe.cpp)
  target_compile_options(test_l2_atomic_probe PRIVATE -Wall -Wextra -Werror)

  # The single-tile Cube baseline. A SECOND runner over the SAME frozen
  # production operator source as add_micro_camodel_probe(paged_decode
  # _attention) above -- not a copy of it and not a variant of it. The
  # Split-K crossover needs T_compute_strip measured on the real operator,
  # and the way to get it without touching a frozen file is another runner
  # against the same translation unit: one strip, one block, the ingress
  # rotation and that strip's score panel checked bit-exactly, and the tick
  # spans read out of the Cube dump by
  # scripts/analyze_cube_tile_baseline.py.
  #
  # "The same translation unit" is now literally true. This used to go through
  # add_micro_camodel_probe, which builds a kernel library per call, so the
  # operator was compiled twice on the simulator side and this runner linked
  # the second copy. Both runners declare and call the same
  # `launch_camodel_paged_decode_attention`, so the second library only ever
  # duplicated the first -- and being a separate target it missed the strict
  # warning set applied below. It links the first library now.
  add_micro_camodel_runner(cube_single_tile_baseline
      camodel_paged_decode_attention_kernel)

  # ------------------------------------------- the Split-K bench's sim twin
  # `tests/npu/bench/bench_turboquant_decode_splitk.cpp` is a hardware
  # benchmark and this host has no NPU, so without this target nothing in it
  # would ever execute: the CLI, the percentile statistics, the tail-delta
  # arithmetic, the watchdog census and the combine soundness check would all
  # be unexercised code shipped on the strength of a clean compile.
  #
  # This builds the SAME source against simulator-mode kernel libraries, with
  # ASCEND_MICRO_BENCH_SIM defined so the binary says at startup that its
  # figures are CAModel ticks and not hardware latencies. The benchmark links
  # the fused AIV Split-K and dynamic M=16 AIC decode libraries, with no
  # synthetic reduction launch.
  #
  # It is deliberately NOT part of any aggregate target: a CAModel decode
  # launch is minutes, so a default context sweep would take days. Run it
  # with small counts, as scripts/run_decode_splitk_bench.sh does.
  ascendc_library(camodel_splitk_reduce_kernel SHARED
      kernels/ops/turboquant/turboquant_splitk_reduce.cpp)
  ascendc_compile_definitions(camodel_splitk_reduce_kernel PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(camodel_splitk_reduce_kernel PRIVATE
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # Keep warnings strict in our source while classifying CANN's own headers as
  # system headers, matching the physical-device target -- the same list, from
  # the top-level CMakeLists.txt, so "matching" is now by construction.
  set(splitk_system_options ${ASCEND_MICRO_CANN_SYSTEM_OPTIONS})
  ascendc_compile_options(camodel_paged_decode_attention_kernel PRIVATE
      -std=c++17 -Wall -Werror ${splitk_system_options}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # test_camodel_decode_splitk_boundary went with the AIV split-K kernel it launched. Its
  # gate -- a byte-exact context against an independent scalar oracle, with every
  # unassigned byte and both exterior guard bands held to 0x7FC0BEEF -- is the fused
  # decode runner's now, and that runner prints the same ASSERT_CANARY_INTACT byte census
  # (tests/sim/kernel/test_camodel_fused_decode_opt.cpp).

  add_executable(bench_decode_splitk_sim
      tests/npu/bench/bench_turboquant_decode_splitk.cpp)
  add_executable(test_splitk_decode_probe tests/cpu/test_splitk_decode_probe.cpp)
  target_compile_options(test_splitk_decode_probe PRIVATE -Wall -Wextra -Werror)
  add_executable(test_aic_splitk_decode_probe tests/cpu/test_aic_splitk_decode_probe.cpp)
  target_compile_options(test_aic_splitk_decode_probe PRIVATE -Wall -Wextra -Werror)
  add_executable(test_aic_splitk_acc_error_model tests/cpu/test_aic_splitk_acc_error_model.cpp)
  target_compile_options(test_aic_splitk_acc_error_model PRIVATE -Wall -Wextra -Werror)
  # ASCEND_MICRO_BENCH_SIM says this is the simulator leg. Like the physical target in
  # cmake/NpuTargets.cmake, this one benchmarks the FUSED operator and nothing else by
  # default now: the AIV split-K decoder it used to compare against is deleted, and the
  # AIV-only dispatches went out of the driver with it.
  #
  # TQ_BENCH_LEGACY_DECODERS -- declared in the top-level CMakeLists.txt, so this leg and
  # the physical leg cannot disagree about it -- restores the AIC paged-decode comparison
  # columns. That is what scripts/profile_camodel_trace.sh,
  # scripts/run_decode_splitk_aic_page_depth.sh, scripts/run_decode_splitk_aic_page_race.sh
  # and scripts/run_decode_splitk_strip_ladder.sh drive with --impl aic, and the AIC
  # operator is not what this deprecation removed, so those scripts configure with
  # -DTQ_BENCH_LEGACY_DECODERS=ON. The CLI, the validation and the CSV schema are identical
  # either way; only which kernels a given binary can launch differs.
  target_compile_definitions(bench_decode_splitk_sim PRIVATE ASCEND_MICRO_BENCH_SIM)
  target_include_directories(bench_decode_splitk_sim PRIVATE
      "${ASCEND_HOME_PATH}/include")
  if(TQ_BENCH_LEGACY_DECODERS)
    target_compile_definitions(bench_decode_splitk_sim PRIVATE TQ_BENCH_LEGACY_DECODERS=1)
    target_link_libraries(bench_decode_splitk_sim PRIVATE
        camodel_paged_decode_attention_kernel camodel_fused_decode_opt_kernel ascendcl
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  else()
    target_link_libraries(bench_decode_splitk_sim PRIVATE
        camodel_fused_decode_opt_kernel ascendcl
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  endif()
  target_link_options(bench_decode_splitk_sim PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(bench_decode_splitk_sim PRIVATE -Wall -Wextra -Werror)
  set_target_properties(bench_decode_splitk_sim PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

  # ------------------------ split-K decode validation twin, on CAModel
  # The simulator leg of NpuTargets.cmake's test_npu_decode_splitk_aic. It matters more
  # than most sim twins here: the NPU leg needs a 950PR this harness's development host
  # does not have, so for anyone without one this is the only place the AIC split-K
  # ladder actually runs rather than merely compiles. It is slow -- every rung is a real
  # launch on the model -- so run it with a narrowed matrix when iterating, and read
  # reports/decode_splitk_validation.md for which rungs have been executed on which leg.
  #
  # There used to be an `aiv` leg beside this one, over
  # tests/npu/validation/test_turboquant_decode_splitk_aiv_validation.cpp and the deleted
  # AIV split-K kernel. Both are gone; the fused operator's Split-K workers, including an
  # empty one, are a rung of scripts/run_fused_decode_canary_ladder.sh at both element
  # types, and its hardware ladder is test_npu_fused_decode_opt.
  #
  # It links the ONE kernel library its driver calls into.
  # tests/npu/turboquant_splitk_rigs.hpp declares the launchers the rigs use, but a
  # declaration is not a reference: each rig's `Launch` is an implicitly inline member
  # emitted only where that rig is used, and this driver uses exactly one. `nm -u` on its
  # object says so -- test_decode_splitk_aic_sim needs only
  # launch_turboquant_paged_decode_splitk_aic.
  #
  # A spare link cost a whole extra ascendc_library build, and because CannRebuild.cmake
  # keeps CANN's BUILD_ALWAYS it cost it on EVERY build: a no-op rebuild of the
  # paged-decode library alone is 54 s and 17 compile invocations here. If a ladder later
  # does reach across, the link error names the missing symbol immediately and the library
  # goes back in one line.
  add_executable(test_decode_splitk_aic_sim
      tests/npu/validation/test_turboquant_decode_splitk_aic_validation.cpp)
  target_compile_definitions(test_decode_splitk_aic_sim PRIVATE
      ASCEND_MICRO_DECODE_SPLITK_SIM)
  target_include_directories(test_decode_splitk_aic_sim PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_decode_splitk_aic_sim PRIVATE
      camodel_paged_decode_attention_kernel ascendcl
      "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(test_decode_splitk_aic_sim PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(test_decode_splitk_aic_sim PRIVATE
      -Wall -Wextra -Werror)
  set_target_properties(test_decode_splitk_aic_sim PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

  # ------------------------ unaligned context lengths, on CAModel
  # test_decode_single_token_unaligned_sim is gone with the two AIV decode kernels it
  # launched. The unaligned ladder it drove -- S = 1, 7, 15, 17, 31, 33, 127, 129, 255,
  # 257, 1023, none of them a multiple of the strip and most not of the page, with a
  # quiet-NaN canary in every cache slot past seq_len -- is the fused decode runner's
  # --tail-rows dimension now: rungs 2, 3 and 8 of scripts/run_fused_decode_canary_ladder.sh
  # end a context inside a tile, and rung 8 does it over the whole eight-page block table.

  # ------------------------------------------- mode-4 research probe, on CAModel
  # The simulator twin of NpuTargets.cmake's test_cluster_sync_probe: the same
  # tests/research/kernel_cluster_sync_probe.cpp, built against the model
  # libraries, because the probe's verdict comes from its instruction trace and
  # only this leg produces one. The probe moves no data, so a clean exit proves
  # nothing by itself -- a sticky latch would retire every wait early and the
  # run would still finish. What settles it is the tick order of
  # SET_INTRA_BLOCK against WAIT_INTRA_BLOCK on a single recycled id, and that
  # is read out of the core dumps.
  #
  # Pass an iteration count above the default 2 to give the recycled ids more
  # than one reuse to go wrong on, as scripts/run_reshape_*_camodel.sh do for
  # the operator.
  ascendc_library(camodel_cluster_sync_probe_kernel SHARED
      tests/research/kernel_cluster_sync_probe.cpp)
  ascendc_compile_definitions(camodel_cluster_sync_probe_kernel PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(camodel_cluster_sync_probe_kernel PRIVATE
      -std=c++17 -Wall -Werror -Wno-missing-braces ${splitk_system_options}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
  add_executable(test_cluster_sync_probe_sim
      tests/research/test_cluster_sync_probe.cpp)
  # Same convention as bench_decode_splitk_sim: the binary says at startup that
  # it is the simulator leg, so a pasted log cannot be mistaken for silicon.
  target_compile_definitions(test_cluster_sync_probe_sim PRIVATE
      ASCEND_MICRO_PROBE_SIM)
  target_include_directories(test_cluster_sync_probe_sim PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_cluster_sync_probe_sim PRIVATE
      camodel_cluster_sync_probe_kernel ascendcl
      "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(test_cluster_sync_probe_sim PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(test_cluster_sync_probe_sim PRIVATE
      -Wall -Wextra -Werror)
  set_target_properties(test_cluster_sync_probe_sim PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

  # The BF16 operand-layout micro-probe. Explicit rather than through
  # add_micro_camodel_probe because its runner is test_micro_layout_bf16.cpp,
  # not test_camodel_<name>.cpp: it is a data-movement probe over one tile, not
  # a second runner for an operator. Everything else -- the forwarding rules
  # for ASCEND_MICRO_CANN_LAUNCH, the simulator link line and the install
  # RPATH -- matches the helper above.
  ascendc_library(camodel_micro_layout_bf16_kernel SHARED
      kernels/camodel/micro_layout_bf16.cpp)
  ascendc_compile_definitions(camodel_micro_layout_bf16_kernel PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(camodel_micro_layout_bf16_kernel PRIVATE
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
  add_executable(test_micro_layout_bf16 tests/camodel/test_micro_layout_bf16.cpp)
  target_include_directories(test_micro_layout_bf16 PRIVATE "${ASCEND_HOME_PATH}/include")
  target_link_libraries(test_micro_layout_bf16 PRIVATE camodel_micro_layout_bf16_kernel
      ascendcl "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(test_micro_layout_bf16 PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(test_micro_layout_bf16 PRIVATE -Wall -Wextra -Werror)
  set_target_properties(test_micro_layout_bf16 PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")

endfunction()
