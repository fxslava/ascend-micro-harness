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
    add_executable(test_camodel_${name} tests/camodel/test_camodel_${name}.cpp)
    if(extra_defs)
      target_compile_definitions(test_camodel_${name} PRIVATE ${extra_defs})
    endif()
    target_include_directories(test_camodel_${name} PRIVATE "${ASCEND_HOME_PATH}/include")
    target_link_libraries(test_camodel_${name} PRIVATE camodel_${name}_kernel ascendcl
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
    target_link_options(test_camodel_${name} PRIVATE
        "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
    target_compile_options(test_camodel_${name} PRIVATE -Wall -Wextra -Werror)
    set_target_properties(test_camodel_${name} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64")
  endfunction()

  add_micro_camodel_probe(dequant4)
  add_micro_camodel_probe(cube_matmul)
  add_micro_camodel_probe(wta_fixpipe)
  add_executable(test_wta_convergence_probe tests/cpu/test_wta_convergence_probe.cpp)
  target_compile_options(test_wta_convergence_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(normalize_store)
  add_micro_camodel_probe(online_softmax)

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

  # The AIV-only single-token decode operator (roadmap No.3): 6 query rows,
  # d = 256, a grid of (request, kv_head) with one task per physical AIV, and
  # no Cube, no L1 and no cross-core flag of any kind. Its runner takes
  # --head/--pages/--strips/--rows at runtime, so one binary covers the
  # single-head canary sweep, the multi-page rung and the row sweep; no build
  # parameter is needed.
  #
  # Run the canary coordinates AND the multi-page rung. A single-page run
  # cannot tell a kernel that walks the block table from one that ignores it,
  # because with one page there is only one page to read -- it takes --pages 4
  # to make a wrong page observable, and the oracle's nibble generator had to
  # be extended before even that was true (tests/common/single_token_decode
  # _oracle.hpp).
  add_micro_camodel_probe(single_token_decode_aiv
      kernels/ops/turboquant/turboquant_single_token_decode_aiv.cpp)
  # Its Tier-1 host oracle. The GEMV's repeat-stride arithmetic, the
  # order-independence the bit-exact panel gate rests on, the egress stage
  # order against the dense rotation and every fixture property are settled
  # here, with no device and no simulator.
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
  add_micro_camodel_probe(cube_single_tile_baseline
      kernels/ops/turboquant/turboquant_paged_decode_attention.cpp)

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

  ascendc_library(camodel_single_token_decode_splitk_kernel SHARED
      kernels/ops/turboquant/turboquant_single_token_decode_splitk_aiv.cpp)
  # Keep warnings strict in our source while classifying CANN's own headers
  # as system headers, matching the physical-device target.
  set(splitk_system_options "")
  foreach(path IN ITEMS
      asc/impl/adv_api asc/impl/basic_api asc/impl/c_api
      asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils
      asc asc/include asc/include/adv_api asc/include/basic_api
      asc/include/aicpu_api asc/include/c_api
      asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils
      tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
    list(APPEND splitk_system_options
        "-isystem${ASCEND_HOME_PATH}/x86_64-linux/${path}")
  endforeach()
  ascendc_compile_definitions(camodel_single_token_decode_splitk_kernel PRIVATE
      ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(camodel_single_token_decode_splitk_kernel PRIVATE
      -std=c++17 -Wall -Werror ${splitk_system_options}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(camodel_paged_decode_attention_kernel PRIVATE
      -std=c++17 -Wall -Werror ${splitk_system_options}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  add_executable(bench_decode_splitk_sim
      tests/npu/bench/bench_turboquant_decode_splitk.cpp)
  add_executable(test_splitk_decode_probe tests/cpu/test_splitk_decode_probe.cpp)
  target_compile_options(test_splitk_decode_probe PRIVATE -Wall -Wextra -Werror)
  add_executable(test_aic_splitk_decode_probe tests/cpu/test_aic_splitk_decode_probe.cpp)
  target_compile_options(test_aic_splitk_decode_probe PRIVATE -Wall -Wextra -Werror)
  target_compile_definitions(bench_decode_splitk_sim PRIVATE ASCEND_MICRO_BENCH_SIM)
  target_include_directories(bench_decode_splitk_sim PRIVATE
      "${ASCEND_HOME_PATH}/include")
  target_link_libraries(bench_decode_splitk_sim PRIVATE
      camodel_single_token_decode_splitk_kernel camodel_paged_decode_attention_kernel ascendcl
      "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
  target_link_options(bench_decode_splitk_sim PRIVATE
      "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
  target_compile_options(bench_decode_splitk_sim PRIVATE -Wall -Wextra -Werror)
  set_target_properties(bench_decode_splitk_sim PROPERTIES
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
