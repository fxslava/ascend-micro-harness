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
  function(add_micro_camodel_probe name)
    set(kernel_src "kernels/camodel/${name}.cpp")
    if(ARGC GREATER 1)
      if(ARGV1 MATCHES "/")
        set(kernel_src "${ARGV1}")
      else()
        set(kernel_src "kernels/camodel/${ARGV1}.cpp")
      endif()
    endif()
    ascendc_library(camodel_${name}_kernel SHARED ${kernel_src})
    ascendc_compile_definitions(camodel_${name}_kernel PRIVATE
        ASCEND_MICRO_CANN_LAUNCH)
    ascendc_compile_options(camodel_${name}_kernel PRIVATE
        -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)
    add_executable(test_camodel_${name} tests/camodel/test_camodel_${name}.cpp)
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

  # AIV-only inverse-CDF int4x2 quantization; --tiles selects the runtime count.
  add_executable(test_icdf_quantize_probe tests/cpu/test_icdf_quantize_probe.cpp)
  target_compile_options(test_icdf_quantize_probe PRIVATE -Wall -Wextra -Werror)
  add_micro_camodel_probe(icdf_quantize)
  # The fused operator uses the same runtime tile selection for its page count.
  add_micro_camodel_probe(fused_attention
      kernels/ops/turboquant/turboquant_paged_attention_opt.cpp)

  # The prefill reshape-and-cache operator. Its grid is fixed by the strip shape
  # -- eight chunks of one 16-token page -- so there is no build parameter: the
  # eight chunks already wrap the depth-4 L1 ring and the depth-2 per-core UB
  # rings twice, which is what V3 asks a second probe for elsewhere.
  add_micro_camodel_probe(turboquant_reshape_and_cache_a8
      kernels/ops/turboquant/turboquant_reshape_and_cache_a8.cpp)
endfunction()
