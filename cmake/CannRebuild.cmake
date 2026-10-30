# CANN's legacy preprocessing links its compiler objects in place, changing
# ET_REL into ET_EXEC. Its always-run merge target cannot link those ET_EXEC
# files a second time. Recompile just the consumed objects before each build;
# keep CANN's BUILD_ALWAYS behavior so source/header changes are still seen.
function(add_ascend_micro_cann_rebuild_steps)
  get_property(targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    if(NOT target MATCHES "_preprocess$")
      continue()
    endif()
    get_target_property(source_dir ${target} _EP_SOURCE_DIR)
    if(NOT source_dir MATCHES "/legacy_modules/device_preprocess_project$")
      continue()
    endif()
    ExternalProject_Get_Property(${target} BINARY_DIR)
    ExternalProject_Add_Step(${target} prepare_core_objects
        COMMAND "${ASCEND_PYTHON_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_cann_preprocess.py"
            "${BINARY_DIR}"
        DEPENDEES configure
        DEPENDERS build
        ALWAYS TRUE)
  endforeach()
endfunction()
