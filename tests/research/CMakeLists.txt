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

ascendc_library(kernel_cluster_sync_probe SHARED
    kernel_cluster_sync_probe.cpp
)

ascendc_compile_definitions(kernel_cluster_sync_probe PRIVATE
    ASCEND_MICRO_CANN_LAUNCH
)

ascendc_compile_options(kernel_cluster_sync_probe PRIVATE
    -std=c++17 -Wall -Werror
    -Wno-missing-braces
    ${CANN_SYSTEM_OPTIONS}
    -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
)

# 2. Хостовый исполняемый файл
add_executable(test_cluster_sync_probe
    test_cluster_sync_probe.cpp
)

target_include_directories(test_cluster_sync_probe PRIVATE
    "${ASCEND_HOME_PATH}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(test_cluster_sync_probe PRIVATE
    kernel_cluster_sync_probe
    ascendcl
    stdc++
)

target_compile_options(test_cluster_sync_probe PRIVATE
    -Wall -Wextra -Werror
)

set_target_properties(test_cluster_sync_probe PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${ASCEND_HOME_PATH}/lib64"
)