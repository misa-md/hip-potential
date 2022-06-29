
include(configure.cmake)

set(HIP_POT_SOURCE_FILES
        hip_pot.cpp
        hip_eam_device.cc
        hip_pot_device_global_vars.cc
        )
set_source_files_properties(${HIP_POT_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

hip_add_library(${HIP_POT_LIB_NAME} STATIC ${HIP_POT_SOURCE_FILES})
# target_compile_options(${HIP_POT_LIB_NAME} PRIVATE "-fPIC")

target_include_directories(
        ${HIP_POT_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        ${HIP_POT_LIB_NAME}
        PUBLIC
        pot::pot
        ${HIP_LINKING_LIBS}
)

set(TARGETS_EXPORT_NAME ${HIP_POT_LIB_NAME}-targets)
set(HIP_POT_LIB_CMAKE_DIR lib/cmake/${HIP_POT_LIB_NAME})
set(HIP_POT_LIB_CMAKE_NAMESPACE ${HIP_POT_LIB_NAME}::)
set(HIP_POT_LIB_CMAKE_CONFIG_IN "${HIP_POT_LIB_NAME}-config.cmake.in")
set(HIP_POT_LIB_CMAKE_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/${HIP_POT_LIB_NAME}-config.cmake")

# lib alias to makes it possible to always use the namespace version consistently
add_library(${HIP_POT_LIB_NAME}::${HIP_POT_LIB_NAME} ALIAS ${HIP_POT_LIB_NAME})


install(TARGETS ${HIP_POT_LIB_NAME}
        EXPORT ${TARGETS_EXPORT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

# install header.
install(DIRECTORY "./"
        DESTINATION "include" # todo adding prefxi "pot": fatal error: 'types.h' file not found
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.inl"
        )

# install export target and config for find_package
install(EXPORT ${TARGETS_EXPORT_NAME} DESTINATION ${HIP_POT_LIB_CMAKE_DIR} NAMESPACE ${HIP_POT_LIB_CMAKE_NAMESPACE})
include(CMakePackageConfigHelpers)
configure_package_config_file(
        ${HIP_POT_LIB_CMAKE_CONFIG_IN} ${HIP_POT_LIB_CMAKE_CONFIG_OUT}
        INSTALL_DESTINATION ${HIP_POT_LIB_CMAKE_DIR}
)
install(FILES ${HIP_POT_LIB_CMAKE_CONFIG_OUT} DESTINATION ${HIP_POT_LIB_CMAKE_DIR})
