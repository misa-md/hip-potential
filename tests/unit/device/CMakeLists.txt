set(HIP_POT_DEVICE_TEST_FILES
        eam_calc_test_device.cpp
        eam_calc_kernel.cpp
        )

set_source_files_properties(${HIP_POT_DEVICE_TEST_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_library(${DEVICE_TEST_LIB_NAME} STATIC ${HIP_POT_DEVICE_TEST_FILES})

target_include_directories(
        ${DEVICE_TEST_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
        ${DEVICE_TEST_LIB_NAME}
        PUBLIC
        pot::pot
        ${HIP_POT_LIB_NAME}
)
