
set(HIP_POT_SOURCE_FILES
        hip_pot.cpp
        hip_pot_device.cc
        hip_eam_device.cc
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
