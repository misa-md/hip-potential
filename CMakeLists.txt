cmake_minimum_required(VERSION 3.8)

project(hip-pot)

set(CMAKE_CXX_STANDARD 11)

include(config.cmake)
include(dependency.cmake)

add_subdirectory(src)

##############################
## test configure
##############################
if (HIP_POT_TEST_BUILD_ENABLE_FLAG)
    MESSAGE(STATUS "TEST is enabled")
    enable_testing()
    add_subdirectory(tests)
endif ()
