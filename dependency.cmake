# enable C++11 support.
set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} "-std=c++11")

# check hip toolchain
# see https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/2_Cookbook/12_cmake_hip_add_executable
if (NOT DEFINED HIP_PATH)
    if (NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else ()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif ()
endif ()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

find_package(HIP REQUIRED)
if (HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else ()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif ()

## if it is NV, set NVCC flags
if (NOT DEFINED HIP_PLATFORM)
    if (DEFINED $ENV{HIP_PLATFORM})
        set(HIP_PLATFORM $ENV{HIP_PLATFORM})
    endif ()
endif ()
if (DEFINED HIP_PLATFORM)
    if (${HIP_PLATFORM} MATCHES "^(Nvidia|NVIDIA|nvidia)$")
        set(NV_PLATFORM ON)
    endif ()
endif ()
if (NV_PLATFORM)
    set(HIP_NVCC_FLAGS "${HIP_NVCC_FLAGS} -rdc=true")
endif ()

## check MPI
if (HIP_POT_MPI_ENABLE_FLAG)
    find_package(MPI REQUIRED)
    MESSAGE(STATUS "MPI_INCLUDE dir:" ${MPI_INCLUDE_PATH})
    MESSAGE(STATUS "MPI_LIBRARIES dir:" ${MPI_LIBRARIES})

    if (MPI_CXX_COMPILE_FLAGS)
        set(COMPILE_FLAGS "${COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    endif ()

    if (MPI_CXX_LINK_FLAGS)
        set(LINK_FLAGS "${LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    endif ()

    include_directories(${MPI_CXX_INCLUDE_PATH})

    set(HIP_POT_EXTRA_LIBS ${POT_EXTRA_LIBS} ${MPI_LIBRARIES}) #add mpi lib
endif ()

include(pkg.dep.cmake)
