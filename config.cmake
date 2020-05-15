option(HIP_POT_MPI_ENABLE_FLAG "Enable MPI" ON) # enable MPI
option(HIP_POT_TEST_BUILD_ENABLE_FLAG "Enable building tests" ON) # enable test

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

if (CMAKE_BUILD_TYPE MATCHES "^(Debug|DEBUG|debug)$")
    set(HIP_POT_DEBUG_ENABLE_FLAG ON)
endif ()

#############
## const ##
#############
set(HIP_POT_LIB_NAME hip_pot)

# test
set(HIP_POT_UINT_TEST_NAME "hip_pot-unit-test")
