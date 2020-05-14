set(TEST_HEADER_FILES
        eam_pot_fixture.h
        )

set(TEST_SOURCE_FILES
        main.cpp
        hip_pot_test.cpp
        )

# set binary path and lib storage path wile compiling process.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_executable(${HIP_POT_UINT_TEST_NAME} ${TEST_SOURCE_FILES} ${TEST_HEADER_FILES})

# Extra linking for the project.
if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
    # googletest is added by `find_package(GTest)`
    target_link_libraries(${HIP_POT_UINT_TEST_NAME}
            PRIVATE GTest::gtest GTest::gtest_main ${HIP_POT_LIB_NAME} ${HIP_POT_EXTRA_LIBS}
            )
else ()
    # googletest is added by `add_subdirectory`
    target_link_libraries(${HIP_POT_UINT_TEST_NAME}
            PRIVATE gtest gtest_main ${HIP_POT_LIB_NAME} ${HIP_POT_EXTRA_LIBS}
            )
endif ()

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable runUnitTests to see those specific tests.
add_test(NAME unit-test COMMAND ${HIP_POT_UINT_TEST_NAME})

##########################
##### install files
##########################
install(TARGETS ${HIP_POT_UINT_TEST_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
