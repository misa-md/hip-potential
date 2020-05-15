//
// Created by genshen on 2020/05/14
//

#include <gtest/gtest.h>

#  ifdef HIP_POT_TEST_MPI_ENABLE_FLAG
#    include "mpi.h"

class MPIEnvironment : public ::testing::Environment {
public:
    virtual void SetUp() {
        int mpiError = MPI_Init(NULL, NULL);
        ASSERT_FALSE(mpiError);
    }

    virtual void TearDown() {
        int mpiError = MPI_Finalize();
        ASSERT_FALSE(mpiError);
    }

    virtual ~MPIEnvironment() {}
};
#  endif  // end HIP_POT_TEST_MPI_ENABLE_FLAG

// see https://github.com/google/googletest/issues/822 for more information.
// main function for adapt mpi environment
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
#ifdef HIP_POT_TEST_MPI_ENABLE_FLAG
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
#endif  // end HIP_POT_TEST_MPI_ENABLE_FLAG
    return RUN_ALL_TESTS();
}
