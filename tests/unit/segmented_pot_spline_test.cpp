//
// Created by genshen on 2022/8/27.
//

#include <gtest/gtest.h>

#include "device/eam_calc_test_device.h"
#include "eam_calc_kernel.h"
#include "hip_pot.h"
#include "segmented_eam_pot_fixture.hpp"

TEST_F(SegmentedEamPotTest, hip_pot_segmented_force_test) {
  constexpr size_t LEN = 8;
  const double lattice_const = 2.85532 * 2.85532;

  atom_type::_type_prop_key key_from[LEN] = {0, 1, 2, 0, 2, 2, 0, 1};
  atom_type::_type_prop_key key_to[LEN] = {2, 1, 0, 1, 2, 1, 0, 2};
  double df_from[LEN] = {0.1, 0.15, 0.13, 0.12, 0.23, 0.45, 0.2, 0.13};
  double df_to[LEN] = {0.23, 0.43, 0.12, 0.26, 0.24, 0.17, 0.65, 0.45};
  double dist2[LEN] = {
      1.5 * lattice_const, 4.1 * lattice_const,  5.5 * lattice_const,  3.1 * lattice_const,
      3.5 * lattice_const, 4.89 * lattice_const, 6.34 * lattice_const, 7.14 * lattice_const,
  };
  double force_device[LEN];

  // test hip_pot::hipToForce
  deviceForce(key_from, key_to, df_from, df_to, dist2, force_device, LEN, kernel_wrapper_EamForceSegmentedSingleType);
  for (int i = 0; i < LEN; i++) {
    double force_host = _pot->toForce(key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
    EXPECT_DOUBLE_EQ(force_host, force_device[i]);
  }
}
