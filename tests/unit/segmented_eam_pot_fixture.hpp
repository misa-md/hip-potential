//
// Created by genshen on 2022/8/27.
//

#ifndef HIP_POT_GTEST_SEGMENTE_EAM_POT_FIXTURE_H
#define HIP_POT_GTEST_SEGMENTE_EAM_POT_FIXTURE_H

#include <eam.h>
#include <gtest/gtest.h>
#include <random>

#include "eam_pot_fixture.h"
#include "hip_pot.h"

class SegmentedEamPotTest : public EamPotTest {
public:
  void SetUp() override {
    EamPotTest::init();
    // copy potential spline to device side.
    std::vector<atom_type::_type_atomic_no> _pot_types(ELE_SIZE);
    for (int i = 0; i < ELE_SIZE; i++) {
      _pot_types[i] = prop_key_list[i];
    }
    // perform copy
    d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
    hip_pot::assignDevicePot(d_pot);
    // copy segmented splines:
    hip_pot::assignDeviceSegmentedSpline(d_pot);
  }
  void TearDown() override { hip_pot::destroyDevicePotTables(d_pot); }

private:
  hip_pot::_type_device_pot d_pot;
};

#endif // HIP_POT_GTEST_SEGMENTE_EAM_POT_FIXTURE_H
