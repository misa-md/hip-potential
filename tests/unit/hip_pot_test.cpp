//
// Created by genshen on 2020/5/14.
//

#include "eam_pot_fixture.h"
#include "hip_pot.h"
#include <gtest/gtest.h>

TEST_F(EamPotTest, hip_pot_init_test) {
  auto _pot_types = std::vector<atom_type::_type_atomic_no>{0, 1, 2};
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::destroyDevicePotTables(d_pot);
}
