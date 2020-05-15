//
// Created by genshen on 2020/5/14.
//

#include "eam_calc_test_device.h"
#include "eam_pot_fixture.h"
#include "hip_pot.h"
#include <gtest/gtest.h>

TEST_F(EamPotTest, hip_pot_init_test) {
  auto _pot_types = std::vector<atom_type::_type_atomic_no>{0, 1, 2};
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);
  hip_pot::destroyDevicePotTables(d_pot);
}

TEST_F(EamPotTest, hip_pot_check_spline_copy_test) {
  auto _pot_types = std::vector<atom_type::_type_atomic_no>{0, 1, 2};

  double fd = 0.0;
  for (hip_pot::_type_device_table_size i = 0; i < ELE_SIZE; i++) {
    // set elec and embed energy splines
    auto elec = _pot->electron_density.getEamItemByType(_pot_types[i]);
    auto emb = _pot->embedded.getEamItemByType(_pot_types[i]);
    for (int k = 0; k < DATA_SIZE; k++) {
      for (int s = 0; s < 7; s++) {
        elec->spline[k][s] = fd;
        fd += 1.0;
        emb->spline[k][s] = fd;
        fd += 1.0;
      }
    }
    // set pair enregy splines
    for (hip_pot::_type_device_table_size j = i; j < ELE_SIZE; j++) {
      auto pair = _pot->eam_phi.getPhiByEamPhiByType(_pot_types[i], _pot_types[j]);
      for (int k = 0; k < DATA_SIZE; k++) {
        for (int s = 0; s < 7; s++) {
          pair->spline[k][s] = fd;
          fd += 1.0;
        }
      }
    }
  }

  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);
  checkPotSplinesCopy(ELE_SIZE, DATA_SIZE);
  hip_pot::destroyDevicePotTables(d_pot);
}

