//
// Created by genshen on 2020/5/14.
//

#include <gtest/gtest.h>

#include "eam_calc_test_device.h"
#include "eam_pot_fixture.h"
#include "hip_pot.h"

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
    // set pair energy splines
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

TEST_F(EamPotTest, hip_pot_force_test) {
  std::vector<atom_type::_type_atomic_no> _pot_types(ELE_SIZE);
  for (int i = 0; i < ELE_SIZE; i++) {
    _pot_types[i] = prop_key_list[i];
  }
  // init
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);

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
  deviceForce<false>(key_from, key_to, df_from, df_to, dist2, force_device, LEN);
  for (int i = 0; i < LEN; i++) {
    double force_host = _pot->toForce(key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
    EXPECT_DOUBLE_EQ(force_host, force_device[i]);
  }
  hip_pot::destroyDevicePotTables(d_pot);
}

// in force computation, the two atoms have the same type.
// then, the computation can be simplified (see kernel `hip_pot::hipToForceWrapper`).
TEST_F(EamPotTest, hip_pot_force_single_type_test) {
  std::vector<atom_type::_type_atomic_no> _pot_types(ELE_SIZE);
  for (int i = 0; i < ELE_SIZE; i++) {
    _pot_types[i] = prop_key_list[i];
  }
  // init
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);

  constexpr size_t LEN = 8;
  const double lattice_const = 2.85532 * 2.85532;

  atom_type::_type_prop_key key_from[LEN] = {0, 1, 2, 0, 0, 1, 1, 2};
  atom_type::_type_prop_key key_to[LEN] = {0, 1, 2, 1, 2, 0, 2, 1};
  double df_from[LEN] = {0.1, 0.15, 0.13, 0.12, 0.23, 0.45, 0.2, 0.13};
  double df_to[LEN] = {0.23, 0.43, 0.12, 0.26, 0.24, 0.17, 0.65, 0.45};
  double dist2[LEN] = {
      1.5 * lattice_const, 4.1 * lattice_const,  5.5 * lattice_const,  3.1 * lattice_const,
      3.5 * lattice_const, 4.89 * lattice_const, 6.34 * lattice_const, 7.14 * lattice_const,
  };
  double force_device[LEN];

  // test hip_pot::hipToForceWrapper
  deviceForce<true>(key_from, key_to, df_from, df_to, dist2, force_device, LEN);
  for (int i = 0; i < LEN; i++) {
    double force_host = _pot->toForce(key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
    EXPECT_DOUBLE_EQ(force_host, force_device[i]);
  }
  hip_pot::destroyDevicePotTables(d_pot);
}

TEST_F(EamPotTest, hip_pot_eam_elec_test) {
  std::vector<atom_type::_type_atomic_no> _pot_types(ELE_SIZE);
  for (int i = 0; i < ELE_SIZE; i++) {
    _pot_types[i] = prop_key_list[i];
  }
  // init
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);

  constexpr size_t LEN = 8;
  const double lattice_const = 2.85532 * 2.85532;

  atom_type::_type_prop_key keys[LEN] = {0, 1, 2, 0, 1, 2, 0, 1};
  double dist2[LEN] = {
      1.5 * lattice_const, 4.1 * lattice_const,  5.5 * lattice_const,  3.1 * lattice_const,
      3.5 * lattice_const, 4.89 * lattice_const, 6.34 * lattice_const, 7.14 * lattice_const,
  };
  double rho_device[LEN];

  deviceEamChargeDensity(keys, dist2, rho_device, LEN);
  for (int i = 0; i < LEN; i++) {
    double rho_host = _pot->chargeDensity(keys[i], dist2[i]);
    EXPECT_DOUBLE_EQ(rho_host, rho_device[i]);
  }

  hip_pot::destroyDevicePotTables(d_pot);
}

TEST_F(EamPotTest, hip_pot_eam_dEmbed_test) {
  std::vector<atom_type::_type_atomic_no> _pot_types(ELE_SIZE);
  for (int i = 0; i < ELE_SIZE; i++) {
    _pot_types[i] = prop_key_list[i];
  }
  // init
  hip_pot::_type_device_pot d_pot = hip_pot::potCopyHostToDevice(_pot, _pot_types);
  hip_pot::assignDevicePot(d_pot);

  constexpr size_t LEN = 8;

  atom_type::_type_prop_key keys[LEN] = {0, 1, 2, 0, 1, 2, 0, 1};
  double rhos[LEN] = {
      2.5, 4.0, 0.5, 3.1, 3.5, 4.89, 6.34, 7.14,
  };
  double df_device[LEN];

  deviceEamDEmbedEnergy(keys, rhos, df_device, LEN);
  for (int i = 0; i < LEN; i++) {
    double rho_host = _pot->dEmbedEnergy(keys[i], rhos[i]);
    EXPECT_DOUBLE_EQ(rho_host, df_device[i]);
  }

  hip_pot::destroyDevicePotTables(d_pot);
}
