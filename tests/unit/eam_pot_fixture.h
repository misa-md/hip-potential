//
// Created by genshen on 2020/05/14.
//

#ifndef HIP_POT_GTEST_EAM_POT_FIXTURE_H
#define HIP_POT_GTEST_EAM_POT_FIXTURE_H

#include <eam.h>
#include <gtest/gtest.h>
#include <random>

class EamPotTest : public ::testing::Test {
public:
  eam *_pot = nullptr;
  atom_type::_type_prop_key *prop_key_list = nullptr;

  static constexpr unsigned int ELE_SIZE = 3;
  static constexpr unsigned int DATA_SIZE = 5000;

  void SetUp() override { init(); }
  void init() {
    const unsigned int root_rank = 0;
    int own_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);

    _pot = eam::newInstance(ELE_SIZE, root_rank, own_rank, MPI_COMM_WORLD);
    prop_key_list = new atom_type::_type_prop_key[ELE_SIZE];

    std::mt19937 rng(27321);
    std::uniform_real_distribution<double> dist_f(0.0, 5.0);

    // data buffer
    double data_buff_emb[5000] = {};
    double data_buff_elec[5000] = {};
    for (int k = 0; k < 5000; k++) {
      data_buff_emb[k] = k * dist_f(rng);
      data_buff_elec[k] = k * dist_f(rng);
    }

    for (int i = 0; i < ELE_SIZE; i++) {
      int key = i; // key is atom number
      prop_key_list[i] = key;
      _pot->embedded.append(key, DATA_SIZE, 0.0, 0.001, data_buff_emb);
      _pot->electron_density.append(key, DATA_SIZE, 0.0, 0.001, data_buff_elec);
    }

    int i, j;
    for (i = 0; i < ELE_SIZE; i++) {
      double data_buff[5000] = {};
      for (int k = 0; k < 5000; k++) {
        data_buff[k] = k * dist_f(rng);
      }
      for (j = 0; j <= i; j++) {
        _pot->eam_phi.append(prop_key_list[i], prop_key_list[j], DATA_SIZE, 0.0, 0.001, data_buff);
      }
    }

    _pot->eamBCast(root_rank, own_rank, MPI_COMM_WORLD);
    _pot->interpolateFile(); // interpolation
  }

  void TearDown() override {
    delete _pot;
    delete[] prop_key_list;
  }
};

#endif // HIP_POT_GTEST_EAM_POT_FIXTURE_H
