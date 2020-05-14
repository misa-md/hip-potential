//
// Created by genshen on 2020/05/14.
//

#ifndef HIP_POT_GTEST_EAM_POT_FIXTURE_H
#define HIP_POT_GTEST_EAM_POT_FIXTURE_H

#include <eam.h>
#include <gtest/gtest.h>

class EamPotTest : public ::testing::Test {
public:
  eam *_pot = nullptr;

  void SetUp() override {
    const unsigned int ELE_SIZE = 3;
    const unsigned int DATA_SIZE = 5000;
    const unsigned int root_rank = 0;
    int own_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);

    _pot = eam::newInstance(ELE_SIZE, root_rank, own_rank, MPI_COMM_WORLD);
    atom_type::_type_prop_key *prop_key_list = new atom_type::_type_prop_key[ELE_SIZE];

    // init data buffer
    double data_buff[5000] = {};
    for (int i = 0; i < 5000; i++) {
      data_buff[i] = i * 0.01;
    }

    for (int i = 0; i < ELE_SIZE; i++) {
      int key = i; // key is atom number
      prop_key_list[i] = key;
      _pot->embedded.append(key, DATA_SIZE, 0.0, 0.001, data_buff);
      _pot->electron_density.append(key, DATA_SIZE, 0.0, 0.001, data_buff);
    }

    int i, j;
    for (i = 0; i < ELE_SIZE; i++) {
      for (j = 0; j <= i; j++) {
        _pot->eam_phi.append(prop_key_list[i], prop_key_list[j], DATA_SIZE, 0.0, 0.001, data_buff);
      }
    }

    _pot->eamBCast(root_rank, own_rank, MPI_COMM_WORLD);
    _pot->interpolateFile(); // interpolation
  }

  void TearDown() override { delete _pot; }
};

#endif // HIP_POT_GTEST_EAM_POT_FIXTURE_H
