//
// Created by genshen on 2020-05-14
//

#ifndef HIP_POT_EAM_CALC_TEST_DEVICE_H
#define HIP_POT_EAM_CALC_TEST_DEVICE_H

#include "types.h"

void checkPotSplinesCopy(int ele_size, int data_size);

void deviceForce(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to, double *df_from, double *df_to,
                 double *dist2, double *forces, size_t len);

void deviceEamChargeDensity(atom_type::_type_prop_key *keys, double *dist2, double *rhos, size_t len);

void deviceEamDEmbedEnergy(atom_type::_type_prop_key *keys, double *rhos, double *dfs, size_t len);

#endif // HIP_POT_EAM_CALC_TEST_DEVICE_H
