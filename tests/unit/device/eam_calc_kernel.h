//
// Created by genshen on 2022/8/28.
//

#ifndef HIP_POT_EAM_CALC_KERNEL_H
#define HIP_POT_EAM_CALC_KERNEL_H

#include "hip_pot.h"
#include "types.h"

typedef void (*LaunchKernelFunction)(atom_type::_type_prop_key *, atom_type::_type_prop_key *, double *, double *,
                                     double *, double *, size_t);

void kernel_wrapper_EamForce(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                             double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces, size_t len);

void kernel_wrapper_EamForceSingleType(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                                       double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces,
                                       size_t len);

void kernel_wrapper_EamForceSegmented(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                                      double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces,
                                      size_t len);

void kernel_wrapper_EamForceSegmentedSingleType(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to,
                                                double *df_from, double *df_to, double *dist2, double *forces,
                                                size_t len);

#endif // HIP_POT_EAM_CALC_KERNEL_H
