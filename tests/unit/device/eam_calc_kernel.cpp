//
// Created by genshen on 2022/8/28.
//

#include <hip/hip_runtime.h>

#include "hip_eam_device.h"

__global__ void _kernelEamForce(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to, double *df_from,
                                double *df_to, double *dist2, double *forces, size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    forces[i] = hip_pot::hipToForce(key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
  }
}

void kernel_wrapper_EamForce(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                             double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces,
                             size_t len) {
  const size_t B = 512; // blocks
  const size_t T = 128; // threads
  hipLaunchKernelGGL(_kernelEamForce, dim3(B, 1), dim3(T, 1), 0, 0, dev_key_from, dev_key_to, dev_df_from, dev_df_to,
                     dev_dist2, dev_forces, len);
}

__global__ void _kernelEamForceSingleType(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to,
                                          double *df_from, double *df_to, double *dist2, double *forces, size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    forces[i] = hip_pot::hipToForceAdaptive(key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
  }
}

void kernel_wrapper_EamForceSingleType(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                                       double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces,
                                       size_t len) {
  const size_t B = 512; // blocks
  const size_t T = 128; // threads
  hipLaunchKernelGGL(_kernelEamForceSingleType, dim3(B, 1), dim3(T, 1), 0, 0, dev_key_from, dev_key_to, dev_df_from,
                     dev_df_to, dev_dist2, dev_forces, len);
}

__global__ void _kernelEamForceSegmented(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to,
                                         double *df_from, double *df_to, double *dist2, double *forces, size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    forces[i] = hip_pot::hipToForce<PairSegmentedSplineLoader, RhoSegmentedSplineLoader>(
        key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
  }
}

void kernel_wrapper_EamForceSegmented(atom_type::_type_prop_key *dev_key_from, atom_type::_type_prop_key *dev_key_to,
                                      double *dev_df_from, double *dev_df_to, double *dev_dist2, double *dev_forces,
                                      size_t len) {
  const size_t B = 512; // blocks
  const size_t T = 128; // threads
  hipLaunchKernelGGL(_kernelEamForceSegmented, dim3(B, 1), dim3(T, 1), 0, 0, dev_key_from, dev_key_to, dev_df_from,
                     dev_df_to, dev_dist2, dev_forces, len);
}

__global__ void _kernelEamForceSegmentedSingleType(atom_type::_type_prop_key *key_from,
                                                   atom_type::_type_prop_key *key_to, double *df_from, double *df_to,
                                                   double *dist2, double *forces, size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    // todo: template parameter: other cases.
    forces[i] = hip_pot::hipToForceAdaptive<double, PairSplineLoader, RhoSegmentedSplineLoader, true>(
        key_from[i], key_to[i], dist2[i], df_from[i], df_to[i]);
  }
}

void kernel_wrapper_EamForceSegmentedSingleType(atom_type::_type_prop_key *dev_key_from,
                                                atom_type::_type_prop_key *dev_key_to, double *dev_df_from,
                                                double *dev_df_to, double *dev_dist2, double *dev_forces, size_t len) {
  const size_t B = 512; // blocks
  const size_t T = 128; // threads
  hipLaunchKernelGGL(_kernelEamForceSegmentedSingleType, dim3(B, 1), dim3(T, 1), 0, 0, dev_key_from, dev_key_to,
                     dev_df_from, dev_df_to, dev_dist2, dev_forces, len);
}
