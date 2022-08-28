//
// Created by genshen on 2020-05-14
//

#include <hip/hip_runtime.h>

#include "eam_calc_test_device.h"
#include "hip_eam_device.h"
#include "hip_pot_dev_tables_compact.hpp"
#include "hip_pot_device_global_vars.h"
#include "hip_pot_macros.h"

#define dev_assert(X)                                                                                                  \
  if (!(X)) {                                                                                                          \
    const int tid = threadIdx.x;                                                                                       \
    printf("assert failed in tid %d: %s, %d\n", tid, __FILE__, __LINE__);                                              \
  }                                                                                                                    \
  return;

__global__ void _kernelCheckSplineCopy(int ele_size, int data_size) {
  double fd = 0.0;
  for (hip_pot::_type_device_table_size i = 0; i < ele_size; i++) {
    // set elec and embed energy splines
    auto elec = pot_table_ele_charge_density_by_key(data_size, i);
    auto emb = pot_table_embedded_energy_by_key(data_size, i);
    for (int k = 0; k < data_size; k++) {
      for (int s = 0; s < 7; s++) {
        if (elec[k][s] != fd) {
          dev_assert(false);
        }
        fd += 1.0;
        if (emb[k][s] != fd) {
          dev_assert(false);
        }
        fd += 1.0;
      }
    }
    // set pair energy splines
    for (hip_pot::_type_device_table_size j = i; j < ele_size; j++) {
      int index = ele_size * i - (i + 1) * i / 2 + j;
      auto pair = pot_table_pair_by_key(data_size, index);
      for (int k = 0; k < data_size; k++) {
        for (int s = 0; s < 7; s++) {
          if (pair[k][s] != fd) {
            dev_assert(false);
          }
          fd += 1.0;
        }
      }
    }
  }
}

void checkPotSplinesCopy(int ele_size, int data_size) {
  hipLaunchKernelGGL(_kernelCheckSplineCopy, dim3(16, 1), dim3(1, 1), 0, 0, ele_size, data_size);
}

void deviceForce(atom_type::_type_prop_key *key_from, atom_type::_type_prop_key *key_to, double *df_from, double *df_to,
                 double *dist2, double *forces, size_t len, LaunchKernelFunction launch_kernel) {
  const size_t B = 512; // blocks
  const size_t T = 128; // threads

  atom_type::_type_prop_key *deviceKeyFrom;
  atom_type::_type_prop_key *deviceKeyTo;
  double *deviceDfFrom;
  double *deviceDfTo;
  double *deviceDist2;
  double *deviceForces; // output of kernel

  HIP_CHECK(hipMalloc((void **)&deviceKeyFrom, len * sizeof(atom_type::_type_prop_key)));
  HIP_CHECK(hipMalloc((void **)&deviceKeyTo, len * sizeof(atom_type::_type_prop_key)));
  HIP_CHECK(hipMalloc((void **)&deviceDfFrom, len * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&deviceDfTo, len * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&deviceDist2, len * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&deviceForces, len * sizeof(double)));

  HIP_CHECK(hipMemcpy(deviceKeyFrom, key_from, len * sizeof(atom_type::_type_prop_key), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceKeyTo, key_to, len * sizeof(atom_type::_type_prop_key), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceDfFrom, df_from, len * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceDfTo, df_to, len * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceDist2, dist2, len * sizeof(double), hipMemcpyHostToDevice));

  //  if (TEST_SEGMENTED_SPLINE) {
  //    if (SINGLE_TYPE) {
  launch_kernel(deviceKeyFrom, deviceKeyTo, deviceDfFrom, deviceDfTo, deviceDist2, deviceForces, len);
  //    } else {
  //  hipLaunchKernelGGL(_kernelEamForceSegmented, dim3(B, 1), dim3(T, 1), 0, 0, deviceKeyFrom, deviceKeyTo,
  //  deviceDfFrom,
  //                     deviceDfTo, deviceDist2, deviceForces, len);
  //    }
  //  } else {
  //    if (SINGLE_TYPE) {
  //      hipLaunchKernelGGL(_kernelEamForceSingleType, dim3(B, 1), dim3(T, 1), 0, 0, deviceKeyFrom, deviceKeyTo,
  //                         deviceDfFrom, deviceDfTo, deviceDist2, deviceForces, len);
  //    } else {
  //      hipLaunchKernelGGL(_kernelEamForce, dim3(B, 1), dim3(T, 1), 0, 0, deviceKeyFrom, deviceKeyTo, deviceDfFrom,
  //                         deviceDfTo, deviceDist2, deviceForces, len);
  //    }
  //  }

  HIP_CHECK(hipMemcpy(forces, deviceForces, len * sizeof(double), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(deviceKeyFrom));
  HIP_CHECK(hipFree(deviceKeyTo));
  HIP_CHECK(hipFree(deviceDfFrom));
  HIP_CHECK(hipFree(deviceDfTo));
  HIP_CHECK(hipFree(deviceDist2));
  HIP_CHECK(hipFree(deviceForces));
}

__global__ void _kernelEamChargeDensity(const atom_type::_type_prop_key *keys, const double *dist2, double *rhos,
                                        size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    rhos[i] = hip_pot::hipChargeDensity(keys[i], dist2[i]);
  }
}

void deviceEamChargeDensity(atom_type::_type_prop_key *keys, double *dist2, double *rhos, size_t len) {
  const size_t B = 512;
  const size_t T = 128;

  atom_type::_type_prop_key *deviceKeys;
  double *deviceDist2;
  double *deviceRhos; // output of kernel

  HIP_CHECK(hipMalloc((void **)&deviceKeys, len * sizeof(atom_type::_type_prop_key)));
  HIP_CHECK(hipMalloc((void **)&deviceDist2, len * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&deviceRhos, len * sizeof(double)));

  HIP_CHECK(hipMemcpy(deviceKeys, keys, len * sizeof(atom_type::_type_prop_key), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceDist2, dist2, len * sizeof(double), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(_kernelEamChargeDensity, dim3(B, 1), dim3(T, 1), 0, 0, deviceKeys, deviceDist2, deviceRhos, len);

  HIP_CHECK(hipMemcpy(rhos, deviceRhos, len * sizeof(double), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(deviceKeys));
  HIP_CHECK(hipFree(deviceDist2));
  HIP_CHECK(hipFree(deviceRhos));
}

__global__ void _kernelEamDEmbedEnergy(const atom_type::_type_prop_key *keys, const double *rhos, double *dfs,
                                       size_t len) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; // global thread id
  int threads = hipGridDim_x * hipBlockDim_x;              // total threads
  for (int i = id; i < len; i += threads) {
    dfs[i] = hip_pot::hipDEmbedEnergy(keys[i], rhos[i]);
  }
}

void deviceEamDEmbedEnergy(atom_type::_type_prop_key *keys, double *rhos, double *dfs, size_t len) {
  const size_t B = 512;
  const size_t T = 128;

  atom_type::_type_prop_key *deviceKeys;
  double *deviceRhos;
  double *deviceDfs; // output of kernel

  HIP_CHECK(hipMalloc((void **)&deviceKeys, len * sizeof(atom_type::_type_prop_key)));
  HIP_CHECK(hipMalloc((void **)&deviceRhos, len * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&deviceDfs, len * sizeof(double)));

  HIP_CHECK(hipMemcpy(deviceKeys, keys, len * sizeof(atom_type::_type_prop_key), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(deviceRhos, rhos, len * sizeof(double), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(_kernelEamDEmbedEnergy, dim3(B, 1), dim3(T, 1), 0, 0, deviceKeys, deviceRhos, deviceDfs, len);

  HIP_CHECK(hipMemcpy(dfs, deviceDfs, len * sizeof(double), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(deviceKeys));
  HIP_CHECK(hipFree(deviceRhos));
  HIP_CHECK(hipFree(deviceDfs));
}
