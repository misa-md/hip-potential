//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_DEVICE_H
#define HIP_POT_DEVICE_H

#include "hip_pot_types.h"
#include <hip/hip_runtime.h>


// fixme: dont use namespace here, until issue closed: https://github.com/ROCm-Developer-Tools/HIP/issues/1904
// namespace hip_pot {
  // potential data on device side
  // gloabl tables pointer, contains several kinds of tables.
  extern __device__ __constant__ hip_pot::_type_device_pot_spline **pot_tables;
  // spline data of electron charge density tables for multiple atomic elements on device.
  extern __device__ __constant__ hip_pot::_type_device_pot_spline **pot_table_ele_charge_density;
  // the similar as above, but it is for embedded energy tables.
  extern __device__ __constant__ hip_pot::_type_device_pot_spline **pot_table_embedded_energy;
  // the similar as above, but it is for pair potential tables.
  extern __device__ __constant__ hip_pot::_type_device_pot_spline **pot_origin_table_pair;

  // number of elements in potential
  extern __device__ __constant__ hip_pot::_type_device_table_size pot_eam_eles;

  // metadata of each potential table.
  extern __device__ __constant__ hip_pot::_type_device_pot_table_meta *pot_tables_metadata;
  // orgin data of electron charge density tables for multiple atomic elements on device.
  extern __device__ __constant__ hip_pot::_type_device_pot_table_meta *pot_ele_charge_table_metadata;
  // the similar as above, but it is for embedded energy tables.
  extern __device__ __constant__ hip_pot::_type_device_pot_table_meta *pot_embedded_energy_table_metadata;
  // the similar as above, but it is for pair potential tables.
  extern __device__ __constant__ hip_pot::_type_device_pot_table_meta *pot_pair_table_metadata;

// } // namespace hip_pot

#endif // HIP_POT_DEVICE_H
