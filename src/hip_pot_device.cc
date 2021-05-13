//
// Created by genshen on 2020-05-14
//

#include "hip_pot_device.h"
#include <hip/hip_runtime.h>

// namespace hip_pot {
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_spline **pot_tables = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_spline **pot_table_ele_charge_density = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_spline **pot_table_embedded_energy = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_spline **pot_table_pair = nullptr;

__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_table_size pot_eam_eles = 0;

__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_tables_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_ele_charge_table_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_embedded_energy_table_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_pair_table_metadata = nullptr;
// } // namespace hip_pot

// As part of optimization,
// if compiler will see global device variables is not used anywhere it would delete it.
// We just use the variables here.
// This kernel function will not never be called on host side.
__global__ void _ref_device_variables() {
  pot_eam_eles = 1;

  pot_tables = nullptr;
  pot_table_ele_charge_density = nullptr;
  pot_table_embedded_energy = nullptr;
  pot_table_pair = nullptr;

  pot_tables_metadata = nullptr;
  pot_ele_charge_table_metadata = nullptr;
  pot_embedded_energy_table_metadata = nullptr;
  pot_pair_table_metadata = nullptr;
}
