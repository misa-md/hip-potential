//
// Created by genshen on 2020-05-14
//

#include <hip/hip_runtime.h>

#include "hip_macros.h"
#include "hip_pot_device.h"
#include "hip_pot_dev_tables_compact.hpp"

// namespace hip_pot {

__device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_tables = nullptr;
// array of spline of each alloy.
__device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_ele_charge_density = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_embedded_energy = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_pair = nullptr;

__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_table_size pot_eam_eles = 0;

__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_tables_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_ele_charge_table_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_embedded_energy_table_metadata = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_pair_table_metadata = nullptr;
// } // namespace hip_pot

void set_device_variables(const hip_pot::_type_device_table_size n_eam_elements,
                          hip_pot::_type_device_pot_table_meta *metadata_ptr,
                          hip_pot::_type_device_pot_spline **spline_ptr) {
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_eam_eles), &(n_eam_elements), sizeof(hip_pot::_type_device_table_size)));

  hip_pot::_type_device_pot_table_meta *dev_meta_ptr = metadata_ptr;
  const size_t meta_ptr_size = sizeof(hip_pot::_type_device_pot_table_meta *);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_tables_metadata), &(dev_meta_ptr), meta_ptr_size));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_ele_charge_table_metadata), &(dev_meta_ptr), meta_ptr_size));
  dev_meta_ptr += n_eam_elements;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_embedded_energy_table_metadata), &(dev_meta_ptr), meta_ptr_size));
  dev_meta_ptr += n_eam_elements;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_pair_table_metadata), &(dev_meta_ptr), meta_ptr_size));

  hip_pot::_type_device_pot_spline **dev_spline_ptr = spline_ptr;
  const size_t spline_ptr_size = sizeof(hip_pot::_type_device_pot_spline **);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_tables), &(dev_spline_ptr), spline_ptr_size));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_ele_charge_density), &(dev_spline_ptr), spline_ptr_size));
  dev_spline_ptr += n_eam_elements;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_embedded_energy), &(dev_spline_ptr), spline_ptr_size));
  dev_spline_ptr += n_eam_elements;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_pair), &(dev_spline_ptr), spline_ptr_size));
}

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
