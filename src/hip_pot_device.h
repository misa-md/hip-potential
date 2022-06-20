//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_DEVICE_H
#define HIP_POT_DEVICE_H

#include <hip/hip_runtime.h>

#include "pot_building_config.h"
#include "hip_pot_types.h"

#if defined POT_NV_PLATFORM
#define __DEVICE_CONSTANT__ // ignore constant
#else
#define __DEVICE_CONSTANT__ __constant__
#endif

// fixme: dont use namespace here, until issue closed: https://github.com/ROCm-Developer-Tools/HIP/issues/1904
// namespace hip_pot {
  // potential data on device side
  // gloabl tables pointer, contains several kinds of tables.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_tables;
  // spline data of electron charge density tables for multiple atomic elements on device.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_ele_charge_density;
  // the similar as above, but it is for embedded energy tables.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_embedded_energy;
  // the similar as above, but it is for pair potential tables.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_spline_colle pot_table_pair;

  // number of elements in potential
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_device_table_size pot_eam_eles;

  // metadata of each potential table.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_tables_metadata;
  // orgin data of electron charge density tables for multiple atomic elements on device.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_ele_charge_table_metadata;
  // the similar as above, but it is for embedded energy tables.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_embedded_energy_table_metadata;
  // the similar as above, but it is for pair potential tables.
  extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_device_pot_table_meta *pot_pair_table_metadata;

  /**
   * set global variables via `hipMemcpyToSymbol` here, including elements size, metadata and spline data.
   * \param n_eam_elements number of elements on eam potential system
   * \param metadata_ptr metadata of the eam potential data and spine.
   * \param spline_ptr spline data of eam potential. It must be a device array in incompact mode,
   *   it must be a host array in compact mode.
   */
  void set_device_variables(const hip_pot::_type_device_table_size n_eam_elements,
                            hip_pot::_type_device_pot_table_meta *metadata_ptr,
                            hip_pot::_type_device_pot_spline **spline_ptr);

// } // namespace hip_pot

#endif // HIP_POT_DEVICE_H
