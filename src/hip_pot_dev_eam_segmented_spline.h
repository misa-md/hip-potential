//
// Created by genshen on 2022/8/26.
//

#ifndef HIP_POT_HIP_EAM_DEVICE_SEGMENT_H
#define HIP_POT_HIP_EAM_DEVICE_SEGMENT_H

#include <hip/hip_runtime.h>

#include "hip_pot_macros.h"
#include "hip_pot_types.h"

// SoA memory layout and compact memory layout for potential table.

namespace hip_pot {
  typedef _type_device_pot_table_item _type_device_pot_spline_1st_derivative[4];
  typedef _type_device_pot_table_item _type_device_pot_spline_2nd_derivative[3];

  // collection of the 1st derivative after spline under compact memory layout mode.
  typedef _type_device_pot_spline_1st_derivative *_type_1st_spline_coll;
  // collection of the 2nd derivative after spline under compact memory layout mode.
  typedef _type_device_pot_spline_2nd_derivative *_type_2nd_spline_coll;
} // namespace hip_pot

extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_ele_charge_density_1st;
extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_embedded_energy_1st;
extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_pair_1st;

extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_ele_charge_density_2nd;
extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_embedded_energy_2nd;
extern __device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_pair_2nd;

constexpr int TAG_RHO_1ST = 0;
constexpr int TAG_RHO_2ND = 1;
constexpr int TAG_DF_1ST = 2;
constexpr int TAG_DF_2ND = 3;
constexpr int TAG_PAIR_1ST = 4;
constexpr int TAG_PAIR_2ND = 5;

#ifdef HIP_POT_COMPACT_MEM_LAYOUT
template <int TYPE>
__device__ __forceinline__ hip_pot::_type_1st_spline_coll posite_spline_by_key(const int single_table_size,
                                                                               const int key) {
  if (TYPE == TAG_RHO_1ST) {
    return pot_table_ele_charge_density_1st + (single_table_size + 1) * key;
  }
  if (TYPE == TAG_DF_1ST) {
    return pot_table_embedded_energy_1st + (single_table_size + 1) * key;
  }
  if (TYPE == TAG_PAIR_1ST) {
    return pot_table_pair_1st + (single_table_size + 1) * key;
  }
}

template <int TYPE>
__device__ __forceinline__ hip_pot::_type_2nd_spline_coll posite_spline_by_key2(const int single_table_size, int key) {
  if (TYPE == TAG_RHO_2ND) {
    return pot_table_ele_charge_density_2nd + (single_table_size + 1) * key;
  }
  if (TYPE == TAG_DF_2ND) {
    return pot_table_embedded_energy_2nd + (single_table_size + 1) * key;
  }
  if (TYPE == TAG_PAIR_2ND) {
    return pot_table_pair_2nd + (single_table_size + 1) * key;
  }
}

#endif // HIP_POT_COMPACT_MEM_LAYOUT

void copy_pot_spline_segmented_wrapper(const hip_pot::_type_device_table_size n_eles,
                                       const hip_pot::_type_device_pot_table_meta *device_meta,
                                       const hip_pot::_type_device_pot_table_meta *host_meta,
                                       hip_pot::_type_spline_colle pot_table_ele_charge_density,
                                       hip_pot::_type_spline_colle pot_table_embedded_energy,
                                       hip_pot::_type_spline_colle pot_table_pair);

#endif // HIP_POT_HIP_EAM_DEVICE_SEGMENT_H
