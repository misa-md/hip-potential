//
// Created by genshen on 2022-05-16
//

#ifndef HIP_POT_DEV_TABLES_COMPACT_HPP
#define HIP_POT_DEV_TABLES_COMPACT_HPP

#include <cassert>
#include <hip/hip_runtime.h>

#include <eam.h>

#include "hip_pot_device_global_vars.h"
#include "hip_pot_types.h"
#include "pot_building_config.h"

#ifndef HIP_POT_COMPACT_MEM_LAYOUT

__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_ele_charge_density_by_key(const int ele_count, const atom_type::_type_prop_key key) {
  return pot_table_ele_charge_density[key];
}

__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_embedded_energy_by_key(const int single_tab_size, const atom_type::_type_prop_key key) {
  return pot_table_embedded_energy[key];
}

__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_pair_by_key(const int single_tab_size, const hip_pot::_type_device_table_size index) {
  return pot_table_pair[index];
}

#endif // HIP_POT_COMPACT_MEM_LAYOUT

#ifdef HIP_POT_COMPACT_MEM_LAYOUT

// single_tab_size: elements count in a single table.
__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_ele_charge_density_by_key(const int single_tab_size, const atom_type::_type_prop_key key) {
  return pot_table_ele_charge_density + (single_tab_size + 1) * key;
}

__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_embedded_energy_by_key(const int single_tab_size, const atom_type::_type_prop_key key) {
  return pot_table_embedded_energy + (single_tab_size + 1) * key;
}

__device__ __forceinline__ hip_pot::_type_device_pot_spline *
pot_table_pair_by_key(const int single_tab_size, const hip_pot::_type_device_table_size index) {
  return pot_table_pair + (single_tab_size + 1) * index;
}

#endif // HIP_POT_COMPACT_MEM_LAYOUT

#endif // HIP_POT_DEV_TABLES_COMPACT_HPP