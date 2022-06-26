//
// Created by genshen on 2022/6/21.
//

#ifndef HIP_POT_HIP_POT_EAM_SPLINE_HPP
#define HIP_POT_HIP_POT_EAM_SPLINE_HPP

#include "hip_pot.h"
#include "hip_pot_dev_tables_compact.hpp"
#include "hip_pot_device_global_vars.h"

struct _device_spline_data {
  const hip_pot::_type_device_pot_table_item *spline; // spline (it is a 1d array with length 7.)
  const double p;                                     // x value in ax^3+bx^2+cx+d for calculating interpolation result.
};

__device__ __forceinline__ _device_spline_data findSpline(const hip_pot::_type_device_pot_table_item value,
                                                          const hip_pot::_type_device_pot_spline *spline,
                                                          const hip_pot::_type_device_pot_table_meta meta) {
  double p = value * meta.inv_dx + 1.0;
  int m = static_cast<int>(p);
  m = fmax(1.0, fmin(static_cast<double>(m), static_cast<double>(meta.n - 1)));
  p -= m;
  p = fmin(p, 1.0);
  return _device_spline_data{&(spline[m][0]), p};
}

// find electron density spline
__device__ __forceinline__ _device_spline_data deviceRhoSpline(const atom_type::_type_prop_key key,
                                                               hip_pot::_type_device_pot_table_item v) {
  const hip_pot::_type_device_pot_table_meta meta = pot_ele_charge_table_metadata[key];
  const hip_pot::_type_device_pot_spline *spline = pot_table_ele_charge_density_by_key(meta.n, key);
  return findSpline(v, spline, meta);
}

// find embed energy spline
__device__ __forceinline__ _device_spline_data deviceEmbedSpline(const atom_type::_type_prop_key key,
                                                                 hip_pot::_type_device_pot_table_item v) {
  const hip_pot::_type_device_pot_table_meta meta = pot_embedded_energy_table_metadata[key];
  const hip_pot::_type_device_pot_spline *spline = pot_table_embedded_energy_by_key(meta.n, key);
  return findSpline(v, spline, meta);
}

// find pair potential spline.
__device__ __forceinline__ _device_spline_data devicePhiSplineByType(atom_type::_type_prop_key key_from,
                                                                     atom_type::_type_prop_key key_to,
                                                                     hip_pot::_type_device_pot_table_item v) {
  if (key_from > key_to) {
    // swap from and to.
    atom_type::_type_prop_key temp = key_from;
    key_from = key_to;
    key_to = temp;
  }
  // for example:
  // FeFe  FeCu  FeNi  CuCu  CuNi  NiNi
  // (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)
  //   0,    1,   2,    3,    4,    5
  // for (i,j) pair, we have: index = [N + (N-1) + (N-i+1)] + (j-i+1) - 1 = Ni-(i+1)*i/2+j
  const hip_pot::_type_device_table_size index = pot_eam_eles * key_from - (key_from + 1) * key_from / 2 + key_to;
  const hip_pot::_type_device_pot_table_meta meta = pot_pair_table_metadata[index];
  const hip_pot::_type_device_pot_spline *spline = pot_table_pair_by_key(meta.n, index);
  return findSpline(v, spline, meta);
}

#endif // HIP_POT_HIP_POT_EAM_SPLINE_HPP
