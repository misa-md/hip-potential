//
// Created by genshen on 2022/6/21.
//

#ifndef HIP_POT_HIP_POT_EAM_SPLINE_HPP
#define HIP_POT_HIP_POT_EAM_SPLINE_HPP

#include "hip_pot.h"
#include "hip_pot_dev_eam_segmented_spline.h"
#include "hip_pot_dev_tables_compact.hpp"
#include "hip_pot_device_global_vars.h"

struct _device_spline_data {
  hip_pot::_type_device_pot_spline spline; // spline (it is a 1d array with length 7.)
  double p;                                // x value in ax^3+bx^2+cx+d for calculating interpolation result.

  __device__ _device_spline_data() = default;

  __device__ _device_spline_data(const hip_pot::_type_device_pot_table_item *data, const double _p) : p(_p) {
    spline[0] = data[0];
    spline[1] = data[1];
    spline[2] = data[2];
    spline[3] = data[3];
    spline[4] = data[4];
    spline[5] = data[5];
    spline[6] = data[6];
  }

  __device__ __forceinline__ double sp1() const {
    return ((spline[3] * p + spline[4]) * p + spline[5]) * p + spline[6];
  }
  __device__ __forceinline__ double sp2() const { return (spline[0] * p + spline[1]) * p + spline[2]; }
};

struct _device_spline_data_p1 {
  hip_pot::_type_device_pot_spline_1st_derivative spline;
  double p;

  __device__ _device_spline_data_p1() = default;

  __device__ _device_spline_data_p1(const hip_pot::_type_device_pot_table_item *data, const double _p) : p(_p) {
    spline[0] = data[0];
    spline[1] = data[1];
    spline[2] = data[2];
    spline[3] = data[3];
  }

  __device__ __forceinline__ double sp1() const {
    return ((spline[0] * p + spline[1]) * p + spline[2]) * p + spline[3];
  }
};

struct _device_spline_data_p2 {
  hip_pot::_type_device_pot_spline_2nd_derivative spline;

  __device__ _device_spline_data_p2() = default;

  __device__ _device_spline_data_p2(const hip_pot::_type_device_pot_table_item *data, const double _p) : p(_p) {
    spline[0] = data[0];
    spline[1] = data[1];
    spline[2] = data[2];
  }

  double p;
  __device__ __forceinline__ double sp2() const { return (spline[0] * p + spline[1]) * p + spline[2]; }
};

template <typename SPLINE_TYPE = hip_pot::_type_device_pot_spline, typename SPLINE_RESULT = _device_spline_data>
__device__ __forceinline__ SPLINE_RESULT findSpline(const hip_pot::_type_device_pot_table_item value,
                                                    SPLINE_TYPE *spline,
                                                    const hip_pot::_type_device_pot_table_meta meta) {
  double p = value * meta.inv_dx + 1.0;
  int m = static_cast<int>(p);
  m = fmax(1.0, fmin(static_cast<double>(m), static_cast<double>(meta.n - 1)));
  p -= m;
  p = fmin(p, 1.0);
  SPLINE_TYPE &sp = spline[m];
  return SPLINE_RESULT{&(sp[0]), p}; // fixme:
}

// find electron density spline
__device__ __forceinline__ _device_spline_data deviceRhoSpline(const atom_type::_type_prop_key key,
                                                               hip_pot::_type_device_pot_table_item v) {
  const hip_pot::_type_device_pot_table_meta meta = pot_ele_charge_table_metadata[key];
  const hip_pot::_type_device_pot_spline *spline = pot_table_ele_charge_density_by_key(meta.n, key);
  return findSpline(v, spline, meta);
}

struct RhoSplineLoader {
  __device__ RhoSplineLoader(const atom_type::_type_prop_key key, const hip_pot::_type_device_pot_table_item v) {
    const hip_pot::_type_device_pot_table_meta meta = pot_ele_charge_table_metadata[key];
    const hip_pot::_type_device_pot_spline *spline = pot_table_ele_charge_density_by_key(meta.n, key);
    spline_data = findSpline(v, spline, meta);
  }

  __device__ __forceinline__ double sp1() { return spline_data.sp1(); }
  __device__ __forceinline__ double sp2() { return spline_data.sp2(); }

private:
  _device_spline_data spline_data;
};

// find embed energy spline
__device__ __forceinline__ _device_spline_data deviceEmbedSpline(const atom_type::_type_prop_key key,
                                                                 hip_pot::_type_device_pot_table_item v) {
  const hip_pot::_type_device_pot_table_meta meta = pot_embedded_energy_table_metadata[key];
  const hip_pot::_type_device_pot_spline *spline = pot_table_embedded_energy_by_key(meta.n, key);
  return findSpline(v, spline, meta);
}

struct EmbedSplineLoader {
  __device__ EmbedSplineLoader(const atom_type::_type_prop_key key, const hip_pot::_type_device_pot_table_item v) {
    const hip_pot::_type_device_pot_table_meta meta = pot_embedded_energy_table_metadata[key];
    const hip_pot::_type_device_pot_spline *spline = pot_table_embedded_energy_by_key(meta.n, key);
    spline_data = findSpline(v, spline, meta);
  }

  __device__ __forceinline__ double sp1() { return spline_data.sp1(); }
  __device__ __forceinline__ double sp2() { return spline_data.sp2(); }

private:
  _device_spline_data spline_data;
};

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
  const hip_pot::_type_device_table_size index = pot_eam_eles * key_from - (key_from + 1) * key_from / 2 + key_to;
  const hip_pot::_type_device_pot_table_meta meta = pot_pair_table_metadata[index];
  const hip_pot::_type_device_pot_spline *spline = pot_table_pair_by_key(meta.n, index);
  return findSpline(v, spline, meta);
}

// find pair potential spline and calculate interpolation.
struct PairSplineLoader {
  __device__ PairSplineLoader(atom_type::_type_prop_key key_from, atom_type::_type_prop_key key_to,
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
    spline_data = findSpline(v, spline, meta);
  }

  __device__ __forceinline__ double sp1() { return spline_data.sp1(); }

  __device__ __forceinline__ double sp2() { return spline_data.sp2(); }

private:
  _device_spline_data spline_data;
};

// find electron density spline (but using segmented table)
struct RhoSegmentedSplineLoader {
  __device__ RhoSegmentedSplineLoader(const atom_type::_type_prop_key key,
                                      const hip_pot::_type_device_pot_table_item _v)
      : v(_v) {
    meta = pot_ele_charge_table_metadata[key];
    this->key = key;
  }

  __device__ __forceinline__ double sp1() {
    hip_pot::_type_device_pot_spline_1st_derivative *spline = posite_spline_by_key<TAG_RHO_1ST>(meta.n, key);
    const _device_spline_data_p1 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_1st_derivative, _device_spline_data_p1>(v, spline, meta);
    return spline_data.sp1();
  }
  __device__ __forceinline__ double sp2() {
    hip_pot::_type_device_pot_spline_2nd_derivative *spline = posite_spline_by_key2<TAG_RHO_2ND>(meta.n, key);
    const _device_spline_data_p2 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_2nd_derivative, _device_spline_data_p2>(v, spline, meta);
    return spline_data.sp2();
  }

private:
  const hip_pot::_type_device_pot_table_item v;
  hip_pot::_type_device_pot_table_meta meta;
  hip_pot::_type_device_table_size key;
};

// find embed energy spline (but using segmented table)
struct EmbedSegmentedSplineLoader {
  __device__ EmbedSegmentedSplineLoader(const atom_type::_type_prop_key key,
                                        const hip_pot::_type_device_pot_table_item _v)
      : v(_v) {
    meta = pot_embedded_energy_table_metadata[key];
    this->key = key;
  }

  __device__ __forceinline__ double sp1() {
    hip_pot::_type_device_pot_spline_1st_derivative *spline = posite_spline_by_key<TAG_DF_1ST>(meta.n, key);
    const _device_spline_data_p1 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_1st_derivative, _device_spline_data_p1>(v, spline, meta);
    return spline_data.sp1();
  }
  __device__ __forceinline__ double sp2() {
    hip_pot::_type_device_pot_spline_2nd_derivative *spline = posite_spline_by_key2<TAG_DF_2ND>(meta.n, key);
    const _device_spline_data_p2 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_2nd_derivative, _device_spline_data_p2>(v, spline, meta);
    return spline_data.sp2();
  }

private:
  const hip_pot::_type_device_pot_table_item v;
  hip_pot::_type_device_pot_table_meta meta;
  hip_pot::_type_device_table_size key;
};

// find pair potential spline (but using segmented table) and calculate interpolation.
struct PairSegmentedSplineLoader {
  __device__ PairSegmentedSplineLoader(atom_type::_type_prop_key key_from, atom_type::_type_prop_key key_to,
                                       hip_pot::_type_device_pot_table_item _v)
      : v(_v) {
    if (key_from > key_to) {
      // swap from and to.
      atom_type::_type_prop_key temp = key_from;
      key_from = key_to;
      key_to = temp;
    }
    index = pot_eam_eles * key_from - (key_from + 1) * key_from / 2 + key_to;
    meta = pot_pair_table_metadata[index];
  }
  __device__ __forceinline__ double sp1() {
    hip_pot::_type_device_pot_spline_1st_derivative *spline_table = posite_spline_by_key<TAG_PAIR_1ST>(meta.n, index);
    const _device_spline_data_p1 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_1st_derivative, _device_spline_data_p1>(v, spline_table, meta);
    return spline_data.sp1();
  }

  __device__ __forceinline__ double sp2() {
    hip_pot::_type_device_pot_spline_2nd_derivative *spline_table = posite_spline_by_key2<TAG_PAIR_2ND>(meta.n, index);
    const _device_spline_data_p2 spline_data =
        findSpline<hip_pot::_type_device_pot_spline_2nd_derivative, _device_spline_data_p2>(v, spline_table, meta);
    return spline_data.sp2();
  }

private:
  const hip_pot::_type_device_pot_table_item v;
  hip_pot::_type_device_pot_table_meta meta;
  hip_pot::_type_device_table_size index;
};

#endif // HIP_POT_HIP_POT_EAM_SPLINE_HPP
