//
// Created by genshen on 2020-05-14
//

#include "hip_eam_device.h"
#include "hip_pot.h"
#include "hip_pot_dev_tables_compact.hpp"
#include "hip_pot_device_global_vars.h"
#include "hip_pot_macros.h"

__device__ HIP_POT_INLINE double hip_pot::hipToForce(const atom_type::_type_prop_key key_from,
                                                     const atom_type::_type_prop_key key_to, const double dist2,
                                                     const double df_from, const double df_to) {
  double fpair;
  double phi, phip, psip, z2, z2p;

  const double r = sqrt(dist2);
  const _device_spline_data phi_s = devicePhiSplineByType(key_from, key_to, r);
  const _device_spline_data ele_from_s = deviceRhoSpline(key_from, key_to, r);
  const _device_spline_data ele_to_s = deviceRhoSpline(key_to, key_from, r);

  z2 = ((phi_s.spline[3] * phi_s.p + phi_s.spline[4]) * phi_s.p + phi_s.spline[5]) * phi_s.p + phi_s.spline[6];
  // z2 = phi*r
  z2p = (phi_s.spline[0] * phi_s.p + phi_s.spline[1]) * phi_s.p + phi_s.spline[2];
  // z2p = (phi * r)' = (phi' * r) + phi

  const double rho_p_from =
      (ele_from_s.spline[0] * ele_from_s.p + ele_from_s.spline[1]) * ele_from_s.p + ele_from_s.spline[2];
  const double rho_p_to = (ele_to_s.spline[0] * ele_to_s.p + ele_to_s.spline[1]) * ele_to_s.p + ele_to_s.spline[2];

  const double recip = 1.0 / r;
  phi = z2 * recip;                 // pair potential energy
  phip = z2p * recip - phi * recip; // phip = phi' = (z2p - phi)/r
  psip = df_from * rho_p_to + df_to * rho_p_from + phip;
  fpair = -psip * recip;

  return fpair;
}

/**
 * special optimization for force calculation if "from type" and "to type" are the same.
 */
template <typename T>
__device__ HIP_POT_INLINE T hip_pot::hipToForceSingleType(const atom_type::_type_prop_key key, const T dist2,
                                                          const T df_from, const T df_to) {
  const T r = sqrt(dist2);
  const _device_spline_data phi_s = devicePhiSplineByType(key, key, r);
  const _device_spline_data ele_s = deviceRhoSpline(key, key, r);

  const T z2 = ((phi_s.spline[3] * phi_s.p + phi_s.spline[4]) * phi_s.p + phi_s.spline[5]) * phi_s.p + phi_s.spline[6];
  // z2 = phi*r
  const T z2p = (phi_s.spline[0] * phi_s.p + phi_s.spline[1]) * phi_s.p + phi_s.spline[2];
  // z2p = (phi * r)' = (phi' * r) + phi

  const T rho_p = (ele_s.spline[0] * ele_s.p + ele_s.spline[1]) * ele_s.p + ele_s.spline[2];

  const T recip = 1.0 / r;
  const T phi = z2 * recip;                 // pair potential energy
  const T phip = z2p * recip - phi * recip; // phip = phi' = (z2p - phi)/r

  const T psip = (df_from + df_to) * rho_p + phip;
  const T fpair = -psip * recip;

  return fpair;
}

/**
 * select hipToForce api between single type and double type version.
 * \tparam T type of float point number: float or double.
 * \tparam ALLOW_WARP_DIVERGENCE In CUDA, warp divergence may cause poor performance.
 * By setting ALLOW_WARP_DIVERGENCE to false can avoid this side effect,
 * if there are atom pairs having the same types and other atom pairs having different types in a warp.
 */
template <typename T, bool ALLOW_WARP_DIVERGENCE>
__device__ HIP_POT_INLINE T hip_pot::hipToForceAdaptive(const atom_type::_type_prop_key key_from,
                                                        const atom_type::_type_prop_key key_to, const T dist2,
                                                        const T df_from, const T df_to) {
  if (ALLOW_WARP_DIVERGENCE && key_from == key_to) {
    return hipToForceSingleType<T>(key_from, dist2, df_from, df_to);
  } else {
    return hip_pot::hipToForce(key_from, key_to, dist2, df_from, df_to);
  }
}

__device__ HIP_POT_INLINE double hip_pot::hipChargeDensity(const atom_type::_type_prop_key _atom_key_me,
                                                           const atom_type::_type_prop_key _atom_key_nei, const double dist2) {
  const double r = sqrt(dist2);
  const _device_spline_data s = deviceRhoSpline(_atom_key_me, _atom_key_nei, r);
  return ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6];
}

__device__ HIP_POT_INLINE double hip_pot::hipDEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho) {
  const _device_spline_data s = deviceEmbedSpline(_atom_key, rho);
  return (s.spline[0] * s.p + s.spline[1]) * s.p + s.spline[2];
}

__device__ HIP_POT_INLINE double hip_pot::hipEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho) {
  const _device_spline_data s = deviceEmbedSpline(_atom_key, rho);
  return ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6];
}

__device__ HIP_POT_INLINE double hip_pot::hipPairPotential(const atom_type::_type_prop_key key_from,
                                                           const atom_type::_type_prop_key key_to, const double dist2) {
  const double r = sqrt(dist2);
  const _device_spline_data s = devicePhiSplineByType(key_from, key_to, r);
  const double phi_r = ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6]; // pair_pot * r
  return phi_r / r;
}

// make instance of template functions:
template __device__ HIP_POT_INLINE double hip_pot::hipToForceSingleType<double>(const atom_type::_type_prop_key key,
                                                                                const double dist2,
                                                                                const double df_from,
                                                                                const double df_to);

template __device__ HIP_POT_INLINE double
hip_pot::hipToForceAdaptive<double, false>(const atom_type::_type_prop_key key_from,
                                           const atom_type::_type_prop_key key_to, const double dist2,
                                           const double df_from, const double df_to);
template __device__ HIP_POT_INLINE double
hip_pot::hipToForceAdaptive<double, true>(const atom_type::_type_prop_key key_from,
                                          const atom_type::_type_prop_key key_to, const double dist2,
                                          const double df_from, const double df_to);
