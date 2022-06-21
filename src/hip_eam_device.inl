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
  const _device_spline_data ele_from_s = deviceRhoSpline(key_from, r);
  const _device_spline_data ele_to_s = deviceRhoSpline(key_to, r);

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

__device__ HIP_POT_INLINE double hip_pot::hipChargeDensity(const atom_type::_type_prop_key _atom_key,
                                                           const double dist2) {
  const double r = sqrt(dist2);
  const _device_spline_data s = deviceRhoSpline(_atom_key, r);
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
