//
// Created by genshen on 2020-05-14
//

#include "hip_eam_device.h"
#include "hip_pot.h"
#include "hip_pot_device.h"

struct _device_spline_data {
  const hip_pot::_type_device_pot_element *spline; // spline (it is a 1d array with length 7.)
  const double p;                                  // x value in ax^3+bx^2+cx+d for calculating interpolation result.
};

inline __device__ _device_spline_data findSpline(const hip_pot::_type_device_pot_element value,
                                                 const hip_pot::_type_device_pot_spline *spline,
                                                 const hip_pot::_type_device_pot_table_meta meta) {
  double p = value * meta.inv_dx + 1.0;
  int m = static_cast<int>(p);
  m = max(1, std::min(m, (meta.n - 1)));
  p -= m;
  p = min(p, 1.0);
  return _device_spline_data{&(spline[m][0]), p};
}

// find electron density spline
inline __device__ _device_spline_data deviceRhoSpline(const atom_type::_type_prop_key key,
                                                      hip_pot::_type_device_pot_element v) {
  return findSpline(v, pot_table_ele_charge_density[key], pot_ele_charge_table_metadata[key]);
}

// find embed energy spline
inline __device__ _device_spline_data deviceEmbedSpline(const atom_type::_type_prop_key key,
                                                        hip_pot::_type_device_pot_element v) {
  return findSpline(v, pot_table_embedded_energy[key], pot_embedded_energy_table_metadata[key]);
}

// find pair potential spilne.
inline __device__ _device_spline_data devicePhiSplineByType(atom_type::_type_prop_key key_from,
                                                            atom_type::_type_prop_key key_to,
                                                            hip_pot::_type_device_pot_element v) {
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
  const hip_pot::_type_device_pot_spline *spline = pot_table_pair[index];
  const hip_pot::_type_device_pot_table_meta meta = pot_pair_table_metadata[index];
  return findSpline(v, spline, meta);
}

__device__ double hip_pot::hipToForce(const atom_type::_type_prop_key key_from, const atom_type::_type_prop_key key_to,
                                      const double dist2, const double df_from, const double df_to) {
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

__device__ double hip_pot::hipChargeDensity(const atom_type::_type_prop_key _atom_key, const double dist2) {
  const double r = sqrt(dist2);
  const _device_spline_data s = deviceRhoSpline(_atom_key, r);
  return ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6];
}

__device__ double hip_pot::hipDEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho) {
  const _device_spline_data s = deviceEmbedSpline(_atom_key, rho);
  return (s.spline[0] * s.p + s.spline[1]) * s.p + s.spline[2];
}

__device__ double hip_pot::hipEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho) {
  const _device_spline_data s = deviceEmbedSpline(_atom_key, rho);
  return ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6];
}

__device__ double hip_pot::hipPairPotential(const atom_type::_type_prop_key key_from,
                                            const atom_type::_type_prop_key key_to, const double dist2) {
  const double r = sqrt(dist2);
  const _device_spline_data s = devicePhiSplineByType(key_from, key_to, r);
  const double phi_r = ((s.spline[3] * s.p + s.spline[4]) * s.p + s.spline[5]) * s.p + s.spline[6]; // pair_pot * r
  return phi_r / r;
}
