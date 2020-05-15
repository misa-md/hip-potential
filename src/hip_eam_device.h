//
// Created by genshen on 2020-05-14
//

#ifndef HIP_POT_HIP_EAM_DEVICE_H
#define HIP_POT_HIP_EAM_DEVICE_H

#include "hip_pot.h"
#include "hip_pot_types.h"
#include <hip/hip_runtime.h>

namespace hip_pot {
  /**
   * calculate force comtribution from atom j to atom i.
   * \param key_from the type of atom j.
   * \param key_to the type of atom i.
   * \param dist2 distance^2 of the two atoms.
   * \param df_from derivative of embedded energy of atom j.
   * \param df_to derivative of embedded energy of atom i.
   */
  __device__ double hipToForce(const atom_type::_type_prop_key key_from, const atom_type::_type_prop_key key_to,
                               const double dist2, const double df_from, const double df_to);
  /**
   * compute the contribution to electron charge density from atom j of type {@var _atom_key} at location of one atom i.
   * whose distance is specified by {@var dist2}
   * @param _atom_key atom type of atom j.
   * @param dist2 the square of the distance between atom i and atom j.
   * @return the contribution to electron charge density from atom j.
   */
  __device__ double hipChargeDensity(const atom_type::_type_prop_key _atom_key, const double dist2);

  /**
   * compute derivative of embedded energy of atom of type {@var _atom_type},
   * whose electron charge density contributed by its neighbor atoms is specified by {@var rho}.
   * @param _atom_key atom type
   * @param rho  electron charge density contributed by all its neighbor atoms.
   * @return derivative of embedded energy of this atom.
   */
  __device__ double hipDEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho);

  /**
   * compute embedded energy of atom of type {@var _atom_type}
   * @param _atom_key n atom type
   * @param rho electron charge density contributed by all its neighbor atoms.
   * @return embedded energy of this atom.
   */
  __device__ double hipEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho);

  /**
   * pair potential energy.
   * @return pair potential energy.
   */
  __device__ double hipPairPotential(const atom_type::_type_prop_key key_from, const atom_type::_type_prop_key key_to,
                                     const double dist2);

} // namespace hip_pot

#endif // HIP_POT_HIP_EAM_DEVICE_H