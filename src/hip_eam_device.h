//
// Created by genshen on 2020-05-14
//

#ifndef HIP_POT_HIP_EAM_DEVICE_H
#define HIP_POT_HIP_EAM_DEVICE_H

#include <hip/hip_runtime.h>

#include "hip_pot.h"
#include "hip_pot_dev_eam_spline.hpp"
#include "hip_pot_macros.h"
#include "hip_pot_types.h"

namespace hip_pot {
  /**
   * calculate force contribution from atom j to atom i.
   * \param key_from the type of atom j.
   * \param key_to the type of atom i.
   * \param dist2 distance^2 of the two atoms.
   * \param df_from derivative of embedded energy of atom j.
   * \param df_to derivative of embedded energy of atom i.
   */
  template <typename PAIR_LOADER = PairSplineLoader, typename RHO_LOADER = RhoSplineLoader>
  __device__ HIP_POT_INLINE double hipToForce(const atom_type::_type_prop_key key_from,
                                              const atom_type::_type_prop_key key_to, const double dist2,
                                              const double df_from, const double df_to);

  /**
   * Eam force computation api for single atom type.
   * What is "single atom type": types of i and j is the same when calculating force contribution from atom j to atom i.
   * In this case, the api implementation can be simplified for better performance.
   * \tparam T type of float point number. (currently, it can only be double).
   * \param key the type of atom i and j.
   * \param dist2 distance^2 of the two atoms.
   * \param df_from derivative of embedded energy of atom j.
   * \param df_to derivative of embedded energy of atom i.
   */
  template <typename T, typename PAIR_LOADER = PairSplineLoader, typename RHO_LOADER = RhoSplineLoader>
  __device__ HIP_POT_INLINE T hipToForceSingleType(const atom_type::_type_prop_key key, const T dist2, const T df_from,
                                                   const T df_to);

  /**
   * Eam force computation api for both normal atom types and single atom type.
   * In its implementation, it will call api `hipToForceSingleType` if key_from and key_to are the same
   * (single atom type).
   * Otherwise, the api `hipToForce` will be called (normal atom types case).
   * \tparam T type of float point number. (currently, it can only be double).
   * \tparam ALLOW_WARP_DIVERGENCE In CUDA, warp divergence may cause poor performance.
   * By setting ALLOW_WARP_DIVERGENCE to false can avoid this side effect,
   * if there are atom pairs having the same types and other atom pairs having different types in a warp.
   */
  template <typename T, typename PAIR_LOADER = PairSplineLoader, typename RHO_LOADER = RhoSplineLoader,
            bool ALLOW_WARP_DIVERGENCE = true>
  __device__ HIP_POT_INLINE T hipToForceAdaptive(const atom_type::_type_prop_key key_from,
                                                 const atom_type::_type_prop_key key_to, const T dist2, const T df_from,
                                                 const T df_to);

  /**
   * compute the contribution to electron charge density from atom j of type {@var _atom_key} at location of one
   * atom i. whose distance is specified by {@var dist2}
   * @param _atom_key atom type of atom j.
   * @param dist2 the square of the distance between atom i and atom j.
   * @return the contribution to electron charge density from atom j.
   */
  __device__ HIP_POT_INLINE double hipChargeDensity(const atom_type::_type_prop_key _atom_key, const double dist2);

  /**
   * compute derivative of embedded energy of atom of type {@var _atom_type},
   * whose electron charge density contributed by its neighbor atoms is specified by {@var rho}.
   * @param _atom_key atom type
   * @param rho  electron charge density contributed by all its neighbor atoms.
   * @return derivative of embedded energy of this atom.
   */
  __device__ HIP_POT_INLINE double hipDEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho);

  /**
   * compute embedded energy of atom of type {@var _atom_type}
   * @param _atom_key n atom type
   * @param rho electron charge density contributed by all its neighbor atoms.
   * @return embedded energy of this atom.
   */
  __device__ HIP_POT_INLINE double hipEmbedEnergy(const atom_type::_type_prop_key _atom_key, const double rho);

  /**
   * pair potential energy.
   * @return pair potential energy.
   */
  __device__ HIP_POT_INLINE double hipPairPotential(const atom_type::_type_prop_key key_from,
                                                    const atom_type::_type_prop_key key_to, const double dist2);

} // namespace hip_pot

#ifdef HIP_POT_DEVICE_API_INLINE
#include "hip_eam_device.inl"
#endif

#endif // HIP_POT_HIP_EAM_DEVICE_H
