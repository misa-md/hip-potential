//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_H
#define HIP_POT_H

#include "hip_pot_types.h"
#include <eam.h>

namespace hip_pot {
  typedef struct {
    _type_device_pot_table_meta *ptr_device_pot_meta;
    _type_device_pot_spline **ptr_device_pot_tables;
    atom_type::_type_atom_types n_eles; // elements in potential tables.
  } _type_device_pot;

  /**
   * copy potential data after splining to devices
   * \param pot: origin potential data in CPU side.
   * \param _pot_types: types of atoms.
   */
  _type_device_pot potCopyHostToDevice(eam *pot, std::vector<atom_type::_type_atomic_no> _pot_types);

  /**
   * remove potential data on devices.
   * \param device_pot pointer of data on devices.
   */
  void destroyDevicePotTables(_type_device_pot device_pot);

} // namespace hip_pot

#endif // HIP_POT_H
