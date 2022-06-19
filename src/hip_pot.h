//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_H
#define HIP_POT_H

#include "hip_pot_types.h"
#include <eam.h>

namespace hip_pot {
  typedef struct {
    // device array of all metadata of all elements type and table type.
    _type_device_pot_table_meta *ptr_device_pot_meta;
    // it is device data and points to ptr_device_pot_data array which is categoried by elements type and table type
    _type_device_pot_spline **ptr_device_pot_tables;
    _type_device_pot_spline *device_pot_data; // pointer to device data
    atom_type::_type_atom_types n_eles;       // elements in potential tables.
  } _type_device_pot;

  /**
   * calculate and copy metadata of each potential table to device side.
   *
   * \param pot origin potential data in CPU side.
   * \param _pot_types list of atoms types.
   * \param p_tables_metadata the array to store metadata on host side.
   * \param p_device_tables_metadata the array to store metadata on device side.
   * \param tables number of potential tables.
   * \param n_eles number of element types in potential tables.
   */
  void potCopyMetadata(eam *_pot, std::vector<atom_type::_type_atomic_no> _pot_types,
                       _type_device_pot_table_meta *p_tables_metadata,
                       _type_device_pot_table_meta *p_device_tables_metadata, const _type_device_table_size tables,
                       const atom_type::_type_atom_types n_eles);

  /**
   * copy potential data after splining to devices
   * \param pot: origin potential data in CPU side.
   * \param _pot_types: types of atoms.
   */
  _type_device_pot potCopyHostToDevice(eam *pot, std::vector<atom_type::_type_atomic_no> _pot_types);

  /**
   * set consts in file hip_pot_device, so that we can access potentail meatdata and splines in kernel function.
   */
  void assignDevicePot(_type_device_pot device_pot);

  /**
   * remove potential data on devices.
   * \param device_pot pointer of data on devices.
   */
  void destroyDevicePotTables(_type_device_pot device_pot);

} // namespace hip_pot

#endif // HIP_POT_H
