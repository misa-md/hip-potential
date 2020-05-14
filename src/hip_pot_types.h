//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_TYPES_H
#define HIP_POT_TYPES_H

#include <iostream>

namespace hip_pot {
  typedef double _type_device_pot_element;
  typedef unsigned int _type_device_table_size;
  typedef _type_device_pot_element _type_device_pot_spline[7];

  typedef struct {
    _type_device_pot_element x0;     // the first element in table
    _type_device_pot_element inv_dx; // 1/dx
    _type_device_table_size n;       // data size in table
  } _type_device_pot_table_meta;
} // namespace hip_pot

#endif // HIP_POT_TYPES_H
