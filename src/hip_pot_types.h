//
// Created by genshen on 2020-05-13
//

#ifndef HIP_POT_TYPES_H
#define HIP_POT_TYPES_H

#include <iostream>

#include "pot_building_config.h"

namespace hip_pot {
  typedef double _type_device_pot_table_item; // type of data in origin potential table or spline table.
  typedef unsigned int _type_device_table_size;
  typedef _type_device_pot_table_item _type_device_pot_spline[7];

#ifndef HIP_POT_COMPACT_MEM_LAYOUT
  typedef _type_device_pot_spline **_type_spline_colle; // array of spline of each alloy.
#endif
#ifdef HIP_POT_COMPACT_MEM_LAYOUT
  typedef _type_device_pot_spline *_type_spline_colle; // array of splines. Splines number for each alloy is the same
#endif

  typedef struct {
    _type_device_pot_table_item x0;     // the first element in table
    _type_device_pot_table_item inv_dx; // 1/dx (dx: delta x)
    _type_device_table_size n;          // data size in table
  } _type_device_pot_table_meta;
} // namespace hip_pot

#endif // HIP_POT_TYPES_H
