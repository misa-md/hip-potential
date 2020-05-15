//
// Created by genshen on 2020-05-14
//

#include "eam_calc_test_device.h"
#include "hip_eam_device.h"
#include "hip_macros.h"
#include "hip_pot_device.h"
#include <hip/hip_runtime.h>


__global__ void _kernelCheckSplineCopy(int ele_size, int data_size) {
  double fd = 0.0;
  for (hip_pot::_type_device_table_size i = 0; i < ele_size; i++) {
    // set elec and embed energy splines
    auto elec = pot_table_ele_charge_density[i];
    auto emb = pot_table_embedded_energy[i];
    for (int k = 0; k < data_size; k++) {
      for (int s = 0; s < 7; s++) {
        if (elec[k][s] != fd) {
          abort();
        }
        fd += 1.0;
        if (emb[k][s] != fd) {
          abort();
        }
        fd += 1.0;
      }
    }
    // set pair enregy splines
    for (hip_pot::_type_device_table_size j = i; j < ele_size; j++) {
      int index = ele_size * i - (i + 1) * i / 2 + j;
      auto pair = pot_table_pair[index];
      for (int k = 0; k < data_size; k++) {
        for (int s = 0; s < 7; s++) {
          if (pair[k][s] != fd) {
            abort();
          }
          fd += 1.0;
        }
      }
    }
  }
}

void checkPotSplinesCopy(int ele_size, int data_size) {
  hipLaunchKernelGGL(_kernelCheckSplineCopy, dim3(16, 1), dim3(1, 1), 0, 0, ele_size, data_size);
}

