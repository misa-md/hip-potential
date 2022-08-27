//
// Created by genshen on 2022/8/27.

#include <hip/hip_runtime.h>

#include "hip_pot_dev_eam_segmented_spline.h"
#include "hip_pot_macros.h"
#include "hip_pot_types.h"

__device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_ele_charge_density_1st = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_embedded_energy_1st = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_1st_spline_coll pot_table_pair_1st = nullptr;

__device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_ele_charge_density_2nd = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_embedded_energy_2nd = nullptr;
__device__ __DEVICE_CONSTANT__ hip_pot::_type_2nd_spline_coll pot_table_pair_2nd = nullptr;

#ifdef HIP_POT_COMPACT_MEM_LAYOUT

// copy segmented spline for one type of tables and only one type of alloy element.
__device__ void copy_segmented_spline(const hip_pot::_type_device_pot_table_meta meta,
                                      hip_pot::_type_spline_colle unsegmented_spline,
                                      hip_pot::_type_1st_spline_coll segmented_spline_1st,
                                      hip_pot::_type_2nd_spline_coll segmented_spline_2nd) {
  const int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  const int global_threads = hipGridDim_x * hipBlockDim_x;
  for (int id = tid; id < meta.n + 1; id += global_threads) {
    hip_pot::_type_device_pot_spline &spline_data = unsegmented_spline[id];
    // coefficient for the 2nd derivative
    segmented_spline_2nd[id][0] = spline_data[0];
    segmented_spline_2nd[id][1] = spline_data[1];
    segmented_spline_2nd[id][2] = spline_data[2];
    // coefficient for the 1st derivative
    segmented_spline_1st[id][0] = spline_data[3];
    segmented_spline_1st[id][1] = spline_data[4];
    segmented_spline_1st[id][2] = spline_data[5];
    segmented_spline_1st[id][3] = spline_data[6];
  }
}

// here, we assume that tables for each alloy element have the same length (compact memory layout mode).
__global__ void copy_pot_spline_segmented(
    const hip_pot::_type_device_table_size n_eles, const hip_pot::_type_device_pot_table_meta *meta,
    hip_pot::_type_spline_colle pot_table_ele_charge_density, hip_pot::_type_spline_colle pot_table_embedded_energy,
    hip_pot::_type_spline_colle pot_table_pair, hip_pot::_type_1st_spline_coll pot_table_ele_charge_density_1st,
    hip_pot::_type_1st_spline_coll pot_table_embedded_energy_1st, hip_pot::_type_1st_spline_coll pot_table_pair_1st,
    hip_pot::_type_2nd_spline_coll pot_table_ele_charge_density_2nd,
    hip_pot::_type_2nd_spline_coll pot_table_embedded_energy_2nd, hip_pot::_type_2nd_spline_coll pot_table_pair_2nd) {
  // copy ele charge density
  for (int i = 0, offset = 0; i < n_eles; i++) {
    copy_segmented_spline(meta[i], pot_table_ele_charge_density + offset, pot_table_ele_charge_density_1st + offset,
                          pot_table_ele_charge_density_2nd + offset);
    offset += meta[i].n + 1;
  }
  // copy embedded_energy
  for (int i = 0, offset = 0; i < n_eles; i++) {
    const int k = i + n_eles;
    copy_segmented_spline(meta[k], pot_table_embedded_energy + offset, pot_table_embedded_energy_1st + offset,
                          pot_table_embedded_energy_2nd + offset);
    offset += meta[k].n + 1;
  }
  // copy pair potential
  for (int i = 0, offset = 0; i < (n_eles * (n_eles + 1) / 2); i++) {
    const int k = i + n_eles + n_eles;
    copy_segmented_spline(meta[k], pot_table_pair + offset, pot_table_pair_1st + offset, pot_table_pair_2nd + offset);
    offset += meta[k].n + 1;
  }
}

void copy_pot_spline_segmented_wrapper(const hip_pot::_type_device_table_size n_eles,
                                       const hip_pot::_type_device_pot_table_meta *dev_meta,
                                       const hip_pot::_type_device_pot_table_meta *host_meta,
                                       hip_pot::_type_spline_colle pot_table_ele_charge_density,
                                       hip_pot::_type_spline_colle pot_table_embedded_energy,
                                       hip_pot::_type_spline_colle pot_table_pair) {
  hip_pot::_type_1st_spline_coll device_ele_charge_density_splines_1st = nullptr;
  hip_pot::_type_1st_spline_coll device_embedded_energy_1st = nullptr;
  hip_pot::_type_1st_spline_coll device_pair_1st = nullptr;

  hip_pot::_type_2nd_spline_coll device_ele_charge_density_splines_2nd = nullptr;
  hip_pot::_type_2nd_spline_coll device_embedded_energy_2nd = nullptr;
  hip_pot::_type_2nd_spline_coll device_pair_2nd = nullptr;

  int eles_sum = 0;
  for (int i = 0; i < n_eles; i++) {
    eles_sum += host_meta[i].n + 1;
  }

  hipMalloc((void **)&device_ele_charge_density_splines_1st,
            sizeof(hip_pot::_type_device_pot_spline_1st_derivative) * eles_sum);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_ele_charge_density_1st), &(device_ele_charge_density_splines_1st),
                              sizeof(hip_pot::_type_1st_spline_coll)));
  hipMalloc((void **)&device_ele_charge_density_splines_2nd,
            sizeof(hip_pot::_type_device_pot_spline_2nd_derivative) * eles_sum);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_ele_charge_density_2nd), &(device_ele_charge_density_splines_2nd),
                              sizeof(hip_pot::_type_2nd_spline_coll)));

  eles_sum = 0;
  for (int i = 0; i < n_eles; i++) {
    eles_sum += host_meta[n_eles + i].n + 1;
  }

  hipMalloc((void **)&device_embedded_energy_1st, sizeof(hip_pot::_type_device_pot_spline_1st_derivative) * eles_sum);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_embedded_energy_1st), &(device_embedded_energy_1st),
                              sizeof(hip_pot::_type_1st_spline_coll)));
  hipMalloc((void **)&device_embedded_energy_2nd, sizeof(hip_pot::_type_device_pot_spline_2nd_derivative) * eles_sum);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_embedded_energy_2nd), &(device_embedded_energy_2nd),
                              sizeof(hip_pot::_type_2nd_spline_coll)));

  eles_sum = 0;
  for (int i = 0; i < (n_eles * (n_eles + 1) / 2); i++) {
    eles_sum += host_meta[2 * n_eles + i].n + 1;
  }
  printf("6\n");
  hipMalloc((void **)&device_pair_1st, sizeof(hip_pot::_type_device_pot_spline_1st_derivative) * eles_sum);
  HIP_CHECK(
      hipMemcpyToSymbol(HIP_SYMBOL(pot_table_pair_1st), &(device_pair_1st), sizeof(hip_pot::_type_1st_spline_coll)));
  hipMalloc((void **)&device_pair_2nd, sizeof(hip_pot::_type_device_pot_spline_2nd_derivative) * eles_sum);
  HIP_CHECK(
      hipMemcpyToSymbol(HIP_SYMBOL(pot_table_pair_2nd), &(device_pair_2nd), sizeof(hip_pot::_type_2nd_spline_coll)));
  printf("7\n");

  copy_pot_spline_segmented<<<1024, 256>>>(
      n_eles, dev_meta, pot_table_ele_charge_density, pot_table_embedded_energy, pot_table_pair,
      device_ele_charge_density_splines_1st, device_embedded_energy_1st, device_pair_1st,
      device_ele_charge_density_splines_2nd, device_embedded_energy_2nd, device_pair_2nd);
  printf("8\n");
}

#endif // HIP_POT_COMPACT_MEM_LAYOUT
