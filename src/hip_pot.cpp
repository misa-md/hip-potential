//
// Created by genshen on 2020-05-13
//

#include "hip_pot.h"
#include "hip_macros.h"
#include "hip_pot_device.h"
#include <hip/hip_runtime.h>

hip_pot::_type_device_pot hip_pot::potCopyHostToDevice(eam *_pot, std::vector<atom_type::_type_atomic_no> _pot_types) {
  atom_type::_type_atom_types n_eles = _pot_types.size();
  const _type_device_table_size tables = n_eles + n_eles + n_eles * (n_eles + 1) / 2;

  // the total number of data in origin potential tables.
  _type_device_table_size orgin_data_n = 0;
  // init tables metadata
  _type_device_pot_table_meta *p_tables_metadata = new _type_device_pot_table_meta[tables];
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto elec = _pot->electron_density.getEamItemByType(_pot_types[i]);
    p_tables_metadata[i] = _type_device_pot_table_meta{
        .x0 = elec->x0, .inv_dx = elec->invDx, .n = static_cast<_type_device_table_size>(elec->n)};
    orgin_data_n += elec->n;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto emb = _pot->embedded.getEamItemByType(_pot_types[i]);
    p_tables_metadata[i] = _type_device_pot_table_meta{
        .x0 = emb->x0, .inv_dx = emb->invDx, .n = static_cast<_type_device_table_size>(emb->n)};
    orgin_data_n += emb->n;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    for (_type_device_table_size j = i; j < n_eles; j++) {
      auto pair = _pot->eam_phi.getPhiByEamPhiByType(_pot_types[i], _pot_types[j]);
      p_tables_metadata[i] = _type_device_pot_table_meta{
          .x0 = pair->x0, .inv_dx = pair->invDx, .n = static_cast<_type_device_table_size>(pair->n)};
      orgin_data_n += pair->n;
    }
  }

  // init potential data: coefficient of splines
  _type_device_pot_spline **p_pot_tables = new _type_device_pot_spline *[tables];
  _type_device_table_size index = 0;
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto elec = _pot->electron_density.getEamItemByType(_pot_types[i]);
    _type_device_pot_spline *spline_data_device = nullptr;
    HIP_CHECK(hipMalloc((void **)&spline_data_device, sizeof(_type_device_pot_spline) * (elec->n + 1)));
    HIP_CHECK(hipMemcpy(spline_data_device, elec->spline, sizeof(_type_device_pot_spline) * (elec->n + 1),
                        hipMemcpyHostToDevice));
    p_pot_tables[index] = spline_data_device;
    index++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto emb = _pot->embedded.getEamItemByType(_pot_types[i]);
    _type_device_pot_spline *spline_data_device = nullptr;
    HIP_CHECK(hipMalloc((void **)&spline_data_device, sizeof(_type_device_pot_spline) * (emb->n + 1)));
    HIP_CHECK(hipMemcpy(spline_data_device, emb->spline, sizeof(_type_device_pot_spline) * (emb->n + 1),
                        hipMemcpyHostToDevice));
    p_pot_tables[index] = spline_data_device;
    index++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    for (_type_device_table_size j = i; j < n_eles; j++) {
      auto pair = _pot->eam_phi.getPhiByEamPhiByType(_pot_types[i], _pot_types[j]);
      _type_device_pot_spline *spline_data_device = nullptr;
      HIP_CHECK(hipMalloc((void **)&spline_data_device, sizeof(_type_device_pot_spline) * (pair->n + 1)));
      HIP_CHECK(hipMemcpy(spline_data_device, pair->spline, sizeof(_type_device_pot_spline) * (pair->n + 1),
                          hipMemcpyHostToDevice));
      p_pot_tables[index] = spline_data_device;
      index++;
    }
  }

  // copy metadat array of poteential tables to device side.
  _type_device_pot_table_meta *p_device_tables_metadata = nullptr;
  HIP_CHECK(hipMalloc((void **)&p_device_tables_metadata, sizeof(_type_device_pot_table_meta *) * tables));
  HIP_CHECK(hipMemcpy(p_device_tables_metadata, p_tables_metadata, sizeof(_type_device_pot_table_meta *) * tables,
                      hipMemcpyHostToDevice));

  // copy potential data array, whose content is also device pointers, to device side.
  _type_device_pot_spline **p_device_pot_tables = nullptr;
  HIP_CHECK(hipMalloc((void **)&p_device_pot_tables, sizeof(_type_device_pot_spline *) * tables));
  HIP_CHECK(
      hipMemcpy(p_device_pot_tables, p_pot_tables, sizeof(_type_device_pot_spline *) * tables, hipMemcpyHostToDevice));

  delete[] p_tables_metadata;
  delete[] p_pot_tables;
  return _type_device_pot{
      .ptr_device_pot_meta = p_device_tables_metadata, .ptr_device_pot_tables = p_device_pot_tables, .n_eles = n_eles};
}

void hip_pot::assignDevicePot(_type_device_pot device_pot) {
  auto n = device_pot.n_eles;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_eam_eles), &(n), sizeof(_type_device_table_size)));

  _type_device_pot_table_meta *dev_meta_ptr = device_pot.ptr_device_pot_meta;
  size_t meta_ptr_size = sizeof(_type_device_pot_table_meta *);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_tables_metadata), &(dev_meta_ptr), meta_ptr_size));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_ele_charge_table_metadata), &(dev_meta_ptr), meta_ptr_size));
  dev_meta_ptr += n;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_embedded_energy_table_metadata), &(dev_meta_ptr), meta_ptr_size));
  dev_meta_ptr += n;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_pair_table_metadata), &(dev_meta_ptr), meta_ptr_size));

  _type_device_pot_spline **dev_spline_ptr = device_pot.ptr_device_pot_tables;
  size_t spline_ptr_size = sizeof(_type_device_pot_spline **);
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_tables), &(dev_spline_ptr), spline_ptr_size));
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_ele_charge_density), &(dev_spline_ptr), spline_ptr_size));
  dev_spline_ptr += n;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_table_embedded_energy), &(dev_spline_ptr), spline_ptr_size));
  dev_spline_ptr += n;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(pot_origin_table_pair), &(dev_spline_ptr), spline_ptr_size));
}

void hip_pot::destroyDevicePotTables(_type_device_pot device_pot) {
  HIP_CHECK(hipFree(device_pot.ptr_device_pot_meta));
  for (atom_type::_type_atom_types i = 0; i < device_pot.n_eles; i++) {
    HIP_CHECK(hipFree(device_pot.ptr_device_pot_tables[i]));
  }
  HIP_CHECK(hipFree(device_pot.ptr_device_pot_tables));
}
