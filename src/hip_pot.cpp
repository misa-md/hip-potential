//
// Created by genshen on 2020-05-13
//

#include <cassert>
#include <hip/hip_runtime.h>

#include "hip_macros.h"
#include "hip_pot.h"
#include "hip_pot_device.h"

void hip_pot::potCopyMetadata(eam *_pot, std::vector<atom_type::_type_atomic_no> _pot_types,
                              _type_device_pot_table_meta *p_tables_metadata,
                              _type_device_pot_table_meta *p_device_tables_metadata,
                              const _type_device_table_size tables, const atom_type::_type_atom_types n_eles) {
  // the total number of data in origin potential tables.
  _type_device_table_size orgin_data_n = 0;
  // generate tables metadata
  _type_device_table_size meta_i = 0;
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto elec = _pot->electron_density.getEamItemByType(_pot_types[i]);
    assert(elec != nullptr);
    p_tables_metadata[meta_i] = _type_device_pot_table_meta{
        .x0 = elec->x0, .inv_dx = elec->invDx, .n = static_cast<_type_device_table_size>(elec->n)};
    orgin_data_n += elec->n;
    meta_i++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto emb = _pot->embedded.getEamItemByType(_pot_types[i]);
    assert(emb != nullptr);
    p_tables_metadata[meta_i] = _type_device_pot_table_meta{
        .x0 = emb->x0, .inv_dx = emb->invDx, .n = static_cast<_type_device_table_size>(emb->n)};
    orgin_data_n += emb->n;
    meta_i++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    for (_type_device_table_size j = i; j < n_eles; j++) {
      auto pair = _pot->eam_phi.getPhiByEamPhiByType(_pot_types[i], _pot_types[j]);
      assert(pair != nullptr);
      p_tables_metadata[meta_i] = _type_device_pot_table_meta{
          .x0 = pair->x0, .inv_dx = pair->invDx, .n = static_cast<_type_device_table_size>(pair->n)};
      orgin_data_n += pair->n;
      meta_i++;
    }
  }

  // malloc and copy: copy metadata array of potential tables to device side.
  HIP_CHECK(hipMemcpy(p_device_tables_metadata, p_tables_metadata, sizeof(_type_device_pot_table_meta) * tables,
                      hipMemcpyHostToDevice));
}

hip_pot::_type_device_pot hip_pot::potCopyHostToDevice(eam *_pot, std::vector<atom_type::_type_atomic_no> _pot_types) {
  atom_type::_type_atom_types n_eles = _pot_types.size();
  const _type_device_table_size tables = n_eles + n_eles + n_eles * (n_eles + 1) / 2;

  // set and copy metadata to device side.
  _type_device_pot_table_meta *p_host_tables_metadata = new _type_device_pot_table_meta[tables];
  _type_device_pot_table_meta *p_device_tables_metadata = nullptr;
  HIP_CHECK(hipMalloc((void **)&p_device_tables_metadata, sizeof(_type_device_pot_table_meta) * tables));
  potCopyMetadata(_pot, _pot_types, p_host_tables_metadata, p_device_tables_metadata, tables, n_eles);

  // calculate spline number (total number of splines) for mallocing device array.
  _type_device_table_size spline_num = 0;
  for (_type_device_table_size i = 0; i < tables; i++) {
    spline_num += p_host_tables_metadata[i].n + 1;
  }
  delete[] p_host_tables_metadata;

  _type_device_pot_spline *spline_data_device = nullptr;
  HIP_CHECK(hipMalloc((void **)&spline_data_device, sizeof(_type_device_pot_spline) * spline_num));
  _type_device_pot_spline *spline_data_device_cursor = spline_data_device;

  // init potential data: coefficient of splines
  _type_device_pot_spline **p_pot_tables = new _type_device_pot_spline *[tables];
  _type_device_table_size index = 0;
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto elec = _pot->electron_density.getEamItemByType(_pot_types[i]);
    assert(elec != nullptr);
    HIP_CHECK(hipMemcpy(spline_data_device_cursor, elec->spline, sizeof(_type_device_pot_spline) * (elec->n + 1),
                        hipMemcpyHostToDevice));
    p_pot_tables[index] = spline_data_device_cursor;
    spline_data_device_cursor += elec->n + 1;
    index++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    auto emb = _pot->embedded.getEamItemByType(_pot_types[i]);
    assert(emb != nullptr);
    HIP_CHECK(hipMemcpy(spline_data_device_cursor, emb->spline, sizeof(_type_device_pot_spline) * (emb->n + 1),
                        hipMemcpyHostToDevice));
    p_pot_tables[index] = spline_data_device_cursor;
    spline_data_device_cursor += emb->n + 1;
    index++;
  }
  for (_type_device_table_size i = 0; i < n_eles; i++) {
    for (_type_device_table_size j = i; j < n_eles; j++) {
      auto pair = _pot->eam_phi.getPhiByEamPhiByType(_pot_types[i], _pot_types[j]);
      assert(pair != nullptr);
      HIP_CHECK(hipMemcpy(spline_data_device_cursor, pair->spline, sizeof(_type_device_pot_spline) * (pair->n + 1),
                          hipMemcpyHostToDevice));
      p_pot_tables[index] = spline_data_device_cursor;
      spline_data_device_cursor += pair->n + 1;
      index++;
    }
  }

  // copy potential data array, whose content is also device pointers, to device side.
  _type_device_pot_spline **p_device_pot_tables = nullptr;
  HIP_CHECK(hipMalloc((void **)&p_device_pot_tables, sizeof(_type_device_pot_spline *) * tables));
  HIP_CHECK(
      hipMemcpy(p_device_pot_tables, p_pot_tables, sizeof(_type_device_pot_spline *) * tables, hipMemcpyHostToDevice));

  delete[] p_pot_tables;
  return _type_device_pot{.ptr_device_pot_meta = p_device_tables_metadata,
                          .ptr_device_pot_tables = p_device_pot_tables,
                          .device_pot_data = spline_data_device,
                          .n_eles = n_eles};
}

void hip_pot::assignDevicePot(_type_device_pot device_pot) {
  set_device_variables(device_pot.n_eles, device_pot.ptr_device_pot_meta, device_pot.ptr_device_pot_tables);
}

void hip_pot::destroyDevicePotTables(_type_device_pot device_pot) {
  HIP_CHECK(hipFree(device_pot.ptr_device_pot_meta));
  HIP_CHECK(hipFree(device_pot.device_pot_data));
  HIP_CHECK(hipFree(device_pot.ptr_device_pot_tables));
}
