//
// Created by genshen on 2020-05-14
//

#ifndef HIP_POT_MACROS_H
#define HIP_POT_MACROS_H

#include <iostream>
#include <hip/hip_runtime.h>

#include "pot_building_config.h"


#if defined HIP_POT_DEVICE_API_INLINE
#define HIP_POT_INLINE __forceinline__
#else
#define HIP_POT_INLINE
#endif


#if defined POT_NV_PLATFORM
#define __DEVICE_CONSTANT__ // ignore constant
#else
#define __DEVICE_CONSTANT__ __constant__
#endif

/**
 * macro for checking result of hip operation.
 */
#define HIP_CHECK(func)                                                                                                \
  {                                                                                                                    \
    hipError_t err = func;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "Error: HIP reports " << hipGetErrorString(err) << std::endl;                                       \
      std::cerr << "Raised in file " << __FILE__ << "#" << __func__ << "#line " << __LINE__ << std::endl;              \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  }

#endif // HIP_POT_MACROS_H
