// -*- c++ -*-
#ifndef DUST_CUDA_CUH
#define DUST_CUDA_CUH

#ifdef __NVCC__
#define DEVICE __device__
#define HOST __host__
#define HOSTDEVICE __host__ __device__
#define KERNEL __global__
#define ALIGN(n) __align__(n)

// This is necessary due to templates which are __host__ __device__;
// whenever a HOSTDEVICE function is called from another HOSTDEVICE
// function the compiler gets confused as it can't tell which one it's
// going to use. This suppresses the warning as it is ok here.
#define __nv_exec_check_disable__ _Pragma("nv_exec_check_disable")

#include <cuda_call.cuh>

#include <device_launch_parameters.h>

#else
#define DEVICE
#define HOST
#define HOSTDEVICE
#define KERNEL
#undef DUST_CUDA_ENABLE_PROFILER
#define __nv_exec_check_disable__
#define ALIGN(n)
#endif

// const definition depends on __host__/__device__
#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

const int warp_size = 32;


#endif
