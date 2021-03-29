
#include <cstdint>

#include "device_launch_parameters.h"

__global__ void d_simplex_3d_float(cudaPitchedPtr dst, uint3 dim, float3 begin, float3 step, uint32_t seed);

__global__ void d_simplex_3d_uint8_t(cudaPitchedPtr dst, uint3 dim, float3 begin, float3 step, uint32_t seed);

__global__ void d_simplex_3d_surface2d_grayscale_argb(cudaSurfaceObject_t surf, uint2 dim, float3 begin, float2 step, uint32_t seed);

cudaError_t launch_simplex_3d_surface2d_grayscale_argb(dim3 threads_per_block, dim3 blocks_per_grid, cudaSurfaceObject_t surf, uint2 dim, float3 begin, float2 step, uint32_t seed);
