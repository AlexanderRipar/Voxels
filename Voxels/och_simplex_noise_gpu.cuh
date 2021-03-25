
#include <cstdint>

#include "device_launch_parameters.h"

__global__ void d_simplex_3d_float(cudaPitchedPtr dst, uint3 dim, float3 begin, float3 step, uint32_t seed);

__global__ void d_simplex_3d_uint8_t(cudaPitchedPtr dst, uint3 dim, float3 begin, float3 step, uint32_t seed);