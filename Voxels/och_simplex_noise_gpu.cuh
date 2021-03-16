
#include <cstdint>

#include "cuda_runtime_api.h"

__global__ void d_simplex_3d_float(cudaPitchedPtr out, uint3 dim, float3 begin, float3 step, uint32_t seed);

__global__ void d_simplex_3d_uint8_t(cudaPitchedPtr out, uint3 dim, float3 begin, float3 step, uint32_t seed);