#include <cstdint>

#include "device_launch_parameters.h"

__global__ void d_uint8_to_bit(cudaPitchedPtr dst, const cudaPitchedPtr src, uint8_t cutoff);

__global__ void d_float_to_bit(cudaPitchedPtr dst, const cudaPitchedPtr src, float cutoff);
