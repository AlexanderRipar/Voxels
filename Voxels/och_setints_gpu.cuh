#include <cstdint>
#include "cuda_runtime.h"

__global__ void dev_set_to(int32_t value, void* beg, int32_t stride, int32_t x_sz, int32_t y_sz);

cudaError_t launch_set_to(dim3 threads_per_block, dim3 blocks_per_grid, int32_t value, void* beg, int32_t stride, int32_t x_sz, int32_t y_sz);
