#include "och_setints_gpu.cuh"

#include <cstdint>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

__global__ void dev_set_to(int32_t value, void* beg, int32_t stride, int32_t x_sz, int32_t y_sz)
{
	const uint32_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx_x >= x_sz || idx_y >= y_sz)
		return;

	int32_t* mem = reinterpret_cast<int32_t*>(beg);

	mem[idx_x + idx_y * stride] = 0;
}

cudaError_t launch_set_to(dim3 threads_per_block, dim3 blocks_per_grid, int32_t value, void* beg, int32_t stride, int32_t x_sz, int32_t y_sz)
{
	dev_set_to<<<threads_per_block, blocks_per_grid>>>(value, beg, stride, x_sz, y_sz);

	return cudaGetLastError();
}