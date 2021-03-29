#include "och_setints_gpu.cuh"

#include <cstdint>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

__global__ void dev_set_memory_to(int32_t value, void* beg, int32_t stride, int32_t w, int32_t h)
{
	const uint32_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx_x >= w || idx_y >= h)
		return;

	int32_t* mem = reinterpret_cast<int32_t*>(beg);

	mem[idx_x + idx_y * stride] = 0;
}

cudaError_t launch_set_memory_to(dim3 threads_per_block, dim3 blocks_per_grid, int32_t value, void* beg, int32_t stride, int32_t w, int32_t h)
{
	dev_set_memory_to<<<threads_per_block, blocks_per_grid>>>(value, beg, stride, w, h);

	return cudaGetLastError();
}

__global__ void dev_set_surface_to(int32_t value, cudaSurfaceObject_t surf, int32_t w, int32_t h)
{
	const uint32_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx_x >= w || idx_y >= h)
		return;

	surf2Dwrite(value, surf, idx_x * 4, idx_y);
}

cudaError_t launch_set_surface_to(dim3 threads_per_block, dim3 blocks_per_grid, int32_t value, cudaSurfaceObject_t surf, int32_t w, int32_t h)
{
	dev_set_surface_to<<<threads_per_block, blocks_per_grid>>>(value, surf, w, h);

	return cudaGetLastError();
}
