#include "voxels.h"

#include <cstdio>
#include <cstdint>

#include "cuda.h"
#include "cuda_texture_types.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "channel_descriptor.h"

#include "och_cudahelpers.cuh"
#include "och_simplex_noise_gpu.cuh"

#include "och_timer.h"

cudaArray_t d_voxel_arr;

texture<uint8_t, cudaTextureType3D, cudaReadModeElementType> d_voxel_tex;

uint8_t* d_slice;

void init_voxels(const uint32_t dim_log2, float noise_limit, uint32_t noise_seed)
{
	const uint32_t dim = 1 << dim_log2;

	CHECK(cudaMalloc(&d_slice, dim * dim));

	och::print("\nStarting init_volume\n");

	//device array
	cudaPitchedPtr d_voxel_lin;

	CHECK(cudaMalloc3D(&d_voxel_lin, make_cudaExtent(dim, dim, dim)));

	dim3 threads_per_block(32, 32, 32);
	dim3 blocks_per_grid(dim / 32, dim / 32, dim / 32);

	och::timer dev_fill_timer;

	for (int i = 0; i != 16 * 16 * 16; ++i)
	{
		d_simplex_3d_uint8_t<<<threads_per_block, blocks_per_grid>>>(d_voxel_lin, make_uint3(dim, dim, dim), make_float3(0.0F, 0.0F, 0.0F), make_float3(16.0F / dim, 16.0F / dim, 1.0F / dim), i);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			och::print("Error (loop #{}): {}\n", i, cudaGetErrorString(err));
		cudaDeviceSynchronize();
	}

	och::print("\n{} for {}^3 noise-calls\n", dev_fill_timer.read(), dim);

	//device cudaArray
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uint8_t>();

	CHECK(cudaMalloc3DArray(&d_voxel_arr, &channel_desc, make_cudaExtent(dim, dim, dim)));

	//Copy from device array to 3D-cudaArray
	cudaMemcpy3DParms cpy_params{ 0 };
	cpy_params.srcPtr = d_voxel_lin;
	cpy_params.dstArray = d_voxel_arr;
	cpy_params.extent = make_cudaExtent(dim, dim, dim);
	cpy_params.kind = cudaMemcpyHostToDevice;

	CHECK(cudaMemcpy3D(&cpy_params));

	d_voxel_tex.normalized = false;
	d_voxel_tex.filterMode = cudaFilterModePoint;
	d_voxel_tex.addressMode[0] = d_voxel_tex.addressMode[1] = d_voxel_tex.addressMode[2] = cudaAddressModeBorder;

	CHECK(cudaBindTextureToArray(&d_voxel_tex, d_voxel_arr, &channel_desc));

	CHECK(cudaFree(d_voxel_lin.ptr));

	och::print("\nFinishing init_volume\n");
}

__global__ void get_slice_kernel(uint8_t* dst, uint32_t z, uint32_t dim)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	dst[x + y * dim] = tex3D<uint8_t>(d_voxel_tex, x, y, z);
}

void get_slice(uint8_t* h_slice, uint32_t z, uint32_t dim)
{
	dim3 threads_per_block(32, 32);

	dim3 blocks_per_grid(dim / 32, dim / 32);
	
	get_slice_kernel<<<threads_per_block, blocks_per_grid>>>(d_slice, z, dim);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error2: %s\n", cudaGetErrorString(err));

	CHECK(cudaMemcpy(h_slice, d_slice, dim * dim, cudaMemcpyDeviceToHost));
}

void launch_voxels(uint32_t dim_log2, float noise_limit, uint32_t noise_seed)
{
	init_voxels(dim_log2, noise_limit, noise_seed);
}
