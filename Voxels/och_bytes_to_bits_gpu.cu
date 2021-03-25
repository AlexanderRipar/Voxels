#include "och_bytes_to_bits_gpu.cuh"

#include <cstdint>
#include <cmath>

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

__global__ void d_uint8_to_bit(cudaPitchedPtr dst, const cudaPitchedPtr src, uint8_t cutoff)
{
	const uint32_t dst_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t dst_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t dst_idx_z = blockIdx.z * blockDim.z + threadIdx.z;

	const uint32_t src_idx_x = dst_idx_x * 2;
	const uint32_t src_idx_y = dst_idx_y * 2;
	const uint32_t src_idx_z = dst_idx_z * 2;
	
	uint32_t idx_0 = src_idx_x + src_idx_y * src.pitch + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_1 = idx_0 + 1;
	uint32_t idx_2 = idx_0 + src.pitch;
	uint32_t idx_3 = idx_1 + src.pitch;
	uint32_t idx_4 = idx_0 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_5 = idx_1 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_6 = idx_2 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_7 = idx_3 + src_idx_z * src.pitch * src.ysize;

	uint8_t output = 0;

	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_0] > cutoff);
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_1] > cutoff) << 1;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_2] > cutoff) << 2;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_3] > cutoff) << 3;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_4] > cutoff) << 4;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_5] > cutoff) << 5;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_6] > cutoff) << 6;
	output |= (reinterpret_cast<const uint8_t*>(src.ptr)[idx_7] > cutoff) << 7;

	reinterpret_cast<uint8_t*>(dst.ptr)[dst_idx_x + dst_idx_y * dst.pitch + dst_idx_z * dst.pitch * dst.ysize] = output;
}

__global__ void d_float_to_bit(cudaPitchedPtr dst, const cudaPitchedPtr src, float cutoff)
{
	const uint32_t dst_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t dst_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t dst_idx_z = blockIdx.z * blockDim.z + threadIdx.z;

	const uint32_t src_idx_x = dst_idx_x * 2;
	const uint32_t src_idx_y = dst_idx_y * 2;
	const uint32_t src_idx_z = dst_idx_z * 2;

	uint32_t idx_0 = src_idx_x + src_idx_y * src.pitch + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_1 = idx_0 + 1;
	uint32_t idx_2 = idx_0 + src.pitch;
	uint32_t idx_3 = idx_1 + src.pitch;
	uint32_t idx_4 = idx_0 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_5 = idx_1 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_6 = idx_2 + src_idx_z * src.pitch * src.ysize;
	uint32_t idx_7 = idx_3 + src_idx_z * src.pitch * src.ysize;

	uint8_t output = 0;

	output |= (reinterpret_cast<const float*>(src.ptr)[idx_0] > cutoff);
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_1] > cutoff) << 1;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_2] > cutoff) << 2;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_3] > cutoff) << 3;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_4] > cutoff) << 4;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_5] > cutoff) << 5;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_6] > cutoff) << 6;
	output |= (reinterpret_cast<const float*>(src.ptr)[idx_7] > cutoff) << 7;

	reinterpret_cast<uint8_t*>(dst.ptr)[dst_idx_x + dst_idx_y * dst.pitch + dst_idx_z * dst.pitch * dst.ysize] = output;
}
