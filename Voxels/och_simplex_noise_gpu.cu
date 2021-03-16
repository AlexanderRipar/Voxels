#include "och_simplex_noise_gpu.cuh"

#include <cstdint>
#include <cmath>

#include "cuda.h"
#include "cuda_texture_types.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "channel_descriptor.h"

//__constant__ float d_grad3[12][3]
//{
//	{  1,  1,  0 }, { -1,  1,  0 }, {  1, -1,  0 }, { -1, -1,  0 },
//	{  1,  0,  1 }, { -1,  0,  1 }, {  1,  0, -1 }, { -1,  0, -1 },
//	{  0,  1,  1 }, {  0, -1,  1 }, {  0,  1, -1 }, {  0, -1, -1 }
//};

inline __device__ float d_dot_with_vec(float i, float j, float k, float x, float y, float z, uint32_t seed)
{
	const uint32_t h = (__float_as_int(i) * 73856093) ^ (__float_as_int(j) * 19349663) ^ (__float_as_int(k) * 83492791) ^ seed;

	//const uint32_t h_12 = ((h >> 4) * 12) >> 28;
	//return d_grad3[h_12][0] * x + d_grad3[h_12][1] * y + d_grad3[h_12][2] * z;
	
	//Two masks, which are either 0.0F or -0.0F, depending on positional hash
	const uint32_t neg1 =  h & 0x8000'0000;
	const uint32_t neg2 = (h & 0x1000'0000) << 3;

	//Get hash in [0, 2]
	const uint32_t h_3 = ((h >> 4) * 3) >> 28;

	uint32_t a, b;

	//Decide which inputs to pick depending on h_3
	if (h_3 == 0)
	{
		a = __float_as_int(y);
		b = __float_as_int(z);
	}
	else if (h_3 == 1)
	{
		a = __float_as_int(x);
		b = __float_as_int(z);
	}
	else
	{
		a = __float_as_int(x);
		b = __float_as_int(y);
	}

	//Return picked inputs, either negated or not, depending on masks
	return __int_as_float(a ^ neg1) + __int_as_float(b ^ neg2);
}

__global__ void d_simplex_3d_float(cudaPitchedPtr out, uint3 dim, float3 begin, float3 step, uint32_t seed)
{
	const uint32_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx_x >= dim.x || idx_y >= dim.y || idx_z >= dim.z)
		return;

	const float x_in = begin.x + step.x * idx_x;
	const float y_in = begin.y + step.y * idx_y;
	const float z_in = begin.z + step.z * idx_z;

	//Begin algorithm

	constexpr float skew_factor = 1.0F / 3.0F;
	constexpr float unskew_factor = 1.0F / 6.0F;

	const float skew = (x_in + y_in + z_in) * skew_factor;

	const float i0 = floorf(x_in + skew);
	const float j0 = floorf(y_in + skew);
	const float k0 = floorf(z_in + skew);

	const float unskew = (i0 + j0 + k0) * unskew_factor;

	const float x0 = x_in - i0 + unskew;
	const float y0 = y_in - j0 + unskew;
	const float z0 = z_in - k0 + unskew;

	const bool x_ge_y = x0 >= y0;
	const bool x_ge_z = x0 >= z0;
	const bool y_ge_z = y0 >= z0;

	const float i1 = static_cast<float>(  x_ge_y  &   x_ge_z );		//max == x
	const float i2 = static_cast<float>(  x_ge_y  |   x_ge_z );		//min != x

	const float j1 = static_cast<float>((!x_ge_y) &   y_ge_z );		//max == y
	const float j2 = static_cast<float>((!x_ge_y) |   y_ge_z );		//min != y

	const float k1 = static_cast<float>((!x_ge_z) & (!y_ge_z));		//max == z
	const float k2 = static_cast<float>((!x_ge_z) | (!y_ge_z));		//min != z

	const float x1 = x0 - i1 + unskew_factor;
	const float y1 = y0 - j1 + unskew_factor;
	const float z1 = z0 - k1 + unskew_factor;

	const float x2 = x0 - i2 + unskew_factor * 2.0F;
	const float y2 = y0 - j2 + unskew_factor * 2.0F;
	const float z2 = z0 - k2 + unskew_factor * 2.0F;

	const float x3 = x0 - 1.0F + unskew_factor * 3.0F;
	const float y3 = y0 - 1.0F + unskew_factor * 3.0F;
	const float z3 = z0 - 1.0F + unskew_factor * 3.0F;

	float t0 = 0.5F - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 < 0) t0 = 0;
	t0 = t0 * t0 * t0 * t0 * d_dot_with_vec(i0, j0, k0, x0, y0, z0, seed);

	float t1 = 0.5F - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 < 0) t1 = 0;
	t1 = t1 * t1 * t1 * t1 * d_dot_with_vec(i0 + i1, j0 + j1, k0 + k1, x1, y1, z1, seed);

	float t2 = 0.5F - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 < 0) t2 = 0;
	t2 = t2 * t2 * t2 * t2 * d_dot_with_vec(i0 + i2, j0 + j2, k0 + k2, x2, y2, z2, seed);

	float t3 = 0.5F - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 < 0) t3 = 0;
	t3 = t3 * t3 * t3 * t3 * d_dot_with_vec(i0 + 1.0F, j0 + 1.0F, k0 + 1.0F, x3, y3, z3, seed);

	//76.0F maps to just within [-1.0F, 1.0F]
	reinterpret_cast<float*>(out.ptr)[idx_x + idx_y * out.pitch + idx_z * out.pitch * out.ysize] = 76.0F * (t0 + t1 + t2 + t3);

	return;
}

__global__ void d_simplex_3d_uint8_t(cudaPitchedPtr out, uint3 dim, float3 begin, float3 step, uint32_t seed)
{
	const uint32_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t idx_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (idx_x >= dim.x || idx_y >= dim.y || idx_z >= dim.z)
		return;

	const float x_in = begin.x + step.x * idx_x;
	const float y_in = begin.y + step.y * idx_y;
	const float z_in = begin.z + step.z * idx_z;

	//Begin algorithm

	constexpr float skew_factor = 1.0F / 3.0F;

	const float skew = (x_in + y_in + z_in) * skew_factor;

	const float i0 = floorf(x_in + skew);
	const float j0 = floorf(y_in + skew);
	const float k0 = floorf(z_in + skew);

	constexpr float unskew_factor = 1.0F / 6.0F;

	const float unskew = (i0 + j0 + k0) * unskew_factor;

	const float x_orig = i0 - unskew;
	const float y_orig = j0 - unskew;
	const float z_orig = k0 - unskew;

	const float x0 = x_in - x_orig;
	const float y0 = y_in - y_orig;
	const float z0 = z_in - z_orig;

	const float i1 = (float)((x0 >= y0) & (x0 >= z0));		//max == x
	const float j1 = (float)((y0 > x0) & (y0 >= z0));		//max == y
	const float k1 = (float)((z0 > x0) & (z0 > y0));		//max == z

	const float i2 = (float)((x0 >= y0) | (x0 >= z0));		//min != x
	const float j2 = (float)((y0 > x0) | (y0 >= z0));		//min != y
	const float k2 = (float)((z0 > x0) | (z0 > y0));		//min != z

	const float x1 = x0 - i1 + unskew_factor;
	const float y1 = y0 - j1 + unskew_factor;
	const float z1 = z0 - k1 + unskew_factor;

	const float x2 = x0 - i2 + unskew_factor * 2.0F;
	const float y2 = y0 - j2 + unskew_factor * 2.0F;
	const float z2 = z0 - k2 + unskew_factor * 2.0F;

	const float x3 = x0 - 1.0F + unskew_factor * 3.0F;
	const float y3 = y0 - 1.0F + unskew_factor * 3.0F;
	const float z3 = z0 - 1.0F + unskew_factor * 3.0F;

	float t0 = 0.5F - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 < 0) t0 = 0;
	t0 = t0 * t0 * t0 * t0 * d_dot_with_vec(i0, j0, k0, x0, y0, z0, seed);

	float t1 = 0.5F - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 < 0) t1 = 0;
	t1 = t1 * t1 * t1 * t1 * d_dot_with_vec(i0 + i1, j0 + j1, k0 + k1, x1, y1, z1, seed);

	float t2 = 0.5F - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 < 0) t2 = 0;
	t2 = t2 * t2 * t2 * t2 * d_dot_with_vec(i0 + i2, j0 + j2, k0 + k2, x2, y2, z2, seed);

	float t3 = 0.5F - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 < 0) t3 = 0;
	t3 = t3 * t3 * t3 * t3 * d_dot_with_vec(i0 + 1.0F, j0 + 1.0F, k0 + 1.0F, x3, y3, z3, seed);

	reinterpret_cast<uint8_t*>(out.ptr)[idx_x + idx_y * out.pitch + idx_z * out.pitch * out.ysize] = (76.0F * (t0 + t1 + t2 + t3)) * 128 + 128;

	return;
}