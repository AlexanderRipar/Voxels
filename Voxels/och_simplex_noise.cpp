#include "och_simplex_noise.h"

#include <cstdint>
#include <cmath>
#include <cstdio>

#include <immintrin.h>

constexpr float grad3[12][3]
{
	{  1,  1,  0 }, { -1,  1,  0 }, {  1, -1,  0 }, { -1, -1,  0 },
	{  1,  0,  1 }, { -1,  0,  1 }, {  1,  0, -1 }, { -1,  0, -1 },
	{  0,  1,  1 }, {  0, -1,  1 }, {  0,  1, -1 }, {  0, -1, -1 }
};

//https://www.researchgate.net/publication/2909661_Optimized_Spatial_Hashing_for_Collision_Detection_of_Deformable_Objects
uint32_t hash(float i, float j, float k, uint32_t seed)
{
	const uint32_t _i = *reinterpret_cast<uint32_t*>(&i);
	const uint32_t _j = *reinterpret_cast<uint32_t*>(&j);
	const uint32_t _k = *reinterpret_cast<uint32_t*>(&k);

	const uint64_t h = (_i * 73856093) ^ (_j * 19349663) ^ (_k * 83492791) ^ seed;

	return (h * 12) >> 32;
}

float dot_with_vec(float i, float j, float k, float x, float y, float z, uint32_t seed)
{
	const int h = hash(i, j, k, seed);

	return grad3[h][0] * x + grad3[h][1] * y + grad3[h][2] * z;
}

float simplex_3d(float x_in, float y_in, float z_in, uint32_t seed)
{
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

	const float i1 = (float) ((x0 >= y0) & (x0 >= z0));		//max == x
	const float j1 = (float) ((y0 >  x0) & (y0 >= z0));		//max == y
	const float k1 = (float) ((z0 >  x0) & (z0 >  y0));		//max == z
	
	const float i2 = (float) ((x0 >= y0) | (x0 >= z0));		//min != x
	const float j2 = (float) ((y0 >  x0) | (y0 >= z0));		//min != y
	const float k2 = (float) ((z0 >  x0) | (z0 >  y0));		//min != z
	
	const float x1 = x0 - i1 + unskew_factor;
	const float y1 = y0 - j1 + unskew_factor;
	const float z1 = z0 - k1 + unskew_factor;

	const float x2 = x0 - i2 + unskew_factor * 2.0F;
	const float y2 = y0 - j2 + unskew_factor * 2.0F;
	const float z2 = z0 - k2 + unskew_factor * 2.0F;

	const float x3 = x0 - 1.0F + unskew_factor * 3.0F;
	const float y3 = y0 - 1.0F + unskew_factor * 3.0F;
	const float z3 = z0 - 1.0F + unskew_factor * 3.0F;

	//Find contributions from vectors
	float t0 = 0.5F - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 < 0)
		t0 = 0;
	else
		t0 = t0 * t0 * t0 * t0 * dot_with_vec(i0, j0, k0, x0, y0, z0, seed);

	float t1 = 0.5F - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 < 0)
		t1 = 0;
	else
		t1 = t1 * t1 * t1 * t1 * dot_with_vec(i0 + i1, j0 + j1, k0 + k1, x1, y1, z1, seed);

	float t2 = 0.5F - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 < 0)
		t2 = 0;
	else
		t2 = t2 * t2 * t2 * t2 * dot_with_vec(i0 + i2, j0 + j2, k0 + k2, x2, y2, z2, seed);

	float t3 = 0.5F - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 < 0)
		t3 = 0;
	else
		t3 = t3 * t3 * t3 * t3 * dot_with_vec(i0 + 1.0F, j0 + 1.0F, k0 + 1.0F, x3, y3, z3, seed);

	return 76.0F * (t0 + t1 + t2 + t3);
}

constexpr float grad_x[12] = {  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 };
constexpr float grad_y[12] = {  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 };
constexpr float grad_z[12] = {  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 };

__forceinline __m256 dot_with_vec(__m256 _i, __m256 _j, __m256 _k, __m256 _x, __m256 _y, __m256 _z, __m256i _seed)
{
	const __m256i _hi = _mm256_mul_epi32(_mm256_castps_si256(_i), _mm256_set1_epi32(73856093));
	const __m256i _hj = _mm256_mul_epi32(_mm256_castps_si256(_j), _mm256_set1_epi32(19349663));
	const __m256i _hk = _mm256_mul_epi32(_mm256_castps_si256(_k), _mm256_set1_epi32(83492791));

	const __m256i _h_raw = _mm256_xor_si256(_hi, _mm256_xor_si256(_hj, _hk));

	const __m256i _h = _mm256_srli_epi32(_mm256_mul_epi32(_mm256_srli_epi32(_h_raw, 4), _mm256_set1_epi32(3)), 26);	//Normalize hash-value to [0, 11]

	const __m256 _gx = _mm256_i32gather_ps(grad_x, _h, 4);
	const __m256 _gy = _mm256_i32gather_ps(grad_y, _h, 4);
	const __m256 _gz = _mm256_i32gather_ps(grad_z, _h, 4);

	const __m256 _px = _mm256_mul_ps(_x, _gx);
	const __m256 _py = _mm256_mul_ps(_y, _gy);
	const __m256 _pz = _mm256_mul_ps(_z, _gz);

	return _mm256_add_ps(_px, _mm256_add_ps(_py, _pz));
}

void simplex_3d_fill(float* dst, float x_beg, float y_beg, float z_beg, float x_size, float y_size, float z_size, uint32_t x_cnt, uint32_t y_cnt, uint32_t z_cnt, uint32_t seed)
{
	constexpr float skew_factor = 1.0F / 3.0F;
	constexpr float unskew_factor = 1.0F / 6.0F;

	const __m256 _skew_factor = _mm256_set1_ps(skew_factor);

	const __m256 _unskew_factor = _mm256_set1_ps(unskew_factor);

	const float x_step = x_size / x_cnt;
	const float y_step = y_size / y_cnt;
	const float z_step = z_size / z_cnt;

	const __m256 _x_offsets = _mm256_set_ps(x_step*7, x_step*6, x_step*5, x_step*4, x_step*3, x_step*2, x_step, 0);		//Offsets in x-coordinate space

	const __m256i _seed = _mm256_set1_epi32(seed);

	for (uint32_t iz = 0; iz != z_cnt; ++iz)
	{
		const float z_in = z_beg + iz * z_step;

		for (uint32_t iy = 0; iy != y_cnt; ++iy)
		{
			const float y_in = y_beg + iy * y_step;

			for (uint32_t ix = 7; ix < x_cnt; ix += 8)
			{
				const float x_in0 = x_beg + ix * x_step;

				const __m256 _x_in0 = _mm256_set1_ps(x_in0);

				const __m256 _x_in = _mm256_add_ps(_x_in0, _x_offsets);

				const __m256 _y_in = _mm256_set1_ps(y_in);

				const __m256 _z_in = _mm256_set1_ps(z_in);

				const __m256 _skew_sum = _mm256_add_ps(_x_in, _mm256_add_ps(_y_in, _z_in));

				const __m256 _skew = _mm256_mul_ps(_skew_sum, _skew_factor);

				const __m256 _i0 = _mm256_floor_ps(_mm256_add_ps(_x_in, _skew));
				const __m256 _j0 = _mm256_floor_ps(_mm256_add_ps(_y_in, _skew));
				const __m256 _k0 = _mm256_floor_ps(_mm256_add_ps(_z_in, _skew));

				const __m256 _unskew_sum = _mm256_add_ps(_i0, _mm256_add_ps(_j0, _k0));

				const __m256 _unskew = _mm256_mul_ps(_unskew_sum, _unskew_factor);

				const __m256 _x_orig = _mm256_sub_ps(_i0, _unskew);
				const __m256 _y_orig = _mm256_sub_ps(_j0, _unskew);
				const __m256 _z_orig = _mm256_sub_ps(_k0, _unskew);

				const __m256 _x0 = _mm256_sub_ps(_x_in, _x_orig);
				const __m256 _y0 = _mm256_sub_ps(_y_in, _y_orig);
				const __m256 _z0 = _mm256_sub_ps(_z_in, _z_orig);

				const __m256 _one = _mm256_set1_ps(1.0F);

				const __m256 _x_ge_y = _mm256_cmp_ps(_x0, _y0, 29);
				const __m256 _x_ge_z = _mm256_cmp_ps(_x0, _z0, 29);
				const __m256 _y_ge_z = _mm256_cmp_ps(_y0, _z0, 29);

				const __m256 _i1 = _mm256_and_ps(   _mm256_and_ps(   _x_ge_y, _x_ge_z), _one);		//max == x
				const __m256 _j1 = _mm256_and_ps(   _mm256_andnot_ps(_x_ge_y, _y_ge_z), _one);		//max == y
				const __m256 _k1 = _mm256_andnot_ps(_mm256_or_ps(    _x_ge_z, _y_ge_z), _one);		//max == z

				const __m256 _i2 = _mm256_and_ps(   _mm256_or_ps(    _x_ge_y, _x_ge_z), _one);		//min != x
				const __m256 _j2 = _mm256_andnot_ps(_mm256_andnot_ps(_y_ge_z, _x_ge_y), _one);		//min != y
				const __m256 _k2 = _mm256_andnot_ps(_mm256_and_ps(   _x_ge_z, _y_ge_z), _one);		//min != z

				const __m256 _x1 = _mm256_add_ps(_mm256_sub_ps(_x0, _i1), _unskew_factor);
				const __m256 _y1 = _mm256_add_ps(_mm256_sub_ps(_y0, _j1), _unskew_factor);
				const __m256 _z1 = _mm256_add_ps(_mm256_sub_ps(_z0, _k1), _unskew_factor);

				const __m256 _two_unskew_factor = _mm256_add_ps(_unskew_factor, _unskew_factor);
				const __m256 _point_five = _mm256_add_ps(_two_unskew_factor, _unskew_factor);

				const __m256 _x2 = _mm256_add_ps(_mm256_sub_ps(_x0, _i2), _two_unskew_factor);
				const __m256 _y2 = _mm256_add_ps(_mm256_sub_ps(_y0, _j2), _two_unskew_factor);
				const __m256 _z2 = _mm256_add_ps(_mm256_sub_ps(_z0, _k2), _two_unskew_factor);

				const __m256 _x3 = _mm256_sub_ps(_x0, _point_five);
				const __m256 _y3 = _mm256_sub_ps(_y0, _point_five);
				const __m256 _z3 = _mm256_sub_ps(_z0, _point_five);

				const __m256 _square_sum0 = _mm256_add_ps(_mm256_mul_ps(_x0, _x0), _mm256_add_ps(_mm256_mul_ps(_y0, _y0), _mm256_mul_ps(_z0, _z0)));
				const __m256 _square_sum1 = _mm256_add_ps(_mm256_mul_ps(_x1, _x1), _mm256_add_ps(_mm256_mul_ps(_y1, _y1), _mm256_mul_ps(_z1, _z1)));
				const __m256 _square_sum2 = _mm256_add_ps(_mm256_mul_ps(_x2, _x2), _mm256_add_ps(_mm256_mul_ps(_y2, _y2), _mm256_mul_ps(_z2, _z2)));
				const __m256 _square_sum3 = _mm256_add_ps(_mm256_mul_ps(_x3, _x3), _mm256_add_ps(_mm256_mul_ps(_y3, _y3), _mm256_mul_ps(_z3, _z3)));

				const __m256 _t0 = _mm256_sub_ps(_point_five, _square_sum0);
				const __m256 _t1 = _mm256_sub_ps(_point_five, _square_sum1);
				const __m256 _t2 = _mm256_sub_ps(_point_five, _square_sum2);
				const __m256 _t3 = _mm256_sub_ps(_point_five, _square_sum3);

				const __m256 _neg0 = _mm256_cmp_ps(_t0, _mm256_setzero_ps(), 29);
				const __m256 _neg1 = _mm256_cmp_ps(_t1, _mm256_setzero_ps(), 29);
				const __m256 _neg2 = _mm256_cmp_ps(_t2, _mm256_setzero_ps(), 29);
				const __m256 _neg3 = _mm256_cmp_ps(_t3, _mm256_setzero_ps(), 29);

				const __m256 _n0 = _mm256_and_ps(_t0, _neg0);
				const __m256 _n1 = _mm256_and_ps(_t1, _neg1);
				const __m256 _n2 = _mm256_and_ps(_t2, _neg2);
				const __m256 _n3 = _mm256_and_ps(_t3, _neg3);

				const __m256 _n_squared0 = _mm256_mul_ps(_n0, _n0);
				const __m256 _n_squared1 = _mm256_mul_ps(_n1, _n1);
				const __m256 _n_squared2 = _mm256_mul_ps(_n2, _n2);
				const __m256 _n_squared3 = _mm256_mul_ps(_n3, _n3);

				const __m256 _n_cubed0 = _mm256_mul_ps(_n_squared0, _n_squared0);
				const __m256 _n_cubed1 = _mm256_mul_ps(_n_squared1, _n_squared1);
				const __m256 _n_cubed2 = _mm256_mul_ps(_n_squared2, _n_squared2);
				const __m256 _n_cubed3 = _mm256_mul_ps(_n_squared3, _n_squared3);

				const __m256 _abs_i1 = _mm256_add_ps(_i0, _i1);
				const __m256 _abs_j1 = _mm256_add_ps(_j0, _j1);
				const __m256 _abs_k1 = _mm256_add_ps(_k0, _k1);
				const __m256 _abs_i2 = _mm256_add_ps(_i0, _i2);
				const __m256 _abs_j2 = _mm256_add_ps(_j0, _j2);
				const __m256 _abs_k2 = _mm256_add_ps(_k0, _k2);
				const __m256 _abs_i3 = _mm256_add_ps(_i0, _one);
				const __m256 _abs_j3 = _mm256_add_ps(_j0, _one);
				const __m256 _abs_k3 = _mm256_add_ps(_k0, _one);

				const __m256 _r0 = _mm256_mul_ps(_n_cubed0, dot_with_vec(    _i0,     _j0,     _k0, _x0, _y0, _z0, _seed));
				const __m256 _r1 = _mm256_mul_ps(_n_cubed1, dot_with_vec(_abs_i1, _abs_j1, _abs_k1, _x1, _y1, _z1, _seed));
				const __m256 _r2 = _mm256_mul_ps(_n_cubed2, dot_with_vec(_abs_i2, _abs_j2, _abs_k2, _x2, _y2, _z2, _seed));
				const __m256 _r3 = _mm256_mul_ps(_n_cubed3, dot_with_vec(_abs_i3, _abs_j3, _abs_k3, _x3, _y3, _z3, _seed));

				const __m256 _thirty_two = _mm256_set1_ps(76.0F);

				const __m256 _r_sum = _mm256_add_ps(_mm256_add_ps(_r0, _r1), _mm256_add_ps(_r2, _r3));

				const __m256 _r = _mm256_sub_ps(_r_sum, _thirty_two);

				_mm256_store_ps(dst + ix + iy * x_cnt + iz * x_cnt * y_cnt, _r);
			}
		}
	}
}
