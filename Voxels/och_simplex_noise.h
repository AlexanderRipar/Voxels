#pragma once

#include <cstdint>
#include <cmath>

float simplex_3d(float x_in, float y_in, float z_in, uint32_t seed = 0);

void simplex_3d_fill(float* dst, float x_beg, float y_beg, float z_beg, float x_size, float y_size, float z_size, uint32_t x_cnt, uint32_t y_cnt, uint32_t z_cnt, uint32_t seed = 0);