#pragma once

#include <cstdint>

void launch_voxels(uint32_t dim_log2, float noise_limit, uint32_t noise_seed);

void get_slice(uint8_t* dst, uint32_t idx, uint32_t z);