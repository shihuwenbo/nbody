#include <stdio.h>
#include <cuda.h>
#include "utils.h"

size_t block_size = 128;
size_t max_grid_x = 65535;

// update acceleration
__global__ void update_acc_kernel(size_t n, float *part_pos,
    float *part_vel, float *part_acc, float *part_mass,
    float grav_const);

// update acceleration
__global__ void update_acc_tile_kernel(size_t n, float *part_pos,
    float *part_vel, float *part_acc, float *part_mass, float grav_const,
    size_t tile_size, size_t num_tiles_per_block, size_t num_tiles);
