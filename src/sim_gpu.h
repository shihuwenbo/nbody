#include <stdio.h>
#include <cuda.h>
#include "utils.h"

size_t block_size = 64;
size_t max_grid_x = 65535;

// update acceleration
__global__ void update_acc_kernel(size_t n, float *part_pos,
    float *part_vel, float *part_acc, float *part_mass,
    float grav_const);
