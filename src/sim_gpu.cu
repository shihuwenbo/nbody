#include "sim_gpu.h"

// update acceleration
extern "C++" void update_acc_gpu(size_t n, float *part_pos,
    float *part_vel, float *part_acc, float *part_mass,
    float grav_const) {

    // get num of threads and blocks needed
    size_t num_block = (n-1)/block_size+1;

    // compute grid dimension
    size_t num_grid_y = (num_block-1)/max_grid_x+1;
    size_t num_grid_x = num_block < max_grid_x ? num_block : max_grid_x;

    // launch kernel
    dim3 dimBlock(block_size, 1);
    dim3 dimGrid(num_grid_x, num_grid_y);
    update_acc_kernel<<<dimGrid,dimBlock>>>(n, part_pos,
        part_vel, part_acc, part_mass, grav_const);

    // check kernel result
    cudaThreadSynchronize();
    cudaError_t crc = cudaGetLastError();
    if(crc) {
        printf("emptyKernel error=%d:%s\n", crc, cudaGetErrorString(crc));
        exit(1);
    }
}

// kernel function for updating acceleration
__global__ void update_acc_kernel(size_t n, float *part_pos,
    float *part_vel, float *part_acc, float *part_mass,
    float grav_const) {
    
    // get absoluate idx of thread
    size_t i = threadIdx.x+blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);

    // check if j is within range
    if(i < n) {
        float xi = part_pos[2*i+0];
        float yi = part_pos[2*i+1];
        float mi = part_mass[i];

        part_acc[2*i+0] = 0.0;
        part_acc[2*i+1] = 0.0;

        // aggregate acceleration
        for(size_t j=0; j<n; j++) {
            
            float xj = part_pos[2*j+0];
            float yj = part_pos[2*j+1];
            float mj = part_mass[j];
            
            if(i != j) {
                float dx = xj-xi;
                float dy = yj-yi;
                float r = sqrt(dx*dx+dy*dy);
                
                // avoid infinite acceleration
                if(r < 1.0) {
                    r = 1.0;
                }

                // update acceleration
                float f = grav_const*mi*mj/(r*r);
                float fx = f*(dx/r);
                float fy = f*(dy/r);
                part_acc[2*i+0] += fx/mi;
                part_acc[2*i+1] += fy/mi;
            }
        }
    }
}
