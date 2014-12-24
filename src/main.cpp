#include <time.h>

#include "utils.h"
#include "sim.h"

int main() {

    // timing variable
    clock_t begin;
    clock_t end;
    float dt_ms;

    // timing information
    float tacc = 0.0;
    float tvel = 0.0;
    float tpos = 0.0;

    // simulation parameters
    const size_t npart = 8192;
    const size_t nsteps = 10;
    const float size_x = 1024.0;
    const float size_y = 512.0;
    const float center_x = size_x/2.0;
    const float center_y = size_y/2.0;
    const float scale_x = size_x;
    const float scale_y = size_y;
    const float delta_t = 1.0e3;
    const float scale_mass = 1000.0;
    const float grav_const = 6.67384e-11;

    // initialize particles
    float *part_pos = (float*) safe_calloc(2*npart, sizeof(float));
    float *part_vel = (float*) safe_calloc(2*npart, sizeof(float));
    float *part_acc = (float*) safe_calloc(2*npart, sizeof(float));
    float *part_mass = (float*) safe_calloc(npart, sizeof(float));
    init(npart, part_pos, part_vel, part_acc, part_mass,
            scale_x, scale_y, center_x, center_y, scale_mass);

    // run iterations
    for(size_t i=0; i<nsteps; i++) {

        // update particle - step 1 update acceleration
        begin = clock();
        update_acc(npart, part_pos, part_vel, part_acc,
                part_mass, grav_const);
        end = clock();
        dt_ms = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
        tacc += dt_ms;

        // update particle - step 2 update velocity
        begin = clock();
        update_vel(npart, part_vel, part_acc, delta_t);
        end = clock();
        dt_ms = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
        tvel += dt_ms;
        
        // update particle - step 3 update velocity
        begin = clock();
        update_pos(npart, part_pos, part_vel, delta_t);
        end = clock();
        dt_ms = ((float)(end-begin))/((float)(CLOCKS_PER_SEC)/1.0e9);
        tpos += dt_ms;
    }

    // print out timing information
    printf("update_acc: %fns/particle/step\n", tacc/npart/nsteps);
    printf("update_vel: %fns/particle/step\n", tvel/npart/nsteps);
    printf("update_pos: %fns/particle/step\n", tpos/npart/nsteps);
}
