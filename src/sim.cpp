#include "sim.h"
#include "utils.h"
#include <math.h>

// initialize particles, position and velocity
void init(size_t n, float *part_pos, float *part_vel,
        float *part_acc, float *part_mas) {

    // initialize uniform position 
    for(size_t i=0; i<n; i++) {
        part_pos[2*i+0] = randu()-0.5;
        part_pos[2*i+1] = randu()-0.5;
        part_pos[2*i+0] *= screen_scale;
        part_pos[2*i+1] *= screen_scale;
        part_pos[2*i+0] += screen_center;
        part_pos[2*i+1] += screen_center;
    }

    // initialize velocity to 0
    for(size_t i=0; i<n; i++) {
        part_vel[2*i+0] = 0.0;
        part_vel[2*i+1] = 0.0;
    }

    // initialize acceleration to 0
    for(size_t i=0; i<n; i++) {
        part_acc[2*i+0] = 0.0;
        part_acc[2*i+1] = 0.0;
    }

    // initialize mass to uniform
    for(size_t i=0; i<n; i++) {
        part_mas[i] = randu();
        part_mas[i] *= mass_scale;
    }
}

// update particles
void update(size_t n, float *part_pos, float *part_vel,
        float *part_acc, float *part_mas, float *part_force) {

    // compute gravity
    for(size_t i=0; i<n; i++) {
        
        float xi = part_pos[2*i+0];
        float yi = part_pos[2*i+1];
        float mi = part_mas[i];

        part_force[2*i+0] = 0.0;
        part_force[2*i+1] = 0.0;

        // aggregate gravity
        for(size_t j=0; j<n; j++) {
            
            float xj = part_pos[2*j+0];
            float yj = part_pos[2*j+1];
            float mj = part_mas[j];
            
            if(i != j) {
                float dx = xj-xi;
                float dy = yj-yi;
                float r = sqrt(dx*dx+dy*dy);
                
                // regularize r to avoid numeric instability
                if(r < 2.0) {
                    r = 2.0;
                }

                // update force
                float f = mi*mj/(r*r);
                float fx = f*(dx/r);
                float fy = f*(dy/r);
                part_force[2*i+0] += fx;
                part_force[2*i+1] += fy;
            }

        }
    }

    // compute acceleration
    for(size_t i=0; i<n; i++) {
        float mi = part_mas[i];
        part_acc[2*i+0] = part_force[2*i+0]/mi;
        part_acc[2*i+1] = part_force[2*i+1]/mi;
    }

    // update velocity
    for(size_t i=0; i<n; i++) {
        part_vel[2*i+0] += part_acc[2*i+0]*delta_t;
        part_vel[2*i+1] += part_acc[2*i+1]*delta_t;
    }

    // update position
    for(size_t i=0; i<n; i++) {
        part_pos[2*i+0] += part_vel[2*i+0]*delta_t;
        part_pos[2*i+1] += part_vel[2*i+1]*delta_t;
    }
}
