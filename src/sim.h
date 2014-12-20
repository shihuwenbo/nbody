#include <stdlib.h>

#ifndef SIM
#define SIM

// simulation parameters
const float npart = 1000;
const float screen_size = 700.0;
const float screen_center = 350.0;
const float screen_scale = 700;
const float mass_scale = 1.0;
const float delta_t = 0.1;
const float grav_const = 6.67384e-11;

// initialize particles, position and velocity
void init(size_t n, float *part_pos, float *part_vel,
        float *part_acc, float *part_mas);

// update particles
void update(size_t n, float *part_pos, float *part_vel,
        float *part_acc, float *part_mas, float *part_force);

#endif
