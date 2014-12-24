as of dec 23, 2014:
parallelized function for updating acceleration
- 1 thread per particle, simple update
- 31x speed up on t10 on 2048 particles
- 52x speed up on t10 on 4096 particles
- 75x speed up on t10 on 8192 particles
- 62x speed up on t10 on 16384 particles

as of dec 19, 2014:
simple 2d sequantial collission less n-body simulation code
- uniform spacial initialization with 0 initial velocity
- particle-particle method (n^2), leapfrog update
- requires sfml to compile and run
