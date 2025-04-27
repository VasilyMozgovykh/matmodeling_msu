__constant double GravityConstant = 6.6743e-11;
__constant double AstronomicalUnit = 1.49597870700e11;
__constant double EarthMass = 5.9722e24;
__constant double SunMass = 1.98892e30;


double get_norm_array(
    const double* array,
    const int N
) {
    double norm = 0.;
    for (int i = 0; i < N; i++) {
        norm += (array[i]) * (array[i]);
    }
    return sqrt(norm);
}


void get_force(
    double *result,
    __global const double *r1,
    __global const double *r2,
    const double m2
) {
    for (int i = 0; i < 3; i++) {
        result[i] = r2[i] - r1[i];
    }
    double norm = get_norm_array(result, 3);
    for (int i = 0; i < 3; i++) {
        result[i] = result[i] * m2 * GravityConstant / (norm * norm * norm);
    }
}


__kernel void get_accelerations(
    __global double *result,
    __global const double *coords,
    __global const double *masses,
    const int N
) {
    int i = get_global_id(0);
    for (int k = 0; k < 3; k++) {
        result[i * 3 + k] = 0.;
    }
    double force[3];
    for (int j = 0; j < N; j++) {
        if (i == j) { continue; }
        get_force(force, coords + 3 * i, coords + 3 * j, masses[j]);
        for (int k = 0; k < 3; k++) {
            result[i * 3 + k] += force[k];
        }
    }
}


__kernel void update_coords(
    __global double *coords,
    __global const double *velocities,
    __global const double *accelerations,
    const double t_step
) {
    int i = get_global_id(0);
    coords[i] += velocities[i] * t_step + 0.5 * accelerations[i] * t_step * t_step;  
}

__kernel void update_velocities(
    __global double *velocities,
    __global const double *accelerations,
    __global const double *new_accelerations,
    const double t_step
) {
    int i = get_global_id(0);
    velocities[i] += 0.5 * (accelerations[i] + new_accelerations[i]) * t_step;
}
