cimport cython
cimport libc.math as cmath
import numpy as np
from cython.view cimport array as cvarray


GravityConstant = 6.6743e-11
AstronomicalUnit = 1.49597870700e11
EarthMass = 5.9722e24
SunMass = 1.98892e30


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double get_norm_array(double[:] array):
    cdef Py_ssize_t N = array.shape[0]
    cdef double norm = 0.
    for i in range(N):
        norm += array[i] * array[i]
    return cmath.sqrt(norm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] get_force(double [:] r1, double[:] r2, double m2):
    cdef double[3] force
    for i in range(3):
        force[i] = r2[i] - r1[i]

    cdef double norm = get_norm_array(force)
    for i in range(3):
        force[i] = force[i] * m2 * GravityConstant / (norm * norm * norm)
    return force


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] get_accelerations(double[:, :] coords, double[:] masses):
    cdef Py_ssize_t N = coords.shape[0]
    cdef double[:, :] accelerations = cvarray(shape=(N, 3), itemsize=sizeof(double), format="d")
    cdef double[:] force
    for i in range(N):
        for j in range(N):
            if i != j:
                force = get_force(coords[i], coords[j], masses[j])
                for k in range(3):
                    accelerations[i, k] += force[k]
    return accelerations


@cython.boundscheck(False)
@cython.wraparound(False)
def solve_verlet(
    double[:, :] coords,
    double[:, :] velocities,
    double[:] masses,
    int n_iterations = 100,
    double t_step = 0.1
):
    cdef Py_ssize_t N = coords.shape[0]
    cdef double[:, :, :, :] result = cvarray(shape=(n_iterations, 3, N, 3), itemsize=sizeof(double), format="d")
    cdef double[:, :] accelerations = get_accelerations(coords, masses)
    cdef double[:, :] new_coords = np.zeros_like(coords)
    cdef double[:, :] new_velocities = np.zeros_like(velocities)
    cdef double[:, :] new_accelerations = np.zeros_like(accelerations)
    for i in range(n_iterations):
        for j in range(N):
            for k in range(3):
                new_coords[j, k] = coords[j, k] + velocities[j, k] * t_step + 0.5 * accelerations[j, k] * t_step * t_step
                result[i, 0, j, k] = new_coords[j, k]
        new_accelerations = get_accelerations(new_coords, masses)
        for j in range(N):
            for k in range(3):
                new_velocities[j, k] = velocities[j, k] + 0.5 * (new_accelerations[j, k] + accelerations[j, k]) * t_step
                result[i, 1, j, k] = new_velocities[j, k]
                result[i, 2, j, k] = new_accelerations[j, k]
        for j in range(N):
            for k in range(3):
                coords[j, k] = new_coords[j, k]
                velocities[j, k] = new_velocities[j, k]
                accelerations[j, k] = new_accelerations[j, k]
    return np.asarray(result)
