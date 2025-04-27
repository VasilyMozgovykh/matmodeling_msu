import numpy as np
import os
import pyopencl as cl

os.environ["PYOPENCL_CTX"] = ""

OPENCL_PROGRAM = os.path.join(os.path.dirname(os.path.realpath(__file__)), "opencl_python_build", "solve_verlet.cl")


def solve_verlet(
    coords: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    n_iterations: int = 100,
    t_step: float = 1e-1
) -> np.ndarray:
    """
        Генератор на каждом шаге метода Верле выдаёт координаты, скорости и ускорения N тел

        coords - радиус-векторы тел [N, 3]
        velocities - векторы скоростей [N, 3]
        masses - вектор масс [N]
        n_iterations - количество шагов по временной сетке
        t_step - шаг по временной сетке
    """
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, open(OPENCL_PROGRAM).read()).build()

    N = coords.shape[0]
    result = np.zeros(shape=(n_iterations, 3, N * 3), dtype=coords.dtype)

    coords = coords.reshape(-1).astype(np.float64)
    velocities = velocities.reshape(-1).astype(np.float64)
    masses = masses.reshape(-1).astype(np.float64)
    accelerations = np.zeros_like(coords)

    coords_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=coords)
    velocities_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=velocities)
    accelerations_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=accelerations)
    new_accelerations_g = cl.Buffer(ctx, mf.READ_WRITE, accelerations.nbytes)
    masses_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=masses)
    
    prg.get_accelerations(queue, (N,), None, accelerations_g, coords_g, masses_g, np.int32(N))
    cl.enqueue_copy(queue, accelerations, accelerations_g)
    for i in range(n_iterations):
        prg.update_coords(queue, coords.shape, None, coords_g, velocities_g, accelerations_g, np.float64(t_step))
        queue.finish()
        prg.get_accelerations(queue, (N,), None, new_accelerations_g, coords_g, masses_g, np.int32(N))
        queue.finish()
        prg.update_velocities(queue, velocities.shape, None, velocities_g, accelerations_g, new_accelerations_g, np.float64(t_step))
        queue.finish()
        cl.enqueue_copy(queue, accelerations_g, new_accelerations_g)
        cl.enqueue_copy(queue, result[i, 0], coords_g)
        cl.enqueue_copy(queue, result[i, 1], velocities_g)
        cl.enqueue_copy(queue, result[i, 2], accelerations_g)
    return result.reshape(n_iterations, 3, N, 3)
