import numpy as np
from common import PhysicsEnum
from scipy.integrate import odeint


def get_force(r1: np.ndarray, r2: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Возвращает напряжённость гравитационного поля, создаваемого вторым телом,
    действующую на первое тело

    r1, r2 - радиус-векторы первого и второго тел соответственно
    m2 - масса второго тела
    """
    diff = r2 - r1
    return PhysicsEnum.GravityConstant.value * m2 * diff / np.linalg.norm(diff)**3


def get_accelerations(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Возвращает ускорения N тел в замкнутой системе

    coords - радиус-векторы тел [N, 3]
    masses - вектор масс [N]
    """
    assert len(coords.shape) == 2 and coords.shape[1] == 3, "The size of coordinates matrix should be [N, 3]"
    assert len(masses.shape) == 1, "The size of masses matrix should be [N]"
    assert coords.shape[0] == masses.shape[0], "The number of points should be equal for coordinates and masses matrices"

    N = coords.shape[0]
    pairwise_forces = np.full(shape=(N, N, 3), fill_value=np.nan)
    for i in range(N):
        for j in range(N):
            if i != j:
                pairwise_forces[i, j] = get_force(coords[i], coords[j], masses[j])
    accelerations = np.nansum(pairwise_forces, axis=1)
    return accelerations


def solve_verlet(
    coords: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    n_iterations: int = 100,
    t_step: float = 1e-1
    ) -> np.ndarray:

    def func(y, t):
        N = y.shape[0] // 6
        coords = y[:3*N].reshape(N, 3)
        accelerations = get_accelerations(coords, masses).reshape(-1)
        res = np.zeros(6 * N, dtype=y.dtype)
        res[:3*N] = y[3*N:6*N]
        res[3*N:6*N] = accelerations
        return res

    N = coords.shape[0]
    y0 = np.concatenate([
        coords.reshape(3 * N),
        velocities.reshape(3 * N),
    ])
    result = odeint(func, y0, np.linspace(0., t_step * n_iterations, n_iterations+1)[1:])
    coords = result[:, :3*N].reshape(n_iterations, N, 3)
    velocities = result[:, 3*N:6*N].reshape(n_iterations, N, 3)
    res = np.zeros((n_iterations, 3, N, 3), dtype=coords.dtype)
    res[:, 0] = coords
    res[:, 1] = velocities
    return res
