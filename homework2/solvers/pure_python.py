import numpy as np
from itertools import product
from common import PhysicsEnum


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
    for i, j in product(range(N), range(N)):
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
    """
        Генератор на каждом шаге метода Верле выдаёт координаты, скорости и ускорения N тел

        coords - радиус-векторы тел [N, 3]
        velocities - векторы скоростей [N, 3]
        masses - вектор масс [N]
        n_iterations - количество шагов по временной сетке
        t_step - шаг по временной сетке
    """
    result = np.zeros(shape=(n_iterations, 3, coords.shape[0], 3), dtype=coords.dtype)
    accelerations = get_accelerations(coords, masses)
    for i in range(n_iterations):
        new_coords = coords + velocities * t_step + 0.5 * accelerations * t_step**2
        new_accelerations = get_accelerations(new_coords, masses)
        new_velocities = velocities + 0.5 * (new_accelerations + accelerations) * t_step

        result[i, 0] = new_coords
        result[i, 1] = new_velocities
        result[i, 2] = new_accelerations

        coords, velocities, accelerations = new_coords, new_velocities, new_accelerations
    return result
