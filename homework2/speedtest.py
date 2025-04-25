import matplotlib.pyplot as plt
import numpy as np
import time

from common import get_solar_system_params

from solvers.pure_python import solve_verlet as solve_python
from solvers.odeint_python import solve_verlet as solve_odeint
from solvers.numba_python import solve_verlet as solve_numba
from solvers.cython_python_build.cython_python import solve_verlet as solve_cython

from typing import Callable, Dict, List, Tuple
from tqdm.auto import tqdm


def perform_calculations(
    solver: Callable,
    n_objects: int = 100,
    n_iterations: int = 100,
    use_solar: bool = False
) -> Tuple[float, np.ndarray]:
    if use_solar:
        coords, velocities, masses = get_solar_system_params()
    else:
        coords = np.random.randn(n_objects, 3).astype(dtype=np.float64)
        velocities = np.random.randn(n_objects, 3).astype(dtype=np.float64)
        masses = np.random.randn(n_objects).astype(dtype=np.float64)
    ts_begin = time.time()
    res = solver(coords, velocities, masses, n_iterations=n_iterations, t_step=0.1)
    ts_end = time.time()
    return (ts_end - ts_begin, res)


def test_solver(
    config: Dict[str, Dict[str, List]],
    name: str,
    solver: Callable,
    n_objects: int = 100,
    n_iterations: int = 100,
    n_trials: int = 1,
    show_speedup=True,
) -> None:
    print(f"Testing {name} solution...")

    times = []
    for _ in tqdm(range(n_trials)):
        ts, _ = perform_calculations(solver, n_objects, n_iterations)
        times.append(ts)
    print(f"{name} average time: {np.mean(times)}")

    _, res = perform_calculations(solver, n_iterations=100, use_solar=True)
    config[name] = {"coords": res[:, 0], "times": times, "avg_time": np.mean(times)}

    if show_speedup and "Python" in config:
        speedup = config["Python"]["avg_time"] / config[name]["avg_time"]
        print(f"{name} speedup: {speedup:.4f} times", end="\n\n")
    else:
        print()


if __name__ == "__main__":
    n_iterations = 100
    for N in (100, 200, 400):
        config = {}
        print("======", f"N = {N}", "======")

        test_solver(config, "Python", solve_python, N, show_speedup=False)
        test_solver(config, "odeint", solve_odeint, N)
        test_solver(config, "Numba", solve_numba, N)
        test_solver(config, "Cython", solve_cython, N)

        fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
        fig.suptitle("Сравнение относительных погрешностей решения")

        coords_odeint = config["odeint"]["coords"]
        for name in config.keys():
            if name == "odeint":
                continue
            coords = config[name]["coords"]
            diff = np.linalg.norm(coords - coords_odeint, axis=(1, 2))
            rel_diff = diff / np.linalg.norm(coords_odeint, axis=(1, 2))
            ax.plot(range(n_iterations), rel_diff, label=name)
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_yscale("log")
        fig.savefig(f"comparison_plot_n={N}.png")
