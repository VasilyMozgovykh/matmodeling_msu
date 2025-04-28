import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from common import get_solar_system_params

from functools import partial

from solvers import (
    python_solver,
    odeint_solver,
    numba_solver,
    cython_solver,
    opencl_solver,
    multiprocessing_solver,
)

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
    t_step = 60*60*240 if use_solar else 0.1
    res = solver(coords, velocities, masses, n_iterations=n_iterations, t_step=t_step)
    ts_end = time.time()
    return (ts_end - ts_begin, res)


def test_solver(
    config: Dict[str, Dict[str, List]],
    name: str,
    solver: Callable,
    n_objects: int = 100,
    n_iterations: int = 100,
    n_trials: int = 3,
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


def plot_avg_time(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
    fig.suptitle("Время работы программ в зависимости от числа тел")
    sns.lineplot(data, x="n_objects", y="avg_time", hue="name")
    ax.set_ylabel("Время работы, сек")
    ax.set_xlabel("Количество тел")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.savefig(f"visualisation_wo_opencl/avg_time_plot.png")


def plot_speedup(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
    fig.suptitle("Ускорение работы программ в зависимости от числа тел")
    sns.lineplot(data, x="n_objects", y="speedup", hue="name")
    ax.set_ylabel("Ускорение, во сколько раз")
    ax.set_xlabel("Количество тел")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.savefig(f"visualisation_wo_opencl/speedup_plot.png")


def plot_rel_diff(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
    fig.suptitle("Относительная погрешность в сравнении с Odeint")
    sns.lineplot(data, x="n_objects", y="rel_diff", hue="name")
    ax.set_ylabel("Относительная погрешность в евклидовой норме")
    ax.set_xlabel("Количество тел")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.savefig(f"visualisation_wo_opencl/rel_diff_plot.png")


if __name__ == "__main__":
    n_iterations = 100
    n_trials = 3
    data = {"name": [], "n_objects": [], "avg_time": [], "speedup": [], "rel_diff": []}
    for N in (100, 200, 400):
        config = {}
        test_fn = partial(test_solver, config=config, n_objects=N, n_trials=n_trials)
        print("======", f"N = {N}", "======")

        test_fn(name="Python", solver=python_solver.solve_verlet, show_speedup=False)
        test_fn(name="Odeint", solver=odeint_solver.solve_verlet)
        # test_fn(name="OpenCL", solver=opencl_solver.solve_verlet)
        test_fn(name="Cython", solver=cython_solver.solve_verlet)
        test_fn(name="Numba", solver=numba_solver.solve_verlet)
        test_fn(name="Multiprocessing", solver=multiprocessing_solver.solve_verlet)

        fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))
        fig.suptitle("Сравнение относительных погрешностей решения")

        coords_odeint = config["Odeint"]["coords"]
        for name in config.keys():
            data["name"].append(name)
            data["n_objects"].append(N)
            data["avg_time"].append(config[name]["avg_time"])
            data["speedup"].append(config["Python"]["avg_time"] / config[name]["avg_time"])
            if name == "Odeint":
                data["rel_diff"].append(None)
                continue
            coords = config[name]["coords"]
            diff = np.linalg.norm(coords - coords_odeint, axis=(1, 2))
            rel_diff = diff / np.linalg.norm(coords_odeint, axis=(1, 2))
            data["rel_diff"].append(rel_diff)
            ax.plot(range(n_iterations), rel_diff, label=name)
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_yscale("log")
        fig.savefig(f"visualisation_wo_opencl/comparison_plot_n={N}.png")
    plot_avg_time(data)
    plot_speedup(data)
    plot_rel_diff(data)
