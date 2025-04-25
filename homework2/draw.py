import argparse
import math
import matplotlib.pyplot as plt
import numpy as np

from common import PhysicsEnum, get_solar_system_params

from solvers.pure_python import solve_verlet as solve_python
from solvers.odeint_python import solve_verlet as solve_odeint
from solvers.numba_python import solve_verlet as solve_numba
from solvers.cython_python_build.cython_python import solve_verlet as solve_cython

from matplotlib.animation import FuncAnimation


SOLVERS = {
    "python": solve_python,
    "odeint": solve_odeint,
    "numba": solve_numba,
    "cython": solve_cython
}


def scale_function(R):
    return math.pow(R + 1e-12, 0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_scale", action="store_true", default=False)
    parser.add_argument("--num_days_for_animation", type=int, default=365)
    parser.add_argument("--num_days_for_frame", type=int, default=30)
    parser.add_argument("--max_tail_size", type=int, default=365)
    parser.add_argument("--solver", type=str, default="python")
    args = parser.parse_args()

    # Моделируем движение планет в Солнечной системе
    N = 10
    R, V, M, names = get_solar_system_params(return_names=True)

    X, Y, lines = [], [], []
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Движение небесных тел Солнечной системы из парада планет", fontsize=18)
    ax.set_xlabel("scaled x" if args.use_scale else "x", fontsize=14)
    ax.set_ylabel("scaled y" if args.use_scale else "y", fontsize=14)
    for i in range(N):
        if args.use_scale:
            X.append([scale_function(R[i, 0])])
            Y.append([scale_function(R[i, 1])])
        else:
            X.append([R[i, 0]])
            Y.append([R[i, 1]])
        line = ax.plot(X[i][0], Y[i][0], label=names[i])[0]
        lines.append(line)

    plot_radius = 45. * PhysicsEnum.AstronomicalUnit.value
    if args.use_scale:
        plot_radius = scale_function(plot_radius)
    ax.set_xlim([-plot_radius, plot_radius])
    ax.set_ylim([-plot_radius, plot_radius])
    ax.legend(loc="upper right")
    
    solve_verlet = SOLVERS[args.solver]
    for R, *_ in solve_verlet(R, V, M, n_iterations=args.num_days_for_animation, t_step=60*60*24):
        for i in range(N):
            if args.use_scale:
                distance = float(np.linalg.norm(R[i, :2]))
                scale_coef = scale_function(distance) / distance
                X[i].append(scale_coef * R[i, 0])
                Y[i].append(scale_coef * R[i, 1])
            else:
                X[i].append(R[i, 0])
                Y[i].append(R[i, 1])

    def update(frame):
        tail = max(0, frame - args.max_tail_size)
        for i in range(N):
            lines[i].set_xdata(X[i][tail:frame])
            lines[i].set_ydata(Y[i][tail:frame])
        return lines

    prefix = "solar_system_scaled" if args.use_scale else "solar_system"
    path = "_".join([
        prefix,
        f"num_days_for_animation={args.num_days_for_animation}",
        f"num_days_for_frame={args.num_days_for_frame}",
        f"max_tail_size={args.max_tail_size}",
        f"solver={args.solver}",
    ])
    ani = FuncAnimation(fig=fig, func=update, frames=args.num_days_for_animation)
    ani.save(f"{path}.gif", fps=args.num_days_for_frame)
