import solvers.odeint_python as odeint_solver
import solvers.pure_python as python_solver
import solvers.numba_python as numba_solver
import solvers.cython_python as cython_solver
import solvers.opencl_python as opencl_solver
import solvers.multiprocessing_python as multiprocessing_solver

__all__ = [
    "python_solver",
    "odeint_solver",
    "cython_solver",
    "numba_solver",
    "opencl_solver",
    "multiprocessing_solver",
]
