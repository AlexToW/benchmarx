import jax
import numpy as np


def run_solver(solver, w_init, *args, **kwargs):
    state = solver.init_state(w_init, *args, **kwargs)
    sol = w_init
    trajectory, errors = [], []

    @jax.jit
    def jitted_update(sol, state):
        return solver.update(sol, state, *args, **kwargs)

    for _ in range(solver.maxiter):
        sol, state = jitted_update(sol, state)
        trajectory.append(sol)
        errors.append(state.error)

    return np.array(trajectory), np.array(errors)
