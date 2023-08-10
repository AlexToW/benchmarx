import sys
import os
import jax
import jax.numpy as jnp
import jaxopt

import traceback
import logging
import functools
import time

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

from benchmarx import Benchmark, QuadraticProblem, Rastrigin, Rosenbrock, QuadraticProblemRealData, CustomOptimizer, Plotter
from benchmarx.src.custom_optimizer import State
from benchmarx.src.metrics import CustomMetric
from typing import Any


class MirrorDescent(CustomOptimizer):
    def __init__(self, x_init, stepsize, problem, tol=0, maxiter=1000, label = 'MD'):
        params = {
            'x_init': x_init,
            'tol': tol,
            'maxiter': maxiter,
            'stepsize': stepsize
        }
        self.stepsize = stepsize
        self.problem = problem
        self.maxiter = maxiter
        self.tol = tol
        super().__init__(params=params, x_init=x_init, label=label)

    def init_state(self, x_init, *args, **kwargs) -> State:
        return State(
            iter_num=1,
            stepsize=self.stepsize
        )
    
    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        Ax = self.problem.A @ sol
        y = [sol[i] * jnp.exp(-state.stepsize * Ax[i]) for i in range(self.problem.n)]
        sol = jnp.array(y) / sum(y)
        state.iter_num += 1
        return sol, state
    
    def stop_criterion(self, sol, state: State) -> bool:
        return False
    

def _main():
    L = 1000
    mu = 1
    d = 2

    problem = QuadraticProblem(
        n=d,
        b=jnp.zeros(d),
        mineig=mu,
        maxeig=L,
        info=f"QP, $\\mu=${mu}, $L=${L}"
    )


    key = jax.random.PRNGKey(110500)
    x_init = jax.random.uniform(key, minval=0, maxval=1, shape=(d,))
    nit = 3

    md_solver = MirrorDescent(
        x_init=x_init,
        stepsize=1/L,
        problem=problem,
        tol=0,
        maxiter=nit,
        label='MD'
    )

    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[{
            "MirrorDescent": md_solver
        },
        {
            'GRADIENT_DESCENT_const_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : 1/L,
                'acceleration': False,
                'label': 'GD'
            }
        }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ],
    )

    result = benchmark.run()
    result.save('custom_method_data.json')

    plotter = Plotter(
        # metrics= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        metrics=["fs", "xs_norm", "f_gap"],
        data_path="custom_method_data.json",
    )
    #plotter.plot()
    l1_norm = CustomMetric(
        func=lambda x: float(jnp.linalg.norm(x, ord=1)),
        label="l1-norm"
    )
    #dfs_dict = plotter.json_to_dataframes(
    #    df_metrics=["Solution norm", "Distance to the optimum", "Primal gap", l1_norm]   # metrics from dataframe_metrics
    #)
    #for problem, df in dfs_dict.items():
    #    print(df.to_string())

    plotter.plot_plotly(
        metrics=["Solution norm", "Distance to the optimum", "Primal gap"]
    )


if __name__ == "__main__":
    _main()