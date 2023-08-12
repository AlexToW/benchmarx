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


class GD_proj(CustomOptimizer):
    """
    GD on standart simplex
    """
    def __init__(self, x_init, stepsize, problem, tol=0, maxiter=1000, label = 'GD_proj'):
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
    
    def proj(self, x):
        """
        Euclidian
        """
        x_sort = sorted(x, reverse=True)
        rho = 0
        s = x_sort[0]
        s_ans = s

        for i in range(1, len(x_sort)):
            s += x_sort[i]
            if x_sort[i] + 1 / (i + 1) * (1 - s) > 0:
                rho = i
                s_ans = s

        l = 1 / (rho + 1) * (1 - s_ans)
        ans = jnp.zeros(len(x_sort))
        for i in range(len(ans)):
            ans = ans.at[i].set(max(x[i] + l, 0))
        return ans

    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        Ax = self.problem.A @ sol
        sol = self.proj(sol - self.stepsize * Ax)
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
        info=f"QP"
    )


    key = jax.random.PRNGKey(110520)
    x_init = jax.random.uniform(key, minval=0, maxval=1, shape=(d,)) / d
    nit = 200

    md_solver = MirrorDescent(
        x_init=x_init,
        stepsize=1/L,
        problem=problem,
        tol=0,
        maxiter=nit,
        label='MD'
    )

    gd_solver = GD_proj(
        x_init=x_init,
        stepsize=1/L,
        problem=problem,
        tol=0,
        maxiter=nit,
        label='GD_proj'
    )

    gap = CustomMetric(
        func=lambda x: x.T @ problem.A @ x - jnp.min(problem.A @ x),
        label="main_gap"
    )
    benchmark = Benchmark(
        runs=1,
        problem=problem,
        methods=[{
            "MirrorDescent": md_solver
        },
        {
            'GradientDescent_proj': gd_solver 
        }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df",
            gap
        ],
    )

    result = benchmark.run()
    result.save('custom_method_data.json')

    plotter = Plotter(
        #metrics= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        #metrics=["fs", "xs_norm", "f_gap"],
        metrics=[],
        data_path="custom_method_data.json",
    )
    #plotter.plot()
    dfs_dict = plotter.json_to_dataframes(
        df_metrics=["Solution norm", "Distance to the optimum", "Primal gap", gap] # metrics from dataframe_metrics
    )
    for problem, df in dfs_dict.items():
        print(df.to_string())

    plotter.plot_plotly(
        metrics=["Solution norm", "Distance to the optimum", "Primal gap", gap],
        write_html=True,
        path_to_write="MD_vs_GD_simplex.html"
    )


if __name__ == "__main__":
    _main()