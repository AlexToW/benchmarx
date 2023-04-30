import jax
import jaxopt
import jax.numpy as jnp
import os
import sys


from benchmark import Benchmark
from plotter import Plotter
from problems.quadratic_problem import QuadraticProblem


def run_experiment():
    n = 2
    key = jax.random.PRNGKey(758493)
    x_init = jax.random.uniform(key, shape=(n,))
    problem = QuadraticProblem(n=n)
    maxiter = 350
    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[
            {
            
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': maxiter,
                    'stepsize' : 1e-1,
                    'label': 'GD_const'
                },
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': maxiter,
                    'stepsize' : lambda iter_num:  1 / (iter_num + 9),
                    'label': 'GD_1/(k+9)'
                },
                
                'GRADIENT_DESCENT_armijo_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': maxiter,
                    'stepsize' : 0.01,
                    'linesearch': 'armijo',
                    'label': 'GD_armijo'
                },
                'GRADIENT_DESCENT_goldstein_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': maxiter,
                    'stepsize' : 0.01,
                    'linesearch': 'goldstein',
                    'label': 'GD_goldstein'
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
    result.save("./draft1/draft1.json")

def draw():
    plotter = Plotter(
        metrics= ['fs', 'xs_norm', 'fs_dist_to_opt', 'xs_dist_to_opt', 'grads_norm'],
        data_path='./draft1/draft1.json',
        dir_path='draft1'
    )
    plotter.plot()


def _main():
    run_experiment()
    draw()


if __name__ == '__main__':
    _main()