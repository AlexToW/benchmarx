import jax
import jaxopt
import jax.numpy as jnp
import os
import sys
import logging


from benchmark import Benchmark
from plotter import Plotter
from problems.quadratic_problem import QuadraticProblem
from custom_optimizer import CustomOptimizer




def run_experiment():
    n = 2
    key = jax.random.PRNGKey(758493)
    x_init = jax.random.uniform(key, shape=(n,))
    problem = QuadraticProblem(n=n)
    maxiter = 50
    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[
            {
            
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'stepsize' : 1e-1,
                    'acceleration': False,
                    'label': 'GD_const'
                },
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'stepsize' : lambda iter_num:  1 / (iter_num + 9),
                    'acceleration': False,
                    'label': 'GD_1/(k+9)'
                },
                
                'GRADIENT_DESCENT_armijo_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'stepsize' : 0.01,
                    'linesearch': 'backtracking',
                    'condition': 'armijo',
                    'acceleration': False,
                    'label': 'GD_armijo'
                },
                'GRADIENT_DESCENT_goldstein_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'stepsize' : 0.01,
                    'linesearch': 'backtracking',
                    'condition': 'goldstein',
                    'acceleration': False,
                    'label': 'GD_goldstein'
                },
                'ArmijoSGD': {
                    'x_init': x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'label': 'Armijo SGD'
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


def run_experiment_small():
    n = 6
    key = jax.random.PRNGKey(758493)
    x_init = jax.random.uniform(key, shape=(n,))
    problem = QuadraticProblem(n=n)
    L = float(jnp.max(jnp.real(jnp.linalg.eigh(problem.A)[0])))
    maxiter = 250
    benchmark = Benchmark(
        runs=1000,
        problem=problem,
        methods=[
            {
                'ArmijoSGD': {
                    'x_init': x_init,
                    'tol': 1e-9,
                    'maxiter': maxiter,
                    'label': 'Armijo SGD'
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
        metrics= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='./draft1/draft1.json',
        dir_path='draft1'
    )
    plotter.plot(log=True)
    #plotter.plot()


def _main():
    logging.getLogger().setLevel(logging.INFO)
    #run_experiment_small()
    draw()


if __name__ == '__main__':
    _main()
