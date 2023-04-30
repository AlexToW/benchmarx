import jax
import jaxopt
import jax.numpy as jnp
import os
import sys

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("src"), '.')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("src"), '..')))

from benchmark import Benchmark
from plotter import Plotter
from problems.quadratic_problem import QuadraticProblem



def _main():
    """
    ВЕРИФИКАЦИЯ - посмотреть 5 алгоритмов на одном 
    графике для квадрат. задачи - GD с двумя разными 
    learning rate + steepest, Armijo, Goldstein.
    """

    n = 5
    key = jax.random.PRNGKey(758493)
    x_init = jax.random.uniform(key, shape=(n,))
    problem = QuadraticProblem(n=n)
    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': 350,
                    'stepsize' : 1e-2,
                    'label': 'GD_const'
                },
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': 350,
                    'stepsize' : lambda iter_num:  1 / (iter_num + 9),
                    'label': 'GD_1/(k+9)'
                },
                'GRADIENT_DESCENT_armijo_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': 350,
                    'stepsize' : 0.1,
                    'linesearch': 'armijo',
                    'label': 'GD_armijo'
                },
                'GRADIENT_DESCENT_goldstein_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': 350,
                    'stepsize' : 0.1,
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
    '''
    my_solver = MyGradientDescent(
        x_init=x_init,
        stepsize=1e-2,
        problem=problem,
        tol=1e-3,
        maxiter=500,
        label='MyGradDescent'
    )
    '''
    #result = benchmark.run(user_method=my_solver)
    result = benchmark.run()
    result.save("draft1.json")



if __name__ == '__main__':
    _main()