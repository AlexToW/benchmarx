import sys
import os
import jax.numpy as jnp
import jaxopt

import traceback
import logging
import functools
import time

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

from benchmarx import Benchmark, QuadraticProblem, Rastrigin, Rosenbrock, QuadraticProblemRealData

test_num = 0
tests_passed = 0

def tester(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        global test_num, tests_passed
        test_num += 1
        logging.info(f'Run test: {test_num}')
        start = time.time()
        value = None
        try:
            value = func()
            end = time.time()
            logging.info(f'Test {test_num} passed. Duration: {round(end - start, 3)} (sec).')
            tests_passed += 1
        except Exception as e:
            logging.error(traceback.format_exc())
            end = time.time()
            logging.info(f'Duration: {round(end - start, 3)} (sec).')
        return value
    return wrapper_decorator

@tester
def test1():
    n = 7
    x_init = jnp.zeros(n)
    benchmark = Benchmark(
        problem=QuadraticProblem(n=7),
        runs=2,
        methods= [
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 1e-1,
                    'acceleration': False,
                    'label': 'GD_const'
                },
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : lambda iter_num:  1 / (iter_num + 9),
                    'acceleration': False,
                    'label': 'GD_1/(k+9)'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        # здесь происходит следующее: объект result сначала сохраняется в json файл,
        # а уже из него Plotter строит что надо.
        data_path='test_1_data.json',
        dir_path='plots',
        fname_append='test1'
    )

@tester
def test2():
    x_init = jnp.zeros(7)
    benchmark = Benchmark(
        problem=QuadraticProblem(n=7),
        runs=2,
        methods= [
            {
                'GRADIENT_DESCENT_armijo_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 0.01,
                    'linesearch': 'backtracking',
                    'condition': 'armijo',
                    'acceleration': False,
                    'label': 'GD_armijo'
                },
                'GRADIENT_DESCENT_goldstein_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 0.01,
                    'linesearch': 'backtracking',
                    'condition': 'goldstein',
                    'acceleration': False,
                    'label': 'GD_goldstein'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_2_data.json',
        dir_path='plots',
        fname_append='test2'
    )

@tester
def test3():
    x_init = jnp.zeros(2)
    benchmark = Benchmark(
        problem=QuadraticProblem(n=2),
        methods= [
            {
                'ArmijoSGD': {
                    'x_init': x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'label': 'Armijo SGD'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_3_data.json',
        dir_path='plots',
        fname_append='test3'
    )

@tester
def test4():
    problem = QuadraticProblem(n=7)
    x_init = jnp.zeros(7)
    ls_armijo = jaxopt.BacktrackingLineSearch(fun=problem.f, maxiter=20, condition="armijo",
                            decrease_factor=0.8)
    ls_strong_wolfe = jaxopt.BacktrackingLineSearch(fun=problem.f, maxiter=20, condition="strong-wolfe",
                            decrease_factor=0.8)
    benchmark = Benchmark(
        problem=problem,
        runs=3,
        methods= [
            {
                'GRADIENT_DESCENT_ls_obj_armijo': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 1e-1,
                    'linesearch': ls_armijo,
                    'acceleration': False,
                    'label': 'GD_ls_obj_armijo'
                },
                'GRADIENT_DESCENT_ls_obj_strong_wolfe': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 1e-1,
                    'linesearch': ls_strong_wolfe,
                    'acceleration': False,
                    'label': 'GD_ls_obj_strong_wolfe'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_4_data.json',
        dir_path='plots',
        fname_append='test4'
    )

@tester
def test5():
    n = 4
    problem = Rastrigin(n=n)
    x_init = 0.1 * jnp.ones(n)
    benchmark = Benchmark(
        problem=problem,
        methods= [
            {
                'GRADIENT_DESCENT_adapt': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : lambda iter_num: 0.1 / (iter_num + 10),
                    'acceleration': False,
                    'label': 'GD_0.1/(k+9)'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_5_data.json',
        dir_path='plots',
        fname_append='test5'
    )

@tester
def test6():
    n = 2
    problem = Rosenbrock(n=n)
    x_init = jnp.array([1.1, 1.])
    benchmark = Benchmark(
        problem=problem,
        methods= [
            {
                'GRADIENT_DESCENT_const': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 1e-5,
                    'acceleration': False,
                    'label': 'GD_1e-5'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_6_data.json',
        dir_path='plots',
        fname_append='test6'
    )

@tester
def test7():
    problem = QuadraticProblemRealData()
    L = float(jnp.linalg.det(problem.A))
    stepsize = 1 / L
    x_init = 0.1 * problem.x_opt
    benchmark = Benchmark(
        problem=problem,
        methods= [
            {
                'GRADIENT_DESCENT_adapt': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 500,
                    'stepsize' : lambda iter_num: stepsize / (iter_num + 10),
                    'acceleration': False,
                    'label': 'GD_1/L(k+9)'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_7_data.json',
        dir_path='plots',
        fname_append='test7'
    )

@tester
def test8():
    n = 4
    problem = Rastrigin(n=n)
    x_init = jnp.ones(n)
    benchmark = Benchmark(
        problem=problem,
        methods= [
            {
                'GRADIENT_DESCENT_adapt': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 100,
                    'stepsize' : 1e-2,
                    'acceleration': False,
                    'label': 'GD_1/(k+9)'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df"
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics_to_plot= ['fs', 'xs_norm', 'f_gap', 'x_gap', 'grads_norm'],
        data_path='test_8_data.json',
        dir_path='plots',
        fname_append='test8'
    )

def tests():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()

    total_test_num = 8
    logging.info(f' {tests_passed} / {total_test_num} tests passed.')


def _main():
    tests()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _main()