import jaxopt
import jax
import jax.numpy as jnp

import time

from problem import Problem
from methods import available_built_in_methods, check_method

# from benchmark_target import BenchmarkTarget
from metrics import *
from benchmark_result import BenchmarkResult


class Benchmark:
    """
    A class that provides the benchmarking of different optimization
    methods on a given problem (like Problem object).
    """

    problem: Problem = None  # Problem to solve
    #methods: list[dict[Method : dict[str:any]]] = None  # Methods for benchmarking
    methods: list[dict[str : dict[str:any]]] = None  # Methods for benchmarking
    available_built_in_methods: list[str] = None # method's keywords. 
    # If you want to call a method from the jaxopt, 
    # the name of the method must begin with one of these keywords.
    metrics: list[str] = None  # List of fields to include in BenchamrkResult

    def __init__(
        self,
        problem: Problem,
        methods: list[dict[str : dict[str:any]]],
        metrics: list[str],
    ) -> None:
        self.problem = problem
        methods_names = list()
        for item in methods:
            for name, params in item.items():
                methods_names.append(name)
        if not check_method(methods_names):
            exit(1)
        self.methods = methods
        self.available_built_in_methods = available_built_in_methods
        if not check_metric(metrics):
            exit(1)
        self.metrics = metrics

    def __run_solver(
        self, solver, x_init, metrics: list[str], *args, **kwargs
    ) -> dict[str, list[any]]:
        """
        A layer for pulling the necessary information according to metrics
        as the "method" solver works (solver like jaxopt.GradientDescent obj)
        """
        result = dict()
        start_time = time.time()
        state = solver.init_state(x_init, *args, **kwargs)
        sol = x_init
        x_prev = sol

        @jax.jit
        def jitted_update(sol, state):
            return solver.update(sol, state, *args, **kwargs)

        tol = None
        if 'tol' in kwargs:
            tol = kwargs['tol']

        for _ in range(solver.maxiter):
            """
            Add stop criterion!!!
            """
            x_prev = sol
            sol, state = jitted_update(sol, state)
            if "history_x" in metrics:
                if not "history_x" in result:
                    result["history_x"] = [sol]
                else:
                    result["history_x"].append(sol)
            if "history_f" in metrics:
                if not "history_f" in result:
                    result["history_f"] = [self.problem.f(sol)]
                else:
                    result["history_f"].append(self.problem.f(sol))
            if "nit" in metrics:
                if not "nit" in result:
                    result["nit"] = [1]
                else:
                    result["nit"][0] += 1
            if "nfev" in metrics:
                # IDK
                pass
            if "njev" in metrics:
                # IDK
                pass
            if "nhev" in metrics:
                # IDK
                pass
            if "errors" in metrics:
                if not "errors" in result:
                    result["errors"] = [state.error]
                else:
                    result["errors"].append(state.error)
        duration = time.time() - start_time
        if "time" in metrics:
            result["time"] = [duration]

        return result

    def run(self, user_method = None) -> BenchmarkResult:
        res = BenchmarkResult(problem=self.problem, methods=list(), metrics=self.metrics)
        data = dict()
        data[self.problem] = dict()
        # methods: list[dict[method(str) : dict[str:any]]]
        for item in self.methods:
            for method, params in item.items():
                # data: dict[Problem, dict[method(str), dict[str, list[any]]]]
                if method.startswith('GRADIENT_DESCENT'):
                    res.methods.append(method)
                    x_init = None
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    solver = jaxopt.GradientDescent(fun=self.problem.f, **params)
                    sub = self.__run_solver(solver=solver, x_init=x_init, metrics=self.metrics, **params)    
                    data[self.problem][method] = sub
                elif user_method is not None:
                    res.methods.append(method)
                    x_init = None
                    if 'x_init' in params:
                        x_init = params['x_init']
                        params.pop('x_init')
                    sub = self.__run_solver(solver=user_method, metrics=self.metrics, x_init=x_init, **params)
                    data[self.problem][method] = sub
        res.data = data
        return res


def test_local():
    from quadratic_problem import QuadraticProblem

    n = 2
    x_init = jnp.array([1.0, 1.0])
    problem = QuadraticProblem(n=n)
    benchamrk = Benchmark(
        problem=problem,
        methods=[
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'stepsize' : 1e-2
                }
            },
            {
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-2,
                    'maxiter': 11,
                    'stepsize' : lambda iter_num: 1 / (iter_num + 20)
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
        ],
    )
    result = benchamrk.run()
    result.save("GD_quadratic.json")


if __name__ == "__main__":
    test_local()
