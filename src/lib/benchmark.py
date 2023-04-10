import jaxopt
import jax
import jax.numpy as jnp

import time

from problem import Problem
from methods import Method
from benchmark_target import BenchmarkTarget
from benchmark_result import BenchmarkResult


class Benchmark:
    """
    A class that provides the benchmarking of different optimization 
    methods on a given problem (like Problem object).
    """
    problem: Problem = None                         # Problem to solve
    methods: list[Method] = None                   # Methods for benchmarking
    result_params: list[BenchmarkTarget] = None     # List of fields to include in BenchamrkResult

    def __init__(self, problem: Problem, methods: list[Method], result_params: list[BenchmarkTarget]) -> None:
        self.problem = problem
        self.methods = methods
        self.result_params = result_params

    def __run_solver(self, solver, x_init, result_params: list[BenchmarkTarget], 
                    *args, **kwargs) -> dict[BenchmarkTarget, list[any]]:
        """
        Прослойка для вытягивания необходимой информации согласно result_params
        по ходу работы "метода" solver (solver like jaxopt.GradientDescent obj)
        """
        result = dict()
        start_time = time.time()
        state = solver.init_state(x_init, *args, **kwargs)
        sol = x_init

        @jax.jit
        def jitted_update(sol, state):
            return solver.update(sol, state, *args, **kwargs)

        for _ in range(solver.maxiter):
            sol, state = jitted_update(sol, state)
            if BenchmarkTarget.trajectory_x in result_params:
                if not BenchmarkTarget.trajectory_x in result:
                    result[BenchmarkTarget.trajectory_x] = [sol]
                else:
                    result[BenchmarkTarget.trajectory_x].append(sol)
            if BenchmarkTarget.trajectory_f in result_params:
                if not BenchmarkTarget.trajectory_f in result:
                    result[BenchmarkTarget.trajectory_f] = [self.problem.f(sol)]
                else:
                    result[BenchmarkTarget.trajectory_f].append(self.problem.f(sol))
            if BenchmarkTarget.trajectory_df in result_params:
                grad = jax.grad(f=self.problem.f)
                if not BenchmarkTarget.trajectory_df in result:
                    result[BenchmarkTarget.trajectory_x] = [grad(sol)]
                else:
                    result[BenchmarkTarget.trajectory_df].append(grad(sol))
            if BenchmarkTarget.nit in result_params:
                if not BenchmarkTarget.nit in result:
                    result[BenchmarkTarget.nit] = [1]
                else:
                    result[BenchmarkTarget.nit][0] += 1
            if BenchmarkTarget.nfev in result_params:
                # IDK
                pass
            if BenchmarkTarget.njev in result_params:
                # IDK
                pass
            if BenchmarkTarget.nhev in result_params:
                # IDK
                pass
            if BenchmarkTarget.errors in result_params:
                if not BenchmarkTarget.errors in result:
                    result[BenchmarkTarget.errors] = [state.error]
                else:
                    result[BenchmarkTarget.errors].append(state.error)
        duration = time.time() - start_time
        if BenchmarkTarget.time in result_params:
            result[BenchmarkTarget.time] = [duration]
        
        return result

    def run(self, x_init, *args, **kwargs) -> BenchmarkResult:
        res = BenchmarkResult(
            problem=self.problem,
            methods=self.methods,
            keys=self.result_params
        )
        data = dict()
        for method in self.methods:
            # data: dict[Method, dict[Problem, dict[BenchmarkTarget, list[any]]]] = None
            if method == Method.GRADIENT_DESCENT:
                solver = jaxopt.GradientDescent(fun=self.problem.f, *args, **kwargs)
                sub = self.__run_solver(
                    solver=solver,
                    x_init=x_init,
                    result_params=self.result_params,
                    args=args,
                    kwargs=kwargs
                )
                data[Method.GRADIENT_DESCENT] = {self.problem : sub}
        res.data = data
        return res


from quadratic_problem import QuadraticProblem

def test_local():
    n = 2
    x_init = jnp.array([1., 1.])
    problem = QuadraticProblem(n=n)
    benchamrk = Benchmark(
        problem=problem,
        methods=[Method.GRADIENT_DESCENT],
        result_params=[BenchmarkTarget.nit, BenchmarkTarget.trajectory_x]
    )
    result = benchamrk.run(x_init=x_init, tol=1e-5)
    result.save('GD_quadratic.json')

test_local()