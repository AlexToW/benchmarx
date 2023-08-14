import sys
import os
import jax
import jax.numpy as jnp
import jaxopt
import time
import random

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

#from benchmarx import Benchmark, QuadraticProblem, Rastrigin, Rosenbrock, QuadraticProblemRealData, CustomOptimizer, Plotter
from benchmarx.custom_optimizer import State
from benchmarx.metrics import CustomMetric
from typing import Any

from benchmarx.benchmark_result import BenchmarkResult
from benchmarx.benchmark import Benchmark
from benchmarx.custom_optimizer import CustomOptimizer
from benchmarx._problems.log_regr import LogisticRegression


class SGD(CustomOptimizer):
    """
    SGD for LogLoss like f(w) = 1/n sum_{i=1}^n f_i(w)
    """
    def __init__(self, x_init, stepsize, problem, tol=0, maxiter=1000, label = 'SGD'):
        params = {
            'x_init': x_init,
            'tol': tol,
            'maxiter': maxiter,
            'stepsize': stepsize
        }
        self.stepsize = stepsize
        self.problem = problem
        self.maxiter = maxiter
        self.batch = 10
        self.tol = tol
        super().__init__(params=params, x_init=x_init, label=label)

    def init_state(self, x_init, *args, **kwargs) -> State:
        return State(
            iter_num=1,
            stepsize=self.stepsize
        )


    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        n = self.problem.n_train
        d = self.problem.d_train
        g = jnp.zeros(d)
        #random.seed(state.iter_num)
        indices = random.sample(
            population=list(range(n)),
            k=self.batch
        )
        for ind in indices:
            g += self.problem.grad_i(sol, ind)
        sol = sol - self.stepsize / self.batch * g
        state.iter_num += 1
        return sol, state
    
    def stop_criterion(self, sol, state: State) -> bool:
        return False
    


def _main():
    problem = LogisticRegression("mushrooms")

    train_acc_metric = CustomMetric(
        func= lambda w: problem.train_accuracy(w),
        label="train accuracy"
    )

    test_acc_metric = CustomMetric(
        func= lambda w: problem.test_accuracy(w),
        label="test accuracy"
    )

    key = jax.random.PRNGKey(110520)
    x_init = jax.random.uniform(key, minval=0, maxval=1, shape=(problem.d_train,))
    nit = 200

    sgd_solver = SGD(
        x_init=x_init,
        stepsize=4/5.25,
        problem=problem,
        tol=0,
        maxiter=nit,
        label='SGD'
    )

    benchmark = Benchmark(
        runs=4,
        problem=problem,
        methods=[{
            "SGD": sgd_solver
        },
        {
            'GRADIENT_DESCENT_const_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : 2/5.25,
                'acceleration': False,
                'label': 'GD'
            },
        }
        ],
        metrics=[
            "f",
            "grad"
        ],
    )

    result = benchmark.run()
    result.plot(
        metrics=["f", "grad_norm", train_acc_metric, test_acc_metric]
    )
    result.save(
        path="logreg_test_res.json"
    )


if __name__ == "__main__":
    _main()
    