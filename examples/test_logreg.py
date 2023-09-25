import sys
import os
import jax
import jax.numpy as jnp
import jaxopt
import time
import random
from math import sqrt

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

#from benchmarx import Benchmark, QuadraticProblem, Rastrigin, Rosenbrock, QuadraticProblemRealData, CustomOptimizer, Plotter
from benchmarx.custom_optimizer import State
from benchmarx.metrics import CustomMetric
from typing import Any

from benchmarx.benchmark_result import BenchmarkResult
from benchmarx.benchmark import Benchmark
from benchmarx.custom_optimizer import CustomOptimizer
from benchmarx._problems.log_regr import LogisticRegression


class CSGD(CustomOptimizer):
    """
    CSGD for LogLoss
    """
    def __init__(self, x_init, stepsize, problem, tol=0, maxiter=1000, label = 'CSGD'):
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
        
        full_grad = jax.grad(self.problem.f)(sol)

        indices = random.sample(
            population=list(range(d)),
            k=self.batch
        )
        g = jnp.zeros(d)
        for ind in indices:
            #print(type(full_grad.at[ind]), full_grad.at[ind])
            g = g.at[ind].set(full_grad[ind])
        sol = sol - self.stepsize * d / self.batch * g
        state.iter_num += 1
        return sol, state
    
    def stop_criterion(self, sol, state: State) -> bool:
        return False


class SGD(CustomOptimizer):
    """
    SGD for LogLoss
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
        self.batch = 1
        self.tol = tol
        super().__init__(params=params, x_init=x_init, label=label)

    def init_state(self, x_init, *args, **kwargs) -> State:
        return State(
            iter_num=1,
            stepsize=self.stepsize
        )


    def update(self, sol, state: State) -> tuple([jnp.array, State]):
        n = self.problem.y_train.shape[0] // 10
        d = self.problem.d_train
        indices = random.sample(
            population=list(range(n)),
            k=self.batch
        )
        g = jnp.zeros(d)
        for ind in indices:
            g += self.problem.grad_log_loss_ind(sol, ind)
        sol = sol - self.stepsize / self.batch * g
        state.iter_num += 1
        return sol, state
    
    def stop_criterion(self, sol, state: State) -> bool:
        return False
    

def logreg_mushrooms():
    problem = LogisticRegression(
        info="Logistic Regression problem on mushrooms dataset, l2-regularization",
        train_data_parts=10,
        problem_type="mushrooms"
    )

    L = problem.estimate_L()

    # l2 regularizer
    regularizer = lambda w: L/2500*jnp.linalg.norm(w, ord=2)**2
    problem.regularizer = regularizer

    key = jax.random.PRNGKey(110520)
    x_init = jax.random.uniform(key, minval=0, maxval=1, shape=(problem.d_train,))
    nit = 250

    csgd_solver = CSGD(
        x_init=x_init,
        stepsize=20/L,
        problem=problem,
        tol=0,
        maxiter=nit,
        label="CSGD"
    )

    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[{
            "CSGD": csgd_solver
        },
        {
            'GRADIENT_DESCENT_const_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : 20/L,
                'acceleration': False,
                'label': 'GD'
            },
        },
        {
            'GRADIENT_DESCENT_adapt_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : lambda iter_num: 20/(L + iter_num/20),
                'acceleration': False,
                'label': 'GD adapt step'
            },
        }
        ],
        metrics=[
            "f",
        ],
    )

    result = benchmark.run()
    result.plot(
        write_html=True,
        path_to_write="logreg_plot.html"
    )
    result.save(
        path="logreg_test_res.json"
    )


def logreg_cancer():
    problem = LogisticRegression("breast_cancer")

    L = problem.estimate_L()

    key = jax.random.PRNGKey(110520)
    x_init = jax.random.uniform(key, minval=0, maxval=1, shape=(problem.d_train,))
    nit = 250


    csgd_solver = CSGD(
        x_init=x_init,
        stepsize=34/L,
        problem=problem,
        tol=0,
        maxiter=nit,
        label="CSGD"
    )

    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[{
            "CSGD": csgd_solver
        },
        {
            'GRADIENT_DESCENT_const_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : 20/L,
                'acceleration': False,
                'label': 'GD'
            },
        },
        {
            'GRADIENT_DESCENT_adapt_step': {
                'x_init' : x_init,
                'tol': 0,
                'maxiter': nit,
                'stepsize' : lambda iter_num: 20/(L + iter_num/20),
                'acceleration': False,
                'label': 'GD adapt step'
            },
        }
        ],
        metrics=[
            "f",
        ],
    )

    result = benchmark.run()
    result.plot(
        write_html=True,
        path_to_write="logreg_plot.html"
    )
    result.save(
        path="logreg_test_res.json"
    )

def _main():
    #logreg_cancer()
    logreg_mushrooms()


if __name__ == "__main__":
    _main()
    