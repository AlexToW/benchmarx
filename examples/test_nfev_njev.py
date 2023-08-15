import sys
import os
import jax
import jax.numpy as jnp
import jaxopt

import traceback
import logging
import functools
import time
import random

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))

from benchmarx.custom_optimizer import State
from benchmarx.metrics import CustomMetric
from typing import Any

from benchmarx.benchmark import Benchmark
from benchmarx.quadratic_problem import QuadraticProblem



def _main():

    start = time.time()
    my_l1_metric = CustomMetric(
        func=lambda x: jnp.linalg.norm(x, ord=1),
        label="l1-norm"
    )

    n = 2
    x_init = jnp.zeros(n)
    benchmark = Benchmark(
        problem=QuadraticProblem(n=n, ),
        runs=3,
        methods= [
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 0,
                    'maxiter': 10,
                    'stepsize' : 1e-1,
                    'acceleration': False,
                    'label': 'GD_const'
                }
            }
        ],
        metrics=[
            "x",
            "f",
            "grad",
            "nfev",
            "njev",
            my_l1_metric
        ]
    )
    result = benchmark.run()
    result.plot(
        metrics=["x_gap", "f", "f_gap", "grad_norm", "x_norm"]
    )
    result.save(
        path="test_nfev_njev_res.json"
    )

    print(benchmark.nfev_global)


if __name__ == "__main__":
    _main()