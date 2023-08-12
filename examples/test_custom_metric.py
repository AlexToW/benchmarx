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
from benchmarx.src.metrics import CustomMetric



def _main():

    start = time.time()
    my_l1_metric = CustomMetric(
        func=lambda x: jnp.linalg.norm(x, ord=1),
        label="l1-norm",
        step=2
    )

    n = 2
    x_init = jnp.zeros(n)
    benchmark = Benchmark(
        problem=QuadraticProblem(n=n),
        runs=1,
        methods= [
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-9,
                    'maxiter': 10,
                    'stepsize' : 1e-1,
                    'acceleration': False,
                    'label': 'GD_const'
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
            "history_df",
            my_l1_metric
        ]
    )
    result = benchmark.run()
    print(f"benchmarking: {time.time() - start} sec")

    start = time.time()
    result.save('test_custom_metric_results.json')
    print(f"saving: {time.time() - start} sec")

    start = time.time()
    result.plot(
        metrics_to_plot= ['l1-norm'],
        data_path="test_custom_metric_results.json",
        dir_path='plots',
        fname_append='qp_real',
        show=True,
        log=True
    )
    print(f"plotting: {time.time() - start} sec")


if __name__ == "__main__":
    _main()