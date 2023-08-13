from context import benchmarx

from benchmarx.problems import LinearLeastSquares
from benchmarx import Benchmark
from jax import numpy as jnp

problem = LinearLeastSquares("random", m=2, n=2)
x_init = jnp.zeros(2)

benchmark = Benchmark(
        problem=problem,
        runs=1,
        methods= [
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'maxiter': 1000,
                    'stepsize' : 1e-2,
                    'acceleration': False,
                    'label': 'GD_const'
                }
            }
        ],
        metrics=[
            "x",
            "f",
            "grad",
            "nit",
            "nfev",
            "njev",
            "nhev",
            "time"
        ]
    )
result = benchmark.run()
result.plot(
        metrics=[
            "x_gap", 
            "f", 
            "f_gap", 
            "grad_norm", 
            "x_norm",
            
                ]
    )