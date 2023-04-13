# My method example/test
from jax import grad
import jax.numpy as jnp


from quadratic_problem import QuadraticProblem
from benchmark import Benchmark


class MyGradientDescent:
    def __init__(self, fun, x_init, stepsize, maxiter: int = 1000, tol=1e-5) -> None:
        self.x_init = x_init
        self.x = x_init
        self.fun = fun
        self.stepsize = stepsize
        self.maxiter = maxiter
        self.tol = tol
        self.nit = 1

    def init_state(self, x_init, *args, **kwargs):
        self.x_init = x_init
        return self.x, self.x
    
    def update(self, sol, state, *args, **kwargs):
        self.nit += 1
        self.stepsize = 1 / (self.nit + 20)
        self.x = sol - self.stepsize * grad(self.fun)(sol)
        return self.x, self.x
    

def test_local():
    n = 2
    x_init = jnp.array([1.0, 1.0])
    problem = QuadraticProblem(n=n)
    benchamrk = Benchmark(
        problem=problem,
        methods=[
            {
                'MY_GRADIENT_DESCENT': {'x_init' : x_init, 'tol' : 1e-5}
            },
            {
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-7,
                    'maxiter': 1000,
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
    my_solver = MyGradientDescent(fun=problem.f, x_init=x_init, maxiter=1000, stepsize=1e-2)
    result = benchamrk.run(user_method=my_solver)
    result.save("GD_quadratic.json")

test_local()