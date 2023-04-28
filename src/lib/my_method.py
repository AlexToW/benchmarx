# My method example/test
from jax import grad
import jax.numpy as jnp


from problems.quadratic_problem import QuadraticProblem
from benchmark import Benchmark
import custom_optimizer


class MyGradientDescent(custom_optimizer.CustomOptimizer):
    """
    Must have the following fields:
        label
    Must have the following methods:
        init_state
        update
        stop_criterion
    """
    def __init__(self, x_init, stepsize, problem, tol=1e-6, maxiter=400, label = 'MyGD'):
        params = {
            'x_init': x_init,
            'tol': tol,
            'maxiter': maxiter,
            'stepsize': stepsize
        }
        self.stepsize = stepsize
        self.problem = problem
        self.maxiter = maxiter
        self.tol = tol
        super().__init__(params=params, x_init=x_init, label=label)

    def init_state(self, x_init, *args, **kwargs) -> custom_optimizer.State:
        return custom_optimizer.State(
            iter_num=1,
            stepsize=self.stepsize
        )
    
    def update(self, sol, state: custom_optimizer.State) -> tuple([jnp.array, custom_optimizer.State]):
        sol -= state.stepsize * grad(self.problem.f)(sol)
        state.iter_num += 1
        state.stepsize = 1 / (state.iter_num + 1)
        return sol, state
    
    def stop_criterion(self, sol, state: custom_optimizer.State) -> bool:
        return jnp.linalg.norm(grad(self.problem.f)(sol))**2 < self.tol


    

def test_local():
    n = 3
    x_init = jnp.array([2.0, 1.0, 0.0])
    problem = QuadraticProblem(n=n)
    benchmark = Benchmark(
        runs=2,
        problem=problem,
        methods=[
            {
                'MY_GRADIENT_DESCENT': {}
            },
            {
                'GRADIENT_DESCENT_const_step': {
                    'x_init' : x_init,
                    'tol': 1e-3,
                    'maxiter': 2,
                    'stepsize' : 1e-2
                }
            }
        ],
        metrics=[
            "nit",
            "history_x",
            "history_f",
        ],
    )
    my_solver = MyGradientDescent(
        x_init=x_init,
        stepsize=1e-2,
        problem=problem,
        tol=1e-3,
        maxiter=2,
        label='MyGradDescent'
    )
    result = benchmark.run(user_method=my_solver)
    result.save("GD_quadratic.json")


if __name__ == '__main__':
    test_local()