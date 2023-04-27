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
        return sol, state
    
    def stop_criterion(self, state: custom_optimizer.State) -> bool:
        return state.iter_num > self.maxiter


    

def test_local():
    n = 2
    x_init = jnp.array([2.0, 1.0])
    problem = QuadraticProblem(n=n)
    benchmark = Benchmark(
        problem=problem,
        methods=[
            {
                'MY_GRADIENT_DESCENT': {}
            },
            {
                'GRADIENT_DESCENT_adaptive_step': {
                    'x_init' : x_init,
                    'tol': 1e-4,
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
    my_solver = MyGradientDescent(
        x_init=x_init,
        stepsize=1e-2,
        problem=problem,
        tol=1e-3,
        maxiter=500,
        label='MyGradDescent'
    )
    result = benchmark.run(user_method=my_solver)
    result.save("GD_quadratic.json")


if __name__ == '__main__':
    test_local()