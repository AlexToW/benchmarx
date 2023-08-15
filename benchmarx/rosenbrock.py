from typing import Callable
import jax
import jax.numpy as jnp
import logging

from benchmarx.problem import Problem
from benchmarx.defaults import default_seed

class Rosenbrock(Problem):
    """
    A class describing the Rosenbrock function.

    The Rosenbrock function is a non-convex optimization problem that is often
    used as a benchmark to test optimization algorithms. It is characterized
    by a narrow, parabolic shaped valley that contains the global minimum.

    The function is defined as:
    f(x) = sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2) for i in range(n-1)

    where n is the dimensionality of the problem and x is the input vector.
    """
    n: int = 2      # problem dimensionality
    seed = default_seed
    def __init__(self, n: int = 2, info: str = 'Rosenbrock') -> None:
        """
        Initialize the Rosenbrock instance.

        Args:
            n (int, optional): Problem dimensionality.
            info (str, optional): Brief information about the problem.
        """
        self.info = info
        self.n = n
        self.x_opt = jnp.ones(self.n)
        self.f_opt = 0.0
        super().__init__(info=info, func=self.f, x_opt=self.x_opt)
    

    def f(self, x, *args, **kwargs):
        """
        Calculate the Rosenbrock function value.

        Args:
            x (Any): Input value for the function.

        Returns:
            Any: The Rosenbrock function value.
        """
        if x.shape[0] != self.n:
            err_msg = f'Wrong x shape: {x.shape}. Expected: ({self.n},)'
            logging.critical(err_msg)
            exit(1)
        return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(self.n - 1)])
