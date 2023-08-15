from typing import Callable
import jax
import jax.numpy as jnp
import logging

from benchmarx.problem import Problem
from benchmarx.defaults import default_seed

class Rastrigin(Problem):
    """
    A class describing the Rastrigin function.

    The function is defined as:
    f(x) = A * n + sum(xi^2 - A * cos(2 * pi * xi)) for xi in x

    where A is a constant, n is the dimensionality of the problem, and x is
    the input vector.
    """
    n: int = 2      # problem dimensionality
    seed = default_seed
    def __init__(self, n: int = 2, info: str = 'Rastrigin') -> None:
        """
        Initialize the Rastrigin instance.

        Args:
            n (int, optional): Problem dimensionality.
            info (str, optional): Brief information about the problem.
        """
        self.info = info
        self.n = n
        self.x_opt = jnp.zeros(self.n)
        self.f_opt = 0.0
        super().__init__(info=info, func=self.f, x_opt=self.x_opt)
    

    def f(self, x, *args, **kwargs):
        """
        Calculate the Rastrigin function value.

        Args:
            x (Any): Input value for the function.

        Returns:
            Any: The Rastrigin function value.
        """
        if x.shape[0] != self.n:
            err_msg = f'Wrong x shape: {x.shape}. Expected: ({self.n},)'
            logging.critical(err_msg)
            exit(1)
        A = 10
        return A * self.n + jnp.sum(jnp.array([xi**2 - A * jnp.cos(2 * jnp.pi * xi) for xi in x]))
