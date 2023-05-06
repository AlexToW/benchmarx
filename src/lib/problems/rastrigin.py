from typing import Callable
import jax
import jax.numpy as jnp
import logging

from problem import Problem


class Rastrigin(Problem):

    n: int = 2      # problem dimensionality
    def __init__(self, n: int = 2, info: str = 'Rastrigin') -> None:
        self.info = info
        self.n = n
        self.x_opt = jnp.zeros(self.n)
        self.f_opt = 0.0
        super().__init__(info=info, func=self.f)
    

    def f(self, x):
        if x.shape[0] != self.n:
            err_msg = f'Wrong x shape: {x.shape}. Expected: ({self.n},)'
            logging.critical(err_msg)
            exit(1)
        A = 10
        return A * self.n + sum([xi**2 - A * jnp.cos(2 * jnp.pi * xi) for xi in x])
