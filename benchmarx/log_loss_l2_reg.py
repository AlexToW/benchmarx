from typing import Callable
import jax
import jax.numpy as jnp
import logging

import sys
import os

from benchmarx.problem import Problem

#sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))
from benchmarx.defaults import *

class LogLossL2Reg(Problem):
    """
    Strongly convex: log loss with l2 regularisation
    y: jnp.array with shape (n,)
    X: jnp.array with shape (n, d), 
        x_i is the ith row of X matrix, 
        x_i shape (d,)
    
    f(w) = 1/n sum i = 1 to n log[1 + exp(-y_i <x_i, w>)] + mu/2 ||w||_2^2
    """
    X: jnp.array = None
    y: jnp.array = None
    mu: float               # function is mu strongly convex
    seed = default_seed
    def __init__(self, y = None, X = None, n: int = 2, d: int = 2, mu: float = 1., info: str='log_loss_with_l2_reg') -> None:
        self.n = n
        self.d = d
        if y is None:
            self.y = self._generate_y()
        else:
            if y.shape[0] == self.n:
                self.y = y
            else:
                err_msg = f'y must be of shape ({self.n},), not {y.shape}'
                logging.critical(err_msg)
                exit(1)
        if X is None:
            self.X = self._generate_X()
        else:
            if X.shape[0] == self.n and X.shape[1] == self.d:
                self.X = X
            else:
                err_msg = f'X must be of shape ({self.n}, {self.d}), not {X.shape}'
                logging.critical(err_msg)
                exit(1)
        self.mu = mu
        super().__init__(info=info, func=self.f)
        
    
    def f(self, w, *args, **kwargs):
        """
        w: jnp.array with shape (d,)
        """
        if w.shape != (self.d,):
            err_msg = f'Argument must be of shape ({self.d},), not {w.shape}'
            logging.critical(err_msg)
            exit(1)
        
        return sum([jnp.log(1 + jnp.exp(-self.y[i] * self.X[i].T @ w)) for i in range(self.n)]) / self.n + self.mu / 2 * jnp.linalg.norm(w)**2


    def _generate_X(self):
        key = jax.random.PRNGKey(self.seed)
        return jax.random.uniform(key, (self.n, self.d))

    def _generate_y(self):
        key = jax.random.PRNGKey(self.seed)
        return jax.random.uniform(key, (self.n,))
