from typing import Callable
import jax
import sys
import os
import jax.numpy as jnp
import pandas as pd

from benchmarx.problem import Problem
from benchmarx.defaults import default_seed


class QuadraticProblemRealData(Problem):
    """
    World bank data https://data.worldbank.org/
    """
    A = None
    b = None
    n: int = 1
    seed = default_seed
    def __init__(self, info: str = 'QP_World_Bank_data') -> None:
        dataset = pd.read_csv('https://raw.githubusercontent.com/MerkulovDaniil/sber21_fmin/sources/data/world_bank_data.csv')
        y_variable = 'Import'
        x_variable = 'Export'

        clean_data = dataset[[x_variable, y_variable]].dropna()
        
        x_data = jnp.array(clean_data[x_variable].values)
        y_data = jnp.array(clean_data[y_variable].values)
        self.n = x_data.shape[0]
        self.A = jnp.array([float(x_data.T @ x_data)]).reshape((1,1))
        self.b = jnp.array([float(-2 * x_data.T @ y_data)]).reshape((1,))

        B = jnp.vstack([x_data, jnp.ones(len(x_data))]).T
        x_opt, _ = jnp.linalg.lstsq(B, y_data, rcond=None)[0]
        self.x_opt = jnp.array([float(x_opt)])
        self.f_opt = self.f(self.x_opt)
        
        super().__init__(info=info, func=self.f, x_opt=self.x_opt)

    def f(self, w, *args, **kwargs):
        """
        w: jnp.array
        """
        return w.T @ self.A @ w + self.b.T @ w

