from typing import Callable
import jax
import sys
import os
import jax.numpy as jnp
import pandas as pd

from problem import Problem


class QuadraticProblemRealData(Problem):
    """
    World bank data https://data.worldbank.org/
    """
    A = None
    b = None
    n: int = 1
    def __init__(self, info: str = 'QP_World_Bank_data') -> None:
        dataset = pd.read_csv('https://raw.githubusercontent.com/MerkulovDaniil/sber21_fmin/sources/data/world_bank_data.csv')
        y_variable = 'Import'
        x_variable = 'Export'

        clean_data = dataset[[x_variable, y_variable]].dropna()

        x_data = clean_data[x_variable]
        y_data = clean_data[y_variable]

        self.n = x_data.shape[0]
        self.A = x_data.T @ x_data
        self.b = -2 * x_data.T @ y_data

        B = jnp.vstack([x_data, jnp.ones(len(x_data))]).T
        self.x_opt, _ = jnp.linalg.lstsq(B, y_data, rcond=None)[0]

        self.f_opt = self.f(self.x_opt)
        
        super().__init__(info=info, func=self.f)

    def f(self, w):
        """
        w: float
        """
        return self.A * w**2 + self.b * w

