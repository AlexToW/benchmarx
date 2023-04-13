import jax
import sys
import os
import jax.numpy as jnp

from problem import Problem

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("src"), "..")))
from defaults import *


class QuadraticProblem(Problem):
    """
    A class describing an unconstrained quadratic problem:
    0.5 x^TAx^T + b^Tx, where the matrix A is positive defined, x \in R^n
    """

    n: int = 1  # problem dimensionality
    A = None  # A-matrix: np.array of shape (n,n)
    b = None  # b-vector: np.array of shape (n,)

    def __init__(self, n: int = 2, A=None, b=None, info: str = "Quadratic problem"):
        self.n = n

        if A is None:
            self.A = self.__get_random_matrix(self.n)
        else:
            self.A = A

        if b is None:
            self.b = self.__get_random_vector(self.n)
        else:
            self.b = b

        # func = lambda x: 0.5 * x.T @ self.A @ x + self.b.T @ x
        super().__init__(info=info, func=self.f)

    def f(self, x, *args, **kwargs):
        """
        Quadratic function
        """
        return 0.5 * x.T @ self.A @ x + self.b.T @ x

    def __get_random_matrix(self, n: int = 2):
        """
        Returns a positive defined matrix of size (n, n)
        Powered by eigendecomposition
        """
        key = jax.random.PRNGKey(default_seed)
        return jax.random.uniform(key, (n, n))

    def __get_random_vector(self, n: int = 2):
        """
        Returns a random vector: np.array of shape (n,)
        """
        key = jax.random.PRNGKey(default_seed)
        return jax.random.uniform(key, (n,))


def local_test():
    n = 3
    qp = QuadraticProblem(n=n)
    print(qp.info, "\n", qp.A)


# local_test()
