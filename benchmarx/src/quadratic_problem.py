import jax
import sys
import os
import jax.numpy as jnp


from benchmarx.src.problem import Problem

#sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("src"), "..")))
from benchmarx.src.defaults import *


class QuadraticProblem(Problem):
    """
    A class describing an unconstrained quadratic problem:
    0.5 x^TAx^T + b^Tx, where the matrix A is positive defined, x \in R^n
    """

    n: int = 1  # problem dimensionality
    A = None  # A-matrix: np.array of shape (n,n)
    b = None  # b-vector: np.array of shape (n,)
    seed = default_seed
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
        if jnp.linalg.det(self.A + self.A.T) != 0:
            self.x_opt = -2 * jnp.linalg.inv(self.A + self.A.T) @ self.b
            self.f_opt = self.f(self.x_opt)

    def f(self, x, *args, **kwargs):
        """
        Quadratic function
        """
        x = jnp.array(x)
        return 0.5 * x.T @ self.A @ x + self.b.T @ x

    def __get_random_matrix(self, n: int = 2):
        """
        Returns a positive defined matrix of size (n, n)
        """
        key = jax.random.PRNGKey(self.seed)
        A = jax.random.uniform(key, (n, n))
        return A @ A.T

    def __get_random_vector(self, n: int = 2):
        """
        Returns a random vector: np.array of shape (n,)
        """
        key = jax.random.PRNGKey(self.seed)
        return jax.random.uniform(key, (n,))


def local_test():
    n = 3
    qp = QuadraticProblem(n=n)
    print(qp.info, "\n", qp.A)


# local_test()
