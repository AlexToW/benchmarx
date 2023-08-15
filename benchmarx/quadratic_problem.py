import jax
import sys
import os
import scipy.linalg as la
import logging
import jax.numpy as jnp


from benchmarx.problem import Problem

#sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname("benchmarx"), "..")))
from benchmarx.defaults import *


class QuadraticProblem(Problem):
    """
    A class describing an unconstrained quadratic problem:
    0.5 x^TAx^T + b^Tx, where the matrix A is positive defined, x \in R^n
    """

    n: int = 1  # problem dimensionality
    A = None  # A-matrix: np.array of shape (n,n)
    b = None  # b-vector: np.array of shape (n,)
    seed = default_seed
    def __init__(self, n: int = 2, A=None, b=None, mineig=0, maxeig=1, info: str = "Quadratic problem"):
        """
        Initialize the QuadraticProblem instance.

        Args:
            n (int): Problem dimensionality.
            A (np.ndarray, optional): A-matrix of shape (n,n).
            b (np.ndarray, optional): b-vector of shape (n,).
            mineig (float, optional): Minimum eigenvalue for random matrix A.
            maxeig (float, optional): Maximum eigenvalue for random matrix A.
            info (str, optional): Brief information about the problem.
        """
        self.n = n

        if A is None:
            self.A = self.__get_random_matrix(self.n, maxeig=maxeig, mineig=mineig)
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
        Calculate the quadratic function value.

        Args:
            x (Any): Input value for the function.

        Returns:
            Any: The quadratic function value.
        """
        x = jnp.array(x)
        return 0.5 * x.T @ self.A @ x + self.b.T @ x

    def __get_random_matrix(self, n: int = 2, mineig=0, maxeig=1):
        """
        Returns a positive defined matrix of size (n, n)
        with eigenvalues in [mineig, maxeig]
        """
        key = jax.random.PRNGKey(self.seed)

        lambdas = list()  # eigenvalues of matrix to generate

        if n == 1:
            if mineig == maxeig:
                lambdas = [mineig]
            else:
                logging.critical(msg="It is impossible to create a matrix of shape (1,1) with two different eigenvalues.")
        if n == 2:
            lambdas = [mineig, maxeig]
        else:
            lambdas = jax.random.uniform(key, minval=mineig, maxval=maxeig, shape=(n-2,))
            lambdas = lambdas.tolist() + [mineig, maxeig]

        A = jnp.diag(jnp.array(lambdas))
        q, _ = la.qr(jax.random.uniform(key, (n, n)))
        A = q.T @ A @ q
        return A

    def __get_random_vector(self, n: int = 2):
        """
        Returns a random vector: np.ndarray of shape (n,)
        """
        key = jax.random.PRNGKey(self.seed)
        return jax.random.uniform(key, (n,))


def local_test():
    n = 3
    qp = QuadraticProblem(n=n)
    print(qp.info, "\n", qp.A)


# local_test()
