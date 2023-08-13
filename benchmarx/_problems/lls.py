from benchmarx.problem import Problem
from benchmarx.defaults import default_seed
import jax.numpy as jnp
from jax import random
import logging
from sklearn.datasets import load_wine
import pandas as pd

class LinearLeastSquares(Problem):
    """
    A class describing an unconstrained linear least squares problem.

    f(x) = 1/2*||Ax - b||_2^2 -> min_x

    Typical usage example:

    problem = LinearLeastSquares("random", m=3, n=3)
    benchmark = Benchmark(problem=problem, ...)
    result = benchmark.run()
    """

    def __init__(self, problem_type: str, 
        A: jnp.array=None, b: jnp.array=None, x_opt: jnp.array=None, 
        m: int=None, n: int=None, seed: int=default_seed,
        reduce: bool=False) -> None:
        """
        Initializes the LinearLeastSquares problem.

        Args:
            problem_type: The type of problem to set up. Can be one of the
                               following:
                               - 'random'
                               - 'boston'
                               - 'wine'
                               - 'custom'
            A (optional): Matrix A for the custom problem. Defaults to None.
            b (optional): Vector b for the custom problem. Defaults to None.
            x_opt(optional): Optimal solution for the custom problem. 
                                Defaults to None.
            m (optional): Dimension of the data for random types. 
                               Defaults to None.
            n (optional): Dimension of the problem for random types. 
                               Defaults to None.
            seed (optional): Seed for RNG. Defaults to benchmarx 
                                default_seed.
        Raises:
            ValueError: If the provided problem_type is not recognized.
            ValueError: If required parameters (A, b for 'custom', 
                m, n for 'random') are not provided.
        """
        super().__init__(
            info=f"Linear Least Squares problem on {problem_type} dataset", 
            func=self.f
            )

        self.seed = seed
        self.reduce = reduce

        if problem_type == 'custom':
            if A is None or b is None:
                raise ValueError("For custom problem type, "
                    "A and b must be provided.")
            self.A = A
            self.b = b
            if x_opt is None:
                logging.debug("Solving custom linear system with "
                                "jnp.linalg.lstsq ")
                self.x_opt = jnp.linalg.lstsq(A, b)[0]
            else:
                self.x_opt = x_opt

        elif problem_type == 'random':
            if m is None or n is None:
                raise ValueError("For random problem types, "
                    "m and n must be provided.")
            self.A, self.b, self.x_opt = self._generate_data(m, n)
        elif problem_type in ['boston', 'wine']:
            self.A, self.b, self.x_opt = self._load_data(problem_type)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        self.f_opt = self.f(self.x_opt)
        
    def _generate_data(self, m, n):
        """
        Generates random data for the linear system.

        Args:
            m (int): Dimension of the data.
            n (int): Dimension of the problem.

        Returns:
            A: Matrix A for the problem.
            b: Vector b for the problem.
            x_opt: Optimal solution for the problem
        """

        # Log system type
        if m < n:
            logging.debug("Generating underdetermined system.")
        elif m == n:
            logging.debug("Generating square system.")
        elif m > n:
            logging.debug("Generating overdetermined system.")

        RNG = random.PRNGKey(self.seed)

        A = random.normal(RNG, (m, n))
        x_opt = random.normal(RNG, (n,))
        b = jnp.dot(A, x_opt)

        return A, b, x_opt

    def _load_data(self, problem_type):
        """
        Loads problem data from available datasets.

        Args:
            problem_type (str): Name of the dataset to load.

        Returns:
            A (jnp.array): Matrix A for the problem.
            b (jnp.array): Vector b for the problem.
            x_opt (jnp.array): Optimal solution vector.
        """
        if problem_type == 'boston':
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22,header=None)
            data = jnp.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]
            logging.debug("Loading boston house-prices datase")
            A = data
            b = target
        elif problem_type == 'wine':
            logging.debug("Loading Wine dataset")
            data = load_wine()
            A = data.data
            b = data.target
        else:
            raise ValueError(f"Unknown dataset name: {problem_type}")

        # Compute optimal solution using least squares
        x_opt = jnp.linalg.lstsq(A, b)[0]

        return A, b, x_opt

    def f(self, x: jnp.array, *args, **kwargs):
        """
        Function to minimize.
        """
        x = jnp.array(x)
        if self.reduce:
            m, n = self.A.shape 
            return 1/(2*m)*jnp.linalg.norm(self.A@x - self.b)**2
        else:
            return 0.5*jnp.linalg.norm(self.A@x - self.b)**2