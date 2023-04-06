import numpy as np
from enum import Enum

from test_functions import TestFunctions, TestFunctionUnit


class BenchmarckType(Enum):
    ITERS = 1,
    FUNC_CALLS = 2,
    JAC_CALLS = 3,
    HESS_CALLS = 4,
    DURATION = 5


class Benchmark:
    n: int = 1      #  problem dimensionality
    def __init__(self, n) -> None:
        self.n = n
        self.test_functions: list[TestFunctionUnit] = [
            TestFunctionUnit(
                func=TestFunctions.Ackley_f,
                x_opt=np.array([0, 0]),
                f_opt=0.
            ),
            TestFunctionUnit(
                func=TestFunctions.Rastrigin_f,
                x_opt=np.zeros(n),
                f_opt=0.
            ),
            TestFunctionUnit(
                func=TestFunctions.Beale_f,
                x_opt=np.array([3., 0.5]),
                f_opt=0.
            ),
            TestFunctionUnit(
                func=TestFunctions.Booth_f,
                x_opt=np.array([1., 3.]),
                f_opt=0.
            ),
            TestFunctionUnit(
                func=TestFunctions.Bukin_f,
                x_opt=np.array([-10., 1.]),
                f_opt=0.
            ),
            TestFunctionUnit(
                func=TestFunctions.Easom_f,
                x_opt=np.array([np.pi, np.pi]),
                f_opt=-1.
            ),
            TestFunctionUnit(
                func=TestFunctions.Goldstein_Price_f,
                x_opt=np.array([0., -1.]),
                f_opt=3.
            ),
            TestFunctionUnit(
                func=TestFunctions.Levi_f,
                x_opt=np.array([1., 1.]),
                f_opt=0.
            )
        ]