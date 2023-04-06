import numpy as np

from optimize_results import OptimizeResults
from time import time


class Optimize:
    def __init__(self):
        pass
    
    @classmethod
    def gradient_descent(self, f, df, x0, step_size: float = 1e-2, max_steps: int = 1000, accuracy: float = 1e-5,
                         trajectory_flag: bool = False, accept_test=None):
        trajectory = list()
        x = x0
        success = False
        nfev = 0
        njev = 0
        nit = 0
        for i in range(max_steps):
            if trajectory_flag:
                trajectory.append(x)
            if accept_test and accept_test(x, accuracy):
                success = True
                break
            x = x - step_size * df(x)
            njev += 1
            nit += 1
        result = OptimizeResults(
            success=success,
            message="",
            fun=f(x),
            jac=df(x),
            nfev=nfev,
            njev=njev,
            nhev=0,
            nit=nit,
            x=x
        )
        if trajectory_flag:
            result.trajectory = trajectory

        return result


    @classmethod
    def binary_search(self, f, a: float, b: float, accuracy: float = 1e-5, max_steps: int = 1000):
        success = False
        nfev = 0
        njev = 0
        nit = 0
        c = (a + b) / 2
        for _ in range(max_steps): 
            if abs(b - a) <= accuracy:
                success = True
                break
            nit += 1
            y = (a + c) / 2.0
            nfev += 2
            if f(y) <= f(c):
                b = c
                c = y
            else:
                nfev += 2
                z = (b + c) / 2.0
                if f(c) <= f(z):
                    a = y
                    b = z
                else:
                    a = c
                    c = z
        result = OptimizeResults(
            success=success,
            message="",
            fun=f(c),
            jac=None,
            nfev=nfev,
            njev=njev,
            nhev=0,
            nit=nit,
            x=c
        )
        return result

    @classmethod
    def golden_search(self, f, a: float, b: float, accuracy: float = 1e-5, max_steps: int = 1000):
        success = False
        nfev = 0
        njev = 0
        nit = 0
        tau = (np.sqrt(5) + 1) / 2
        y = a + (b - a) / tau**2
        z = a + (b - a) / tau
        for _ in range(max_steps):
            if b - a <= accuracy:
                success = True
                break
            nit += 1
            nfev += 2
            if f(y) <= f(z):
                b = z
                z = y
                y = a + (b - a) / tau**2
            else:
                a = y
                y = z
                z = a + (b - a) / tau
        x_opt = (b-a)/2
        result = OptimizeResults(
            success=success,
            message="",
            fun=f(x_opt),
            jac=None,
            nfev=nfev,
            njev=njev,
            nhev=0,
            nit=nit,
            x=x_opt
        )
        return result




def small_based_tests():
    def f(x):
        return sum([(x_i - 1)**2 for x_i in x])
    def df(x):
        return np.array([2*(x_i - 1) for x_i in x])
    def accept_test(x, eps):
        return np.linalg.norm(df(x))**2 <= eps

    x0 = np.array([1, 2, 0])
    x_opt = np.array([1, 1, 1])
    results = Optimize.gradient_descent(f=f, df=df, accuracy=1e-6, accept_test=accept_test, x0=x0, max_steps=1000)
    print(f'|x-x*|: {np.linalg.norm(results.x - x_opt)}')
    print(results)


small_based_tests()