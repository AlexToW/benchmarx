import numpy as np

from optimize_results import OptimizeResults


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
    def binary_search(self):
        pass

    @classmethod
    def golden_search(self):
        pass




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