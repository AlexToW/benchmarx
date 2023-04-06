import numpy as np
from benchmark import TestFunctions
from optimize import Optimize


def f(x):
    return x.T @ x + x

def df(x):
    return 2 * x + 1

def accept_test(x, eps):
    return np.linalg.norm(df(x)) <= eps


def test():
    x0 = np.array([1, 1, 1])
    eps = 1e-8
    optimizer = Optimize()
    opts = optimizer.gradient_descent(f=f, df=df, x0=x0, accept_test=accept_test, accuracy=eps)
    print(opts)



def _main():
    test()


if __name__ == '__main__':
    _main()
