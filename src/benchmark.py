import numpy as np
from enum import Enum


class TestFunctionUnit:
    def __init__(self, func, x_opt, f_opt):
        self.func = func
        self.x_opt = x_opt
        self.f_opt = f_opt

    def func(self, **kwargs):
        return self.func(**kwargs)


class TestFunctions:
    def __init__(self):
        pass

    @classmethod
    def Rastrigin_f(self, x):
        """
        Global optimum: f(0, ..., 0) = 0
        :param x: np.array of shape (n,)
        :return: float
        """
        A = 10
        n = x.shape[0]
        return A * n + np.sum(x_i ** 2 - A * np.cos(2 * np.pi * x_i) for x_i in x)

    @classmethod
    def Rastrigin_grad_f(self, x):
        """
        Gradient of Rastrigin function
        :param x: np.array of shape (n,)
        :return:
        """
        A = 10
        return np.array(2*x_i + 2*np.pi*A*np.sin(2*np.pi*x_i) for x_i in x)

    @classmethod
    def Ackley_f(self, z):
        """
        Global optimum: f(0, 0) = 0
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20
    
    @classmethod
    def Ackley_grad_f(self, z):
        """
        Gradient of Ackley function
        :param x: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        sqr = np.sqrt(x**2/2 + y**2/2)
        trig = (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))/2
        df_x = 2 * x * np.exp(-0.2*sqr) / sqr + np.pi * np.sin(2*np.pi*x)*np.exp(trig)
        df_y = 2 * y * np.exp(-0.2*sqr) / sqr + np.pi * np.sin(2*np.pi*y)*np.exp(trig)
        return np.array([df_x, df_y])

    @classmethod
    def Sphere_f(self, x):
        """
        Global optimum: f(0, ..., 0) = 0
        :param x: np.array of shape (n,)
        :return: float
        """
        return np.sum(x_i ** 2 for x_i in x)
    
    @classmethod
    def Sphere_grad_f(self, x):
        """
        Gradient of Sphere function
        :param x: np.array of shape (n,)
        :return: np.array of shape (n,)
        """
        return np.array(2*x_i for x_i in x)

    @classmethod
    def Rosenbrock_f(self, x):
        """
        Global optimum: f(1,..., 1) = 0
        :param x: np.array of shape (n,)
        :return: float
        """
        n = x.shape[0]
        res = 0
        for i in range(0, n - 1):
            res += 100 * (x[i + 1] - x[i]) ** 2 + (1 - x[i]) ** 2

        return res
    
    @classmethod
    def Rosenbrock_grad_f(self, x):
        """
        Gradient of Rosenbrock function
        :param x: np.array of shape (n,)
        :return: np.array of shape (n,)
        """
        n = x.shape[0]
        grad = np.zeros((n,))
        for k in range(n):
            grad_k = 0
            if k == i + 1:
                for i in range(0, n - 1):
                    grad_k += 100 * (x[i+1] - x[i])
            if k == i:
                for i in range(0, n - 1):
                    grad_k += -100 * (x[i+1] - x[i]) - 2 * (1 - x[i])
            grad[k] = grad_k

        return grad

    @classmethod
    def Beale_f(self, z):
        """
        Global optimum: f(3, 0.5) = 0
        -4.5 <= x, y <= 4.5
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return (1.5 - x - x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2
    
    @classmethod
    def Beale_grad_f(self, z):
        """
        Gradient of Beale function
        -4.5 <= x, y <= 4.5
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        a1 = 1.5 - x + x*y
        a2 = 2.25 - x - x*y**2
        a3 = 2.625 - x - x*y**3
        df_x = 2*a1*(y-1) + 2*a2*(y**2-1) + 2*a3*(y**3-1)
        df_y = 2*x*(a1 + a2 + a3)
        return np.array([df_x, df_y])

    @classmethod
    def Goldstein_Price_f(self, z):
        """
        Global optimum: f(0, -1) = 3
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    
    @classmethod
    def Goldstein_Price_grad_f(self, z):
        '''
        Gradient of Goldstein-Price function
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        '''
        x = z[0]
        y = z[1]
        return np.array([24* (8* x**3 - 4* x**2* (9* y + 4) + 6* x* (9* y**2 + 8* y + 1) - 9* y* (3* y**2 + 4* y + 1))* ((3* x**2 + 2* x* (3* y - 7) + 3* y**2 - 14* y + 19)* (x + y + 1)**2 + 1) + 12* (x**3 + x**2* (3* y - 2) + x* (3* y**2 - 4* y - 1) + y**3 - 2* y**2 - y + 2)* ((12* x**2 - 4* x* (9* y + 8) + 3* (9* y**2 + 16* y + 6))* (2* x - 3* y)**2 + 30), 12* (x**3 + x**2* (3* y - 2) + x* (3* y**2 - 4* y - 1) + y**3 - 2* y**2 - y + 2)* ((12* x**2 - 4* x* (9* y + 8) + 3* (9* y**2 + 16* y + 6))* (2* x - 3* y)**2 + 30) - 36* (8* x**3 - 4* x**2* (9* y + 4) + 6* x* (9* y**2 + 8* y + 1) - 9* y* (3* y**2 + 4* y + 1))* ((3* x**2 + 2* x* (3* y - 7) + 3* y**2 - 14* y + 19)* (x + y + 1)**2 + 1)])

    @classmethod
    def Booth_f(self, z):
        """
        Global optimum: f(1, 3) = 0
        -10 <= x, y <= 10
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    
    @classmethod
    def Booth_grad_f(self, z):
        '''
        Gradient of Booth function
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        '''
        x = z[0]
        y = z[1]
        df_x = 10*x + 8*y - 34
        df_y = 8*x + 10*y - 38
        return np.array([df_x, df_y])

    @classmethod
    def Bukin_f(self, z):
        """
        Global optimum: f(-10, 1) = 0
        -15 <= x <= -5
        -3 <= y <= 3
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)

    @classmethod
    def Bukin_grad_f(self, z):
        '''
        Gradient of Bukin function
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        '''
        def sign(a):
            return int(a >= 0)
        x = z[0]
        y = z[1]
        tok = y - x**2/100
        df_x = -x*sign(tok) / np.sqrt(abs(tok)) + sign(x+10)/100
        df_y = 50*sign(tok) / np.sqrt(abs(tok))
        return np.array([df_x, df_y])

    @classmethod
    def Matyas_f(self, z):
        """
        Global optimum: f(0, 0) = 0
        -10 <= x, y <= 10
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return 0.26 * (x ** 2 + y ** 2) - 0.28 * x * y

    @classmethod
    def Matyas_grad_f(self, z):
        """
        Gradient of Matyas function
        -10 <= x, y <= 10
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        return np.array([0.52*x - 0.48*y, 0.52*y - 0.48*x])

    @classmethod
    def Levi_f(self, z):
        """
        Global optimum: f(1, 1) = 0
        -10 <= x, y <= 10
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
                    1 + np.sin(2 * np.pi * y) ** 2)

    @classmethod
    def Levi_grad_f(self, z):
        """
        Gradient of Levi function
        -10 <= x, y <= 10
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        df_x = 3*np.pi*np.sin(6*np.pi*x) + 2*(x-1)*(1+(np.sin(3*np.pi*y)**2))
        df_y = (x-1)**2*(1 + 3*np.pi*np.sin(6*np.pi*y)) + 2*(y-1)*(1+(np.sin(3*np.pi*y)**2)) + (y-1)**2*(1+2*np.pi*np.cos(4*np.pi*y))
        return np.array([df_x, df_y])

    @classmethod
    def Three_hump_camel_f(self, z):
        """
        Global optimum: f(0, 0) = 0
        -5 <= x, y <= 5
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return 2 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y ** 2
    
    @classmethod
    def Three_hump_camel_grad_f(self, z):
        """
        Gradient of Three hump camel function
        -5 <= x, y <= 5
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        df_x = 4*x - 4.2*x**3 + x**5 + y
        df_y = x + 2*y
        return np.array([df_x, df_y])

    @classmethod
    def Easom_f(self, z):
        """
        Global optimum: f(pi, pi) = -1
        -100 <= x, y <= 100
        :param z: np.array of shape (2,)
        :return: float
        """
        x = z[0]
        y = z[1]
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

    @classmethod
    def Easom_grad_f(self, z):
        """
        Gradient of Easom function
        -100 <= x, y <= 100
        :param z: np.array of shape (2,)
        :return: np.array of shape (2,)
        """
        x = z[0]
        y = z[1]
        exp = np.exp(-(x-np.pi)**2 - (y-np.pi)**2)
        df_x = np.sin(x)*np.cos(y)*exp + np.cos(x)*np.cos(y)*exp*2*(x-np.pi)
        df_y = np.cos(x)*np.sin(y)*exp + np.cos(x)*np.cos(y)*exp*2*(y-np.pi)
        return np.array([df_x, df_y])


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