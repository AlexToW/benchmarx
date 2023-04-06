from typing import Callable, List
import time
from optimize import Optimize
from test_functions import TestFunctions, TestFunctionUnit


class OptimizeTestResult:
    def __init__(self, x0=0.,  x_opt=0., f_opt=0., nit=0, nfev=0, duration=0., success=False) -> None:
        self.x0 = x0
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.nit = nit
        self.nfev = nfev
        self.duration = duration
        self.success = success

    def __str__(self) -> str:
        return f'Success: {self.success}. Initial point: {self.x0}. Optimal point: {self.x_opt}. Target function optimal value: {self.f_opt}. Iterations: {self.nit}. Function calls: {self.nfev}. Duration: {self.duration} (sec)'


class OptimizeTestUnit:
    def __init__(self, target_func: Callable, x0, x_opt, f_opt, lb = 0., rb = 0., accuracy = 1e-5, max_iters = 1000, info: str = ""):
        """
            lb (like left bound) and rb (like right bound) -- for line search methods or methods that requier initial point bounds
            :info: string -- Test description 
        """
        self.target_func = target_func
        self.x0 = x0
        self.x_opt = x_opt
        self.f_opt = f_opt
        self.lb = lb
        self.rb = rb
        self.accuracy = accuracy
        self.max_iters = max_iters
        self.info = info
        


class OptimizeTester:
    n: int = 1          # problem dimensionality
    def __init__(self, n: int = 1) -> None:
        self.n = n
        self.line_search_tests: list[OptimizeTestUnit] = [
            OptimizeTestUnit(
                target_func=TestFunctions.func_00_1d,
                x0 = None,
                x_opt=0.,
                f_opt=0.,
                lb=-2.,
                rb=2.,
                accuracy=1e-5,
                max_iters=1000
            ),
            OptimizeTestUnit(
                target_func=TestFunctions.func_01_1d,
                x0 = None,
                x_opt=5.145735,
                f_opt=TestFunctions.func_01_1d(5.145735),
                lb=2.,
                rb=8.,
                accuracy=1e-5,
                max_iters=1000
            ),
            OptimizeTestUnit(
                target_func=TestFunctions.func_02_1d,
                x0 = None,
                x_opt=-3.85045,
                f_opt=TestFunctions.func_02_1d(-3.85045),
                lb=2.,
                rb=4.,
                accuracy=1e-5,
                max_iters=1000
            ),
            OptimizeTestUnit(
                target_func=TestFunctions.func_03_1d,
                x0 = None,
                x_opt=0.96609,
                f_opt=TestFunctions.func_03_1d(0.96609),
                lb=0.,
                rb=1.2,
                accuracy=1e-5,
                max_iters=1000
            ),
            OptimizeTestUnit(
                target_func=TestFunctions.func_04_1d,
                x0 = None,
                x_opt=0.67956,
                f_opt=TestFunctions.func_04_1d(0.67956),
                lb=-10.,
                rb=10.,
                accuracy=1e-5,
                max_iters=1000
            ),
            OptimizeTestUnit(
                target_func=TestFunctions.func_05_1d,
                x0 = None,
                x_opt=5.19978,
                f_opt=TestFunctions.func_05_1d(5.19978),
                lb=2.7,
                rb=7.5,
                accuracy=1e-5,
                max_iters=1000
            )
        ]

    def run_line_search_full_test(self, method: Callable):
        """
        :return: list of list [OptimizeTestUnit, OptimizeTestResult]
        """
        results = list()
        for test in self.line_search_tests:
            start = time.time()
            tmp_res = method(test.target_func, test.lb, test.rb, test.accuracy, test.max_iters)
            duration = time.time() - start
            opt_test_result = OptimizeTestResult(
                x0=None,
                x_opt=tmp_res.x,
                f_opt=tmp_res.fun,
                nit=tmp_res.nit,
                nfev=tmp_res.nfev,
                duration=duration,
                success=bool(abs(tmp_res.x - test.x_opt) <= test.accuracy)
            )
            results.append([test, opt_test_result])
        return results


def line_search_tests():
    tester = OptimizeTester()
    results_binary = tester.run_line_search_full_test(method=Optimize.binary_search)
    print('Binary search:')
    for pair in results_binary:
        test, result = pair
        print(abs(test.x_opt - result.x_opt))
    print('Golden search:')
    results_golden = tester.run_line_search_full_test(method=Optimize.golden_search)
    for pair in results_binary:
        test, result = pair
        print(abs(test.x_opt - result.x_opt))


def _main():
    line_search_tests()


if __name__ == '__main__':
    _main()