import jaxopt

from problem import Problem
from methods import Methods
from benchmark_target import BenchmarkTarget
from benchmark_result import BenchmarkResult


class Benchmark:
    """
    A class that provides the benchmarking of different optimization methods on a given problem (like Problem object).
    """
    problem: Problem = None                         # Problem to solve
    methods: list[Methods] = None                   # Methods for benchmarking
    result_params: list[BenchmarkTarget] = None     # List of fields to include in BenchamrkResult

    def __init__(self, problem: Problem, methods: list[Methods], result_params: list[BenchmarkTarget]) -> None:
        self.problem = problem
        self.methods = methods
        self.result_params = result_params

    def __sub_run(self, method: Methods, method_params) -> dict[BenchmarkTarget, list[any]]:
        """
            хочу: функция (метод), которая принимает Methods, [f (like jax function) -- можно взять как self.problem.f], 
            массив из BenchmarkTarget -- можно взять как self.result_params, 
            необходимые для запуска метода параметры и возвращает словарь BechmarkTarget : _суть_
        """
        if method == Methods.GRADIENT_DESCENT:
            """
            Здесь будет вызываться "прослойка", позволяющая получить необходимые данные, если они не доступны в "выхлопе" jaxopt.метода по умолчанию
            (например, trajectory_x)
            """
            pass

    def run(self) -> BenchmarkResult:
        res = BenchmarkResult(
            problem=self.problem,
            methods=self.methods,
            keys=self.result_params
        )
        pre_data = list()
        data = dict()
        for method in self.methods:
            # data: dict[Methods, dict[Problem, dict[BenchmarkTarget, list[any]]]] = None
            sub = self.__sub_run()
            data[Methods.GRADIENT_DESCENT] = dict((self.problem, sub))

        