from problem import Problem
from methods import Methods
from benchmark_target import BenchmarkTarget
from benchmark_result import BenchmarkResult


class Benchmark:
    """
    A class that provides the benchmarking of different optimization methods on a given problem (like jax function).
    """
    problem: Problem = None                         # Problem to solve
    methods: list[Methods] = None                   # Methods for benchmarking
    result_params: list[BenchmarkTarget] = None     # List of fields to include in BenchamrkResult

    def __init__(self, problem: Problem, methods: list[Methods], result_params: list[BenchmarkTarget]):
        self.problem = problem
        self.methods = methods
        self.result_params = result_params

    def run(self):
        pass
        