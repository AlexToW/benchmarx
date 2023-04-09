import json


from methods import Methods
from benchmark_target import BenchmarkTarget
from problem import Problem


class BenchmarkResult:
    methods: list[Methods] = None       # methods that have been benchmarked
    keys: list[BenchmarkTarget] = None  # an array of fields that will be assigned to each of the methods from self.methods
    problem: Problem = None             # the Problem on which the benchmarking was performed
    data: dict[Methods, dict[Problem, dict[BenchmarkTarget, list[any]]]] = None

    def __init__(self, problem: Problem, methods: list[Methods], keys: list[BenchmarkTarget], 
                 data: dict[Methods, dict[Problem, dict[BenchmarkTarget, list[any]]]] = None) -> None:
        self.problem = problem
        self.methods = methods
        self.keys = keys
        self.data = data
    
    def save(self, path: str) -> None:
        """
        Saves benchmarking data to a json file by path.
        """
        with open(path, 'w') as file:
            json.dump(self.data, file)

    def send_wandb(self):
        pass

    def __str__(self) -> str:
        return ''