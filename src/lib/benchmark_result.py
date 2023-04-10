import json


from methods import Method
from metrics import *
from problem import Problem


class BenchmarkResult:
    methods: list[Method] = None  # methods that have been benchmarked
    metrics: list[
        str
    ] = None  # an array of fields that will be assigned to each of the methods from self.methods
    problem: Problem = None  # the Problem on which the benchmarking was performed
    data: dict[Problem, dict[Method, dict[str, list[any]]]] = None

    def __init__(
        self,
        problem: Problem,
        methods: list[Method],
        metrics: list[str],
        data: dict[Problem, dict[Method, dict[str, list[any]]]] = None,
    ) -> None:
        self.problem = problem
        self.methods = methods
        if not check_metric(metrics):
            exit(1)
        self.metrics = metrics
        self.data = data

    def save(self, path: str) -> None:
        """
        Saves benchmarking data to a json file by path.
        """
        # dict[Problem, dict[Method, dict[str, list[any]]]]
        data_str = dict()
        for problem, ddict in self.data.items():
            tmp2 = dict()
            for method, dddict in ddict.items():
                tmp1 = dict()
                for target, lst in dddict.items():
                    tmp1[str(target)] = [str(val) for val in lst]
                tmp2[str(method)] = tmp1
            data_str[str(problem)] = tmp2
        with open(path, "w") as file:
            json.dump(data_str, file, indent=2)

    def send_wandb(self):
        pass

    def __str__(self) -> str:
        return ""
