import json
import wandb


from metrics import *
from problem import Problem


class BenchmarkResult:
    #methods: list[Method] = None  # methods that have been benchmarked
    methods: list[str] = None
    metrics: list[str] = None  # an array of fields that will be assigned to each of the methods from self.methods
    problem: Problem = None  # the Problem on which the benchmarking was performed
    #data: dict[Problem, dict[Method, dict[str, list[any]]]] = None
    data: dict[Problem, dict[str, dict[str, list[any]]]] = None
    def __init__(
        self,
        problem: Problem,
        methods: list[str],
        metrics: list[str],
        data: dict[Problem, dict[str, dict[str, list[any]]]] = None,
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
        # data: dict[Problem, dict[str, dict[str, list[any]]]]
        data_str = dict()
        for problem, ddict in self.data.items():
            tmp2 = dict()
            for method, dddict in ddict.items():
                tmp1 = dict()
                for metric, lst in dddict.items():
                    tmp1[str(metric)] = [str(val) for val in lst]
                tmp2[method] = tmp1
            data_str[str(problem)] = tmp2
        with open(path, "w") as file:
            json.dump(data_str, file, indent=2)

    def send_wandb(self):
        """
        In progress.
        """
        # 1. Start a W&B Run
        run = wandb.init(
            project="Benchmarx",
            notes="My first experiment",
            tags=["Gradient descent", "Quadratic problem"]
        )
        wandb.config = {
            "maxiter": 1000, 
            "learning_rate": 0.01, 
            "tol": 1e-5
        }
        #wan_data = dict()
        history_f_const = self.data[self.problem]['GRADIENT_DESCENT_const_step']['history_f']
        for n in range(len(history_f_const)):
            #wan_data[n] = history_f[n]
            wandb.log({"f_const" : history_f_const[n]})
        history_f_adapt = self.data[self.problem]['GRADIENT_DESCENT_adaptive_step']['history_f']
        for n in range(len(history_f_adapt)):
            wandb.log({"f_adapt" : history_f_adapt[n]})
        wandb.finish()
        

    def __str__(self) -> str:
        return ""

