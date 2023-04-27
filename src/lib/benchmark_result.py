import json
import wandb

import jax
import jax.numpy as jnp
from metrics import *
from problem import Problem
import metrics


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
                hyper_dict = dict()
                runs_dict = dict()
                for field, d in dddict.items():
                    # field is 'hyperparams' or 'runs'.
                    if str(field) == 'hyperparams':
                        hyper_dict = {key: str(val) for key, val in d.items()}

                    if str(field) == 'runs':
                        for run_num, run_dict in d.items():
                            run_num_dict_str = dict()
                            metric_lst_str = list()
                            for metric, metric_lst in run_dict.items():
                                metric_lst_str = [str(val) for val in metric_lst]
                                if len(metric_lst_str) > 1:
                                    run_num_dict_str[str(metric)] = metric_lst_str
                                else:
                                    run_num_dict_str[str(metric)] = metric_lst_str[0]
                            runs_dict[str(run_num)] = run_num_dict_str
                    
                    tmp1 = {'hyperparams': hyper_dict, 'runs': runs_dict}
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
        history_f_const = list()
        for item in self.data[self.problem]['GRADIENT_DESCENT_const_step']['history_x']:
            l = jnp.array(item)
            p = self.problem.f(jnp.array([l[i] for i in range(len(l))]))
            history_f_const.append(p)
        
        history_f_adapt = list()
        for item in self.data[self.problem]['GRADIENT_DESCENT_adaptive_step']['history_x']:
            l = jnp.array(item)
            p = self.problem.f(jnp.array([l[i] for i in range(len(l))]))
            history_f_adapt.append(p)
        
        data = [[x, y] for (x, y) in zip(range(1, len(history_f_const) + 1), history_f_const)]
        for item in data:
            wandb.log({'GD_const' : {'f' : item[1]}})

        wandb.finish()
        

    def __str__(self) -> str:
        return ""

