import json
import wandb
import logging
import sys
import uuid

import jax
import jax.numpy as jnp

from benchmarx.src import metrics as _metrics
from benchmarx.src.metrics import CustomMetric
from benchmarx.src.problem import Problem
from benchmarx.src.plotter import Plotter 
from benchmarx.src.quadratic_problem import QuadraticProblem

from typing import List, Dict


class BenchmarkResult:
    """
    BenchmarkResult class contains benchmarking results.
    """
    methods: List[str] = None
    problem: Problem = None  # the Problem on which the benchmarking was performed
    data: Dict[Problem, Dict[str, Dict[str, List[any]]]] = None

    def __init__(
        self,
        problem: Problem = None,
        methods: List[str] = None,
        data: Dict[Problem, Dict[str, Dict[str, List[any]]]] = None,
    ) -> None:
        self.problem = problem
        self.methods = methods
        self.data = data
    
    @staticmethod
    def _convert(val):
        """
        Converts val from str to apropriate type.
        ['[2. 1.]', '[7.5 8.]']
        ['4.5', '-0.1']
        '[1.1 -7.7]'
        '0.01'
        'MyGD'
        """

        if isinstance(val, float) or isinstance(val, jnp.ndarray):
            return val

        if isinstance(val, List):
            if len(val) > 0 and isinstance(val[0], str):
                if val[0][0] == "[" and val[0][-1] == "]":
                    # val is like ['[2. 1.]', '[7.5 8.]']
                    res = list()
                    for item in val:
                        item = item.replace("\n", "")
                        tmp = jnp.array(
                            [float(x) for x in item[1:-1].split(" ") if len(x) > 0]
                        )
                        res.append(tmp)
                    return res
                else:
                    # val is like ['4.5', '-0.1']
                    return [float(x) for x in val]

        elif isinstance(val, str):
            val = val.replace("\n", "")
            if val[0] == "[" and val[-1] == "]":
                # val is like '[2. 1.]'
                return jnp.array(
                    [float(x) for x in (val[1:-1]).split(" ") if len(x) > 0]
                )
            else:
                flag = True
                try:
                    tmp = int(val)
                except:
                    flag = False
                if flag:
                    return int(val)

                flag = True
                try:
                    tmp = float(val)
                except:
                    flag = False

                if flag:
                    return float(val)

                return val
        else:
            # something went wrong
            logging.critical(f"Can't convert {val} of type {type(val)}")
            return "wtf"

    @staticmethod
    def _matrix_from_str(A_str: str):
        """
        A_str in format:
        "[[0.96531415 0.84779143 0.72762513]\n [0.31114805 0.03425407 0.31510842]\n [0.12594318 0.42591357 0.8050107 ]]"
        """
        pre_raws = A_str.split("]")
        pre_raws[0] = pre_raws[0][1:]
        pre_raws = [val.strip() + "]" for val in pre_raws if len(val) > 0]
        raws = [raw[1:-1].strip().replace("\n", "") for raw in pre_raws]
        A = jnp.array([jnp.fromstring(raw, sep=" ") for raw in raws])
        return A

    @classmethod
    def load(cls, path: str) -> None:
        """
        Loads benchmark data to self.data from json-file by path.
        """
        raw_data = dict()
        with open(path) as json_file:
            raw_data = json.load(json_file)

        good_data = dict()
        problem_obj = None
        methods = list()
        for problem, problem_dict in raw_data.items():
            x_opt = None
            A = None
            b = None
            f_opt = None
            if "A" in raw_data[problem]:
                A = BenchmarkResult._matrix_from_str(raw_data[problem]["A"])
                raw_data[problem].pop("A")
            if "b" in raw_data[problem]:
                b = jnp.fromstring(raw_data[problem]["b"][1:-1], sep=" ")
                raw_data[problem].pop("b")
            if "x_opt" in raw_data[problem]:
                x_opt = BenchmarkResult._convert(raw_data[problem]["x_opt"])
                raw_data[problem].pop("x_opt")
            if "f_opt" in raw_data[problem]:
                f_opt = BenchmarkResult._convert(raw_data[problem]["f_opt"])
                raw_data[problem].pop("f_opt")

            problem_dict_good = dict()
            for method, method_dict in problem_dict.items():
                methods.append(str(method))
                hyperparams_good = dict()
                runs_good = dict()
                for field, field_dict in method_dict.items():
                    # field is 'hyperparams' or 'runs'
                    if field == "hyperparams":
                        for hyperparam, val in field_dict.items():
                            hyperparams_good[str(hyperparam)] = BenchmarkResult._convert(val)
                    if field == "runs":
                        for run_num, run_dict in field_dict.items():
                            tmp_run_good = dict()
                            for metric, metric_val in run_dict.items():
                                tmp_run_good[str(metric)] = BenchmarkResult._convert(metric_val)
                            runs_good[str(run_num)] = tmp_run_good
                method_dict_good = {"hyperparams": hyperparams_good, "runs": runs_good}
                problem_dict_good[str(method)] = method_dict_good

            good_data[str(problem)] = problem_dict_good
            if x_opt is not None:
                good_data[str(problem)]["x_opt"] = x_opt
            if A is not None:
                good_data[problem]["A"] = A
            if b is not None:
                good_data[problem]["b"] = b
            if f_opt is not None:
                good_data[problem]["f_opt"] = f_opt
            
            if A is not None and b is not None:
                problem_obj = QuadraticProblem(
                    n=A.shape[0],
                    A=A,
                    b=b,
                    info=problem
                )
            if x_opt is not None:
                problem_obj.x_opt = x_opt
            if f_opt is not None:
                problem_obj.f_opt = f_opt

        return cls(
            problem=problem_obj,
            methods=methods,
            data=good_data
        )

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

            if self.problem is not None:
                if self.problem.x_opt is not None:
                    data_str[str(problem)]['x_opt'] = str(self.problem.x_opt)

                if self.problem.f_opt is not None:
                    data_str[str(problem)]['f_opt'] = str(self.problem.f_opt)
                
                if isinstance(problem, QuadraticProblem):
                    jnp.set_printoptions(threshold=sys.maxsize)
                    data_str[str(problem)]['A'] = jnp.array_str(self.problem.A)
                    data_str[str(problem)]['b'] = str(self.problem.b)

        with open(path, "w") as file:
            json.dump(data_str, file, indent=2)


    def plot(self, 
             metrics_to_plot = ['f', 'x_norm', 'f_gap', 'x_gap', 'grad_norm'], 
             data_path = '', 
             dir_path = '.',
             save: bool = True,
             show: bool = False,
             log: bool = True,
             fname_append: str = ''):
        """
        Plot metrics using matplotlib.pyplot. 
        metrics_to_plot: List[str | CustomMetric], metrics to plot. String metrics must be from Metrics.available_metrics_to_plot
        """
        res_file_path = str(uuid.uuid4())
        if len(data_path) > 0:
            res_file_path = data_path
        self.save(res_file_path)
        plotter_ = Plotter(metrics=metrics_to_plot + self.custom_metrics, data_path=res_file_path, dir_path=dir_path)
        plotter_.plot(save=save, show=show, log=log, fname_append=fname_append)

    def __str__(self) -> str:
        return ""

