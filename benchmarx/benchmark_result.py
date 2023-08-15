import json
import logging
import sys
import uuid

import jax
import jax.numpy as jnp
import pandas as pd

from benchmarx.metrics import Metrics, CustomMetric
from benchmarx.problem import Problem
#from benchmarx.plotter import Plotter 
from benchmarx.quadratic_problem import QuadraticProblem
from benchmarx.defaults import default_plotly_config

from typing import List, Dict, Tuple


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
        """
        Initialize BenchmarkResult instance.

        :param problem: The Problem on which the benchmarking was performed.
        :param methods: List of method names.
        :param data: Dictionary containing benchmarking data.
        """
        self.problem = problem
        self.methods = methods
        self.data = data
    
    @staticmethod
    def _convert(val):
        """
        Converts val from str to appropriate type.

        :param val: Value to convert.
        :return: Converted value.
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
        Converts string representation of a matrix to a numpy array.

        :param A_str: String representation of the matrix.
        :return: Numpy array representing the matrix.

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
        Loads benchmark data from a JSON file.

        :param path: Path to the JSON file.
        :return: Loaded BenchmarkResult instance.
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
        Saves benchmarking data to a JSON file.

        :param path: Path to save the JSON file.
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

    def get_dataframes(self, df_metrics: List[str | CustomMetric]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Create DataFrames from rows.

        :param df_metrics: List of metrics to put in columns.
        :return: Tuple of DataFrames and list of successfully calculated metrics.

        Create DataFrame form rows.
        Returns tuple of:
            [0] dictionary {problem: problem's DataFrame}.
            [1] A list of metrics that have been successfully calculated.
        Only successfully calculated metrics will be plotted.
        df_metrics -- metrics to put in columns, subset of
        Metrics.metrics_to_plot or your CustomMetric objects
        """
        result_dict = {}
        problem_rows = {}

        success_metrics = []

        for problem, problem_data in self.data.items():
            x_opt = None
            f_opt = None
            if "A" in problem_data:
                problem_data.pop("A")
            if "b" in problem_data:
                problem_data.pop("b")

            # after these manipulations either x_opt is not None, or it can no longer be counted
            if "x_opt" in problem_data:
                x_opt = self._convert(problem_data["x_opt"])
                problem_data.pop("x_opt")
            elif isinstance(self.problem, Problem) and hasattr(self.problem, "x_opt"):
                x_opt = self.problem.x_opt

            # after these manipulations either f_opt is not None, or it can no longer be counted
            if "f_opt" in problem_data:
                f_opt = self._convert(problem_data["f_opt"])
                problem_data.pop("f_opt")
            elif isinstance(self.problem, Problem):
                if hasattr(self.problem, "f_opt"):
                    f_opt = self.problem.f_opt
                elif hasattr(self.problem, "x_opt"):
                    f_opt = self.problem.f(self.problem.x_opt)

            rows = []

            for method, method_data in problem_data.items():
                if "hyperparams" in method_data and "runs" in method_data:
                    hyperparams = method_data["hyperparams"]
                    runs = method_data["runs"]
                    nit = int(runs["run_0"]["nit"][0])

                    for iteration in range(nit):
                        row = {
                            "Problem": problem,
                            "Method": method_data["hyperparams"]["label"],
                            "Hyperparameters": hyperparams,
                            "Iteration": iteration,
                        }
                        if x_opt is not None:
                            row["x_opt"] = x_opt
                        if f_opt is not None:
                            row["f_opt"] = f_opt

                        to_means_stds = {}
                        for run, run_data in runs.items():
                            run_num = int(run[4:])
                            for df_metric in df_metrics:
                                df_metric_key = str(df_metric)  # string representation of metric (label)
                                if df_metric_key not in success_metrics:
                                    success_metrics.append(df_metric_key)
                                custom_df_metric_flag = isinstance(df_metric, CustomMetric)
                                if isinstance(df_metric, str):
                                    if df_metric not in Metrics.metrics_to_plot and df_metric not in Metrics.model_metrics_to_plot:
                                        logging.warning(
                                            f"Unknown metric '{df_metric}'. Use CustomMetric to specify your own metric."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)

                                val = None
                                if df_metric_key in run_data.keys():
                                    val = run_data[df_metric_key][iteration]
                                elif custom_df_metric_flag:
                                    try:
                                        val = df_metric.func(
                                            run_data["x"][iteration]
                                        )
                                    except:
                                        logging.warning(
                                            msg=f"Something went wrong while calculating the CustomMetric '{df_metric_key}'."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                elif df_metric_key == "x_gap":
                                    if x_opt is not None:
                                        val = float(jnp.linalg.norm(run_data["x"][iteration] - x_opt))
                                    else:
                                        logging.warning(
                                            msg=f"Cannot calculate x_gap because there is no information about x_opt."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                elif df_metric_key == "f_gap":
                                    if f_opt is not None:
                                        if "f" in run_data.keys():
                                            val = jnp.abs(run_data["f"][iteration] - f_opt)
                                        elif isinstance(self.problem, Problem):
                                            val = jnp.abs(self.problem.f(run_data["x"][iteration]) - f_opt)
                                        else:
                                            logging.warning(
                                                msg="Cannot calculate f_gap because there is impossible to compute f_opt."
                                            )
                                            if df_metric_key in success_metrics:
                                                success_metrics.remove(df_metric_key)
                                    else:
                                        logging.warning(
                                            msg="Cannot calculate f_gap because there is impossible to compute f_opt."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                elif df_metric_key == "grad_norm":
                                    if "grad" in run_data.keys():
                                        val = float(jnp.linalg.norm(run_data["grad"][iteration]))
                                    elif isinstance(self.problem, Problem):
                                        if hasattr(self.problem, "grad"):
                                            val = float(jnp.linalg.norm(self.problem.grad(run_data["x"][iteration])))
                                        else:
                                            val = float(jnp.linalg.norm(jax.grad(self.problem.f)(run_data["x"][iteration])))
                                    else:
                                        logging.warning(
                                            msg="Cannot calculate 'grad_norm' because there is no information about grad."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                elif df_metric_key == "x_norm":
                                    val = float(jnp.linalg.norm(run_data["x"][iteration]))
                                elif df_metric_key == "f":
                                    if isinstance(self.problem, Problem):
                                        val = self.problem.f(run_data["x"][iteration])
                                    else:
                                        logging.warning(
                                            msg="Cannot calculate 'f' metric because there is no information about problem."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                elif df_metric_key == "relative_x_gap":
                                    if x_opt is not None:
                                        if jnp.linalg.norm(x_opt) != 0:
                                            val = jnp.linalg.norm(run_data["x"][iteration] - x_opt) / jnp.linalg.norm(x_opt)
                                        else:
                                            logging.warning(
                                                msg="Cannot calculate 'relative_x_gap' metric because ||x_opt|| = 0."
                                            )
                                            if df_metric_key in success_metrics:
                                                success_metrics.remove(df_metric_key)
                                    else:
                                        logging.warning(
                                            msg="Cannot calculate 'relative_x_gap' metric because there is no information about x_opt."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)
                                
                                elif df_metric_key == "relative_f_gap":
                                    if f_opt is not None:
                                        if f_opt != 0:
                                            if "f" in run_data.keys():
                                                val = jnp.abs(run_data["f"][iteration] - f_opt) / jnp.abs(f_opt)
                                            elif isinstance(self.problem, Problem):
                                                f_val = self.problem.f(run_data["x"][iteration])
                                                val = jnp.abs(f_val - f_opt) / jnp.abs(f_opt)
                                            else:
                                                logging.warning(
                                                    msg="Cannot calculate 'relative_f_gap' metric because there is no information about f."
                                                )
                                                if df_metric_key in success_metrics:
                                                    success_metrics.remove(df_metric_key)
                                        else:
                                            logging.warning(
                                                msg="Cannot calculate 'relative_f_gap' metric because f_opt = 0."
                                            )
                                            if df_metric_key in success_metrics:
                                                success_metrics.remove(df_metric_key)
                                    else:
                                        logging.warning(
                                            msg="Cannot calculate 'relative_f_gap' metric because there is no information about f_opt."
                                        )
                                        if df_metric_key in success_metrics:
                                            success_metrics.remove(df_metric_key)

                                row[df_metric_key + "_" + str(run_num)] = val
                                if df_metric_key in to_means_stds.keys():
                                    to_means_stds[df_metric_key].append(val)
                                else:
                                    to_means_stds[df_metric_key] = [val]

                        for metric, val1 in to_means_stds.items():
                            row[metric + "_mean"] = jnp.mean(jnp.array(val1), axis=0)
                            row[metric + "_std"] = jnp.std(jnp.array(val1), axis=0)

                        rows.append(row)
            problem_rows[problem] = rows

        for problem, rows in problem_rows.items():
            result_dict[problem] = pd.DataFrame(rows)

        return result_dict, success_metrics

    def plot(
        self,
        metrics: List[str | CustomMetric] = [],
        plotly_config=default_plotly_config,
        write_html: bool = False,
        path_to_write: str = "",
        include_plotlyjs: str = "cdn",
        full_html: bool = False
    ) -> None:
        """
        Plot benchmarking data.

        :param metrics: List of metrics to plot.
        :param plotly_config: Plotly configuration.
        :param write_html: Flag to write HTML plot.
        :param path_to_write: Path to write HTML plot.
        :param include_plotlyjs: Option for including Plotly JS.
        :param full_html: Flag for full HTML plot.
        """
        from benchmarx.plotter import Plotter 
        plotter = Plotter(
            benchmark_result=self
        )
        plotter.plot(
            metrics=metrics,
            plotly_config=plotly_config,
            write_html=write_html,
            path_to_write=path_to_write,
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
