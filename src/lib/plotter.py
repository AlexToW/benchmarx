# https://habr.com/ru/articles/502958/

#=======================
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

#import pandas as pd
#=======================


import json
import re
import jax.numpy as jnp

from problems.quadratic_problem import QuadraticProblem


import metrics as _metrics


class Plotter:
    metrics: list[str]      # metrics to plot
    data_path: str          # path to a json file with the necessary data
    dir_path: str          # path to directory to save plots 
    def __init__(self, metrics: list[str], data_path: str, dir_path: str = '.') -> None:
        if not _metrics.check_plot_metric(metrics):
            exit(1)
        self.metrics = metrics
        self.data_path = data_path
        self.dir_path = dir_path


    def _matrix_from_str(self, A_str: str):
        """
        A_str in format:
        "[[0.96531415 0.84779143 0.72762513]\n [0.31114805 0.03425407 0.31510842]\n [0.12594318 0.42591357 0.8050107 ]]"
        """
        #pre_raws = A_str.split('\n')
        pre_raws = [s.strip() for s in A_str.split('\n')]
        if len(pre_raws) > 0 and pre_raws[0][0] == '[' and pre_raws[0][1] == '[':
            pre_raws[0] = pre_raws[0][1:]
        if len(pre_raws) > 0 and pre_raws[-1][-1] == ']' and pre_raws[-1][-2] == ']':
            pre_raws[-1] = pre_raws[-1][:-1]
        raws = [raw[1:-1].strip() for raw in pre_raws]
        return jnp.array([jnp.fromstring(raw, sep=' ') for raw in raws])
    
    def _convert(self, val):
        """
        Converts val from str to apropriate type.
        ['[2. 1.]', '[7.5 8.]']
        ['4.5', '-0.1']
        '[1.1 -7.7]'
        '0.01'
        'MyGD'
        """

        if isinstance(val, list):
            if len(val) > 0 and isinstance(val[0], str):
                if val[0][0] == '[' and val[0][-1] == ']':
                    # val is like ['[2. 1.]', '[7.5 8.]']
                    res = list()
                    for item in val:
                        tmp = jnp.array([float(x) for x in item[1:-1].split(' ') if len(x) > 0])
                        res.append(tmp)
                    return res
                else:
                    # val is like ['4.5', '-0.1']
                    return [float(x) for x in val]
            
        elif isinstance(val, str):
            if val[0] == '[' and val[-1] == ']':
                # val is like '[2. 1.]'
                return jnp.array([float(x) for x in (val[1:-1]).split(' ') if len(x) > 0])
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
            print(f'Can\'t convert {val}')
            return 'wtf'

    def _sparse_data(self) -> dict:
        """
        Returns the dictionary from the file (data_path), in which 
        fields are converted from strings to the appropriate type.
        """
        raw_data = dict()
        with open(self.data_path) as json_file:
            raw_data = json.load(json_file)

        good_data = dict()
        for problem, problem_dict in raw_data.items():
            x_opt = None
            A = None
            b = None
            if 'A' in raw_data[problem]:
                A = self._matrix_from_str(raw_data[problem]['A'])
                raw_data[problem].pop('A')
            if 'b' in raw_data[problem]:
                b = jnp.fromstring(raw_data[problem]['b'][1:-1], sep=' ')
                raw_data[problem].pop('b')
            if 'x_opt' in raw_data[problem]:
                x_opt = self._convert(raw_data[problem]['x_opt'])
                raw_data[problem].pop('x_opt')

            problem_dict_good = dict()
            for method, method_dict in problem_dict.items():
                hyperparams_good = dict()
                runs_good = dict()
                for field, field_dict in method_dict.items():
                    # field is 'hyperparams' or 'runs'
                    if field == 'hyperparams':
                        for hyperparam, val in field_dict.items():
                            hyperparams_good[str(hyperparam)] = self._convert(val)
                    if field == 'runs':
                        for run_num, run_dict in field_dict.items():
                            tmp_run_good = dict()
                            for metric, metric_val in run_dict.items():
                                tmp_run_good[str(metric)] = self._convert(metric_val)
                            runs_good[str(run_num)] = tmp_run_good
                method_dict_good = {'hyperparams' : hyperparams_good, 'runs' : runs_good}
                problem_dict_good[str(method)] = method_dict_good

            good_data[str(problem)] = problem_dict_good
            if x_opt is not None:
                good_data[str(problem)]['x_opt'] = x_opt
            if A is not None:
                good_data[problem]['A'] = A
            if b is not None:
                good_data[problem]['b'] = b

        return good_data


    def _get_fs(self, data: dict) -> dict:
        """
        Returns dict {method: [list[func vals run_0], ... ,list[func vals run_N]]}
        """
        result = dict()
        for problem, problem_dict in data.items():
            method_trg = dict()
            for method, method_dict in problem_dict.items():
                f_vals_vals = list()
                if isinstance(method_dict, dict):
                    for run_num, run_dict in method_dict['runs'].items():
                        if 'history_f' in run_dict:
                            f_vals_vals.append(run_dict['history_f'])
                        else:
                            print('Maaaan(')
                            exit(1)
                    method_trg[method_dict['hyperparams']['label']] = f_vals_vals
                result[problem] = method_trg
        return result




    def plot(self, save: bool = True):
        """
        Create plots according to the self.metrics. Saves to
        dir_path if save is True.
        """

        data = self._sparse_data()
        for metric in self.metrics:
            if metric == 'fs':
                print(self._get_fs(data))
            if metric == 'xs_norm':
                pass
            if metric == 'fs_dist_to_opt':
                pass
            if metric == 'xs_dist_to_opt':
                pass
            if metric == 'grads_norm':
                pass


def test_local():
    plotter = Plotter(
        metrics= ['fs'],
        data_path='/Users/aleksandrtrisin/Documents/6 семестр/метопты/Benchmark_Opt/src/lib/GD_quadratic.json'
    )
    plotter.plot()
    #print(plotter._sparse_data())


if __name__ == '__main__':
    test_local()