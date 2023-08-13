import logging
from typing import List, Callable



class Metrics:
    """
    Metrics to track as the method runs and to save to a json file:

    [1]	x	    every json-file must include this metric
    [2]	f	    objective function value 
    [3]	grad	gradient of the objective function
    [4]	nit 	total number of iteration -- every json-file must include 
                        this metric!
    [5]	nfev 	number of the objective function evaluations
    [6]	njev 	number of the objective function's gradient evaluations
    [7]	nhev	number of the objective function's hessian evaluation
    [8] time    At the k-th iteration, the value of the time metric is the 
                time elapsed from the start of the method to the k-th iteration.

    Metrics to plot: 
    (to plot a metric, it is not necessary for the corresponding 
    metric to be tracked over the course of the method. 
    Each of these metrics can be calculated along the method trajectory.)

        label		value
    [1]	x_gap		||x-x_opt||
    [2] 	f		f
    [3]	f_gap		|f-f_opt|
    [4]	grad_norm	||grad||
    [5]	x_norm		||x||

    Note: if one of the compulsory metrics to track is not 
    contained in the list of metrics to track (it is set when 
    the benchmark object is created) then this compulsory metric 
    will be added to metrics to track.
    """
    compulsory_metrics_to_track = [
        "x",
        "nit"
    ]
    metrics_to_track = [
        "x",
        "f",
        "grad",
        "nit",
        "nfev",
        "njev",
        "nhev"
    ]
    metrics_to_plot = [
        "x_gap",
        "f",
        "f_gap",
        "grad_norm",
        "x_norm"
    ]
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def check_metrics_to_track(metrics_to_check: List[str]):
        for metric in metrics_to_check:
            if metric not in Metrics.metrics_to_track:
                logging.warning(
                    msg=f"Metric '{metric}' is not contained in default metrics_to_track list. Use CustomMetric instead to specify your own metric."
                )
    
    @staticmethod
    def fix_metrics_to_track(metrics_to_fix: List[str]) -> List[str]:
        """
        Add compulsory metrics and remove metrics that are 
        not contained in self.metrics_to_track.
        """
        fixed_metrics = Metrics.compulsory_metrics_to_track
        for metric in metrics_to_fix:
            if metric in Metrics.metrics_to_track:
                fixed_metrics.append(metric)
        
        return fixed_metrics

    @staticmethod
    def check_metrics_to_plot(metrics_to_check: List[str]):
        for metric in metrics_to_check:
            if metric not in Metrics.metrics_to_plot:
                logging.warning(
                    msg=f"Metric '{metric}' is not contained in default metrics_to_plot list. Use CustomMetric instead to specify your own metric."
                )


class CustomMetric:
    """
    CustomMetric class allows to compute your own metric to plot.
    It is assumed that the metric is calculated using points from 
    the method trajectory, i.e. function func is real-valued and 
    takes as an argument a point from R^d, where d is dimensionality 
    of the problem. 
    The step parameter is responsible for the frequency of func 
    calculation. 
    self.func will be calculated at each iteration whose number 
    is a multiple of self.step.
    """
    def __init__(self, func: Callable, label: str, step: int = 1) -> None:
        self.func = func
        self.label = label
        self.step = step
    
    def __str__(self) -> str:
        return self.label


# metrics that will be tracked as the method 
# runs and saved to a json file
available_metrics = [
    "history_x",
    "history_f",
    "history_df",
    "nit",
    "nfev",
    "njev",
    "nhev",
    "errors",
    "time"
]


# metrics to plot 

aval_metric_to_df_metric = {
    "history_x" : "x",
    "history_f" : "Function value",
    "history_df" : "Gradient"
}

df_metric_to_aval_metric = {
    "x" : "history_x",
    "Function value" : "history_f",
    "Gradient" : "history_df"
}

dataframe_metrics = [
    "x",
    "Solution norm",
    "Distance to the optimum",
    "Optimal solution",
    "Function value",
    "Primal gap",
    "Gradient",
    "Gradient norm"
]

available_metrics_to_plot = [
    'fs',
    'xs_norm',
    'f_gap',       # TODO: rename to f_gap
    'x_gap',       # TODO: rename to x_gap
    'grads_norm'
]

nn_aval_metrics = [
    'test_acc',
    'train_acc',
    'test_loss',
    'train_loss'
]


def check_plot_metric(metric: str):
    return metric in available_metrics_to_plot or metric in nn_aval_metrics


def check_plot_metric(metric: List[str]):
    for item in metric:
        if item not in available_metrics_to_plot and item not in nn_aval_metrics:
            logging.critical(f'Unsupported metric \'{item}\' to plot. Available metrics to plot: {available_metrics_to_plot}, {nn_aval_metrics}')
            return False
    return True


def check_metric(metric: str):
    return metric in available_metrics


def check_metric(metric: List[str]):
    for item in metric:
        if item not in available_metrics:
            logging.critical(f'Unsupported metric \'{item}\'. Available metrics: {available_metrics}')
            return False
    return True
