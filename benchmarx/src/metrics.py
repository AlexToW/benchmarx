import logging
from typing import List, Callable

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