import logging

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


def check_plot_metric(metric: list[str]):
    for item in metric:
        if item not in available_metrics_to_plot and item not in nn_aval_metrics:
            logging.critical(f'Unsupported metric \'{item}\' to plot. Available metrics to plot: {available_metrics_to_plot}, {nn_aval_metrics}')
            return False
    return True


def check_metric(metric: str):
    return metric in available_metrics


def check_metric(metric: list[str]):
    for item in metric:
        if item not in available_metrics:
            logging.critical(f'Unsupported metric \'{item}\'. Available metrics: {available_metrics}')
            return False
    return True
