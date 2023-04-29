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
    'fs_dist_to_opt',
    'xs_dist_to_opt',
    'grads_norm'
]


def check_plot_metric(metric: str):
    return metric in available_metrics_to_plot


def check_plot_metric(metric: list[str]):
    for item in metric:
        if item not in available_metrics_to_plot:
            print(f'Unsupported metric \'{item}\' to plot. Available metrics to plot: {available_metrics_to_plot}')
            return False
    return True


def check_metric(metric: str):
    return metric in available_metrics


def check_metric(metric: list[str]):
    for item in metric:
        if item not in available_metrics:
            print(f'Unsupported metric \'{item}\'. Available metrics: {available_metrics}')
            return False
    return True
