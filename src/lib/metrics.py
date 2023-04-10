available_metrics = [
    "history_x",
    "history_f",
    "nit",
    "nfev",
    "njev",
    "nhev",
    "errors",
    "time"
]

def check_metric(metric: str):
    return metric in available_metrics

def check_metric(metric: list[str]):
    for item in metric:
        if item not in available_metrics:
            print(f'Unsupported metric \'{item}\'. Available metrics: {available_metrics}')
            return False
    return True
