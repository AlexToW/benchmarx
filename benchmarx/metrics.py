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
        "nhev",
        "time"
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
