
import json
import jax.numpy as jnp


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


    def plot(self, save: bool = True):
        """
        Create plots according to the self.metrics. Saves to
        dir_path if save is True.
        """
        raw_data = dict()
        with open(self.data_path) as json_file:
            raw_data = json.load(json_file)

        for metric in self.metrics:
            if metric == 'fs':
                pass
            if metric == 'xs_norm':
                pass
            if metric == 'fs_dist_to_opt':
                pass
            if metric == 'xs_dist_to_opt':
                pass
            if metric == 'grads_norm':
                pass




def test_local():
    pass



if __name__ == '__main__':
    test_local()