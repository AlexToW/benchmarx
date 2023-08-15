import pandas as pd
import plotly
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"
import plotly.graph_objects as go
import os
import pathlib
import json
import logging
import jax.numpy as jnp
from typing import List, Dict

from benchmarx.defaults import default_plotly_config, default_log_threshold
from benchmarx.metrics import Metrics, CustomMetric

from benchmarx.benchmark_result import BenchmarkResult


class Plotter:
    """
    Class for plotting benchmark results using Plotly.

    Attributes:
        benchmark_result (BenchmarkResult): The benchmark result object.
    """
    benchmark_result: BenchmarkResult

    def __init__(self, benchmark_result) -> None:
        self.benchmark_result = benchmark_result

    def plotly_figure(
        self, dataframe: pd.DataFrame, dropdown_options: List[Dict[str, str]]
    ) -> go.Figure:
        """
        Create a Plotly figure for plotting benchmark results.

        Args:
            dataframe (pd.DataFrame): DataFrame containing benchmark data.
            dropdown_options (List[Dict[str, str]]): Dropdown options for selecting metrics.

        Returns:
            go.Figure: The Plotly figure.
        """
        markers = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "triangle-ne",
            "triangle-se",
            "triangle-sw",
            "triangle-nw",
        ]
        colors_rgba = [
            "rgba(31, 119, 180,  1)",
            "rgba(255, 127, 14,  1)",
            "rgba(44, 160, 44,   1)",
            "rgba(214, 39, 40,   1)",
            "rgba(148, 103, 189, 1)",
            "rgba(140, 86, 75,   1)",
            "rgba(227, 119, 194, 1)",
            "rgba(127, 127, 127, 1)",
            "rgba(188, 189, 34,  1)",
            "rgba(23, 190, 207,  1)",
        ]
        colors_rgba_faint = [
            "rgba(31, 119, 180,  0.3)",
            "rgba(255, 127, 14,  0.3)",
            "rgba(44, 160, 44,   0.3)",
            "rgba(214, 39, 40,   0.3)",
            "rgba(148, 103, 189, 0.3)",
            "rgba(140, 86, 75,   0.3)",
            "rgba(227, 119, 194, 0.3)",
            "rgba(127, 127, 127, 0.3)",
            "rgba(188, 189, 34,  0.3)",
            "rgba(23, 190, 207,  0.3)",
        ]
        fig = go.Figure()

        # Add traces for each method and each dropdown option
        for i_method, method in enumerate(dataframe["Method"].unique()):
            method_df = dataframe[dataframe["Method"] == method]
            marker = dict(
                symbol=markers[i_method % len(markers)],
                color=colors_rgba[i_method % len(colors_rgba)],
            )
            fillcolor = colors_rgba_faint[i_method % len(colors_rgba_faint)]
            for option in dropdown_options:
                trace_mean = go.Scatter(
                    x=method_df["Iteration"],
                    y=method_df[option["value"] + "_mean"],
                    mode="lines+markers",
                    marker=marker,
                    hovertext=f"{method} - {option['label']}",
                    name=f"{method}",
                    visible=option["value"] == dropdown_options[0]["value"],
                )
                fig.add_trace(trace_mean)
                if not all([val == 0 for val in method_df[option["value"] + "_std"]]):
                    trace_plus_std = go.Scatter(
                        name="mean + std",
                        x=method_df["Iteration"],
                        y=method_df[option["value"] + "_mean"]
                        + method_df[option["value"] + "_std"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hovertext=f"{method} - {option['label']}_upper",
                        visible=option["value"] == dropdown_options[0]["value"],
                    )
                    fig.add_trace(trace_plus_std)

                    trace_minus_std = go.Scatter(
                        name="mean - std",
                        x=method_df["Iteration"],
                        y=[max(val, default_log_threshold) for val in method_df[option["value"] + "_mean"]- method_df[option["value"] + "_std"]],
                        line=dict(width=0),
                        mode="lines",
                        fillcolor=fillcolor,
                        fill="tonexty",
                        showlegend=False,
                        hovertext=f"{method} - {option['label']}_lower",
                        visible=option["value"] == dropdown_options[0]["value"],
                    )
                    fig.add_trace(trace_minus_std)
        # Update layout
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "method": "update",
                            "label": option["label"],
                            "args": [
                                {
                                    "visible": [
                                        trace.hovertext.endswith(
                                            f" - {option['value']}"
                                        )
                                        or trace.hovertext.endswith(
                                            f" - {option['value']}_upper"
                                        )
                                        or trace.hovertext.endswith(
                                            f" - {option['value']}_lower"
                                        )
                                        for trace in fig.data
                                    ]
                                }
                            ],
                            "args2": [
                                {"yaxis": {"title": option["label"], "type": "log"}}
                            ],
                        }
                        for option in dropdown_options
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": -0.14,
                    "xanchor": "left",
                    "y": 1.2,
                    "yanchor": "top",
                }
            ],
            xaxis={"title": "Iteration"},
            yaxis={"title": "", "type": "log"},
            title=str(dataframe.T[0]["Problem"]),  # Set your problem title here
        )

        fig.update_layout(
            dragmode="pan",
            title={
                "x": 0.5,
                "xanchor": "center",
            },
        )

        return fig

    def plot(
        self,
        metrics: List[str | CustomMetric] = [],
        plotly_config=default_plotly_config,
        write_html: bool = False,
        path_to_write: str = "",
        include_plotlyjs: str = "cdn",
        full_html: bool = False,
    ) -> None:
        """
        Plot benchmark results using Plotly.

        Args:
            metrics (List[str | CustomMetric]): Metrics to plot.
            plotly_config: Plotly config.
            write_html (bool): If True, write an HTML file.
            path_to_write (str): Path to write the HTML file.
            include_plotlyjs (str): Include Plotly JS in the HTML.
            full_html (bool): Create a full HTML file.

        Returns:
            None
        """

        # Metrics.model_metrics_to_plot are tracking in Benchmark, if the problem
        # inherits from ModelProblem. Thus if the problem inherits from ModelProblem,
        # metrics Metrics.model_metrics_to_plot are always will be "successful", 
        # i.e. this metrics will be in good_str_metrics returned from benchmark_result.get_dataframes
        # (if the problem inherits from ModelProblem), and vice versa: 
        # if the problem does NOT inherit from ModelProblem, Metrics.model_metrics_to_plot
        # will not be contained in good_str_metrics.
        # Therefore, by adding Metrics.model_metrics_to_plot to df_metrics for passing to
        # benchmark_result.get_dataframes, Metrics.model_metrics_to_plot will be plotted in case 
        # if problem inherits from ModelProblem, and will NOT be plotted in case if problem does NOT
        # inherit from ModelProblem.
        dfs, good_str_metrics = self.benchmark_result.get_dataframes(df_metrics=metrics + Metrics.model_metrics_to_plot)
        for _, df in dfs.items():
            dropdown_options = [
                {"label": metric, "value": metric} for metric in good_str_metrics
            ]
            figure = self.plotly_figure(dataframe=df, dropdown_options=dropdown_options)
            figure.show(config=plotly_config)

            if write_html:
                figure.write_html(
                    path_to_write,
                    config=plotly_config,
                    include_plotlyjs="cdn",
                    full_html=False,
                )
