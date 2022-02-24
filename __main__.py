"""
usage: python3 -m itz <command> [-v]

========
COMMANDS
========

diagram <model> <path>
--------------
Save a diagram of the SEM as a PNG image.

Parameters:
- path: path to output image file.

fit <model>
-----------
Fit an SEM model and print results.

Parameters:
- model: name of the model to fit.

Model options:
- long-term
- short-term-with-emissions
- short-term-no-emissions

graph <x> [<y>] <path1> [<path2>]
---------------------------------
Create a visualization of one or two variables from our dataset. Descriptive statistics will be
printed to the console.

Parameters:
- x: name of explanatory variable to graph
- y (optional): name of response variable to graph
- path1: path to first output image file
- path2 (optional): path to second output image file

If y is specified, a scatterplot with an LSRL and/or a residual plot will be produced.
If y is not specified, a histogram of will be produced.

Use -v for verbosity.
"""

import argparse
import pandas as pd

import itz


def parse_data(parse=False):
    if parse:
        itz.get_data()
    # else:
    itz_df = pd.read_csv("itz-data.csv")
    return itz_df

def make_graph(x, path1, y=None, path2=None):
    """Create a visualization of one or two variables from our dataset.
    
    Returns descriptive statistics as a dictionary.
    """
    if y is None:
        return itz.make_histogram(x, path1)
    regression_stats = itz.make_regression_plot(x, y, path1)
    residual_stats = itz.make_residual_plot(x, y, path2)
    return {**regression_stats, **residual_stats}


if __name__ == "__main__":
    _ = parse_data(parse=True)
    
    # parser = argparse.ArgumentParser(usage=__doc__)
    # parser.add_argument("-v", action="store_true")
    # subparsers = parser.add_subparsers()
    
    # diagram_parser = subparsers.add_parser("diagram")
    # diagram_parser.add_argument("path")
    # diagram_parser.set_defaults(func=itz.make_sem_diagram)

    # fit_parser = subparsers.add_parser("fit")
    # fit_parser.add_argument("model", choices=["long-term", "short-term-with-emissions",
    #                                           "short-term-no-emissions"])
    # fit_parser.set_defaults(func=lambda model: itz.fit(itz.get_description(model)))

    # graph_parser = subparsers.add_parser("graph")
    # graph_parser.add_argument("x", choices=itz.data.VAR_NAMES)
    # graph_parser.add_argument("y", choices=itz.data.VAR_NAMES, required=False)
    # graph_parser.add_argument("path1")
    # graph_parser.add_argument("path2", required=False)
    # graph_parser.set_defaults(func=make_graph)

    # args = parser.parse_args()
    # stats = args.func(**vars(args))
    # for key, val in stats.items():
    #     print(f"{key}: {val}")