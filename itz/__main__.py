"""
usage: python3 -m itz <command> [-v]

========
COMMANDS
========

Model options:
- LONG_TERM
- SHORT_TERM_WITH_EMISSIONS
- SHORT_TERM_NO_EMISSIONS

Variable names specified in itz.data.VAR_NAMES

diagram <model> <path>
----------------------
Save a diagram of the SEM as a PNG image.

Parameters:
- model: name of SEM model to create diagram of.
- path: path to output image file.

fit <model> <model_path> <data_path>
------------------------------------
Fit an SEM model and print results.

Parameters:
- model: name of the model to fit.
- model_path: path to file to store model estimated covariance matrix.
- data_path: path to CSV with data for model.

graph <x> <img_path1> <data_path> [--y Y_VAR_NAME] [--img_path2 IMG_PATH2]
--------------------------------------------------------------------------
Create a visualization of one or two variables from our dataset. Descriptive
statistics will be printed to the console.

Parameters:
- x: name of explanatory variable to graph
- img_path1: path to first output image file
- data_path: path to dataset CSV.
- y (optional): name of response variable to graph
- path2 (optional): path to second output image file

If y is specified, a scatterplot with an LSRL and/or a residual plot will be produced.
If y is not specified, a histogram of will be produced.

parse [--lot_data_path LOT_DATA_PATH] [--tract_data_paths TRACT_DATA_PATHS] output_path
---------------------------------------------------------------------------------------
Parse and save data for SEM models.

Parameters:
- output_path: path to directory in which CSV files will be outputted.
- lot_data_path (optional): path to CSV file containing pre-parsed lot data.
- tract_data_paths (optional): paths to CSV files containing pre-parsed tract data.

Use -v for verbosity.
"""

from typing import Dict, List
import argparse
import os

import numpy as np
import pandas as pd

import itz


def _make_diagram(model_string: str, img_path: str, verbose):
    """Makes an SEM diagram.
    """
    model = itz.model.ModelName.__dict__[model_string]
    itz.make_sem_diagram(model, img_path)


def _fit(model_string: str, model_path: str, data_path: str, verbose):
    """Fits a model to the data and prints evaluation metrics.
    """
    model_name = itz.model.ModelName.__dict__[model_string]
    data = pd.read_csv(data_path)
    model = itz.fit(itz.get_description(model_name), data, verbose)
    np.savetxt(model_path, model.calc_sigma()[0], delimiter=",")
    print(itz.evaluate(model))


def _make_graph(x, data_path, img_path1, y, img_path2, verbose, log_x=False, log_y=False) -> Dict[str, float]:
    """Create a visualization of one or two variables from our dataset.
    
    Prints descriptive statistics.
    """
    data = pd.read_csv(data_path)
    if y is None:
        if x == 'all_vars':
            try:
                os.mkdir("histogram-data/")
            except FileExistsError:
                pass
            for column in data.columns:
                if column.startswith("Unnamed") or column in ["all_vars"]:
                    continue
                print("working on:", column)
                itz.make_histogram(column, data, "histogram-data/"+column+".png")
                print("Histogram created:", column)
            return
        else:
            if img_path1:
                return itz.make_histogram(x, data, img_path1)
            else:
                return itz.make_histogram(x, data, x+".png")
    if img_path1:
        regression_stats = itz.make_regression_plot(x, y, data, img_path1, log_x, log_y)
    else:
        try:
            os.mkdir("regression-plots")
        except FileExistsError:
            pass
        if y == "all_vars":
            for column in data.columns:
                if column.startswith("Unnamed") or column in ["all_vars"]:
                    continue
                print("working on:", x, column)
                try:
                    if log_x:
                        os.mkdir("regression-plots/"+column+"/")
                    else:
                        os.mkdir("regression-plots/"+column+"/")
                except FileExistsError:
                    pass
                if log_x and log_y:
                    regression_stats = itz.make_regression_plot(x, column, data, "regression-plots/"+column+"/"+"log_"+x+"_"+"log_"+column+".png", log_x, log_y)
                elif log_x:
                    regression_stats = itz.make_regression_plot(x, column, data, "regression-plots/"+column+"/"+"log_"+x+"_"+column+".png", log_x, log_y)
                elif log_y:
                    regression_stats = itz.make_regression_plot(x, column, data, "regression-plots/"+column+"/"+x+"_"+"log_"+column+".png", log_x, log_y)
                else:
                    regression_stats = itz.make_regression_plot(x, column, data, "regression-plots/"+column+"/"+x+"_"+column+".png", log_x, log_y)
        else:
            if log_x and log_y:
                regression_stats = itz.make_regression_plot(x, y, data, "regression-plots/"+"log_"+x+"_"+"log_"+y+".png", log_x, log_y)
            elif log_x:
                regression_stats = itz.make_regression_plot(x, y, data, "regression-plots/"+"log_"+x+"_"+y+".png", log_x, log_y)
            elif log_y:
                regression_stats = itz.make_regression_plot(x, y, data, "regression-plots/"+x+"_"+"log_"+y+".png", log_x, log_y)
            else:
                regression_stats = itz.make_regression_plot(x, y, data, "regression-plots/"+x+"_"+y+".png", log_x, log_y)
    if img_path2:
        residual_stats = itz.make_residual_plot(x, y, data, img_path2, log_x, log_y)
        stats = {**regression_stats, **residual_stats}
        for key, val in stats.items():
                print(f"{key}: {val}")
        return stats
    else:
        # pass
        # **regression_stats is not a thing???
        # print(regression_stats)
        return {"sum of squared_residuals":regression_stats[0]}


def _parse(output_path: str, lot_data_path: str, tract_data_paths: List[str], verbose):
    """Parse the raw ACS and PLUTO data into a directory of CSV files.
    """
    tract_data = (
        [pd.read_csv(path, index_col="ITZ_GEOID") for path in tract_data_paths]
        if tract_data_paths is not None else [])
    lot_data = (pd.read_csv(lot_data_path)
                if lot_data_path is  not None else None)
    lot_data, tract_data, model_data = itz.get_data(lot_data, tract_data, verbose)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    for i, tract_df in enumerate(tract_data):
        tract_df.to_csv(os.path.join(output_path, f"tract-data-{i}.csv"))
    model_data.to_csv(os.path.join(output_path, "itz-data.csv"))
    lot_data.to_csv(os.path.join(output_path, "lot-data.csv"))

def _correlate(data_path: str, output_path: str, img_path: str, verbose):
    data = pd.read_csv(data_path)
    itz.make_correlation_matrix(data, output_path, img_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers()
    
    diagram_parser = subparsers.add_parser("diagram")
    diagram_parser.add_argument("model_string")
    diagram_parser.add_argument("img_path")
    diagram_parser.set_defaults(func=_make_diagram)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("model_string", choices=itz.model.MODEL_NAMES)
    fit_parser.add_argument("model_path")
    fit_parser.add_argument("data_path")
    fit_parser.set_defaults(func=_fit)

    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("x", choices=itz.data.VAR_NAMES)
    graph_parser.add_argument("data_path")
    graph_parser.add_argument("--img_path1", required=False)
    graph_parser.add_argument("--y", choices=itz.data.VAR_NAMES, required=False)
    graph_parser.add_argument("--img_path2", required=False)
    graph_parser.add_argument("--log_x", required=False, action="store_true")
    graph_parser.add_argument("--log_y", required=False, action="store_true")
    graph_parser.set_defaults(func=_make_graph)

    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("output_path")
    parse_parser.add_argument("--lot_data_path", required=False)
    parse_parser.add_argument("--tract_data_paths", action="extend", required=False)
    parse_parser.set_defaults(func=_parse)

    correlate_parser = subparsers.add_parser("correlate")
    correlate_parser.add_argument("data_path")
    correlate_parser.add_argument("output_path")
    correlate_parser.add_argument("--img_path", required=False)
    correlate_parser.set_defaults(func=_correlate)

    args = parser.parse_args()
    print(args)
    args.func(**{key: val for key, val in vars(args).items() if key != "func"})
