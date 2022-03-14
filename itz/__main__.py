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

diagram <model> <data_path> <img_path>
--------------------------------------
Save a diagram of the SEM as a PNG image.

Parameters:
- model: name of SEM model to create diagram of.
- data_path: path to dataset CSV.
- img_path: path to output image file.

fit <model> <model_path> <data_path> [--cov_mat_path COV_MAT_PATH]
------------------------------------------------------------------
Fit an SEM model and print results.

Parameters:
- model: name of the model to fit.
- model_path: path to file to store mode.
- data_path: path to CSV with data for model.
- cov_mat_path (optional): path to file to store covariance matrix (CSV).

regress <x> <y> <data_path> [--regression_plot_path PATH1] [--residual_plot_path PATH2] [--histogram_path PATH3] [--log_x] [--log_y]
------------------------------------------------------------------------------------------------------------------------------------
Create visualizations for a regression between two variables. Descriptive statistics will be
printed to the console.

Parameters:
- x: name of explanatory variable (can also be all_vars)
- y: name of response variable (can also be all_vars)
- data_path: path to dataset CSV.
- regression_plot_path (optional): path to regression plot
- residual_plot_path (optional): path to residual plot
- histogram_path (optional): path to histogram of residuals
- log_x: log-transforms x
- log_y: log-transforms y

distribute <x> <data_path> [--img_path IMG_PATH] [--log]
--------------------------------------------------------
Create a histogram of a variable.

Parameters:
- x: name of variable to graph (can also be all_vars)
- data_path: path to dataset CSV.
- img_path: path to histogram
- log: log-transforms x

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
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

import itz


def _print_stats(dict_: Dict[str, float]):
    """Prints a dictionary of statistics.
    """
    for key, val in dict_.items():
        print(f"{key}: {round(val, 3)}")


def _make_diagram(model_string: str, data_path: str, img_path: str, verbose: bool):
    """Makes an SEM diagram.
    """
    model_name = itz.model.ModelName.__dict__[model_string]
    data = pd.read_csv(data_path)
    itz.make_sem_diagram(model_name, data, img_path, verbose)


def _fit(model_string: str, model_path: str, data_path: str, cov_mat_path: str, verbose: bool):
    """Fits a model to the data and prints evaluation metrics.
    """
    model_name = itz.model.ModelName.__dict__[model_string]
    if verbose:
        print("Loading data... ", end="")
        sys.stdout.flush()
    data = pd.read_csv(data_path)
    if verbose:
        print("done!")
    model = itz.fit(*itz.get_description(model_name, data, verbose), data, verbose)
    # TODO: figure out how to save/load a model
    if cov_mat_path is not None:
        np.savetxt(cov_mat_path, model.calc_sigma()[0], delimiter=",")
    if verbose:
        print("Evaluating model...", end="")
        sys.stdout.flush()
    stats, params = itz.evaluate(model)
    if verbose:
        print("done!")
    _print_stats(stats)
    print(params)


def _make_histogram(x: str, data_path: str, img_path: str, log: bool, verbose: bool):
    """Visualize the distribution of a variable.
    """
    data = pd.read_csv(data_path)
    data.set_index("ITZ_GEOID", inplace=True)

    if x == 'all_vars':
        try:
            os.mkdir("histogram-data/")
        except FileExistsError:
            pass
        for column in data.columns:
            if column.startswith("Unnamed") or column == "all_vars":
                continue
            if verbose:
                print("working on:", column)
            itz.make_histogram(column, data, "histogram-data/" + column + ".png")
            if verbose:
                print("Histogram created:", column)
        return
    else:
        histogram_path = img_path if img_path else x + ".png"
        transformation = ((lambda x_: math.log(x_ + itz.util.LOG_TRANSFORM_SHIFT)) if log
                          else (lambda x_: x_))
        _print_stats(itz.make_histogram(x, data, histogram_path, transformation))


def _make_regression(x, y, data_path, regression_plot_path, residual_plot_path, histogram_path,
        log_x, log_y, verbose):
    """Create a regression visualization.
    """
    data = pd.read_csv(data_path)
    data.set_index("ITZ_GEOID", inplace=True)

    if regression_plot_path:
        regression_stats = itz.make_regression_plot(x, y, data, regression_plot_path, log_x, log_y)
    else:
        try:
            os.mkdir("regression-plots")
        except FileExistsError:
            pass
        if x == "all_vars" and y == "all_vars":
            # for x_column in itz.data.CONTROL_VARS.extend(["2011_2019_percent_upzoned","2016_2019_percent_upzoned","2011_2016_percent_upzoned"]):
            for x_column in itz.data.INDEPENDENT_VARS:
                for y_column in itz.data.DEPENDENT_VARS:
                    if verbose:
                        print("\nworking on:", x_column, y_column)
                    try:
                        os.mkdir("regression-plots/"+y_column+"/")
                    except FileExistsError:
                        pass
                    if log_x and log_y:
                        regression_stats = itz.make_regression_plot(x_column, y_column, data, "regression-plots/"+y_column+"/"+"log_"+x_column+"_"+"log_"+y_column+".png", log_x, log_y)
                    elif log_x:
                        regression_stats = itz.make_regression_plot(x_column, y_column, data, "regression-plots/"+y_column+"/"+"log_"+x_column+"_"+y_column+".png", log_x, log_y)
                    elif log_y:
                        regression_stats = itz.make_regression_plot(x_column, y_column, data, "regression-plots/"+y_column+"/"+x_column+"_"+"log_"+y_column+".png", log_x, log_y)
                    else:
                        regression_stats = itz.make_regression_plot(x_column, y_column, data, "regression-plots/"+y_column+"/"+x_column+"_"+y_column+".png", log_x, log_y)
        elif y == "all_vars":
            for column in data.columns:
                if column.startswith("Unnamed") or column in ["all_vars"]:
                    continue
                print("working on:", x, column)
                try:
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
    
    if residual_plot_path:
        resid_plot_stats = itz.make_residual_plot(x, y, data, residual_plot_path, log_x, log_y)
        regression_stats = {**regression_stats, **resid_plot_stats}
    
    if histogram_path:
        func = itz.util.regress(x, y, data, log_x, log_y)[-1]
        residual_df = pd.DataFrame()
        residual_df["resids"] = data[x].transform(func) - data[y]
        resid_hist_stats = itz.make_histogram("resids", residual_df, histogram_path)
        original_keys = list(resid_hist_stats.keys())
        for key in original_keys:
            resid_hist_stats["resid " + key] = resid_hist_stats[key]
            del resid_hist_stats[key]
        regression_stats = {**regression_stats, **resid_hist_stats}
    
    if verbose:
        _print_stats(regression_stats)


def _parse(output_path: str, itz_data_path: str, lot_data_path: str, tract_data_paths: List[str], verbose: bool):
    """Parse the raw ACS and PLUTO data into a directory of CSV files.
    Integrates it with generated greenspace data. 
    """
    if not itz_data_path:
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
    else:
        model_data = pd.read_csv(itz_data_path, index_col="ITZ_GEOID")
    orthoimagery_2010 = pd.read_csv("in-the-zone-data/2010-greenspace-orthoimagery.csv")
    orthoimagery_2010.set_index("ITZ_GEOID", inplace=True) 
    orthoimagery_2018 = pd.read_csv("in-the-zone-data/2018-greenspace-orthoimagery.csv") 
    orthoimagery_2018.set_index("ITZ_GEOID", inplace=True) 
    distance_to_park = pd.read_csv("in-the-zone-data/tract_distance_from_park.csv") 
    distance_to_park.set_index("ITZ_GEOID", inplace=True) 
    for index, _ in model_data.iterrows():
        try:
            model_data.loc[index, "orig_feet_distance_to_park"] = distance_to_park.loc[index, "2010_distance_from_park"]
            model_data.loc[index, "d_2010_2018_feet_distance_to_park"] = distance_to_park.loc[index, "d_2010_2018_distance_from_park"]
            model_data.loc[index, "orig_square_meter_greenspace_coverage"] = orthoimagery_2010.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]
            model_data.loc[index, "d_2010_2018_square_meter_greenspace_coverage"] = orthoimagery_2018.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]-orthoimagery_2010.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]
        except:
            model_data.loc[index, "orig_feet_distance_to_park"] = None
            model_data.loc[index, "d_2010_2018_feet_distance_to_park"] = None
            model_data.loc[index, "orig_square_meter_greenspace_coverage"] = None
            model_data.loc[index, "d_2010_2018_square_meter_greenspace_coverage"] = None
    model_data.to_csv(os.path.join(output_path, "integrated-itz-data.csv"))


def _correlate(data_path: str, output_path: str, img_path: str, verbose: bool):
    data = pd.read_csv(data_path)
    itz.make_correlation_matrix(data, output_path, img_path)


def _visualize(geodata_path: str, data_path: str, output_path: str, columns: List[str], lots: bool,
        verbose: bool):
    with open(geodata_path, "r") as f:
        geodata = json.load(f)
    with open(data_path, "r") as f:
        data = pd.read_csv(data_path)
    columns.insert(0, "ITZ_GEOID")
    if output_path:
        itz.make_map_vis(geodata, data, output_path, columns, int(lots)+1)
    else:
        itz.make_map_vis(geodata, data, "vis.html", columns, int(lots)+1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers()
    
    diagram_parser = subparsers.add_parser("diagram")
    diagram_parser.add_argument("model_string")
    diagram_parser.add_argument("data_path")
    diagram_parser.add_argument("img_path")
    diagram_parser.set_defaults(func=_make_diagram)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("model_string", choices=itz.model.MODEL_NAMES)
    fit_parser.add_argument("model_path")
    fit_parser.add_argument("data_path")
    fit_parser.add_argument("--cov_mat_path", required=False)
    fit_parser.set_defaults(func=_fit)

    histogram_parser = subparsers.add_parser("distribute")
    histogram_parser.add_argument("x", choices=itz.data.VAR_NAMES + ("all_vars",))
    histogram_parser.add_argument("data_path")
    histogram_parser.add_argument("--img_path", required=False)
    histogram_parser.add_argument("--log", required=False, action="store_true")
    histogram_parser.set_defaults(func=_make_histogram)

    regress_parser = subparsers.add_parser("regress")
    regress_parser.add_argument("x", choices=itz.data.VAR_NAMES + ("all_vars",))
    regress_parser.add_argument("y", choices=itz.data.VAR_NAMES + ("all_vars",))
    regress_parser.add_argument("data_path")
    regress_parser.add_argument("--regression_plot_path", required=False)
    regress_parser.add_argument("--residual_plot_path", required=False)
    regress_parser.add_argument("--histogram_path", required=False)
    regress_parser.add_argument("--log_x", required=False, action="store_true")
    regress_parser.add_argument("--log_y", required=False, action="store_true")
    regress_parser.set_defaults(func=_make_regression)

    parse_parser = subparsers.add_parser("parse")
    parse_parser.add_argument("output_path")
    parse_parser.add_argument("--itz_data_path", required=False)
    parse_parser.add_argument("--lot_data_path", required=False)
    parse_parser.add_argument("--tract_data_paths", action="extend", required=False)
    parse_parser.set_defaults(func=_parse)

    correlate_parser = subparsers.add_parser("correlate")
    correlate_parser.add_argument("data_path")
    correlate_parser.add_argument("output_path")
    correlate_parser.add_argument("--img_path", required=False)
    correlate_parser.set_defaults(func=_correlate)

    vis_parser = subparsers.add_parser("vis")
    vis_parser.add_argument("geodata_path")
    vis_parser.add_argument("data_path")
    vis_parser.add_argument("--columns", action="extend", nargs="+", required=True)
    vis_parser.add_argument("--output_path", required=False)
    vis_parser.add_argument("--lots", required=False, default=False, action="store_true")
    vis_parser.set_defaults(func=_visualize)

    args = parser.parse_args()
    args.func(**{key: val for key, val in vars(args).items() if key != "func"})