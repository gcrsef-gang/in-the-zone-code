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

regress <x> <y> <data_path> [--regression_plot_path PATH1] [--residual_plot_path PATH2] [--histogram_path PATH3] [--transform_x] [--transform_y]
------------------------------------------------------------------------------------------------------------------------------------------------
Create visualizations for a regression between two variables. Descriptive statistics will be
printed to the console.

Parameters:
- x: name of explanatory variable (can also be all_vars)
- y: name of response variable (can also be all_vars)
- data_path: path to dataset CSV.
- regression_plot_path (optional): path to regression plot
- residual_plot_path (optional): path to residual plot
- histogram_path (optional): path to histogram of residuals
- transform_x: applies a transformation to x
- transform_y: applies a transformation to y

distribute <x> <data_path> [--img_path IMG_PATH] [--transform]
--------------------------------------------------------------
Create a histogram of a variable.

Parameters:
- x: name of variable to graph (can also be all_vars)
- data_path: path to dataset CSV.
- img_path: path to histogram
- transform: log-transforms x

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
from os.path import dirname
import pickle
import sys
import subprocess

import semopy
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


def _fit(model_string: str, model_type: str, model_path: str, data_path: str, output_path:str, cov_mat_path: str, verbose: bool):
    """Fits a model to the data and prints evaluation metrics.
    """
    # TODO: figure out if semopy.efa.explore_cfa_model() is something worth exploring (see semopy documentation)
    model_name = itz.model.ModelName.__dict__[model_string]
    model_type_string = itz.model.MODEL_TYPE_UPZONED_VARS[model_type]
    if verbose:
        print("Loading data... ", end="")
        sys.stdout.flush()
    data = pd.read_csv(data_path)
    data = data[~data[model_type_string].isna()]
    print(data)
    if verbose:
        print("done!")
    model = itz.fit(*itz.get_description(model_name, model_type_string, data, verbose), data, verbose)
    # TODO: figure out how to save/load a model
    if cov_mat_path is not None:
        pd.DataFrame(model.calc_sigma()[0])
        np.savetxt(cov_mat_path, model.calc_sigma()[0], delimiter=",")
    if verbose:
        print("Evaluating model...", end="")
        sys.stdout.flush()
    stats, params = itz.evaluate(model)
    if verbose:
        print("done!")
    _print_stats(stats)
    print(params)

    factors = model.predict_factors(data)
    print(factors)
    print(type(factors))
    for stat in stats.keys():
        stats[stat] = str(stats[stat])
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    with open(os.path.join(output_path, "model_stats.json"), "w") as f:
        json.dump(stats, f)
        # print(stats)
        # print(type(stats))
        # f.write(stats)
    params.to_csv(os.path.join(output_path, "model_inspection.csv"))
    # factors.to_csv(os.path.join(output_path, "model_factors.csv"))

    # semopy.semplot(model, os.path.join(output_path, "model_diagram.png"))
    # TODO: learn more about robust p-values (see semopy FAQ)
    semopy.report(model, "report", output_path)
    # subprocess.run(f"dot {os.path.join(output_path, 'report/plots/1')} -Tpng -Granksep=3 > {os.path.join(output_path, 'model_diagram.png')}")
    # subprocess.run(f"dot {os.path.join(output_path, 'report/plots/2')} -Tpng -Granksep=3 > {os.path.join(output_path, 'with_estimation_model_diagram.png')}")
    # subprocess.run(f"dot {os.path.join(output_path, 'report/plots/3')} -Tpng -Granksep=3 > {os.path.join(output_path, 'with_covariances_model_diagram.png')}")
    # subprocess.run(f"dot {os.path.join(output_path, 'report/plots/4')} -Tpng -Granksep=3 > {os.path.join(output_path, 'with_both_model_diagram.png')}")
    subprocess.run(["dot", os.path.join(output_path, "'report'/plots/1"), "-Tpng", "-Granksep=3", ">", os.path.join(output_path, "model_diagram.png")])
    subprocess.run(["dot", os.path.join(output_path, "'report'/plots/2"), "-Tpng", "-Granksep=3", ">", os.path.join(output_path, "with_estimation_model_diagram.png")])
    subprocess.run(["dot", os.path.join(output_path, "'report'/plots/3"), "-Tpng", "-Granksep=3", ">", os.path.join(output_path, "with_covariances_model_diagram.png")])
    subprocess.run(["dot", os.path.join(output_path, "'report'/plots/4"), "-Tpng", "-Granksep=3", ">", os.path.join(output_path, "with_both_model_diagram.png")])

def _make_histogram(x: str, data_path: str, img_path: str, transform: str, verbose: bool):
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
        if transform:
            transformation = itz.util.Transformations.__dict__[transform]
        else:
            transformation = lambda x: x
        _print_stats(itz.make_histogram(x, data, histogram_path, transformation))


def _make_regression(x, y, data_path, regression_plot_path, residual_plot_path, histogram_path,
        transform_x, transform_y, verbose):
    """Create a regression visualization.
    """
    data = pd.read_csv(data_path)
    data.set_index("ITZ_GEOID", inplace=True)

    if regression_plot_path:
        if transform_x:
            transformation_x = itz.util.Transformations.__dict__[transform_x]
        else:
            transformation_x = lambda x: x
        if transform_y:
            transformation_y = itz.util.Transformations.__dict__[transform_y]
        else:
            transformation_y = lambda x: x
        regression_stats = itz.make_regression_plot(x, y, data, regression_plot_path, transformation_x, transformation_y)
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
                    if transform_x and transform_y:
                        image_path = "regression-plots/"+y_column+"/"+str(transform_x)+"_"+x_column+"_"+str(transform_y)+"_"+y_column+".png"
                    elif transform_x:
                        image_path = "regression-plots/"+y_column+"/"+str(transform_x)+"_"+x_column+"_"+y_column+".png"
                    elif transform_y:
                        image_path = "regression-plots/"+y_column+"/"+x_column+"_"+str(transform_y)+"_"+y_column+".png"
                    else:
                        image_path = "regression-plots/"+y_column+"/"+x_column+"_"+y_column+".png"
                    if transform_x:
                        transformation_x = itz.util.Transformations.__dict__[transform_x]
                    else:
                        transformation_x = lambda x: x
                    if transform_y:
                        transformation_y = itz.util.Transformations.__dict__[transform_y]
                    else:
                        transformation_y = lambda x: x
                    regression_stats = itz.make_regression_plot(x_column, y_column, data, image_path, transformation_x, transformation_y)
        elif y == "all_vars":
            for y_column in data.columns:
                if y_column.startswith("Unnamed") or y_column in ["all_vars"]:
                    continue
                print("working on:", x, y_column)
                try:
                    os.mkdir("regression-plots/"+y_column+"/")
                except FileExistsError:
                    pass
                if transform_x and transform_y:
                    image_path = "regression-plots/"+y_column+"/"+str(transform_x)+"_"+x+"_"+str(transform_y)+"_"+y_column+".png"
                elif transform_x:
                    image_path = "regression-plots/"+y_column+"/"+str(transform_x)+"_"+x+"_"+y_column+".png"
                elif transform_y:
                    image_path = "regression-plots/"+y_column+"/"+x+"_"+str(transform_y)+"_"+y_column+".png"
                else:
                    image_path = "regression-plots/"+y_column+"/"+x+"_"+y_column+".png"
                if transform_x:
                    transformation_x = itz.util.Transformations.__dict__[transform_x]
                else:
                    transformation_x = lambda x: x
                if transform_y:
                    transformation_y = itz.util.Transformations.__dict__[transform_y]
                else:
                    transformation_y = lambda x: x
                regression_stats = itz.make_regression_plot(x, y_column, data, image_path, transformation_x, transformation_y)
        else:
            if transform_x and transform_y:
                image_path = "regression-plots/"+y+"/"+str(transform_x)+"_"+x+"_"+str(transform_y)+"_"+y+".png"
            elif transform_x:
                image_path = "regression-plots/"+y+"/"+str(transform_x)+"_"+x+"_"+y+".png"
            elif transform_y:
                image_path = "regression-plots/"+y+"/"+x+"_"+str(transform_y)+"_"+y+".png"
            else:
                image_path = "regression-plots/"+y+"/"+x+"_"+y+".png"
            if transform_x:
                transformation_x = itz.util.Transformations.__dict__[transform_x]
            else:
                transformation_x = lambda x: x
            if transform_y:
                transformation_y = itz.util.Transformations.__dict__[transform_y]
            else:
                transformation_y = lambda x: x
            regression_stats = itz.make_regression_plot(x, y_column, data, image_path, transformation_x, transformation_y)
    if residual_plot_path:
        if transform_x:
            transformation_x = itz.util.Transformations.__dict__[transform_x]
        else:
            transformation_x = lambda x: x
        if transform_y:
            transformation_y = itz.util.Transformations.__dict__[transform_y]
        else:
            transformation_y = lambda x: x
        resid_plot_stats = itz.make_residual_plot(x, y, data, residual_plot_path, transformation_x, transformation_y)
        regression_stats = {**regression_stats, **resid_plot_stats}
    
    if histogram_path:
        if transform_x:
            transformation_x = itz.util.Transformations.__dict__[transform_x]
        else:
            transformation_x = lambda x: x
        if transform_y:
            transformation_y = itz.util.Transformations.__dict__[transform_y]
        else:
            transformation_y = lambda x: x
        func = itz.util.regress(x, y, data, transformation_x, transformation_y)[-1]
        residual_df = pd.DataFrame()
        residual_df["resids"] = data[x].apply(func) - data[y]
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
    orthoimagery_2010 = pd.read_csv("in-the-zone-data/greenspace-orthoimagery/2010-greenspace-orthoimagery.csv")
    orthoimagery_2010.set_index("ITZ_GEOID", inplace=True) 
    orthoimagery_2018 = pd.read_csv("in-the-zone-data/greenspace-orthoimagery/2018-greenspace-orthoimagery.csv") 
    orthoimagery_2018.set_index("ITZ_GEOID", inplace=True) 
    distance_from_park = pd.read_csv("in-the-zone-data/greenspace-distance/tract_distance_from_park.csv") 
    distance_from_park.set_index("ITZ_GEOID", inplace=True) 
    for index, _ in model_data.iterrows():
        try:
            model_data.loc[index, "orig_feet_distance_from_park"] = distance_from_park.loc[index, "2010_distance_from_park"]
            model_data.loc[index, "d_2010_2018_feet_distance_from_park"] = distance_from_park.loc[index, "d_2010_2018_distance_from_park"]
            model_data.loc[index, "orig_square_meter_greenspace_coverage"] = orthoimagery_2010.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]
            model_data.loc[index, "d_2010_2018_square_meter_greenspace_coverage"] = orthoimagery_2018.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]-orthoimagery_2010.loc[index, "SQUARE_METER_GREENSPACE_COVERAGE"]
        except:
            model_data.loc[index, "orig_feet_distance_from_park"] = None
            model_data.loc[index, "d_2010_2018_feet_distance_from_park"] = None
            model_data.loc[index, "orig_square_meter_greenspace_coverage"] = None
            model_data.loc[index, "d_2010_2018_square_meter_greenspace_coverage"] = None
    model_data.to_csv(os.path.join(output_path, "integrated-itz-data.csv"))


def _correlate(data_path: str, output_path: str, img_path: str, verbose: bool):
    data = pd.read_csv(data_path)
    itz.make_correlation_matrix(data, output_path, img_path)

def _covariance(data_path: str, output_path: str, img_path: str, verbose: bool):
    data = pd.read_csv(data_path)
    itz.make_covariance_matrix(data, output_path, img_path)


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
    fit_parser.add_argument("model_type", choices=itz.model.MODEL_TYPE_UPZONED_VARS)
    fit_parser.add_argument("model_path")
    fit_parser.add_argument("data_path")
    fit_parser.add_argument("output_path")
    fit_parser.add_argument("--cov_mat_path", required=False)
    fit_parser.set_defaults(func=_fit)

    histogram_parser = subparsers.add_parser("distribute")
    histogram_parser.add_argument("x", choices=itz.data.VAR_NAMES + ("all_vars",))
    histogram_parser.add_argument("data_path")
    histogram_parser.add_argument("--img_path", required=False)
    histogram_parser.add_argument("--transform", required=False, choices=itz.util.TRANSFORMATION_NAMES)
    histogram_parser.set_defaults(func=_make_histogram)

    regress_parser = subparsers.add_parser("regress")
    regress_parser.add_argument("x", choices=itz.data.VAR_NAMES + ("all_vars",))
    regress_parser.add_argument("y", choices=itz.data.VAR_NAMES + ("all_vars",))
    regress_parser.add_argument("data_path")
    regress_parser.add_argument("--regression_plot_path", required=False)
    regress_parser.add_argument("--residual_plot_path", required=False)
    regress_parser.add_argument("--histogram_path", required=False)
    regress_parser.add_argument("--transform_x", required=False, choices=itz.util.TRANSFORMATION_NAMES)
    regress_parser.add_argument("--transform_y", required=False, choices=itz.util.TRANSFORMATION_NAMES)
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

    covariance_parser = subparsers.add_parser("covariance")
    covariance_parser.add_argument("data_path")
    covariance_parser.add_argument("output_path")
    covariance_parser.add_argument("--img_path", required=False)
    covariance_parser.set_defaults(func=_covariance)

    vis_parser = subparsers.add_parser("vis")
    vis_parser.add_argument("geodata_path")
    vis_parser.add_argument("data_path")
    vis_parser.add_argument("--columns", action="extend", nargs="+", required=True)
    vis_parser.add_argument("--output_path", required=False)
    vis_parser.add_argument("--lots", required=False, default=False, action="store_true")
    vis_parser.set_defaults(func=_visualize)

    args = parser.parse_args()
    args.func(**{key: val for key, val in vars(args).items() if key != "func"})