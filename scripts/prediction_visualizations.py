"""Usage:
python3 scripts/prediction_visualizations.py <model_path> <data_path> <x> <y> <output_path> [--full_model] [--include_endogenous] [--all]

--full_model creates a regression plot of model predictions considering the entire
model. Not including this flag creates a graph considering only the effect of the 
explanatory variable on the response variable.

if --full_model is specified, --include_endogenous can also be specified in order to
set the values of endogenous variables to known in the model's predictions.

--all does all of the script's functionality.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import semopy

from scripts import effect_evaluator
import itz


ENDOGENOUS_VARS = {'d_2010_2018_percent_car_commuters', 'd_2010_2018_percent_car_trips_under_45_min', 'd_2010_2018_resid_unit_density', 'd_2010_2018_percent_hispanic_any_race', 'd_2010_2018_percent_public_transport_trips_under_45_min', 'd_2010_2018_percent_multi_family_units', '2010_2018_percent_upzoned', 'd_2010_2018_per_capita_income', 'd_2010_2018_percent_non_hispanic_asian_alone', 'd_2010_2018_percent_bachelor_degree_or_higher', 'd_2010_2018_percent_households_with_people_under_18', 'd_2010_2018_percent_public_transport_commuters', 'd_2010_2018_median_gross_rent', 'd_2010_2018_percent_occupied_housing_units', 'd_2010_2018_percent_non_hispanic_black_alone', 'd_2010_2018_percent_non_hispanic_or_latino_white_alone', 'd_2010_2018_pop_density', 'd_2010_2018_square_meter_greenspace_coverage', 'd_2010_2018_percent_of_households_in_same_house_year_ago', 'd_2010_2018_median_home_value'}

ALL_VARS = {'orig_percent_public_transport_trips_under_45_min', 'd_2010_2018_percent_car_commuters', 'sqrt_orig_percent_non_hispanic_asian_alone', 'd_2010_2018_percent_car_trips_under_45_min', 'd_2010_2018_resid_unit_density', 'orig_median_age', 'orig_percent_residential', 'log_orig_pop_density', '2002_2010_percent_upzoned', 'orig_percent_public_transport_commuters', 'd_2010_2018_percent_hispanic_any_race', 'orig_percent_subsidized_properties', 'log_orig_resid_unit_density', 'd_2010_2018_percent_public_transport_trips_under_45_min', '2010_2018_percent_upzoned', 'orig_percent_households_with_people_under_18', 'd_2010_2018_per_capita_income', 'd_2010_2018_percent_non_hispanic_asian_alone', 'd_2010_2018_percent_multi_family_units', 'sqrt_orig_percent_non_hispanic_black_alone', 'd_2010_2018_percent_bachelor_degree_or_higher', 'd_2010_2018_percent_households_with_people_under_18', 'orig_feet_distance_from_park', 'd_2010_2018_percent_public_transport_commuters', 'd_2010_2018_median_gross_rent', 'orig_median_home_value', 'orig_percent_bachelor_degree_or_higher', 'd_2010_2018_percent_occupied_housing_units', 'sqrt_orig_percent_hispanic_any_race', 'd_2010_2018_percent_non_hispanic_black_alone', 'orig_percent_multi_family_units', 'log_orig_per_capita_income', 'orig_percent_of_households_in_same_house_year_ago', 'orig_percent_non_hispanic_or_latino_white_alone', 'd_2010_2018_percent_non_hispanic_or_latino_white_alone', 'orig_percent_car_commuters', 'd_2010_2018_median_home_value', 'log_orig_percent_mixed_development', 'd_2010_2018_percent_of_households_in_same_house_year_ago', 'd_2010_2018_pop_density', 'orig_median_gross_rent', 'orig_percent_car_trips_under_45_min', 'log_orig_square_meter_greenspace_coverage', 'd_2010_2018_square_meter_greenspace_coverage', 'orig_percent_occupied_housing_units'}

SIMPLE_DESC = """
d_2010_2018_pop_density ~ 2002_2010_percent_upzoned + orig_percent_multi_family_units
"""


def get_model_predictions(model, data, y, include_endogenous=False, excluding=None, only_consider=None) -> np.ndarray:
    """Gets full-model predictions of y given a set of variables.

    Returns 2d numpy array containing one array of x and another array of predicted y.
    """
    explanatory_data = pd.DataFrame(data)
    explanatory_data[y] = pd.Series([np.nan] * len(data))

    if only_consider is not None:
        for var in ALL_VARS:
            if var != y and var != only_consider:
                explanatory_data[var] = pd.Series([0.0] * len(data))
    else:
        if not include_endogenous:
            for var in ENDOGENOUS_VARS:
                explanatory_data[var] = pd.Series([np.nan] * len(data))

        if excluding is not None:
            explanatory_data[excluding] = pd.Series([0.0] * len(data))

    return model.predict(explanatory_data, intercepts=True)


def make_model_evaluation_graph(data, model, x, y, output_path, include_endogenous):

    means = semopy.estimate_means(model)
    intercept = means[means["lval"] == y].iloc[0]["Estimate"]

    predictions = get_model_predictions(model, data, y, include_endogenous)

    regression_graph = effect_evaluator.get_regression_graph(model.inspect())
    total_effect_coef = effect_evaluator.get_total_effect_dfs(model.inspect(), x, y)
    direct_effect_coef = [est for lval, est, _ in regression_graph[x] if lval == y][0]

    graph_data = pd.DataFrame()
    graph_data[x] = data[x]
    graph_data[y] = data[y]
    graph_data[f"total effect pred"] = data[x] * total_effect_coef + intercept
    graph_data[f"direct effect pred"] = data[x] * direct_effect_coef + intercept
    graph_data[f"pred {y}"] = predictions[y]
    graph_data.to_csv(os.path.join(output_path, "model_eval_graph_data.csv"))

    plt.subplots(figsize=(18, 10))
    plt.scatter(data[x], data[y], label="Observed data")
    pred_label = "Model predictions"
    if not include_endogenous:
        pred_label += " using exogenous variables"
    plt.scatter(predictions[x], predictions[y], color="red", label=pred_label)
    plt.plot(data[x] * total_effect_coef + intercept, color="black", label="Total effect")
    plt.plot(data[x] * direct_effect_coef + intercept, color="black", label="Direct effect", linestyle="dashed")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.title(f"{x} vs. {y}")
    fname = "model evaluation"
    if include_endogenous:
        fname += " including endogenous"
    plt.savefig(os.path.join(output_path, f"{fname}.pdf"), format="pdf")
    plt.clf()

    plt.subplots(figsize=(18, 10))
    plt.scatter(predictions[x], data[y] - predictions[y])
    plt.plot(predictions[x], np.zeros(len(predictions)), color="red")
    plt.xlabel(x)
    plt.ylabel(f"Residuals for {y}")
    plt.title(f"{y} Residual Plot")
    fname = "residual plot"
    if include_endogenous:
        fname += " including endogenous"
    plt.savefig(os.path.join(output_path, f"{fname}.pdf"), format="pdf")
    plt.clf()


def make_model_regression_graph(data, x, y, output_path):

    # means = semopy.estimate_means(model)
    # intercept = means[means["lval"] == y].iloc[0]["Estimate"]

    no_x_predictions = get_model_predictions(
        model, data, y, include_endogenous=True, excluding=x)[y]
    only_x_predictions = get_model_predictions(model, data, y,
        include_endogenous=True)[y] - no_x_predictions

    plt.subplots(figsize=(18, 10))

    plt.scatter(data[x], data[y] - no_x_predictions,
                label=f"{y} - predicted {y} without accounting for X")

    # regression_graph = effect_evaluator.get_regression_graph(inspection)
    # total_effect_coef = effect_evaluator.get_total_effect_dfs(
    #     inspection, x, y)
    # direct_effect_coef = [est for lval, est, _ in regression_graph[x] if lval == y][0]

    graph_data = pd.DataFrame()
    graph_data[x] = data[x]
    graph_data[f"partial {y}"] = data[y] - no_x_predictions
    graph_data[f"only x pred"] = only_x_predictions
    # graph_data[f"total effect pred"] = data[x] * total_effect_coef + intercept
    # graph_data[f"direct effect pred"] = data[x] * direct_effect_coef + intercept
    graph_data.to_csv(os.path.join(output_path, "regression_graph_data.csv"))

    # plt.plot(data[x], data[x] * total_effect_coef + intercept, color="red", label="Total effect")
    # plt.plot(data[x], data[x] * direct_effect_coef + intercept, color="green", label="Direct effect")
    plt.scatter(data[x], only_x_predictions, color="red", label=f"{x}-based prediction")

    plt.xlabel(x)
    plt.ylabel(f"Part of {y} influenced by {x}")
    plt.title(f"Model prediciton of {y} based on {x}")
    plt.legend()
    plt.savefig(os.path.join(output_path, "individual_effect.pdf"), format="pdf")
    plt.clf()

    plt.subplots(figsize=(18, 10))
    plt.scatter(data[x], data[y], label=y)
    plt.scatter(data[x], no_x_predictions, color="red", label=f"Prediction without considering {x}")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Model prediciton of {y} based on all variables but {x}")
    plt.legend()
    plt.savefig(os.path.join(output_path, "all_other_effect.pdf"), format="pdf")
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("data_path")
    parser.add_argument("x")
    parser.add_argument("y")
    parser.add_argument("output_path")
    parser.add_argument("--full_model", action="store_true")
    parser.add_argument("--include_endogenous", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    untransformed_vars = {"_".join(var.split("_")[1:]) if var.startswith("log") or var.startswith("sqrt") or var.startswith("square") else var for var in ALL_VARS}
    for var in data.columns:
        if var not in untransformed_vars:
            del data[var]
    data = data.dropna()
    
    try:
        os.mkdir(args.output_path)
    except FileExistsError:
        pass

    with open(args.model_path, "r") as f:
        desc = f.read()
    # desc = SIMPLE_DESC
    print(desc)

    model = itz.fit(desc, ALL_VARS, data, verbose=True)
    # model = itz.fit(desc, {"2002_2010_percent_upzoned", "d_2010_2018_pop_density", "orig_percent_multi_family_units"}, data, verbose=True)
    inspection = model.inspect()

    if args.all:
        make_model_evaluation_graph(data, model, args.x, args.y, args.output_path, True)
        make_model_evaluation_graph(data, model, args.x, args.y, args.output_path, False)
        make_model_regression_graph(data, model, args.x, args.y, args.output_path)

    elif args.full_model:
        make_model_evaluation_graph(data, model, args.x, args.y, args.output_path, args.include_endogenous)
    else:
        make_model_regression_graph(data, args.x, args.y, args.output_path)