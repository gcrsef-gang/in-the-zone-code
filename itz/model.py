"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
import itertools
import sys
import time

import pandas as pd
import semopy

from .data import DENSIFICATION_MEASURES, CONTROL_VARS, DEPENDENT_VARS, EARLY_UPZONING
from .util import log_transform, square_transform, sqrt_transform, regress


# DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.005
DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD = 1
# CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.01
CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.05
# REGRESSION_SIGNIFICANCE_THRESHOLD = 0.01
REGRESSION_SIGNIFICANCE_THRESHOLD = 0.5


class ModelName(Enum):
    LONG_TERM = 0
    # SHORT_TERM_NO_CARBON_EMISSIONS = 1
    # SHORT_TERM_CARBON_EMISSIONS = 2


# MODEL_NAMES = ("LONG_TERM", "SHORT_TERM_NO_CARBON_EMISSIONS", "SHORT_TERM_CARBON_EMISSIONS")
MODEL_NAMES = ("LONG_TERM",)

MODEL_YEARS = {
    ModelName.LONG_TERM: ("2002", "2010", "2018"),
    # ModelName.SHORT_TERM_CARBON_EMISSIONS: ("2010", "2014"),
    # ModelName.SHORT_TERM_NO_CARBON_EMISSIONS: ("2010", "2014")
}

LOG_TRANSFORM_VARS = {
    # ModelName.LONG_TERM: ('2002_2010_percent_upzoned', '2010_2018_percent_upzoned')
    # ModelName.LONG_TERM: ('orig_square_meter_greenspace_coverage',
    #     'orig_per_capita_income',
    #     'orig_percent_mixed_development',
    #     'orig_public_transit_trips_under_45_mins',
    #     'orig_car_trips_under_45_mins',
    #     'orig_pop_density',
    #     'orig_resid_unit_density',
    #     ),
    ModelName.LONG_TERM: ()
}

SQRT_TRANSFORM_VARS = {
    # ModelName.LONG_TERM: ('2002_2010_percent_upzoned', '2010_2018_percent_upzoned')
    # ModelName.LONG_TERM: ('d_2010_2018_square_meter_greenspace_coverage',
        # ''),
    # ModelName.LONG_TERM: (
    #     'orig_percent_non_hispanic_asian_alone',
    #     'orig_percent_non_hispanic_black_alone',
    #     'orig_percent_hispanic_any_race',
    # )
    ModelName.LONG_TERM: ()
}
# SQUARE_TRANSFORM_VARS = {
#     # ModelName.LONG_TERM: ('2002_2010_percent_upzoned', '2010_2018_percent_upzoned')
#     ModelName.LONG_TERM: (',
#         ''),
#     ModelName.LONG_TERM: ()
# }

MODEL_TYPE_UPZONED_VARS = {
    # "MANHATTAN" : "2002_2010_percent_upzoned_manhattan",
    # "NON_MANHATTAN" : "2002_2010_percent_upzoned_non_manhattan",
    "UNIFIED" : "2002_2010_percent_upzoned",
}


def fit(desc: str, variables: Set[str], data: pd.DataFrame, verbose=False) -> semopy.Model:
    """Fits an SEM to a dataset. 
    """
    # Transform data

    log_transform_vars = set()
    square_transform_vars = set()
    sqrt_transform_vars = set()

    model_data = pd.DataFrame()
    for var in variables:
        # print(var, var[5:])
        if var in data.columns:
            model_data[var] = data[var]
        elif var.startswith("log_"):
            model_data[var[4:]] = data[var[4:]]
            log_transform_vars.add(var[4:])
        elif var[:4].startswith("square_"):
            model_data[var[7:]] = data[var[7:]]
            square_transform_vars.add(var[7:])
        elif var.startswith("sqrt_"):
            model_data[var[5:]] = data[var[5:]]
            # sqrt_transform_vars.add(var[var.find("sqrt_")+5:])
            sqrt_transform_vars.add(var[5:])
            # print("sqrt caught!")
    # model_data.index = data["ITZ_GEOID"]
    # print(data.index)
    model_data = model_data[model_data["orig_pop_density"] > 0]
    # model_data = model_data.dropna()
    model_data.to_csv("in-the-zone-data/all-data-integrated-itz-data.csv")
    if verbose:
        print(model_data, "after fit drop")
    if verbose:
        print("Log transforming: " + ", ".join(log_transform_vars), end="" + "... ")
        sys.stdout.flush()
    for var in log_transform_vars:
        model_data["log_" + var] = log_transform(model_data[var])
    for var in square_transform_vars:
        model_data["square_" + var] = square_transform(model_data[var])
    for var in sqrt_transform_vars:
        model_data["sqrt_" + var] = sqrt_transform(model_data[var])
        print(var, "sqrt_"+var)
    if verbose:
        print("done!")

    # Create and fit model

    if verbose:
        print("Constructing SEM model... ", end="")
        sys.stdout.flush()
    model = semopy.Model(desc)
    # model = semopy.ModelMeans(desc)
    if verbose:
        print("done!")

    if verbose:
        print("Fitting SEM to data... ", end="")
        sys.stdout.flush()
    start_time = time.time()
    model.fit(model_data)
    # model.fit(model_data, obj='FIML')
    duration = time.time() - start_time
    if verbose:
        print(f"done! Model fitted in {duration // 60}m {round(duration, 1) % 60}s")
    return model


# def get_description(model_name: ModelName, covariances: List[Tuple[str, str]]=[],
#                     control_regressions: Dict[str, List[str]]={}, verbose=False
#                         ) -> Tuple[str, Set[str]]:
#     """Creates a semopy model description for one of three possible SEMs.

#     Returns the description as a string as well as a set of all variable names.
#     """
#     relations = []
#     variables = set()

#     def _add_relation(var_names: List[str], operators: List[str]):
#         """Adds a relation to the SEM description.

#         There must be one fewer operator than there are variables.
#         """
#         operators = operators + [""]
#         relation = []
#         for var, operator in zip(var_names, operators):
#             if var in LOG_TRANSFORM_VARS[model_name]:
#                 relation.append("log_" + var)
#                 variables.add("log_" + var)
#             else:
#                 relation.append(var)
#                 variables.add(var)
#             relation.append(operator)
#         relations.append(" ".join(relation[:-1]))

#     # Regressions
#     num_examined_regressions = 0
#     num_control_regressions = 0

#     # Early upzoning into densification.
#     for densification_var in DENSIFICATION_MEASURES:
#         _add_relation([densification_var, EARLY_UPZONING], ["~"])
#         num_examined_regressions += 1

#     # Densification into dependent variables.
#     for dep_var in DEPENDENT_VARS:
#         _add_relation([dep_var, *DENSIFICATION_MEASURES], ["~"] + ["+"] * (len(CONTROL_VARS) - 1))
#         num_examined_regressions += len(DENSIFICATION_MEASURES)

#     # Controls.
#     for dep_var, explanatory_vars in control_regressions.items():
#         _add_relation([dep_var, *explanatory_vars], ["~"] + ["+"] * (len(explanatory_vars) - 1))
#         num_control_regressions += len(explanatory_vars)

#     # Covariances
#     num_covariances = len(covariances)

#     for var_1, var_2 in covariances:
#         _add_relation([var_1, var_2], ["~~"])

#     model_description = "\n".join(relations)

#     if verbose:
#         print(f"{num_examined_regressions=}")
#         print(f"{num_control_regressions=}")
#         print(f"{num_covariances=}")
#         print(f"{variables=}")
#         print(model_description)

#     return model_description, variables


def get_description(model_name: ModelName, model_type: str, data: pd.DataFrame, covariances: List[Tuple[str, str]]=[],
                    control_regressions: Dict[str, List[str]]={}, verbose=False
                        ) -> Tuple[str, Set[str]]:
    """Creates a semopy model description for one of three possible SEMs.

    Returns the description as a string as well as a set of all variable names.
    """
    relations = []
    variables = set()

    def _add_relation(var_names: List[str], operators: List[str]):
        """Adds a relation to the SEM description.

        There must be one fewer operator than there are variables.
        """
        operators = operators + [""]
        relation = []
        for var, operator in zip(var_names, operators):
            if var in LOG_TRANSFORM_VARS[model_name]:
                relation.append("log_" + var)
                variables.add("log_" + var)
                print("YOO!!", var)
            elif var in SQRT_TRANSFORM_VARS[model_name]:
                relation.append("sqrt_" + var)
                variables.add("sqrt_" + var)
            else:
                relation.append(var)
                variables.add(var)
            relation.append(operator)
        relations.append(" ".join(relation[:-1]))

    # Regressions
    num_examined_regressions = 0
    num_control_regressions = 0

    # Early upzoning into densification.
    for densification_var in DENSIFICATION_MEASURES:
        _add_relation([densification_var, EARLY_UPZONING], ["~"])
        num_examined_regressions += 1

    regressions = []
    # Densification into dependent variables.
    for dep_var in DEPENDENT_VARS:
        _add_relation([dep_var, *DENSIFICATION_MEASURES], ["~"] + ["+"] * (len(CONTROL_VARS) - 1))
        num_examined_regressions += len(DENSIFICATION_MEASURES)
        for densification_measure in DENSIFICATION_MEASURES:
            regressions.append([dep_var, densification_measure])

    # Controls.
    for dep_var in DEPENDENT_VARS:
        significant_controls = [control for control in CONTROL_VARS
                    if abs(regress(dep_var, control, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD]
        _add_relation([dep_var, *significant_controls], ["~"] + ["+"] * (len(CONTROL_VARS) - 1))
        for control in significant_controls:
            regressions.append([dep_var, control])
        num_control_regressions += len(significant_controls)

    for densification_var in DENSIFICATION_MEASURES:
        _add_relation([densification_var, "2010_2018_percent_upzoned"], ["~"])
        for control in CONTROL_VARS: 
            if [densification_var, control] in regressions:
                continue
            if control == "2010_2018_percent_upzoned":
                continue
            # if abs(regress(densification_var, control, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            _add_relation([densification_var, control], ["~"])
            regressions.append([densification_var, control])
            num_control_regressions += 1

    for var_1, var_2 in itertools.combinations(DEPENDENT_VARS, 2):
        if [var_1, var_2] not in regressions:
            if abs(regress(var_1, var_2, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([var_1, var_2], ["~"])
                regressions.append([var_1, var_2])
                num_examined_regressions += 1
        if [var_2, var_1] not in regressions:
            if abs(regress(var_2, var_1, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([var_2, var_1], ["~"])
                regressions.append([var_2, var_1])
                num_examined_regressions += 1

    # UPZONING_AFFECTED = ['d_2010_2018_percent_multi_family_units',
    #    'd_2010_2018_percent_occupied_housing_units',
    #    'd_2010_2018_median_gross_rent', 'd_2010_2018_median_home_value',
    #    'd_2010_2018_square_meter_greenspace_coverage']
    # for var in UPZONING_AFFECTED:
    #     if [var, EARLY_UPZONING] in regressions:
    #         continue
    #     _add_relation([var, EARLY_UPZONING], ["~"])
    #     _add_relation([var, "2010_2018_percent_upzoned"], ["~"])
    #     regressions.append([var, EARLY_UPZONING])
    #     regressions.append([var, '2010_2018_percent_upzoned'])
    #     num_examined_regressions += 2

    for var in DEPENDENT_VARS:
        if [var, EARLY_UPZONING] in regressions:
            continue
        # if abs(regress(var, EARLY_UPZONING, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            # _add_relation([var, EARLY_UPZONING], ["~"])
            # regressions.append([var, EARLY_UPZONING])
        _add_relation([var, EARLY_UPZONING], ["~"])
        regressions.append([var, EARLY_UPZONING])

    # for var in CONTROL_VARS:
    #     _add_relation([var, EARLY_UPZONING], ["~"])
    # for var in CONTROL_VARS:
    #     if abs(regress("2010_2018_percent_upzoned", var, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
    #         _add_relation(["2010_2018_percent_upzoned", var], ["~"])

    for var in DEPENDENT_VARS:
        # if var not in UPZONING_AFFECTED:
        #     if abs(regress(var, "2010_2018_percent_upzoned", data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
        #         _add_relation([var, "2010_2018_percent_upzoned"], ["~"])
        if [var, "2010_2018_percent_upzoned"] in regressions:
            continue
        if abs(regress(var, "2010_2018_percent_upzoned", data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            _add_relation([var, "2010_2018_percent_upzoned"], ["~"])
            regressions.append([var, "2010_2018_percent_upzoned"])

    for var in CONTROL_VARS:
        if "2010_2018_percent_upzoned" == var:
            continue
        if abs(regress(var, "2010_2018_percent_upzoned", data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            _add_relation(["2010_2018_percent_upzoned", var], ["~"])
            regressions.append(["2010_2018_percent_upzoned", var])

    # Covariances
    # num_covariances = 0

    # for var_1, var_2 in itertools.combinations(CONTROL_VARS, 2):
    #     if var_1 == "2010_2018_percent_upzoned" or var_2 == "2010_2018_percent_upzoned":
    #         continue
    #     if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
    #         _add_relation([var_1, var_2], ["~~"])
    #         num_covariances += 1
    #     _add_relation([var_1, var_2], ["~~"])
    #     num_covariances += 1

    # Add covariance between densification variables.
    # Could be messing with the stuff. 
    # for var_1, var_2 in itertools.combinations(DENSIFICATION_MEASURES, 2):
    #     if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
    #         _add_relation([var_1, var_2], ["~~"])
    #         num_covariances += 1
        # _add_relation([var_1, var_2], ["~~"])
        # num_covariances += 1

    # _add_relation([EARLY_UPZONING, "2010_2018_percent_upzoned"], ["~~"])
    # _add_relation(["2010_2018_percent_upzoned", EARLY_UPZONING], ["~"])

    # # Add covariances between densification indicators and dependent variables
    # for densification_indicator in densification_indicators:
    #     for var in dependent_vars:
    #         if abs(regress(densification_indicator, var, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
    #             _add_relation([densification_indicator, var], ["~~"])
    #             num_covariances += 1

    # for var_1, var_2 in itertools.combinations(DEPENDENT_VARS, 2):
    #     if [var_1, var_2] in regressions:
    #         continue
    #     if abs(regress(var_1, var_2, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
    #         _add_relation([var_1, var_2], ["~~"])
    #         num_covariances += 1
        # _add_relation([early_upzoning, var], ["~~"])
        # num_covariances += 1

    model_description = "\n".join(relations)

    if verbose:
        print(f"{num_examined_regressions=}")
        print(f"{num_control_regressions=}")
        print(f"{num_covariances=}")
        print(f"{variables=}")
        print(model_description)

    return model_description, variables


def evaluate(model: semopy.Model) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Returns evaluations of how well an SEM fits a dataset.
    """
    stats = semopy.calc_stats(model)
    return {col: stats[col][0] for col in stats.columns}, model.inspect()