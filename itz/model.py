"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
import itertools
import sys

import pandas as pd
import semopy

from .data import INDEPENDENT_VARS, DEPENDENT_VARS
from .util import log_transform, regress


DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.005
# DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD = 1
CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.01
# CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD = 1
REGRESSION_SIGNIFICANCE_THRESHOLD = 0.01
# REGRESSION_SIGNIFICANCE_THRESHOLD = 1


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
    ModelName.LONG_TERM: ()
}

MODEL_TYPE_UPZONED_VARS = {
    "manhattan" : "2002_2010_percent_upzoned_manhattan",
    "non_manhattan" : "2002_2010_percent_upzoned_non_manhattan",
    "unified" : "2002_2010_percent_upzoned",
}


def fit(desc: str, variables: Set[str], data: pd.DataFrame, verbose=False) -> semopy.Model:
    """Fits an SEM to a dataset. 
    """
    # Transform data

    log_transform_vars = set()

    model_data = pd.DataFrame()
    for var in variables:
        if var in data.columns:
            model_data[var] = data[var]
        elif var[:4] == "log_":
            model_data[var[4:]] = data[var[4:]]
            log_transform_vars.add(var[4:])
    model_data = model_data.dropna()
    print(model_data, "after fit drop")
    if verbose:
        print("Log transforming: " + ", ".join(log_transform_vars), end="" + "... ")
        sys.stdout.flush()
    for var in log_transform_vars:
        model_data["log_" + var] = log_transform(model_data[var])
    if verbose:
        print("done!")

    # Create and fit model

    if verbose:
        print("Constructing SEM model... ", end="")
        sys.stdout.flush()
    model = semopy.Model(desc)
    if verbose:
        print("done!")

    if verbose:
        print("Fitting SEM to data... ", end="")
        sys.stdout.flush()
    model.fit(model_data)
    # model.fit(model_data, obj='GLS')
    if verbose:
        print("done!")
    return model


def get_description(model_name: ModelName, model_type: str, data: pd.DataFrame, verbose=False
                        ) -> Tuple[str, Set[str]]:
    """Creates a semopy model description for one of three possible SEMs.

    Returns the description as a string as well as a set of all variable names.
    """
    start_yr, mid_yr, end_yr = MODEL_YEARS[model_name]
    early_upzoning = model_type
    print(early_upzoning, "early_upzoning")
    # control_vars = {var for var in INDEPENDENT_VARS if var not in ["2002_2010_percent_upzoned","2002_2010_percent_upzoned_manhattan","2002_2010_percent_upzoned_non_manhattan", "d_2010_2018_pop_density", "d_2010_2018_resid_unit_density"]}
    control_vars = {var for var in INDEPENDENT_VARS if var not in ["2002_2010_percent_upzoned","2002_2010_percent_upzoned_manhattan","2002_2010_percent_upzoned_non_manhattan"]}
    # dependent_vars = {var for var in DEPENDENT_VARS}
    dependent_vars = {var for var in DEPENDENT_VARS if var not in ["d_2010_2018_pop_density", "d_2010_2018_resid_unit_density"  ]}
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
            else:
                relation.append(var)
                variables.add(var)
            relation.append(operator)
        relations.append(" ".join(relation[:-1]))

    # Latent variables
    densification_indicators = [
        f"d_{mid_yr}_{end_yr}_pop_density",
        f"d_{mid_yr}_{end_yr}_resid_unit_density",
        # f"d_{mid_yr}_{end_yr}_percent_multi_family_units"  # Provisional
    ]
    _add_relation(["densification"] + densification_indicators,
                 ["=~"] + ["+"] * (len(densification_indicators) - 1))

    # Regressions
    num_examined_regressions = 0
    num_control_regressions = 0
    for dep_var in DEPENDENT_VARS:
        # Densification as an explanatory variable
        if dep_var in densification_indicators:
            continue
        _add_relation([dep_var, "densification"], ["~"])
        num_examined_regressions += 1
        # Controls for dependent variables
        # maybe change so that densification indicators aren't getting controlled also?
        for control in control_vars:
            if abs(regress(dep_var, control, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([dep_var, control], ["~"])
                num_control_regressions += 1

    # Later upzoning as a control for densificiation
    _add_relation(["densification", f"{mid_yr}_{end_yr}_percent_upzoned"], ["~"])

    # Early upzoning as an explanatory variable for densification
    # early_upzoning = f"{start_yr}_{mid_yr}_percent_upzoned"
    _add_relation(["densification", early_upzoning], ["~"])



    _add_relation([f"d_{mid_yr}_{end_yr}_median_home_value", early_upzoning], ["~"])

    # Add regressions between upzoning and dependent variables
    for dep_var in DEPENDENT_VARS:
        if dep_var == f"d_{mid_yr}_{end_yr}_median_home_value" or dep_var in densification_indicators:
            continue
        if abs(regress(dep_var, early_upzoning, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            _add_relation([dep_var, early_upzoning], ["~"])
            num_examined_regressions += 1

    dependent_regressions = []
    for dep_var_1, dep_var_2 in itertools.combinations(DEPENDENT_VARS, 2):
        if dep_var_1 in densification_indicators and dep_var_2 in densification_indicators:
            continue
        if dep_var_1 not in densification_indicators:
            if abs(regress(dep_var_1, dep_var_2, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([dep_var_1, dep_var_2], ["~"])
                num_examined_regressions += 1
                dependent_regressions.append((dep_var_1, dep_var_2))
        if dep_var_2 not in densification_indicators:
            if abs(regress(dep_var_2, dep_var_1, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([dep_var_2, dep_var_1], ["~"])
                num_examined_regressions += 1
                dependent_regressions.append((dep_var_2, dep_var_1))

    num_control_regressions += 3
    
    # Covariances

    # Controls

    # Remove controls not involved in regressions. 
    # Note: does this actually do that? i see no indication that variables are actually being
    # removed from control_vars. - jam
    print(len(control_vars), "control vars pre removal")
    control_vars = variables.intersection(control_vars)
    print(len(control_vars), "control vars post removal")

    # Add all controls for densification
    for control in control_vars:
        if control in densification_indicators or control == f"{mid_yr}_{end_yr}_percent_upzoned":
            continue
        print(control)
        _add_relation(["densification", control], ["~"])


    # Add covariances between controls
    num_covariances = 0
    for var_1, var_2, in itertools.combinations(control_vars, 2):
        if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1
        # _add_relation([var_1, var_2], ["~~"])
        # num_covariances += 1

    # # Add covariances between densification indicators
    # Commented out - could be causing problems with interactions between densification
    # for var_1, var_2 in itertools.combinations(densification_indicators, 2):
    #     _add_relation([var_1, var_2], ["~~"])
    #     num_covariances += 1

    # Add covariances between dependent variables

    for var_1, var_2 in itertools.combinations(dependent_vars, 2):
        if (var_1, var_2) in dependent_regressions or (var_2, var_1) in dependent_regressions:
            continue
        if var_1 in densification_indicators and var_2 in densification_indicators:
            continue
        if abs(regress(var_1, var_2, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1
        # _add_relation([var_1, var_2], ["~~"])
        # num_covariances += 1

    # Add covariances between densification indicators and dependent variables
    for densification_indicator in densification_indicators:
        for var in dependent_vars:
            if abs(regress(densification_indicator, var, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
                _add_relation([densification_indicator, var], ["~~"])
                num_covariances += 1

    # Add covariances for upzoning and control variables
    for var in control_vars:
        if var in densification_indicators:
            continue
        if abs(regress(early_upzoning, var, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([early_upzoning, var], ["~~"])
            num_covariances += 1
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