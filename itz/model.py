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
CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD = 0.01
REGRESSION_SIGNIFICANCE_THRESHOLD = 0.01


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
    ModelName.LONG_TERM: ('2002_2010_percent_upzoned', '2010_2018_percent_upzoned')
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
    if verbose:
        print("done!")
    return model


def get_description(model_name: ModelName, data: pd.DataFrame, verbose=False
                        ) -> Tuple[str, Set[str]]:
    """Creates a semopy model description for one of three possible SEMs.

    Returns the description as a string as well as a set of all variable names.
    """
    start_yr, mid_yr, end_yr = MODEL_YEARS[model_name]
    control_vars = {var for var in INDEPENDENT_VARS if var != "2002_2010_percent_upzoned"}

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
        f"d_{mid_yr}_{end_yr}_percent_multi_family_units"  # Provisional
    ]
    _add_relation(["densification"] + densification_indicators,
                 ["=~"] + ["+"] * (len(densification_indicators) - 1))

    # Regressions
    num_examined_regressions = 0
    num_control_regressions = 0
    for dep_var in DEPENDENT_VARS:
        # Densification as an explanatory variable
        _add_relation([dep_var, "densification"], ["~"])
        num_examined_regressions += 1
        # Controls for densification
        for control in control_vars:
            if abs(regress(dep_var, control, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
                _add_relation([dep_var, control], ["~"])
                num_control_regressions += 1

    # Later upzoning as a control for densificiation
    _add_relation(["densification", f"{mid_yr}_{end_yr}_percent_upzoned"], ["~"])

    # Early upzoning as an explanatory variable
    early_upzoning = f"{start_yr}_{mid_yr}_percent_upzoned"
    _add_relation(["densification", early_upzoning], ["~"])
    _add_relation([f"d_{mid_yr}_{end_yr}_median_home_value", early_upzoning], ["~"])
    for dep_var in DEPENDENT_VARS:
        if abs(regress(early_upzoning, dep_var, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD:
            _add_relation([early_upzoning, dep_var], ["~"])
            num_examined_regressions += 1

    num_control_regressions += 3
    
    # Covariances

    # Controls

    # Remove controls not involved in regressions.
    control_vars = variables.intersection(control_vars)

    num_covariances = 0
    for var_1, var_2, in itertools.combinations(control_vars, 2):
        if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1

    # Densification indicators

    for var_1, var_2 in itertools.combinations(densification_indicators, 2):
        _add_relation([var_1, var_2], ["~~"])
        num_covariances += 1

    # Dependent variables

    # for var_1, var_2 in itertools.combinations()

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