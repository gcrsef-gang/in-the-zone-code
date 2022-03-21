"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
import itertools
import sys

import pandas as pd
import semopy

from .data import DENSIFICATION_MEASURES, CONTROL_VARS, DEPENDENT_VARS, EARLY_UPZONING
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
    # ModelName.LONG_TERM: ('d_2010_2018_square_meter_greenspace_coverage'),
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
    if verbose:
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

    # Densification into dependent variables.
    for dep_var in DEPENDENT_VARS:
        _add_relation([dep_var, *DENSIFICATION_MEASURES], ["~"] + ["+"] * (len(CONTROL_VARS) - 1))
        num_examined_regressions += len(DENSIFICATION_MEASURES)

    # Controls.
    for dep_var in DEPENDENT_VARS:
        significant_controls = [control for control in CONTROL_VARS
                    if abs(regress(dep_var, control, data)[3]) < REGRESSION_SIGNIFICANCE_THRESHOLD]
        _add_relation([dep_var, *significant_controls], ["~"] + ["+"] * (len(CONTROL_VARS) - 1))
        num_control_regressions += len(significant_controls)

    # Covariances
    num_covariances = 0

    for var_1, var_2 in itertools.combinations(CONTROL_VARS, 2):
        if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1

    for var_1, var_2 in itertools.combinations(DENSIFICATION_MEASURES, 2):
        if abs(regress(var_1, var_2, data)[3]) < CONTROL_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1

    for var_1, var_2 in itertools.combinations(DEPENDENT_VARS, 2):
        if abs(regress(var_1, var_2, data)[3]) < DEPENDENT_VARIABLE_COVARIANCE_SIGNIFICANCE_THRESHOLD:
            _add_relation([var_1, var_2], ["~~"])
            num_covariances += 1

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