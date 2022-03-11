"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum
from typing import Dict, Tuple

import pandas as pd
import semopy

from .data import VAR_NAMES


class ModelName(Enum):
    LONG_TERM = 0
    SHORT_TERM_NO_CARBON_EMISSIONS = 1
    SHORT_TERM_CARBON_EMISSIONS = 2

MODEL_NAMES = ("LONG_TERM", "SHORT_TERM_NO_CARBON_EMISSIONS", "SHORT_TERM_CARBON_EMISSIONS")

MODEL_YEARS = {
    ModelName.LONG_TERM: ("2010", "2018"),
    # ModelName.SHORT_TERM_CARBON_EMISSIONS: ("2010", "2014"),
    # ModelName.SHORT_TERM_NO_CARBON_EMISSIONS: ("2010", "2014")
}


DEPENDENT_VARIABLES = {
    ModelName.LONG_TERM: (
        "d_2010_2018_percent_car_commuters",
        "d_2010_2018_percent_car_commuters",
        "d_2010_2018_percent_public_transport_commuters",
        "d_2010_2018_percent_public_transport_trips_under_45_min",
        "d_2010_2018_percent_car_trips_under_45_min",
        "d_2010_2018_percent_non_hispanic_or_latino_white_alone",
        "d_2010_2018_percent_occupied_housing_units",
        "d_2010_2018_median_home_value",
        "d_2010_2018_per_capita_income",
        "d_2010_2018_median_gross_rent"),
    ModelName.SHORT_TERM_CARBON_EMISSIONS: (),
    ModelName.SHORT_TERM_NO_CARBON_EMISSIONS: ()
}


def fit(desc: str, data: pd.DataFrame, verbose=False) -> semopy.Model:
    """Fits an SEM to a dataset.
    """
    if verbose:
        print(desc)
    model = semopy.Model(desc)
    model.fit(data)
    return model


def get_description(model_name: ModelName) -> str:
    """Creates a semopy model description for one of three possible SEMs.
    """
    start_yr, end_yr = MODEL_YEARS[model_name]
    relationships = []

    # Latent variables
    densification_indicators = [
        f"d_{start_yr}_{end_yr}_pop_density",
        f"d_{start_yr}_{end_yr}_resid_unit_density",
        f"d_{start_yr}_{end_yr}_percent_multi_family_units"
    ]
    relationships.append(f"densification =~ " + " + ".join(densification_indicators))

    controls = [var for var in VAR_NAMES if var[:4] == "orig"]
    controls.append(f"{start_yr}_{end_yr}_average_years_since_upzoning")
    
    # Regressions

    for dep_var in DEPENDENT_VARIABLES[model_name]:
        relationships.append(f"{dep_var} ~ densification")
        for control in controls:
            relationships.append(f"{dep_var} ~ {control}")

    percent_upzoned = f"{start_yr}_{end_yr}_percent_upzoned"
    relationships.append(f"densification ~ {percent_upzoned}")
    relationships.append(f"d_{start_yr}_{end_yr}_median_home_value ~ {percent_upzoned}")
    
    return "\n".join(relationships)


def evaluate(model: semopy.Model) -> pd.DataFrame:
    """Returns evaluations of how well an SEM fits a dataset.
    """
    return model.inspect()