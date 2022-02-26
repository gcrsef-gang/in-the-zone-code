"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum
from typing import Dict, Tuple

import pandas as pd
import semopy


class ModelName(Enum):
    LONG_TERM = 0
    SHORT_TERM_NO_CARBON_EMISSIONS = 1
    SHORT_TERM_CARBON_EMISSIONS = 2

MODEL_NAMES = ("LONG_TERM", "SHORT_TERM_NO_CARBON_EMISSIONS", "SHORT_TERM_CARBON_EMISSIONS")

MODEL_YEARS = {
    ModelName.LONG_TERM: ("2011", "2019"),
    ModelName.SHORT_TERM_CARBON_EMISSIONS: ("2011", "2016"),
    ModelName.SHORT_TERM_NO_CARBON_EMISSIONS: ("2011", "2016")
}


def fit(desc: str, data: pd.DataFrame, verbose=False) -> Tuple[semopy.Model, Dict[str, float]]:
    """Fits an SEM to a dataset. Returns model parameters and evaluation metrics.
    """
    print("fitting", desc)


def get_description(model_name: ModelName) -> str:
    """Creates a semopy model description for one of three possible SEMs.
    """

def evaluate(model: semopy.Model, data: pd.DataFrame) -> Dict[str, float]:
    """Returns evaluations of how well an SEM fits a dataset.
    """