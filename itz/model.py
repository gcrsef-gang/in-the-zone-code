"""Implementation of SEM for modeling the effect of upzoning on various urban metrics.
"""

from enum import Enum




class ModelName(Enum):
    LONG_TERM = 0
    SHORT_TERM_NO_CARBON_EMISSIONS = 1
    SHORT_TERM_CARBON_EMISSIONS = 2

MODEL_YEARS = {
    ModelName.LONG_TERM: ("2011", "2019"),
    ModelName.SHORT_TERM_CARBON_EMISSIONS: ("2011", "2016"),
    ModelName.SHORT_TERM_NO_CARBON_EMISSIONS: ("2011", "2016")
}

def fit(desc, data):
    """Fits an SEM to a dataset. Returns model parameters and evaluation metrics.
    """

def get_description(model_name):
    """Creates a semopy model description for one of three possible SEMs.

    Model options:
    - long-term
    - short-term-with-emissions
    - short-term-no-emissions
    """

def _evaluate(model, data):
    """Returns evaluations of how well an SEM fits a dataset.
    """