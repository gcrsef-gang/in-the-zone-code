from typing import Dict
import itertools

import pandas as pd

from itz.data import CONTROL_VARS, DEPENDENT_VARS, DENSIFICATION_MEASURES
import itz


def _print_stats(dict_: Dict[str, float]):
    """Prints a dictionary of statistics.
    """
    for key, val in dict_.items():
        print(f"{key}: {round(val, 3)}")


def do_stepwise_regression():

    data = pd.read_csv("in-the-zone-data/extra-updated-itz-data.csv")

    # covariances = []
    # for var1, var2 in itertools.combinations(CONTROL_VARS):
    #     covariances.append((var1, var2))
    #     desc = itz.get_description(itz.model.ModelName.LONG_TERM, covariances, control_regressions)
    desc, variables = itz.get_description(itz.model.ModelName.LONG_TERM, verbose=True)
    model = itz.fit(desc, variables, data, verbose=True)
    model_stats, inspection = itz.evaluate(model)
    _print_stats(model_stats)
    print(inspection)

if __name__ == "__main__":
    do_stepwise_regression()