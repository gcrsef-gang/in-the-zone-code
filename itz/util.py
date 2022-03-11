"""Statistics utility functions.
"""

import math

import numpy as np
import pandas as pd
import scipy


def log_transform(var: str, data: pd.DataFrame) -> pd.Series:
    """Returns the log-transformed version of a variable.
    """
    return data[var].transform(math.log)


def regress(x: str, y: str, data: pd.DataFrame) -> float:
    """Returns the slope, intercept, correlation coefficient, two-tailed p-value, and coefficient of
    determination between two variables.
    """
    data = data[data[x].notnull()][data[y].notnull()]
    X = data[x]
    Y = data[y]

    r, p = scipy.stats.pearsonr(X, Y)
    fit, _, *_ = np.polyfit(X, Y, 1, full=True)
    return fit[0], fit[1], r, p, r ** 2