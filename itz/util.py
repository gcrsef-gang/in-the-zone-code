"""Statistics utility functions.
"""

from typing import Tuple
import math

import numpy as np
import pandas as pd
import scipy


# Value for shifting variables before log transformations. 
LOG_TRANSFORM_SHIFT = 0.1


def log_transform(X: pd.Series) -> pd.Series:
    """Returns the log-transformed version of a variable.

    NOTE: Assumes:
    - All values are >= 0
    - No NaNs present in data
    """
    return (X + LOG_TRANSFORM_SHIFT).transform(math.log)


def get_data_linreg(x: str, y: str, data: pd.DataFrame, transformation_x=lambda x:x, transformation_y=lambda x:x
        ) -> Tuple[pd.Series, pd.Series]:
    """Obtains data from a DataFrame for a linear regression.
    """
    X = data[x][data[x].notnull()][data[y].notnull()]
    Y = data[y][data[x].notnull()][data[y].notnull()]
    if transformation_x:
        try:
            X = X.transform(transformation_x)
        except OverflowError:
            raise Exception("Transformation out of range! Are all values >= 0?")
    if transformation_y:
        try:
            Y = Y.transform(transformation_y)
        except OverflowError:
            raise Exception("Transformation out of range! Are all values >= 0?")
    return X, Y


def regress(x: str, y: str, data: pd.DataFrame, transformation_x=lambda x:x, transformation_y=lambda x:x) -> Tuple:
    """Returns the slope, intercept, correlation coefficient, two-tailed p-value, coefficient of
    determination, and the resulting regression function.

    NOTE: function returned expects UNTRANSFORMED inputs and gives UNTRANSFORMED outputs.
    """
    X, Y = get_data_linreg(x, y, data, transformation_x, transformation_y)

    r, p = scipy.stats.pearsonr(X, Y)
    fit, _, *_ = np.polyfit(X, Y, 1, full=True)

    # transform_x = lambda x_: math.log(x_ + LOG_TRANSFORM_SHIFT) if transformation_x else x_
    # transform_y = lambda y_: math.e ** y_ - LOG_TRANSFORM_SHIFT if transformation_y else y_

    # return fit[0], fit[1], r, p, r ** 2, lambda x_: transform_y(fit[0] * transform_x(x_) + fit[1])
    return fit[0], fit[1], r, p, r ** 2, lambda x_: transformation_y(fit[0] * transformation_x(x_) + fit[1])