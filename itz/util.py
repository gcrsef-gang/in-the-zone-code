"""Statistics utility functions.
"""

from enum import Enum
from typing import Tuple
import math

import numpy as np
import pandas as pd
import scipy


# Value for shifting variables before log transformations. 
LOG_TRANSFORM_SHIFT = 0.001
RECIPROCAL_TRANSFORM_SHIFT = 0.001


class Transformations(Enum):
    log = (lambda x: math.log(x + LOG_TRANSFORM_SHIFT, math.e))
    # duplicates for convenient use
    ln = (lambda x: math.log(x + LOG_TRANSFORM_SHIFT, math.e))
    log10 = (lambda x: math.log(x + LOG_TRANSFORM_SHIFT, 10))
    log2 = (lambda x: math.log(x + LOG_TRANSFORM_SHIFT, 2))
    expe = lambda y: math.e ** y - LOG_TRANSFORM_SHIFT
    exp2 = lambda y: 2 ** y - LOG_TRANSFORM_SHIFT
    exp10 = lambda y: 10 ** y - LOG_TRANSFORM_SHIFT
    square = lambda x: x*x
    cube = lambda x: x**3
    cbrt = lambda x: x**(1/3)
    sqrt = lambda x: math.sqrt(x)
    reciprocal = lambda x: 1/(x+RECIPROCAL_TRANSFORM_SHIFT)
    identity = lambda x: x


TRANSFORMATION_NAMES = (
    'log',
    'ln',
    'log10',
    'log2',
    'expe',
    'exp2',
    'exp10',
    'square',
    'cube',
    'cbrt',
    'sqrt',
    'reciprocal',
    'identity'
)


def log_transform(X: pd.Series) -> pd.Series:
    """Returns the log-transformed version of a variable.

    NOTE: Assumes:
    - All values are >= 0
    - No NaNs present in data
    """
    return (X + LOG_TRANSFORM_SHIFT).transform(math.log)

def square_transform(X: pd.Series) -> pd.Series:
    """Returns the log-transformed version of a variable.

    NOTE: Assumes:
    - All values are >= 0
    - No NaNs present in data
    """
    return (X).transform(lambda x: x*x)/10000

def sqrt_transform(X: pd.Series) -> pd.Series:
    """Returns the log-transformed version of a variable.

    NOTE: Assumes:
    - All values are >= 0
    - No NaNs present in data
    """
    return (X).transform(lambda x: math.pow(x, 0.5))


def get_data_linreg(x: str, y: str, data: pd.DataFrame, transformation_x=Transformations.identity, transformation_y=Transformations.identity
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


def regress(x: str, y: str, data: pd.DataFrame, transformation_x=Transformations.identity, transformation_y=Transformations.identity) -> Tuple:
    """Returns the slope, intercept, correlation coefficient, two-tailed p-value, coefficient of
    determination, and the resulting regression function.

    NOTE: function returned expects UNTRANSFORMED inputs and gives UNTRANSFORMED outputs.
    """
    X, Y = get_data_linreg(x, y, data, transformation_x, transformation_y)
    assert not X.isnull().values.any()
    assert not Y.isnull().values.any()

    r, p = scipy.stats.pearsonr(X, Y)
    fit, _, *_ = np.polyfit(X, Y, 1, full=True)

    # transform_x = lambda x_: math.log(x_ + LOG_TRANSFORM_SHIFT) if transformation_x else x_
    # transform_y = lambda y_: math.e ** y_ - LOG_TRANSFORM_SHIFT if transformation_y else y_

    # return fit[0], fit[1], r, p, r ** 2, lambda x_: transform_y(fit[0] * transform_x(x_) + fit[1])
    return fit[0], fit[1], r, p, r ** 2, lambda x_: transformation_y(fit[0] * transformation_x(x_) + fit[1])