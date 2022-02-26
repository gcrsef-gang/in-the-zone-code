"""Implementation of methods described in *In the Zone: The effects of zoning regulation changes on urban life* by Arin Khare, James Lian, and Kai Vernooy.
"""

from itz.data import get_data
from itz.model import evaluate, fit, get_description
from itz.visualization import (make_sem_diagram, make_regression_plot, make_residual_plot,
                               make_histogram)