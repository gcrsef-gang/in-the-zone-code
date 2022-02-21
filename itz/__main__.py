"""
Usage: python3 -m itz <command> [-v]

========
COMMANDS
========

diagram <path>
--------------
Save a diagram of the SEM as a PNG image.

Parameters:
- path: path to output image file.

fit <model>
-----------
Fit an SEM model and print results.

Parameters:
- model: name of the model to fit.

Current model options:
- long-term
- short-term-with-emissions
- short-term-no-emissions

graph <x> [<y>] <path1> [<path2>]
---------------------------------
Create a visualization of one or two variables from our dataset. Descriptive statistics will be
printed to the console.

Parameters:
- x: name of explanatory variable to graph
- y (optional): name of response variable to graph
- path1: path to first output image file
- path2 (optional): path to second output image file

If y is specified, a scatterplot with an LSRL and/or a residual plot will be produced.
If y is not specified, a histogram of will be produced.

Use -v for verbosity.
"""

from itz import data, model, visualization


if __name__ == "__main__":
    pass