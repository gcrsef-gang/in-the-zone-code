"""Various visualization functions.
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import semopy

from .model import ModelName


def make_sem_diagram(model_name: ModelName, path: str):
    """Saves a diagram of an SEM to a PNG image.
    """
    # semplot()


def make_regression_plot(x: str, y: str, data: pd.DataFrame, path: str, log_x: bool, log_y: bool):
    """Creates a scatterplot with LSRL and returns descriptive statistics as a dictionary.
    """
    if log_x and log_y:
        X = data[x][data[x].notnull()][data[y].notnull()][data[x] > 0][data[y] > 0].apply(math.log)
        Y = data[y][data[x].notnull()][data[y].notnull()][data[x] > 0][data[y] > 0].apply(math.log)
    elif log_x:
        X = data[x][data[x].notnull()][data[y].notnull()][data[x] > 0].apply(math.log)
        Y = data[y][data[x].notnull()][data[y].notnull()][data[x] > 0]
    elif log_y:
        X = data[x][data[x].notnull()][data[y].notnull()][data[y] > 0]
        Y = data[y][data[x].notnull()][data[y].notnull()][data[y] > 0].apply(math.log)
    else:
        X = data[x][data[x].notnull()][data[y].notnull()]
        Y = data[y][data[x].notnull()][data[y].notnull()]

    # fit, stats, *args = np.polyfit(X, Y, 2, full=True)
    fit, stats, *args = np.polyfit(X, Y, 1, full=True)
    # print(np.polyfit(X,Y,1,full=True))
    # slope, intercept, residuals = np.polyfit(X, Y, 1, full=True)
    # slope, intercept, residuals = np.polyfit(X, Y, 1, full=True)
    # print(fit)
    # print(stats)

    # fit here is actually a scalar
    plt.plot(X, (fit[0] * np.asarray(X) + fit[1]), '-r')
    # plt.plot(X, (fit[0] * np.asarray(X.apply(lambda x: x*x)) + fit[1]*np.asarray(X)+fit[2]), '-r')
    # plt.plot(X, (slope * np.asarray(X) + intercept), '-r')
    if log_x:
        plt.xlabel("log_"+x)
    else:
        plt.xlabel(x)
    if log_y:
        plt.ylabel("log_"+y)
    else:
        plt.ylabel(y)
    plt.scatter(X, Y)
    plt.savefig(path)
    return stats



def make_residual_plot(x: str, y: str, data: pd.DataFrame, path: str):
    """Creates a residual plot for a least-squares linear regression and returns descriptive
    statistics as a dictionary.
    """
    X = data[x][data[x].notnull()][data[y].notnull()]
    Y = data[y][data[x].notnull()][data[y].notnull()]

    coef = np.polyfit(X, Y, 1)
    fn = np.poly1d(coef)

    plt.plot(X, np.array([0] * len(X)), '-r')
    plt.scatter(X, Y - fn(X))

    plt.savefig(path)



def make_histogram(x: str, data: pd.DataFrame, path: str):
    """"Creates a histogram and returns descriptive statistics as a dictionary.
    """
    # plt.hist((data[x][data[x] != 0]).apply(math.log))
    # plt.hist((data[x][data[x] != 0]).apply(math.log2))
    # plt.hist((data[x][data[x] != 0]))
    # plt.hist((data[x][data[x] != 0]), bins=200)
    plt.hist(data[x], bins=200)
    plt.title(x)
    # plt.hist((data[x][data[x] != 0]), bins=np.arange(-200, 200, 0.5))
    # plt.hist((data[x][data[x] != 0]))
    plt.savefig(path)
    plt.clf()


def make_map_vis(geoset, values, path):
    """Creates an html file containing an interactive choropleth map based on the specified values
    """

    bins = list(values.quantile([0, 0.25, 0.5, 0.75, 1]))

    # determine location by geoset?
    m = folium.Map(location=[48, -102], zoom_start=3)

    folium.Choropleth(
        geo_data=geoset,
        data=values,
        columns=["State", "Unemployment"],
        key_on="feature.id",
        fill_color="BuPu",
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name="Unemployment Rate (%)",
        bins=bins,
        reset=True,
    ).add_to(m)

    m.save(path)