"""Various visualization functions.
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import semopy

from .model import ModelName


def make_sem_diagram(model_name: ModelName, path: str):
    """Saves a diagram of an SEM to a PNG image.
    """
    # semplot()


def make_regression_plot(x: str, y: str, data: pd.DataFrame, path: str):
    """Creates a scatterplot with LSRL and returns descriptive statistics as a dictionary.
    """
    X = data[x]
    Y = data[y]

    fit, stats = np.polyfit(X, Y, 1)
    
    plt.plot(X, (fit[0] * np.asarray(X) + fit[1]), '-r')
    plt.scatter(X, Y)
    plt.savefig(path)



def make_residual_plot(x: str, y: str, data: pd.DataFrame, path: str):
    """Creates a residual plot for a least-squares linear regression and returns descriptive
    statistics as a dictionary.
    """
    X = data[x]
    Y = data[y]

    coef = np.polyfit(X, Y, 1)
    fn = np.poly1d(coef)

    plt.plot(X, np.array([0] * len(X)), '-r')
    plt.scatter(X, Y - fn(X))

    plt.savefig(path)



def make_histogram(x: str, data: pd.DataFrame, path: str):
    """"Creates a histogram and returns descriptive statistics as a dictionary.
    """
    plt.hist(data[x])
    plt.savefig(path)


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