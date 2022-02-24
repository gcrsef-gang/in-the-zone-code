"""Various visualization functions.
"""

# import semopy

import folium
import matplotlib.pyplot as plt
import numpy as np




def make_sem_diagram(model, path):
    """Saves a diagram of an SEM to a PNG image.

    Model options:
    - long-term
    - short-term-with-emissions
    - short-term-no-emissions
    """
    # semplot()


def make_regression_plot(x, y, path):
    """Creates a scatterplot with LSRL and returns descriptive statistics as a dictionary.
    """
    fit, stats = np.polyfit(x, y, 1)
    
    plt.plot(x, (m * np.asarray(x) + b), '-r')
    plt.scatter(x, y)
    plt.savefig(path)



def make_residual_plot(x, y, path):
    """Creates a residual plot for a least-squares linear regression and returns descriptive
    statistics as a dictionary.
    """
    coef = np.polyfit(x, y, 1)
    fn = np.poly1d(coef)

    plt.plot(x, np.array([0] * len(x)), '-r')
    plt.scatter(x, y - fn(x))

    plt.savefig(path)



def make_histogram(x, path):
    """"Creates a histogram and returns descriptive statistics as a dictionary.
    """
    plt.hist(x)
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