"""Various visualization functions.
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import semopy
import seaborn as sn
import scipy

from .model import ModelName


def make_sem_diagram(model_name: ModelName, path: str):
    """Saves a diagram of an SEM to a PNG image.
    """
    # semplot()
def log_transformation(data: pd.Series):
    pass

def make_regression_plot(x: str, y: str, data: pd.DataFrame, path: str, log_x: bool, log_y: bool):
    """Creates a scatterplot with LSRL and returns descriptive statistics as a dictionary.
    """
    if log_x and log_y:
        X = data[x][data[x].notnull()][data[y].notnull()][data[x] > 0][data[y] > 0].transform(math.log)
        Y = data[y][data[x].notnull()][data[y].notnull()][data[x] > 0][data[y] > 0].transform(math.log)
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

    x_average = X.sum()/len(X)
    y_average = Y.sum()/len(Y)
    print(f"X Average: {x_average}")
    print(f"Y Average: {y_average}")
    print(fit[0], "slope", fit[1], "intercept")
    
    total_sum_of_squares = Y-y_average
    total_sum_of_squares *= total_sum_of_squares
    r_squared = 1 - stats[0]/total_sum_of_squares.sum()
    print("R squared", r_squared)
    r, two_tailed_p = scipy.stats.pearsonr(X,Y)
    # r = np.corrcoef(X, Y)[0][1]
    print("R (correlation)", r)
    print("Two-tailed p value", two_tailed_p)

    # fit here is actually a scalar
    plt.plot(X, (fit[0] * np.asarray(X) + fit[1]), '-r')

    if log_x:
        plt.xlabel("log_"+x)
    else:
        plt.xlabel(x)
    if log_y:
        plt.ylabel("log_"+y)
    else:
        plt.ylabel(y)
    plt.scatter(X, Y)
    plt.title(f"R^2: {round(r_squared, 3)} r: {round(r, 3)} p: {round(two_tailed_p, 5)} Num Obsv: {len(X)} Equation: {round(fit[0],5)}x + {round(fit[1],3)}")
    plt.savefig(path)
    plt.clf()
    # print(np.corrcoef(dat a))
    return stats

def make_correlation_matrix(data: pd.DataFrame, output_path: str, path: str):
    x = "2011_2019_percent_upzoned"
    logged = data[x][data[x].notnull()][data[x] > 0].apply(math.log)
    data[x] = logged
    x = data.corr()
    # x.to_csv("correlation_matrix.csv")
    x["d_2011_2019_pop_density"].to_csv("d_2011_2019_pop_density_correlations.csv")
    x.to_csv(output_path)
    # plt.imshow(x)
    plt.figure(figsize=(40,40))
    sn.heatmap(x, cmap='coolwarm')
    # plt.savefig("test.png")
    if path:
        plt.savefig(path)
    return x


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
    # plt.hist((data[x][data[x] != 0]), bins=np.arange(-200, 200, 0.5))
    # plt.hist((data[x][data[x] != 0]))
    average = data[x].sum()/len(data[x])
    print(f"Average: {average}")
    plt.title(f"{x}  Average: {average}")
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