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
from enum import Enum
from typing import List, Tuple

from .model import ModelName, get_description
from .util import get_data_linreg, regress, Transformations


def make_sem_diagram(model_name: ModelName, data: pd.DataFrame, path: str, verbose: bool=False):
    """Saves a diagram of an SEM to a PNG image.
    """
    desc, _ = get_description(model_name, data, verbose)
    _ = semopy.semplot(semopy.Model(desc), filename=path)


def make_regression_plot(x: str, y: str, data: pd.DataFrame, path: str,
        transformation_x=Transformations.identity, transformation_y=Transformations.identity):
    """Creates a scatterplot with LSRL and returns descriptive statistics as a dictionary.
    """
    X, Y = get_data_linreg(x, y, data, transformation_x, transformation_y)
    print(transformation_x, transformation_x)

    slope, intercept, r, two_tailed_p, r_squared, _ = regress(x, y, data, transformation_x, transformation_y)
    # slope = -188
    # intercept = 7865.1
    plt.plot(X, (slope * np.asarray(X) + intercept), '-r')

    if transformation_x:
        plt.xlabel("transformed_" + x)
    else:
        plt.xlabel(x)
    if transformation_y:
        plt.ylabel("transformed_" + y)
    else:
        plt.ylabel(y)

    # plt.figure(figsize=(10, 40))
    plt.scatter(X, Y)
    plt.title(f"R^2: {round(r_squared, 3)} r: {round(r, 3)} p: {round(two_tailed_p, 5)}"
              f"Num Obsv: {len(X)}")
    plt.savefig(path)
    plt.clf()

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "p": two_tailed_p,
        "R^2": r_squared,
        "Transformed mean" if transformation_x else "X mean": X.mean(),
        "Transformed mean" if transformation_y else "Y mean": Y.mean(),
        # "log(X) mean" if transformation_X else "X mean": X.mean(),
        # "log(Y) mean" if transformation_y else "Y mean": Y.mean(),
        "n": len(X)
    }


def make_correlation_matrix(data: pd.DataFrame, output_path: str, path: str):
    # x = "2010_2018_percent_upzoned"
    # logged = data[x][data[x].notnull()][data[x] > 0].transform(math.log)
    # data[x] = logged
    x = data.corr()
    print("Number of columns:", len(data.columns))
    # x.to_csv("correlation_matrix.csv")
    # x["d_2010_2018_pop_density"].to_csv("d_2010_2018_pop_density_correlations.csv")
    x.to_csv(output_path)
    # plt.imshow(x)
    plt.figure(figsize=(40,40))
    sn.heatmap(x, cmap='coolwarm')
    # plt.savefig("test.png")
    if path:
        plt.savefig(path)
        plt.clf()
    return x

def make_covariance_matrix(data: pd.DataFrame, output_path: str, path: str):
    # x = "2010_2018_percent_upzoned"
    # logged = data[x][data[x].notnull()][data[x] > 0].transform(math.log)
    # data[x] = logged
    x = data.cov()
    print("Number of columns:", len(data.columns))
    # x.to_csv("correlation_matrix.csv")
    # x["d_2010_2018_pop_density"].to_csv("d_2010_2018_pop_density_correlations.csv")
    x.to_csv(output_path)
    # plt.imshow(x)
    plt.figure(figsize=(40,40))
    sn.heatmap(x, cmap='coolwarm')
    # plt.savefig("test.png")
    if path:
        plt.savefig(path)
        plt.clf()
    return x


def make_residual_plot(x: str, y: str, data: pd.DataFrame, path: str,
        transformation_x=Transformations.identity, transformation_y=Transformations.identity):
    """Creates a residual plot for a least-squares linear regression and returns descriptive
    statistics as a dictionary.
    """
    X, Y = get_data_linreg(x, y, data, transformation_x, transformation_y)
    slope, intercept, _, _, _, _ = regress(x, y, data, transformation_x, transformation_y)
    resids = (Y - (slope * X + intercept)).to_numpy()

    digitized = np.digitize(X, bins=np.linspace(min(X), max(X), num=100))
    bin_size = (max(X) - min(X)) / 100
    bins = [[] for _ in range(100)]
    for i, bin_ in enumerate(digitized):
        bins[bin_ - 1].append(resids[i])
    variances = [np.var(np.array(bin_)) for bin_ in bins]

    # plt.plot(X, np.zeros(len(X)), '-r')
    plt.title("Variance of residuals")
    plt.scatter(np.arange(len(bins)) * bin_size - min(X), variances)
    plt.savefig(path)
    plt.clf()

    return {
        "resid mean": resids.mean(),
        "resid stdev": resids.std()
    }


def make_histogram(x: str, data: pd.DataFrame, path: str, transformation=lambda x: x):
    """"Creates a histogram and returns descriptive statistics as a dictionary.
    """
    print(data[x])
    print(data.columns)
    # subdata = data[data["2002_2010_percent_upzoned"] < 50]
    # superdata = data[data["2002_2010_percent_upzoned"] > 50]
    X = (data[x][data[x].notnull()]).transform(transformation)
    # Y = (subdata[x][subdata[x].notnull()]).transform(transformation)
    # Z = (superdata[x][superdata[x].notnull()]).transform(transformation)
    mean, stdev = X.mean(), X.std()
    # print(Y.mean(), Z.mean())
    print(X[X > 100])
    plt.hist(X, bins=100)
    # plt.hist(Y, bins=200)
    # plt.hist(Z, bins=200)
    # plt.hist(X, bins=20)
    plt.title(f"{x} M: {round(mean, 3)} S: {round(stdev, 3)}")
    plt.savefig(path)
    plt.clf()
    return {
        "mean": mean,
        "stdev": stdev,
        "n": len(X)
    }


def make_map_vis(geoset: dict, data: pd.DataFrame, path: str, columns: List[str], tracts: bool):
    """Creates an html file containing an interactive choropleth map based on the specified values
    """


    # determine location by geoset?
    m = folium.Map(location=[40.7, -74], zoom_start=10)

    to_remove = []
    if tracts:
        for tract in geoset['features']:
            # print(data["ITZ_GEOID"], tract["id"])
            if tract["properties"]["ITZ_GEOID"] not in data["ITZ_GEOID"].values:
                to_remove.append(tract)
    else:
        for tract in geoset['features']:
            # print(data["BBL"], tract["id"])
            if tract["properties"]["BBL"] not in data["BBL"].values:
                to_remove.append(tract)

    for tract in to_remove:
        geoset['features'].remove(tract)

    # for index in data.index:  
    # for tract in geoset['features']:
        # tract['properties']['ITZ_GEOID'] = data.loc[tract['id'], "ITZ_GEOID"] + "\n"
    # print(len(geoset['features']))
    for column in columns[1:]:
        if "upzoned" in column:
            bins = [0,0.1,5,10,25,50,75,90,95,100]
        else:
            bins = list(data[column].quantile([0, .25, .5, .75, 1]))
        # bins.append(0)
        # bins = sorted(bins)
        if tracts:
            choropleth = folium.Choropleth(
                geo_data=geoset,
                data=data,
                columns=["ITZ_GEOID", column],
                key_on="feature.properties.ITZ_GEOID",
                fill_color="PuBuGn",
                fill_opacity=0.7,
                line_opacity=0.5,
                legend_name=column,
                bins=bins,
                reset=True,
                name=column,
                highlight=True,
            ).add_to(m)
        else:
            choropleth = folium.Choropleth(
                geo_data=geoset,
                data=data,
                columns=["BBL", column],
                key_on="feature.properties.BBL",
                fill_color="PuBuGn",
                fill_opacity=0.7,
                line_opacity=0.5,
                legend_name=column,
                bins=bins,
                reset=True,
                name=column,
                highlight=True,
            ).add_to(m)

        # prepare the customised text
        tooltip_text = {}
        for index in data.index:
            if tracts:
                tooltip_text[data.loc[index, "ITZ_GEOID"]] = str(round(data.loc[index, column], 3))
            else:
                tooltip_text[data.loc[index, "BBL"]] = str(round(data.loc[index, column], 3))
        for tract in geoset['features']:
            if tracts:
                tract['properties'][column] = tooltip_text[tract['properties']['ITZ_GEOID']]
            else:
                tract['properties'][column] = tooltip_text[tract['properties']['BBL']]
        # Append a tooltip column with customised text
        # Display Region Label
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(columns)
            # folium.features.GeoJsonTooltip(columns, aliases=columns)
        )
    folium.LayerControl().add_to(m)
    m.save(path)