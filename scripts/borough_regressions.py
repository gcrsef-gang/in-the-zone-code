# NOTE: remember to run this from the project root

import pandas as pd

import itz


# BOROUGHS = ["MN", "BK", "SI", "QN", "BX"]
# BOROUGHS = ["MN", "BK", "QN", "BX"]
BOROUGHS = ["MN"]
# Nothing was upzoned in SI
BOROUGH_TRANSFORMATIONS = {
    "MN": (itz.util.Transformations.ln, itz.util.Transformations.identity),
    "BK": (itz.util.Transformations.identity, itz.util.Transformations.identity),
    "QN": (itz.util.Transformations.identity, itz.util.Transformations.identity),
    "BX": (itz.util.Transformations.identity, itz.util.Transformations.identity)
}


def _print_stats(dict_):
    """Prints a dictionary of statistics.
    """
    for key, val in dict_.items():
        print(f"{key}: {round(val, 3)}")


def do_regression(df, name, transform_x, transform_y):
    print(transform_x, transform_y)
    x, y = "2002_2010_percent_upzoned", "d_2010_2018_pop_density"
    regression_stats = itz.make_regression_plot(x, y, df, f"{name}_regression.png", transform_x, transform_y)
    resid_plot_stats = itz.make_residual_plot(x, y, df, f"{name}_resid_plot.png", transform_x, transform_y)
    # func = itz.util.regress(x, y, df)[-1]
    func = itz.util.regress(x, y, df, transform_x, transform_y)[-1]
    residual_df = pd.DataFrame()
    residual_df["resids"] = df[x].transform(func) - df[y]
    resid_hist_stats = itz.make_histogram("resids", residual_df, f"{name}_resid_hist.png")
    _print_stats({**regression_stats, **resid_plot_stats, **resid_hist_stats})

if __name__ == "__main__":
    data = pd.read_csv("in-the-zone-data/itz-data.csv")
    for borough in BOROUGHS:
        do_regression(data.loc[data["ITZ_GEOID"].str.contains(borough)],
                      borough, *BOROUGH_TRANSFORMATIONS[borough])