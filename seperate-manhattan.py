import pandas as pd


data = pd.read_csv("in-the-zone-data/updated-itz-data.csv")
manhattan = data.loc[data["ITZ_GEOID"].str.contains("MN")]
non_manhattan = data.loc[(1 - data["ITZ_GEOID"].str.contains("MN").astype(int)).astype(bool)]
manhattan.reset_index(inplace=True)
non_manhattan.reset_index(inplace=True)


import itz

def _print_stats(dict_):
    """Prints a dictionary of statistics.
    """
    for key, val in dict_.items():
        print(f"{key}: {round(val, 3)}")

def do_regression(df, name):
    x, y = "2002_2010_percent_upzoned", "d_2010_2018_pop_density"
    regression_stats = itz.make_regression_plot(x, y, df, f"{name}_regression.png")
    resid_plot_stats = itz.make_residual_plot(x, y, df, f"{name}_resid_plot.png")
    func = itz.util.regress(x, y, df)[-1]
    residual_df = pd.DataFrame()
    residual_df["resids"] = df[x].transform(func) - df[y]
    resid_hist_stats = itz.make_histogram("resids", residual_df, f"{name}_resid_hist.png")
    _print_stats({**regression_stats, **resid_plot_stats, **resid_hist_stats})

do_regression(manhattan, "MN")
do_regression(non_manhattan, "non_MN")

for index, row in data.iterrows():
    if index in manhattan.index:
        data.loc[index, "2002_2010_percent_upzoned_manhattan"] = data.loc[index, "2002_2010_percent_upzoned"]
        data.loc[index, "2002_2010_percent_upzoned_non_manhattan"] = None
    else:
        data.loc[index, "2002_2010_percent_upzoned_manhattan"] = None
        data.loc[index, "2002_2010_percent_upzoned_non_manhattan"] = data.loc[index, "2002_2010_percent_upzoned"]

data.to_csv("in-the-zone-data/extra-updated-itz-data.csv")