# NOTE: remember to run this from the project root

import pandas as pd


# data = pd.read_csv("in-the-zone-data/updated-itz-data.csv")
data = pd.read_csv("in-the-zone-data/integrated-itz-data.csv")
manhattan = data.loc[data["ITZ_GEOID"].str.contains("MN")]
print(manhattan)
non_manhattan = data.loc[(1 - data["ITZ_GEOID"].str.contains("MN").astype(int)).astype(bool)]
print(non_manhattan)
manhattan.reset_index(inplace=True)
non_manhattan.reset_index(inplace=True)



import itz

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
    print(f"{name}_resid_plot.png")
    # func = itz.util.regress(x, y, df)[-1]
    func = itz.util.regress(x, y, df, transform_x, transform_y)[-1]
    residual_df = pd.DataFrame()
    residual_df["resids"] = df[x].transform(func) - df[y]
    resid_hist_stats = itz.make_histogram("resids", residual_df, f"{name}_resid_hist.png")
    _print_stats({**regression_stats, **resid_plot_stats, **resid_hist_stats})

do_regression(manhattan, "MN", itz.util.Transformations.cube, itz.util.Transformations.identity)
do_regression(non_manhattan, "non_MN", itz.util.Transformations.identity, itz.util.Transformations.identity)


data.set_index("ITZ_GEOID", inplace=True)

for index, row in data.iterrows():
    if index[:2] == "MN":
        data.loc[index, "2002_2010_percent_upzoned_manhattan"] = data.loc[index, "2002_2010_percent_upzoned"]
        data.loc[index, "2002_2010_percent_upzoned_non_manhattan"] = None
    else:
        data.loc[index, "2002_2010_percent_upzoned_manhattan"] = None
        data.loc[index, "2002_2010_percent_upzoned_non_manhattan"] = data.loc[index, "2002_2010_percent_upzoned"]
        if index == "SI17":
            print(data.loc[index, ["2002_2010_percent_upzoned_manhattan", "2002_2010_percent_upzoned_non_manhattan"]], "yees")

print(data[["2002_2010_percent_upzoned_manhattan", "2002_2010_percent_upzoned_non_manhattan"]])

# raise Exception("done")
data.to_csv("in-the-zone-data/extra-updated-itz-data.csv")