# NOTE: remember to run this from the project root

import pandas as pd

import itz


BOROUGHS = ["MN", "BK", "SI", "QN", "BX"]
BOROUGH_TRANSFORMATIONS = {
    "MN": itz.util.Transformations.identity,
    "BK": itz.util.Transformations.identity,
    "QN": itz.util.Transformations.identity,
    "BX": itz.util.Transformations.identity,
    "SI": itz.util.Transformations.identity
}
VARIABLES = ["2002_2010_percent_upzoned"]


def _print_stats(dict_):
    """Prints a dictionary of statistics.
    """
    for key, val in dict_.items():
        print(f"{key}: {round(val, 3)}")


def make_histogram(df, name, transform):
    print(name.upper())
    print("---------------")
    for variable in VARIABLES:
        print(variable)
        df_no_zeros = df[df[variable] != 0.0]
        _print_stats(itz.make_histogram(variable, df_no_zeros, f"{name}_{variable}.png", transformation=transform))
    print()

if __name__ == "__main__":
    data = pd.read_csv("in-the-zone-data/itz-data.csv")
    
    # All boroughs
    # for borough in BOROUGHS:
    #     make_histogram(data.loc[data["ITZ_GEOID"].str.contains(borough)],
    #                    borough, BOROUGH_TRANSFORMATIONS[borough])

    # Manhattan/Non-Manhattan
    make_histogram(data.loc[(1 - data["ITZ_GEOID"].str.contains("MN").astype(int)).astype(bool)], "non_MN_log", itz.util.Transformations.log)
    make_histogram(data.loc[data["ITZ_GEOID"].str.contains("MN")], "MN_log", itz.util.Transformations.log)
    make_histogram(data.loc[(1 - data["ITZ_GEOID"].str.contains("MN").astype(int)).astype(bool)], "non_MN", itz.util.Transformations.identity)
    make_histogram(data.loc[data["ITZ_GEOID"].str.contains("MN")], "MN", itz.util.Transformations.identity)