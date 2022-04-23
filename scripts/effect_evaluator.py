import math
import sys
import os

import json
import pandas as pd

from itz.data import VAR_NAMES

DEPENDENT_VARS = [
        # 'd_2010_2018_resid_units',
       'd_2010_2018_per_capita_income',
       'd_2010_2018_percent_non_hispanic_or_latino_white_alone',
       'd_2010_2018_percent_non_hispanic_black_alone',
       'd_2010_2018_percent_hispanic_any_race',
       'd_2010_2018_percent_non_hispanic_asian_alone',
       'd_2010_2018_percent_multi_family_units',
       'd_2010_2018_percent_occupied_housing_units',
       'd_2010_2018_median_gross_rent', 'd_2010_2018_median_home_value',
       'd_2010_2018_percent_households_with_people_under_18',
       'd_2010_2018_percent_of_households_in_same_house_year_ago',
       'd_2010_2018_percent_bachelor_degree_or_higher',
       'd_2010_2018_percent_car_commuters',
       'd_2010_2018_percent_public_transport_commuters',
       'd_2010_2018_percent_public_transport_trips_under_45_min',
       'd_2010_2018_percent_car_trips_under_45_min',
    #    'd_2010_2018_feet_distance_from_park',
       'd_2010_2018_square_meter_greenspace_coverage']

def effect_aggregator(dependent_var, independent_var, data, consider_nonsignificant=True):
    filtered_data = data[data["lval"] == dependent_var]
    filtered_data = filtered_data[filtered_data["op"] == "~"]
    try:
        direct_effect = filtered_data[filtered_data["rval"] == independent_var]["Estimate"].astype(float).iloc[0]
    except IndexError:
        direct_effect = 0
    filtered_data = filtered_data[filtered_data["rval"] != independent_var]
    print(filtered_data)
    print("Independent to mediating estmate", "Mediating to dependent estimate", "Indirect effect", "Mediating variable")
    indirect_effects = 0
    for index, row in filtered_data.iterrows():
        if not consider_nonsignificant:
            if row["p-value"] > 0.05:
                continue
        indirect_data = data[data["lval"] == row["rval"]]
        if len(indirect_data) == 0:
            continue
        indirect_data = indirect_data[indirect_data["op"] == "~"]
        indirect_effect = indirect_data[indirect_data["rval"] == independent_var]["Estimate"]
        if len(indirect_effect) == 0:
            continue
        print(indirect_effect.iloc[0], row["Estimate"], indirect_effect.iloc[0] * row["Estimate"], row["rval"])
        indirect_effects += indirect_effect.iloc[0] * row["Estimate"]
    print(f"Direct Effect of {independent_var} on {dependent_var}: {direct_effect}")
    print(f"Indirect Effects of {independent_var} on {dependent_var}: {indirect_effects}")
    print(f"Total Effects of {independent_var} on {dependent_var}: {direct_effect + indirect_effects}")

def get_total_effect_dfs(regression_graph, x, y):
    visited = set()
    paths = set()
    current_path = []

    def _dfs_util(u):
        """Returns the set of path sums discovered.
        """
        print(current_path)
        if u in visited:
            return
        visited.add(u)
        current_path.append(u)
        if u == y:
            paths.add(tuple(current_path))
            visited.remove(u)
            current_path.pop()
            return
        for v, _ in regression_graph[u]:
            if v not in visited:
                _dfs_util(v)
                try:
                    visited.pop(v)
                except:
                    pass
        current_path.pop()
        visited.remove(u)

    _dfs_util(x)
    print("dfs created!")
    # print(f"{paths=}")

    # path_totals = set()
    path_totals = []
    for path in paths:
        path_total = 1
        for i, v in enumerate(path[1:], 1):
            w = 0
            for neighbor, weight in regression_graph[path[i - 1]]:
                if neighbor == v:
                    w = weight
                    break
            path_total *= w
        # path_totals.add(path_total)
        path_totals.append([path, path_total])
    path_totals = pd.DataFrame(path_totals, columns=["path", "estimate"])
    abs_path_totals = path_totals["estimate"].apply(lambda x: abs(x))
    path_totals["abs_estimate"] = abs_path_totals
    path_totals.sort_values(by="abs_estimate", inplace=True, ascending=False)
    path_totals.to_csv("test.csv")
    for _, row in path_totals.iterrows():
        print(str(row["path"])+" ", end=None)
        print(row["estimate"])
    print(path_totals)
    print(path_totals.iloc[:,1].sum())
    return path_totals.iloc[:,1].sum()


if __name__ == "__main__":
    mode = sys.argv[1]
    model_path = sys.argv[2]
    inspection = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    regression_graph = {
        var: [] for var in set(inspection["lval"]) | set(inspection["rval"])
    }
    for i, row in inspection.iterrows():
        if row["op"] == "~":
            regression_graph[row["rval"]].append((row["lval"], row["Estimate"]))
    if mode == "all":
        output = sys.argv[3]
        results = {}
        for dependent_var in DEPENDENT_VARS:
            dep_results = {}
            for independent_var in VAR_NAMES:
                if independent_var == "all_vars":
                    continue
                if independent_var == dependent_var:
                    continue
                print(independent_var, dependent_var)
                sum = get_total_effect_dfs(regression_graph, independent_var, dependent_var)
                if sum != 0:
                    dep_results[independent_var] = sum
            results[dependent_var] = dep_results
        with open(output, "w") as f:
            json.dump(results, f, indent=4)
    elif mode == "one":
        independent_var = sys.argv[3]
        dependent_var = sys.argv[4]
        # effect_aggregator(dependent_var, independent_var, inspection)
        get_total_effect_dfs(regression_graph, independent_var, dependent_var)
    # df = pd.DataFrame(columns=["lval", "op", "rval", "Estimate"], index=list(range(1, 9)))
    # df.loc[8] = pd.Series({"lval": "y", "op": "~", "rval": "x", "Estimate": 5})
    # df.loc[7] = pd.Series({"lval": "y", "op": "~", "rval": "c", "Estimate": 6})
    # df.loc[6] = pd.Series({"lval": "y", "op": "~", "rval": "d", "Estimate": 8})
    # df.loc[5] = pd.Series({"lval": "d", "op": "~", "rval": "x", "Estimate": 7})
    # df.loc[4] = pd.Series({"lval": "c", "op": "~", "rval": "b", "Estimate": 2})
    # df.loc[3] = pd.Series({"lval": "b", "op": "~", "rval": "a", "Estimate": 4})
    # df.loc[2] = pd.Series({"lval": "b", "op": "~", "rval": "x", "Estimate": 1})
    # df.loc[1] = pd.Series({"lval": "a", "op": "~", "rval": "c", "Estimate": 3})
    # print(get_total_effect_dfs(df, "x", "y"))