import math
import sys
import os

import pandas as pd

def effect_aggregator(dependent_var, independent_var, data, consider_nonsignificant=True):
    filtered_data = data[data["lval"] == dependent_var]
    filtered_data = filtered_data[filtered_data["op"] == "~"]
    try:
        direct_effect = filtered_data[filtered_data["rval"] == independent_var]["Estimate"].astype(float).iloc[0]
    except IndexError:
        direct_effect = 0
        
    filtered_data = filtered_data[filtered_data["rval"] != independent_var]
    print(filtered_data)
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
        print(indirect_effect.iloc[0], row["rval"])
        indirect_effects += indirect_effect.iloc[0] * row["Estimate"]
    print(f"Direct Effect of {independent_var} on {dependent_var}: {direct_effect}")
    print(f"Indirect Effects of {independent_var} on {dependent_var}: {indirect_effects}")
    print(f"Total Effects of {independent_var} on {dependent_var}: {direct_effect + indirect_effects}")


def get_total_effect_dfs(inspection, x, y):
    regression_graph = {
        var: [] for var in set(inspection["lval"]) | set(inspection["rval"])
    }
    for i, row in inspection.iterrows():
        if row["op"] == "~":
            regression_graph[row["rval"]].append((row["lval"], row["Estimate"]))

    visited = set()
    paths = set()
    current_path = []

    def _dfs_util(u):
        """Returns the set of path sums discovered.
        """
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
            _dfs_util(v)
        current_path.pop()
        visited.remove(u)

    _dfs_util(x)
    print(f"{paths=}")

    path_totals = set()
    for path in paths:
        path_total = 1
        for i, v in enumerate(path[1:], 1):
            w = 0
            for neighbor, weight in regression_graph[path[i - 1]]:
                if neighbor == v:
                    w = weight
                    break
            path_total *= w
        path_totals.add(path_total)

    return sum(path_totals)


if __name__ == "__main__":
    model_path = sys.argv[1]
    inspection = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    independent_var = sys.argv[2]
    dependent_var = sys.argv[3]
    get_total_effect_dfs(inspection, independent_var, dependent_var)
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