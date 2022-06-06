"""
usage:
python3 scripts/effect_evaluator.py <model_path> <x> <y> <output_path>
python3 scripts/effect_evaluator.py <model_path> all <output_path>
"""


import itertools
import math
import sys
import os

import json
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


SEARCH_DEPTH = 2


def get_regression_graph(inspection):
    regression_graph = {
        var: [] for var in set(inspection["lval"]) | set(inspection["rval"])
    }
    for i, row in inspection.iterrows():
        if row["op"] == "~":
            regression_graph[row["rval"]].append((row["lval"], row["Std. Err"], row["Estimate"], row["p-value"]))
    return regression_graph


def get_total_effect_dfs(regression_graph, x, y, output_path=None):
    # regression_graph = get_regression_graph(inspection)

    visited = set()
    paths = set()
    current_path = []

    def _dfs_util(u, depth=0):
        """Returns the set of path sums discovered.
        """
        # print(current_path)
        if u in visited:
            return
        visited.add(u)
        current_path.append(u)
        if u == y or depth == SEARCH_DEPTH:
            if u == y:
                paths.add(tuple(current_path))
            visited.remove(u)
            current_path.pop()
            return
        for v, _, _, _ in regression_graph[u]:
            _dfs_util(v, depth + 1)
        current_path.pop()
        visited.remove(u)

    _dfs_util(x)
    # print("dfs created!")
    # print(f"{paths=}")

    path_totals = []
    for path in paths:
        path_weights = []
        path_errors = []
        path_p_values = []
        path_total = 1
        for i, v in enumerate(path[1:], 0):
            w = 0
            p_val = 0
            e = 0
            for neighbor, error, weight, p_value in regression_graph[path[i]]:
                if neighbor == v:
                    w = weight
                    p_val = p_value
                    e = error
                    break
            path_total *= w
            path_errors.append(e)
            path_p_values.append(p_val)
            path_weights.append(w)
        # if path[0] == x and path[1] == y:
            # print(f"Direct: {regression_graph[path[0]][1]}")
        path_string = path[0]
        for i in range(len(path_weights)):
            path_string += f" ---- Est: {round(path_weights[i], 3)} Err: {round(path_errors[i], 3)} P-val: {round(path_p_values[i], 3)} ---> {path[i + 1]}"
        path_totals.append([path_string, path_total])

    path_totals = pd.DataFrame(path_totals, columns=["path", "estimate"])
    abs_path_totals = path_totals["estimate"].apply(lambda x: abs(x))
    path_totals["abs_estimate"] = abs_path_totals
    path_totals.sort_values(by="abs_estimate", inplace=True, ascending=False)
    # if output_path != None:
    print(path_totals)
    print("Number of paths: ", len(path_totals))
    if output_path is not None:
        path_totals.to_csv(output_path)
    path_totals.to_csv("test.csv")
    # print(path_totals.iloc[:1000,1].cumsum().tolist())
    return path_totals.iloc[:,1].sum(), path_totals


if __name__ == "__main__":
    mode = sys.argv[1]
    model_path = sys.argv[2]
    inspection = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    regression_graph = get_regression_graph(inspection)
    if mode == "all":
        variables = set(inspection["lval"]) | set(inspection["rval"])
        num_perms = math.comb(len(variables), 2) * 2

        effects = pd.DataFrame(columns=["x", "y", "total effect", "abs total effect"], index=list(range(num_perms)))
        i = 0
        # for combo in itertools.combinations(variables, 2):
            # x, y = combo
                
            # print("\r                                             ", end="")
            # print(f"\r{round((i / num_perms) * 100, 1)}% complete... (permutation {i}/{num_perms})", end="")
            # sys.stdout.flush()
            # total_effect_xy = get_total_effect_dfs(inspection, x, y)
            # effects.loc[i] = [x, y, total_effect_xy, abs(total_effect_xy)]
            # i += 1
            # total_effect_yx = get_total_effect_dfs(inspection, y, x)
            # effects.loc[i] = [y, x, total_effect_yx, abs(total_effect_yx)]
            # i += 1
        for y in variables:
            for x in variables:
                if y != x:
                    print("\r                                             ", end="")
                    print(f"\r{round((i / num_perms) * 100, 1)}% complete... (permutation {i}/{num_perms})", end="")
                    sys.stdout.flush()
                    total_effect_xy, path_totals = get_total_effect_dfs(regression_graph, x, y)
                    if total_effect_xy == 0:
                        continue
                    if len(path_totals) == 0:
                        continue
                    # path_totals.to_csv("test.csv")
                    sorted_path_totals = path_totals.sort_values("abs_estimate", ascending=False)
                    # print(type(path_totals))
                    # print(type(sorted_path_totals))
                    print(sorted_path_totals.columns, "yoo??")
                    print(sorted_path_totals, "sorted path")
                    for path in sorted_path_totals.iterrows():
                        # print(path, type(path), "a path!")
                        print(path[0], "asf", type(path[1]))
                        print(path_totals)
                        effects.loc[i] = [path[1][0], "", path[1][1], path[1][2]]   
                        i += 1
                    effects.loc[i] = [x, y, total_effect_xy, abs(total_effect_xy)]
                    i += 1
                    # total_effect_yx = get_total_effect_dfs(inspection, y, x)
                    # effects.loc[i] = [y, x, total_effect_yx, abs(total_effect_yx)]
                    # i += 1
        print("\r                                             ", end="")
        print(f"\r{round((i / num_perms) * 100, 1)}% complete... (permutation {i}/{num_perms})", end="")
        sys.stdout.flush()
        print()

        # effects = effects.sort_values("abs total effect", ascending=False)
        effects.to_csv(sys.argv[3])
    else:
        independent_var = sys.argv[3]
        dependent_var = sys.argv[4]
        # print(get_total_effect_dfs(inspection, independent_var, dependent_var, output_path=sys.argv[4]+".csv"))
        total_effect, path_totals = get_total_effect_dfs(regression_graph, independent_var, dependent_var)
        print(total_effect)
        path_totals.to_csv("test.csv")
