"""
usage:
python3 scripts/effect_evaluator.py <model_path> <x> <y> <output_path>
python3 scripts/effect_evaluator.py <model_path> all <output_path>
"""


import itertools
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


# SEARCH_DEPTH = 2


def get_total_effect_dfs(inspection, x, y, output_path=None):
    regression_graph = {
        var: [] for var in set(inspection["lval"]) | set(inspection["rval"])
    }
    for i, row in inspection.iterrows():
        if row["op"] == "~":
            regression_graph[row["rval"]].append((row["lval"], row["Estimate"], row["p-value"]))
    visited = set()
    paths = set()
    current_path = []

    def _dfs_util(u, depth=0):
        """Returns the set of path sums discovered.
        """
        if u in visited:
            return
        visited.add(u)
        current_path.append(u)
        # if u == y or depth == SEARCH_DEPTH:
        if u == y:
            paths.add(tuple(current_path))
            visited.remove(u)
            current_path.pop()
            return
        for v, _, _ in regression_graph[u]:
            _dfs_util(v, depth + 1)
        current_path.pop()
        visited.remove(u)

    _dfs_util(x)

    path_totals = []
    for path in paths:
        path_weights = []
        path_p_values = []
        path_total = 1
        for i, v in enumerate(path[1:], 1):
            w = 0
            p_val = 0
            for neighbor, weight, p_value in regression_graph[path[i - 1]]:
                if neighbor == v:
                    w = weight
                    p_val = p_value
                    break
            path_total *= w
            path_p_values.append(p_val)
            path_weights.append(w)
        
        path_string = path[0]
        for i in range(len(path_weights)):
            path_string += f" ---- Est: {round(path_weights[i], 3)} P-val: {round(path_p_values[i], 3)} ---> {path[i + 1]}"
        path_totals.append([path_string, path_total])

    path_totals = pd.DataFrame(path_totals, columns=["path", "estimate"])
    abs_path_totals = path_totals["estimate"].apply(lambda x: abs(x))
    path_totals["abs_estimate"] = abs_path_totals
    path_totals.sort_values(by="abs_estimate", inplace=True, ascending=False)
    path_totals.to_csv("test.csv")

    if output_path is not None:
        path_totals.to_csv(output_path)

    return path_totals.iloc[:,1].sum()


if __name__ == "__main__":
    model_path = sys.argv[1]
    inspection = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    
    if sys.argv[2] == "all":
        variables = set(inspection["lval"]) | set(inspection["rval"])
        num_perms = math.comb(len(variables), 2) * 2

        effects = pd.DataFrame(columns=["x", "y", "total effect", "abs total effect"], index=list(range(num_perms)))
        i = 0
        for combo in itertools.combinations(variables, 2):
            x, y = combo
            print("\r                                             ", end="")
            print(f"\r{round((i / num_perms) * 100, 1)}% complete... (permutation {i}/{num_perms})", end="")
            sys.stdout.flush()
            total_effect_xy = get_total_effect_dfs(inspection, x, y)
            effects.loc[i] = [x, y, total_effect_xy, abs(total_effect_xy)]
            i += 1
            total_effect_yx = get_total_effect_dfs(inspection, y, x)
            effects.loc[i] = [y, x, total_effect_yx, abs(total_effect_yx)]
            i += 1
        print("\r                                             ", end="")
        print(f"\r{round((i / num_perms) * 100, 1)}% complete... (permutation {i}/{num_perms})", end="")
        sys.stdout.flush()
        print()
        effects = effects.sort_values("abs total effect", ascending=False)
        effects.to_csv(sys.argv[3])
    else:
        independent_var = sys.argv[2]
        dependent_var = sys.argv[3]
        print(get_total_effect_dfs(inspection, independent_var, dependent_var, output_path=sys.argv[4]))