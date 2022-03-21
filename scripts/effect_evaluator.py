import os
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
        

if __name__ == "__main__":
    model_path = sys.argv[1]
    data = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    dependent_var = sys.argv[2]
    independent_var = sys.argv[3]
    effect_aggregator(dependent_var, independent_var, data)