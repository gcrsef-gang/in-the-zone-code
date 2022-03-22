import sys
import os

import pandas as pd

def filter_model_description(model_path, output_path, threshold=0.2):
    model_inspection = pd.read_csv(os.path.join(model_path, "model_inspection.csv"))
    filtered_model_inspection = []
    for _, row in model_inspection.iterrows():
        if row["lval"] == "2010_2018_percent_upzoned":
            filtered_model_inspection.append(row["lval"]+" "+row["op"]+" "+row["rval"]+"\n")
        
        if row["rval"] in ["d_2010_2018_pop_density", "d_2010_2018_resid_unit_density"]:
            if row["lval"] == row["rval"]:
                continue
            filtered_model_inspection.append(row["lval"]+" "+row["op"]+" "+row["rval"]+"\n")
        elif row["p-value"] < float(threshold):
        # if row["p-value"] < float(threshold):
            if row["lval"] == row["rval"]:
                continue
            filtered_model_inspection.append(row["lval"]+" "+row["op"]+" "+row["rval"]+"\n")
        print(_, len(filtered_model_inspection))
    with open(output_path, "w+") as f:
        f.writelines(filtered_model_inspection)

if __name__ == "__main__":
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        threshold = sys.argv[3]
        filter_model_description(model_path, output_path, threshold)
    else:
        filter_model_description(model_path, output_path)