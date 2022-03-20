import semopy
import pandas as pd


data = pd.read_csv("in-the-zone-data/extra-updated-itz-data.csv")
data.set_index("ITZ_GEOID", inplace=True)
print(data)
data.drop(columns=["2002_2010_percent_upzoned_manhattan", "2002_2010_percent_upzoned_non_manhattan"], inplace=True)
data.dropna(inplace=True)
print(data)

print(semopy.efa.explore_cfa_model(data))