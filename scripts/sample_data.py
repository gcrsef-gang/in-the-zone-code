import random
import pandas as pd

data = pd.read_csv("in-the-zone-data/integrated-itz-data.csv")
data.set_index("ITZ_GEOID", inplace=True)

sampled_data = []

for i in range(100):
    print(i)
    numbers = random.choices(data.index.tolist(), k=100)
    average_row = []
    for x in data.columns:
        average_row.append([])
        for number in numbers:
            average_row[-1].append(data.loc[number, x])
    new_average_row = []
    for stuff in average_row:
        new_average_row.append(sum(stuff)/100)

    sampled_data.append(new_average_row)

sampled_data = pd.DataFrame(sampled_data, columns=data.columns)

sampled_data.to_csv("in-the-zone-data/sampled-itz-data.csv")