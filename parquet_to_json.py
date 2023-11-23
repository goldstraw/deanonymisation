import pandas as pd
import json

path = "./train-00000-of-00001-cced8514c7ed782a.parquet"
data = pd.read_parquet(path)
df = pd.DataFrame(data)
json_data = []
for i in range(len(df)):
    json_data.append(df["conversation_b"][i][0])
    json_data.append(df["conversation_b"][i][1])

with open("train.json", "w") as f:
    json.dump(json_data, f)