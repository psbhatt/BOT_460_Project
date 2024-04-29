import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

model_prob = pd.read_csv("23_results.csv")
implied_prob = pd.read_csv("historical_odds.csv")
# print(model_prob['Model Win Prob'].shape, implied_prob.shape)


implied_prob["Model Prob"] = model_prob['Model Win Prob']
implied_prob["Result"] = model_prob['WL']

implied_prob["Difference"] = implied_prob["Model Prob"] - implied_prob["Estimated Win Percentage"]

implied_prob.sort_values(by="Difference", ascending=False, inplace=True, key=abs)


def checkResult(row):
    return (row["Difference"] > 0) == (row['Result'] == 1)

implied_prob["Would have profitted"] = implied_prob.apply(checkResult, axis=1)

print(implied_prob[:100]["Would have profitted"].value_counts())
