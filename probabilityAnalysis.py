import pandas as pd
import numpy as np
import datetime as dt


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

model_prob = pd.read_csv("23_results.csv")
implied_prob = pd.read_csv("historical_odds.csv")
# print(model_prob['Model Win Prob'].shape, implied_prob.shape)

model_prob = model_prob[['TEAM_NAME', 'GAME_DATE', 'Model Win Prob', 'WL']]
implied_prob['Date'] = pd.to_datetime(implied_prob['Date']).dt.date
model_prob['GAME_DATE'] = pd.to_datetime(model_prob['GAME_DATE']).dt.date
# print(implied_prob['Date'], model_prob['GAME_DATE'])

results = pd.merge(implied_prob, model_prob, how='inner', left_on=['Date', 'Team'], right_on=['GAME_DATE', 'TEAM_NAME'])

results["Difference"] = results["Model Win Prob"] - results["Estimated Win Percentage"]

results.sort_values(by="Difference", ascending=False, inplace=True, key=abs)

def calcProfit(row):
    if row["Difference"] > 0:
        if row['WL']:
            if row['Moneyline Odds'] > 0:
                return row["Moneyline Odds"]
            return 10000 / abs(row["Moneyline Odds"])
        return -100

    if row["Difference"] < 0:
        if row['WL']:
            return -100
        if row["Opposing Odds"] > 0:
            return row["Opposing Odds"]
        return 10000 / abs(row["Opposing Odds"])


results["Profit"] = results.apply(calcProfit, axis=1)
print("Total Profit over the 100 games where we most disagreed with the Moneyline Odds: ", results["Profit"][:100].sum())

results.to_csv("results.csv")