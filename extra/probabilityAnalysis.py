import pandas as pd
pd.set_option('display.max_columns', None)

# load nba data with model results
model_prob = pd.read_csv("data/25_results.csv")
# load odds with implied probabilities
implied_prob = pd.read_csv("data/todays_odds.csv")

# remove unneeded columns from nba data
model_prob = model_prob[['TEAM_NAME', 'GAME_DATE', 'Model Win Prob', 'WL']]

# convert dates to datetime objects for comparison
implied_prob['Date'] = pd.to_datetime(implied_prob['Date']).dt.date
model_prob['GAME_DATE'] = pd.to_datetime(model_prob['GAME_DATE']).dt.date
print(implied_prob)
print(model_prob)

# merge odds and model results together based on team name and date of game
results = pd.merge(implied_prob, model_prob, how='inner', left_on=['Date', 'Team'], right_on=['GAME_DATE', 'TEAM_NAME'])
print(results)

# calculate difference between model win probability and implied win prob from odds
results["Difference"] = results["Model Win Prob"] - results["Estimated Win Percentage"]

# sort by greatest differences
results.sort_values(by="Difference", ascending=False, inplace=True, key=abs)

print(results)

# method to calculate expected profit based on appropriate bet from odds comparison
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


# apply profit calculation to data
results["Profit"] = results.apply(calcProfit, axis=1)

# print results
print("Total Profit over the 100 games where we most disagreed with the Moneyline Odds: ", results["Profit"][:100].sum())
print("Total Profit over the games with difference > 20%: ", results["Profit"][results['Difference'] > 0.2].sum())

# output to a final csv
results.to_csv("data/results.csv")
