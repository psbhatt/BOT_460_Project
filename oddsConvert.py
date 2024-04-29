import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


odds = pd.read_csv("historical_odds.csv")


def converter(line):
    if line < 0:
        return -line / (-line + 100)
    return 100 / (line + 100)


groups = odds.groupby(["Date", "Matchup"])

opposing_odds_dict = {}

# Iterate through each group and calculate opposing team's odds
for group_id, group in groups:
    team1 = group.iloc[0]["Team"]
    team2 = group.iloc[1]["Team"]
    odds1 = group.loc[group["Team"] != team1, "Moneyline Odds"].values[0]
    odds2 = group.loc[group["Team"] != team2, "Moneyline Odds"].values[0]
    opposing_odds_dict[group_id] = {team1: odds1, team2: odds2}

# Add opposing team's odds to the DataFrame
odds["Opposing Odds"] = [opposing_odds_dict[(date, matchup)][team] for date, matchup, team in zip(odds["Date"], odds["Matchup"], odds["Team"])]

print(odds.columns)
odds["Estimated Win Percentage"] = odds["Moneyline Odds"].apply(converter)
print(odds.loc[:5])


odds.to_csv("historical_odds.csv")