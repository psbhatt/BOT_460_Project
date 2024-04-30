import pandas as pd


# define method to convert given odds to an implied win probability
def converter(line):
    if line < 0:
        return -line / (-line + 100)
    return 100 / (line + 100)


# import odds pulled from odds_api
odds = pd.read_csv("data/historical_odds.csv")

# group games together
groups = odds.groupby(["Date", "Matchup"])

opposing_odds_dict = {}

# iterate through each group and collect the opposing teams odds
for group_id, group in groups:
    team1 = group.iloc[0]["Team"]
    team2 = group.iloc[1]["Team"]
    odds1 = group.loc[group["Team"] != team1, "Moneyline Odds"].values[0]
    odds2 = group.loc[group["Team"] != team2, "Moneyline Odds"].values[0]
    opposing_odds_dict[group_id] = {team1: odds1, team2: odds2}

# add opposing team's odds to the df
odds["Opposing Odds"] = [opposing_odds_dict[(date, matchup)][team] for date, matchup, team in zip(odds["Date"], odds["Matchup"], odds["Team"])]

# apply converter to the moneyline odds
odds["Estimated Win Percentage"] = odds["Moneyline Odds"].apply(converter)

# export updated dataset to the original csv
odds.to_csv("data/historical_odds.csv")