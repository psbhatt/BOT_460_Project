import requests
from datetime import datetime, timedelta
import time
import json
import pandas as pd

# Replace with your actual API key
# API_KEY = '0316c24097211ce1cd18e57e3bbf2ef5'
API_KEY = '4bc436933957f838dddbee900dd709da'
SPORT = 'basketball_nba'
MARKET = 'h2h'  # moneyline odds
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

odds_url = (
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'
)

params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': MARKET,
    'oddsFormat': ODDS_FORMAT,
    'dateFormat': DATE_FORMAT
}

response = requests.get(odds_url, params=params)
if response.status_code != 200:
    print(f'Failed to fetch odds. Status: {response.status_code}, Msg: {response.text}')
    exit()

odds_data = response.json()
events = []

for event in odds_data:
    home_team = event.get('home_team')
    away_team = event.get('away_team')
    commence_time = event.get('commence_time')
    for bookmaker in event.get('bookmakers', []):
        if bookmaker['key'] == 'draftkings':  # Change or remove to include other books
            for market in bookmaker.get('markets', []):
                if market['key'] == MARKET:
                    for outcome in market.get('outcomes', []):
                        team = outcome['name']
                        price = outcome['price']
                        events.append({
                            'Date': commence_time,
                            'Matchup': f"{home_team} vs {away_team}",
                            'Bookmaker': bookmaker['key'],
                            'Team': team,
                            'Moneyline Odds': price
                        })

with open('nba_odds_today.json', 'w') as outfile:
    json.dump(events, outfile)

print(f"Saved odds for {len(events)} lines to nba_odds_today.json")

# Read JSON file containing historical odds data
with open('nba_odds_today.json', 'r') as infile:
    odds_data = json.load(infile)

# Prepare lists to hold extracted data
events = []
for event in odds_data:
    date = event['Date']
    matchup = event['Matchup']
    bookmaker = event['Bookmaker']
    team = event['Team']
    price = event['Moneyline Odds']
    events.append([date, matchup, bookmaker, team, price])

# Create a DataFrame from the events list
columns = ['Date', 'Matchup', 'Bookmaker', 'Team', 'Moneyline Odds']
df = pd.DataFrame(events, columns=columns)

# Output DataFrame to a CSV file
csv_filename = 'data/todays_odds.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")


# define method to convert given odds to an implied win probability
def converter(line):
    if line < 0:
        return -line / (-line + 100)
    return 100 / (line + 100)


# import odds pulled from odds_api
odds = pd.read_csv("data/todays_odds.csv")

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
odds.to_csv("data/todays_odds.csv")