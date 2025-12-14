import requests
from datetime import datetime, timedelta
import time
import json

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