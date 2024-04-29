import requests
from datetime import datetime, timedelta
import time
import json

# Replace with your actual API key
API_KEY = '0316c24097211ce1cd18e57e3bbf2ef5'
SPORT = 'basketball_nba'
MARKET = 'h2h'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# Define the start and end dates
start_date = datetime(2023, 10, 24)
end_date = datetime(2024, 4, 14)

# Initialize start date
current_date = start_date

# List to hold extracted data
events = []

while current_date <= end_date:
    # Convert current date to the desired format
    current_date_str = current_date.strftime('%Y-%m-%dT12:00:00Z')

    # Make API request for the current date
    response = requests.get(
        f'https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds',
        params={
            'api_key': API_KEY,
            'regions': 'us',
            'markets': MARKET,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
            'date': current_date_str,
        }
    )

    if response.status_code == 200:
        odds_data = response.json()
        # Check if there is a 'X-RateLimit-Remaining' header
        remaining_requests = response.headers.get('X-RateLimit-Remaining')
        if remaining_requests is not None:
            print(f'Remaining requests: {remaining_requests}')
        # Process the odds data
        for event in odds_data['data']:
            if (datetime.strptime(event['commence_time'], "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=4, minutes=0)).date() == current_date.date():
                    for bookmaker in event['bookmakers']:
                        if bookmaker['key'] == 'draftkings':
                            for market in bookmaker['markets']:
                                if market['key'] == MARKET:
                                    for outcome in market['outcomes']:
                                        home_team = event['home_team']
                                        away_team = event['away_team']
                                        team = outcome['name']
                                        price = outcome['price']
                                        events.append({
                                            'Date': current_date_str,
                                            'Matchup': f"{home_team} vs {away_team}",
                                            'Bookmaker': bookmaker['key'],
                                            'Team': team,
                                            'Moneyline Odds': price
                                        })

    else:
        print(f'Failed to fetch data for {current_date_str}')

    # Move to the next day
    current_date += timedelta(days=1)

    # Add a small delay to comply with rate limits
    time.sleep(1)

# Save extracted data to a JSON file
with open('historical_odds.json', 'w') as outfile:
    json.dump(events, outfile)

print("Data saved to historical_odds.json")
