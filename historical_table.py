import pandas as pd
import json

# Read JSON file containing historical odds data
with open('historical_odds.json', 'r') as infile:
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
csv_filename = 'historical_odds.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")
