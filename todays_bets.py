import pandas as pd
pd.set_option('display.max_columns', None)

# Load NBA data (model results) and odds (with implied probabilities)
model_prob = pd.read_csv("data/upcoming_game_predictions.csv")
implied_prob = pd.read_csv("data/todays_odds.csv")

# Keep only relevant columns from model predictions
model_prob = model_prob[['TEAM_NAME', 'GAME_DATE', 'Model Win Prob']]

# Convert date columns for merging
implied_prob['Date'] = pd.to_datetime(implied_prob['Date']).dt.date
model_prob['GAME_DATE'] = pd.to_datetime(model_prob['GAME_DATE']).dt.date

# Merge on team and date
results = pd.merge(implied_prob, model_prob, how='inner', left_on=['Date', 'Team'], right_on=['GAME_DATE', 'TEAM_NAME'])
# Difference between model prediction and book implied win prob
results["Difference"] = results["Model Win Prob"] - results["Estimated Win Percentage"]
print(results)
results.to_csv("data/edges_today.csv", index=False)

# Only keep strong edges (absolute > 20%)
edge = results[results["Difference"].abs() > 0.20].copy()

# Decide which bet is indicated by the model's edge
# If the model is more optimistic than the odds: bet on that team, otherwise the edge is on the opponent
def get_edge(row):
    # Model says to bet on the team if it likes them more than the book by at least 20%
    if row["Difference"] > 0.20:
        return f"BET ON {row['Team']} at {row['Moneyline Odds']}"
    elif row["Difference"] < -0.20:
        # The model prefers the opponent; must also get their odds (if available in your data)
        # Here, Opposing Odds assumed to exist 
        return f"BET AGAINST {row['Team']} at {row['Opposing Odds']}"
    return "No major edge"

edge["Suggested Bet"] = edge.apply(get_edge, axis=1)

# Show most actionable edges first
edge = edge.sort_values(by="Difference", ascending=False, key=abs)

# Print and save
print(edge[['Date', 'Matchup', 'Team', 'Model Win Prob', 'Estimated Win Percentage', 'Moneyline Odds', 'Difference', 'Suggested Bet']])
