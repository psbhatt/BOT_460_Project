import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
import stats
import re

# method to calculate the expanding mean
def expanding_mean(group):
    return group.shift(1).expanding().mean()

def main():
    # Step 1: Pull historic data and calculate season stats
    season_list = []
    for i in range(2019, 2019):
        games = LeagueGameLog(season=i, season_type_all_star='Regular Season').get_data_frames()[0]
        games = games.merge(games, how='inner', on='GAME_ID', suffixes=[None, '_OPP'])
        games = games[games["TEAM_ID"] != games["TEAM_ID_OPP"]].reset_index(drop=True)

        # games = games[games["MATCHUP"].str.contains('vs')].reset_index(drop=True)


        games.drop(columns=['SEASON_ID', 'VIDEO_AVAILABLE', 'SEASON_ID_OPP', 'VIDEO_AVAILABLE_OPP', 'WL_OPP'], inplace=True)

        # Advanced stats for team
        games['PACE'] = stats.pace(games)
        games['POSSESSIONS'] = stats.possessions(games)
        games['OFF_RTG'] = stats.off_rating(games)
        games['DEF_RTG'] = stats.def_rating(games)
        games['NET_RTG'] = games['OFF_RTG'] - games['DEF_RTG']
        games['EFG'] = stats.eff_fg_pct(games)
        games['TOV_PCT'] = stats.tov_pct(games)
        games['OREB_PCT'] = stats.oreb_pct(games)
        games['DREB_PCT'] = stats.dreb_pct(games)
        games['FT_PER_FGA'] = stats.ft_per_fga(games)
       

        # Advanced stats for opposing team
        games['PACE_OPP'] = stats.pace_OPP(games)
        games['POSSESSIONS_OPP'] = stats.possessions_OPP(games)
        games['OFF_RTG_OPP'] = stats.off_rating_OPP(games)
        games['DEF_RTG_OPP'] = stats.def_rating_OPP(games)
        games['NET_RTG_OPP'] = games['OFF_RTG_OPP'] - games['DEF_RTG_OPP']
        games['EFG_OPP'] = stats.eff_fg_pct_OPP(games)
        games['TOV_PCT_OPP'] = stats.tov_pct_OPP(games)
        games['OREB_PCT_OPP'] = stats.oreb_pct_OPP(games)
        games['DREB_PCT_OPP'] = stats.dreb_pct_OPP(games)
        games['FT_PER_FGA_OPP'] = stats.ft_per_fga_OPP(games)
        games['HOME'] = games['MATCHUP'].apply(lambda x: 0 if re.search(r'@', str(x)) else 1)

        # name_cols is the list of metadata columns that we dont want to apply expanding means on
        name_cols =  ['GAME_DATE', 'GAME_DATE_OPP', 'MATCHUP', 'MATCHUP_OPP',
             'TEAM_ABBREVIATION', 'TEAM_ABBREVIATION_OPP', 'TEAM_NAME',
             'TEAM_NAME_OPP', 'WL', 'GAME_ID', 'MIN', 'MIN_OPP', 'HOME', 'TEAM_ID', 'TEAM_ID_OPP']
        
        # game_stats are the remaining columns, or stats, that we wish to apply expanding mean on
        game_stats = games.columns.difference(name_cols)

        # get all the previous games for both the home team and the away team, and then calculate their average stats up until the current matchup
        grouped_home = games.groupby('TEAM_ID', group_keys=False)[[col for col in game_stats if not col.endswith('_OPP')]].apply(expanding_mean)
        grouped_opp = games.groupby('TEAM_ID_OPP', group_keys=False)[[col for col in game_stats if col.endswith('_OPP')]].apply(expanding_mean)
        
        # create a new row for this matchup with the averaged stats and add to our set for the season
        grouped = grouped_home.join(grouped_opp).join(games[name_cols]).loc[:, games.columns.tolist()]        
        season_list.append(grouped)

    historic_df = pd.concat(season_list, ignore_index=True)

    # Step 2: Load odds with upcoming games info
    odds_df = pd.read_csv("data/todays_odds.csv")  # Columns: Date, Matchup, Bookmaker, Team, Moneyline Odds, Opposing Odds, Estimated Win Percentage
    odds_df['Date'] = pd.to_datetime(odds_df['Date']).dt.date

    # Step 3: Build new rows for upcoming games with rolling averages for those teams
    future_rows = []
    max_game_date = historic_df['GAME_DATE'].max()
    # print(teams_group.groups.keys())

    for _, row in odds_df.iterrows():
        
        matchup = row['Matchup']
        game_date = row['Date']

        pattern = r"(.+?)\s+vs\s+(.+)"
        match = re.match(pattern, matchup)
        if match:
            home_team = match.group(1)
            away_team = match.group(2)

        home = row['Team'] == home_team

        if home_team == 'Los Angeles Clippers':
            home_team = 'LA Clippers'

        if away_team == 'Los Angeles Clippers':
            away_team = 'LA Clippers'

        home_games = historic_df.groupby('TEAM_NAME').get_group(home_team)
        away_games = historic_df.groupby('TEAM_NAME_OPP').get_group(away_team)
        home_last_game = home_games[home_games['GAME_DATE'] <= max_game_date].sort_values('GAME_DATE').iloc[-1].copy()
        away_last_game = away_games[away_games['GAME_DATE'] <= max_game_date].sort_values('GAME_DATE').iloc[-1].copy()


        name_cols =  ['GAME_DATE', 'GAME_DATE_OPP', 'MATCHUP', 'MATCHUP_OPP',
            'TEAM_ABBREVIATION', 'TEAM_ABBREVIATION_OPP', 'TEAM_NAME',
            'TEAM_NAME_OPP', 'WL', 'GAME_ID', 'MIN', 'MIN_OPP', 'HOME']
        game_cols = games.columns.difference(name_cols)

        new_row = {
            'GAME_DATE': game_date,
            'GAME_DATE_OPP': game_date,
            'MATCHUP': matchup,
            'MATCHUP_OPP': matchup,
            'HOME': home
        }

        if home:
            game_stats_home = home_last_game[[col for col in game_cols if not col.endswith('_OPP')]]
            game_stats_opp = away_last_game[[col for col in game_cols if col.endswith('_OPP')]]

            new_row.update({
                'GAME_ID': home_last_game['GAME_ID'],
                'TEAM_NAME': home_team,
                'TEAM_NAME_OPP': away_team,
                'TEAM_ABBREVIATION': home_last_game['TEAM_ABBREVIATION'],
                'TEAM_ABBREVIATION_OPP': away_last_game['TEAM_ABBREVIATION_OPP'],
                'MIN': home_last_game['MIN'],
                'MIN_OPP': away_last_game['MIN_OPP'],
                **game_stats_home,
                **game_stats_opp,
            })
        else:
            game_stats_home = home_last_game[[col for col in game_cols if col.endswith('_OPP')]]
            game_stats_opp = away_last_game[[col for col in game_cols if not col.endswith('_OPP')]]

            new_row.update({
                'GAME_ID': away_last_game['GAME_ID'],
                'TEAM_NAME': away_team,
                'TEAM_NAME_OPP': home_team,
                'TEAM_ABBREVIATION': away_last_game['TEAM_ABBREVIATION_OPP'],
                'TEAM_ABBREVIATION_OPP': home_last_game['TEAM_ABBREVIATION'],
                'MIN': away_last_game['MIN'],
                'MIN_OPP': home_last_game['MIN_OPP'],
                **game_stats_home,
                **game_stats_opp,
            })

        future_rows.append(new_row)

    upcoming_df = pd.DataFrame(future_rows)

    # Step 4: Append upcoming rows to historic dataframe
    full_df = pd.concat([historic_df, upcoming_df], ignore_index=True)

    # Optionally save or return full dataset
    full_df.to_csv("data/19_25_with_upcoming.csv", index=False)
    print(f"Appended {len(upcoming_df)} upcoming game rows to historic data. Saved to data/19_25_with_upcoming.csv")

if __name__ == '__main__':
    main()