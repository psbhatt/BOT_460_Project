import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playercareerstats
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import LeagueGameLog, CumeStatsTeam, CumeStatsTeamGames
from nba_api.stats.static import teams, players
import stats
import re


def expanding_mean(group):
    return group.expanding().mean()


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    games = LeagueGameLog(season=2023, season_type_all_star='Regular Season').get_data_frames()[0]
    games = games.merge(games, how='inner', on='GAME_ID', suffixes=[None, '_OPP'])
    games = games[games["TEAM_ID"] != games["TEAM_ID_OPP"]].reset_index()
    games.drop(columns=['SEASON_ID', 'VIDEO_AVAILABLE', 'SEASON_ID_OPP', 'VIDEO_AVAILABLE_OPP'], inplace=True)

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

    # for every team, want their stats from every game played up unto that point
    # so will have 82 * 30 = 2460 entry points

    # construct each team average up until that point for each game
    # first want to get every game of every team
    name_cols = ['TEAM_ID_OPP', 'GAME_DATE', 'GAME_DATE_OPP', 'MATCHUP', 'MATCHUP_OPP',
                 'TEAM_ABBREVIATION', 'TEAM_ABBREVIATION_OPP', 'TEAM_NAME',
                 'TEAM_NAME_OPP', 'WL', 'WL_OPP']

    game_stats = games.columns.difference(name_cols)

    grouped = games[game_stats].groupby('TEAM_ID', group_keys=False).apply(expanding_mean).join(games[name_cols]).loc[:,
              games.columns.tolist()]
    print(grouped.loc[0])

    grouped.to_csv("Dataset1.csv")


def addingSeasons():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    szns = [LeagueGameLog(season=i, season_type_all_star='Regular Season').get_data_frames()[0] for i in range(2019, 2024)]
    # curr_szn = LeagueGameLog(season=2023, season_type_all_star='Regular Season').get_data_frames()[0]
    # last_szn = LeagueGameLog(season=2022, season_type_all_star='Regular Season').get_data_frames()[0]
    games = pd.concat(szns, ignore_index=True, axis=0)
    games = games.merge(games, how='inner', on='GAME_ID', suffixes=[None, '_OPP'])
    games = games[games["TEAM_ID"] != games["TEAM_ID_OPP"]].reset_index()
    games.drop(columns=['SEASON_ID', 'VIDEO_AVAILABLE', 'SEASON_ID_OPP', 'VIDEO_AVAILABLE_OPP'], inplace=True)

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

    # for every team, want their stats from every game played up unto that point
    # so will have 82 * 30 = 2460 entry points

    # construct each team average up until that point for each game
    # first want to get every game of every team
    name_cols = ['TEAM_ID_OPP', 'GAME_DATE', 'GAME_DATE_OPP', 'MATCHUP', 'MATCHUP_OPP',
                 'TEAM_ABBREVIATION', 'TEAM_ABBREVIATION_OPP', 'TEAM_NAME',
                 'TEAM_NAME_OPP', 'WL', 'WL_OPP', 'GAME_ID', 'MIN', 'MIN_OPP', 'index']

    game_stats = games.columns.difference(name_cols)

    grouped = games[game_stats].groupby('TEAM_ID', group_keys=False).apply(expanding_mean).join(games[name_cols]).loc[:,
              games.columns.tolist()]
    print(grouped.loc[0])

    grouped.to_csv("19_23.csv")


if __name__ == '__main__':
    # main()
    addingSeasons()
