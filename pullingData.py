import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog
import stats
import re


# method to calculate the expanding mean
def expanding_mean(group):
    return group.expanding().mean()


def main():
    # moving averages of each teams stats of each season
    season_list = []
    for i in range(2019, 2024):
        # pull a season of game data from api
        games = LeagueGameLog(season=i, season_type_all_star='Regular Season').get_data_frames()[0]
        # add the opposing teams stats to each row
        games = games.merge(games, how='inner', on='GAME_ID', suffixes=[None, '_OPP'])
        # remove team A vs team A rows that came as a result of the merge
        games = games[games["TEAM_ID"] != games["TEAM_ID_OPP"]].reset_index()
        # remove useless features
        games.drop(columns=['SEASON_ID', 'VIDEO_AVAILABLE', 'SEASON_ID_OPP', 'VIDEO_AVAILABLE_OPP'], inplace=True)

        # calculate advanced stats for team
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

        # advanced stats for opposing team
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

        # columns we dont want an expanding average for
        name_cols = ['TEAM_ID_OPP', 'GAME_DATE', 'GAME_DATE_OPP', 'MATCHUP', 'MATCHUP_OPP',
                     'TEAM_ABBREVIATION', 'TEAM_ABBREVIATION_OPP', 'TEAM_NAME',
                     'TEAM_NAME_OPP', 'WL', 'WL_OPP', 'GAME_ID', 'MIN', 'MIN_OPP', 'index']
        # all the other columns
        game_stats = games.columns.difference(name_cols)

        # construct each team average up until that point for each game
        grouped = games[game_stats].groupby('TEAM_ID', group_keys=False).apply(expanding_mean).join(
            games[name_cols]).loc[:, games.columns.tolist()]

        # appends this seasons moving average to season_list
        season_list.append(grouped)

    # combine all season to one dataset
    games_updated = pd.concat(season_list, ignore_index=True, axis=0)
    # output to csv
    games_updated.to_csv("data/19_23_updated.csv")


if __name__ == '__main__':
    main()
