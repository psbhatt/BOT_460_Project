def eff_fg_pct(data, group=''):
    return (data[group + 'FGM'] + 0.5 * data[group + 'FG3M']) / data[group + 'FGA']


def eff_fg_pct_OPP(data, group=''):
    return (data[group + 'FGM_OPP'] + 0.5 * data[group + 'FG3M_OPP']) / data[group + 'FGA_OPP']


def ft_per_fga(data, group=''):
    return data[group + 'FTM'] / data[group + 'FGA']


def ft_per_fga_OPP(data, group=''):
    return data[group + 'FTM_OPP'] / data[group + 'FGA_OPP']


def tov_pct(data, group=''):
    return data[group + 'TOV'] / (
            data[group + 'FGA'] + 0.44 * data[group + 'FTA'] + data[group + 'TOV']
    )


def tov_pct_OPP(data, group=''):
    return data[group + 'TOV_OPP'] / (
            data[group + 'FGA_OPP'] + 0.44 * data[group + 'FTA_OPP'] + data[group + 'TOV_OPP']
    )


def blk_pct(data):
    return data.BLK / (data.FGA_OPP - data.FG3A_OPP)


def blk_pct_OPP(data):
    return data.BLK_OPP / (data.FGA - data.FG3A)


def def_rating(data):
    return 100 * data.PTS_OPP / possessions(data)


def def_rating_OPP(data):
    return 100 * data.PTS / possessions_OPP(data)


def dreb_pct(data):
    return data.DREB / (data.DREB + data.OREB_OPP)


def dreb_pct_OPP(data):
    return data.DREB_OPP / (data.DREB_OPP + data.OREB)


def off_rating(data):
    return 100 * data.PTS / possessions(data)


def off_rating_OPP(data):
    return 100 * data.PTS_OPP / possessions_OPP(data)


def oreb_pct(data):
    return data.OREB / (data.OREB + data.DREB_OPP)


def oreb_pct_OPP(data):
    return data.OREB_OPP / (data.OREB_OPP + data.DREB)


def pace(data):
    min_per_game = 240
    min_played = data.MIN

    if min_played[0] < min_per_game:
        min_played *= 5

    return possessions(data) / min_played * min_per_game


def pace_OPP(data):
    min_per_game = 240
    min_played = data.MIN_OPP

    if min_played[0] < min_per_game:
        min_played *= 5

    return possessions(data) / min_played * min_per_game


def possessions(data):
    return (
                   data.FGA
                   + 0.4 * data.FTA
                   + data.TOV
                   - 1.07
                   * (data.OREB / (data.OREB + data.DREB_OPP))
                   * (data.FGA - data.FGM)
                   + data.FGA_OPP
                   + 0.4 * data.FTA_OPP
                   + data.TOV_OPP
                   - 1.07
                   * (data.OREB_OPP / (data.OREB_OPP + data.DREB))
                   * (data.FGA_OPP - data.FGM_OPP)
           ) / 2


def possessions_OPP(data):
    return (
                   data.FGA_OPP
                   + 0.4 * data.FTA_OPP
                   + data.TOV_OPP
                   - 1.07
                   * (data.OREB_OPP / (data.OREB_OPP + data.DREB))
                   * (data.FGA_OPP - data.FGM_OPP)
                   + data.FGA
                   + 0.4 * data.FTA
                   + data.TOV
                   - 1.07
                   * (data.OREB / (data.OREB + data.DREB_OPP))
                   * (data.FGA - data.FGM)
           ) / 2
