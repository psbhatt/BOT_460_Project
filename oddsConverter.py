import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


odds = pd.read_csv("historical_odds.csv")


def converter(line):
    if line < 0:
        return -line / (-line + 100)
    return 100 / (line + 100)


print(odds.columns)
odds["Estimated Win Percentage"] = odds["Moneyline Odds"].apply(converter)
print(odds.loc[:5])
odds.to_csv("historical_odds.csv")
