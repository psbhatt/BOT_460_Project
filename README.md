CS 460 Final Project
Praneeth Bhatt, Luke Olsen, Nathan Turlington

To reproduce results here's an explanation of what each Python script does, in the order they should be run:

1. pullingData.py: pulls data from the NBA API and saves it into 19_23_updated.csv. (This uses the methods in stats.csv, you don't need to run that file)
2. model.py: Takes data from 19_23_updated.csv and returns model predictions for the test 23 season in 23_results.csv
3. historical_odds.py: **DO NOT RUN** This file creates a JSON of odds data from odds_api, but since we paid for the API key and there is a limit to the number of requests we can make, please do not run this file. 
4. historical_table.py: **DO NOT RUN** Converts the odds JSON to historical_odds.csv
5. oddsConvert.py: takes historical_odds.csv and modifies it so every row has the odds for both teams in the game, converts the odds of one team to an implied win probability, then saves back to the same csv
6. probabilityAnalysis.py: takes both 23_results.csv and historical_odds.csv. This merges those two datasets based on team and date and then computes the difference between the model's estimated win probability and the implied win probability. It then tests what the profits would have been had we bet on different sets of games based on the difference.


The file named CS_460_Project__Copy_ is a report of the project.
