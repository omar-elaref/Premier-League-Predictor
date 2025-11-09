import pandas as pd
import numpy as np

season_2010 = pd.read_csv("laliga/2010-11.csv")
season_2011 = pd.read_csv("laliga/2011-12.csv")
season_2012 = pd.read_csv("laliga/2012-13.csv")
season_2013 = pd.read_csv("laliga/2013-14.csv")
season_2014 = pd.read_csv("laliga/2014-15.csv")
season_2015 = pd.read_csv("laliga/2015-16.csv")
season_2016 = pd.read_csv("laliga/2016-17.csv")
season_2017 = pd.read_csv("laliga/2017-18.csv")
season_2018 = pd.read_csv("laliga/2018-19.csv")
season_2019 = pd.read_csv("laliga/2019-20.csv")
season_2020 = pd.read_csv("laliga/2020-21.csv")
season_2021 = pd.read_csv("laliga/2021-22.csv")
season_2022 = pd.read_csv("laliga/2022-23.csv")
season_2023 = pd.read_csv("laliga/2023-24.csv")
season_2024 = pd.read_csv("laliga/2024-25.csv")

cols_to_keep = [
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"
]


def _parse_fd_dates(s):
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", dayfirst=True, errors="coerce")
    m = d.isna()
    if m.any():
        d.loc[m] = pd.to_datetime(s[m], format="%d/%m/%y", dayfirst=True, errors="coerce")
    return d

laliga_season_data = {}

def load_laliga_season(df, year):
    keep = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, keep].copy()

    if "Date" in df.columns:
        df.loc[:, "Date"] = _parse_fd_dates(df["Date"])

    laliga_season_data[year] = df
    
    
load_laliga_season(season_2010, "10-11")
load_laliga_season(season_2011, "11-12")
load_laliga_season(season_2012, "12-13")
load_laliga_season(season_2013, "13-14")
load_laliga_season(season_2014, "14-15")
load_laliga_season(season_2015, "15-16")
load_laliga_season(season_2016, "16-17")
load_laliga_season(season_2017, "17-18")
load_laliga_season(season_2018, "18-19")
load_laliga_season(season_2019, "19-20")
load_laliga_season(season_2020, "20-21")
load_laliga_season(season_2021, "21-22")
load_laliga_season(season_2022, "22-23")
load_laliga_season(season_2023, "23-24")
load_laliga_season(season_2024, "24-25")

def create_home_row(row, match_num):
    res = row.get("FTR", np.nan)
    win, draw, lose = (np.nan, np.nan, np.nan)
    if pd.notna(res):
        if res == "H": win, draw, lose = 1, 0, 0
        elif res == "D": win, draw, lose = 0, 1, 0
        else: win, draw, lose = 0, 0, 1
        
        goals = row.get("FTHG", np.nan)
        sot = row.get("HST", 0)
        out = {
            "match": match_num,
            "ground": "H",
            "Date": row.get("Date", np.nan),
            "TeamAgainst": row.get("AwayTeam", np.nan),
            "Goals": goals,
            "GoalsConceded": row.get("FTAG", np.nan),
            "HTGoals": row.get("HTHG", np.nan),
            "HTResult": row.get("HTR", np.nan),
            "Shots": row.get("HS", np.nan),
            "ShotsAgainst": row.get("AS", np.nan),
            "ShotsOnTarget": sot,
            "ShotsAgainstOnTarget": row.get("AST", np.nan),
            "Corners": row.get("HC", np.nan),
            "CornersAgainst": row.get("AC", np.nan),
            "FoulsCommitted": row.get("HF", np.nan),
            "FoulsAgainst": row.get("AF", np.nan),
            "YCards": row.get("HY", np.nan),
            "YCardsAgainst": row.get("AY", np.nan),
            "RCards": row.get("HR", np.nan),
            "RCardsAgainst": row.get("AR", np.nan),
            "Win": win, "Draw": draw, "Lose": lose,
            "BigChancesCreated": (0 if pd.isna(goals) else goals) + (0 if pd.isna(sot) else sot)
        }
        
        return pd.DataFrame([out], index=[match_num])
    

def create_away_row(row, match_num):
    res = row.get("FTR", np.nan)
    win, draw, lose = (np.nan, np.nan, np.nan)
    if pd.notna(res):
        if res == "A": win, draw, lose = 1, 0, 0
        elif res == "D": win, draw, lose = 0, 1, 0
        else: win, draw, lose = 0, 0, 1
        
        goals = row.get("FTHG", np.nan)
        sot = row.get("HST", 0)
        out = {
            "match": match_num,
            "ground": "A",
            "Date": row.get("Date", np.nan),
            "TeamAgainst": row.get("AwayTeam", np.nan),
            "Goals": goals,
            "GoalsConceded": row.get("FTAG", np.nan),
            "HTGoals": row.get("HTHG", np.nan),
            "HTResult": row.get("HTR", np.nan),
            "Shots": row.get("HS", np.nan),
            "ShotsAgainst": row.get("AS", np.nan),
            "ShotsOnTarget": sot,
            "ShotsAgainstOnTarget": row.get("AST", np.nan),
            "Corners": row.get("HC", np.nan),
            "CornersAgainst": row.get("AC", np.nan),
            "FoulsCommitted": row.get("HF", np.nan),
            "FoulsAgainst": row.get("AF", np.nan),
            "YCards": row.get("HY", np.nan),
            "YCardsAgainst": row.get("AY", np.nan),
            "RCards": row.get("HR", np.nan),
            "RCardsAgainst": row.get("AR", np.nan),
            "Win": win, "Draw": draw, "Lose": lose,
            "BigChancesCreated": (0 if pd.isna(goals) else goals) + (0 if pd.isna(sot) else sot)
        }
        
        return pd.DataFrame([out], index=[match_num])
    
def build_away_table_for_team(team_name, season_key, season_dict):
    df = season_dict[season_key]
    sub = (df[df["AwayTeam"] == team_name]
           .sort_values("Date")
           .reset_index(drop=True))
    if sub.empty:
        return pd.DataFrame()

    rows = [create_away_row(sub.loc[i], i + 1) for i in range(len(sub))]
    out = pd.concat(rows)
    out.insert(0, "Season", season_key)  # handy label
    return out

def build_away_dict_per_season(season_key, seasons_dict):
    df = seasons_dict[season_key]
    teams = sorted(df["AwayTeam"].dropna().unique())
    return {t: build_away_table_for_team(t, season_key, seasons_dict) for t in teams}

away_games_laliga = {sk: build_away_dict_per_season(sk, laliga_season_data) for sk in laliga_season_data.keys()}


def build_home_table_for_team(team_name, season_key, season_dict):
    df = season_dict[season_key]
    sub = (df[df["HomeTeam"] == team_name]
           .sort_values("Date")
           .reset_index(drop=True))
    if sub.empty:
        return pd.DataFrame()

    rows = [create_home_row(sub.loc[i], i + 1) for i in range(len(sub))]
    out = pd.concat(rows)
    out.insert(0, "Season", season_key)  
    return out

def build_home_dict_per_season(season_key, seasons_dict):
    df = seasons_dict[season_key]
    teams = sorted(df["HomeTeam"].dropna().unique())
    return {t: build_home_table_for_team(t, season_key, seasons_dict) for t in teams}

home_games_laliga = {sk: build_home_dict_per_season(sk, laliga_season_data) for sk in laliga_season_data.keys()}

#print(home_games_laliga["24-25"]["Barcelona"])

def build_team_year_stats(team_name, season_key, seasons_dict):
    home_df = build_home_table_for_team(team_name, season_key, seasons_dict)
    away_df = build_away_table_for_team(team_name, season_key, seasons_dict)
    combined_df = pd.concat([home_df, away_df], ignore_index=True)
    combined = {
        "Wins": int(home_df["Win"].sum() + away_df["Win"].sum()),
        "Draws": int(home_df["Draw"].sum() + away_df["Draw"].sum()),
        "Losses": int(home_df["Lose"].sum() + away_df["Lose"].sum()),
        "Goals": int(home_df["Goals"].sum() + away_df["GoalsConceded"].sum()),
        "GoalsConceded": int(home_df["GoalsConceded"].sum() + away_df["Goals"].sum()),
        "GoalDifference": int((home_df["Goals"].sum() + away_df["GoalsConceded"].sum()) - (home_df["GoalsConceded"].sum() + away_df["Goals"].sum())),
        "GoalsFor": np.mean(combined_df["Goals"]),
        "GoalsAgainst": np.mean(combined_df["GoalsConceded"]),
        "ShotsFor": np.mean(combined_df["Shots"]),
        "ShotsAgainst": np.mean(combined_df["ShotsAgainst"]),
        "ShotsOnTargetFor": np.mean(combined_df["ShotsOnTarget"]),
        "ShotsOnTargetAgainst": np.mean(combined_df["ShotsAgainstOnTarget"]),
        "CornersFor": np.mean(combined_df["Corners"]),
        "CornersAgainst": np.mean(combined_df["CornersAgainst"]),
        "FoulsCommitted": np.mean(combined_df["FoulsCommitted"]),
        "FoulsAgainst": np.mean(combined_df["FoulsAgainst"]),
        "YellowCards": int(home_df["YCards"].sum() + away_df["YCards"].sum()),
        "YellowCardsAgainst": int(home_df["YCardsAgainst"].sum() + away_df["YCardsAgainst"].sum()),
        "RedCards": int(home_df["RCards"].sum() + away_df["RCards"].sum()),
        "RedCardsAgainst": int(home_df["RCardsAgainst"].sum() + away_df["RCardsAgainst"].sum()),
        "BigChancesCreated": np.mean(combined_df["BigChancesCreated"]),
        "Points": int(home_df["Win"].sum() * 3 + home_df["Draw"].sum() + away_df["Win"].sum() * 3 + away_df["Draw"].sum())
    }
    return pd.DataFrame([combined])

# Example usage:
stats = build_team_year_stats("Barcelona", "24-25", laliga_season_data)
#print(stats)


def build_season_team(season_key, season_dict):
    #"""Builds a dictionary of all teams' season stats for one season."""
    season_stat = {}
    df = season_dict[season_key]
    teams = sorted(set(df["HomeTeam"]).union(set(df["AwayTeam"])))

    for team in teams:
        try:
            team_stats = build_team_year_stats(team, season_key, season_dict)
            season_stat[team] = team_stats
        except Exception as e:
            print(f"Error processing {team} in season {season_key}: {e}")
            continue

    return season_stat




# Run the full database build
laliga_database = build_season_team("24-25",laliga_season_data)

combined_stats = []
for team_name, team_df in laliga_database.items():
    team_df_copy = team_df.copy()
    team_df_copy.insert(0, "Team", team_name)
    combined_stats.append(team_df_copy)

all_teams_table = pd.concat(combined_stats, ignore_index=True).sort_values(by="Points", ascending=False)
all_teams_table.insert(1, "Position", range(1, len(all_teams_table) + 1))
#print(all_teams_table)

def prepare_training_data(seasons_dict, target_column="Points"):

    all_seasons_stats = {}
    for season_key in seasons_dict.keys():
        all_seasons_stats[season_key] = build_season_team(season_key, seasons_dict)

    combined_stats = []
    for season_key, season_data in all_seasons_stats.items():
        for team_name, team_df in season_data.items():
            team_df_copy = team_df.copy()
            team_df_copy.insert(0, "Team", team_name)
            team_df_copy.insert(0, "Season", season_key)
            combined_stats.append(team_df_copy)

    all_teams_table = pd.concat(combined_stats, ignore_index=True)

    exclude_cols = ['Team', 'Season', target_column]
    X = all_teams_table.drop(columns=exclude_cols)
    y = all_teams_table[target_column]

    return X, y

X,y = prepare_training_data(laliga_season_data)
#print(X)
#print(y)   
