import pandas as pd
import numpy as np


season_2010_prem = pd.read_csv("Premier League/2010-11.csv")
season_2011_prem = pd.read_csv("Premier League/2011-12.csv")
season_2012_prem = pd.read_csv("Premier League/2012-13.csv")
season_2013_prem = pd.read_csv("Premier League/2013-14.csv")
season_2014_prem = pd.read_csv("Premier League/2014-15.csv")
season_2015_prem = pd.read_csv("Premier League/2015-16.csv")
season_2016_prem = pd.read_csv("Premier League/2016-17.csv")
season_2017_prem = pd.read_csv("Premier League/2017-18.csv")
season_2018_prem = pd.read_csv("Premier League/2018-19.csv")
season_2019_prem = pd.read_csv("Premier League/2019-20.csv")
season_2020_prem = pd.read_csv("Premier League/2020-21.csv")
season_2021_prem = pd.read_csv("Premier League/2021-22.csv")
season_2022_prem = pd.read_csv("Premier League/2022-23.csv")
season_2023_prem = pd.read_csv("Premier League/2023-24.csv")
season_2024_prem = pd.read_csv("Premier League/2024-25.csv")


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


prem_season_data = {}

def load_prem_season(df, year):
    keep = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, keep].copy()

    if "Date" in df.columns:
        df.loc[:, "Date"] = _parse_fd_dates(df["Date"])

    prem_season_data[year] = df


load_prem_season(season_2010_prem, "10-11")
load_prem_season(season_2011_prem, "11-12")
load_prem_season(season_2012_prem, "12-13")
load_prem_season(season_2013_prem, "13-14")
load_prem_season(season_2014_prem, "14-15")
load_prem_season(season_2015_prem, "15-16")
load_prem_season(season_2016_prem, "16-17")
load_prem_season(season_2017_prem, "17-18")
load_prem_season(season_2018_prem, "18-19")
load_prem_season(season_2019_prem, "19-20")
load_prem_season(season_2020_prem, "20-21")
load_prem_season(season_2021_prem, "21-22")
load_prem_season(season_2022_prem, "22-23")
load_prem_season(season_2023_prem, "23-24")
load_prem_season(season_2024_prem, "24-25")
