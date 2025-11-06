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


