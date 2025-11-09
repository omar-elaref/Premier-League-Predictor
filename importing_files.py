import pandas as pd
import numpy as np


season_2010_laliga = pd.read_csv("laliga/2010-11.csv")
season_2011_laliga = pd.read_csv("laliga/2011-12.csv")
season_2012_laliga = pd.read_csv("laliga/2012-13.csv")
season_2013_laliga = pd.read_csv("laliga/2013-14.csv")
season_2014_laliga = pd.read_csv("laliga/2014-15.csv")
season_2015_laliga = pd.read_csv("laliga/2015-16.csv")
season_2016_laliga = pd.read_csv("laliga/2016-17.csv")
season_2017_laliga = pd.read_csv("laliga/2017-18.csv")
season_2018_laliga = pd.read_csv("laliga/2018-19.csv")
season_2019_laliga = pd.read_csv("laliga/2019-20.csv")
season_2020_laliga = pd.read_csv("laliga/2020-21.csv")
season_2021_laliga = pd.read_csv("laliga/2021-22.csv")
season_2022_laliga = pd.read_csv("laliga/2022-23.csv")
season_2023_laliga = pd.read_csv("laliga/2023-24.csv")
season_2024_laliga = pd.read_csv("laliga/2024-25.csv")



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


season_2010_seriea = pd.read_csv("seriea/2010-11.csv")
season_2011_seriea = pd.read_csv("seriea/2011-12.csv")
season_2012_seriea = pd.read_csv("seriea/2012-13.csv")
season_2013_seriea = pd.read_csv("seriea/2013-14.csv")
season_2014_seriea = pd.read_csv("seriea/2014-15.csv")
season_2015_seriea = pd.read_csv("seriea/2015-16.csv")
season_2016_seriea = pd.read_csv("seriea/2016-17.csv")
season_2017_seriea = pd.read_csv("seriea/2017-18.csv")
season_2018_seriea = pd.read_csv("seriea/2018-19.csv")
season_2019_seriea = pd.read_csv("seriea/2019-20.csv")
season_2020_seriea = pd.read_csv("seriea/2020-21.csv")
season_2021_seriea = pd.read_csv("seriea/2021-22.csv")
season_2022_seriea = pd.read_csv("seriea/2022-23.csv")
season_2023_seriea = pd.read_csv("seriea/2023-24.csv")
season_2024_seriea = pd.read_csv("seriea/2024-25.csv")




season_2010_bundesliga = pd.read_csv("bundesliga/2010-11.csv")
season_2011_bundesliga = pd.read_csv("bundesliga/2011-12.csv")
season_2012_bundesliga = pd.read_csv("bundesliga/2012-13.csv")
season_2013_bundesliga = pd.read_csv("bundesliga/2013-14.csv")
season_2014_bundesliga = pd.read_csv("bundesliga/2014-15.csv")
season_2015_bundesliga = pd.read_csv("bundesliga/2015-16.csv")
season_2016_bundesliga = pd.read_csv("bundesliga/2016-17.csv")
season_2017_bundesliga = pd.read_csv("bundesliga/2017-18.csv")
season_2018_bundesliga = pd.read_csv("bundesliga/2018-19.csv")
season_2019_bundesliga = pd.read_csv("bundesliga/2019-20.csv")
season_2020_bundesliga = pd.read_csv("bundesliga/2020-21.csv")
season_2021_bundesliga = pd.read_csv("bundesliga/2021-22.csv")
season_2022_bundesliga = pd.read_csv("bundesliga/2022-23.csv")
season_2023_bundesliga = pd.read_csv("bundesliga/2023-24.csv")
season_2024_bundesliga = pd.read_csv("bundesliga/2024-25.csv")

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
    
seriea_season_data = {}

def load_seriea_season(df, year):
    keep = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, keep].copy()

    if "Date" in df.columns:
        df.loc[:, "Date"] = _parse_fd_dates(df["Date"])

    seriea_season_data[year] = df
    
bundesliga_season_data = {}

def load_bundesliga_season(df, year):
    keep = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, keep].copy()

    if "Date" in df.columns:
        df.loc[:, "Date"] = _parse_fd_dates(df["Date"])

    bundesliga_season_data[year] = df

prem_season_data = {}

def load_prem_season(df, year):
    keep = [c for c in cols_to_keep if c in df.columns]
    df = df.loc[:, keep].copy()

    if "Date" in df.columns:
        df.loc[:, "Date"] = _parse_fd_dates(df["Date"])

    prem_season_data[year] = df

load_laliga_season(season_2010_laliga, "10-11")
load_laliga_season(season_2011_laliga, "11-12")
load_laliga_season(season_2012_laliga, "12-13")
load_laliga_season(season_2013_laliga, "13-14")
load_laliga_season(season_2014_laliga, "14-15")
load_laliga_season(season_2015_laliga, "15-16")
load_laliga_season(season_2016_laliga, "16-17")
load_laliga_season(season_2017_laliga, "17-18")
load_laliga_season(season_2018_laliga, "18-19")
load_laliga_season(season_2019_laliga, "19-20")
load_laliga_season(season_2020_laliga, "20-21")
load_laliga_season(season_2021_laliga, "21-22")
load_laliga_season(season_2022_laliga, "22-23")
load_laliga_season(season_2023_laliga, "23-24")
load_laliga_season(season_2024_laliga, "24-25")

load_seriea_season(season_2010_seriea, "10-11")
load_seriea_season(season_2011_seriea, "11-12")
load_seriea_season(season_2012_seriea, "12-13")
load_seriea_season(season_2013_seriea, "13-14")
load_seriea_season(season_2014_seriea, "14-15")
load_seriea_season(season_2015_seriea, "15-16")
load_seriea_season(season_2016_seriea, "16-17")
load_seriea_season(season_2017_seriea, "17-18")
load_seriea_season(season_2018_seriea, "18-19")
load_seriea_season(season_2019_seriea, "19-20")
load_seriea_season(season_2020_seriea, "20-21")
load_seriea_season(season_2021_seriea, "21-22")
load_seriea_season(season_2022_seriea, "22-23")
load_seriea_season(season_2023_seriea, "23-24")
load_seriea_season(season_2024_seriea, "24-25")

load_bundesliga_season(season_2010_bundesliga, "10-11")
load_bundesliga_season(season_2011_bundesliga, "11-12")
load_bundesliga_season(season_2012_bundesliga, "12-13")
load_bundesliga_season(season_2013_bundesliga, "13-14")
load_bundesliga_season(season_2014_bundesliga, "14-15")
load_bundesliga_season(season_2015_bundesliga, "15-16")
load_bundesliga_season(season_2016_bundesliga, "16-17")
load_bundesliga_season(season_2017_bundesliga, "17-18")
load_bundesliga_season(season_2018_bundesliga, "18-19")
load_bundesliga_season(season_2019_bundesliga, "19-20")
load_bundesliga_season(season_2020_bundesliga, "20-21")
load_bundesliga_season(season_2021_bundesliga, "21-22")
load_bundesliga_season(season_2022_bundesliga, "22-23")
load_bundesliga_season(season_2023_bundesliga, "23-24")
load_bundesliga_season(season_2024_bundesliga, "24-25")

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
