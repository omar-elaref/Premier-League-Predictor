import numpy as np
import pandas as pd
import torch
from collections import defaultdict, deque

def _build_all_matches_prem(seasons_dict):
    rows = []
    for sk in sorted(seasons_dict.keys()):
        df = seasons_dict[sk].copy()
        df["Season"] = sk
        rows.append(df)

    all_matches = pd.concat(rows, ignore_index=True)

    all_matches["Date"] = pd.to_datetime(all_matches["Date"])
    all_matches = all_matches.sort_values(["Season", "Date"]).reset_index(drop=True)

    return all_matches


def _game_features_from_perspective(row, team_name):
    res = row["FTR"]

    if row["HomeTeam"] == team_name:
        gf = row["FTHG"]
        ga = row["FTAG"]
        is_home = 1.0

        if res == "H":
            win, draw, lose = 1.0, 0.0, 0.0
        elif res == "D":
            win, draw, lose = 0.0, 1.0, 0.0
        else:  # "A"
            win, draw, lose = 0.0, 0.0, 1.0
    else:
        gf = row["FTAG"]
        ga = row["FTHG"]
        is_home = 0.0

        if res == "A":
            win, draw, lose = 1.0, 0.0, 0.0
        elif res == "D":
            win, draw, lose = 0.0, 1.0, 0.0
        else:  # "H"
            win, draw, lose = 0.0, 0.0, 1.0

    gd = gf - ga
    return np.array([gf, ga, gd, is_home, win, draw, lose], dtype=np.float32)


def build_match_history_tensors(
    seasons_dict,
    k_form: int = 5,
    k_h2h: int = 5,
    odds_cols=("B365H", "B365D", "B365A"),
):
    
    all_matches = _build_all_matches_prem(seasons_dict)

    all_matches = all_matches.dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(drop=True)

    for c in odds_cols:
        if c in all_matches.columns:
            all_matches[c] = pd.to_numeric(all_matches[c], errors="coerce").fillna(0.0)
        else:
            all_matches[c] = 0.0

    # team -> id
    teams = sorted(
        set(all_matches["HomeTeam"].dropna().unique())
        | set(all_matches["AwayTeam"].dropna().unique())
    )
    team_to_id = {t: i for i, t in enumerate(teams)}

    N = len(all_matches)
    hist_dim = 7  

    team1_ids = np.zeros(N, dtype=np.int64)
    team2_ids = np.zeros(N, dtype=np.int64)
    ground_flags = np.zeros(N, dtype=np.int64) 
    meta_numeric = np.zeros((N, len(odds_cols)), dtype=np.float32)
    y_goals = np.zeros((N, 2), dtype=np.float32)

    h2h_seq = np.zeros((N, k_h2h, hist_dim), dtype=np.float32)
    team1_seq = np.zeros((N, k_form, hist_dim), dtype=np.float32)
    team2_seq = np.zeros((N, k_form, hist_dim), dtype=np.float32)

    # histories
    team_hist = defaultdict(lambda: deque(maxlen=k_form))
    pair_hist = defaultdict(lambda: deque(maxlen=k_h2h))

    for idx, row in all_matches.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        team1_ids[idx] = team_to_id[home]
        team2_ids[idx] = team_to_id[away]
        ground_flags[idx] = 0  

        meta_numeric[idx] = np.array([row[c] for c in odds_cols], dtype=np.float32)
        y_goals[idx] = np.array([row["FTHG"], row["FTAG"]], dtype=np.float32)

        hist1 = list(team_hist[home])
        pad1 = k_form - len(hist1)
        if pad1 > 0:
            hist1 = [np.zeros(hist_dim, dtype=np.float32)] * pad1 + hist1
        team1_seq[idx] = np.stack(hist1[-k_form:], axis=0)

        hist2 = list(team_hist[away])
        pad2 = k_form - len(hist2)
        if pad2 > 0:
            hist2 = [np.zeros(hist_dim, dtype=np.float32)] * pad2 + hist2
        team2_seq[idx] = np.stack(hist2[-k_form:], axis=0)

        key_h2h = (home, away)
        hist_h2h = list(pair_hist[key_h2h])
        pad_h = k_h2h - len(hist_h2h)
        if pad_h > 0:
            hist_h2h = [np.zeros(hist_dim, dtype=np.float32)] * pad_h + hist_h2h
        h2h_seq[idx] = np.stack(hist_h2h[-k_h2h:], axis=0)

        feat_home = _game_features_from_perspective(row, home)
        feat_away = _game_features_from_perspective(row, away)

        team_hist[home].append(feat_home)
        team_hist[away].append(feat_away)

        pair_hist[(home, away)].append(feat_home)  
        pair_hist[(away, home)].append(feat_away)  

    tensors = {
        "team1_ids": torch.from_numpy(team1_ids),
        "team2_ids": torch.from_numpy(team2_ids),
        "ground_flags": torch.from_numpy(ground_flags),
        "meta_numeric": torch.from_numpy(meta_numeric),
        "h2h_seq": torch.from_numpy(h2h_seq),
        "team1_seq": torch.from_numpy(team1_seq),
        "team2_seq": torch.from_numpy(team2_seq),
        "y_goals": torch.from_numpy(y_goals),
        "team_to_id": team_to_id,
        "matches_df": all_matches[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].reset_index(drop=True),
    }
    return tensors

