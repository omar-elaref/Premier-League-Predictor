# rnn_season_predictor.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_imports import laliga_season_data, LABEL_MAP, REV_LABEL, FEATS, build_match_table
#from SeasonRNN import SeasonRNN
from EncoderRnn import *

matches = build_match_table(laliga_season_data)



import numpy as np
import pandas as pd
import torch
from collections import defaultdict, deque

# --- helper: build one big, chronologically sorted match table ---

def _build_all_matches_prem(seasons_dict):
    """
    seasons_dict: e.g. prem_season_data from importing_files.py
    Returns a single DataFrame with columns:
      Season, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A, ...
    """
    rows = []
    for sk in sorted(seasons_dict.keys()):
        df = seasons_dict[sk].copy()
        df["Season"] = sk
        rows.append(df)

    all_matches = pd.concat(rows, ignore_index=True)

    # ensure date is datetime and sort
    all_matches["Date"] = pd.to_datetime(all_matches["Date"])
    all_matches = all_matches.sort_values(["Season", "Date"]).reset_index(drop=True)

    return all_matches


def _game_features_from_perspective(row, team_name):
    """
    Convert one match row into a feature vector from the perspective of `team_name`.
    Feature vector (dim = 7):
        [gf, ga, goal_diff, is_home, win, draw, lose]
    """
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
    """
    Main dataset builder for your architecture.

    Returns a dict of torch tensors:
        team1_ids:   (N,)
        team2_ids:   (N,)
        ground_flags:(N,)  # 0 = team1 is home (always in this setup)
        meta_numeric:(N, len(odds_cols))
        h2h_seq:     (N, k_h2h, hist_dim)
        team1_seq:   (N, k_form, hist_dim)
        team2_seq:   (N, k_form, hist_dim)
        y_goals:     (N, 2)  # [FTHG, FTAG]
        team_to_id:  dict mapping team name -> int
        matches_df:  info DataFrame for debugging
    """
    all_matches = _build_all_matches_prem(seasons_dict)

    # keep only rows with known final score
    all_matches = all_matches.dropna(subset=["FTHG", "FTAG", "FTR"])

    # make sure odds exist numerically (fill NaN with 0.0 so you still keep the match)
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
    hist_dim = 7  # as defined in _game_features_from_perspective

    team1_ids = np.zeros(N, dtype=np.int64)
    team2_ids = np.zeros(N, dtype=np.int64)
    ground_flags = np.zeros(N, dtype=np.int64)  # 0 = team1 is home

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

        # --- metadata ---
        team1_ids[idx] = team_to_id[home]
        team2_ids[idx] = team_to_id[away]
        ground_flags[idx] = 0  # team1 is always home in this representation

        meta_numeric[idx] = np.array([row[c] for c in odds_cols], dtype=np.float32)
        y_goals[idx] = np.array([row["FTHG"], row["FTAG"]], dtype=np.float32)

        # --- sequences BEFORE updating with this match (no leakage) ---

        # Enc2: team1 recent form
        hist1 = list(team_hist[home])
        # left-pad with zeros so oldest history is first, newest last
        pad1 = k_form - len(hist1)
        if pad1 > 0:
            hist1 = [np.zeros(hist_dim, dtype=np.float32)] * pad1 + hist1
        team1_seq[idx] = np.stack(hist1[-k_form:], axis=0)

        # Enc3: team2 recent form
        hist2 = list(team_hist[away])
        pad2 = k_form - len(hist2)
        if pad2 > 0:
            hist2 = [np.zeros(hist_dim, dtype=np.float32)] * pad2 + hist2
        team2_seq[idx] = np.stack(hist2[-k_form:], axis=0)

        # Enc1: head-to-head history (from home/team1 perspective)
        key_h2h = (home, away)
        hist_h2h = list(pair_hist[key_h2h])
        pad_h = k_h2h - len(hist_h2h)
        if pad_h > 0:
            hist_h2h = [np.zeros(hist_dim, dtype=np.float32)] * pad_h + hist_h2h
        h2h_seq[idx] = np.stack(hist_h2h[-k_h2h:], axis=0)

        # --- update histories WITH this match ---

        feat_home = _game_features_from_perspective(row, home)
        feat_away = _game_features_from_perspective(row, away)

        team_hist[home].append(feat_home)
        team_hist[away].append(feat_away)

        # store both directions so next time (away, home) is team1/team2 we also have history
        pair_hist[(home, away)].append(feat_home)  # features from home perspective
        pair_hist[(away, home)].append(feat_away)  # features from away perspective

    # convert everything to torch tensors
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









class MatchHistoryEncoder(nn.Module):
    """
    Generic GRU encoder for a sequence of past matches.

    Input:  x of shape (batch, seq_len, feat_dim)
    Output: h of shape (batch, hidden_dim)
    """
    def __init__(self, feat_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (B, T, F)
        _, h_n = self.gru(x)  # h_n: (num_layers * num_directions, B, H)
        # take last layer
        h_last = h_n[-1]      # (B, H * directions)
        return h_last


class FootballScorePredictor(nn.Module):
    """
    Your architecture:

      metadata ----\
      Enc1 (H2H) ---+--> concat --> FF --> [goals_team1, goals_team2]
      Enc2 (team1) -+
      Enc3 (team2) -+

    """
    def __init__(
        self,
        num_teams: int,
        team_id_emb_dim: int,
        hist_feat_dim: int,
        h2h_hidden_dim: int,
        team_hidden_dim: int,
        meta_numeric_dim: int,   # e.g. 3 for B365H/B365D/B365A + maybe 1 for home/away flag
        ff_hidden_dim: int = 128,
        dropout: float = 0.2,
        share_team_encoders: bool = True,
    ):
        super().__init__()

        # --- Team ID embeddings ---
        self.team_emb = nn.Embedding(num_teams, team_id_emb_dim)

        # Ground (home/away) as tiny embedding (2 values: 0=home,1=away) if you want
        self.ground_emb = nn.Embedding(2, 4)

        # --- Encoders ---
        # Enc1: head-to-head history
        self.h2h_encoder = MatchHistoryEncoder(hist_feat_dim, h2h_hidden_dim)

        # Enc2 and Enc3: form of each team
        self.share_team_encoders = share_team_encoders
        self.team_encoder1 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)
        if share_team_encoders:
            self.team_encoder2 = self.team_encoder1
        else:
            self.team_encoder2 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)

        # --- Metadata projection ---
        # metadata = [team1_emb, team2_emb, ground_emb, numeric_meta]
        meta_in_dim = 2 * team_id_emb_dim + 4 + meta_numeric_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Final FF combining everything ---
        total_in = self.h2h_encoder.output_dim + 2 * self.team_encoder1.output_dim + ff_hidden_dim
        self.ff = nn.Sequential(
            nn.Linear(total_in, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim // 2, 2),  # [goals_team1, goals_team2]
        )

    def forward(
        self,
        team1_ids,          # (B,)
        team2_ids,          # (B,)
        ground_flags,       # (B,) 0=team1_home, 1=team2_home or similar
        meta_numeric,       # (B, meta_numeric_dim) e.g. [B365H,B365D,B365A]
        h2h_seq,            # (B, T_h2h, hist_feat_dim)
        team1_seq,          # (B, T_team, hist_feat_dim)
        team2_seq           # (B, T_team, hist_feat_dim)
    ):
        # --- Embeddings for metadata ---
        t1_emb = self.team_emb(team1_ids)       # (B, team_id_emb_dim)
        t2_emb = self.team_emb(team2_ids)       # (B, team_id_emb_dim)
        g_emb = self.ground_emb(ground_flags)   # (B, 4)

        meta = torch.cat([t1_emb, t2_emb, g_emb, meta_numeric], dim=-1)  # (B, meta_in_dim)
        meta_repr = self.meta_mlp(meta)  # (B, ff_hidden_dim)

        # --- Encoders ---
        h_h2h = self.h2h_encoder(h2h_seq)           # (B, h2h_hidden_dim[*2])
        h_t1  = self.team_encoder1(team1_seq)       # (B, team_hidden_dim[*2])
        h_t2  = self.team_encoder2(team2_seq)       # (B, team_hidden_dim[*2])

        # --- Combine ---
        combined = torch.cat([meta_repr, h_h2h, h_t1, h_t2], dim=-1)
        goals = self.ff(combined)   # (B, 2)

        # Optional: enforce non-negativity with softplus
        goals = F.softplus(goals)

        return goals  # predicted [goals_team1, goals_team2]


from torch.utils.data import Dataset

class FootballSequenceDataset(Dataset):
    def __init__(self, seasons_dict, k_form=5, k_h2h=5, odds_cols=("B365H", "B365D", "B365A")):
        data = build_match_history_tensors(
            seasons_dict,
            k_form=k_form,
            k_h2h=k_h2h,
            odds_cols=odds_cols,
        )
        self.team1_ids = data["team1_ids"]
        self.team2_ids = data["team2_ids"]
        self.ground_flags = data["ground_flags"]
        self.meta_numeric = data["meta_numeric"]
        self.h2h_seq = data["h2h_seq"]
        self.team1_seq = data["team1_seq"]
        self.team2_seq = data["team2_seq"]
        self.y_goals = data["y_goals"]

        self.team_to_id = data["team_to_id"]
        self.matches_df = data["matches_df"]  # optional, for debugging

    def __len__(self):
        return self.y_goals.shape[0]

    def __getitem__(self, idx):
        return {
            "team1_ids": self.team1_ids[idx],
            "team2_ids": self.team2_ids[idx],
            "ground_flags": self.ground_flags[idx],
            "meta_numeric": self.meta_numeric[idx],
            "h2h_seq": self.h2h_seq[idx],
            "team1_seq": self.team1_seq[idx],
            "team2_seq": self.team2_seq[idx],
            "y": self.y_goals[idx],
        }






# dataset already built somewhere above:
# dataset = FootballSequenceDataset(prem_season_data, k_form=5, k_h2h=5)

def make_time_split_loaders(dataset, val_ratio=0.2, batch_size=64):
    """
    Splits the dataset chronologically: first part = train, last part = val.
    """
    N = len(dataset)
    split_idx = int(N * (1 - val_ratio))

    train_idx = np.arange(0, split_idx)
    val_idx   = np.arange(split_idx, N)

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in train_loader:
        batch = to_device(batch, device)
        y = batch["y"]                    # (B, 2) true goals

        preds = model(
            batch["team1_ids"],
            batch["team2_ids"],
            batch["ground_flags"],
            batch["meta_numeric"],
            batch["h2h_seq"],
            batch["team1_seq"],
            batch["team2_seq"],
        )

        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = y.size(0)
        running_loss += loss.item() * bsz
        n_samples += bsz

    avg_loss = running_loss / n_samples
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    exact_score_correct = 0
    wdl_correct = 0

    for batch in data_loader:
        batch = to_device(batch, device)
        y = batch["y"]                    # (B, 2)

        preds = model(
            batch["team1_ids"],
            batch["team2_ids"],
            batch["ground_flags"],
            batch["meta_numeric"],
            batch["h2h_seq"],
            batch["team1_seq"],
            batch["team2_seq"],
        )

        loss = criterion(preds, y)

        bsz = y.size(0)
        running_loss += loss.item() * bsz
        n_samples += bsz

        # ---- metrics ----
        # round goals to nearest int for accuracy metrics
        pred_goals = torch.round(preds).long()   # (B, 2)
        true_goals = y.long()

        # exact scoreline (e.g. 2-1 exactly)
        exact_score_correct += (pred_goals == true_goals).all(dim=1).sum().item()

        # W/D/L accuracy
        pred_diff = pred_goals[:, 0] - pred_goals[:, 1]
        true_diff = true_goals[:, 0] - true_goals[:, 1]

        pred_result = torch.sign(pred_diff)  # -1,0,1
        true_result = torch.sign(true_diff)

        wdl_correct += (pred_result == true_result).sum().item()

    avg_loss = running_loss / n_samples

    # RMSE over both goals
    # (we recompute quickly here)
    # Collecting all preds+y in memory is nicer, but this is fine for now
    # If you want exact RMSE, you can accumulate squared error instead.
    rmse = avg_loss ** 0.5   # since we used MSELoss

    exact_score_acc = exact_score_correct / n_samples
    wdl_acc = wdl_correct / n_samples

    return {
        "loss": avg_loss,
        "rmse": rmse,
        "exact_score_acc": exact_score_acc,
        "wdl_acc": wdl_acc,
    }


from importing_files import prem_season_data
# === build dataset and loaders ===
dataset = FootballSequenceDataset(prem_season_data, k_form=5, k_h2h=5)
train_loader, val_loader = make_time_split_loaders(dataset, val_ratio=0.2, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === build model ===
hist_feat_dim = 7        # we defined 7 features in _game_features_from_perspective
meta_numeric_dim = 3     # B365H, B365D, B365A

model = FootballScorePredictor(
    num_teams=len(dataset.team_to_id),
    team_id_emb_dim=16,
    hist_feat_dim=hist_feat_dim,
    h2h_hidden_dim=32,
    team_hidden_dim=32,
    meta_numeric_dim=meta_numeric_dim,
    ff_hidden_dim=128,
    dropout=0.3,
    share_team_encoders=True,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# === training loop ===
num_epochs = 30

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_rmse={val_metrics['rmse']:.3f} | "
        f"val_exact={val_metrics['exact_score_acc']*100:.2f}% | "
        f"val_wdl={val_metrics['wdl_acc']*100:.2f}%"
    )
