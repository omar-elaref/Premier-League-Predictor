# rnn_season_predictor.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- bring your LaLiga season dict
from data_imports import laliga_season_data  # keys like "10-11", ..., "24-25"

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
REV_LABEL = {v: k for k, v in LABEL_MAP.items()}
FEATS = ["B365H", "B365D", "B365A"]  # leakage-safe, pre-match odds only

# ----------------------------
# Data: build a match-level table
# ----------------------------
def build_match_table(seasons_dict):
    rows = []
    for sk in sorted(seasons_dict.keys()):
        df = seasons_dict[sk].copy()
        # keep only what we need
        need = ["Date", "HomeTeam", "AwayTeam", "FTR"] + FEATS
        have = [c for c in need if c in df.columns]
        df = df.loc[:, have].copy()

        # numeric features
        for c in FEATS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # clean rows
        df = df.dropna(subset=FEATS + ["FTR", "HomeTeam", "AwayTeam", "Date"])
        df = df[df["FTR"].isin(LABEL_MAP)]

        df["y"] = df["FTR"].map(LABEL_MAP)
        df["Season"] = sk
        rows.append(df)

    all_matches = pd.concat(rows, ignore_index=True)
    # sort inside each season by date to respect chronology
    all_matches["Date"] = pd.to_datetime(all_matches["Date"])
    all_matches = all_matches.sort_values(["Season", "Date"]).reset_index(drop=True)
    return all_matches

matches = build_match_table(laliga_season_data)

# split
train_seasons = [s for s in matches["Season"].unique() if s != "24-25"]
val_seasons   = ["24-25"]

train_df = matches[matches["Season"].isin(train_seasons)].copy()
val_df   = matches[matches["Season"].isin(val_seasons)].copy()

# team index map
teams = sorted(set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"])).union(
               set(val_df["HomeTeam"])).union(set(val_df["AwayTeam"])))
team2id = {t:i for i,t in enumerate(teams)}
num_teams = len(teams)

# scaler fit on train only
scaler = StandardScaler()
scaler.fit(train_df[FEATS].to_numpy(dtype=np.float32))

def feats_tensor(df):
    X = scaler.transform(df[FEATS].to_numpy(dtype=np.float32))
    # We'll also pass a small role signal in the GRU update (home=+1, away=-1) inside the model
    return torch.tensor(X, dtype=torch.float32)

# ----------------------------
# Model: two GRUCells + prediction head
# ----------------------------
class SeasonRNN(nn.Module):
    def __init__(self, feat_dim=3, hidden_dim=32, head_dim=32, num_classes=3):
        super().__init__()
        # encode match features once
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.ReLU(),
        )
        # separate GRUCells for home/away updates (small role-specific dynamics)
        self.gru_home = nn.GRUCell(input_size=feat_dim+1, hidden_size=hidden_dim)  # +1 for role signal
        self.gru_away = nn.GRUCell(input_size=feat_dim-0+1, hidden_size=hidden_dim)

        # prediction head uses both team states + encoded features
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2 + head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes)
        )

    def forward_predict(self, h_home, h_away, x_feat):
        """
        Predict outcome logits given both team states and match features.
        h_home, h_away: (hidden_dim,) each
        x_feat: (feat_dim,)
        """
        xh = self.enc(x_feat)            # (head_dim,)
        z  = torch.cat([h_home, h_away, xh], dim=-1).unsqueeze(0)  # (1, 2H+head_dim)
        logits = self.head(z).squeeze(0) # (3,)
        return logits

    def update_states(self, h_home, h_away, x_feat):
        """
        Update each team's hidden state after this match.
        Use a simple role scalar: home=+1.0, away=-1.0 appended to features.
        """
        role_home = torch.tensor([+1.0], dtype=torch.float32)
        role_away = torch.tensor([-1.0], dtype=torch.float32)
        xh = torch.cat([x_feat, role_home], dim=-1).unsqueeze(0)  # (1, feat_dim+1)
        xa = torch.cat([x_feat, role_away], dim=-1).unsqueeze(0)

        new_h_home = self.gru_home(xh, h_home.unsqueeze(0)).squeeze(0)
        new_h_away = self.gru_away(xa, h_away.unsqueeze(0)).squeeze(0)
        return new_h_home, new_h_away

# ----------------------------
# Training / Validation Loops
# ----------------------------
def train_season_rnn(model, train_df, num_teams, hidden_dim=32, lr=1e-3, epochs=5):
    """
    Chronological, season-by-season training.
    - Reset hidden states per season to zeros.
    - For each match: predict from latest states; compute loss; update states; step optimizer.
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        n = 0

        for season in sorted(train_df["Season"].unique()):
            season_df = train_df[train_df["Season"] == season]
            # hidden state per team for this season
            H = torch.zeros((num_teams, hidden_dim), dtype=torch.float32)

            for _, r in season_df.iterrows():
                hi = team2id[r["HomeTeam"]]
                ai = team2id[r["AwayTeam"]]

                h_home = H[hi]
                h_away = H[ai]
                x_feat = feats_tensor(pd.DataFrame([r]))[0]  # (feat_dim,)

                # predict
                logits = model.forward_predict(h_home, h_away, x_feat)
                y = torch.tensor([r["y"]], dtype=torch.long)
                loss = criterion(logits.unsqueeze(0), y)

                # step
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss += float(loss.item()); n += 1

                # update team states after the match
                with torch.no_grad():
                    new_h_home, new_h_away = model.update_states(h_home, h_away, x_feat)
                    H[hi] = new_h_home
                    H[ai] = new_h_away

        avg = total_loss / max(1, n)
        if ep == 1 or ep % 1 == 0:
            print(f"Epoch {ep:02d} | train CE {avg:.4f}")

def validate_season_rnn(model, val_df, num_teams, hidden_dim=32):
    """
    Validate on 24-25:
    - Reset hidden states for the season
    - Predict each match
    - Aggregate per-team W/D/L from predictions
    """
    model.eval()
    season = "24-25"
    season_df = val_df[val_df["Season"] == season]

    H = torch.zeros((num_teams, hidden_dim), dtype=torch.float32)

    preds, trues = [], []
    rows = []  # per-match predictions with teams & date

    with torch.no_grad():
        for _, r in season_df.iterrows():
            hi = team2id[r["HomeTeam"]]
            ai = team2id[r["AwayTeam"]]
            h_home = H[hi]
            h_away = H[ai]
            x_feat = feats_tensor(pd.DataFrame([r]))[0]

            logits = model.forward_predict(h_home, h_away, x_feat)
            pred = int(torch.argmax(logits).item())
            true = int(r["y"])

            preds.append(pred); trues.append(true)

            rows.append({
                "Date": r["Date"], "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
                "pred": REV_LABEL[pred], "true": REV_LABEL[true]
            })

            # update states after match
            new_h_home, new_h_away = model.update_states(h_home, h_away, x_feat)
            H[hi] = new_h_home
            H[ai] = new_h_away

    # basic accuracy
    acc = accuracy_score(trues, preds) if preds else 0.0
    print(f"\nValidation 24-25 match accuracy: {acc:.3f}")
    print("Confusion (rows=true, cols=pred) [H,D,A]:")
    print(confusion_matrix(trues, preds, labels=[0,1,2]))
    print(classification_report(trues, preds, target_names=["Home","Draw","Away"], digits=3))

    # Aggregate per-team W/D/L from predictions
    per_match = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    agg = []
    for _, m in per_match.iterrows():
        h, a, pr = m["HomeTeam"], m["AwayTeam"], m["pred"]
        if pr == "H":
            agg.append((h,"W")); agg.append((a,"L"))
        elif pr == "A":
            agg.append((a,"W")); agg.append((h,"L"))
        else:
            agg.append((h,"D")); agg.append((a,"D"))
    per_team = pd.DataFrame(agg, columns=["Team","Res"])
    summary = (per_team.pivot_table(index="Team", columns="Res", aggfunc="size", fill_value=0)
                        .reindex(columns=["W","D","L"], fill_value=0)
                        .reset_index()
                        .sort_values(["W","D","L"], ascending=[False,False,True])
                        .reset_index(drop=True))
    print("\nPredicted 24-25 summary (by W/D/L):")
    print(summary.head(10))

    return per_match, summary

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    hidden_dim = 32
    head_dim   = 32

    model = SeasonRNN(feat_dim=len(FEATS), hidden_dim=hidden_dim, head_dim=head_dim, num_classes=3)

    print("Training on seasons <= 23-24...")
    train_season_rnn(model, train_df, num_teams=num_teams, hidden_dim=hidden_dim, lr=1e-3, epochs=5)

    print("\nValidating on 24-25...")
    per_match_preds, per_team_summary = validate_season_rnn(model, val_df, num_teams=num_teams, hidden_dim=hidden_dim)
