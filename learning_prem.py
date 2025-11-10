# rnn_season_predictor.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_imports import prem_season_data, LABEL_MAP, REV_LABEL, FEATS, build_match_table
from SeasonRNN import SeasonRNN

matches = build_match_table(prem_season_data)

train_seasons = [s for s in matches["Season"].unique() if s != "24-25"]
val_seasons   = ["24-25"]

train_df = matches[matches["Season"].isin(train_seasons)].copy()
val_df   = matches[matches["Season"].isin(val_seasons)].copy()

teams = sorted(set(train_df["HomeTeam"]).union(set(train_df["AwayTeam"])).union(
               set(val_df["HomeTeam"])).union(set(val_df["AwayTeam"])))
team2id = {t:i for i,t in enumerate(teams)}
num_teams = len(teams)

scaler = StandardScaler()
scaler.fit(train_df[FEATS].to_numpy(dtype=np.float32))

def feats_tensor(df):
    X = scaler.transform(df[FEATS].to_numpy(dtype=np.float32))
    return torch.tensor(X, dtype=torch.float32)



def train_season_rnn(model, train_df, num_teams, hidden_dim=32, lr=1e-3, epochs=5):
    
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        n = 0
        
        H = torch.zeros((num_teams, hidden_dim), dtype=torch.float32)

        for season in sorted(train_df["Season"].unique()):
            season_df = train_df[train_df["Season"] == season]

            for _, r in season_df.iterrows():
                hi = team2id[r["HomeTeam"]]
                ai = team2id[r["AwayTeam"]]

                h_home = H[hi]
                h_away = H[ai]
                x_feat = feats_tensor(pd.DataFrame([r]))[0]  # (feat_dim,)

                logits = model.forward_predict(h_home, h_away, x_feat)
                y = torch.tensor([r["y"]], dtype=torch.long)
                loss = criterion(logits.unsqueeze(0), y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss += float(loss.item()); n += 1

                with torch.no_grad():
                    new_h_home, new_h_away = model.update_states(h_home, h_away, x_feat)
                    H[hi] = new_h_home
                    H[ai] = new_h_away

        avg = total_loss / max(1, n)
        if ep == 1 or ep % 1 == 0:
            print(f"Epoch {ep:02d} | train CE {avg:.4f}")
    return H

def validate_season_rnn(model, val_df, num_teams, hidden_dim=32, init_H=None):
    
    model.eval()
    season = "24-25"
    season_df = val_df[val_df["Season"] == season]

    if init_H is not None:
        H = init_H.clone() 
    else:
        H = torch.zeros((num_teams, hidden_dim), dtype=torch.float32)

    preds, trues = [], []
    rows = []  
    
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

            new_h_home, new_h_away = model.update_states(h_home, h_away, x_feat)
            H[hi] = new_h_home
            H[ai] = new_h_away

    acc = accuracy_score(trues, preds) if preds else 0.0
    print(f"\nValidation 24-25 match accuracy: {acc:.3f}")
    print("Confusion (rows=true, cols=pred) [H,D,A]:")
    print(confusion_matrix(trues, preds, labels=[0,1,2]))
    print(classification_report(trues, preds, target_names=["Home","Draw","Away"], digits=3))

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
    print(summary)

    return per_match, summary


hidden_dim = 32
head_dim   = 32

model = SeasonRNN(feat_dim=len(FEATS), hidden_dim=hidden_dim, head_dim=head_dim, num_classes=3)

print("Training on seasons <= 23-24...")
H_final = train_season_rnn(model, train_df, num_teams=num_teams, hidden_dim=hidden_dim, lr=1e-3, epochs=5)

print("\nValidating on 24-25...")
per_match_preds, per_team_summary = validate_season_rnn(model, val_df, num_teams=num_teams, hidden_dim=hidden_dim, init_H=H_final)
