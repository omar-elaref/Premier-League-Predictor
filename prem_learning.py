import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict, deque
from build_dataset import *
from architecture import *
from importing_files import prem_season_data
from torch.utils.data import Dataset, DataLoader, Subset
from print_results import *




def make_time_split_loaders(dataset, batch_size=64):
    
    # Extract all seasons present in the dataset
    seasons = dataset.matches_df["Season"].unique()
    seasons_sorted = sorted(seasons)      # ensures correct season order
    last_season = seasons_sorted[-1]      # choose the final season

    # Build boolean mask for validation rows
    season_col = dataset.matches_df["Season"].values
    val_mask = (season_col == last_season)

    # Indices for train and val
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    print(f"Validation Season: {last_season}")
    print(f"Train matches: {len(train_idx)}, Validation matches: {len(val_idx)}")

    # Build subsets
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    # Dataloaders
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

# training and evaluation functions
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

        # metrics 
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




dataset = FootballSequenceDataset(prem_season_data, k_form=5, k_h2h=5)
train_loader, val_loader = make_time_split_loaders(dataset, batch_size=64)


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



print_validation_table_predicted(dataset, val_loader, model, device)
