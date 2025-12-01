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

    seasons = dataset.matches_df["Season"].unique()
    seasons_sorted = sorted(seasons)
    last_season = seasons_sorted[-1]

    season_col = dataset.matches_df["Season"].values
    val_mask = (season_col == last_season)

    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    print(f"Validation Season: {last_season}")
    print(f"Train matches: {len(train_idx)}, Validation matches: {len(val_idx)}")

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
        y = batch["y"].float()

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
    se_sum = 0.0

    for batch in data_loader:
        batch = to_device(batch, device)
        y = batch["y"].float()

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
        se_sum += ((preds - y) ** 2).sum().item()

        pred_goals = torch.round(preds).long()
        true_goals = y.long()

        exact_score_correct += (pred_goals == true_goals).all(dim=1).sum().item()

        true_diff = true_goals[:, 0] - true_goals[:, 1]
        true_result = torch.sign(true_diff)

        pred_diff = preds[:, 0] - preds[:, 1] 

        draw_margin = 0.55
        pred_result = torch.zeros_like(true_diff)

        pred_result[pred_diff >  draw_margin]  = 1  
        pred_result[pred_diff < -draw_margin] = -1  

        wdl_correct += (pred_result == true_result).sum().item()

    avg_loss = running_loss / n_samples

    rmse = (se_sum / (n_samples * 2)) ** 0.5

    return {
        "loss": avg_loss,
        "rmse": rmse,
    }




dataset = FootballSequenceDataset(prem_season_data, k_form=5, k_h2h=4)
train_loader, val_loader = make_time_split_loaders(dataset, batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hist_feat_dim = 7
meta_numeric_dim = 3

model = FootballScorePredictor(
    num_teams=len(dataset.team_to_id),
    team_id_emb_dim=32,
    hist_feat_dim=hist_feat_dim,
    h2h_hidden_dim=64,
    team_hidden_dim=64,
    meta_numeric_dim=meta_numeric_dim,
    ff_hidden_dim=256,
    dropout=0.4,
    share_team_encoders=True,
).to(device)

criterion = nn.PoissonNLLLoss(log_input=False)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

num_epochs = 25

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_rmse={val_metrics['rmse']:.3f} | "
    )



print_validation_table_predicted(dataset, val_loader, model, device)
print_confusion_matrix(dataset, val_loader, model, device, save_path="confusion_matrix.png")
