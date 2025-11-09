# learning.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_imports import prepare_training_data, laliga_season_data  # <- your code

# ----------------------
# 1) Build dataset table (simple version)
# ----------------------
def build_training_table(seasons_dict, target_column="Points"):
    """
    Create one DataFrame with Season, Team, numeric feature columns, and the target.
    - Pulls per-team season tables via your build_season_team(...)
    - Adds Season/Team columns
    - Concatenates into a single DataFrame
    - Returns (full_df, feature_cols)
    """
    from data_imports import build_season_team  # local import to avoid circulars

    rows = []
    # Iterate seasons in a stable order
    for season_key in sorted(seasons_dict.keys()):
        season_dict = build_season_team(season_key, seasons_dict)  # { team_name: team_df }
        for team_name, team_df in season_dict.items():
            df = team_df.copy()
            df.insert(0, "Team", team_name)      # add 'Team' as first col
            df.insert(0, "Season", season_key)   # add 'Season' before 'Team'
            rows.append(df)

    # Stack everything together
    full = pd.concat(rows, ignore_index=True)

    # Pick numeric columns, then drop Season/Team/target to get the feature list
    numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_column]

    return full, feature_cols

full, feature_cols = build_training_table(laliga_season_data, target_column="Points")

# ----------------------
# 2) Train/Val split by season (simple, explicit)
# ----------------------
# Train on all seasons up to 23-24 => i.e., everything EXCEPT 24-25
train_seasons = [sk for sk in full["Season"].unique() if sk != "24-25"]
val_seasons   = ["24-25"]

train_df = full[full["Season"].isin(train_seasons)].reset_index(drop=True)
val_df   = full[full["Season"].isin(val_seasons)].reset_index(drop=True)

# Build matrices (float32 for PyTorch)
X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
y_train = train_df["Points"].to_numpy(dtype=np.float32).reshape(-1, 1)

X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val   = val_df["Points"].to_numpy(dtype=np.float32).reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)

# ----------------------
# 3) Torch Dataset/DataLoader
# ----------------------
Xtr = torch.from_numpy(X_train)
ytr = torch.from_numpy(y_train)
Xva = torch.from_numpy(X_val)
yva = torch.from_numpy(y_val)

train_ds = TensorDataset(Xtr, ytr)
val_ds   = TensorDataset(Xva, yva)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
# ----------------------
# 4) Model (simple MLP)
# ----------------------
class FootballModel(nn.Module):
    def __init__(self, input_size, h1=64, h2=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim)   # final output layer (regression -> 1)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FootballModel(input_size=len(feature_cols), h1=64, h2=32, out_dim=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 200

# ----------------------
# 5) Train/eval loops
# ----------------------
def run_epoch(loader, train=True):
    model.train(mode=train)
    total_loss = 0.0
    yhats = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        yhats.append(pred.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    N = len(loader.dataset)
    avg_loss = total_loss / N if N else 0.0
    yhats = np.vstack(yhats) if yhats else np.zeros((0,1))
    ys = np.vstack(ys) if ys else np.zeros((0,1))
    return avg_loss, yhats, ys

best_state = None
best_val = float("inf")

for ep in range(1, epochs+1):
    tr_loss, _, _ = run_epoch(train_loader, train=True)
    va_loss, _, _ = run_epoch(val_loader, train=False)
    if va_loss < best_val:
        best_val = va_loss
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if ep % 20 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | train MSE {tr_loss:.3f} | val MSE {va_loss:.3f}")

# load best
if best_state is not None:
    model.load_state_dict(best_state)

# ----------------------
# 6) Final validation metrics + per-team table
# ----------------------
_, yhat_val, ytrue_val = run_epoch(val_loader, train=False)

# Flatten to 1-D for sklearn
y_true = ytrue_val.ravel()
y_pred = yhat_val.ravel()

# Compute metrics (no 'squared' kwarg)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"\n24-25 regression metrics -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R^2: {r2:.3f}")

val_out = val_df[["Season","Team"]].copy()
val_out["Points_true"] = y_true
val_out["Points_pred"] = y_pred
val_out["Error"] = val_out["Points_pred"] - val_out["Points_true"]
val_out = val_out.sort_values("Points_pred", ascending=False).reset_index(drop=True)

print("\nPredicted 24-25 table (by points prediction):")
print(val_out[["Team","Points_pred","Points_true","Error"]].head(10))