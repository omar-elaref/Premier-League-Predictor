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


def calculate_full_loss(model, criterion, X, y):
    """Compute loss over the entire dataset (no grads)."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        loss = criterion(preds, y)
    model.train()
    return float(loss.item())

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


def train_with_minibatch(model, criterion, optimizer,
                         X_train, y_train, X_val, y_val,
                         num_iterations, batch_size, check_every,
                         shuffle_each_epoch=True):
    """
    Minibatch trainer with simple modulo batch selection.
    - No DataLoader; works on full tensors directly.
    - Optionally reshuffles at the start of each 'epoch' (i.e., when batches wrap).
    - Logs full-dataset train/val loss every `check_every` iterations.
    """
    model.train()

    train_losses, val_losses, iterations = [], [], []

    # Initial logged losses at iteration 0
    train_losses.append(calculate_full_loss(model, criterion, X_train, y_train))
    val_losses.append(calculate_full_loss(model, criterion, X_val, y_val))
    iterations.append(0)

    n_train = X_train.shape[0]
    batch_size = int(min(max(1, batch_size), n_train))
    num_batches = (n_train + batch_size - 1) // batch_size  # ceil division

    # Index order (shuffled per epoch if requested)
    order = torch.arange(n_train)

    for it in range(1, num_iterations + 1):
        # If we wrapped around (new epoch), optionally reshuffle
        if shuffle_each_epoch and ((it - 1) % num_batches == 0):
            order = order[torch.randperm(n_train)]

        batch_id = (it - 1) % num_batches
        start = batch_id * batch_size
        end = min(start + batch_size, n_train)

        idx = order[start:end]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(xb)
        loss = criterion(y_hat, yb)
        loss.backward()
        optimizer.step()

        if it % check_every == 0 or it == num_iterations:
            tr_loss = calculate_full_loss(model, criterion, X_train, y_train)
            va_loss = calculate_full_loss(model, criterion, X_val,   y_val)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            iterations.append(it)

    return train_losses, val_losses, iterations, model

model = FootballModel(input_size=len(feature_cols), h1=64, h2=32, out_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

NUM_ITERS   = 5000
BATCH_SIZE  = 32
CHECK_EVERY = 100

train_losses, val_losses, iters, model = train_with_minibatch(
    model, criterion, optimizer,
    X_train_t, y_train_t, X_val_t, y_val_t,
    num_iterations=NUM_ITERS,
    batch_size=BATCH_SIZE,
    check_every=CHECK_EVERY,
    shuffle_each_epoch=True,   # set False to make it fully deterministic
)

# ----------------------
# 6) Final validation metrics + per-team table
# ----------------------
def predict_full(model, X):
    """Return model predictions for the entire tensor X as a 1-D NumPy array."""
    model.eval()
    with torch.no_grad():
        yhat = model(X)
    model.train()
    return yhat.detach().cpu().numpy().ravel()

# 1) Get predictions and flatten targets
y_pred = predict_full(model, X_val_t)               # shape -> (N,)
y_true = y_val_t.detach().cpu().numpy().ravel()     # shape -> (N,)

# 2) Metrics (version-proof RMSE)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print(f"\n24-25 regression metrics -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R^2: {r2:.3f}")

# 3) Predicted table for 24-25 (by points prediction)
val_out = val_df[["Season", "Team"]].copy()
val_out["Points_true"] = y_true
val_out["Points_pred"] = y_pred
val_out["Error"] = val_out["Points_pred"] - val_out["Points_true"]
val_out = val_out.sort_values("Points_pred", ascending=False).reset_index(drop=True)

print("\nPredicted 24-25 table (by points prediction):")
print(val_out[["Team","Points_pred","Points_true","Error"]].head(10))