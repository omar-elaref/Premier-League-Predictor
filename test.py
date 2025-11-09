# learning.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_imports import prepare_training_data, laliga_season_data  # <- your code

# ----------------------
# 1) Build dataset tables
# ----------------------
# We need the Season + Team columns to split by season, so lightly modify how we call prepare_training_data.
# The function returns X,y but we can reconstruct the full table by calling it and redoing feature selection here:

def build_training_table(seasons_dict, target_column="Points"):
    """
    Returns one DataFrame with Season, Team, all numeric features, and the target column.
    Uses your build_season_team() logic under the hood via prepare_training_data.
    """
    # Rebuild the combined full table with Season/Team retained:
    all_seasons_stats = {}
    for season_key in seasons_dict.keys():
        # This calls your build_season_team under the hood
        pass

    # Use the same logic as in prepare_training_data but keep Season + Team + target
    combined_stats = []
    from data_imports import build_season_team  # import here to avoid circular
    for season_key in seasons_dict.keys():
        season_data = build_season_team(season_key, seasons_dict)
        for team_name, team_df in season_data.items():
            df = team_df.copy()
            df.insert(0, "Team", team_name)
            df.insert(0, "Season", season_key)
            combined_stats.append(df)

    full = pd.concat(combined_stats, ignore_index=True)

    # Feature matrix = all numeric columns except Season, Team, and target
    non_features = {"Season", "Team", target_column}
    feature_cols = [c for c in full.columns if c not in non_features and np.issubdtype(full[c].dtype, np.number)]
    print(full)
    print(feature_cols)
    return full, feature_cols

full, feature_cols = build_training_table(laliga_season_data, target_column="Points")

# Split by season: train <= 23-24, validate == 24-25
train_mask = full["Season"] <= "23-24"
val_mask   = full["Season"] == "24-25"

train_df = full.loc[train_mask].reset_index(drop=True)
val_df   = full.loc[val_mask].reset_index(drop=True)

X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
y_train = train_df["Points"].to_numpy(dtype=np.float32).reshape(-1, 1)

X_val   = val_df[feature_cols].to_numpy(dtype=np.float32)
y_val   = val_df["Points"].to_numpy(dtype=np.float32).reshape(-1, 1)

# ----------------------
# 2) Scaling
# ----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)

# ----------------------
# 3) Torch Dataset/DataLoader
# ----------------------
class TabDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = TabDS(X_train, y_train)
val_ds   = TabDS(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

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