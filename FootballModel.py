import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def calculate_full_loss(model, criterion, X, y):
    """Compute loss over the entire dataset (no grads)."""
    model.eval()
    with torch.no_grad():
        preds = model(X)
        loss = criterion(preds, y)
    model.train()
    return float(loss.item())

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