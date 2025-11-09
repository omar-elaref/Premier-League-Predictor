
import platform
import psutil
import getpass

# --- Setup: Imports ---
import os, seaborn, sklearn, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

from data_imports import (
    home_games_laliga,
    away_games_laliga,
    build_team_year_stats, #Function
    build_season_team, #Function
    prepare_training_data #Function, 
    laliga_season_data #Dictionary
)

class FootballModel(nn.Module):

    def __init__(self, input_size, home_game_layer, away_game_layer, output_size):
        super(FootballModel, self).__init__()
        self.home_game_layer = nn.Linear(input_size, home_game_layer)
        self.away_game_layer = nn.Linear(home_game_layer, away_game_layer)

    def forward(self, x):
        x = torch.relu(self.home_game_layer(x))
        x = torch.relu(self.away_game_layer(x))
        return x

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):


    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model = None

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
            val_losses.append(val_loss.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

        train_losses.append(loss.item())

    return train_losses, val_losses, best_model

X, y = prepare_training_data(laliga_season_data)
model = FootballModel(input_size=X.shape[1], home_game_layer=32, away_game_layer=32, output_size=32)
train_losses, val_losses, best_model = train_model(model, X, y, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001)
print(train_losses)
print(val_losses)
print(best_model)