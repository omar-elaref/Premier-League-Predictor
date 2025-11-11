import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""
Defining the SeasonRNN model, which is a GRU based RNN model that takes in the features of a match and the hidden states of the home and away teams and outputs the logits for the match outcome.
"""
class SeasonRNN(nn.Module):
    def __init__(self, feat_dim=3, hidden_dim=32, head_dim=32, num_classes=3):
        super().__init__()
        # Encoding the match features once
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.ReLU(),
        )
        # Split the GRU into home and away GRU cells
        self.gru_home = nn.GRUCell(input_size=feat_dim+1, hidden_size=hidden_dim) 
        self.gru_away = nn.GRUCell(input_size=feat_dim-0+1, hidden_size=hidden_dim)

        # Prediction head is the final layer of the model that outputs the logits for the match outcome
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2 + head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, num_classes)
        )

    """
    Forward pass through the model to predict the match outcome.
    """
    def forward_predict(self, h_home, h_away, x_feat):
        encoded_feature = self.enc(x_feat)            
        z  = torch.cat([h_home, h_away, encoded_feature], dim=-1).unsqueeze(0)  
        logits_result = self.head(z).squeeze(0) 
        return logits_result

    """
    Updating the hidden states for the home and away teams.
    """
    def update_states(self, h_home, h_away, x_feat):
        role_home = torch.tensor([+1.0], dtype=torch.float32)
        role_away = torch.tensor([-1.0], dtype=torch.float32)
        encoded_feature_home = torch.cat([x_feat, role_home], dim=-1).unsqueeze(0)
        encoded_feature_away = torch.cat([x_feat, role_away], dim=-1).unsqueeze(0)

        new_h_home = self.gru_home(encoded_feature_home, h_home.unsqueeze(0)).squeeze(0)
        new_h_away = self.gru_away(encoded_feature_away, h_away.unsqueeze(0)).squeeze(0)
        return new_h_home, new_h_away