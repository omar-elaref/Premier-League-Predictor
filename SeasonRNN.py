import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
        xh = self.enc(x_feat)            
        z  = torch.cat([h_home, h_away, xh], dim=-1).unsqueeze(0)  
        logits = self.head(z).squeeze(0) 
        return logits

    def update_states(self, h_home, h_away, x_feat):
        role_home = torch.tensor([+1.0], dtype=torch.float32)
        role_away = torch.tensor([-1.0], dtype=torch.float32)
        xh = torch.cat([x_feat, role_home], dim=-1).unsqueeze(0)  # (1, feat_dim+1)
        xa = torch.cat([x_feat, role_away], dim=-1).unsqueeze(0)

        new_h_home = self.gru_home(xh, h_home.unsqueeze(0)).squeeze(0)
        new_h_away = self.gru_away(xa, h_away.unsqueeze(0)).squeeze(0)
        return new_h_home, new_h_away