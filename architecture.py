import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from build_dataset import *
from torch.utils.data import Dataset

class MatchHistoryEncoder(nn.Module):
    
    def __init__(self, feat_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_last = h_n[-1]      
        return h_last


class FootballScorePredictor(nn.Module):
    
    def __init__(
        self,
        num_teams: int,
        team_id_emb_dim: int,
        hist_feat_dim: int,
        h2h_hidden_dim: int,
        team_hidden_dim: int,
        meta_numeric_dim: int,
        ff_hidden_dim: int = 128,
        dropout: float = 0.2,
        share_team_encoders: bool = True,
    ):
        super().__init__()

        self.team_emb = nn.Embedding(num_teams, team_id_emb_dim)

        self.ground_emb = nn.Embedding(2, 4)

        self.h2h_encoder = MatchHistoryEncoder(hist_feat_dim, h2h_hidden_dim)

        self.share_team_encoders = share_team_encoders
        self.team_encoder1 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)
        if share_team_encoders:
            self.team_encoder2 = self.team_encoder1
        else:
            self.team_encoder2 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)

        meta_in_dim = 2 * team_id_emb_dim + 4 + meta_numeric_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        total_in = self.h2h_encoder.output_dim + 2 * self.team_encoder1.output_dim + ff_hidden_dim
        self.ff = nn.Sequential(
            nn.Linear(total_in, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim // 2, 2),
        )

    def forward(
        self,
        team1_ids,          
        team2_ids,          
        ground_flags,       
        meta_numeric,      
        h2h_seq,            
        team1_seq,          
        team2_seq          
    ):

        t1_emb = self.team_emb(team1_ids)
        t2_emb = self.team_emb(team2_ids)
        g_emb = self.ground_emb(ground_flags)

        meta = torch.cat([t1_emb, t2_emb, g_emb, meta_numeric], dim=-1)
        meta_repr = self.meta_mlp(meta)

        h_h2h = self.h2h_encoder(h2h_seq)
        h_t1  = self.team_encoder1(team1_seq)
        h_t2  = self.team_encoder2(team2_seq)       

        combined = torch.cat([meta_repr, h_h2h, h_t1, h_t2], dim=-1)
        goals = self.ff(combined)

        goals = F.softplus(goals)

        return goals


class FootballSequenceDataset(Dataset):
    def __init__(self, seasons_dict, k_form=5, k_h2h=5, odds_cols=("B365H", "B365D", "B365A")):
        data = build_match_history_tensors(
            seasons_dict,
            k_form=k_form,
            k_h2h=k_h2h,
            odds_cols=odds_cols,
        )
        self.team1_ids = data["team1_ids"]
        self.team2_ids = data["team2_ids"]
        self.ground_flags = data["ground_flags"]
        self.meta_numeric = data["meta_numeric"]
        self.h2h_seq = data["h2h_seq"]
        self.team1_seq = data["team1_seq"]
        self.team2_seq = data["team2_seq"]
        self.y_goals = data["y_goals"]

        self.team_to_id = data["team_to_id"]
        self.matches_df = data["matches_df"]

    def __len__(self):
        return self.y_goals.shape[0]

    def __getitem__(self, idx):
        return {
            "team1_ids": self.team1_ids[idx],
            "team2_ids": self.team2_ids[idx],
            "ground_flags": self.ground_flags[idx],
            "meta_numeric": self.meta_numeric[idx],
            "h2h_seq": self.h2h_seq[idx],
            "team1_seq": self.team1_seq[idx],
            "team2_seq": self.team2_seq[idx],
            "y": self.y_goals[idx],
        }

