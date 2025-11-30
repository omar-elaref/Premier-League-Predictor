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
        # x: (B, T, F)
        _, h_n = self.gru(x)  # h_n: (num_layers * num_directions, B, H)
        # previous layer
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
        meta_numeric_dim: int,   # e.g. 3 for B365H/B365D/B365A + maybe 1 for home/away flag
        ff_hidden_dim: int = 128,
        dropout: float = 0.2,
        share_team_encoders: bool = True,
    ):
        super().__init__()

        # Team ID embeddings 
        self.team_emb = nn.Embedding(num_teams, team_id_emb_dim)

        # Ground (home/away) as tiny embedding (2 values: 0=home,1=away)
        self.ground_emb = nn.Embedding(2, 4)

        # Encoders
        # Enc1: head-to-head history
        self.h2h_encoder = MatchHistoryEncoder(hist_feat_dim, h2h_hidden_dim)

        # Enc2 and Enc3: form of each team
        self.share_team_encoders = share_team_encoders
        self.team_encoder1 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)
        if share_team_encoders:
            self.team_encoder2 = self.team_encoder1
        else:
            self.team_encoder2 = MatchHistoryEncoder(hist_feat_dim, team_hidden_dim)

        # Metadata projection 
        # metadata = [team1_emb, team2_emb, ground_emb, numeric_meta]
        meta_in_dim = 2 * team_id_emb_dim + 4 + meta_numeric_dim
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_in_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final FF combining everything
        total_in = self.h2h_encoder.output_dim + 2 * self.team_encoder1.output_dim + ff_hidden_dim
        self.ff = nn.Sequential(
            nn.Linear(total_in, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim // 2, 2),  # goals for both teams, [goals_team1, goals_team2]
        )

    def forward(
        self,
        team1_ids,          # (B,)
        team2_ids,          # (B,)
        ground_flags,       # (B,) 0=team1_home, 1=team2_home or similar
        meta_numeric,       # (B, meta_numeric_dim) e.g. [B365H,B365D,B365A]
        h2h_seq,            # (B, T_h2h, hist_feat_dim)
        team1_seq,          # (B, T_team, hist_feat_dim)
        team2_seq           # (B, T_team, hist_feat_dim)
    ):
        # Embeddings for metadata 
        t1_emb = self.team_emb(team1_ids)       # (B, team_id_emb_dim)
        t2_emb = self.team_emb(team2_ids)       # (B, team_id_emb_dim)
        g_emb = self.ground_emb(ground_flags)   # (B, 4)

        meta = torch.cat([t1_emb, t2_emb, g_emb, meta_numeric], dim=-1)  # (B, meta_in_dim)
        meta_repr = self.meta_mlp(meta)  # (B, ff_hidden_dim)

        # Encoders
        h_h2h = self.h2h_encoder(h2h_seq)           # (B, h2h_hidden_dim[*2])
        h_t1  = self.team_encoder1(team1_seq)       # (B, team_hidden_dim[*2])
        h_t2  = self.team_encoder2(team2_seq)       # (B, team_hidden_dim[*2])

        # Combine
        combined = torch.cat([meta_repr, h_h2h, h_t1, h_t2], dim=-1)
        goals = self.ff(combined)   # (B, 2)

        # Optional: enforce non-negativity with softplus
        goals = F.softplus(goals)

        return goals  # predicted [goals_team1, goals_team2]


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
        self.matches_df = data["matches_df"]  # optional, used for debugging

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

