import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict, deque

# ---------- Model ----------
class EncoderRNN(nn.Module):
    """Encodes a sequence (L,F) into a single vector (E) with a GRU."""
    def __init__(self, feat_dim: int, enc_hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=enc_hidden, batch_first=True)
    def forward(self, seq):                 # seq: (B, L, F)
        _, h_n = self.gru(seq)              # h_n: (1, B, E)
        return h_n.squeeze(0)               # -> (B, E)

class MatchSeqModelNoLeak(nn.Module):
    """
    Predict with ONLY past encodings + previous form states (no x_cur in logits).
    After predicting, update form states using encodings that include current match.
    """
    def __init__(self, feat_dim: int, enc_hidden: int = 32, form_hidden: int = 64,
                 head_hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.enc = EncoderRNN(feat_dim, enc_hidden)
        # GRUCells update with encoding + role scalar (+1 home, -1 away)
        self.gru_home = nn.GRUCell(input_size=enc_hidden + 1, hidden_size=form_hidden)
        self.gru_away = nn.GRUCell(input_size=enc_hidden + 1, hidden_size=form_hidden)

        comb_in = (form_hidden * 2) + (enc_hidden * 2)   # h_home + h_away + e_home_past + e_away_past
        self.head = nn.Sequential(
            nn.Linear(comb_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes)
        )

    def forward_predict(self, h_home, h_away, e_home_past, e_away_past):
        """Logits from past-only information (no current features)."""
        z = torch.cat([h_home, h_away, e_home_past, e_away_past], dim=-1).unsqueeze(0)  # (1, D)
        return self.head(z).squeeze(0)  # (3,)

    @torch.no_grad()
    def update_states(self, h_home, h_away, e_home_now, e_away_now):
        """Advance states using encodings that include the current match (no grad)."""
        home_role = torch.tensor([+1.0], dtype=e_home_now.dtype, device=e_home_now.device)
        away_role = torch.tensor([-1.0], dtype=e_away_now.dtype, device=e_away_now.device)
        in_home = torch.cat([e_home_now, home_role], dim=-1)    # (E+1,)
        in_away = torch.cat([e_away_now, away_role], dim=-1)    # (E+1,)
        new_h_home = self.gru_home(in_home, h_home)
        new_h_away = self.gru_away(in_away, h_away)
        return new_h_home, new_h_away
