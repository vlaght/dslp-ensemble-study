"""
DSLP (Dual Stream LSTM Phoneme) model definitions (h=256, l=2).

Mean pooling aggregation — ablation showed attention-MIL adds no significant
benefit (p=0.473, d=0.198 on AUC across 10-fold CV).

Classes:
  UniDSLP  — unidirectional LSTM, mean pooling  (flagship)
  BiDSLP   — bidirectional LSTM, mean pooling
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMB_DIM  = 8
HIDDEN   = 256
N_LAYERS = 2
DROPOUT  = 0.3


class _DSLPBase(nn.Module):
    """Shared mean-pool stream logic. Subclasses set bidirectional flag."""

    bidirectional = False

    def __init__(self, num_visual, num_audio, num_phonemes,
                 embedding_dim=EMB_DIM, hidden_dim=HIDDEN,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        D = hidden_dim * (2 if self.bidirectional else 1)
        self.phoneme_embedding = nn.Embedding(num_phonemes, embedding_dim)

        self.vis_lstm     = nn.LSTM(num_visual + embedding_dim, hidden_dim, num_layers,
                                    batch_first=True, bidirectional=self.bidirectional,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.vis_instance = nn.Linear(D, 1)

        self.aud_lstm     = nn.LSTM(num_audio + embedding_dim, hidden_dim, num_layers,
                                    batch_first=True, bidirectional=self.bidirectional,
                                    dropout=dropout if num_layers > 1 else 0.0)
        self.aud_instance = nn.Linear(D, 1)

        self.fusion = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )

    def _stream(self, lstm, inst_fc, feats, phon_emb, lengths):
        T = feats.shape[1]
        x = torch.cat([feats, phon_emb], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        inst  = inst_fc(out)  # (B, T, 1)
        valid = (torch.arange(T, device=feats.device).unsqueeze(0)
                 < lengths.to(feats.device).unsqueeze(1))
        inst  = inst.masked_fill(~valid.unsqueeze(-1), 0.0)
        return inst.sum(dim=1) / lengths.float().unsqueeze(-1).to(feats.device)

    def forward(self, visual, audio, phons, lengths):
        phon_emb = self.phoneme_embedding(phons)
        vis = self._stream(self.vis_lstm, self.vis_instance, visual, phon_emb, lengths)
        aud = self._stream(self.aud_lstm, self.aud_instance, audio, phon_emb, lengths)
        return self.fusion(torch.cat([vis, aud], dim=-1)).squeeze(-1)


class UniDSLP(_DSLPBase):
    bidirectional = False


class BiDSLP(_DSLPBase):
    bidirectional = True
