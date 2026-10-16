import math

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Paper section 3.3: FFN(x) = max(0, xW1 + b1)W2 + b2, applied to each position separately."""

    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class PositionalEncoding(nn.Module):
    """Paper section 3.5: fixed sine/cosine encodings added to the embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_len).unsqueeze(1)                       # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                    # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)  # paper applies dropout to embeddings + positional encodings