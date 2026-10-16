import torch.nn as nn

from model.attention import MultiHeadAttention
from model.layers import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Paper section 3.1: two sub-layers, each wrapped as LayerNorm(x + Dropout(Sublayer(x)))."""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """A stack of N identical layers (each with its own weights)."""

    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x