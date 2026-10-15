import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V   (paper, Eq. 1)

    q, k, v: (..., seq, d_k)
    mask:    broadcastable to (..., seq_q, seq_k); True = may attend, False = blocked
    Returns (output, attention_weights).
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Paper section 3.2.2: h parallel heads, each with its own learned projections."""

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = None  # saved for attention visualizations later

    def _split_heads(self, x):
        b, n, _ = x.shape
        return x.view(b, n, self.h, self.d_k).transpose(1, 2)  # (b, h, n, d_k)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, self.attn = scaled_dot_product_attention(q, k, v, mask)

        b = query.size(0)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.w_o(out)