import math

import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder
from model.layers import PositionalEncoding
from model.masks import causal_mask, padding_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, dropout=0.1, pad_id=0, max_len=5000):
        super().__init__()
        self.d_model, self.pad_id = d_model, pad_id

        self.embed = nn.Embedding(vocab_size, d_model)          # shared by encoder and decoder
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight                # paper 3.4: tie the three matrices

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)

    def _embed(self, tokens):
        # paper 3.4: multiply embedding weights by sqrt(d_model), then add positions
        return self.pos(self.embed(tokens) * math.sqrt(self.d_model))

    def encode(self, src):
        src_mask = padding_mask(src, self.pad_id)
        return self.encoder(self._embed(src), src_mask), src_mask

    def decode(self, tgt, memory, src_mask):
        tgt_mask = causal_mask(tgt.size(1)).to(tgt.device) & padding_mask(tgt, self.pad_id)
        return self.out_proj(self.decoder(self._embed(tgt), memory, src_mask, tgt_mask))

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)   # logits: (batch, tgt_len, vocab)