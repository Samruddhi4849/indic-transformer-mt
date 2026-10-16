import math

import torch

from model.encoder import Encoder, EncoderLayer
from model.layers import PositionalEncoding, PositionwiseFeedForward
from model.masks import padding_mask


def test_ffn_shape_and_position_independence():
    ffn = PositionwiseFeedForward(64, 256)
    x = torch.randn(2, 5, 64)
    x2 = x.clone()
    x2[:, 2] = torch.randn(2, 64)                  # change only position 2
    out, out2 = ffn(x), ffn(x2)
    assert out.shape == (2, 5, 64)
    assert torch.allclose(out[:, [0, 1, 3, 4]], out2[:, [0, 1, 3, 4]])


def test_positional_encoding_known_values():
    pe = PositionalEncoding(64, dropout=0.0, max_len=100).pe[0]
    assert pe.shape == (100, 64)
    assert torch.allclose(pe[0, 0::2], torch.zeros(32))   # sin(0) = 0
    assert torch.allclose(pe[0, 1::2], torch.ones(32))    # cos(0) = 1
    assert abs(pe[3, 0].item() - math.sin(3)) < 1e-5      # first frequency is 1


def test_positional_encoding_relative_offset_is_linear():
    # Paper: PE(pos+k) is a linear function of PE(pos). For the first sin/cos pair
    # this is just a rotation matrix by angle k.
    pe = PositionalEncoding(64, dropout=0.0, max_len=100).pe[0]
    k = 5
    rot = torch.tensor([[math.cos(k), math.sin(k)], [-math.sin(k), math.cos(k)]])
    assert torch.allclose(pe[k:k + 50, :2], pe[:50, :2] @ rot.T, atol=1e-4)


def test_encoder_layer_parameter_count():
    layer = EncoderLayer(512, 8, 2048)
    assert sum(p.numel() for p in layer.parameters()) == 3_152_384


def test_encoder_shapes_and_independent_layers():
    enc = Encoder(num_layers=6, d_model=64, num_heads=4, d_ff=128)
    assert enc(torch.randn(2, 9, 64)).shape == (2, 9, 64)
    w0 = enc.layers[0].self_attn.w_q.weight
    w1 = enc.layers[1].self_attn.w_q.weight
    assert w0.data_ptr() != w1.data_ptr()          # separate weights, not shared


def test_padding_does_not_leak_into_real_tokens():
    torch.manual_seed(0)
    enc = Encoder(num_layers=2, d_model=64, num_heads=4, d_ff=128).eval()
    tokens = torch.tensor([[5, 6, 7, 0, 0]])       # last two are padding
    mask = padding_mask(tokens)
    x = torch.randn(1, 5, 64)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 64)              # change only the padded positions
    assert torch.allclose(enc(x, mask)[:, :3], enc(x2, mask)[:, :3], atol=1e-5)