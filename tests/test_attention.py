import torch
import torch.nn.functional as F

from model.attention import MultiHeadAttention, scaled_dot_product_attention
from model.masks import causal_mask, padding_mask


def test_output_shapes():
    mha = MultiHeadAttention(512, 8)
    x = torch.randn(2, 10, 512)
    assert mha(x, x, x).shape == (2, 10, 512)
    assert mha.attn.shape == (2, 8, 10, 10)


def test_attention_rows_sum_to_one():
    q = k = v = torch.randn(2, 4, 7, 16)
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 7), atol=1e-5)


def test_matches_pytorch_reference():
    q, k, v = (torch.randn(2, 4, 7, 16) for _ in range(3))
    mask = causal_mask(7)
    ours, _ = scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_causal_mask_blocks_future():
    mha = MultiHeadAttention(64, 4)
    x = torch.randn(1, 6, 64)
    mha(x, x, x, causal_mask(6))
    assert torch.all(torch.triu(mha.attn[0, 0], diagonal=1) == 0)


def test_changing_future_tokens_does_not_change_past_outputs():
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 4).eval()
    x = torch.randn(1, 6, 64)
    out1 = mha(x, x, x, causal_mask(6))
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 64)
    out2 = mha(x2, x2, x2, causal_mask(6))
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-6)


def test_padding_gets_zero_attention():
    mha = MultiHeadAttention(64, 4)
    tokens = torch.tensor([[5, 6, 7, 0, 0]])  # last two are padding
    x = torch.randn(1, 5, 64)
    mha(x, x, x, padding_mask(tokens))
    assert torch.all(mha.attn[..., 3:] == 0)