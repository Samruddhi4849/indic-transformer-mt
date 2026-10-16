import torch

from model.decoder import DecoderLayer
from model.transformer import Transformer


def tiny(vocab=30):
    return Transformer(vocab, d_model=32, num_layers=2, num_heads=4, d_ff=64, dropout=0.0)


def test_output_shape():
    src = torch.randint(3, 30, (2, 7))
    tgt = torch.randint(3, 30, (2, 5))
    assert tiny()(src, tgt).shape == (2, 5, 30)


def test_weight_sharing():
    m = tiny()
    assert m.out_proj.weight is m.embed.weight


def test_decoder_layer_parameter_count():
    # 2 attention blocks (2,101,248) + FFN (2,099,712) + 3 LayerNorms (3,072)
    assert sum(p.numel() for p in DecoderLayer(512, 8, 2048).parameters()) == 4_204_032


def test_future_target_tokens_do_not_change_earlier_logits():
    torch.manual_seed(0)
    m = tiny().eval()
    src = torch.randint(3, 30, (1, 6))
    tgt = torch.randint(3, 30, (1, 5))
    tgt2 = tgt.clone()
    tgt2[:, 3:] = torch.randint(3, 30, (1, 2))
    assert torch.allclose(m(src, tgt)[:, :3], m(src, tgt2)[:, :3], atol=1e-5)


def test_source_padding_is_ignored():
    torch.manual_seed(0)
    m = tiny().eval()
    src = torch.randint(3, 30, (1, 5))
    src_padded = torch.cat([src, torch.zeros(1, 3, dtype=torch.long)], dim=1)
    tgt = torch.randint(3, 30, (1, 4))
    assert torch.allclose(m(src, tgt), m(src_padded, tgt), atol=1e-5)