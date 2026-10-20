import torch

from model.decoding import beam_search, greedy_decode
from model.transformer import Transformer

BOS, EOS = 2, 3


def tiny():
    torch.manual_seed(0)
    return Transformer(30, d_model=32, num_layers=2, num_heads=4, d_ff=64, dropout=0.0).eval()


def test_beam_of_one_equals_greedy():
    m = tiny()
    for seed in range(5):
        torch.manual_seed(seed)
        src = torch.randint(4, 30, (1, 7))
        assert beam_search(m, src, BOS, EOS, beam_size=1, max_len=12) == \
               greedy_decode(m, src, BOS, EOS, max_len=12)


def test_beam_output_is_clean_and_bounded():
    m = tiny()
    src = torch.randint(4, 30, (1, 7))
    out = beam_search(m, src, BOS, EOS, beam_size=4, max_len=12)
    assert len(out) <= 12 and BOS not in out and EOS not in out