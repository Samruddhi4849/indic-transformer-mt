import random

from training.dataset import PAD, collate, make_batches
from training.schedule import noam


def fake_pairs(n=300):
    rng = random.Random(0)
    out = []
    for _ in range(n):
        a, b = rng.randint(3, 40), rng.randint(3, 40)
        out.append(([5] * a + [3], [2] + [6] * b, [6] * b + [3]))
    return out


def test_noam_peaks_exactly_at_warmup():
    d, w, f = 256, 800, 0.5
    lrs = [noam(s, d, w, f) for s in range(1, 3000)]   # lrs[k] is step k+1
    assert lrs.index(max(lrs)) + 1 == w
    assert abs(max(lrs) - f * d ** -0.5 * w ** -0.5) < 1e-9
    assert lrs[0] < lrs[w // 2] < lrs[w - 1]            # rising during warmup
    assert lrs[w + 500] > lrs[w + 1500]                 # then decaying


def test_every_pair_used_exactly_once():
    pairs = fake_pairs()
    flat = sorted(i for b in make_batches(pairs, max_tokens=500) for i in b)
    assert flat == list(range(len(pairs)))


def test_batches_respect_token_budget():
    pairs = fake_pairs()
    for b in make_batches(pairs, max_tokens=500):
        src, tgt_in, tgt_out = collate(pairs, b)
        assert src.numel() <= 500 and tgt_in.numel() <= 500


def test_collate_pads_with_pad_id():
    pairs = [([5, 5, 3], [2, 6], [6, 3]), ([5, 3], [2, 6, 6, 6], [6, 6, 6, 3])]
    src, tgt_in, _ = collate(pairs, [0, 1])
    assert src.shape == (2, 3) and tgt_in.shape == (2, 4)
    assert src[1, 2].item() == PAD and tgt_in[0, 3].item() == PAD