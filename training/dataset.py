import random

import torch

PAD, UNK, BOS, EOS = 0, 1, 2, 3  # must match the ids used in train_tokenizer.py


def load_pairs(prefix, sp, max_len=100):
    """Return (src_ids, tgt_in_ids, tgt_out_ids) per sentence pair.

    src     = tokens + EOS
    tgt_in  = BOS + tokens        (what the decoder reads)
    tgt_out = tokens + EOS        (what the decoder must predict; shifted by one)
    """
    from pathlib import Path

    en = Path(f"{prefix}.en").read_text(encoding="utf-8").splitlines()
    hi = Path(f"{prefix}.hi").read_text(encoding="utf-8").splitlines()
    assert len(en) == len(hi)
    pairs = []
    for e, h in zip(en, hi):
        s, t = sp.encode(e), sp.encode(h)
        if len(s) + 1 > max_len or len(t) + 1 > max_len:
            continue
        pairs.append((s + [EOS], [BOS] + t, t + [EOS]))
    return pairs


def make_batches(pairs, max_tokens=3000, shuffle=True):
    """Group sentences of similar length so little padding is wasted.

    A batch's padded size (longest sentence x number of sentences) stays under max_tokens.
    """
    length = lambda i: max(len(pairs[i][0]), len(pairs[i][1]))
    order = sorted(range(len(pairs)), key=length)
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = length(i)
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(pairs, batch_idx):
    def pad(seqs):
        L = max(len(s) for s in seqs)
        return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])

    return (
        pad([pairs[i][0] for i in batch_idx]),
        pad([pairs[i][1] for i in batch_idx]),
        pad([pairs[i][2] for i in batch_idx]),
    )