import torch


def _length_penalty(seq, alpha):
    # GNMT-style penalty (the paper's ref [38]): ((5 + length) / 6) ** alpha
    return ((5 + len(seq) - 1) / 6) ** alpha        # length excludes BOS


@torch.no_grad()
def greedy_decode(model, src, bos, eos, max_len=100):
    """src: (1, S) token ids. Returns generated ids without BOS/EOS."""
    memory, src_mask = model.encode(src)
    ys = torch.tensor([[bos]], device=src.device)
    for _ in range(max_len):
        nxt = model.decode(ys, memory, src_mask)[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == eos:
            break
        ys = torch.cat([ys, nxt], dim=1)
    return ys[0, 1:].tolist()


@torch.no_grad()
def beam_search(model, src, bos, eos, beam_size=4, alpha=0.6, max_len=100):
    """src: (1, S) token ids. Returns the best hypothesis without BOS/EOS."""
    memory, src_mask = model.encode(src)
    beams = [([bos], 0.0)]            # (tokens, summed log-probability)
    finished = []

    for _ in range(max_len):
        tokens = torch.tensor([b[0] for b in beams], device=src.device)
        mem = memory.expand(len(beams), -1, -1)
        mask = src_mask.expand(len(beams), -1, -1, -1)
        logp = torch.log_softmax(model.decode(tokens, mem, mask)[:, -1], dim=-1)

        cands = []
        for i, (seq, score) in enumerate(beams):
            top_lp, top_id = logp[i].topk(beam_size)
            for lp, idx in zip(top_lp.tolist(), top_id.tolist()):
                cands.append((seq + [idx], score + lp))
        cands.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for seq, score in cands[:beam_size]:
            (finished if seq[-1] == eos else beams).append((seq, score))
        if not beams or len(finished) >= beam_size:
            break

    pool = finished if finished else beams
    best, _ = max(pool, key=lambda c: c[1] / _length_penalty(c[0], alpha))
    out = best[1:]
    return out[:-1] if out and out[-1] == eos else out