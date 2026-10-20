import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.transformer import Transformer
from training.dataset import PAD, collate, load_pairs, make_batches

sp = spm.SentencePieceProcessor(model_file="data/spm.model")
ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
cfg = ckpt["cfg"]
model = Transformer(sp.get_piece_size(), cfg["d_model"], cfg["num_layers"], cfg["num_heads"],
                    cfg["d_ff"], cfg["dropout"], pad_id=PAD)
model.load_state_dict(ckpt["model"])
model.eval()


def loss_on(pairs):
    total, ntok = 0.0, 0
    with torch.no_grad():
        for b in make_batches(pairs, 3000, shuffle=False):
            src, tgt_in, tgt_out = collate(pairs, b)
            logits = model(src, tgt_in)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                                     ignore_index=PAD, reduction="sum").item()
            ntok += (tgt_out != PAD).sum().item()
    return total / ntok


pairs = load_pairs("data/train.clean", sp, 100)
print("pairs:", len(pairs))
print(f"first 200 (the overfit set):     {loss_on(pairs[:200]):.3f}")
print(f"pairs 200-1200 (trained on):     {loss_on(pairs[200:1200]):.3f}")
print(f"last 500 (validation):           {loss_on(pairs[-500:]):.3f}")