import argparse
import math

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
from matplotlib import font_manager

from model.decoding import greedy_decode
from model.transformer import Transformer
from training.dataset import BOS, EOS, PAD

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="The government has announced a new policy.")
ap.add_argument("--ckpt", default="checkpoints/best.pt")
ap.add_argument("--spm", default="data/spm.model")
ap.add_argument("--out", default="docs/attention_example.png")
args = ap.parse_args()

# pick a font that can draw Devanagari
available = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Nirmala UI", "Mangal", "Noto Sans Devanagari", "DejaVu Sans"]:
    if name in available:
        plt.rcParams["font.family"] = name
        break

sp = spm.SentencePieceProcessor(model_file=args.spm)
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt["cfg"]
model = Transformer(sp.get_piece_size(), cfg["d_model"], cfg["num_layers"], cfg["num_heads"],
                    cfg["d_ff"], cfg["dropout"], pad_id=PAD)
model.load_state_dict(ckpt["model"])
model.eval()

src_ids = sp.encode(args.text) + [EOS]
src = torch.tensor([src_ids])
out_ids = greedy_decode(model, src, BOS, EOS, max_len=30)
print("EN :", args.text)
print("HI :", sp.decode(out_ids))

# run the decoder once on the generated output to read its cross-attention weights
tgt_in = torch.tensor([[BOS] + out_ids])
with torch.no_grad():
    model(src, tgt_in)
attn = model.decoder.layers[-1].cross_attn.attn[0]        # (heads, target_len, source_len)

src_labels = [sp.id_to_piece(i) for i in src_ids[:-1]] + ["</s>"]
tgt_labels = [sp.id_to_piece(i) for i in out_ids] + ["</s>"]

h = attn.shape[0]
rows = math.ceil(h / 2)
fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), squeeze=False)
for i in range(rows * 2):
    ax = axes[i // 2][i % 2]
    if i >= h:
        ax.axis("off")
        continue
    ax.imshow(attn[i].numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_labels)))
    ax.set_xticklabels(src_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_labels)))
    ax.set_yticklabels(tgt_labels, fontsize=8)
    ax.set_title(f"Head {i + 1}")
fig.suptitle("Decoder cross-attention, last layer\n"
             "rows = Hindi output tokens, columns = English tokens")
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print("saved", args.out)