import argparse
import csv
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch

from model.decoding import greedy_decode
from model.transformer import Transformer
from training.dataset import BOS, EOS, PAD

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--ckpt", default="checkpoints/best.pt")
ap.add_argument("--spm", default="data/spm.model")
ap.add_argument("--out", default="results/error_analysis.csv")
ap.add_argument("--summary", action="store_true", help="count the categories you filled in")
args = ap.parse_args()

if args.summary:
    rows = list(csv.DictReader(open(args.out, encoding="utf-8-sig")))
    counts = Counter(r["category"].strip() or "(unlabeled)" for r in rows)
    for cat, n in counts.most_common():
        print(f"{n:3d}  {cat}")
    print(f"total {len(rows)}")
    raise SystemExit

sp = spm.SentencePieceProcessor(model_file=args.spm)
ckpt = torch.load(args.ckpt, map_location="cpu")
cfg = ckpt["cfg"]
model = Transformer(sp.get_piece_size(), cfg["d_model"], cfg["num_layers"], cfg["num_heads"],
                    cfg["d_ff"], cfg["dropout"], pad_id=PAD)
model.load_state_dict(ckpt["model"])
model.eval()

en = Path("data/flores.dev.en").read_text(encoding="utf-8").splitlines()[: args.n]
hi = Path("data/flores.dev.hi").read_text(encoding="utf-8").splitlines()[: args.n]

Path("results").mkdir(exist_ok=True)
with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "english", "reference", "hypothesis", "len_ratio", "repeats", "category", "notes"])
    for i, (e, r) in enumerate(zip(en, hi), 1):
        ids = torch.tensor([sp.encode(e) + [EOS]])
        cap = min(100, 2 * ids.size(1) + 10)
        h = sp.decode(greedy_decode(model, ids, BOS, EOS, max_len=cap))
        words = h.split()
        trigrams = [tuple(words[j:j + 3]) for j in range(len(words) - 2)]
        repeats = "yes" if len(trigrams) != len(set(trigrams)) else "no"
        ratio = round(len(words) / max(len(r.split()), 1), 2)
        w.writerow([i, e, r, h, ratio, repeats, "", ""])
        print(f"done {i}/{args.n}", flush=True)
print("saved", args.out)