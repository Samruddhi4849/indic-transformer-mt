import argparse
import csv
import time
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch

from model.decoding import beam_search
from model.transformer import Transformer
from training.dataset import BOS, EOS, PAD

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "devtest"])
ap.add_argument("--beam", type=int, default=4)
ap.add_argument("--alpha", type=float, default=0.6)
ap.add_argument("--max_len", type=int, default=100)
ap.add_argument("--n", type=int, default=0, help="only the first N sentences (quick test)")
ap.add_argument("--ckpt", default="checkpoints/best.pt")
ap.add_argument("--spm", default="data/spm.model")
ap.add_argument("--tag", default="")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
sp = spm.SentencePieceProcessor(model_file=args.spm)

ckpt = torch.load(args.ckpt, map_location=device)
cfg = ckpt["cfg"]
model = Transformer(sp.get_piece_size(), cfg["d_model"], cfg["num_layers"], cfg["num_heads"],
                    cfg["d_ff"], cfg["dropout"], pad_id=PAD).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

src_lines = Path(f"data/flores.{args.split}.en").read_text(encoding="utf-8").splitlines()
refs = Path(f"data/flores.{args.split}.hi").read_text(encoding="utf-8").splitlines()
if args.n:
    src_lines, refs = src_lines[: args.n], refs[: args.n]

t0 = time.time()
hyps = []
for i, line in enumerate(src_lines, 1):
    ids = torch.tensor([sp.encode(line) + [EOS]], device=device)
    cap = min(args.max_len, 2 * ids.size(1) + 10)   # a translation can't be hugely longer than its source
    hyps.append(sp.decode(beam_search(model, ids, BOS, EOS, args.beam, args.alpha, max_len=cap)))
    if i % 10 == 0:
        print(f"translated {i}/{len(src_lines)}  ({time.time() - t0:.0f}s)", flush=True)

bleu = sacrebleu.corpus_bleu(hyps, [refs])
chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)     # word_order=2 -> chrF++
print(f"\ntag={args.tag} split={args.split} beam={args.beam} alpha={args.alpha} sentences={len(hyps)}")
print(bleu)
print(chrf)
try:
    spbleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    print(f"spBLEU (flores200 tokenizer): {spbleu:.2f}")
except Exception:
    spbleu = float("nan")

Path("results").mkdir(exist_ok=True)
name = f"{args.tag}.{args.split}" if args.tag else args.split
Path(f"results/hyp.{name}.hi").write_text("\n".join(hyps) + "\n", encoding="utf-8")
new = not Path("results/results.csv").exists()
with open("results/results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["tag", "split", "beam", "alpha", "n", "bleu", "chrf++", "spbleu"])
    w.writerow([args.tag, args.split, args.beam, args.alpha, len(hyps),
                f"{bleu.score:.2f}", f"{chrf.score:.2f}", f"{spbleu:.2f}"])

print("\nFirst 3 examples:")
for s, r, h in list(zip(src_lines, refs, hyps))[:3]:
    print(f"  EN : {s}\n  REF: {r}\n  HYP: {h}\n")