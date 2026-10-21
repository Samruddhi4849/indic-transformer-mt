import argparse
import csv
import math
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from model.transformer import Transformer
from training.dataset import BOS, EOS, PAD, collate, load_pairs, make_batches
from training.schedule import noam

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--warmup", type=int, default=800)
ap.add_argument("--overfit", type=int, default=0, help="train on N pairs only (pipeline sanity check)")
ap.add_argument("--data", default="data/train.clean", help="prefix of the .en/.hi training files")
ap.add_argument("--spm", default="data/spm.model")
ap.add_argument("--size", type=int, default=0, help="use only the first N training pairs")
ap.add_argument("--tag", default="", help="name for this run's checkpoint and log")
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without improvement (0 = off)")
args = ap.parse_args()

CFG = dict(d_model=256, num_layers=3, num_heads=4, d_ff=1024, dropout=args.dropout,
           lr_factor=0.5, max_tokens=3000, label_smoothing=0.1, max_len=100, n_val=500)

tag = args.tag or ("overfit" if args.overfit else "")
ckpt_path = f"checkpoints/{tag}.pt" if tag else "checkpoints/best.pt"
log_path = f"logs/{tag}.csv" if tag else "logs/train_log.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)
print("device:", device)

sp = spm.SentencePieceProcessor(model_file=args.spm)
pairs = load_pairs(args.data, sp, CFG["max_len"])
if args.overfit:
    train_pairs = val_pairs = pairs[: args.overfit]
    args.warmup = 100
    samples = [(sp.decode(p[0][:-1]), sp.decode(p[2][:-1])) for p in pairs[:3]]
else:
    train_pairs, val_pairs = pairs[: -CFG["n_val"]], pairs[-CFG["n_val"]:]
    if args.size:
        train_pairs = train_pairs[: args.size]
    en = Path("data/flores.dev.en").read_text(encoding="utf-8").splitlines()[:3]
    hi = Path("data/flores.dev.hi").read_text(encoding="utf-8").splitlines()[:3]
    samples = list(zip(en, hi))
print(f"train pairs: {len(train_pairs):,}   val pairs: {len(val_pairs):,}")

model = Transformer(sp.get_piece_size(), CFG["d_model"], CFG["num_layers"], CFG["num_heads"],
                    CFG["d_ff"], CFG["dropout"], pad_id=PAD).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: noam(s + 1, CFG["d_model"], args.warmup, CFG["lr_factor"]))


@torch.no_grad()
def translate(text, max_len=100):
    model.eval()
    src = torch.tensor([sp.encode(text) + [EOS]], device=device)
    memory, src_mask = model.encode(src)
    ys = torch.tensor([[BOS]], device=device)
    for _ in range(max_len):
        nxt = model.decode(ys, memory, src_mask)[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == EOS:
            break
        ys = torch.cat([ys, nxt], dim=1)
    return sp.decode(ys[0, 1:].tolist())


@torch.no_grad()
def evaluate(eval_pairs):
    model.eval()
    total, ntok = 0.0, 0
    for b in make_batches(eval_pairs, CFG["max_tokens"], shuffle=False):
        src, tgt_in, tgt_out = [t.to(device) for t in collate(eval_pairs, b)]
        logits = model(src, tgt_in)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                                 ignore_index=PAD, reduction="sum").item()
        ntok += (tgt_out != PAD).sum().item()
    return total / ntok


Path("checkpoints").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
log = open(log_path, "w", newline="")
writer = csv.writer(log)
writer.writerow(["epoch", "step", "train_loss", "val_loss", "val_ppl", "lr"])

best, best_epoch, step, bad = float("inf"), 0, 0, 0
for epoch in range(1, args.epochs + 1):
    model.train()
    t0, total, ntok = time.time(), 0.0, 0
    for b in make_batches(train_pairs, CFG["max_tokens"]):
        src, tgt_in, tgt_out = [t.to(device) for t in collate(train_pairs, b)]
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1),
                               ignore_index=PAD, label_smoothing=CFG["label_smoothing"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        n = (tgt_out != PAD).sum().item()
        total += loss.item() * n
        ntok += n

    train_loss, val_loss = total / ntok, evaluate(val_pairs)
    lr = opt.param_groups[0]["lr"]
    writer.writerow([epoch, step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{math.exp(val_loss):.2f}", f"{lr:.6f}"])
    log.flush()
    print(f"epoch {epoch:3d}  step {step:5d}  train {train_loss:.3f}  "
          f"val {val_loss:.3f}  ppl {math.exp(val_loss):7.2f}  lr {lr:.2e}  {time.time() - t0:.0f}s")

    if val_loss < best:
        best, best_epoch, bad = val_loss, epoch, 0
        torch.save({"model": model.state_dict(), "cfg": CFG}, ckpt_path)
    else:
        bad += 1
    if epoch % 10 == 0:
        for src_text, ref in samples:
            print(f"  EN : {src_text}\n  REF: {ref}\n  HYP: {translate(src_text)}\n")
    if args.patience and bad >= args.patience:
        print(f"early stop: no improvement for {args.patience} epochs")
        break

log.close()
print(f"BEST epoch {best_epoch}  val {best:.3f}  ppl {math.exp(best):.2f}")
new = not Path("logs/summary.csv").exists()
with open("logs/summary.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["tag", "train_pairs", "dropout", "best_epoch", "best_val_loss", "best_val_ppl"])
    w.writerow([tag or "best", len(train_pairs), CFG["dropout"], best_epoch,
                f"{best:.4f}", f"{math.exp(best):.2f}"])