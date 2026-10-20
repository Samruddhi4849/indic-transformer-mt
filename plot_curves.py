import csv
from pathlib import Path

import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("logs/train_log.csv")))
ep = [int(r["epoch"]) for r in rows]
tr = [float(r["train_loss"]) for r in rows]
va = [float(r["val_loss"]) for r in rows]
best = min(range(len(va)), key=va.__getitem__)

Path("docs").mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(ep, tr, label="train (label-smoothed)")
plt.plot(ep, va, label="validation")
plt.axvline(ep[best], ls="--", color="gray", label=f"best epoch {ep[best]}")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Training curves: English to Hindi, 19K pairs")
plt.savefig("docs/loss_curves.png", dpi=150, bbox_inches="tight")
print("best epoch:", ep[best], "val loss:", va[best])