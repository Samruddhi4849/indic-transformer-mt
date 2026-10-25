import csv
from pathlib import Path

import matplotlib.pyplot as plt

SUMMARY = Path("results/exp/summary.csv")
RESULTS = Path("results/exp/results.csv")


def read_csv(path):
    # tolerant reader: strips spaces and a possible BOM, skips blank lines
    with open(path, encoding="utf-8-sig") as f:
        lines = [[c.strip() for c in row] for row in csv.reader(f) if any(c.strip() for c in row)]
    header, body = lines[0], lines[1:]
    return [dict(zip(header, row)) for row in body]


if not SUMMARY.exists():
    raise SystemExit("Missing results/exp/summary.csv")

rows = {}
for r in read_csv(SUMMARY):
    if r["tag"].startswith("n"):
        rows[r["tag"]] = r
tags = sorted(rows, key=lambda t: int(rows[t]["train_pairs"]))
size = [int(rows[t]["train_pairs"]) for t in tags]
ppl = [float(rows[t]["best_val_ppl"]) for t in tags]

scores = {}
if RESULTS.exists():
    for r in read_csv(RESULTS):
        scores[r["tag"]] = r

print(f"{'pairs':>8} {'val ppl':>9} {'BLEU':>7} {'chrF++':>8}")
for t, n, p in zip(tags, size, ppl):
    s = scores.get(t)
    print(f"{n:>8} {p:>9.1f} {(s['bleu'] if s else '-'):>7} {(s['chrf++'] if s else '-'):>8}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(size, ppl, marker="o")
ax[0].set_xscale("log")
ax[0].set_xlabel("training pairs (log scale)")
ax[0].set_ylabel("validation perplexity (lower is better)")
ax[0].set_title("Validation perplexity vs data size")

st = [t for t in tags if t in scores]
if st:
    sn = [int(rows[t]["train_pairs"]) for t in st]
    ax[1].plot(sn, [float(scores[t]["chrf++"]) for t in st], marker="o", label="chrF++")
    ax[1].plot(sn, [float(scores[t]["bleu"]) for t in st], marker="s", label="BLEU")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training pairs (log scale)")
    ax[1].set_ylabel("score (higher is better)")
    ax[1].set_title("FLORES dev, first 200 sentences, greedy")
    ax[1].legend()

plt.tight_layout()
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/data_efficiency.png", dpi=150, bbox_inches="tight")
print("saved docs/data_efficiency.png")