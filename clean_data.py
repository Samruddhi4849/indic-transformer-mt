import re
from pathlib import Path

DATA = Path("data")
MAX_WORDS = 80
MIN_RATIO, MAX_RATIO = 0.4, 2.5

en = (DATA / "train20k.en").read_text(encoding="utf-8").splitlines()
hi = (DATA / "train20k.hi").read_text(encoding="utf-8").splitlines()
assert len(en) == len(hi), "files are misaligned"

devanagari = re.compile(r"[\u0900-\u097F]")
kept, seen = [], set()
dropped = {"empty": 0, "too_long": 0, "bad_ratio": 0, "not_hindi": 0, "duplicate": 0}

for e, h in zip(en, hi):
    e, h = e.strip(), h.strip()
    if not e or not h:
        dropped["empty"] += 1; continue
    ne, nh = len(e.split()), len(h.split())
    if ne > MAX_WORDS or nh > MAX_WORDS:
        dropped["too_long"] += 1; continue
    if not (MIN_RATIO <= ne / nh <= MAX_RATIO):
        dropped["bad_ratio"] += 1; continue
    if not devanagari.search(h):
        dropped["not_hindi"] += 1; continue
    if (e, h) in seen:
        dropped["duplicate"] += 1; continue
    seen.add((e, h))
    kept.append((e, h))

print(f"Kept {len(kept):,} of {len(en):,} pairs")
print("Dropped:", dropped)

(DATA / "train.clean.en").write_text("\n".join(p[0] for p in kept) + "\n", encoding="utf-8")
(DATA / "train.clean.hi").write_text("\n".join(p[1] for p in kept) + "\n", encoding="utf-8")