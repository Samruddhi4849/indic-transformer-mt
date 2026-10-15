import random
import tarfile
import urllib.request
from pathlib import Path

from datasets import load_dataset

SEED = 42
N_SUBSET = 20_000
DATA = Path("data")
DATA.mkdir(exist_ok=True)

# ---------- 1. IITB English-Hindi (training data) ----------
ds = load_dataset("cfilt/iitb-english-hindi")
print(ds)

train = ds["train"]
print(f"\nTotal training pairs: {len(train):,}")

print("\n--- 10 sample pairs ---")
for i in range(10):
    t = train[i]["translation"]
    print(f"EN: {t['en']}\nHI: {t['hi']}\n")

# ---------- 2. Random 20K subset (fixed seed = reproducible) ----------
rng = random.Random(SEED)
idx = rng.sample(range(len(train)), N_SUBSET)
subset = train.select(idx)

def clean(s):
    return " ".join(s.split())  # remove stray newlines/tabs/extra spaces

en = [clean(x["translation"]["en"]) for x in subset]
hi = [clean(x["translation"]["hi"]) for x in subset]

# ---------- 3. Quick exploration ----------
en_len = sorted(len(s.split()) for s in en)
hi_len = sorted(len(s.split()) for s in hi)

def pct(a, p):
    return a[int(len(a) * p)]

print("--- Subset stats (words per sentence) ---")
print(f"EN: median={pct(en_len, .5)}, 95th pct={pct(en_len, .95)}, max={en_len[-1]}")
print(f"HI: median={pct(hi_len, .5)}, 95th pct={pct(hi_len, .95)}, max={hi_len[-1]}")

empty = sum(1 for a, b in zip(en, hi) if not a or not b)
bad_ratio = sum(
    1 for a, b in zip(en, hi)
    if a and b and not (0.4 <= len(a.split()) / len(b.split()) <= 2.5)
)
print(f"Empty pairs: {empty}")
print(f"Suspicious length ratio (<0.4 or >2.5): {bad_ratio}")

# ---------- 4. Save the subset ----------
(DATA / "train20k.en").write_text("\n".join(en) + "\n", encoding="utf-8")
(DATA / "train20k.hi").write_text("\n".join(hi) + "\n", encoding="utf-8")
print("\nSaved data/train20k.en and data/train20k.hi")

# ---------- 5. FLORES-200 (test set) ----------
URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
tar_path = DATA / "flores200.tar.gz"
if not tar_path.exists():
    print("\nDownloading FLORES-200 (about 25 MB)...")
    urllib.request.urlretrieve(URL, tar_path)

langs = {"eng_Latn": "en", "hin_Deva": "hi", "mar_Deva": "mr"}
with tarfile.open(tar_path) as tar:
    for split in ["dev", "devtest"]:
        for code, short in langs.items():
            member = f"flores200_dataset/{split}/{code}.{split}"
            text = tar.extractfile(member).read().decode("utf-8")
            (DATA / f"flores.{split}.{short}").write_text(text, encoding="utf-8")

print("Saved FLORES files:", sorted(p.name for p in DATA.glob("flores.*")))