import random
import re
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

SEED, N = 42, 200_000
MAX_WORDS, MIN_RATIO, MAX_RATIO = 80, 0.4, 2.5
N_TOKENIZER = 20_000   # tokenizer sees only the first 20K pairs, same budget as the pilot
DATA = Path("data")
DATA.mkdir(exist_ok=True)

ds = load_dataset("cfilt/iitb-english-hindi")["train"]
order = list(range(len(ds)))
random.Random(SEED).shuffle(order)
ds = ds.select(order[: int(N * 1.25)])        # a few extra, since cleaning drops some

devanagari = re.compile(r"[\u0900-\u097F]")
kept, seen = [], set()
for row in ds:
    e = " ".join(row["translation"]["en"].split())
    h = " ".join(row["translation"]["hi"].split())
    if not e or not h:
        continue
    ne, nh = len(e.split()), len(h.split())
    if ne > MAX_WORDS or nh > MAX_WORDS or not (MIN_RATIO <= ne / nh <= MAX_RATIO):
        continue
    if not devanagari.search(h) or (e, h) in seen:
        continue
    seen.add((e, h))
    kept.append((e, h))
    if len(kept) == N:
        break
print(f"kept {len(kept):,} pairs")

(DATA / "train200k.en").write_text("\n".join(p[0] for p in kept) + "\n", encoding="utf-8")
(DATA / "train200k.hi").write_text("\n".join(p[1] for p in kept) + "\n", encoding="utf-8")

tok_text = [p[0] for p in kept[:N_TOKENIZER]] + [p[1] for p in kept[:N_TOKENIZER]]
(DATA / "tok_exp.txt").write_text("\n".join(tok_text) + "\n", encoding="utf-8")
spm.SentencePieceTrainer.train(
    input=str(DATA / "tok_exp.txt"), model_prefix=str(DATA / "spm_exp"),
    vocab_size=8000, model_type="bpe", character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)
print("saved data/train200k.{en,hi} and data/spm_exp.model")