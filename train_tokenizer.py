import sentencepiece as spm
from pathlib import Path

DATA = Path("data")

# One shared vocabulary over both languages (as the paper does for EN-DE)
combined = DATA / "tok_train.txt"
combined.write_text(
    (DATA / "train.clean.en").read_text(encoding="utf-8")
    + (DATA / "train.clean.hi").read_text(encoding="utf-8"),
    encoding="utf-8",
)

spm.SentencePieceTrainer.train(
    input=str(combined),
    model_prefix=str(DATA / "spm"),
    vocab_size=8000,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
)

sp = spm.SentencePieceProcessor(model_file=str(DATA / "spm.model"))
for line in ["How are you today?", "आप आज कैसे हैं?"]:
    ids = sp.encode(line)
    print(line, "->", sp.encode(line, out_type=str), ids)
    print("   decoded:", sp.decode(ids))

# Fertility = tokens per word (lower is better)
for lang in ["en", "hi"]:
    lines = (DATA / f"train.clean.{lang}").read_text(encoding="utf-8").splitlines()[:2000]
    tokens = sum(len(sp.encode(l)) for l in lines)
    words = sum(len(l.split()) for l in lines)
    print(f"{lang} fertility: {tokens / words:.2f} tokens/word")