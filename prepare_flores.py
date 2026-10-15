import tarfile
from pathlib import Path

DATA = Path("data")
tar_path = DATA / "flores200.tar.gz"
langs = {"eng_Latn": "en", "hin_Deva": "hi", "mar_Deva": "mr"}

with tarfile.open(tar_path) as tar:
    names = tar.getnames()
    print("First 8 entries in archive:", names[:8])

    for split in ["dev", "devtest"]:
        for code, short in langs.items():
            suffix = f"/{split}/{code}.{split}"
            match = [n for n in names if n.endswith(suffix)]
            if not match:
                print(f"NOT FOUND: {suffix}")
                continue
            text = tar.extractfile(match[0]).read().decode("utf-8")
            (DATA / f"flores.{split}.{short}").write_text(text, encoding="utf-8")
            print(f"Saved flores.{split}.{short} ({len(text.splitlines())} lines)")