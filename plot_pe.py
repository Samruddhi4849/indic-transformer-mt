from pathlib import Path

import matplotlib.pyplot as plt

from model.layers import PositionalEncoding

Path("docs").mkdir(exist_ok=True)
pe = PositionalEncoding(128, dropout=0.0, max_len=100).pe[0]
plt.figure(figsize=(8, 4))
plt.imshow(pe.numpy(), aspect="auto", cmap="RdBu")
plt.xlabel("Embedding dimension")
plt.ylabel("Position")
plt.title("Sinusoidal positional encoding")
plt.colorbar()
plt.savefig("docs/pos_encoding.png", dpi=150, bbox_inches="tight")