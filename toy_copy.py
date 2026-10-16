import torch
import torch.nn.functional as F

from model.transformer import Transformer

PAD, BOS, EOS = 0, 1, 2
VOCAB, MAXLEN = 20, 10
torch.manual_seed(0)


def make_batch(n):
    L = MAXLEN + 1
    src = torch.full((n, L), PAD)
    tgt_in = torch.full((n, L), PAD)
    tgt_out = torch.full((n, L), PAD)
    for i in range(n):
        k = torch.randint(3, MAXLEN + 1, ()).item()
        s = torch.randint(3, VOCAB, (k,))
        src[i, :k], src[i, k] = s, EOS
        tgt_in[i, 0], tgt_in[i, 1:k + 1] = BOS, s
        tgt_out[i, :k], tgt_out[i, k] = s, EOS
    return src, tgt_in, tgt_out


@torch.no_grad()
def greedy(model, src):
    memory, src_mask = model.encode(src)
    ys = torch.full((src.size(0), 1), BOS)
    for _ in range(MAXLEN + 1):
        nxt = model.decode(ys, memory, src_mask)[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
    return ys[:, 1:]


model = Transformer(VOCAB, d_model=64, num_layers=2, num_heads=4, d_ff=128, dropout=0.1)
opt = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)

for step in range(1, 2001):
    src, tgt_in, tgt_out = make_batch(64)
    logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.view(-1, VOCAB), tgt_out.view(-1), ignore_index=PAD)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

model.eval()
src, _, tgt_out = make_batch(500)
pred = greedy(model, src)
correct = 0
for i in range(500):
    n = int((tgt_out[i] != PAD).sum())
    correct += int(torch.equal(pred[i, :n], tgt_out[i, :n]))
print(f"Exact-match accuracy on 500 fresh sequences: {correct / 500:.1%}")
print("example:", src[0].tolist(), "->", pred[0].tolist())