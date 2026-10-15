import torch

torch.manual_seed(0)
for d_k in [4, 64, 512]:
    q = torch.randn(10000, d_k)
    k = torch.randn(10000, d_k)
    raw = (q * k).sum(-1)
    print(f"d_k={d_k:4d}  std of q.k: raw={raw.std():.1f}  scaled={(raw / d_k**0.5).std():.1f}")

scores = torch.randn(1, 10) * 8          # what unscaled scores look like at d_k=64
print("softmax max weight, unscaled:", torch.softmax(scores, -1).max().item())
print("softmax max weight, scaled:  ", torch.softmax(scores / 8, -1).max().item())