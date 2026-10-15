import torch


def causal_mask(size):
    """(size, size) lower-triangular: position i may attend to positions <= i."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool))


def padding_mask(tokens, pad_id=0):
    """(batch, seq) token ids -> (batch, 1, 1, seq); False where the token is padding."""
    return (tokens != pad_id).unsqueeze(1).unsqueeze(2)