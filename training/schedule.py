def noam(step, d_model, warmup, factor=1.0):
    """Paper Eq. 3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)"""
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)