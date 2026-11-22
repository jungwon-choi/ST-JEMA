import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def sce_loss(x, y, alpha=3):
    """Scaled cosine error (SCE) loss."""
    if x.dtype != torch.float32:
        x = x.float()
    if y.dtype != torch.float32:
        y = y.float()
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def cos_loss(x, y):
    """Cosine similarity loss."""
    loss = 1 - F.cosine_similarity(x, y, dim=-1)
    return loss.mean()


def setup_loss_fn(loss_fn, alpha_l):
    """Factory function for SSL loss functions."""
    if loss_fn == "mse":
        return nn.MSELoss()
    elif loss_fn == "bce":
        return nn.BCELoss()
    elif loss_fn == "sce":
        return partial(sce_loss, alpha=alpha_l)
    elif loss_fn == "cos":
        return cos_loss
    else:
        raise NotImplementedError(f"Loss function '{loss_fn}' not supported.")
