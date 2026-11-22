"""Dynamic Functional Connectivity computation from BOLD signals."""

import torch
from random import randrange


def corrcoef(x):
    """Pearson correlation coefficient matrix.

    Reference: https://github.com/pytorch/pytorch/issues/1254
    """
    eps = torch.finfo(x.dtype).eps
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    stddev = torch.where(stddev < eps, torch.full_like(stddev, eps), stddev)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def get_fc(timeseries, sampling_point, window_size, self_loop):
    fc = corrcoef(timeseries[sampling_point:sampling_point + window_size].T)
    if not self_loop:
        fc -= torch.eye(fc.shape[0])
    return fc


def get_minibatch_fc(minibatch_timeseries, sampling_point, window_size, self_loop):
    fc_list = [get_fc(ts, sampling_point, window_size, self_loop) for ts in minibatch_timeseries]
    return torch.stack(fc_list)


def process_dynamic_fc(minibatch_timeseries, window_size, window_stride,
                       dynamic_length=None, sampling_init=None, self_loop=True):
    """Compute dynamic functional connectivity using sliding windows.

    Args:
        minibatch_timeseries: [B, T, N] BOLD signal tensor
        window_size: Sliding window size for FC computation
        window_stride: Step size between windows
        dynamic_length: Total time length to use (random crop if shorter)
        sampling_init: Starting point (random if None)
        self_loop: Whether to include self-loops in FC

    Returns:
        dynamic_fc: [B, W, N, N] dynamic FC matrices
        sampling_points: List of window start indices
    """
    if dynamic_length is None:
        dynamic_length = minibatch_timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert minibatch_timeseries.shape[1] >= sampling_init + dynamic_length

    assert sampling_init is None or isinstance(sampling_init, int)
    assert minibatch_timeseries.ndim == 3
    assert dynamic_length >= window_size

    if sampling_init is None:
        sampling_init = randrange(minibatch_timeseries.shape[1] - dynamic_length + 1)

    sampling_points = list(range(sampling_init, sampling_init + dynamic_length - window_size + 1, window_stride))
    minibatch_fc_list = [get_minibatch_fc(minibatch_timeseries, sp, window_size, self_loop) for sp in sampling_points]
    dynamic_fc = torch.stack(minibatch_fc_list, dim=1)

    return dynamic_fc, sampling_points
