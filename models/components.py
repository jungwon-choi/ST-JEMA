import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def sample_from_multiple_ranges(min_values, max_values):
    """Sample integers uniformly from per-element [min, max] ranges."""
    assert min_values.size() == max_values.size()
    random_uniform = torch.rand(min_values.size())
    scaled_values = random_uniform * (max_values - min_values + 1) + min_values
    return torch.floor(scaled_values).long()


class EdgeConverter(nn.Module):
    """Convert node features to edge features via dot-product similarity.

    Input:  [B, N, C] node features
    Output: [B, N, N] edge features (sigmoid of pairwise dot products)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y=None):
        if y is not None:
            edge1 = torch.sigmoid(torch.matmul(x, y.transpose(1, 2)))
            edge2 = torch.sigmoid(torch.matmul(y, x.transpose(1, 2)))
            return (edge1 + edge2) * 0.5
        return torch.sigmoid(torch.matmul(x, x.transpose(1, 2)))


# MLP-Mixer decoder
# Reference: https://github.com/lucidrains/mlp-mixer-pytorch
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, transpose=False):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.transpose = transpose
        if transpose:
            self.bf = Rearrange('b n d -> b d n')
            self.af = Rearrange('b d n -> b n d')

    def forward(self, x):
        if self.transpose:
            return self.af(self.fn(self.bf(self.norm(x)))) + x
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class MLP_Mixer(nn.Module):
    """MLP-Mixer decoder for node/edge reconstruction."""
    def __init__(self, dim, num_patches, expansion_factor=4, expansion_factor_token=0.5, num_layers=1, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(PreNormResidual(dim, FeedForward(num_patches, expansion_factor=expansion_factor, dropout=dropout), transpose=True))
            self.layers.append(PreNormResidual(dim, FeedForward(dim, expansion_factor=expansion_factor_token, dropout=dropout)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Percentile(torch.autograd.Function):
    """Percentile computation for adjacency thresholding.

    Reference: https://github.com/aliutkus/torchpercentile
    """
    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input)
        input_dtype = input.dtype
        input_shape = input.shape

        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.as_tensor(percentiles, dtype=torch.double)

        input = input.double().view(input.shape[0], -1)
        percentiles = percentiles.to(input.device).double()

        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1

        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled

        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]

        self.save_for_backward(input_shape, in_argsort, floored.long(), ceiled.long(), weight_floored, weight_ceiled)

        result = (d0 + d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        cols_offsets = torch.arange(0, input_shape[1], device=in_argsort.device)[None, :].long()
        in_argsort = (in_argsort * input_shape[1] + cols_offsets).view(-1).long()
        floored = (floored[:, None] * input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (ceiled[:, None] * input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=in_argsort.device)
        grad_input[in_argsort[floored]] += (grad_output * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input
