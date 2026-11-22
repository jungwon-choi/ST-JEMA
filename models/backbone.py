"""STAGIN backbone modules: GNN layers, readout, and temporal aggregation."""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Optional: torch_scatter is only needed for GAT encoder
try:
    from torch_scatter import scatter_softmax, scatter_add
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


# ==============================================================================
# Temporal Encoding
# ==============================================================================

class ModuleTimestamping(nn.Module):
    """GRU-based temporal encoding of BOLD signals."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p - 1 for p in sampling_endpoints]]


# ==============================================================================
# GNN Layers
# ==============================================================================

class LayerGIN(nn.Module):
    """Graph Isomorphism Network layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon:
            self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))
        else:
            self.epsilon = 0.0
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(),
        )

    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v
        v_combine = self.mlp(v_aggregate)
        return v_combine


class LayerGAT(nn.Module):
    """Graph Attention Network layer with sparse adjacency support.

    Requires torch_scatter for efficient sparse operations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1, dropout=0.1, concat=False):
        super().__init__()
        if not HAS_TORCH_SCATTER:
            raise ImportError("torch_scatter is required for GAT encoder. Install with: pip install torch_scatter")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)

        self.attn = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim * 2))
        nn.init.xavier_uniform_(self.attn)

        if concat:
            self.out_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.out_proj = nn.Linear(self.head_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, v, a):
        num_nodes = v.size(0)
        device = v.device

        q = self.W_q(v).view(num_nodes, self.num_heads, self.head_dim)
        k = self.W_k(v).view(num_nodes, self.num_heads, self.head_dim)
        value = self.W_v(v).view(num_nodes, self.num_heads, self.head_dim)

        if a.is_sparse:
            edge_index = a._indices()
        else:
            edge_index = a.nonzero().t()

        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        num_edges = edge_index.size(1)

        if num_edges == 0:
            if self.concat:
                out = value.reshape(num_nodes, -1)
            else:
                out = value.mean(dim=1)
            out = self.out_proj(out)
            out = self.layer_norm(out)
            return out

        q_edge = q[src_idx]
        k_edge = k[dst_idx]
        qk_cat = torch.cat([q_edge, k_edge], dim=-1)

        attn_logits = (qk_cat * self.attn.squeeze(0)).sum(dim=-1)
        attn_logits = self.leaky_relu(attn_logits)

        attn_weights = scatter_softmax(attn_logits, src_idx, dim=0)
        attn_weights = self.dropout(attn_weights)

        value_edge = value[dst_idx]
        weighted_value = value_edge * attn_weights.unsqueeze(-1)

        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=device)
        out = scatter_add(weighted_value, src_idx, dim=0, out=out)

        if self.concat:
            out = out.reshape(num_nodes, -1)
        else:
            out = out.mean(dim=1)

        out = self.out_proj(out)
        out = self.layer_norm(out)
        return out


# ==============================================================================
# Readout Modules
# ==============================================================================

class ModuleMeanReadout(nn.Module):
    """Simple mean readout over the node axis."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    """Squeeze-and-Excitation Readout: attention-weighted node pooling."""
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, round(upscale * hidden_dim)),
            nn.BatchNorm1d(round(upscale * hidden_dim)),
            nn.GELU(),
        )
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1], -1)
        permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(range(node_axis, len(x_graphattention.shape) - 1))
        x_graphattention = x_graphattention.permute(permute_idx)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleGARO(nn.Module):
    """Graph Attention Readout: query-key attention-based node pooling."""
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale * hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(
            torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n')) / np.sqrt(x_q.shape[-1])
        ).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


# ==============================================================================
# Temporal Aggregation
# ==============================================================================

class ModuleTransformer(nn.Module):
    """Transformer for temporal aggregation over graph readout sequence."""
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # x: [T, B, D]
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix
