"""ST-MAE: Spatio-Temporal Masked Autoencoder for dynamic brain graphs.

Choi et al., "A Generative Self-Supervised Framework using Functional Connectivity
in fMRI Data", Temporal Graph Learning Workshop @ NeurIPS 2023.

ST-MAE shares the encoder of ST-JEMA and differs in the pre-training objective:
  - Random node masking (per timepoint) and random edge masking
  - Reconstruction of the input node embeddings (no EMA target encoder)
  - Spatial (same-timepoint) and temporal (from two other timepoints) reconstruction
    of masked node embeddings and of the binary adjacency
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .stjema import ModelSTAGIN
from .components import EdgeConverter, sample_from_multiple_ranges
from .losses import setup_loss_fn


def _mlp_decoder(hidden_dim, factor):
    return nn.Sequential(
        nn.Linear(hidden_dim, int(hidden_dim * factor)),
        nn.PReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(hidden_dim * factor), hidden_dim),
    )


class ModelSTMAE(ModelSTAGIN):
    """STAGIN encoder with the ST-MAE self-supervised objective.

    In supervised mode (ssl=False) this model is identical to ModelSTAGIN, so
    fine-tuning works the same way for both pre-training methods.
    """

    def _build_ssl_heads(self, cfg, hidden_dim, num_layers, input_dim, *args):
        factor_node = cfg.get('node_decoder_hidden_dim_factor', 2.0)
        factor_edge = cfg.get('edge_decoder_hidden_dim_factor', 2.0)

        # Node reconstruction
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temporal_node_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.node_decoder = _mlp_decoder(hidden_dim, factor_node)
        self.temporal_node_decoder = self.node_decoder  # shared decoder
        self.node_ssl_criterion = setup_loss_fn(cfg.get('node_recon_loss_type', 'sce'), cfg.get('node_alpha_l', 2))

        # Edge reconstruction
        self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_decoder = _mlp_decoder(hidden_dim, factor_edge)
        self.edge_converter = EdgeConverter()
        self.edge_ssl_criterion = setup_loss_fn(cfg.get('edge_recon_loss_type', 'bce'), cfg.get('edge_alpha_l', 2))

    # ======================================================================
    # Adjacency with random edge masking
    # ======================================================================
    def _collate_adjacency_random(self, a, sparsity, mask_rate):
        """Threshold adjacency matrices and randomly drop a fraction of edges.

        Returns the sparse (masked) adjacency used for message passing and the
        binary (unmasked) adjacency used as the reconstruction target.
        """
        i_list, v_list, batch_adj_list = [], [], []
        for sample, _dyn_a in enumerate(a):
            adj_list = []
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                adj_list.append(thresholded_a.long())

                if mask_rate > 0.0:
                    num_node = thresholded_a.shape[0]
                    triu_mask = torch.triu(torch.ones(num_node, num_node), diagonal=1).to(a.device)
                    upper_adj = thresholded_a * triu_mask
                    adj_flatten = upper_adj.reshape(-1)
                    edge_idx = torch.arange(adj_flatten.shape[0]).to(a.device)[adj_flatten == True]
                    num_edge_mask = int(mask_rate * edge_idx.shape[0])
                    mask_idx = edge_idx[torch.randperm(edge_idx.shape[0])[:num_edge_mask]]
                    adj_flatten[mask_idx] = False
                    upper_adj = adj_flatten.reshape(num_node, num_node)
                    thresholded_a = upper_adj + upper_adj.T + torch.diag(torch.ones(num_node)).to(a.device)

                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
            batch_adj_list.append(torch.stack(adj_list))

        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)
        adj_matrix = torch.sparse_coo_tensor(_i, _v, (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))
        binalized_adj = torch.stack(batch_adj_list).to(a.device)
        return adj_matrix, binalized_adj

    def _min_temporal_dist(self):
        cfg = self.cfg
        if cfg.get('min_temporal_dist') is not None:
            return cfg.min_temporal_dist
        return cfg.window_size // cfg.window_stride + 1

    def _sample_pairs(self, time_idx, differences, num_timepoints):
        """Sample (A, B) with A < time_idx < B and B - A = differences."""
        min_As = torch.max(torch.tensor(0), time_idx - torch.div(differences, 2, rounding_mode='floor'))
        max_As = torch.ones(min_As.size()) * time_idx - 1
        As = sample_from_multiple_ranges(min_As, max_As)
        Bs = As + differences
        exceed_mask = Bs > num_timepoints - 1
        exceed_values = Bs[exceed_mask] - (num_timepoints - 1)
        As[exceed_mask] = torch.clamp(As[exceed_mask] - exceed_values, min=0)
        Bs[exceed_mask] = num_timepoints - 1
        return As, Bs

    # ======================================================================
    # Forward
    # ======================================================================
    def forward(self, v, a, t, sampling_endpoints):
        cfg = self.cfg
        if not cfg.get('ssl', False):
            return super().forward(v, a, t, sampling_endpoints)

        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        max_dist = cfg.get('max_temporal_dist') or num_timepoints - 1
        min_dist = self._min_temporal_dist()

        # ========== 1. Input Encoding ==========
        time_encoding = self.timestamp_encoder(t, sampling_endpoints)
        time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)
        h = torch.cat([v, time_encoding], dim=3)
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)

        # ========== 2. Adjacency with random edge masking ==========
        a, a_origin = self._collate_adjacency_random(a, self.sparsity, cfg.get('edge_mask_rate', 0.0))

        # ========== 3. Random node masking with temporal pairs ==========
        h_node_origin = h.clone()  # reconstruction target: input node embeddings
        num_mask_nodes = int(cfg.node_mask_rate * num_nodes)
        total_mask_nodes = torch.zeros(minibatch_size, num_timepoints - 2, num_mask_nodes)
        total_A_nodes = torch.zeros(minibatch_size, num_timepoints - 2, num_mask_nodes)
        total_B_nodes = torch.zeros(minibatch_size, num_timepoints - 2, num_mask_nodes)

        for i in range(minibatch_size):
            for j in range(1, num_timepoints - 1):  # first and last timepoints are not masked
                mask_nodes = torch.randperm(num_nodes)[:num_mask_nodes]
                base = i * num_timepoints * num_nodes + j * num_nodes
                total_mask_nodes[i, j - 1, :] = base + mask_nodes

                differences = torch.randint(min_dist, max_dist + 1, size=(num_mask_nodes,))
                As, Bs = self._sample_pairs(j, differences, num_timepoints)
                total_A_nodes[i, j - 1, :] = i * num_timepoints * num_nodes + As * num_nodes + mask_nodes
                total_B_nodes[i, j - 1, :] = i * num_timepoints * num_nodes + Bs * num_nodes + mask_nodes

        mask_nodes = total_mask_nodes.long().view(-1).to(h.device)
        A_nodes = total_A_nodes.long().view(-1).to(h.device)
        B_nodes = total_B_nodes.long().view(-1).to(h.device)

        h[mask_nodes, :] = 0.0
        h[mask_nodes, :] += self.enc_mask_token

        # ========== 4. GNN Forward ==========
        reg_ortho = torch.tensor(0.0, device=h.device)
        for G in self.gnn_layers:
            h = G(h, a)
        enc_rep = h

        ssl_loss = torch.tensor(0.0, device=h.device)
        ssl_loss_dict = {}
        self_loop_rate = cfg.get('self_loop_rate', 0.5)

        # ========== 5. Node Reconstruction ==========
        cat_rep = torch.cat([enc_rep[A_nodes], enc_rep[B_nodes]], dim=1)
        node_recon_temporal = self.node_decoder(self.temporal_node_projection(cat_rep))
        node_temporal_loss = self.node_ssl_criterion(h_node_origin[mask_nodes], node_recon_temporal)

        node_recon_spatial = self.node_decoder(self.node_projection(enc_rep))[mask_nodes]
        node_spatial_loss = self.node_ssl_criterion(h_node_origin[mask_nodes], node_recon_spatial)

        node_ssl_loss = self_loop_rate * node_spatial_loss + (1 - self_loop_rate) * node_temporal_loss
        ssl_loss_dict.update(node_spatial_ssl_loss=node_spatial_loss.item(),
                             node_temporal_ssl_loss=node_temporal_loss.item(),
                             node_ssl_loss=node_ssl_loss.item())
        ssl_loss = ssl_loss + cfg.get('ssl_lambda_node', 1.0) * node_ssl_loss

        # ========== 6. Edge Reconstruction ==========
        edge_code = self.edge_decoder(self.edge_projection(enc_rep))
        code_bridge = rearrange(edge_code, '(b t n) c -> (b t) n c', b=minibatch_size, t=num_timepoints, n=num_nodes)
        a_origin = a_origin.float()

        total_A_times = torch.zeros(minibatch_size, num_timepoints - 2)
        total_B_times = torch.zeros(minibatch_size, num_timepoints - 2)
        for i in range(minibatch_size):
            differences = torch.randint(min_dist, max_dist + 1, size=(num_timepoints - 2,))
            for j in range(1, num_timepoints - 1):
                As, Bs = self._sample_pairs(j, differences[j - 1], num_timepoints)
                total_A_times[i, j - 1] = i * num_timepoints + As
                total_B_times[i, j - 1] = i * num_timepoints + Bs
        A_times = total_A_times.long().view(-1).to(a.device)
        B_times = total_B_times.long().view(-1).to(a.device)

        a_recon_temporal = self.edge_converter(code_bridge[A_times], code_bridge[B_times])
        edge_temporal_loss = self.edge_ssl_criterion(
            a_recon_temporal, rearrange(a_origin[:, 1:-1], 'b t n c -> (b t) n c'))

        a_recon_spatial = self.edge_converter(code_bridge)
        edge_spatial_loss = self.edge_ssl_criterion(
            a_recon_spatial, rearrange(a_origin, 'b t n c -> (b t) n c'))

        edge_ssl_loss = self_loop_rate * edge_spatial_loss + (1 - self_loop_rate) * edge_temporal_loss
        ssl_loss_dict.update(edge_spatial_ssl_loss=edge_spatial_loss.item(),
                             edge_temporal_ssl_loss=edge_temporal_loss.item(),
                             edge_ssl_loss=edge_ssl_loss.item())
        ssl_loss = ssl_loss + cfg.get('ssl_lambda_edge', 1.0) * edge_ssl_loss

        attention = {'node-attention': [], 'time-attention': []}
        return 0.0, attention, None, reg_ortho, ssl_loss, ssl_loss_dict
