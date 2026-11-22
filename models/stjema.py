"""ST-JEMA: Spatio-Temporal Joint Embedding Masked Autoencoder for brain fMRI.

This model implements JEPA-style self-supervised learning on dynamic brain graphs:
  - Online encoder processes masked input
  - EMA target encoder provides latent reconstruction targets
  - Node + Edge reconstruction with spatial and temporal objectives
  - Block-based masking strategy for multi-scale prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from einops import rearrange, repeat

from .backbone import (
    ModuleTimestamping, LayerGIN, LayerGAT,
    ModuleSERO, ModuleGARO, ModuleMeanReadout, ModuleTransformer,
)
from .components import EdgeConverter, MLP_Mixer, Percentile, sample_from_multiple_ranges
from .losses import setup_loss_fn


class ModelSTAGIN(nn.Module):
    """STAGIN with ST-JEMA self-supervised learning.

    Supports two modes:
      - SSL (pretrain): Block masking + node/edge reconstruction with EMA targets
      - Supervised (finetune): Standard classification/regression with pretrained encoder

    Args:
        input_dim: Number of input node features (ROIs)
        hidden_dim: Hidden dimension for all layers
        num_classes: Number of output classes (1 for regression)
        num_heads: Number of attention heads for readout
        num_layers: Number of GNN layers
        sparsity: Top-k percentage for adjacency thresholding
        dropout: Dropout rate
        cls_token: Temporal aggregation method ('sum', 'mean', 'param')
        readout: Spatial readout method ('sero', 'garo', 'mean')
        cfg: Configuration dict with all hyperparameters
    """
    def __init__(
        self,
        input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity,
        dropout=0.5, cls_token='sum', readout='sero', cfg=None,
    ):
        super().__init__()

        # Temporal aggregation token
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise ValueError(f"cls_token must be one of ['sum', 'mean', 'param'], got '{cls_token}'")

        # Readout module selection
        readout_module = {'garo': ModuleGARO, 'sero': ModuleSERO, 'mean': ModuleMeanReadout}[readout]

        self.cfg = cfg
        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token == 'param' else None
        self.num_classes = num_classes
        self.sparsity = sparsity
        self.num_layers = num_layers
        self.time_aggregation = cfg.time_aggregation
        self.percentile = Percentile()

        # ========== Temporal Encoding ==========
        if not cfg.get('not_use_time_encoding', False):
            self.timestamp_encoder = ModuleTimestamping(input_dim, hidden_dim, hidden_dim)

        # ========== Initial Projection ==========
        if input_dim != hidden_dim:
            if (cfg.get('static', False) and not cfg.get('use_time_encoding', False)) or cfg.get('not_use_time_encoding', False):
                self.initial_linear = nn.Linear(input_dim, hidden_dim)
            else:
                self.initial_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        else:
            self.initial_linear = nn.Identity()

        # ========== Encoder Layers ==========
        encoder_type = cfg.get('encoder_type', 'gin')
        gat_num_heads = cfg.get('gat_num_heads', 1)
        gat_dropout = cfg.get('gat_dropout', 0.1)

        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.penultimate_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(num_layers):
            # GNN layer
            if encoder_type == 'gat':
                self.gnn_layers.append(LayerGAT(hidden_dim, hidden_dim, hidden_dim, num_heads=gat_num_heads, dropout=gat_dropout))
            else:
                self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))

            # Readout
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))

            # Temporal aggregation
            if self.time_aggregation == 'mean':
                self.penultimate_layers.append(nn.Identity())
            elif self.time_aggregation == 'gru':
                self.penultimate_layers.append(nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=False))
            elif self.time_aggregation == 'transformer':
                self.penultimate_layers.append(ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.0))
            else:
                raise NotImplementedError(f"time_aggregation '{self.time_aggregation}' not supported")

            # Classification head
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

        self.current_step = 0
        self.apply_edge_loss = True

        # ========== SSL Components (only if ssl=True) ==========
        if cfg.get('ssl', False):
            self._build_ssl_heads(cfg, hidden_dim, num_layers, input_dim, encoder_type, gat_num_heads, gat_dropout)

    def _build_ssl_heads(self, cfg, hidden_dim, num_layers, input_dim, encoder_type, gat_num_heads, gat_dropout):
        """Build SSL-specific modules for node and edge reconstruction."""
        self.num_pred_masks = cfg.num_pred_masks
        self.pred_mask_scale = cfg.pred_mask_scale

        # ---------- Node Reconstruction ----------
        if cfg.get('recon_node', False):
            self.enc_node_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
            self.node_mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.ssl_lambda_node = cfg.ssl_lambda_node
            self.node_mask_rate = cfg.node_mask_rate
            self._node_replace_rate = cfg.get('node_replace_rate', 0.0)
            self._node_mask_rate = 1 - self._node_replace_rate

            # Projection layers
            if cfg.get('add_projection_layer', True):
                proj_output_dim = hidden_dim
                initial_hidden_dim = hidden_dim

                if cfg.get('temporal_node_recon', False) and cfg.get('temporal_node_concat_type', 'feat') == 'feat':
                    initial_hidden_dim += hidden_dim

                self.node_projection = nn.Linear(hidden_dim, proj_output_dim, bias=False)
                if cfg.get('temporal_node_recon', False) and cfg.get('temporal_node_concat_type', 'feat') == 'feat':
                    self.temporal_node_projection = nn.Linear(initial_hidden_dim, proj_output_dim, bias=False)
                else:
                    self.temporal_node_projection = self.node_projection
            else:
                self.node_projection = nn.Identity()

            # Node decoder
            if cfg.get('add_node_decoder', True):
                decoder_type = cfg.get('node_decoder_type', 'mlp-mixer')
                if decoder_type == 'gin':
                    self.node_decoder = LayerGIN(hidden_dim, hidden_dim, hidden_dim)
                elif decoder_type == 'gat':
                    self.node_decoder = LayerGAT(hidden_dim, hidden_dim, hidden_dim, num_heads=gat_num_heads, dropout=gat_dropout)
                elif decoder_type == 'mlp':
                    factor = cfg.get('node_decoder_hidden_dim_factor', 2.0)
                    self.node_decoder = nn.Sequential(
                        nn.Linear(hidden_dim, int(hidden_dim * factor)), nn.PReLU(),
                        nn.Dropout(0.2), nn.Linear(int(hidden_dim * factor), hidden_dim),
                    )
                elif decoder_type == 'mlp-mixer':
                    self.node_decoder = MLP_Mixer(hidden_dim, num_patches=input_dim, expansion_factor=4, num_layers=1)
                else:
                    raise NotImplementedError(f"node_decoder_type '{decoder_type}' not supported")
                self.temporal_node_decoder = self.node_decoder

            self.node_ssl_criterion = setup_loss_fn(cfg.get('node_recon_loss_type', 'mse'), cfg.get('node_alpha_l', 2))

        # ---------- Edge Reconstruction ----------
        if cfg.get('recon_edge', False):
            self.enc_edge_mask_token = nn.Parameter(torch.randn(1))
            self.edge_mask_token = nn.Parameter(torch.randn(1, hidden_dim))
            self.ssl_lambda_edge = cfg.ssl_lambda_edge

            if cfg.get('add_projection_layer', True):
                self.edge_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            else:
                self.edge_projection = nn.Identity()

            if cfg.get('add_edge_decoder', True):
                decoder_type = cfg.get('edge_decoder_type', 'mlp-mixer')
                if decoder_type == 'gin':
                    self.edge_decoder = LayerGIN(hidden_dim, hidden_dim, hidden_dim)
                elif decoder_type == 'gat':
                    self.edge_decoder = LayerGAT(hidden_dim, hidden_dim, hidden_dim, num_heads=gat_num_heads, dropout=gat_dropout)
                elif decoder_type == 'mlp':
                    factor = cfg.get('edge_decoder_hidden_dim_factor', 2.0)
                    self.edge_decoder = nn.Sequential(
                        nn.Linear(hidden_dim, int(hidden_dim * factor)), nn.PReLU(),
                        nn.Dropout(0.2), nn.Linear(int(hidden_dim * factor), hidden_dim),
                    )
                elif decoder_type == 'mlp-mixer':
                    self.edge_decoder = MLP_Mixer(hidden_dim, num_patches=input_dim, expansion_factor=4, num_layers=1)
                else:
                    raise NotImplementedError(f"edge_decoder_type '{decoder_type}' not supported")

            self.edge_converter = EdgeConverter()
            self.edge_ssl_criterion = setup_loss_fn(cfg.get('edge_recon_loss_type', 'bce'), cfg.get('edge_alpha_l', 2))

    # ======================================================================
    # Adjacency Processing
    # ======================================================================
    def _collate_adjacency(self, a, sparsity, mask_rate=0.0, get_unmasked=False, block_masks=None):
        """Threshold adjacency matrices and apply edge block masking.

        Args:
            a: [B, T, N, N] raw adjacency matrices
            sparsity: Top-k percentage to keep
            mask_rate: Random edge masking rate
            get_unmasked: Whether to return unmasked adjacency
            block_masks: Pre-computed block mask indices for edge masking

        Returns:
            adj_matrix: Sparse adjacency for GNN forward
            binalized_adj: [B, T, N, N] binary adjacency (for edge target)
            unmasked_adj_matrix: Unmasked sparse adjacency (for EMA target)
            block_masked_a_indices: Indices of masked edge blocks
        """
        i_list, v_list = [], []
        unmasked_i_list, unmasked_v_list = [], []
        num_timepoints = a.shape[1]
        batch_adj_list = []

        edge_block_mask = self.cfg.get('edge_block_mask', False)
        if edge_block_mask:
            batch_block_masked_a_indices = [[] for _ in range(self.num_pred_masks)]
            all_indices = torch.arange(a.shape[2] * a.shape[3]).reshape(a.shape[2], a.shape[3])
            min_width = int(a.shape[2] * self.pred_mask_scale[0])
            max_width = int(a.shape[2] * self.pred_mask_scale[1])

        for sample, _dyn_a in enumerate(a):
            adj_list = []
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100 - sparsity))
                adj_list.append(thresholded_a.long())

                if get_unmasked:
                    u_i = thresholded_a.nonzero(as_tuple=False)
                    u_v = torch.ones(len(u_i))
                    u_i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                    unmasked_i_list.append(u_i)
                    unmasked_v_list.append(u_v)

                # Edge block masking
                if edge_block_mask:
                    if timepoint != 0 and timepoint != num_timepoints - 1:
                        for g in range(self.num_pred_masks):
                            edge_random_block = self.cfg.get('edge_random_block', True)
                            if not edge_random_block and block_masks is not None:
                                block_masks_per_time = block_masks[g][sample][timepoint - 1]
                                s_i, e_i = block_masks_per_time[0], block_masks_per_time[-1]
                                block_idx = all_indices[s_i:e_i + 1, s_i:e_i + 1].reshape(-1)
                                thresholded_a[s_i:e_i + 1, s_i:e_i + 1] = 0
                            else:
                                block_width = torch.randint(min_width, max_width, (1,)).item()
                                start_x = torch.randint(0, a.shape[2] - block_width + 1, (1,)).item()
                                start_y = torch.randint(0, a.shape[2] - block_width + 1, (1,)).item()
                                block_idx = all_indices[start_x:start_x + block_width, start_y:start_y + block_width].reshape(-1)
                                thresholded_a[start_x:start_x + block_width, start_y:start_y + block_width] = 0

                            offset = sample * (a.shape[1] - 2) * a.shape[2] * a.shape[3] + (timepoint - 1) * a.shape[2] * a.shape[3]
                            batch_block_masked_a_indices[g].append(offset + block_idx)

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

        if get_unmasked:
            _u_i = torch.cat(unmasked_i_list).T.to(a.device)
            _u_v = torch.cat(unmasked_v_list).to(a.device)
            unmasked_adj_matrix = torch.sparse_coo_tensor(_u_i, _u_v, adj_matrix.shape)
        else:
            unmasked_adj_matrix = None

        if edge_block_mask:
            block_masked_a_indices = [torch.cat(b).to(a.device) for b in batch_block_masked_a_indices]
        else:
            block_masked_a_indices = None

        return adj_matrix, binalized_adj, unmasked_adj_matrix, block_masked_a_indices

    # ======================================================================
    # Masking
    # ======================================================================
    def _create_masks(self, h, minibatch_size, num_timepoints, num_nodes):
        """Create block masks for nodes and temporal pairs (A, B).

        Returns dict with keys: block_mask_nodes, total_mask_nodes,
        block_A_nodes, block_B_nodes, block_A_times, block_B_times, block_masks
        """
        cfg = self.cfg
        min_nodes = int(num_nodes * self.pred_mask_scale[0])
        max_nodes = int(num_nodes * self.pred_mask_scale[1])
        num_mask_nodes = int(self.node_mask_rate * num_nodes)
        min_temporal_dist = cfg.get('min_temporal_dist', None)
        if min_temporal_dist is None:
            min_temporal_dist = math.ceil(cfg.window_size / cfg.window_stride)

        block_mask_nodes = [torch.empty(0) for _ in range(self.num_pred_masks)]
        block_A_nodes = [torch.empty(0) for _ in range(self.num_pred_masks)]
        block_B_nodes = [torch.empty(0) for _ in range(self.num_pred_masks)]
        block_A_times = [torch.empty(0) for _ in range(self.num_pred_masks)]
        block_B_times = [torch.empty(0) for _ in range(self.num_pred_masks)]
        temporal_node_valid = [torch.empty(0, dtype=torch.bool) for _ in range(self.num_pred_masks)]
        temporal_time_valid = [torch.empty(0, dtype=torch.bool) for _ in range(self.num_pred_masks)]

        # Edge block masks (for non-random edge block masking)
        edge_block_mask = cfg.get('edge_block_mask', False) and not cfg.get('edge_random_block', True)
        block_masks = [[[] for _ in range(minibatch_size)] for _ in range(self.num_pred_masks)] if edge_block_mask else None

        for g in range(self.num_pred_masks):
            batch_mask_list, batch_A_list, batch_B_list = [], [], []
            batch_A_times_list, batch_B_times_list = [], []
            batch_node_valid_list, batch_time_valid_list = [], []

            for i in range(minibatch_size):
                block_masks_per_time = []

                for j in range(num_timepoints):
                    if j == 0 or j == num_timepoints - 1:
                        continue  # Skip first and last timepoints

                    # Block mask for nodes
                    if cfg.get('recon_node', False):
                        block_size = torch.randint(min_nodes, max_nodes, (1,)).item()
                        start = torch.randint(0, num_nodes - block_size + 1, (1,)).item()
                        mask_nodes = torch.arange(start, start + block_size)

                        if edge_block_mask:
                            block_masks_per_time.append(mask_nodes)

                        base_idx = i * num_timepoints * num_nodes + j * num_nodes
                        batch_mask_list.append(base_idx + mask_nodes)

                    # Temporal pairs (A, B) for temporal reconstruction
                    if cfg.get('temporal_node_recon', False) or cfg.get('temporal_edge_recon', False):
                        # Context windows of A and B must not overlap the window of the target j
                        valid = (j - min_temporal_dist >= 0) and (j + min_temporal_dist <= num_timepoints - 1)
                        if valid:
                            A = torch.randint(0, j - min_temporal_dist + 1, (1,)).item()
                            B = torch.randint(j + min_temporal_dist, num_timepoints, (1,)).item()
                        else:
                            A, B = 0, num_timepoints - 1  # placeholder, excluded from the loss

                        if cfg.get('recon_node', False) and cfg.get('temporal_node_recon', False):
                            base_A = i * num_timepoints * num_nodes + A * num_nodes
                            base_B = i * num_timepoints * num_nodes + B * num_nodes
                            batch_A_list.append(base_A + mask_nodes)
                            batch_B_list.append(base_B + mask_nodes)
                            batch_node_valid_list.append(torch.full((len(mask_nodes),), valid, dtype=torch.bool))

                        if cfg.get('recon_edge', False) and cfg.get('temporal_edge_recon', False):
                            batch_A_times_list.append(torch.tensor([i * num_timepoints + A]))
                            batch_B_times_list.append(torch.tensor([i * num_timepoints + B]))
                            batch_time_valid_list.append(torch.tensor([valid]))

                if edge_block_mask:
                    block_masks[g][i] = block_masks_per_time

            if batch_mask_list:
                block_mask_nodes[g] = torch.cat(batch_mask_list, dim=0)
            if batch_A_list:
                block_A_nodes[g] = torch.cat(batch_A_list, dim=0)
                block_B_nodes[g] = torch.cat(batch_B_list, dim=0)
                temporal_node_valid[g] = torch.cat(batch_node_valid_list, dim=0).to(h.device)
            if batch_A_times_list:
                block_A_times[g] = torch.cat(batch_A_times_list, dim=0)
                block_B_times[g] = torch.cat(batch_B_times_list, dim=0)
                temporal_time_valid[g] = torch.cat(batch_time_valid_list, dim=0).to(h.device)

        # Compute total mask (union of all pred masks)
        if cfg.get('recon_node', False):
            block_mask_nodes = [bmn.long().view(-1).to(h.device) for bmn in block_mask_nodes]
            total_mask_nodes = torch.cat(block_mask_nodes, dim=0).unique()
            total_mask_nodes = total_mask_nodes.long().to(h.device)
            if cfg.get('temporal_node_recon', False):
                block_A_nodes = [bn.long().view(-1).to(h.device) for bn in block_A_nodes]
                block_B_nodes = [bn.long().view(-1).to(h.device) for bn in block_B_nodes]
        else:
            total_mask_nodes = None

        if cfg.get('temporal_edge_recon', False):
            block_A_times = [bt.long().view(-1).to(h.device) for bt in block_A_times]
            block_B_times = [bt.long().view(-1).to(h.device) for bt in block_B_times]

        return {
            'block_mask_nodes': block_mask_nodes,
            'total_mask_nodes': total_mask_nodes,
            'block_A_nodes': block_A_nodes,
            'block_B_nodes': block_B_nodes,
            'block_A_times': block_A_times,
            'block_B_times': block_B_times,
            'temporal_node_valid': temporal_node_valid,
            'temporal_time_valid': temporal_time_valid,
            'block_masks': block_masks,
        }

    # ======================================================================
    # Forward Pass
    # ======================================================================
    def forward(self, v, a, t, sampling_endpoints):
        """Forward pass for both SSL (pretrain) and supervised (finetune) modes.

        Args:
            v: [B, T, N, N] identity node features
            a: [B, T, N, N] dynamic adjacency matrices
            t: [T, B, N] BOLD timeseries for temporal encoding
            sampling_endpoints: List of time indices for temporal encoding

        Returns (SSL mode):
            logit, attention, latent, reg_ortho, ssl_loss, ssl_loss_dict
        Returns (Supervised mode):
            logit, attention, latent, reg_ortho
        """
        cfg = self.cfg
        minibatch_size, num_timepoints, num_nodes = a.shape[:3]
        ema_update = cfg.get('ema_update', False) and cfg.get('ssl', False)

        # ========== 1. Input Encoding ==========
        if cfg.get('static', False) and not cfg.get('use_time_encoding', False):
            h = rearrange(v, 'b t n c -> (b t n) c')
            if ema_update:
                target_h = self.target_model.initial_linear(h.clone())
            h = self.initial_linear(h)
        else:
            if cfg.get('not_use_time_encoding', False):
                h = v
            else:
                time_encoding = self.timestamp_encoder(t, sampling_endpoints)
                time_encoding = repeat(time_encoding, 'b t c -> t b n c', n=num_nodes)
                if ema_update:
                    target_time_encoding = self.target_model.timestamp_encoder(t, sampling_endpoints)
                    target_time_encoding = repeat(target_time_encoding, 'b t c -> t b n c', n=num_nodes)
                    target_h = torch.cat([v, target_time_encoding], dim=3)
                h = torch.cat([v, time_encoding], dim=3)

            h = rearrange(h, 'b t n c -> (b t n) c')
            h = self.initial_linear(h)
            if ema_update:
                target_h = rearrange(target_h, 'b t n c -> (b t n) c')
                target_h = self.target_model.initial_linear(target_h)

        # ========== 2. SSL Masking ==========
        masks = None
        block_masks_for_edge = []
        if cfg.get('ssl', False):
            if ema_update:
                h_origin = h.clone()

            if cfg.get('recon_node', False):
                h_node_origin = h.clone()

            masks = self._create_masks(h, minibatch_size, num_timepoints, num_nodes)
            block_masks_for_edge = masks['block_masks'] if masks['block_masks'] is not None else []

            # Apply node masking
            if masks['total_mask_nodes'] is not None:
                h[masks['total_mask_nodes'], :] = 0.0

        # ========== 3. Adjacency Processing ==========
        a, a_origin, unmasked_a, block_masked_a_indices = self._collate_adjacency(
            a, self.sparsity,
            mask_rate=cfg.get('edge_mask_rate', 0.0),
            get_unmasked=ema_update,
            block_masks=block_masks_for_edge,
        )

        if cfg.get('ssl', False) and not self.apply_edge_loss and unmasked_a is not None:
            a = unmasked_a.clone()

        # ========== 4. GNN Forward ==========
        logit = 0.0
        reg_ortho = torch.tensor(0.0, device=h.device)
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        node_features = []

        if ema_update:
            target_h = h_origin if ema_update else None
            target_node_features = []

        for layer, (G, R, T_layer, L) in enumerate(zip(
            self.gnn_layers, self.readout_modules, self.penultimate_layers, self.linear_layers
        )):
            # GNN layer
            h = G(h, a)

            # EMA target encoder (no gradient)
            if ema_update and cfg.get('target_model_type', 'latent') == 'latent':
                with torch.no_grad():
                    target_h = self.target_model.gnn_layers[layer](target_h, unmasked_a)

            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)

            # Orthogonal regularization
            if cfg.get('reg_lambda', 0.0) > 0.0:
                ortho = rearrange(h_bridge, 't b n c -> (t b) n c')
                mat_inner = torch.bmm(ortho, ortho.permute(0, 2, 1))
                reg_ortho += (mat_inner / (mat_inner.max(-1)[0].unsqueeze(-1) + 1e-8) - torch.eye(num_nodes, device=mat_inner.device)).triu().add(1e-8).norm(dim=(1, 2)).mean()

            # Collect node features for SSL
            if cfg.get('ssl', False) and (cfg.get('recon_node', False) or cfg.get('recon_edge', False)):
                node_features.append(h.clone())
                if ema_update and cfg.get('target_model_type', 'latent') == 'latent':
                    target_node_features.append(target_h.clone())
                continue  # Skip readout during SSL (only need node features)

            # Readout + temporal aggregation (for supervised mode)
            h_readout, node_attn = R(h_bridge, node_axis=2)

            if self.time_aggregation == 'mean':
                h_mean = h_readout.mean(dim=0)
                latent = T_layer(h_mean)
            elif self.time_aggregation == 'transformer':
                h_temporal, time_attn = T_layer(h_readout)
                latent = self.cls_token(h_temporal)
                attention['time-attention'].append(time_attn)
            elif self.time_aggregation == 'gru':
                h_temporal, _ = T_layer(h_readout)
                latent = self.cls_token(h_temporal)
            else:
                raise NotImplementedError

            logit += self.dropout(L(latent))
            latent_list.append(latent)

        # ========== 5. Return (Supervised Mode) ==========
        if not cfg.get('ssl', False):
            latent = torch.stack(latent_list, dim=1)
            return logit, attention, latent, reg_ortho

        # ========== 6. SSL Reconstruction ==========
        ssl_loss = torch.tensor(0.0, device=h.device)
        ssl_loss_dict = {}

        enc_rep = node_features[-1]
        use_latent_target = ema_update and cfg.get('target_model_type', 'latent') == 'latent'
        target_enc_rep = target_node_features[-1] if use_latent_target else None

        # ---------- Node Reconstruction ----------
        if cfg.get('recon_node', False):
            node_spatial_loss = torch.tensor(0.0, device=h.device)
            node_temporal_loss = torch.tensor(0.0, device=h.device)
            node_code = self.node_projection(enc_rep)
            decoder_type = cfg.get('node_decoder_type', 'mlp-mixer')

            for g in range(self.num_pred_masks):
                mask_nodes = masks['block_mask_nodes'][g]
                total_mask = masks['total_mask_nodes']
                context_nodes = torch.arange(enc_rep.size(0), device=h.device)
                context_nodes = context_nodes[~torch.isin(context_nodes, total_mask)].long()
                non_target = total_mask[~torch.isin(total_mask, mask_nodes)].long()

                # Spatial node reconstruction
                if cfg.get('spatial_node_recon', False):
                    code_g = node_code.clone()
                    if decoder_type == 'mlp-mixer':
                        code_g[mask_nodes] = self.node_mask_token
                        code_g[non_target] = 0.0
                        x = rearrange(code_g, '(b t n) c -> (b t) n c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                        x = self.node_decoder(x)
                        x = rearrange(x, '(b t) n c -> (b t n) c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                        recon = x[mask_nodes]
                    elif decoder_type == 'mlp':
                        recon = self.node_decoder(code_g[mask_nodes])
                    else:
                        recon = self.node_decoder(code_g, a)[mask_nodes]

                    target = target_enc_rep[mask_nodes] if use_latent_target else h_node_origin[mask_nodes]
                    node_spatial_loss += self.node_ssl_criterion(target, recon)

                # Temporal node reconstruction
                if cfg.get('temporal_node_recon', False):
                    A_nodes = masks['block_A_nodes'][g]
                    B_nodes = masks['block_B_nodes'][g]

                    temporal_code = torch.zeros_like(node_code)
                    cat_rep = torch.cat([node_code[A_nodes], node_code[B_nodes]], dim=1)
                    temporal_code[mask_nodes] = self.temporal_node_projection(cat_rep)

                    if decoder_type == 'mlp-mixer':
                        temporal_code[context_nodes] = node_code[context_nodes].clone()
                        temporal_code[non_target] = 0.0
                        x = rearrange(temporal_code, '(b t n) c -> (b t) n c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                        x = self.node_decoder(x)
                        x = rearrange(x, '(b t) n c -> (b t n) c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                        recon = x[mask_nodes]
                    elif decoder_type == 'mlp':
                        recon = self.node_decoder(temporal_code[mask_nodes])
                    else:
                        recon = self.node_decoder(temporal_code, a)[mask_nodes]

                    target = target_enc_rep[mask_nodes] if use_latent_target else h_node_origin[mask_nodes]
                    valid = masks['temporal_node_valid'][g]
                    if valid.any():
                        node_temporal_loss += self.node_ssl_criterion(target[valid], recon[valid])

            if cfg.get('spatial_node_recon', False):
                node_spatial_loss /= self.num_pred_masks
            if cfg.get('temporal_node_recon', False):
                node_temporal_loss /= self.num_pred_masks

            spatial_rate = cfg.get('spatial_rate', 0.5)
            if cfg.get('spatial_node_recon', False) and cfg.get('temporal_node_recon', False):
                node_ssl_loss = spatial_rate * node_spatial_loss + (1 - spatial_rate) * node_temporal_loss
            else:
                node_ssl_loss = node_spatial_loss + node_temporal_loss

            ssl_loss += self.ssl_lambda_node * node_ssl_loss
            ssl_loss_dict['node_ssl_loss'] = node_ssl_loss.item()
            ssl_loss_dict['node_spatial_ssl_loss'] = node_spatial_loss.item()
            ssl_loss_dict['node_temporal_ssl_loss'] = node_temporal_loss.item()

        # ---------- Edge Reconstruction ----------
        if cfg.get('recon_edge', False) and self.apply_edge_loss:
            edge_spatial_loss = torch.tensor(0.0, device=h.device)
            edge_temporal_loss = torch.tensor(0.0, device=h.device)
            edge_code = self.edge_projection(enc_rep)

            # Insert mask tokens at masked node positions for edge decoder
            if masks is not None and masks['total_mask_nodes'] is not None:
                edge_code[masks['total_mask_nodes']] = self.edge_mask_token

            # Apply edge decoder
            decoder_type = cfg.get('edge_decoder_type', 'mlp-mixer')
            if cfg.get('add_edge_decoder', True):
                if decoder_type in ('mlp',):
                    decoded = self.edge_decoder(edge_code)
                elif decoder_type == 'mlp-mixer':
                    x = rearrange(edge_code, '(b t n) c -> (b t) n c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                    x = self.edge_decoder(x)
                    decoded = rearrange(x, '(b t) n c -> (b t n) c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                else:
                    decoded = self.edge_decoder(edge_code, a)
            else:
                decoded = edge_code

            code_bridge = rearrange(decoded, '(b t n) c -> (b t) n c', b=minibatch_size, t=num_timepoints, n=num_nodes)

            # Spatial edge reconstruction
            if cfg.get('spatial_edge_recon', False):
                a_recon = self.edge_converter(code_bridge)
                if cfg.get('edge_block_mask', False) and block_masked_a_indices is not None:
                    a_recon_flat = rearrange(a_recon, '(b t) n c -> b t n c', b=minibatch_size, t=num_timepoints, n=num_nodes)
                    a_recon_flat = rearrange(a_recon_flat[:, 1:-1], 'b t n c -> (b t n c)')
                    a_target = rearrange(a_origin[:, 1:-1], 'b t n c -> (b t n c)').float()
                    all_block_idx = torch.cat(block_masked_a_indices, dim=0).unique().to(h.device)
                    if cfg.get('recon_block_only', True):
                        edge_spatial_loss += self.edge_ssl_criterion(a_recon_flat[all_block_idx], a_target[all_block_idx])
                    else:
                        edge_spatial_loss += self.edge_ssl_criterion(a_recon_flat, a_target)
                else:
                    a_target = rearrange(a_origin, 'b t n c -> (b t) n c').float()
                    edge_spatial_loss += self.edge_ssl_criterion(a_recon, a_target)

            # Temporal edge reconstruction
            if cfg.get('temporal_edge_recon', False):
                g = torch.randint(self.num_pred_masks, (1,)).item()
                A_times = masks['block_A_times'][g]
                B_times = masks['block_B_times'][g]

                time_valid = masks['temporal_time_valid'][g]

                a_recon = self.edge_converter(code_bridge[A_times], code_bridge[B_times])

                if not time_valid.any():
                    pass
                elif cfg.get('edge_block_mask', False) and block_masked_a_indices is not None:
                    a_recon_flat = rearrange(a_recon, '(b t) n c -> (b t n c)', b=minibatch_size, t=num_timepoints - 2, n=num_nodes)
                    a_target = rearrange(a_origin[:, 1:-1], 'b t n c -> (b t n c)').float()
                    all_block_idx = torch.cat(block_masked_a_indices, dim=0).unique().to(h.device)
                    all_block_idx = all_block_idx[time_valid[all_block_idx // (num_nodes * num_nodes)]]
                    if cfg.get('recon_block_only', True):
                        edge_temporal_loss += self.edge_ssl_criterion(a_recon_flat[all_block_idx], a_target[all_block_idx])
                    else:
                        valid_flat = time_valid.repeat_interleave(num_nodes * num_nodes)
                        edge_temporal_loss += self.edge_ssl_criterion(a_recon_flat[valid_flat], a_target[valid_flat])
                else:
                    a_target = rearrange(a_origin[:, 1:-1], 'b t n c -> (b t) n c').float()
                    edge_temporal_loss += self.edge_ssl_criterion(a_recon[time_valid], a_target[time_valid])

            spatial_rate = cfg.get('spatial_rate', 0.5)
            if cfg.get('spatial_edge_recon', False) and cfg.get('temporal_edge_recon', False):
                edge_ssl_loss = spatial_rate * edge_spatial_loss + (1 - spatial_rate) * edge_temporal_loss
            else:
                edge_ssl_loss = edge_spatial_loss + edge_temporal_loss

            ssl_loss += self.ssl_lambda_edge * edge_ssl_loss
            ssl_loss_dict['edge_ssl_loss'] = edge_ssl_loss.item()
            ssl_loss_dict['edge_spatial_ssl_loss'] = edge_spatial_loss.item()
            ssl_loss_dict['edge_temporal_ssl_loss'] = edge_temporal_loss.item()

        return logit, attention, None, reg_ortho, ssl_loss, ssl_loss_dict
