"""ST-JEMA Pre-training: Self-supervised learning on brain fMRI dynamic graphs.

Usage:
    python pretrain.py --config configs/pretrain_stjema.yaml --data_dir ./data --output_dir ./output

This script implements:
  1. Block-based node/edge masking on dynamic brain graphs
  2. JEPA-style reconstruction with EMA target encoder
  3. Spatial + temporal reconstruction objectives
"""

import os
import json
import logging
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from einops import repeat

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import LoggerSTAGIN
from data.dataset import DatasetRest, build_dataloaders
from data.bold import process_dynamic_fc
from models.stjema import ModelSTAGIN
from models.stmae import ModelSTMAE


def parse_args():
    parser = argparse.ArgumentParser(description='ST-JEMA Pre-training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--fold', type=int, default=None, help='Fold index (overrides config train_k)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (overrides config)')
    return parser.parse_args()


def build_momentum_scheduler(ema_decay, ema_decay_final, total_steps):
    """Linear EMA momentum schedule from ema_decay to ema_decay_final."""
    for i in range(total_steps + 1):
        yield ema_decay + i * (ema_decay_final - ema_decay) / total_steps


def train_one_epoch(model, dataloader, dataset, optimizer, scheduler, momentum_scheduler,
                    cfg, device, epoch, num_nodes, dyn_v_template):
    """Run one training epoch."""
    model.train()
    losses = []
    minibatch_size = cfg.minibatch_size

    pbar = tqdm(dataloader, desc=f'Train epoch {epoch}', leave=False)
    for i, batch in enumerate(pbar):
        # Compute dynamic functional connectivity
        if cfg.static:
            dynamic_length = batch['timeseries'].shape[1]
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], dynamic_length, dynamic_length, dynamic_length)
            sampling_endpoints = [p + dynamic_length for p in sampling_points]
        else:
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], cfg.window_size, cfg.window_stride, cfg.dynamic_length)
            sampling_endpoints = [p + cfg.window_size for p in sampling_points]

        # Node identity features
        if i == 0:
            dyn_v = repeat(dyn_v_template, 'n1 n2 -> b t n1 n2',
                           t=len(sampling_points), b=minibatch_size)

        actual_batch = len(dyn_a)
        if actual_batch < minibatch_size:
            dyn_v_batch = dyn_v[:actual_batch]
        else:
            dyn_v_batch = dyn_v

        t = batch['timeseries'].permute(1, 0, 2)  # [T, B, N]

        # Enable edge loss after warmup
        if model.current_step >= cfg.get('edge_recon_start_from', 0) and not model.apply_edge_loss:
            model.apply_edge_loss = True

        # Forward pass (SSL mode)
        logit, attention, latent, reg_ortho, ssl_loss, ssl_loss_dict = model(
            dyn_v_batch.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)

        loss = cfg.ssl_lambda_total * ssl_loss
        loss += cfg.get('reg_lambda', 0.0) * reg_ortho

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if cfg.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # EMA update target model
        if cfg.ema_update:
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param, target_param in zip(model.parameters(), model.target_model.parameters()):
                    target_param.data.mul_(m).add_(param.data, alpha=1.0 - m)

        model.current_step += 1
        losses.append(ssl_loss.item())

        pbar.set_postfix({'ssl_loss': f'{ssl_loss.item():.4f}'})

    return np.mean(losses)


@torch.no_grad()
def validate(model, eval_dataloader, dataset, cfg, device, epoch, num_nodes, dyn_v_template):
    """Run validation and return average SSL loss."""
    model.eval()
    losses = []
    minibatch_size = cfg.get('eval_minibatch_size', 1)

    pbar = tqdm(eval_dataloader, desc=f'Val   epoch {epoch}', leave=False)
    for i, batch in enumerate(pbar):
        if cfg.static:
            dynamic_length = batch['timeseries'].shape[1]
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], dynamic_length, dynamic_length, dynamic_length)
            sampling_endpoints = [p + dynamic_length for p in sampling_points]
        else:
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], cfg.window_size, cfg.window_stride, cfg.dynamic_length,
                sampling_init=0)
            sampling_endpoints = [p + cfg.window_size for p in sampling_points]

        if i == 0:
            dyn_v = repeat(dyn_v_template, 'n1 n2 -> b t n1 n2',
                           t=len(sampling_points), b=minibatch_size)

        actual_batch = len(dyn_a)
        if actual_batch < minibatch_size:
            dyn_v_batch = dyn_v[:actual_batch]
        else:
            dyn_v_batch = dyn_v

        t = batch['timeseries'].permute(1, 0, 2)

        logit, attention, latent, reg_ortho, ssl_loss, ssl_loss_dict = model(
            dyn_v_batch.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)

        losses.append(ssl_loss.item())

    return np.mean(losses)


def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.fold is not None:
        cfg.train_k = args.fold
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs

    set_seed(cfg.seed)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Output directory
    output_dir = Path(args.output_dir)
    model_dir = output_dir / 'model' / str(cfg.train_k)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'pretrain.log'),
            logging.StreamHandler(),
        ],
    )

    # Save config
    import yaml
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)

    # Build dataset
    logging.info('Building dataset...')
    dataset, dataloader, eval_dataloader = build_dataloaders(cfg, root=args.data_dir)
    num_nodes = dataset.num_nodes
    logging.info(f'Dataset: {cfg.dataset}, nodes: {num_nodes}, fold: {cfg.train_k}')

    # Time dimension
    if cfg.static:
        time_dim = 1
    else:
        time_dim = len(range(0, cfg.dynamic_length - cfg.window_size + 1, cfg.window_stride))
    logging.info(f'Time dimension: {time_dim}')

    # Build model
    logging.info('Building model...')
    # ssl_type selects the pre-training objective: stjema (default) or stmae
    model_cls = ModelSTMAE if cfg.get('ssl_type', 'stjema') == 'stmae' else ModelSTAGIN
    model = model_cls(
        input_dim=num_nodes,
        hidden_dim=cfg.hidden_dim,
        num_classes=dataset.num_classes,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        sparsity=cfg.sparsity,
        dropout=cfg.dropout,
        cls_token=cfg.cls_token,
        readout=cfg.readout,
        cfg=cfg,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Model parameters: {num_params:,}')

    model.to(device)

    # Setup EMA target model
    if cfg.ema_update:
        target_model = model_cls(
            input_dim=num_nodes,
            hidden_dim=cfg.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            sparsity=cfg.sparsity,
            dropout=cfg.dropout,
            cls_token=cfg.cls_token,
            readout=cfg.readout,
            cfg=cfg,
        )
        target_model.load_state_dict(model.state_dict(), strict=True)
        model.target_model = target_model
        for param in model.target_model.parameters():
            param.requires_grad = False
        model.target_model.to(device)

        total_steps = cfg.num_epochs * len(dataloader)
        momentum_scheduler = build_momentum_scheduler(
            cfg.ema_decay, cfg.ema_decay_final, total_steps)
    else:
        momentum_scheduler = None

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # Learning rate scheduler
    if cfg.get('scheduler_type', 'cosinedecay') == 'cosinedecay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 + np.cos(step * np.pi / (cfg.num_epochs * len(dataloader)))) * 0.5,
        )
    elif cfg.scheduler_type == 'constant':
        scheduler = None
    elif cfg.scheduler_type == 'cawr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=len(dataloader), T_mult=1,
            eta_min=cfg.get('scheduler_gamma', 0.001) * cfg.lr)
    else:
        scheduler = None

    # Precompute identity node features template
    dyn_v_template = torch.eye(num_nodes)

    # Training loop
    logging.info(f'Starting pre-training for {cfg.num_epochs} epochs...')
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        # Train
        dataset.set_fold(cfg.train_k, mode='train')
        train_loss = train_one_epoch(
            model, dataloader, dataset, optimizer, scheduler, momentum_scheduler,
            cfg, device, epoch, num_nodes, dyn_v_template)

        # Save model (target model weights for JEPA)
        if cfg.ema_update:
            torch.save(model.target_model.state_dict(), model_dir / 'model.pth')
        else:
            torch.save(model.state_dict(), model_dir / 'model.pth')

        logging.info(f'Epoch {epoch}: train_ssl_loss={train_loss:.4f}')

        # Validate
        if cfg.get('validate', True) and cfg.get('val_ratio', 0.1) > 0:
            if (epoch + 1) % cfg.get('validate_freq', 1) == 0:
                dataset.set_fold(cfg.train_k, mode='val')
                val_loss = validate(
                    model, eval_dataloader, dataset, cfg, device, epoch, num_nodes, dyn_v_template)

                logging.info(f'Epoch {epoch}: val_ssl_loss={val_loss:.4f}')

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if cfg.ema_update:
                        torch.save(model.target_model.state_dict(), model_dir / 'best_model.pth')
                    else:
                        torch.save(model.state_dict(), model_dir / 'best_model.pth')
                    logging.info(f'  -> New best model saved (val_loss={val_loss:.4f})')

    logging.info(f'Pre-training complete. Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})')


if __name__ == '__main__':
    main()
