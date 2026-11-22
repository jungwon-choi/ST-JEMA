"""ST-JEMA Fine-tuning: Downstream classification/regression on brain fMRI.

Usage:
    python finetune.py --config configs/finetune_classification.yaml \
                       --data_dir ./data --output_dir ./output_ft \
                       --pretrain_dir ./output_pt

This script loads a pre-trained ST-JEMA encoder and fine-tunes it for
downstream classification (e.g., sex, diagnosis) or regression (e.g., age).
"""

import os
import json
import logging
import argparse
from pathlib import Path
from copy import deepcopy

import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import repeat

from utils.config import load_config
from utils.seed import set_seed
from utils.logger import LoggerSTAGIN
from data.dataset import DatasetRest, build_dataloaders
from data.bold import process_dynamic_fc
from models.stjema import ModelSTAGIN


def parse_args():
    parser = argparse.ArgumentParser(description='ST-JEMA Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--pretrain_dir', type=str, default=None, help='Pre-trained model directory')
    parser.add_argument('--fold', type=int, default=None, help='Fold index (overrides config train_k)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze pretrained encoder weights')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and evaluate the fine-tuned model saved in output_dir on the test split')
    return parser.parse_args()


def load_pretrained_weights(model, pretrain_dir, fold, cfg, device):
    """Load pretrained encoder weights from checkpoint.

    Loads encoder weights (GNN layers, readout, etc.) while keeping
    the classification head randomly initialized.
    """
    # Validate config compatibility
    pretrain_config_path = os.path.join(pretrain_dir, 'config.yaml')
    if os.path.exists(pretrain_config_path):
        with open(pretrain_config_path) as f:
            pt_cfg = yaml.safe_load(f)

        pt_encoder = pt_cfg.get('encoder_type', 'gin')
        if pt_encoder != cfg.encoder_type:
            raise ValueError(
                f'Encoder type mismatch: pretrain={pt_encoder}, finetune={cfg.encoder_type}')

        pt_hidden = pt_cfg.get('hidden_dim')
        if pt_hidden and pt_hidden != cfg.hidden_dim:
            raise ValueError(
                f'Hidden dim mismatch: pretrain={pt_hidden}, finetune={cfg.hidden_dim}')

        pt_layers = pt_cfg.get('num_layers')
        if pt_layers and pt_layers != cfg.num_layers:
            raise ValueError(
                f'Num layers mismatch: pretrain={pt_layers}, finetune={cfg.num_layers}')

        logging.info(f'Pretrain config validated: encoder={pt_encoder}, '
                     f'hidden_dim={pt_hidden}, layers={pt_layers}')

    # Load weights
    fix_k = cfg.get('fix_pretrain_k', None)
    model_fold = fix_k if fix_k is not None else fold

    use_best = cfg.get('use_bestval_model', False)
    model_file = 'best_model.pth' if use_best else 'model.pth'
    weight_path = os.path.join(pretrain_dir, 'model', str(model_fold), model_file)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f'Pretrained weights not found: {weight_path}')

    initial_dict = deepcopy(model.state_dict())
    pretrain_dict = torch.load(weight_path, map_location=device)

    # Remove classification head from pretrained weights
    for key in list(pretrain_dict.keys()):
        if 'linear_layers' in key:
            del pretrain_dict[key]

    # Load encoder weights (strict=False: classification head missing is OK)
    model.load_state_dict(pretrain_dict, strict=False)

    # Log which parameters were loaded
    current_dict = model.state_dict()
    loaded_modules = set()
    new_modules = set()
    for key in current_dict:
        if key in initial_dict and (current_dict[key] != initial_dict[key]).any():
            loaded_modules.add(key.split('.')[0])
        else:
            new_modules.add(key.split('.')[0])

    logging.info(f'Loaded pretrained modules: {sorted(loaded_modules)}')
    logging.info(f'Randomly initialized modules: {sorted(new_modules)}')
    logging.info(f'Loaded from: {weight_path}')

    return model, initial_dict


def train_one_epoch(model, dataloader, dataset, criterion, optimizer, scheduler,
                    cfg, device, epoch, num_nodes, dyn_v_template):
    """Run one training epoch for supervised fine-tuning."""
    model.train()
    losses = []
    all_labels, all_preds, all_probs = [], [], []
    minibatch_size = cfg.minibatch_size

    pbar = tqdm(dataloader, desc=f'Train epoch {epoch}', leave=False)
    for i, batch in enumerate(pbar):
        # Compute dynamic FC
        if cfg.static:
            dynamic_length = batch['timeseries'].shape[1]
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], dynamic_length, dynamic_length, dynamic_length)
            sampling_endpoints = [p + dynamic_length for p in sampling_points]
        else:
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], cfg.window_size, cfg.window_stride, cfg.dynamic_length)
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
        label = batch['label']

        # Forward pass (supervised mode: returns logit, attention, latent, reg_ortho)
        logit, attention, latent, reg_ortho = model(
            dyn_v_batch.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)

        # Compute loss
        if cfg.regression:
            loss = criterion(logit.squeeze(-1), label.to(device))
        else:
            loss = criterion(logit, label.to(device))

        loss += cfg.get('reg_lambda', 0.0) * reg_ortho

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if cfg.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

        # Collect predictions
        with torch.no_grad():
            if dataset.num_classes > 1:
                pred = logit.argmax(dim=1).cpu().numpy()
                prob = logit.softmax(dim=1).cpu().numpy()
            else:
                pred = logit.squeeze(-1).cpu().numpy()
                prob = pred  # regression: prob = pred
            all_labels.append(label.numpy())
            all_preds.append(pred)
            all_probs.append(prob)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return np.mean(losses), all_labels, all_preds, all_probs


@torch.no_grad()
def evaluate(model, eval_dataloader, dataset, criterion, cfg, device, epoch,
             num_nodes, dyn_v_template, mode='val'):
    """Run evaluation (validation or test)."""
    model.eval()
    losses = []
    all_labels, all_preds, all_probs = [], [], []
    minibatch_size = cfg.get('eval_minibatch_size', 1)
    full_length = cfg.get('eval_full_length', False) and not cfg.static
    if full_length:
        assert minibatch_size == 1, 'eval_full_length requires eval_minibatch_size: 1 (subjects differ in length)'

    pbar = tqdm(eval_dataloader, desc=f'{mode:5s} epoch {epoch}', leave=False)
    for i, batch in enumerate(pbar):
        if cfg.static:
            dynamic_length = batch['timeseries'].shape[1]
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], dynamic_length, dynamic_length, dynamic_length)
            sampling_endpoints = [p + dynamic_length for p in sampling_points]
        else:
            dyn_a, sampling_points = process_dynamic_fc(
                batch['timeseries'], cfg.window_size, cfg.window_stride,
                None if full_length else cfg.dynamic_length,
                sampling_init=0)  # deterministic for eval
            sampling_endpoints = [p + cfg.window_size for p in sampling_points]

        if i == 0 or full_length:
            dyn_v = repeat(dyn_v_template, 'n1 n2 -> b t n1 n2',
                           t=len(sampling_points), b=minibatch_size)

        actual_batch = len(dyn_a)
        if actual_batch < minibatch_size:
            dyn_v_batch = dyn_v[:actual_batch]
        else:
            dyn_v_batch = dyn_v

        t = batch['timeseries'].permute(1, 0, 2)
        label = batch['label']

        logit, attention, latent, reg_ortho = model(
            dyn_v_batch.to(device), dyn_a.to(device), t.to(device), sampling_endpoints)

        if cfg.regression:
            loss = criterion(logit.squeeze(-1), label.to(device))
        else:
            loss = criterion(logit, label.to(device))

        losses.append(loss.item())

        if dataset.num_classes > 1:
            pred = logit.argmax(dim=1).cpu().numpy()
            prob = logit.softmax(dim=1).cpu().numpy()
        else:
            pred = logit.squeeze(-1).cpu().numpy()
            prob = pred
        all_labels.append(label.numpy())
        all_preds.append(pred)
        all_probs.append(prob)

    return np.mean(losses), all_labels, all_preds, all_probs


def compute_metrics(logger, dataset, cfg, preds, labels, probs):
    """Evaluate predictions; regression outputs are mapped back to the original label scale."""
    preds, labels, probs = np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)
    if cfg.regression:
        preds = dataset.label_encoder.inverse_transform(preds)
        labels = dataset.label_encoder.inverse_transform(labels)
        probs = preds
    logger.initialize(cfg.train_k)
    logger.add(k=cfg.train_k, pred=preds, true=labels, prob=probs)
    metrics = logger.evaluate(cfg.train_k)
    if cfg.regression:
        metrics['nmae'] = metrics['mae'] / dataset.label_encoder.std_  # MAE normalized by the label std
    return metrics


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
    if args.pretrain_dir is not None:
        cfg.pretraindir = args.pretrain_dir
    if args.freeze_encoder:
        cfg.freeze_pretrain = True

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
            logging.FileHandler(output_dir / ('eval.log' if args.eval_only else 'finetune.log')),
            logging.StreamHandler(),
        ],
    )

    # Save config (kept unchanged when only evaluating an existing run)
    with open(output_dir / ('eval_config.yaml' if args.eval_only else 'config.yaml'), 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)

    # Build dataset
    logging.info('Building dataset...')
    dataset, dataloader, eval_dataloader = build_dataloaders(cfg, root=args.data_dir)
    num_nodes = dataset.num_nodes
    logging.info(f'Dataset: {cfg.dataset}, nodes: {num_nodes}, classes: {dataset.num_classes}, fold: {cfg.train_k}')

    # Build model (supervised mode: ssl=False in config)
    logging.info('Building model...')
    model = ModelSTAGIN(
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

    # Load pretrained weights
    if cfg.get('pretraindir') is not None:
        model, initial_dict = load_pretrained_weights(
            model, cfg.pretraindir, cfg.train_k, cfg, device)

        # Optionally freeze encoder
        if cfg.get('freeze_pretrain', False):
            current_dict = model.state_dict()
            for key, param in model.named_parameters():
                if key in initial_dict and (current_dict[key] != initial_dict[key]).any():
                    param.requires_grad = False
            logging.info('Encoder weights frozen')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Trainable parameters: {num_params:,}')

    model.to(device)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss() if dataset.num_classes > 1 else torch.nn.MSELoss()

    if args.eval_only:
        ckpt = model_dir / 'model.pth'
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logging.info(f'Evaluation only: loaded {ckpt} (eval_full_length={cfg.get("eval_full_length", False)})')
        dataset.set_fold(cfg.train_k, mode='test')
        test_loss, test_labels, test_preds, test_probs = evaluate(
            model, eval_dataloader, dataset, criterion, cfg, device, -1,
            num_nodes, torch.eye(num_nodes), mode='test')
        logger = LoggerSTAGIN(cfg.k_fold, dataset.num_classes)
        test_metrics = compute_metrics(logger, dataset, cfg, test_preds, test_labels, test_probs)
        logging.info('Final test metrics (eval only: ' +
                     ' '.join(f'{k}={v:.4f}' for k, v in test_metrics.items()) + ')')
        return

    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.wd,
    )

    # Learning rate scheduler
    scheduler_type = cfg.get('scheduler_type', 'onecycle')
    if scheduler_type == 'onecycle':
        max_lr = cfg.get('max_lr') or cfg.lr * 2
        div_factor = 2
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=cfg.num_epochs,
            steps_per_epoch=len(dataloader), pct_start=0.2,
            div_factor=div_factor, final_div_factor=1000)
    elif scheduler_type == 'cosinedecay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 + np.cos(step * np.pi / (cfg.num_epochs * len(dataloader)))) * 0.5,
        )
    elif scheduler_type == 'constant':
        scheduler = None
    else:
        scheduler = None

    # Logger
    logger = LoggerSTAGIN(cfg.k_fold, dataset.num_classes)
    dyn_v_template = torch.eye(num_nodes)

    # Training loop
    logging.info(f'Starting fine-tuning for {cfg.num_epochs} epochs...')

    if cfg.regression:
        best_score = float('inf')  # lower is better (MSE)
    else:
        best_score = 0.0  # higher is better (AUC-ROC)
    best_epoch = 0

    final_test_str = 'not evaluated'
    for epoch in range(cfg.num_epochs):
        # Train
        dataset.set_fold(cfg.train_k, mode='train')
        train_loss, train_labels, train_preds, train_probs = train_one_epoch(
            model, dataloader, dataset, criterion, optimizer, scheduler,
            cfg, device, epoch, num_nodes, dyn_v_template)

        logging.info(f'Epoch {epoch}: train_loss={train_loss:.4f}')

        # Save latest model
        torch.save(model.state_dict(), model_dir / 'model.pth')

        # Validate + Test
        is_last_epoch = epoch == cfg.num_epochs - 1
        if cfg.get('validate', True) and ((epoch + 1) % cfg.get('validate_freq', 1) == 0 or is_last_epoch):
            # Validation
            if cfg.get('val_ratio', 0.0) > 0:
                dataset.set_fold(cfg.train_k, mode='val')
                val_loss, val_labels, val_preds, val_probs = evaluate(
                    model, eval_dataloader, dataset, criterion, cfg, device, epoch,
                    num_nodes, dyn_v_template, mode='val')

                val_metrics = compute_metrics(logger, dataset, cfg, val_preds, val_labels, val_probs)

                val_metric_str = ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items())
                logging.info(f'Epoch {epoch}: val_loss={val_loss:.4f} {val_metric_str}')

                # Best model selection
                improved = False
                if cfg.regression:
                    if val_metrics['mse'] <= best_score:
                        best_score = val_metrics['mse']
                        improved = True
                else:
                    if val_metrics['roc_auc'] >= best_score:
                        best_score = val_metrics['roc_auc']
                        improved = True

                if improved:
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_dir / 'best_model.pth')
                    logging.info(f'  -> New best model saved')

            # Test
            dataset.set_fold(cfg.train_k, mode='test')
            test_loss, test_labels, test_preds, test_probs = evaluate(
                model, eval_dataloader, dataset, criterion, cfg, device, epoch,
                num_nodes, dyn_v_template, mode='test')

            test_metrics = compute_metrics(logger, dataset, cfg, test_preds, test_labels, test_probs)

            test_metric_str = ' '.join(f'{k}={v:.4f}' for k, v in test_metrics.items())
            logging.info(f'Epoch {epoch}: test_loss={test_loss:.4f} {test_metric_str}')

            final_test_str = f'epoch {epoch}: {test_metric_str}'

            if cfg.get('val_ratio', 0.0) <= 0:
                best_epoch = epoch

            # Save test predictions at best epoch
            if best_epoch == epoch:
                logger.save_predictions(str(model_dir), k=cfg.train_k, mode='test', epoch=epoch)
                logger.to_csv(str(model_dir), k=cfg.train_k, epoch=epoch)

    logging.info(f'Fine-tuning complete. Best epoch: {best_epoch}')
    if cfg.get('val_ratio', 0.0) > 0:
        score_name = 'MSE' if cfg.regression else 'AUC-ROC'
        logging.info(f'Best validation {score_name}: {best_score:.4f}')
    logging.info(f'Final test metrics ({final_test_str})')


if __name__ == '__main__':
    main()
