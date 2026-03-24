"""Synthetic-aware sample weighting using frozen-reference features vs real class centroids."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.metrics import compute_accuracy
from src.models.backbone import forward_features


@torch.no_grad()
def compute_real_class_centroids(
    frozen_model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    """
    Accumulate mean penultimate features per class on the given loader (typically 5% real train).
    """
    frozen_model.eval()
    feat_dim: Optional[int] = None
    sums: Optional[torch.Tensor] = None
    counts = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for batch in data_loader:
        if len(batch) == 4:
            images, targets, _, _ = batch
        else:
            images, targets, _ = batch
        images = images.to(device)
        targets = targets.to(device)
        _, feat = forward_features(frozen_model, images)
        if feat_dim is None:
            feat_dim = feat.shape[1]
            sums = torch.zeros(num_classes, feat_dim, device=device, dtype=torch.float64)
        assert sums is not None
        for c in range(num_classes):
            mask = targets == c
            if mask.any():
                sums[c] += feat[mask].double().sum(0)
                counts[c] += mask.double().sum()

    counts_safe = counts.clamp(min=1.0).unsqueeze(1)
    centroids = (sums / counts_safe).float()
    return centroids


def train_one_epoch_synthetic_aware(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    frozen_model: nn.Module,
    centroids: torch.Tensor,
    synthetic_lambda: float,
    mixed_precision: bool = True,
) -> Tuple[float, float]:
    """
    Per-sample CE with weights: real=1, synthetic=exp(-lambda * ||f_ref - mu_y||^2), f_ref from frozen_model.
    """
    model.train()
    frozen_model.eval()
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        if len(batch) == 4:
            images, targets, _, syn_flag = batch
        else:
            images, targets, _ = batch
            syn_flag = torch.zeros(images.size(0))

        images = images.to(device)
        targets = targets.to(device)
        syn_flag = syn_flag.to(device).float()

        optimizer.zero_grad()

        with torch.no_grad():
            _, feat_ref = forward_features(frozen_model, images)
            mu = centroids[targets]
            dist_sq = (feat_ref - mu).pow(2).sum(dim=-1)
            w_syn = torch.exp(-synthetic_lambda * dist_sq)
            w = torch.where(syn_flag > 0.5, w_syn, torch.ones_like(w_syn))
            w = w / (w.mean() + 1e-8)

        if mixed_precision:
            with autocast():
                outputs = model(images)
                loss_vec = F.cross_entropy(outputs, targets, reduction="none")
                loss = (loss_vec * w).sum() / (w.sum() + 1e-8)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_vec = F.cross_entropy(outputs, targets, reduction="none")
            loss = (loss_vec * w).sum() / (w.sum() + 1e-8)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        acc1 = compute_accuracy(outputs.float(), targets)[0]
        running_correct += acc1 / 100.0 * batch_size
        running_total += batch_size

    return running_loss / running_total, running_correct / running_total
