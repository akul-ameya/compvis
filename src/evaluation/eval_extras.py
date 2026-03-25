"""Temperature scaling, linear probe, feature eigen-spectrum, linear CKA (Stage 2)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.models.backbone import forward_features


def fit_temperature_scalar(logits: np.ndarray, labels: np.ndarray, num_grid: int = 80) -> float:
    """Grid search T > 0 minimizing NLL on validation logits."""
    lt = torch.from_numpy(logits).float()
    y = torch.from_numpy(labels).long()
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.05, 8.0, num_grid):
        loss = F.cross_entropy(lt / t, y).item()
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def ece_from_logits_scaled(
    logits: np.ndarray, labels: np.ndarray, temperature: float, ece_bins: int
) -> float:
    from src.metrics.metrics import ExpectedCalibrationError

    lt = torch.from_numpy(logits).float() / temperature
    yl = torch.from_numpy(labels).long()
    ece_computer = ExpectedCalibrationError(n_bins=ece_bins)
    ece, _, _, _ = ece_computer.compute_from_logits(lt, yl)
    return float(ece)


def effective_rank(eigenvalues: np.ndarray) -> float:
    ev = np.maximum(eigenvalues.astype(np.float64), 0.0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 1e-16]
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_cka_features(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two n×d feature matrices (same n rows = paired samples)."""
    X = X.astype(np.float64) - X.mean(0, keepdims=True)
    Y = Y.astype(np.float64) - Y.mean(0, keepdims=True)
    cxy = X.T @ Y
    cxx = X.T @ X
    cyy = Y.T @ Y
    num = np.linalg.norm(cxy, ord="fro") ** 2
    den = np.linalg.norm(cxx, ord="fro") * np.linalg.norm(cyy, ord="fro") + 1e-12
    return float(num / den)


@torch.no_grad()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    fs, ys = [], []
    n = 0
    for batch in loader:
        if len(batch) == 4:
            x, y, _, _ = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        _, feat = forward_features(model, x)
        fs.append(feat.float().cpu().numpy())
        ys.append(y.cpu().numpy())
        n += len(y)
        if n >= max_samples:
            break
    Fm = np.concatenate(fs, axis=0)[:max_samples]
    Ym = np.concatenate(ys, axis=0)[:max_samples]
    return Fm, Ym


@torch.no_grad()
def collect_paired_features_two_models(
    model_a: nn.Module,
    model_b: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model_a.eval()
    model_b.eval()
    fa, fb = [], []
    n = 0
    for batch in loader:
        if len(batch) == 4:
            x, _, _, _ = batch
        else:
            x, _, _ = batch
        x = x.to(device)
        _, za = forward_features(model_a, x)
        _, zb = forward_features(model_b, x)
        fa.append(za.float().cpu().numpy())
        fb.append(zb.float().cpu().numpy())
        n += x.size(0)
        if n >= max_samples:
            break
    return (
        np.concatenate(fa, axis=0)[:max_samples],
        np.concatenate(fb, axis=0)[:max_samples],
    )


def linear_probe_sklearn_accuracy(
    train_feats: np.ndarray,
    train_y: np.ndarray,
    val_feats: np.ndarray,
    val_y: np.ndarray,
) -> float:
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )
    clf.fit(train_feats, train_y)
    pred = clf.predict(val_feats)
    return float((pred == val_y).mean())


def covariance_top_eigenvalues(feats: np.ndarray, k: int = 40) -> Tuple[List[float], float]:
    X = feats.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    n = X.shape[0]
    if n < 2:
        return [], 0.0
    cov = (X.T @ X) / max(n - 1, 1)
    w = np.linalg.eigvalsh(cov)
    w = np.sort(np.real(w))[::-1][:k]
    er = effective_rank(w)
    return [float(x) for x in w], er


def plot_eigenvalues_overlay(
    eigenvalues: List[float],
    out_path: Path,
    title: str = "Top covariance eigenvalues (penultimate)",
) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", ms=3)
    plt.xlabel("Index (sorted desc)")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
