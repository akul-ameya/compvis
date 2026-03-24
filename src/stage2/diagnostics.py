"""Per-class diagnostics from a trained model on the validation set."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.backbone import forward_features


@torch.no_grad()
def collect_val_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logits (N,C), labels (N,), features (N,D)."""
    model.eval()
    xs, ys, fs = [], [], []
    for batch in val_loader:
        images, targets, _ = batch
        images = images.to(device)
        targets = targets.to(device)
        logits, feat = forward_features(model, images)
        xs.append(logits.float().cpu().numpy())
        ys.append(targets.cpu().numpy())
        fs.append(feat.float().cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(fs)


def compute_class_diagnostics(
    logits: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    class_ids_ordered: list,
) -> pd.DataFrame:
    """
    class_ids_ordered: list of str class ids for index 0..num_classes-1 (label order).
    """
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)

    rows = []
    num_classes = len(class_ids_ordered)
    centroids = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            centroids[c] = 0.0
            continue
        centroids[c] = features[mask].mean(axis=0)
        counts[c] = int(mask.sum())

    # nearest other centroid distance per class
    for c in range(num_classes):
        mask = labels == c
        cid = class_ids_ordered[c]
        if mask.sum() == 0:
            rows.append(
                {
                    "class_id": cid,
                    "baseline_acc": 0.0,
                    "mean_confidence": 0.0,
                    "prediction_entropy": 0.0,
                    "feature_compactness": 0.0,
                    "feature_separation": 0.0,
                    "nearest_centroid_margin": 0.0,
                }
            )
            continue
        acc_c = (preds[mask] == labels[mask]).mean()
        mean_conf = conf[mask].mean()
        mean_ent = ent[mask].mean()
        fc = np.linalg.norm(features[mask] - centroids[c], axis=1).mean()
        dists = np.linalg.norm(centroids - centroids[c], axis=1)
        dists[c] = np.inf
        j = int(np.argmin(dists))
        sep = float(np.linalg.norm(centroids[c] - centroids[j]))
        own = np.linalg.norm(features[mask] - centroids[c], axis=1).mean()
        other_d = np.linalg.norm(features[mask] - centroids[j], axis=1).mean()
        margin = float(other_d - own)
        rows.append(
            {
                "class_id": cid,
                "baseline_acc": float(acc_c),
                "mean_confidence": float(mean_conf),
                "prediction_entropy": float(mean_ent),
                "feature_compactness": float(fc),
                "feature_separation": float(sep),
                "nearest_centroid_margin": margin,
            }
        )

    df = pd.DataFrame(rows).set_index("class_id")
    return df


def merge_synthetic_quality(df: pd.DataFrame, quality_csv: Optional[Path]) -> pd.DataFrame:
    if quality_csv is None or not Path(quality_csv).exists():
        df = df.copy()
        df["synthetic_quality"] = 0.0
        return df
    q = pd.read_csv(quality_csv)
    if "class_id" in q.columns:
        q = q.set_index("class_id")
    col = "mean_distance" if "mean_distance" in q.columns else q.columns[0]
    df = df.copy()
    df["synthetic_quality"] = df.index.map(lambda x: float(q.loc[x, col]) if x in q.index else 0.0)
    return df


def per_class_accuracy(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    preds = logits.argmax(axis=1)
    out = {}
    for c in range(num_classes):
        m = labels == c
        if m.sum() == 0:
            out[c] = 0.0
        else:
            out[c] = float((preds[m] == labels[m]).mean())
    return out


def utility_from_accs(
    acc_baseline: Dict[int, float],
    acc_uniform: Dict[int, float],
    class_ids_ordered: list,
) -> Dict[str, float]:
    util = {}
    for c, cid in enumerate(class_ids_ordered):
        util[cid] = acc_uniform.get(c, 0.0) - acc_baseline.get(c, 0.0)
    return util
