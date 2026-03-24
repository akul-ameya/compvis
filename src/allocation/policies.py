"""
Allocation policies: uniform, hard_class, uncertainty, predicted_utility.
Total budget B = num_classes * max_per_class at 15x; distribute with normalize + clamp [min_floor, max_cap].
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _normalize_and_clamp(
    raw: np.ndarray,
    class_ids: List[str],
    total_budget: int,
    min_floor: int,
    max_cap: int,
) -> Dict[str, int]:
    raw = np.maximum(raw, 1e-8)
    w = raw / raw.sum() * total_budget
    w = np.round(w).astype(int)
    # Fix rounding drift
    diff = total_budget - int(w.sum())
    if diff != 0:
        w[np.argmax(w)] += diff
    w = np.clip(w, min_floor, max_cap)
    # If clipping broke total, redistribute (iterative simple fix)
    for _ in range(1000):
        s = int(w.sum())
        if s == total_budget:
            break
        if s > total_budget:
            idx = int(np.argmax(w))
            if w[idx] > min_floor:
                w[idx] -= 1
            else:
                break
        else:
            idx = int(np.argmin(w))
            if w[idx] < max_cap:
                w[idx] += 1
            else:
                break
    return {cid: int(w[i]) for i, cid in enumerate(class_ids)}


def uniform_allocation(num_classes: int, per_class: int) -> Dict[str, int]:
    # class_ids unknown here — caller passes ordered list
    raise NotImplementedError("use uniform_from_class_ids")


def uniform_from_class_ids(class_ids: List[str], per_class: int) -> Dict[str, int]:
    return {cid: per_class for cid in class_ids}


def hard_class_allocation(
    class_ids: List[str],
    baseline_acc: Dict[str, float],
    total_budget: int,
    min_floor: int,
    max_cap: int,
) -> Dict[str, int]:
    arr = np.array([1.0 - baseline_acc[c] for c in class_ids], dtype=np.float64)
    return _normalize_and_clamp(arr, class_ids, total_budget, min_floor, max_cap)


def uncertainty_allocation(
    class_ids: List[str],
    entropy: Dict[str, float],
    total_budget: int,
    min_floor: int,
    max_cap: int,
) -> Dict[str, int]:
    arr = np.array([entropy[c] for c in class_ids], dtype=np.float64)
    return _normalize_and_clamp(arr, class_ids, total_budget, min_floor, max_cap)


def predicted_utility_allocation(
    class_ids: List[str],
    features_df: pd.DataFrame,
    utility: Dict[str, float],
    total_budget: int,
    min_floor: int,
    max_cap: int,
    cv_folds: int = 10,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Ridge to predict utility; allocate proportional to max(0, pred).
    features_df rows indexed by class_id.
    """
    feat = features_df.reindex(class_ids).fillna(0.0)
    X = feat.values.astype(np.float64)
    y = np.array([utility.get(c, 0.0) for c in class_ids], dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if len(class_ids) >= 3:
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=min(cv_folds, len(class_ids)))
    else:
        ridge = Ridge(alpha=1.0)
    ridge.fit(Xs, y)
    pred = ridge.predict(Xs)
    raw = np.maximum(pred, 0.0) + 1e-6
    alloc = _normalize_and_clamp(raw, class_ids, total_budget, min_floor, max_cap)
    meta = {
        "train_r2": float(ridge.score(Xs, y)),
        "cv_r2_mean": float(np.mean(cross_val_r2(Xs, y, min(cv_folds, len(class_ids))))),
    }
    if hasattr(ridge, "alpha_"):
        meta["ridge_alpha"] = float(ridge.alpha_)
    elif hasattr(ridge, "alpha"):
        meta["ridge_alpha"] = float(ridge.alpha)
    return alloc, meta


def cross_val_r2(X: np.ndarray, y: np.ndarray, k: int) -> List[float]:
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=42)
    scores = []
    for train_i, val_i in kf.split(X):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[train_i])
        Xv = scaler.transform(X[val_i])
        model = RidgeCV(alphas=np.logspace(-3, 3, 15))
        model.fit(Xt, y[train_i])
        pred = model.predict(Xv)
        ss_res = ((y[val_i] - pred) ** 2).sum()
        ss_tot = ((y[val_i] - y[val_i].mean()) ** 2).sum() + 1e-8
        scores.append(1.0 - ss_res / ss_tot)
    return scores


def compute_allocations(
    policy: str,
    class_ids: List[str],
    total_budget: int,
    min_floor: int,
    max_cap: int,
    diagnostics_df: pd.DataFrame,
    utility: Optional[Dict[str, float]] = None,
    cv_folds: int = 10,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    diagnostics_df: index = class_id, columns include
      baseline_acc, mean_confidence, prediction_entropy, feature_compactness,
      feature_separation, nearest_centroid_margin, synthetic_quality (optional)
    """
    df = diagnostics_df.copy()
    if df.index.name != "class_id":
        df = df.set_index("class_id") if "class_id" in df.columns else df

    baseline_acc = {c: float(df.loc[c, "baseline_acc"]) for c in class_ids}
    entropy = {c: float(df.loc[c, "prediction_entropy"]) for c in class_ids}

    if policy == "uniform":
        n = len(class_ids)
        per = total_budget // n
        rem = total_budget - per * n
        alloc = {cid: per for cid in class_ids}
        for i, cid in enumerate(sorted(class_ids)):
            if i < rem:
                alloc[cid] += 1
        return alloc, {}

    if policy == "hard_class":
        return hard_class_allocation(class_ids, baseline_acc, total_budget, min_floor, max_cap), {}

    if policy == "uncertainty":
        return uncertainty_allocation(class_ids, entropy, total_budget, min_floor, max_cap), {}

    if policy == "predicted_utility":
        if utility is None:
            raise ValueError("predicted_utility requires utility dict")
        for col in [
            "baseline_acc",
            "mean_confidence",
            "prediction_entropy",
            "feature_compactness",
            "feature_separation",
            "nearest_centroid_margin",
        ]:
            if col not in df.columns:
                df[col] = 0.0
        if "synthetic_quality" not in df.columns:
            df["synthetic_quality"] = 0.0
        feat = df.loc[class_ids][
            [
                "baseline_acc",
                "mean_confidence",
                "prediction_entropy",
                "feature_compactness",
                "feature_separation",
                "nearest_centroid_margin",
                "synthetic_quality",
            ]
        ].copy()
        return predicted_utility_allocation(
            class_ids, feat, utility, total_budget, min_floor, max_cap, cv_folds=cv_folds
        )

    raise ValueError(f"Unknown policy {policy}")


def save_allocation_csv(alloc: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "allocated_count"])
        for cid, n in sorted(alloc.items(), key=lambda x: x[0]):
            w.writerow([cid, n])


def load_allocation_csv(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["class_id"]] = int(row["allocated_count"])
    return out


def allocation_to_dict(path: Optional[Path]) -> Optional[Dict[str, int]]:
    if path is None or not path.exists():
        return None
    return load_allocation_csv(path)
