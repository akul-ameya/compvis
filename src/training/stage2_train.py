"""Training with cosine scheduler, early stopping, JSON history, run directory."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.training.synthetic_loss import compute_real_class_centroids, train_one_epoch_synthetic_aware
from src.training.train_eval import evaluate, train_one_epoch


def _plateau_stop(val_accs: List[float], window: int = 5, tol: float = 1e-2) -> bool:
    """Return True when the last *window* val accuracies span < *tol*."""
    if len(val_accs) < window:
        return False
    recent = val_accs[-window:]
    return (max(recent) - min(recent)) < tol


def train_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    mixed_precision: bool = True,
    early_stopping_patience: int = 7,
    synthetic_aware: bool = False,
    frozen_reference_model: Optional[nn.Module] = None,
    centroid_loader: Optional[DataLoader] = None,
    num_classes: int = 200,
    synthetic_lambda: float = 5.0,
) -> Dict[str, List[float]]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=mixed_precision)

    use_syn = bool(
        synthetic_aware
        and frozen_reference_model is not None
        and centroid_loader is not None
    )
    centroids: Optional[torch.Tensor] = None
    if use_syn:
        frozen_reference_model.eval()
        centroids = compute_real_class_centroids(
            frozen_reference_model, centroid_loader, device, num_classes
        )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top1": [],
        "lr": [],
    }
    best_path = run_dir / "best.pt"
    final_path = run_dir / "final.pt"
    ckpt_path = run_dir / "checkpoint.pt"

    model.to(device)
    best_val = 0.0
    start_epoch = 1
    wall_elapsed = 0.0

    # ── Resume from checkpoint if interrupted ────────────────────
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history = ckpt["history"]
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"] + 1
        wall_elapsed = ckpt.get("wall_elapsed", 0.0)
        print(f"  ↻ Resuming from epoch {start_epoch} (best_val={best_val:.4f})")

    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(cur_lr)
        if use_syn and centroids is not None and frozen_reference_model is not None:
            tl, ta = train_one_epoch_synthetic_aware(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                frozen_reference_model,
                centroids,
                synthetic_lambda,
                mixed_precision=mixed_precision,
            )
        else:
            tl, ta = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                mixed_precision=mixed_precision,
            )
        vl, va = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_top1"].append(va)

        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)

        scheduler.step()

        print(f"Epoch {epoch}/{epochs}  train_loss={tl:.4f} acc={ta:.4f}  val_loss={vl:.4f} acc={va:.4f}")

        # Save resumable checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_val": best_val,
                "wall_elapsed": wall_elapsed + (time.time() - t0),
            },
            ckpt_path,
        )

        # Plateau early stopping: last 5 epochs within 0.01 pp
        if _plateau_stop(history["val_acc"], window=5, tol=1e-2):
            print(f"  Early stopping at epoch {epoch} (plateau: last 5 val_acc range < 0.01%)")
            break

    torch.save(model.state_dict(), final_path)
    wall = wall_elapsed + (time.time() - t0)
    curves = {k: v for k, v in history.items() if isinstance(v, list)}
    with (run_dir / "training_curves.json").open("w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "wall_time_s": wall,
                "best_val_top1": best_val,
                "synthetic_aware_loss_active": use_syn,
                "synthetic_lambda": synthetic_lambda if use_syn else None,
            },
            f,
            indent=2,
        )
    # Clean up checkpoint after successful completion
    if ckpt_path.is_file():
        ckpt_path.unlink()

    history["wall_time_s"] = wall
    history["best_val_top1"] = best_val
    return history
