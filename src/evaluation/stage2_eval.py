"""Stage-2 evaluation: accuracy suite, ECE, corruptions (saved to run_dir)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import ExperimentConfig
from src.data.registry import build_real_train_subset
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_train_transform
from src.evaluation.eval_extras import (
    collect_features,
    collect_paired_features_two_models,
    covariance_top_eigenvalues,
    ece_from_logits_scaled,
    fit_temperature_scalar,
    linear_cka_features,
    linear_probe_sklearn_accuracy,
    plot_eigenvalues_overlay,
)
from src.metrics.metrics import ExpectedCalibrationError
from src.stage2.diagnostics import collect_val_predictions, per_class_accuracy


def _raw_val_dataset_for_corruption(cfg: ExperimentConfig, class_to_idx: Dict[str, int]):
    """Tiny or CIFAR val with tensor in [0,1] before norm — for corruption eval."""
    from PIL import Image
    from src.data.cifar100 import CIFAR100ValDataset
    from src.data.tiny_imagenet import TinyImageNetValDataset

    base = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(cfg.dataset.image_size),
            transforms.ToTensor(),
        ]
    )

    if cfg.dataset.name == "tiny_imagenet":
        return TinyImageNetValDataset(root=cfg.path_raw, transform=base, class_to_idx=class_to_idx)
    return CIFAR100ValDataset(
        cifar_root=cfg.project_root / "data" / "raw" / "cifar100",
        transform=base,
        class_to_idx={i: i for i in range(100)},
    )


def _corrupt_loader(cfg: ExperimentConfig, class_to_idx: Dict, name: str, batch_size: int):
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    c2i = class_to_idx if cfg.dataset.name == "tiny_imagenet" else {i: i for i in range(100)}
    ds = _raw_val_dataset_for_corruption(cfg, c2i)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    class CorruptDS(torch.utils.data.Dataset):
        def __init__(self, base_ds, corrupt_fn):
            self.base = base_ds
            self.fn = corrupt_fn

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, y, cid = self.base[i]
            x = self.fn(img)
            x = (x - mean) / std
            return x, y, cid

    if name == "clean":
        norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

        class CleanDS(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base
                self.norm = norm

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                img, y, cid = self.base[i]
                return self.norm(img), y, cid

        return DataLoader(
            CleanDS(ds),
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )

    if name == "gaussian_noise":

        def fn(x):
            return (x + 0.1 * torch.randn_like(x)).clamp(0, 1)

    elif name == "blur":
        blur = T.GaussianBlur(5)

        def fn(x):
            return blur(x)

    elif name == "brightness":

        def fn(x):
            return (x + 0.2).clamp(0, 1)

    else:
        raise ValueError(name)

    return DataLoader(
        CorruptDS(ds, fn),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )


@torch.no_grad()
def _accuracy_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def evaluate_stage2(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    class_to_idx: Dict[str, int],
    device: torch.device,
    run_dir: Optional[Path] = None,
    ref_model_for_cka: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Compute Top-1, macro, worst-k, ECE, corruptions, temperature scaling, probe, CKA, eigen-spectrum."""
    run_dir = Path(run_dir) if run_dir else None
    num_classes = cfg.dataset.num_classes
    logits, labels, feats = collect_val_predictions(model, val_loader, device, num_classes)
    preds = logits.argmax(axis=1)
    top1 = float((preds == labels).mean())
    pc = per_class_accuracy(logits, labels, num_classes)
    macro = float(np.mean(list(pc.values())))
    worst_k = cfg.metrics.worst_k_classes
    sorted_acc = sorted(pc.values())
    worst20 = float(np.mean(sorted_acc[: min(worst_k, len(sorted_acc))]))

    ece_computer = ExpectedCalibrationError(n_bins=cfg.metrics.ece_bins)
    lt = torch.from_numpy(logits)
    yl = torch.from_numpy(labels)
    ece, _, _, _ = ece_computer.compute_from_logits(lt, yl)

    t_star = fit_temperature_scalar(logits, labels)
    ece_cal = ece_from_logits_scaled(logits, labels, t_star, cfg.metrics.ece_bins)

    cor = {}
    for name in ["clean", "gaussian_noise", "blur", "brightness"]:
        cl = _corrupt_loader(cfg, class_to_idx, name, cfg.training.batch_size)
        cor[name] = _accuracy_loader(model, cl, device)
    mean_corrupt = float(np.mean([cor[k] for k in cor if k != "clean"]))

    # Linear probe (sklearn) on frozen features: train on 5% real subset, test on val features
    tr_t = get_train_transform(cfg.dataset.image_size)
    real_train_ds, _ = build_real_train_subset(cfg, tr_t)
    probe_train_loader = DataLoader(
        real_train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    max_tr = min(5000, len(real_train_ds))
    max_va = min(5000, len(val_loader.dataset))  # type: ignore[arg-type]
    tr_f, tr_y = collect_features(model, probe_train_loader, device, max_samples=max_tr)
    va_f, va_y = collect_features(model, val_loader, device, max_samples=max_va)
    probe_top1 = linear_probe_sklearn_accuracy(tr_f, tr_y, va_f, va_y)

    ev_top, eff_rank = covariance_top_eigenvalues(feats, k=40)

    cka_vs_ref: Optional[float] = None
    if ref_model_for_cka is not None:
        fa, fb = collect_paired_features_two_models(
            model,
            ref_model_for_cka,
            val_loader,
            device,
            max_samples=min(cfg.metrics.cka_subsample, 4096),
        )
        cka_vs_ref = linear_cka_features(fa, fb)

    out: Dict[str, Any] = {
        "top1": top1,
        "macro_acc": macro,
        "worst_k_acc": worst20,
        "ece": float(ece),
        "temperature_scaling": {
            "T": t_star,
            "ece_after_scaling": ece_cal,
        },
        "linear_probe_top1": probe_top1,
        "feature_cov_eigenvalues_top": ev_top[:20],
        "feature_effective_rank": eff_rank,
        "linear_cka_vs_ref": cka_vs_ref,
        "corruption": cor,
        "mean_corruption_acc": mean_corrupt,
        "per_class_acc": {str(k): float(v) for k, v in pc.items()},
    }
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_eigenvalues_overlay(
            ev_top[:30],
            run_dir / "eval_eigenvalues.png",
            title="Top covariance eigenvalues (val penultimate)",
        )
        if cka_vs_ref is not None:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(3, 2.5))
                im = ax.imshow([[cka_vs_ref]], vmin=0, vmax=1, cmap="magma")
                ax.set_xticks([0])
                ax.set_yticks([0])
                ax.set_xticklabels(["ref"])
                ax.set_yticklabels(["trained"])
                fig.colorbar(im, ax=ax)
                plt.title("Linear CKA vs ref (penultimate)")
                plt.tight_layout()
                plt.savefig(run_dir / "eval_cka_heatmap.png", dpi=200)
                plt.close()
            except Exception:  # noqa: BLE001
                pass
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out
