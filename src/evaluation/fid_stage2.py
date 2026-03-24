"""
Global FID: 5% real subset vs uniformly subsampled synthetic pool at each budget ratio.
Uses clean-fid if installed (`pip install clean-fid`).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torchvision import transforms

from src.config import ExperimentConfig
from src.data.registry import build_real_train_subset, class_ids_in_label_order, get_baseline_loaders
from src.data.synthetic_dataset import list_synthetic_paths_uniform
from src.data.transforms import get_train_transform, get_val_transform
from src.migration.paths import get_resolved_synthetic_root


def _export_real_subset_images(cfg: ExperimentConfig, out_dir: Path) -> int:
    """Write PNGs from 5% train index into flat folder (class prefix in filename)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("*.png")):
        return sum(1 for _ in out_dir.glob("*.png"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_t = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(cfg.dataset.image_size),
            transforms.ToTensor(),
        ]
    )
    ds, _ = build_real_train_subset(cfg, raw_t)
    n = 0
    for idx in range(len(ds)):
        img, _, cid = ds[idx]
        cid = str(cid)
        arr = (img.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"{cid}_{idx:06d}.png")
        n += 1
    return n


def _export_synthetic_uniform(
    cfg: ExperimentConfig,
    synth_root: Path,
    k_per_class: int,
    out_dir: Path,
) -> int:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        return sum(1 for _ in out_dir.glob("*.png"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_t = get_train_transform(cfg.dataset.image_size)
    _, _, c2i = get_baseline_loaders(cfg, tr_t, get_val_transform(cfg.dataset.image_size))
    cids = class_ids_in_label_order(c2i)
    items = list_synthetic_paths_uniform(
        synth_root, cids, k_per_class, seed=cfg.generation.seed
    )
    for idx, (p, cid) in enumerate(items):
        shutil.copy2(p, out_dir / f"{cid}_{idx:06d}.png")
    return len(items)


def compute_global_fid_for_ratios(
    cfg: ExperimentConfig,
    ratios: Optional[List[int]] = None,
    force_resync_images: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Returns e.g. {"fid_5x": float, ...} or None if clean-fid missing / error.
    Caches image folders under results/{dataset}/fid_cache/.
    """
    try:
        from cleanfid import fid as clean_fid_mod  # type: ignore
    except ImportError:
        return {"_skipped": "clean-fid not installed (pip install clean-fid)"}

    ratios = ratios or [int(x) for x in cfg.synthetic_scaling.ratios]
    cache_root = cfg.path_results_root / cfg.dataset.name / "fid_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    real_dir = cache_root / "real_5pct_png"
    if force_resync_images and real_dir.exists():
        shutil.rmtree(real_dir)
    _export_real_subset_images(cfg, real_dir)

    synth_root = get_resolved_synthetic_root(cfg)
    out: Dict[str, Optional[float]] = {}
    for r in ratios:
        k = cfg.synthetic_scaling.per_class_at_ratio.get(str(r), r * cfg.dataset.real_images_per_class)
        syn_dir = cache_root / f"synth_uniform_{r}x_{k}pc"
        if force_resync_images and syn_dir.exists():
            shutil.rmtree(syn_dir)
        _export_synthetic_uniform(cfg, synth_root, int(k), syn_dir)
        try:
            score = float(
                clean_fid_mod.compute_fid(str(real_dir), str(syn_dir), mode="clean", num_workers=0)
            )
            out[f"fid_{r}x"] = score
        except Exception as e:  # noqa: BLE001
            out[f"fid_{r}x"] = None
            out[f"fid_{r}x_error"] = str(e)

    summary_path = cache_root / "fid_summary.json"
    summary_serializable = {
        k: v for k, v in out.items() if not str(k).endswith("_error") and isinstance(v, (float, int, type(None)))
    }
    summary_path.write_text(json.dumps(summary_serializable, indent=2), encoding="utf-8")
    return out
