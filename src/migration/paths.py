"""
One-time layout migration: legacy synthetic_sd -> canonical data/synthetic/tiny_imagenet,
and old checkpoints -> results/.../legacy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import ExperimentConfig


def _has_class_png_layout(root: Path) -> bool:
    if not root.is_dir():
        return False
    subs = [d for d in root.iterdir() if d.is_dir()]
    if len(subs) < 2:
        return False
    for d in subs[:5]:
        if any(d.glob("*.png")):
            return True
    return False


def get_resolved_synthetic_root(cfg: ExperimentConfig) -> Path:
    """
    Return directory containing <class_id>/*.png for synthetic images.
    Prefers canonical synthetic_root; falls back to legacy path or sidecar redirect.
    """
    canonical = cfg.path_synthetic
    legacy = (cfg.project_root / cfg.paths.synthetic_legacy).resolve() if cfg.paths.synthetic_legacy else None

    if _has_class_png_layout(canonical):
        return canonical

    sidecar = canonical.parent / ".synthetic_redirect.txt"
    if sidecar.is_file():
        target = Path(sidecar.read_text(encoding="utf-8").strip())
        if target.is_dir() and _has_class_png_layout(target):
            return target

    if legacy and legacy.is_dir() and _has_class_png_layout(legacy):
        return legacy

    return canonical


def migrate_tiny_synthetic_layout(cfg: ExperimentConfig) -> Path:
    """
    Ensure Tiny ImageNet synthetic images are discoverable at cfg.path_synthetic or via redirect.
    Does not move image bytes: symlink, or write .synthetic_redirect.txt to legacy path.
    """
    canonical = cfg.path_synthetic
    legacy = (cfg.project_root / cfg.paths.synthetic_legacy).resolve() if cfg.paths.synthetic_legacy else None

    if _has_class_png_layout(canonical):
        return canonical

    if not legacy or not legacy.is_dir() or not _has_class_png_layout(legacy):
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)

    if _has_class_png_layout(canonical):
        return canonical

    try:
        if canonical.exists() or canonical.is_symlink():
            if canonical.is_symlink() or canonical.is_dir():
                return get_resolved_synthetic_root(cfg)
        canonical.symlink_to(legacy, target_is_directory=True)
        return canonical
    except OSError:
        sidecar = canonical.parent / ".synthetic_redirect.txt"
        sidecar.write_text(str(legacy.resolve()), encoding="utf-8")
        return legacy


def link_legacy_checkpoints(cfg: ExperimentConfig) -> Optional[Path]:
    """
    Symlink results/{dataset}/legacy/checkpoints -> project checkpoints/ so Stage 1 artifacts stay in place.
    No file moves (avoids breaking paths).
    """
    ckpt = (cfg.project_root / cfg.paths.checkpoints_legacy).resolve()
    if not ckpt.is_dir():
        return None
    dest = cfg.path_results_root / cfg.dataset.name / "legacy" / "checkpoints"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return dest.resolve()
    try:
        dest.symlink_to(ckpt, target_is_directory=True)
    except OSError:
        return ckpt
    return dest.resolve()
