"""Load cached synthetic images from class subfolders (Tiny wnid or CIFAR 000-099)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


class SyntheticImageListDataset(Dataset):
    """(path, class_id_str) pairs with optional integer label via class_to_idx."""

    def __init__(
        self,
        items: List[Tuple[Path, str]],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.items = items
        self.transform = transform
        if class_to_idx is None:
            ids = sorted({cid for _, cid in items})
            self.class_to_idx = {c: i for i, c in enumerate(ids)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, cid = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[cid]
        return img, label, cid


def list_synthetic_paths_uniform(
    synthetic_root: Path,
    class_ids: List[str],
    k_per_class: int,
    seed: int = 42,
) -> List[Tuple[Path, str]]:
    rng = random.Random(seed)
    items: List[Tuple[Path, str]] = []
    root = Path(synthetic_root)
    for cid in class_ids:
        folder = root / cid
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.png"))
        if len(files) < k_per_class:
            raise ValueError(f"Synthetic class {cid}: need {k_per_class}, found {len(files)}")
        chosen = rng.sample(files, k_per_class)
        for p in chosen:
            items.append((p, cid))
    return items


def list_synthetic_paths_from_allocation(
    synthetic_root: Path,
    allocation: Dict[str, int],
    seed: int = 42,
) -> List[Tuple[Path, str]]:
    rng = random.Random(seed)
    items: List[Tuple[Path, str]] = []
    root = Path(synthetic_root)
    for cid, n in allocation.items():
        if n <= 0:
            continue
        folder = root / cid
        files = sorted(folder.glob("*.png"))
        if len(files) < n:
            raise ValueError(f"Synthetic class {cid}: need {n}, found {len(files)}")
        chosen = rng.sample(files, n)
        for p in chosen:
            items.append((p, cid))
    return items


def concat_real_synthetic(
    real_ds: Dataset,
    synth_ds: Dataset,
) -> ConcatDataset:
    return ConcatDataset([real_ds, synth_ds])


class RealSyntheticMixDataset(Dataset):
    """
    Concatenation of real + synthetic with a 4th tensor flag (1.0 = synthetic, 0.0 = real)
    for synthetic-aware weighted CE.
    """

    def __init__(self, real_ds: Dataset, synth_ds: Dataset) -> None:
        self.real_ds = real_ds
        self.synth_ds = synth_ds

    def __len__(self) -> int:
        return len(self.real_ds) + len(self.synth_ds)

    def __getitem__(self, idx: int):
        if idx < len(self.real_ds):
            img, y, cid = self.real_ds[idx]
            return img, y, cid, torch.tensor(0.0, dtype=torch.float32)
        j = idx - len(self.real_ds)
        img, y, cid = self.synth_ds[j]
        return img, y, cid, torch.tensor(1.0, dtype=torch.float32)
