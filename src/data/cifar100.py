"""CIFAR-100: 5% stratified subset index and PyTorch datasets (224×224, ImageNet norm)."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100

from src.data.tiny_imagenet import get_data_root


def get_cifar100_root(project_root: Optional[Path] = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return Path(project_root) / "data" / "raw" / "cifar100"


def build_5pct_subset_index_cifar(
    output_csv: Optional[Path] = None,
    samples_per_class: int = 25,
    seed: int = 42,
    cifar_root: Optional[Path] = None,
) -> Path:
    if cifar_root is None:
        cifar_root = get_cifar100_root()
    cifar_root.mkdir(parents=True, exist_ok=True)
    ds = CIFAR100(root=cifar_root, train=True, download=True)

    if output_csv is None:
        output_csv = get_data_root() / "cifar100_5pct" / "train_index.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    by_class: Dict[int, List[int]] = {c: [] for c in range(100)}
    for idx, (_, y) in enumerate(ds):
        by_class[int(y)].append(idx)

    rows: List[Tuple[int, int]] = []
    for c in range(100):
        idxs = by_class[c]
        if len(idxs) < samples_per_class:
            raise ValueError(f"CIFAR class {c} has only {len(idxs)} samples")
        picked = random.sample(idxs, samples_per_class)
        for i in picked:
            rows.append((i, c))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "class_id"])
        w.writerows(rows)

    print(f"Wrote CIFAR-100 5% index ({len(rows)} rows) to {output_csv}")
    return output_csv


class CIFAR100SubsetDataset(Dataset):
    """Train subset from CSV (sample_index, class_id)."""

    def __init__(
        self,
        index_csv: Path,
        cifar_root: Path,
        transform: Optional[Callable] = None,
        train: bool = True,
        class_to_idx: Optional[Dict[int, int]] = None,
    ) -> None:
        self.cifar_root = Path(cifar_root)
        self.transform = transform
        self.base = CIFAR100(root=self.cifar_root, train=train, download=True)

        with Path(index_csv).open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.entries = [(int(r["sample_index"]), int(r["class_id"])) for r in reader]

        if class_to_idx is None:
            self.class_to_idx = {i: i for i in range(100)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        si, cid = self.entries[idx]
        img, y = self.base[si]
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[cid]
        cid_str = f"{cid:03d}"
        return img, label, cid_str


class CIFAR100ValDataset(Dataset):
    """Official CIFAR-100 test split as validation."""

    def __init__(
        self,
        cifar_root: Path,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[int, int]] = None,
    ) -> None:
        self.cifar_root = Path(cifar_root)
        self.transform = transform
        self.base = CIFAR100(root=self.cifar_root, train=False, download=True)
        if class_to_idx is None:
            self.class_to_idx = {i: i for i in range(100)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        y = int(y)
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[y]
        return img, label, f"{y:03d}"


class CIFAR100FullTrainDataset(Dataset):
    """All 50k training images."""

    def __init__(
        self,
        cifar_root: Path,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[int, int]] = None,
    ) -> None:
        self.cifar_root = Path(cifar_root)
        self.transform = transform
        self.base = CIFAR100(root=self.cifar_root, train=True, download=True)
        if class_to_idx is None:
            self.class_to_idx = {i: i for i in range(100)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        y = int(y)
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[y]
        return img, label, f"{y:03d}"


def create_cifar100_loaders(
    project_root: Path,
    subset_csv: Path,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    cifar_root = project_root / "data" / "raw" / "cifar100"
    if not subset_csv.exists():
        build_5pct_subset_index_cifar(output_csv=subset_csv, cifar_root=cifar_root)

    class_to_idx = {str(i).zfill(3): i for i in range(100)}
    # Use integer class id 0..99 as keys matching __getitem__ cid_str
    int_map = {i: i for i in range(100)}

    train_ds = CIFAR100SubsetDataset(
        index_csv=subset_csv,
        cifar_root=cifar_root,
        transform=train_transform,
        train=True,
        class_to_idx=int_map,
    )
    val_ds = CIFAR100ValDataset(
        cifar_root=cifar_root,
        transform=val_transform,
        class_to_idx=int_map,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # class_to_idx for models: wnid-style strings
    str_map = {f"{i:03d}": i for i in range(100)}
    return train_loader, val_loader, str_map
