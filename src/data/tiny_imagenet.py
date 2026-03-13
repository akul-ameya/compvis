import csv
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_tiny_imagenet_root() -> Path:
    return get_data_root() / "raw" / "tiny-imagenet-200"


def download_tiny_imagenet(dest_root: Optional[Path] = None) -> Path:
    """
    Download and extract Tiny ImageNet if it does not already exist.
    This is written to be idempotent; it will not re-download if the folder exists.
    """
    import zipfile
    from urllib.request import urlretrieve

    if dest_root is None:
        dest_root = get_data_root() / "raw"
    dest_root.mkdir(parents=True, exist_ok=True)

    target_dir = dest_root / "tiny-imagenet-200"
    if target_dir.exists():
        return target_dir

    zip_path = dest_root / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet to {zip_path}...")
        urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_root)

    return target_dir


def verify_tiny_imagenet_structure(root: Optional[Path] = None) -> bool:
    if root is None:
        root = get_tiny_imagenet_root()

    train_dir = root / "train"
    val_dir = root / "val"
    val_annotations = val_dir / "val_annotations.txt"

    if not train_dir.exists() or not val_dir.exists() or not val_annotations.exists():
        raise FileNotFoundError(
            f"Expected Tiny ImageNet structure under {root}, "
            "but train/ or val/ or val_annotations.txt is missing."
        )

    return True


def build_5pct_subset_index(
    root: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    samples_per_class: int = 25,
    seed: int = 42,
) -> Path:
    """
    Build a deterministic 5% subset index for Tiny ImageNet train split.
    The index CSV will have columns: class_id, image_path (relative to train root).
    """
    if root is None:
        root = get_tiny_imagenet_root()
    train_root = root / "train"

    if output_csv is None:
        output_csv = get_data_root() / "tiny_imagenet_5pct" / "train_index.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    rows: List[Tuple[str, str]] = []
    class_ids = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

    for class_id in class_ids:
        images_dir = train_root / class_id / "images"
        image_files = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
        )
        if len(image_files) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(image_files)} images, "
                f"cannot sample {samples_per_class}."
            )

        sampled = random.sample(image_files, samples_per_class)
        for img_path in sampled:
            rel_path = img_path.relative_to(train_root)
            rows.append((class_id, str(rel_path)))

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "image_path"])
        writer.writerows(rows)

    print(f"Wrote 5% subset index with {len(rows)} entries to {output_csv}")
    return output_csv


class TinyImageNetSubsetDataset(Dataset):
    """
    Tiny ImageNet train subset defined by an index CSV.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        index_csv: Optional[Path] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if root is None:
            root = get_tiny_imagenet_root() / "train"
        if index_csv is None:
            index_csv = get_data_root() / "tiny_imagenet_5pct" / "train_index.csv"

        self.root = Path(root)
        self.index_csv = Path(index_csv)
        self.transform = transform

        with self.index_csv.open("r") as f:
            reader = csv.DictReader(f)
            self.entries = list(reader)

        if class_to_idx is None:
            class_ids = sorted({row["class_id"] for row in self.entries})
            self.class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        class_id = entry["class_id"]
        rel_path = entry["image_path"]
        img_path = self.root / rel_path

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.class_to_idx[class_id]
        return img, label, class_id


class TinyImageNetValDataset(Dataset):
    """
    Tiny ImageNet validation set, using val_annotations.txt for labels.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if root is None:
            root = get_tiny_imagenet_root()

        self.root = Path(root)
        self.transform = transform

        val_dir = self.root / "val"
        images_dir = val_dir / "images"
        annotations_path = val_dir / "val_annotations.txt"

        if not annotations_path.exists():
            raise FileNotFoundError(f"Missing {annotations_path}")

        self.samples: List[Tuple[Path, str]] = []
        with annotations_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename, class_id = parts[0], parts[1]
                self.samples.append((images_dir / filename, class_id))

        if class_to_idx is None:
            class_ids = sorted({cid for _, cid in self.samples})
            self.class_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_id = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.class_to_idx[class_id]
        return img, label, class_id


class TinyImageNetFullTrainDataset(Dataset):
    """
    Full Tiny ImageNet training set (100% — 200 classes x 500 images = 100,000 images).
    Used as the performance ceiling in Stage 2 experiments.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if root is None:
            root = get_tiny_imagenet_root()
        train_root = Path(root) / "train"
        self.transform = transform

        self.samples: List[Tuple[Path, str]] = []
        class_dirs = sorted([d for d in train_root.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            cid = class_dir.name
            images_dir = class_dir / "images"
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((img_path, cid))

        if class_to_idx is None:
            cids = sorted({cid for _, cid in self.samples})
            self.class_to_idx = {c: i for i, c in enumerate(cids)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_id = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[class_id]
        return img, label, class_id


def create_dataloaders(
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 64,
    num_workers: int = 4,
    subset_index_csv: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create train/val dataloaders for the 5% Tiny ImageNet subset (train)
    and the full validation set (val).
    """
    root = get_tiny_imagenet_root()

    # Build index if needed
    if subset_index_csv is None:
        subset_index_csv = get_data_root() / "tiny_imagenet_5pct" / "train_index.csv"
    if not subset_index_csv.exists():
        build_5pct_subset_index(root=root, output_csv=subset_index_csv)

    # Shared class_to_idx mapping for train and val
    train_dataset = TinyImageNetSubsetDataset(
        root=root / "train",
        index_csv=subset_index_csv,
        transform=train_transform,
        class_to_idx=None,
    )
    class_to_idx = train_dataset.class_to_idx

    val_dataset = TinyImageNetValDataset(
        root=root,
        transform=val_transform,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_to_idx

