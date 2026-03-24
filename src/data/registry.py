"""Unified dataloaders from ExperimentConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.cifar100 import (
    CIFAR100FullTrainDataset,
    CIFAR100SubsetDataset,
    CIFAR100ValDataset,
    build_5pct_subset_index_cifar,
)
from src.data.tiny_imagenet import (
    TinyImageNetFullTrainDataset,
    TinyImageNetSubsetDataset,
    TinyImageNetValDataset,
    build_5pct_subset_index,
    verify_tiny_imagenet_structure,
)


def class_ids_in_label_order(class_to_idx: Dict[str, int]) -> list:
    return [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]


def build_real_train_subset(
    cfg: ExperimentConfig,
    train_transform: Callable,
) -> Tuple[object, Dict[str, int]]:
    """Training subset only (5% real), no DataLoader."""
    if cfg.dataset.name == "tiny_imagenet":
        verify_tiny_imagenet_structure(cfg.path_raw)
        if not cfg.path_subset_index.exists():
            build_5pct_subset_index(root=cfg.path_raw, output_csv=cfg.path_subset_index)
        root = cfg.path_raw
        train_ds = TinyImageNetSubsetDataset(
            root=root / "train",
            index_csv=cfg.path_subset_index,
            transform=train_transform,
            class_to_idx=None,
        )
        return train_ds, train_ds.class_to_idx
    if cfg.dataset.name == "cifar100":
        cifar_root = cfg.project_root / "data" / "raw" / "cifar100"
        if not cfg.path_subset_index.exists():
            build_5pct_subset_index_cifar(output_csv=cfg.path_subset_index, cifar_root=cifar_root)
        int_map = {i: i for i in range(100)}
        train_ds = CIFAR100SubsetDataset(
            index_csv=cfg.path_subset_index,
            cifar_root=cifar_root,
            transform=train_transform,
            train=True,
            class_to_idx=int_map,
        )
        class_to_idx = {f"{i:03d}": i for i in range(100)}
        return train_ds, class_to_idx
    raise ValueError(cfg.dataset.name)


def get_baseline_loaders(
    cfg: ExperimentConfig,
    train_transform: Callable,
    val_transform: Callable,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    nw = cfg.training.num_workers
    bs = cfg.training.batch_size
    pin = torch.cuda.is_available()

    if cfg.dataset.name == "tiny_imagenet":
        verify_tiny_imagenet_structure(cfg.path_raw)
        if not cfg.path_subset_index.exists():
            build_5pct_subset_index(root=cfg.path_raw, output_csv=cfg.path_subset_index)
        root = cfg.path_raw
        train_ds = TinyImageNetSubsetDataset(
            root=root / "train",
            index_csv=cfg.path_subset_index,
            transform=train_transform,
            class_to_idx=None,
        )
        class_to_idx = train_ds.class_to_idx
        val_ds = TinyImageNetValDataset(
            root=root,
            transform=val_transform,
            class_to_idx=class_to_idx,
        )
    elif cfg.dataset.name == "cifar100":
        cifar_root = cfg.project_root / "data" / "raw" / "cifar100"
        if not cfg.path_subset_index.exists():
            build_5pct_subset_index_cifar(output_csv=cfg.path_subset_index, cifar_root=cifar_root)
        int_map = {i: i for i in range(100)}
        train_ds = CIFAR100SubsetDataset(
            index_csv=cfg.path_subset_index,
            cifar_root=cifar_root,
            transform=train_transform,
            train=True,
            class_to_idx=int_map,
        )
        class_to_idx = {f"{i:03d}": i for i in range(100)}
        val_ds = CIFAR100ValDataset(
            cifar_root=cifar_root,
            transform=val_transform,
            class_to_idx=int_map,
        )
    else:
        raise ValueError(cfg.dataset.name)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )
    return train_loader, val_loader, class_to_idx


def get_ceiling_loaders(
    cfg: ExperimentConfig,
    train_transform: Callable,
    val_transform: Callable,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    nw = cfg.training.num_workers
    bs = cfg.training.batch_size
    pin = torch.cuda.is_available()

    if cfg.dataset.name == "tiny_imagenet":
        verify_tiny_imagenet_structure(cfg.path_raw)
        root = cfg.path_raw
        train_ds = TinyImageNetFullTrainDataset(
            root=root,
            transform=train_transform,
            class_to_idx=None,
        )
        class_to_idx = train_ds.class_to_idx
        val_ds = TinyImageNetValDataset(
            root=root,
            transform=val_transform,
            class_to_idx=class_to_idx,
        )
    elif cfg.dataset.name == "cifar100":
        cifar_root = cfg.project_root / "data" / "raw" / "cifar100"
        int_map = {i: i for i in range(100)}
        train_ds = CIFAR100FullTrainDataset(
            cifar_root=cifar_root,
            transform=train_transform,
            class_to_idx=int_map,
        )
        class_to_idx = {f"{i:03d}": i for i in range(100)}
        val_ds = CIFAR100ValDataset(
            cifar_root=cifar_root,
            transform=val_transform,
            class_to_idx=int_map,
        )
    else:
        raise ValueError(cfg.dataset.name)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin
    )
    return train_loader, val_loader, class_to_idx
