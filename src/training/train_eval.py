from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.metrics import compute_accuracy


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    mixed_precision: bool = True,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0

    for images, targets, _ in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if mixed_precision:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        acc1 = compute_accuracy(outputs, targets)[0]
        running_correct += acc1 / 100.0 * batch_size
        running_total += batch_size

    return running_loss / running_total, running_correct / running_total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            acc1 = compute_accuracy(outputs, targets)[0]
            running_correct += acc1 / 100.0 * batch_size
            running_total += batch_size

    return running_loss / running_total, running_correct / running_total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
    mixed_precision: bool = True,
    checkpoint_path: Optional[Path] = None,
    scheduler: Optional[object] = None,
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, List[float]]:
    model.to(device)
    scaler = GradScaler(enabled=mixed_precision)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        print(f"Epoch {epoch}/{epochs}  (lr={current_lr:.6f})")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            mixed_precision=mixed_precision,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"  Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Saved best checkpoint ({val_acc:.4f})")

        if scheduler is not None:
            scheduler.step()

        if early_stopping is not None:
            if early_stopping.step(val_acc):
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stopping.patience} epochs)")
                break

    return history
