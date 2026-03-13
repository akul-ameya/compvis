from typing import Iterable, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


def compute_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[float]:
    """Return accuracies for each k in topk in percentage."""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class ExpectedCalibrationError:
    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    def _compute(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = predictions == labels

        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        bin_acc = np.zeros(self.n_bins)
        bin_conf = np.zeros(self.n_bins)
        bin_frac = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if not np.any(mask):
                continue
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            frac = mask.mean()
            ece += np.abs(acc - conf) * frac
            bin_acc[i] = acc
            bin_conf[i] = conf
            bin_frac[i] = frac

        return ece, bin_boundaries, bin_acc, bin_conf, bin_frac

    def compute_from_logits(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        ece, bin_boundaries, bin_acc, bin_conf, _ = self._compute(probs, labels_np)
        return ece, bin_boundaries, bin_acc, bin_conf

    def reliability_diagram(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        save_path: str,
        title: str = "Reliability Diagram",
    ) -> float:
        ece, bin_boundaries, bin_acc, bin_conf = self.compute_from_logits(logits, labels)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.bar(bin_centers, bin_acc, width=1.0 / self.n_bins, alpha=0.7, edgecolor="black")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title(f"{title}\nECE={ece:.3f}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return ece


def collect_logits_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            all_logits.append(outputs.detach().cpu())
            all_labels.append(targets.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels


def evaluate_corruptions(
    model: nn.Module,
    base_dataloader: DataLoader,
    corruption_loader_fn,
    device: torch.device,
) -> dict:
    """
    Evaluate model accuracy under different corruption dataloaders.
    corruption_loader_fn: function(name) -> DataLoader
    """
    model.eval()
    results = {}
    for name in ["clean", "gaussian_noise", "blur", "brightness"]:
        dataloader = corruption_loader_fn(name)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets, _ in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        results[name] = correct / total if total > 0 else 0.0
    return results


def plot_robustness_barplot(results: dict, save_path: str) -> None:
    names = list(results.keys())
    values = [results[k] for k in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, values)
    plt.ylabel("Accuracy")
    plt.title("Robustness to Corruptions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_feature_cov_eigs(features: np.ndarray) -> np.ndarray:
    """
    features: (N, D) array from penultimate layer.
    Returns sorted eigenvalues (descending).
    """
    features_centered = features - features.mean(axis=0, keepdims=True)
    cov = np.cov(features_centered, rowvar=False)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals_sorted = np.sort(eigvals)[::-1]
    return eigvals_sorted


def plot_eig_spectrum(eigvals: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.semilogy(eigvals)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Covariance Eigenvalue Spectrum")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

