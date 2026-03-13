"""
Centered Kernel Alignment (CKA) for comparing neural network representations.
Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
"""

from typing import Optional

import numpy as np


def _centering_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def linear_kernel(X: np.ndarray) -> np.ndarray:
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        - 2 * X @ X.T
        + np.sum(X ** 2, axis=1, keepdims=True).T
    )
    if sigma is None:
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]))
    return np.exp(-sq_dists / (2 * sigma ** 2))


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    H = _centering_matrix(n)
    return float(np.trace(K @ H @ L @ H) / ((n - 1) ** 2))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices X (n, d1) and Y (n, d2).
    Both must have the same number of samples n.
    """
    assert X.shape[0] == Y.shape[0], "Sample counts must match"
    Kx = linear_kernel(X)
    Ky = linear_kernel(Y)
    return hsic(Kx, Ky) / np.sqrt(hsic(Kx, Kx) * hsic(Ky, Ky))


def rbf_cka(
    X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None
) -> float:
    assert X.shape[0] == Y.shape[0], "Sample counts must match"
    Kx = rbf_kernel(X, sigma)
    Ky = rbf_kernel(Y, sigma)
    return hsic(Kx, Ky) / np.sqrt(hsic(Kx, Kx) * hsic(Ky, Ky))


def cka_matrix(
    feature_dict: dict, kernel: str = "linear"
) -> tuple:
    """
    Compute pairwise CKA between all models in feature_dict.
    feature_dict: {"model_name": features_array (n, d), ...}
    Returns: (names list, cka_matrix ndarray)
    """
    names = list(feature_dict.keys())
    n = len(names)
    mat = np.ones((n, n))
    cka_fn = linear_cka if kernel == "linear" else rbf_cka
    for i in range(n):
        for j in range(i + 1, n):
            val = cka_fn(feature_dict[names[i]], feature_dict[names[j]])
            mat[i, j] = val
            mat[j, i] = val
    return names, mat
