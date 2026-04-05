"""Objective functions for codebook optimization.

Provides pluggable objectives that codebook solvers (CMA-ES, fuzzy, etc.)
can use to evaluate candidate codebook quality.

References:
    - MSE distortion: standard Lloyd-Max objective
    - Attention-aware objective: inspired by APTQ (arXiv:2402.14866)
    - Cosine preservation: motivated by PolarQuant (arXiv:2502.02617)
"""

from __future__ import annotations

import numpy as np


def mse_objective(centroids: np.ndarray, samples: np.ndarray) -> float:
    """Mean squared error distortion: E[|X - Q(X)|²].

    Args:
        centroids: Sorted centroid array, shape ``(n_levels,)``.
        samples: 1D samples from the coordinate distribution, shape ``(N,)``.

    Returns:
        Average squared quantization error.
    """
    centroids_sorted = np.sort(centroids)
    # Assign each sample to its nearest centroid
    diffs = np.abs(samples[:, None] - centroids_sorted[None, :])
    indices = np.argmin(diffs, axis=1)
    reconstructed = centroids_sorted[indices]
    return float(np.mean((samples - reconstructed) ** 2))


def cosine_objective(
    centroids: np.ndarray,
    samples_x: np.ndarray,
    d: int,
    rotation: np.ndarray,
) -> float:
    """Cosine similarity loss through full quantize-dequantize pipeline.

    Simulates the full PolarQuant pipeline on random vectors and measures
    the average cosine distance (1 - cos_sim) between original and
    reconstructed vectors.

    Args:
        centroids: Sorted centroid array, shape ``(n_levels,)``.
        samples_x: Random vectors, shape ``(N, d)``, drawn from N(0, I/d).
        d: Head dimension.
        rotation: Orthogonal rotation matrix, shape ``(d, d)``.

    Returns:
        Average cosine distance (lower is better).
    """
    centroids_sorted = np.sort(centroids)
    # Normalize to unit sphere
    norms = np.linalg.norm(samples_x, axis=-1, keepdims=True)
    safe_norms = norms + 1e-10
    x_unit = samples_x / safe_norms

    # Rotate
    y = x_unit @ rotation.T

    # Quantize each coordinate
    diffs = np.abs(y[:, :, None] - centroids_sorted[None, None, :])
    indices = np.argmin(diffs, axis=-1)
    y_hat = centroids_sorted[indices]

    # Inverse rotate and scale
    x_hat = (y_hat @ rotation) * norms

    # Cosine similarity
    dot = np.sum(samples_x * x_hat, axis=-1)
    norm_orig = np.linalg.norm(samples_x, axis=-1)
    norm_hat = np.linalg.norm(x_hat, axis=-1)
    cos_sim = dot / (norm_orig * norm_hat + 1e-12)
    return float(1.0 - np.mean(cos_sim))


def attention_score_objective(
    centroids: np.ndarray,
    samples_q: np.ndarray,
    samples_k: np.ndarray,
    d: int,
    rotation: np.ndarray,
) -> float:
    """Attention score preservation loss.

    Measures how well attention distributions are preserved after
    quantizing key vectors.  Minimizes:
        E[||softmax(Q·K^T/√d) - softmax(Q·K_hat^T/√d)||²]

    Inspired by APTQ (arXiv:2402.14866) which showed attention-aware
    sensitivity metrics outperform per-layer MSE for quantization.

    Args:
        centroids: Sorted centroid array, shape ``(n_levels,)``.
        samples_q: Query vectors, shape ``(n_q, d)``.
        samples_k: Key vectors, shape ``(n_k, d)``.
        d: Head dimension.
        rotation: Orthogonal rotation matrix, shape ``(d, d)``.

    Returns:
        Average L2 distance between original and quantized softmax
        attention distributions (lower is better).
    """
    centroids_sorted = np.sort(centroids)
    scale = 1.0 / np.sqrt(d)

    # Original attention scores
    scores_orig = (samples_q @ samples_k.T) * scale

    # Quantize keys through full pipeline
    norms_k = np.linalg.norm(samples_k, axis=-1, keepdims=True)
    safe_norms = norms_k + 1e-10
    k_unit = samples_k / safe_norms
    y = k_unit @ rotation.T
    diffs = np.abs(y[:, :, None] - centroids_sorted[None, None, :])
    indices = np.argmin(diffs, axis=-1)
    y_hat = centroids_sorted[indices]
    k_hat = (y_hat @ rotation) * norms_k

    # Quantized attention scores
    scores_quant = (samples_q @ k_hat.T) * scale

    # Softmax (numerically stable)
    def _softmax(x):
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attn_orig = _softmax(scores_orig)
    attn_quant = _softmax(scores_quant)

    return float(np.mean(np.sum((attn_orig - attn_quant) ** 2, axis=-1)))
