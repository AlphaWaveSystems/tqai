"""Fuzzy/soft quantization codebook solver.

Replaces the hard nearest-centroid assignment in Lloyd-Max with
temperature-controlled soft membership (fuzzy C-means style).
Temperature annealing from soft to hard produces smoother optimization
and can escape local optima.

At high temperature, each sample contributes to all centroids with
soft weights.  As temperature anneals to zero, the algorithm converges
to hard assignment — equivalent to Lloyd-Max but potentially at a
better local optimum.

References:
    - DSQ (differentiable soft quantization): arXiv:1908.05033
    - IDE-LBG (fuzzy VQ with differential evolution): arXiv:1710.05311
    - Soft quantization via weight coupling: arXiv:2601.21219

No extra dependencies — pure numpy.
"""

from __future__ import annotations

import math

import numpy as np


def solve_fuzzy(
    d: int,
    bits: int,
    tau_init: float | None = None,
    tau_min: float = 1e-6,
    alpha: float = 0.95,
    n_samples: int = 100_000,
    max_iter: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize codebook via fuzzy C-means with temperature annealing.

    Args:
        d: Head dimension (coordinate variance = 1/d).
        bits: Number of quantization bits.
        tau_init: Initial temperature. If None, set to ``2.0 / d``.
        tau_min: Final temperature (near-hard assignment).
        alpha: Temperature decay per iteration: ``tau *= alpha``.
        n_samples: Monte Carlo sample size.
        max_iter: Maximum iterations.
        seed: RNG seed.

    Returns:
        centroids: float32 array ``(2^bits,)``, sorted ascending.
        boundaries: float32 array ``(2^bits - 1,)``.
    """
    from tqai.codebook.lloyd_max import solve_lloyd_max

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)

    if tau_init is None:
        tau_init = 2.0 / d

    # Generate fixed Monte Carlo samples from N(0, 1/d)
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, sigma, size=n_samples).astype(np.float64)

    # Initialize from Lloyd-Max
    centroids, _ = solve_lloyd_max(d, bits)
    centroids = centroids.astype(np.float64)

    tau = tau_init

    for _ in range(max_iter):
        # Compute soft membership weights: mu[i, k] = softmax(-|x_i - c_k|^2 / tau)
        # Shape: (n_samples, n_levels)
        dists_sq = (samples[:, None] - centroids[None, :]) ** 2
        logits = -dists_sq / max(tau, 1e-30)

        # Numerically stable softmax
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        mu = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Update centroids as weighted means
        weighted_sum = mu.T @ samples  # (n_levels,)
        weight_total = np.sum(mu, axis=0)  # (n_levels,)

        for k in range(n_levels):
            if weight_total[k] > 1e-30:
                centroids[k] = weighted_sum[k] / weight_total[k]

        # Sort to maintain ordering
        centroids = np.sort(centroids)

        # Anneal temperature
        tau = max(tau * alpha, tau_min)

        # Early stopping when effectively hard
        if tau <= tau_min:
            break

    centroids = centroids.astype(np.float32)
    boundaries = np.array(
        [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)],
        dtype=np.float32,
    )

    return centroids, boundaries
