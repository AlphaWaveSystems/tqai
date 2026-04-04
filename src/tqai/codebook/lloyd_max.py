"""Lloyd-Max optimal scalar quantizer for rotated coordinate distributions.

After random orthogonal rotation, each coordinate of a d-dimensional unit vector
follows a distribution well-approximated by N(0, 1/d) for d >= 64. The Lloyd-Max
algorithm finds the optimal set of 2^b centroids (reconstruction levels) and
2^b - 1 boundaries (decision thresholds) that minimise MSE for this distribution.

Reference: TurboQuant (arXiv:2504.19874), PolarQuant (arXiv:2502.02617).
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_pdf(x: float, d: int) -> float:
    """PDF of N(0, 1/d) — the coordinate distribution after rotation."""
    sigma2 = 1.0 / d
    return math.exp(-0.5 * x * x / sigma2) / math.sqrt(2.0 * math.pi * sigma2)


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal Lloyd-Max codebook for rotated coordinate distribution.

    Args:
        d: Head dimension (determines variance = 1/d).
        bits: Number of quantization bits.
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on distortion change.

    Returns:
        centroids: float64 array of shape (2^bits,), sorted ascending.
        boundaries: float64 array of shape (2^bits - 1,), sorted ascending.
    """
    from scipy.integrate import quad

    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)

    # Initial uniform partition over [-4*sigma, 4*sigma]
    lo, hi = -4.0 * sigma, 4.0 * sigma
    boundaries = np.linspace(lo, hi, n_levels + 1)
    # Interior boundaries only
    inner_bounds = boundaries[1:-1].copy()
    centroids = np.zeros(n_levels, dtype=np.float64)

    def pdf(x):
        return gaussian_pdf(x, d)

    def _centroid(a: float, b: float) -> float:
        """Conditional expectation E[X | a <= X < b] under pdf."""
        num, _ = quad(lambda x: x * pdf(x), a, b)
        den, _ = quad(pdf, a, b)
        if den < 1e-30:
            return (a + b) / 2.0
        return num / den

    prev_distortion = float("inf")

    for _ in range(max_iter):
        # Full boundary list with sentinels
        full_bounds = np.concatenate([[-np.inf], inner_bounds, [np.inf]])

        # Step 1: Update centroids (conditional expectations)
        for j in range(n_levels):
            centroids[j] = _centroid(full_bounds[j], full_bounds[j + 1])

        # Step 2: Update boundaries (midpoints of adjacent centroids)
        for j in range(n_levels - 1):
            inner_bounds[j] = (centroids[j] + centroids[j + 1]) / 2.0

        # Check convergence via distortion
        full_bounds = np.concatenate([[-np.inf], inner_bounds, [np.inf]])
        distortion = 0.0
        for j in range(n_levels):
            c = centroids[j]
            d_j, _ = quad(lambda x, c=c: (x - c) ** 2 * pdf(x), full_bounds[j], full_bounds[j + 1])
            distortion += d_j

        if abs(prev_distortion - distortion) < tol:
            break
        prev_distortion = distortion

    return centroids.astype(np.float32), inner_bounds.astype(np.float32)
