"""Sheaf Laplacian harmonicity scorer.

Classifies KV entries by computing the sheaf Laplacian of their vectors
on the spatial-temporal token graph.  Harmonious entries (smooth across
neighbors) are redundant and can be compressed aggressively.  Non-harmonious
entries carry local discontinuity information and should be preserved.

References:
    - Sheaf Attention & Local Consistency: AAAI 2026, arXiv:2601.21207
    - DiTFastAttn window attention justification: arXiv:2406.08552
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tqai.pipeline.base import ScoredEntry


class SheafScorer:
    """Score entries by sheaf Laplacian harmonicity.

    Computes how smoothly K/V vectors vary across neighboring token
    positions.  Low harmonicity (large Laplacian) → high information →
    high score (preserve).  High harmonicity (small Laplacian) → redundant →
    low score (compress).

    For 4D tensors ``[batch, heads, seq, dim]``, the Laplacian is computed
    along the sequence axis, treating consecutive tokens as neighbors.

    Args:
        neighborhood: Number of neighbors on each side for Laplacian
            computation (default 1, i.e., immediate neighbors).
        ema_decay: EMA decay for running harmonicity estimate.
    """

    name = "sheaf"

    def __init__(
        self,
        neighborhood: int = 1,
        ema_decay: float = 0.9,
    ):
        self._neighborhood = neighborhood
        self._ema_decay = ema_decay
        self._running_harmonicity: float | None = None
        self._step_count = 0

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        self._step_count += 1
        x_np = _to_numpy(x).astype(np.float64)

        # Compute discrete Laplacian along sequence axis
        harmonicity = self._compute_harmonicity(x_np)

        # Update running estimate
        if self._running_harmonicity is None:
            self._running_harmonicity = harmonicity
        else:
            alpha = self._ema_decay
            self._running_harmonicity = alpha * self._running_harmonicity + (1 - alpha) * harmonicity

        # Score: low harmonicity = non-smooth = high information = high score
        # Normalize: harmonicity in [0, inf), map to score in [0, 1]
        score_val = 1.0 - min(self._running_harmonicity, 2.0) / 2.0
        score_val = max(0.0, min(1.0, score_val))

        tier = _score_to_tier(score_val)

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=tier,
                metadata={
                    "harmonicity": harmonicity,
                    "running_harmonicity": self._running_harmonicity,
                    "layer_idx": layer_idx,
                },
            )
        ]

    def _compute_harmonicity(self, x_np: np.ndarray) -> float:
        """Compute discrete Laplacian harmonicity along sequence axis.

        Harmonicity measures how close each token is to the average of
        its neighbors.  Value of 0 = perfectly harmonic (constant signal).
        """
        # Ensure at least 3D for Laplacian computation
        if x_np.ndim < 2:
            return 0.0

        # Work on the sequence axis (axis=-2 for [batch, heads, seq, dim])
        seq_axis = -2 if x_np.ndim >= 3 else 0
        seq_len = x_np.shape[seq_axis]

        if seq_len < 3:
            return 0.0  # too short for meaningful Laplacian

        # Discrete Laplacian: L(x_i) = x_i - 0.5*(x_{i-1} + x_{i+1})
        # Compute on interior tokens
        slices_center = [slice(None)] * x_np.ndim
        slices_left = [slice(None)] * x_np.ndim
        slices_right = [slice(None)] * x_np.ndim

        slices_center[seq_axis] = slice(1, -1)
        slices_left[seq_axis] = slice(0, -2)
        slices_right[seq_axis] = slice(2, None)

        center = x_np[tuple(slices_center)]
        left = x_np[tuple(slices_left)]
        right = x_np[tuple(slices_right)]

        laplacian = center - 0.5 * (left + right)

        # Harmonicity: ratio of Laplacian energy to signal energy
        signal_energy = np.mean(center ** 2) + 1e-10
        laplacian_energy = np.mean(laplacian ** 2)
        harmonicity = 1.0 - min(laplacian_energy / signal_energy, 1.0)

        return float(harmonicity)

    def reset(self) -> None:
        self._running_harmonicity = None
        self._step_count = 0


def _score_to_tier(score: float) -> int:
    if score < 0.2:
        return 0
    if score < 0.4:
        return 1
    if score < 0.7:
        return 2
    return 3


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)
