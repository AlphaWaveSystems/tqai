"""Palm: information-theoretic adaptive bit allocation for quantization.

Scores tokens by novelty and surprise to determine per-element compression
aggressiveness.  Tokens carrying redundant information get fewer bits;
novel or surprising tokens get more.

References:
    - TurboQuant: arXiv:2504.19874
    - APTQ attention-aware quantization: arXiv:2402.14866
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tqai.pipeline.base import ScoredEntry


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)


class _EMATracker:
    """Exponential moving average tracker for novelty baseline."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._count: int = 0

    def update(self, x) -> float:
        x_np = _to_numpy(x).astype(np.float64).ravel()

        if self._mean is None:
            self._mean = x_np.copy()
            self._var = np.ones_like(x_np)
            self._count = 1
            return 0.0

        diff = x_np - self._mean
        std = np.sqrt(self._var + 1e-10)
        novelty = float(np.mean(np.abs(diff) / std))

        alpha = self._alpha
        self._mean = (1 - alpha) * self._mean + alpha * x_np
        self._var = (1 - alpha) * self._var + alpha * diff * diff
        self._count += 1
        return novelty


class PalmScorer:
    """Score KV entries by novelty for adaptive bit allocation.

    Implements the ``Scorer`` protocol from ``tqai.pipeline.base``.

    Args:
        alpha: EMA decay for novelty baseline (default 0.5).
        ema_decay: Alias for *alpha* (for config compat).
        tier_boundaries: Score thresholds for 4-tier allocation.
        bits_per_tier: Bit widths per tier (ascending quality).
        warmup_steps: Steps before adaptive scoring kicks in.
    """

    name = "palm"

    def __init__(
        self,
        alpha: float = 0.5,
        ema_decay: float | None = None,
        tier_boundaries: list[float] | None = None,
        bits_per_tier: list[int] | None = None,
        warmup_steps: int = 10,
    ):
        self._alpha = ema_decay if ema_decay is not None else alpha
        self._tier_boundaries = tier_boundaries or [0.1, 0.3, 0.7]
        self._bits_per_tier = bits_per_tier or [2, 3, 4, 8]
        self._warmup_steps = warmup_steps
        self._tracker = _EMATracker(alpha=self._alpha)
        self._step_count = 0

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        self._step_count += 1
        novelty = self._tracker.update(x)

        # During warmup, assign mid-tier
        if self._step_count <= self._warmup_steps:
            tier = 2
            score_val = 0.5
        else:
            score_val = min(novelty, 2.0) / 2.0  # normalize to [0, 1]
            tier = self._get_tier(score_val)

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=tier,
                metadata={"novelty": novelty, "layer_idx": layer_idx},
            )
        ]

    def _get_tier(self, score: float) -> int:
        for tier, threshold in enumerate(self._tier_boundaries):
            if score < threshold:
                return tier
        return len(self._tier_boundaries)

    def get_bits(self, score: float) -> int:
        tier = self._get_tier(score)
        return self._bits_per_tier[min(tier, len(self._bits_per_tier) - 1)]

    def reset(self) -> None:
        self._tracker = _EMATracker(alpha=self._alpha)
        self._step_count = 0
