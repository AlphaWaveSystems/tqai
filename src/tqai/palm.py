"""Palm: information-theoretic adaptive bit allocation for quantization.

Scores tokens (or activations) by novelty and surprise to determine
per-element compression aggressiveness.  Tokens that carry redundant
information get fewer bits; novel or surprising tokens get more.

Novelty: how different is this token from the recent running average?
Surprise: how much information was lost by quantization?

The combined info_score maps to bit-allocation tiers:
- Tier 0 (redundant): 2-bit, aggressive compression
- Tier 1 (expected): 3-bit, standard
- Tier 2 (novel): 4-bit, preserve detail
- Tier 3 (critical): 8-bit or skip compression

For diffusion transformers, the info_score can be driven by the
denoising schedule SNR instead of per-token statistics.

References:
    - TurboQuant: arXiv:2504.19874
    - APTQ attention-aware quantization: arXiv:2402.14866
    - Online softmax for numerical stability: Milakov & Gimelshein (2018)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PalmConfig:
    """Configuration for Palm adaptive bit allocation."""

    alpha: float = 0.1  # EMA decay for novelty baseline
    tier_boundaries: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.7]
    )
    bits_per_tier: list[int] = field(
        default_factory=lambda: [2, 3, 4, 8]
    )
    warmup_steps: int = 10  # steps before adaptive kicks in


class EMATracker:
    """Exponential moving average tracker for novelty baseline.

    Maintains a running average of vector norms/features to detect
    when a new input deviates significantly from recent history.
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._mean: Any = None
        self._var: Any = None
        self._count: int = 0

    def update(self, x) -> float:
        """Update EMA and return novelty score (normalized deviation).

        Args:
            x: Input array/tensor of any shape.

        Returns:
            Novelty score in [0, inf). Values > 1.0 indicate the input
            deviates more than one standard deviation from the running mean.
        """
        x_np = _to_numpy(x).astype(np.float64)
        x_flat = x_np.ravel()

        if self._mean is None:
            self._mean = x_flat.copy()
            self._var = np.ones_like(x_flat)
            self._count = 1
            return 0.0

        # Novelty: normalized distance from running mean
        diff = x_flat - self._mean
        std = np.sqrt(self._var + 1e-10)
        novelty = float(np.mean(np.abs(diff) / std))

        # Update EMA
        alpha = self._alpha
        self._mean = (1 - alpha) * self._mean + alpha * x_flat
        self._var = (1 - alpha) * self._var + alpha * diff * diff
        self._count += 1

        return novelty

    @property
    def count(self) -> int:
        return self._count


class PalmScorer:
    """Score inputs by novelty and surprise for adaptive bit allocation.

    Usage::

        scorer = PalmScorer(PalmConfig())

        # Per-token or per-activation scoring
        info_score = scorer.score(x)
        tier = scorer.get_tier(info_score)
        bits = scorer.get_bits(info_score)

        # For diffusion: score by denoising step
        info_score = scorer.score_diffusion_step(step_idx, total_steps)
    """

    def __init__(self, config: PalmConfig | None = None):
        self._config = config or PalmConfig()
        self._novelty_tracker = EMATracker(alpha=self._config.alpha)
        self._surprise_history: list[float] = []

    def score(self, x, quantization_residual: float | None = None) -> float:
        """Compute combined info_score for an input.

        Args:
            x: Input tensor/array.
            quantization_residual: Optional MSE from previous quantization
                (used as surprise signal).

        Returns:
            info_score in [0, inf). Higher = more information, needs more bits.
        """
        novelty = self._novelty_tracker.update(x)

        surprise = 0.0
        if quantization_residual is not None:
            surprise = math.log1p(quantization_residual)
            self._surprise_history.append(surprise)

        # Combined score: weighted geometric mean
        if surprise > 0:
            info_score = math.sqrt(novelty * surprise)
        else:
            info_score = novelty

        return info_score

    def score_diffusion_step(
        self, step_idx: int, total_steps: int, snr: float | None = None
    ) -> float:
        """Score a denoising step for adaptive bit allocation.

        Early steps (high noise) → high info_score → more bits.
        Late steps (refinement) → low info_score → fewer bits.

        Args:
            step_idx: Current denoising step (0 = most noise).
            total_steps: Total number of denoising steps.
            snr: Optional signal-to-noise ratio from the scheduler.

        Returns:
            info_score in [0, 1]. Maps linearly from early (1.0) to late (0.0).
        """
        if snr is not None:
            # SNR-based: high SNR = clean = low info, low SNR = noisy = high info
            return 1.0 / (1.0 + snr)

        # Linear schedule fallback
        progress = step_idx / max(total_steps - 1, 1)
        return 1.0 - progress

    def get_tier(self, info_score: float) -> int:
        """Map info_score to a compression tier (0 = most aggressive)."""
        boundaries = self._config.tier_boundaries
        for tier, threshold in enumerate(boundaries):
            if info_score < threshold:
                return tier
        return len(boundaries)  # highest tier

    def get_bits(self, info_score: float) -> int:
        """Map info_score to bit width via tier lookup."""
        tier = self.get_tier(info_score)
        bits_list = self._config.bits_per_tier
        return bits_list[min(tier, len(bits_list) - 1)]

    def reset(self) -> None:
        """Reset scorer state (e.g., between videos/prompts)."""
        self._novelty_tracker = EMATracker(alpha=self._config.alpha)
        self._surprise_history.clear()


def _to_numpy(x) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)
