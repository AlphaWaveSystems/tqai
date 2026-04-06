"""Diffusion SNR schedule scorer.

Scores denoising steps by their signal-to-noise ratio.  Early steps
(high noise, low SNR) get more bits; late steps (low noise, high SNR)
get fewer bits.

References:
    - Min-SNR weighting: arXiv:2303.09556
    - TurboQuant: arXiv:2504.19874
"""

from __future__ import annotations

import math
from typing import Any

from tqai.pipeline.base import ScoredEntry


class SNRScorer:
    """Score entries by diffusion schedule SNR.

    Implements the ``Scorer`` protocol.  Expects ``step`` and optionally
    ``context["total_steps"]`` and ``context["snr"]`` to be provided.

    Args:
        schedule: SNR schedule type (``"cosine"`` or ``"linear"``).
        total_steps: Default total denoising steps (overridden by context).
    """

    name = "snr"

    def __init__(
        self,
        schedule: str = "cosine",
        total_steps: int = 50,
    ):
        self._schedule = schedule
        self._total_steps = total_steps

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        ctx = context or {}
        total = ctx.get("total_steps", self._total_steps)
        snr = ctx.get("snr", None)

        if snr is not None:
            # SNR provided directly: high SNR = clean = low info
            score_val = 1.0 / (1.0 + snr)
        elif step is not None:
            score_val = self._schedule_score(step, total)
        else:
            score_val = 0.5  # no step info, default mid

        tier = self._score_to_tier(score_val)

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=tier,
                metadata={
                    "step": step,
                    "total_steps": total,
                    "schedule": self._schedule,
                },
            )
        ]

    def _schedule_score(self, step: int, total: int) -> float:
        progress = step / max(total - 1, 1)
        if self._schedule == "cosine":
            # Cosine schedule: maps [0, 1] -> [1, 0] smoothly
            return 0.5 * (1 + math.cos(math.pi * progress))
        # Linear fallback
        return 1.0 - progress

    def _score_to_tier(self, score: float) -> int:
        if score < 0.25:
            return 0
        if score < 0.5:
            return 1
        if score < 0.75:
            return 2
        return 3

    def reset(self) -> None:
        pass  # stateless
