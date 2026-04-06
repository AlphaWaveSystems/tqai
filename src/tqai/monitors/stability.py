"""Attention stability monitor.

Tracks attention entropy over time to detect instability.  When entropy
drops (attention concentrating) or spikes (attention dispersing), the
monitor can adjust scorer thresholds to compensate.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class StabilityMonitor:
    """Track attention stability and suggest parameter adjustments.

    Implements the ``Monitor`` protocol.

    Observes attention entropy and score distributions over time.
    Returns adjustments when significant shifts are detected.

    Args:
        window: Number of recent observations to keep.
        entropy_threshold: Relative entropy change that triggers adjustment.
    """

    name = "stability"

    def __init__(
        self,
        window: int = 50,
        entropy_threshold: float = 0.3,
    ):
        self._window = window
        self._entropy_threshold = entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=window)
        self._score_history: deque[float] = deque(maxlen=window)

    def observe(
        self,
        layer_idx: int,
        step: int,
        attention_state: dict,
    ) -> dict | None:
        entropy = attention_state.get("entropy")
        score = attention_state.get("score")

        if entropy is not None:
            self._entropy_history.append(entropy)
        if score is not None:
            self._score_history.append(score)

        if len(self._entropy_history) < 2:
            return None

        # Check for entropy shift
        recent_mean = _mean(list(self._entropy_history)[-10:])
        overall_mean = _mean(list(self._entropy_history))

        if overall_mean > 0:
            relative_shift = abs(recent_mean - overall_mean) / overall_mean
        else:
            relative_shift = 0.0

        if relative_shift > self._entropy_threshold:
            # Attention is shifting — adjust scoring sensitivity
            if recent_mean < overall_mean:
                # Attention concentrating → increase compression on low-info tokens
                return {"_high_threshold": 0.3}
            else:
                # Attention dispersing → more conservative compression
                return {"_high_threshold": 0.6}

        return None

    @property
    def stats(self) -> dict:
        return {
            "observations": len(self._entropy_history),
            "mean_entropy": _mean(list(self._entropy_history)) if self._entropy_history else 0.0,
            "mean_score": _mean(list(self._score_history)) if self._score_history else 0.0,
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
