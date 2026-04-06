"""Finite-time Lyapunov exponent monitor.

Estimates the finite-time Lyapunov exponent (FTLE) from sequential
attention states to detect chaotic/unstable dynamics in the model.
When FTLE > 0 (positive Lyapunov exponent), the system is diverging
and compression should be more conservative.

References:
    - Finite-time Lyapunov exponents in dynamical systems
    - TurboQuant: arXiv:2504.19874
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


class LyapunovMonitor:
    """Monitor finite-time Lyapunov exponent from attention states.

    Implements the ``Monitor`` protocol.

    Estimates local divergence rate between consecutive states.  When the
    exponent is positive (diverging), suggests more conservative compression.

    Args:
        window: Number of recent FTLE estimates to average.
        positive_threshold: FTLE above which adjustments are returned.
    """

    name = "lyapunov"

    def __init__(
        self,
        window: int = 20,
        positive_threshold: float = 0.1,
    ):
        self._window = window
        self._positive_threshold = positive_threshold
        self._prev_state: np.ndarray | None = None
        self._ftle_history: deque[float] = deque(maxlen=window)

    def observe(
        self,
        layer_idx: int,
        step: int,
        attention_state: dict,
    ) -> dict | None:
        # Accept either a raw tensor/array or a key in attention_state
        state_data = attention_state.get("hidden_state")
        if state_data is None:
            state_data = attention_state.get("attention_weights")
        if state_data is None:
            return None

        current = _to_numpy(state_data).ravel().astype(np.float64)

        if self._prev_state is not None and current.shape == self._prev_state.shape:
            diff = current - self._prev_state
            diff_norm = np.linalg.norm(diff)
            prev_norm = np.linalg.norm(self._prev_state) + 1e-10

            if diff_norm > 0 and prev_norm > 0:
                # FTLE ≈ log(||δx(t)||/||δx(0)||) / Δt
                ftle = math.log(diff_norm / prev_norm + 1e-10)
                self._ftle_history.append(ftle)

        self._prev_state = current.copy()

        if len(self._ftle_history) < 2:
            return None

        mean_ftle = sum(self._ftle_history) / len(self._ftle_history)

        if mean_ftle > self._positive_threshold:
            # Diverging: be more conservative with compression
            return {"_high_threshold": 0.6}
        elif mean_ftle < -self._positive_threshold:
            # Converging: can compress more aggressively
            return {"_high_threshold": 0.3}

        return None

    @property
    def stats(self) -> dict:
        ftle_list = list(self._ftle_history)
        return {
            "observations": len(ftle_list),
            "mean_ftle": sum(ftle_list) / len(ftle_list) if ftle_list else 0.0,
            "max_ftle": max(ftle_list) if ftle_list else 0.0,
            "min_ftle": min(ftle_list) if ftle_list else 0.0,
        }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy()
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)
