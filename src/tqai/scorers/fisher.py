"""Fisher Information diagonal scorer.

Scores KV entries by the diagonal of the Fisher Information matrix,
which approximates the sensitivity of the loss to each parameter.
Higher Fisher values indicate more important entries that need more bits.

References:
    - Fisher Information for neural network compression: arXiv:1906.08589
    - APTQ attention-aware quantization: arXiv:2402.14866
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tqai.pipeline.base import ScoredEntry


class FisherScorer:
    """Score entries by approximated Fisher Information diagonal.

    Uses the squared gradient (outer product diagonal) as a proxy
    for Fisher Information.  Falls back to variance-based scoring
    when gradient information is unavailable.

    Args:
        ema_decay: EMA decay for running Fisher estimate (default 0.9).
    """

    name = "fisher"

    def __init__(self, ema_decay: float = 0.9):
        self._ema_decay = ema_decay
        self._running_fisher: np.ndarray | None = None
        self._step_count = 0

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        x_np = _to_numpy(x).astype(np.float64)
        self._step_count += 1

        # Approximate Fisher diagonal by squared activations (proxy)
        fisher_diag = np.mean(x_np ** 2, axis=tuple(range(x_np.ndim - 1)))

        if self._running_fisher is None:
            self._running_fisher = fisher_diag
        else:
            alpha = self._ema_decay
            self._running_fisher = alpha * self._running_fisher + (1 - alpha) * fisher_diag

        # Score: mean Fisher value, normalized
        mean_fisher = float(np.mean(self._running_fisher))
        # Use log scale for better tier separation
        score_val = min(float(np.log1p(mean_fisher)), 3.0) / 3.0

        tier = _score_to_tier(score_val)

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=tier,
                metadata={
                    "mean_fisher": mean_fisher,
                    "layer_idx": layer_idx,
                },
            )
        ]

    def reset(self) -> None:
        self._running_fisher = None
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
