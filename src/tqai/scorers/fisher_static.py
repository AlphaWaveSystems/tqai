"""Static Fisher scorer that uses an offline calibration file.

The companion to ``tqai.optimization.fisher_calibration.calibrate_fisher``:
loads a precomputed per-layer Fisher Information table from JSON and
serves it as a constant-time lookup at scoring time. No proxy, no
gradients during inference, no per-call overhead beyond a dictionary
read.

This is the **correct** way to use Fisher Information for KV cache
compression on Apple Silicon. The runtime ``FisherScorer`` uses
``mean(x^2)`` as a proxy because real gradients per-attention-call are
too expensive — but the proxy over-estimates importance and routes
everything to the high-bit tier, defeating the purpose. The offline
calibration computes real gradients once on a small representative
dataset, freezes the per-layer values, and the static scorer reads
them at runtime.

Usage::

    from tqai.scorers.fisher_static import FisherStaticScorer

    scorer = FisherStaticScorer(calibration_path="qwen-3b-fisher.json")
    # ... pass to tqai.patch via pipeline config

Or via the registry::

    cache = tqai.patch(
        model,
        bits_k=4, bits_v=2,
        pipeline={
            "scorer": "fisher_static",
            "scorer_kwargs": {"calibration_path": "qwen-3b-fisher.json"},
            "strategy": "tiered",
        },
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqai.pipeline.base import ScoredEntry


class FisherStaticScorer:
    """Score entries using a precomputed Fisher Information table.

    Loads per-layer K and V Fisher diagonals from a JSON file produced
    by :func:`tqai.optimization.fisher_calibration.calibrate_fisher`,
    normalizes them to ``[0, 1]``, and assigns a score and tier per
    incoming entry by ``layer_idx``.

    Args:
        calibration_path: Path to the calibration JSON file.
        kv_mode: Which projection to use for the score. ``"k"`` (default)
            uses the K projection Fisher values, ``"v"`` uses the V
            projection, ``"max"`` uses the per-layer maximum, ``"mean"``
            uses the average.

    Raises:
        FileNotFoundError: If ``calibration_path`` does not exist.
        ValueError: If the JSON is malformed or has zero layers.
    """

    name = "fisher_static"

    def __init__(self, calibration_path: str | Path, kv_mode: str = "k"):
        from tqai.optimization.fisher_calibration import FisherCalibration

        self._path = str(calibration_path)
        self._kv_mode = kv_mode
        self._calibration = FisherCalibration.load(calibration_path)

        if self._calibration.num_layers == 0:
            raise ValueError(
                f"Calibration file {calibration_path!r} contains zero layers"
            )

        if kv_mode == "k":
            raw = self._calibration.layer_fisher_k
        elif kv_mode == "v":
            raw = self._calibration.layer_fisher_v
        elif kv_mode == "max":
            raw = [
                max(k, v)
                for k, v in zip(
                    self._calibration.layer_fisher_k,
                    self._calibration.layer_fisher_v,
                )
            ]
        elif kv_mode == "mean":
            raw = [
                (k + v) / 2.0
                for k, v in zip(
                    self._calibration.layer_fisher_k,
                    self._calibration.layer_fisher_v,
                )
            ]
        else:
            raise ValueError(
                f"kv_mode must be one of 'k'|'v'|'max'|'mean', got {kv_mode!r}"
            )

        # Normalize to [0, 1] across layers so that the most important
        # layer maps to 1.0 and the least important to 0.0
        max_val = max(raw) if raw else 0.0
        min_val = min(raw) if raw else 0.0
        span = max_val - min_val
        if span > 0:
            self._scores = [(v - min_val) / span for v in raw]
        else:
            # All layers equal — assign mid-tier
            self._scores = [0.5] * len(raw)

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        # Wrap layer_idx if it exceeds the calibration's layer count
        # (e.g., if the calibration was for a smaller model). This
        # provides graceful degradation rather than an IndexError.
        n = len(self._scores)
        if n == 0:
            score_val = 0.5
        else:
            score_val = self._scores[layer_idx % n]

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=_score_to_tier(score_val),
                metadata={
                    "layer_idx": layer_idx,
                    "kv_mode": self._kv_mode,
                    "calibration_layers": n,
                    "calibration_path": self._path,
                },
            )
        ]

    def reset(self) -> None:
        """Static scorer has no state to reset."""
        pass

    @property
    def num_layers(self) -> int:
        return len(self._scores)


def _score_to_tier(score: float) -> int:
    if score < 0.2:
        return 0
    if score < 0.4:
        return 1
    if score < 0.7:
        return 2
    return 3
