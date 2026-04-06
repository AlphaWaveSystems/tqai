"""Tiered bit allocation strategy.

Uses scorer output to route entries to different quantization bit widths.
High-score (novel/critical) entries get more bits; low-score (redundant)
entries get fewer bits.

When a low-bit quantizer is available (``quantizer_low`` passed via
pipeline state), entries below the ``high_tier_threshold`` are compressed
with it.  Otherwise, all entries use the primary quantizer.
"""

from __future__ import annotations

from typing import Any

from tqai.pipeline.base import ScoredEntry


class TieredStrategy:
    """Route entries to high or low bit quantizer based on score tier.

    Implements the ``CompressionStrategy`` protocol.

    Args:
        tiers: Score thresholds separating tiers (ascending).
        high_tier_threshold: Score above which the primary (high-bit)
            quantizer is used (default 0.4).
    """

    name = "tiered"

    def __init__(
        self,
        tiers: list[float] | None = None,
        high_tier_threshold: float = 0.4,
    ):
        self._tiers = tiers or [0.1, 0.4, 0.8]
        self._high_threshold = high_tier_threshold

    def compress(
        self,
        entry: Any,
        quantizer: Any,
        prev_state: dict | None = None,
    ) -> tuple[Any, dict]:
        state = dict(prev_state) if prev_state else {}
        quantizer_low = state.pop("_quantizer_low", None)

        # Extract data and score from ScoredEntry list or raw tensor
        if isinstance(entry, list) and entry and isinstance(entry[0], ScoredEntry):
            data = entry[0].data
            score = entry[0].score
        else:
            data = entry
            score = 1.0  # default to high-quality

        # Track tier distribution
        tier_counts = state.get("tier_counts", [0, 0])
        use_high = score >= self._high_threshold

        if use_high:
            compressed = quantizer.quantize(data)
            tier_counts[1] += 1
        else:
            # Use quantizer_low if available, else fall back to primary
            q = quantizer_low if quantizer_low is not None else quantizer
            compressed = q.quantize(data)
            tier_counts[0] += 1

        state["tier_counts"] = tier_counts
        state["last_score"] = score
        state["last_use_high"] = use_high

        return ("tiered", use_high, compressed), state

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any:
        _tag, use_high, inner = compressed
        # Use the same quantizer that was used for compression
        # (quantizer_low is passed via state during decompress too)
        q = quantizer
        if not use_high and state:
            q = state.get("_quantizer_low", quantizer)

        if isinstance(inner, tuple) and len(inner) >= 2:
            indices, norms = inner[0], inner[1]
            qjl = inner[2] if len(inner) > 2 else None
            return q.dequantize(indices, norms, qjl)

        raise ValueError(f"Unknown inner compressed format: {type(inner)}")
