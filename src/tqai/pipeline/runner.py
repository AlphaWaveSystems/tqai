"""Composable compression pipeline.

Wraps ``PolarQuantizer`` with optional scorer, strategy, and monitor.
When no middleware is configured, behaves identically to the v0.3.1 path
(direct ``PolarQuantizer.quantize`` / ``dequantize``).
"""

from __future__ import annotations

from typing import Any

from tqai.quantizer import PolarQuantizer


class CompressionPipeline:
    """Scorer -> Strategy -> Quantizer -> Monitor composition.

    Usage::

        # Minimal (backward-compatible)
        pipe = CompressionPipeline(quantizer=pq)
        indices, norms = pipe.compress(x)

        # With scorer + strategy
        pipe = CompressionPipeline(
            quantizer=pq,
            scorer=PalmScorer(alpha=0.5),
            strategy=TieredStrategy(tiers=[0.1, 0.4, 0.8]),
        )
        compressed = pipe.compress(x, layer_idx=0, step=5)
    """

    def __init__(
        self,
        quantizer: PolarQuantizer,
        scorer: Any | None = None,
        strategy: Any | None = None,
        monitor: Any | None = None,
        quantizer_low: PolarQuantizer | None = None,
    ):
        self._quantizer = quantizer
        self._quantizer_low = quantizer_low
        self._scorer = scorer
        self._strategy = strategy
        self._monitor = monitor
        self._state: dict = {}

    # -- Properties --------------------------------------------------------

    @property
    def quantizer(self) -> PolarQuantizer:
        return self._quantizer

    @property
    def quantizer_low(self) -> PolarQuantizer | None:
        return self._quantizer_low

    @property
    def has_middleware(self) -> bool:
        return self._scorer is not None or self._strategy is not None

    # -- Compress / Decompress ---------------------------------------------

    def compress(
        self,
        x: Any,
        layer_idx: int = 0,
        step: int | None = None,
        context: dict | None = None,
    ) -> Any:
        """Compress input through the pipeline.

        When no middleware is active, returns the same format as
        ``PolarQuantizer.quantize()`` — ``(indices, norms)`` or
        ``(indices, norms, qjl_data)``.
        """
        # No middleware: direct quantization (v0.3.1 behavior)
        if not self.has_middleware:
            return self._quantizer.quantize(x)

        # Check skip_layers: some layers must not be compressed (arXiv:2504.10317)
        skip_layers = self._state.get("_skip_layers")
        if skip_layers and layer_idx in skip_layers:
            return self._quantizer.quantize(x)

        # Score entries
        scored = None
        if self._scorer is not None:
            scored = self._scorer.score(x, layer_idx, step, context)

        # Apply strategy — pass quantizer_low via state so strategies can use it
        if self._strategy is not None:
            strategy_state = dict(self._state)
            if self._quantizer_low is not None:
                strategy_state["_quantizer_low"] = self._quantizer_low
            compressed, new_state = self._strategy.compress(
                scored if scored is not None else x,
                self._quantizer,
                strategy_state,
            )
            new_state.pop("_quantizer_low", None)
            self._state.update(new_state)
            return compressed

        # Scorer but no strategy: standard quantize
        return self._quantizer.quantize(x)

    def decompress(self, compressed: Any, layer_idx: int = 0) -> Any:
        """Decompress through the pipeline."""
        # Check if this layer was skipped (raw quantizer output, not tagged)
        skip_layers = self._state.get("_skip_layers")
        if skip_layers and layer_idx in skip_layers:
            if isinstance(compressed, tuple) and len(compressed) >= 2:
                indices, norms = compressed[0], compressed[1]
                qjl = compressed[2] if len(compressed) > 2 else None
                return self._quantizer.dequantize(indices, norms, qjl)

        if self._strategy is not None:
            decompress_state = dict(self._state)
            if self._quantizer_low is not None:
                decompress_state["_quantizer_low"] = self._quantizer_low
            return self._strategy.decompress(
                compressed, self._quantizer, decompress_state
            )
        # Standard path
        if isinstance(compressed, tuple) and len(compressed) >= 2:
            indices, norms = compressed[0], compressed[1]
            qjl_data = compressed[2] if len(compressed) > 2 else None
            return self._quantizer.dequantize(indices, norms, qjl_data)
        raise ValueError(f"Unknown compressed format: {type(compressed)}")

    # -- Observation -------------------------------------------------------

    def observe(self, layer_idx: int, step: int, attention_state: dict) -> None:
        """Feed runtime observations to the monitor."""
        if self._monitor is None:
            return
        adjustments = self._monitor.observe(layer_idx, step, attention_state)
        if adjustments and self._scorer:
            for key, value in adjustments.items():
                if hasattr(self._scorer, key):
                    setattr(self._scorer, key, value)

    # -- Lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Reset pipeline state between generations."""
        self._state.clear()
        if self._scorer is not None:
            self._scorer.reset()
