"""First-order inter-step delta compression strategy.

Compresses the difference between current and previous step's KV
activations.  When the delta is small (late denoising steps), it can
be stored at fewer bits than the full activation.

References:
    - TurboQuant: arXiv:2504.19874
    - Inter-step KV sharing: DiTFastAttn (arXiv:2406.08552)
"""

from __future__ import annotations

from typing import Any

from tqai.pipeline.base import ScoredEntry


class DeltaStrategy:
    """Compress inter-step deltas when they are small enough.

    Implements the ``CompressionStrategy`` protocol.

    When ``||delta|| / ||current|| < threshold``, stores the delta instead
    of the full tensor.  Otherwise falls back to full compression.

    Args:
        threshold: Relative norm threshold for delta vs full (default 0.1).
    """

    name = "delta"

    def __init__(self, threshold: float = 0.1):
        self._threshold = threshold

    def compress(
        self,
        entry: Any,
        quantizer: Any,
        prev_state: dict | None = None,
    ) -> tuple[Any, dict]:
        state = dict(prev_state) if prev_state else {}

        # Extract data from ScoredEntry or raw tensor
        if isinstance(entry, list) and entry and isinstance(entry[0], ScoredEntry):
            data = entry[0].data
        else:
            data = entry

        prev = state.get("prev_data")
        stats = state.get("delta_stats", {"delta_used": 0, "full_used": 0})

        if prev is not None and prev.shape == data.shape:
            delta = data - prev
            # Compute relative delta norm
            delta_norm = _tensor_norm(delta)
            data_norm = _tensor_norm(data) + 1e-10
            relative = delta_norm / data_norm

            if relative < self._threshold:
                compressed = quantizer.quantize(delta)
                stats["delta_used"] += 1
                state["prev_data"] = data.detach() if hasattr(data, "detach") else data
                state["delta_stats"] = stats
                return ("delta", True, compressed, prev), state

        # Full compression
        compressed = quantizer.quantize(data)
        stats["full_used"] += 1
        state["prev_data"] = data.detach() if hasattr(data, "detach") else data
        state["delta_stats"] = stats
        return ("delta", False, compressed), state

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any:
        is_delta = compressed[1]

        if is_delta:
            _, _, inner, prev = compressed
            indices, norms = inner[0], inner[1]
            qjl = inner[2] if len(inner) > 2 else None
            delta_recon = quantizer.dequantize(indices, norms, qjl)
            return prev + delta_recon

        _, _, inner = compressed
        indices, norms = inner[0], inner[1]
        qjl = inner[2] if len(inner) > 2 else None
        return quantizer.dequantize(indices, norms, qjl)


def _tensor_norm(x) -> float:
    if hasattr(x, "norm"):
        return float(x.norm())
    import numpy as np
    return float(np.linalg.norm(np.asarray(x)))
