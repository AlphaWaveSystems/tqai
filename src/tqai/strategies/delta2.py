"""Second-order delta compression strategy (QuantSparse-inspired).

Compresses the *second-order* residual: Δ² = current - 2*prev + prev_prev.
When consecutive steps are smooth (late denoising), Δ² is much smaller
than first-order Δ, enabling even more aggressive compression.

Falls back to first-order delta when second-order is not yet available
(first two steps), and to full compression when deltas are large.

References:
    - QuantSparse: arXiv:2509.23681 (second-order residual quantization
      for Wan2.1 KV cache; the second-order Δ² formulation here is
      directly inspired by their reparameterization)
    - TurboQuant: arXiv:2504.19874
"""

from __future__ import annotations

from typing import Any

from tqai.pipeline.base import ScoredEntry


class SecondOrderDelta:
    """Compress second-order inter-step deltas.

    Implements the ``CompressionStrategy`` protocol.

    Args:
        threshold: Relative norm threshold for delta2 vs fallback.
        delta1_threshold: Threshold for first-order delta fallback.
    """

    name = "delta2"

    def __init__(
        self,
        threshold: float = 0.05,
        delta1_threshold: float = 0.1,
    ):
        self._threshold = threshold
        self._delta1_threshold = delta1_threshold

    def compress(
        self,
        entry: Any,
        quantizer: Any,
        prev_state: dict | None = None,
    ) -> tuple[Any, dict]:
        state = dict(prev_state) if prev_state else {}

        if isinstance(entry, list) and entry and isinstance(entry[0], ScoredEntry):
            data = entry[0].data
        else:
            data = entry

        prev = state.get("prev_data")
        prev_prev = state.get("prev_prev_data")
        stats = state.get("delta2_stats", {"delta2_used": 0, "delta1_used": 0, "full_used": 0})

        if prev is not None and prev_prev is not None and prev.shape == data.shape:
            # Second-order delta: Δ² = x - 2*prev + prev_prev
            delta2 = data - 2 * prev + prev_prev
            relative = _relative_norm(delta2, data)

            if relative < self._threshold:
                compressed = quantizer.quantize(delta2)
                stats["delta2_used"] += 1
                state.update(prev_prev_data=prev, prev_data=_detach(data), delta2_stats=stats)
                return ("delta2", 2, compressed, prev, prev_prev), state

        if prev is not None and prev.shape == data.shape:
            delta1 = data - prev
            relative = _relative_norm(delta1, data)

            if relative < self._delta1_threshold:
                compressed = quantizer.quantize(delta1)
                stats["delta1_used"] += 1
                state.update(prev_prev_data=prev, prev_data=_detach(data), delta2_stats=stats)
                return ("delta2", 1, compressed, prev), state

        # Full compression
        compressed = quantizer.quantize(data)
        stats["full_used"] += 1
        state.update(prev_prev_data=prev, prev_data=_detach(data), delta2_stats=stats)
        return ("delta2", 0, compressed), state

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any:
        order = compressed[1]

        if order == 2:
            _, _, inner, prev, prev_prev = compressed
            delta2_recon = _dequant(quantizer, inner)
            return 2 * prev - prev_prev + delta2_recon

        if order == 1:
            _, _, inner, prev = compressed
            delta1_recon = _dequant(quantizer, inner)
            return prev + delta1_recon

        # Full
        _, _, inner = compressed
        return _dequant(quantizer, inner)


def _dequant(quantizer, inner):
    indices, norms = inner[0], inner[1]
    qjl = inner[2] if len(inner) > 2 else None
    return quantizer.dequantize(indices, norms, qjl)


def _detach(x):
    return x.detach() if hasattr(x, "detach") else x


def _relative_norm(delta, reference) -> float:
    if hasattr(delta, "norm"):
        return float(delta.norm()) / (float(reference.norm()) + 1e-10)
    import numpy as np
    delta_norm = float(np.linalg.norm(np.asarray(delta)))
    ref_norm = float(np.linalg.norm(np.asarray(reference)))
    return delta_norm / (ref_norm + 1e-10)
