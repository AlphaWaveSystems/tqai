"""Window attention + residual sharing strategy (DiTFastAttn-inspired).

For diffusion transformers, attention patterns often repeat across
adjacent denoising steps.  This strategy caches attention output and
reuses it within a sliding window, only recomputing when the input
has changed significantly.

References:
    - DiTFastAttn: arXiv:2406.08552 (Window Attention with Residual Sharing)
    - TurboQuant: arXiv:2504.19874
"""

from __future__ import annotations

from typing import Any

from tqai.pipeline.base import ScoredEntry


class WindowStrategy:
    """Cache and reuse quantized KV within a sliding window.

    When the input is similar to the cached version, returns the cached
    (dequantized) output directly.  Otherwise, requantizes and updates
    the cache.

    Args:
        window_size: Number of steps to cache before forced refresh.
        similarity_threshold: Cosine similarity threshold for reuse.
    """

    name = "window"

    def __init__(
        self,
        window_size: int = 5,
        similarity_threshold: float = 0.95,
    ):
        self._window_size = window_size
        self._similarity_threshold = similarity_threshold

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

        cached_compressed = state.get("cached_compressed")
        cached_data = state.get("cached_data")
        steps_since_refresh = state.get("steps_since_refresh", 0)
        stats = state.get("window_stats", {"reused": 0, "refreshed": 0})

        reuse = False
        if cached_data is not None and cached_compressed is not None:
            if steps_since_refresh < self._window_size:
                sim = _cosine_similarity(data, cached_data)
                if sim >= self._similarity_threshold:
                    reuse = True

        if reuse:
            stats["reused"] += 1
            state["steps_since_refresh"] = steps_since_refresh + 1
            state["window_stats"] = stats
            return ("window", True, cached_compressed), state

        # Fresh compression
        compressed = quantizer.quantize(data)
        stats["refreshed"] += 1
        state["cached_compressed"] = compressed
        state["cached_data"] = _detach(data)
        state["steps_since_refresh"] = 0
        state["window_stats"] = stats
        return ("window", False, compressed), state

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any:
        _, _, inner = compressed
        indices, norms = inner[0], inner[1]
        qjl = inner[2] if len(inner) > 2 else None
        return quantizer.dequantize(indices, norms, qjl)


def _cosine_similarity(a, b) -> float:
    if hasattr(a, "flatten"):
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        dot = (a_flat * b_flat).sum()
        return float(dot / (a_flat.norm() * b_flat.norm() + 1e-10))
    import numpy as np
    a_np = np.asarray(a).ravel()
    b_np = np.asarray(b).ravel()
    dot = np.dot(a_np, b_np)
    return float(dot / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10))


def _detach(x):
    return x.detach() if hasattr(x, "detach") else x
