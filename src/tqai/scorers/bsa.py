"""BSA (Bidirectional Sparse Attention) scorer.

Scores tokens by their saliency relative to block centers.  Tokens that
are semantically redundant within their spatial block get low scores
(compress aggressively).  Salient tokens that differ from their block
center get high scores (preserve).

References:
    - BSA: Bidirectional Sparse Attention, arXiv:2509.01085
    - Query-side sparsification for video DiT
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tqai.pipeline.base import ScoredEntry


class BSAScorer:
    """Score entries by query-KV saliency relative to block centers.

    Partitions the sequence into blocks, computes the centroid of each
    block, and scores each token by its distance from the centroid.
    Tokens close to their block center are redundant; distant tokens
    carry unique information.

    Args:
        block_size: Number of tokens per spatial block (default 16).
        ema_decay: EMA decay for running saliency estimate.
    """

    name = "bsa"

    def __init__(
        self,
        block_size: int = 16,
        ema_decay: float = 0.9,
    ):
        self._block_size = block_size
        self._ema_decay = ema_decay
        self._running_saliency: float | None = None
        self._step_count = 0

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]:
        self._step_count += 1
        x_np = _to_numpy(x).astype(np.float64)

        saliency = self._compute_saliency(x_np)

        if self._running_saliency is None:
            self._running_saliency = saliency
        else:
            alpha = self._ema_decay
            self._running_saliency = alpha * self._running_saliency + (1 - alpha) * saliency

        # Normalize to [0, 1]: high saliency = high score
        score_val = min(self._running_saliency, 2.0) / 2.0
        score_val = max(0.0, min(1.0, score_val))

        tier = _score_to_tier(score_val)

        return [
            ScoredEntry(
                data=x,
                score=score_val,
                tier=tier,
                metadata={
                    "saliency": saliency,
                    "block_size": self._block_size,
                    "layer_idx": layer_idx,
                },
            )
        ]

    def _compute_saliency(self, x_np: np.ndarray) -> float:
        """Compute average saliency as distance from block centroids."""
        if x_np.ndim < 2:
            return 0.0

        # Flatten to [tokens, dim]
        orig_shape = x_np.shape
        dim = orig_shape[-1]
        x_2d = x_np.reshape(-1, dim)
        n_tokens = x_2d.shape[0]

        if n_tokens < 2:
            return 0.0

        bs = min(self._block_size, n_tokens)
        n_blocks = max(1, n_tokens // bs)

        total_dist = 0.0
        count = 0
        for i in range(n_blocks):
            start = i * bs
            end = min(start + bs, n_tokens)
            block = x_2d[start:end]
            centroid = block.mean(axis=0, keepdims=True)
            dists = np.sqrt(np.sum((block - centroid) ** 2, axis=-1))
            total_dist += np.sum(dists)
            count += len(dists)

        mean_dist = total_dist / max(count, 1)
        # Normalize by average vector norm
        mean_norm = np.mean(np.sqrt(np.sum(x_2d ** 2, axis=-1))) + 1e-10
        return float(mean_dist / mean_norm)

    def reset(self) -> None:
        self._running_saliency = None
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
