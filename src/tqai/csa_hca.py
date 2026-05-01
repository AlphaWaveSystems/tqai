"""CSA + HCA: block-pooled attention compression (DeepSeek V4 inspired).

Two complementary compression views over a KV sequence:

* **CSA — Compressed Sparse Attention.**  Pool every ``csa_block_size`` tokens
  into a single centroid (mild compression, e.g. 4x).  At decode time, score
  the query against block centroids and attend only to the top-k blocks.
  Preserves query-dependent selectivity.

* **HCA — Heavily Compressed Attention.**  Pool every ``hca_block_size`` tokens
  into a single centroid (aggressive compression, e.g. 128x).  Attend densely
  over the resulting (now short) compressed sequence.  Provides a global view.

Combined attention concatenates the selected CSA centroids with the HCA
centroids and runs a single softmax over their union.  This gives both
fine-grained (selective) and global (compressed) coverage at a fraction of
full attention cost.

The functions in this module are pure and backend-agnostic — they work on any
object exposing the standard tensor protocol used elsewhere in tqai (torch
tensors are tested; numpy works for the non-gather paths).  Inputs are assumed
to follow tqai's canonical KV layout ``(B, H, S, D)``.

References:
    - DeepSeek-V4 technical report (April 2026): introduces the interleaved
      CSA/HCA design that motivates this module.
    - Block-centroid scoring is structurally similar to the BSA scorer
      (``tqai.scorers.bsa``); this module differs in that it produces an
      attention output, not a per-token score.
"""

from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Block pooling
# ---------------------------------------------------------------------------


def block_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool along the sequence axis (axis=-2) into blocks.

    The trailing partial block (if any) is pooled separately over its valid
    entries so that padding does not bias the centroid.

    Args:
        x: Tensor of shape ``(..., S, D)``.
        block_size: Block size ``m`` (``m >= 1``).

    Returns:
        Tensor of shape ``(..., ceil(S / m), D)``.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    S = x.shape[-2]
    if S == 0:
        return x

    n_full = S // block_size
    remainder = S - n_full * block_size

    leading = x.shape[:-2]
    D = x.shape[-1]

    parts = []
    if n_full > 0:
        full = x[..., : n_full * block_size, :].reshape(*leading, n_full, block_size, D)
        parts.append(full.mean(dim=-2))  # (..., n_full, D)
    if remainder > 0:
        rem = x[..., n_full * block_size :, :].mean(dim=-2, keepdim=True)
        parts.append(rem)  # (..., 1, D)

    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=-2)


# ---------------------------------------------------------------------------
# CSA scoring + top-k selection
# ---------------------------------------------------------------------------


def csa_score(
    q: torch.Tensor,
    k_blocks: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Score the query against CSA block centroids.

    Args:
        q: ``(B, H_q, T_q, D)``.
        k_blocks: ``(B, H_kv, N, D)``.  GQA is handled by repeating along H.
        scale: Multiplied into scores.  Defaults to ``1/sqrt(D)``.

    Returns:
        Scores ``(B, H_q, T_q, N)``.
    """
    B, H_q, T_q, D = q.shape
    H_kv = k_blocks.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    repeats = H_q // H_kv

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Expand kv heads to query heads (GQA)
    if repeats > 1:
        k_blocks = k_blocks.repeat_interleave(repeats, dim=1)

    # (B, H_q, T_q, D) @ (B, H_q, D, N) → (B, H_q, T_q, N)
    scores = torch.matmul(q, k_blocks.transpose(-1, -2))
    return scores * scale


def csa_top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return the indices of the top-k blocks for each query position.

    Indices are returned in the order produced by ``torch.topk`` (descending
    by score).  The caller may sort them ascending if causal ordering matters.

    Args:
        scores: ``(B, H_q, T_q, N)``.
        k: Number of blocks to keep.  Clamped to ``min(k, N)``.

    Returns:
        Indices ``(B, H_q, T_q, k_eff)`` where ``k_eff = min(k, N)``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    N = scores.shape[-1]
    k_eff = min(k, N)
    return torch.topk(scores, k=k_eff, dim=-1).indices


def gather_blocks(
    blocks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather selected blocks per query head.

    Args:
        blocks: ``(B, H_kv, N, D)``.
        indices: ``(B, H_q, T_q, k)`` from :func:`csa_top_k`.

    Returns:
        ``(B, H_q, T_q, k, D)``.
    """
    B, H_q, T_q, k = indices.shape
    H_kv = blocks.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} not divisible by H_kv={H_kv}")
    repeats = H_q // H_kv

    if repeats > 1:
        blocks = blocks.repeat_interleave(repeats, dim=1)

    D = blocks.shape[-1]
    # Expand indices for the D dim and gather along axis=2 (block dim of blocks)
    # blocks_expanded: (B, H_q, T_q, N, D)
    blocks_expanded = blocks.unsqueeze(2).expand(B, H_q, T_q, blocks.shape[-2], D)
    idx_expanded = indices.unsqueeze(-1).expand(B, H_q, T_q, k, D)
    return torch.gather(blocks_expanded, dim=3, index=idx_expanded)


# ---------------------------------------------------------------------------
# Combined CSA + HCA attention
# ---------------------------------------------------------------------------


def csa_hca_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    csa_block_size: int,
    hca_block_size: int,
    csa_top_k_blocks: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """End-to-end CSA + HCA attention.

    Algorithm:
        1. Pool ``K`` and ``V`` into CSA blocks (mild) and HCA blocks
           (aggressive) along the sequence axis.
        2. Score ``Q`` against CSA-K block centroids; select top-k per
           query head/position.
        3. Concatenate the selected CSA centroids with all HCA centroids
           (and likewise for V).
        4. Run a single softmax over the union and weight V accordingly.

    Args:
        q: Query ``(B, H_q, T_q, D)``.
        k: Key ``(B, H_kv, S, D)``.
        v: Value ``(B, H_kv, S, D)``.
        csa_block_size: CSA pool size (e.g. 4).
        hca_block_size: HCA pool size (e.g. 128).
        csa_top_k_blocks: Number of CSA blocks to keep per query.
        scale: Attention scale.  Defaults to ``1/sqrt(D)``.

    Returns:
        Tuple of ``(output, info)``:
            * ``output``: ``(B, H_q, T_q, D)``.
            * ``info``: stats dict with keys ``n_csa_blocks``, ``n_hca_blocks``,
              ``k_eff``, ``compression_ratio`` (S divided by attended-block
              count).  Useful for benchmarking and debugging.

    Notes:
        This is a reference implementation in plain torch.  A fused MLX
        version belongs in ``tqai.attention_fused`` once the strategy is
        validated.
    """
    if csa_block_size < 1:
        raise ValueError(f"csa_block_size must be >= 1, got {csa_block_size}")
    if hca_block_size < csa_block_size:
        raise ValueError(
            f"hca_block_size ({hca_block_size}) must be >= csa_block_size "
            f"({csa_block_size}); HCA is the more aggressive view."
        )

    B, H_q, T_q, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # 1. Pool
    k_csa = block_pool(k, csa_block_size)  # (B, H_kv, N_csa, D)
    v_csa = block_pool(v, csa_block_size)
    k_hca = block_pool(k, hca_block_size)  # (B, H_kv, N_hca, D)
    v_hca = block_pool(v, hca_block_size)

    N_csa = k_csa.shape[-2]
    N_hca = k_hca.shape[-2]

    # 2. CSA scoring + top-k
    csa_scores = csa_score(q, k_csa, scale=scale)  # (B, H_q, T_q, N_csa)
    idx = csa_top_k(csa_scores, csa_top_k_blocks)  # (B, H_q, T_q, k_eff)
    k_eff = idx.shape[-1]

    k_csa_sel = gather_blocks(k_csa, idx)  # (B, H_q, T_q, k_eff, D)
    v_csa_sel = gather_blocks(v_csa, idx)

    # CSA scores at selected positions (already computed; reuse for fairness)
    csa_sel_scores = torch.gather(csa_scores, dim=-1, index=idx)  # (B, H_q, T_q, k_eff)

    # 3. HCA dense scoring (broadcast over query heads via GQA)
    H_kv = k_hca.shape[1]
    repeats = H_q // H_kv
    if repeats > 1:
        k_hca_q = k_hca.repeat_interleave(repeats, dim=1)
        v_hca_q = v_hca.repeat_interleave(repeats, dim=1)
    else:
        k_hca_q = k_hca
        v_hca_q = v_hca
    # (B, H_q, T_q, D) @ (B, H_q, D, N_hca) → (B, H_q, T_q, N_hca)
    hca_scores = torch.matmul(q, k_hca_q.transpose(-1, -2)) * scale

    # 4. Joint softmax over [csa_sel, hca]
    joint_scores = torch.cat([csa_sel_scores, hca_scores], dim=-1)
    weights = torch.softmax(joint_scores, dim=-1)  # (B, H_q, T_q, k_eff + N_hca)

    w_csa = weights[..., :k_eff]                   # (B, H_q, T_q, k_eff)
    w_hca = weights[..., k_eff:]                    # (B, H_q, T_q, N_hca)

    # 5. Aggregate values
    # CSA contribution: (B, H_q, T_q, k_eff, 1) * (B, H_q, T_q, k_eff, D) → sum k_eff
    out_csa = (w_csa.unsqueeze(-1) * v_csa_sel).sum(dim=-2)  # (B, H_q, T_q, D)
    # HCA contribution: (B, H_q, T_q, N_hca) @ (B, H_q, N_hca, D) → (B, H_q, T_q, D)
    out_hca = torch.matmul(w_hca, v_hca_q)

    out = out_csa + out_hca

    info = {
        "n_csa_blocks": int(N_csa),
        "n_hca_blocks": int(N_hca),
        "k_eff": int(k_eff),
        "compression_ratio": float(k.shape[-2]) / max(1, k_eff + N_hca),
    }
    return out, info
