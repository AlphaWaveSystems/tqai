"""Fused dequant-attention for compressed KV caches.

Implements attention over PolarQuantizer- and RotorQuantizer-compressed
KV caches without materializing float32 KV buffers.  DRAM reads drop
from ``T_kv × D × 4 bytes`` (float32) to ``T_kv × D × 0.5 bytes`` (4-bit)
plus ``T_kv × 2 bytes`` (float16 norms) — roughly a 4× bandwidth reduction.

Algorithm (single-head, single-query decode step):

  Scoring  (K side):
    1. q_rotated = R @ q                     (once per step, O(D²) or O(D))
    2. scores[k] = norm_k * dot(q_rotated, centroids[k_indices[k]])  (Metal)
    3. weights   = softmax(scores * scale)

  Aggregation  (V side):
    4. out_rotated[d] = sum_k weights[k] * norm_k * centroids[v_indices[k,d]]  (Metal)
    5. output = R.T @ out_rotated            (once per step, O(D²) or O(D))

Step 2 and 4 are the fused Metal kernels in ``tqai.kernels``.
Steps 1 and 5 reuse MLX matmul or RotorQuantizer's block-diagonal rotation.

Public API:
    ``fused_polar_decode_step`` — single-head, single-query PolarQuant decode.
    ``fused_rotor_decode_step`` — single-head, single-query RotorQuant decode.
    ``batched_fused_polar_decode`` — multi-head, single-query PolarQuant decode.
"""

from __future__ import annotations

import mlx.core as mx


# ---------------------------------------------------------------------------
# Single-head helpers (PolarQuant)
# ---------------------------------------------------------------------------


def fused_polar_decode_step(
    q: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    rotation: mx.array,
    centroids: mx.array,
    scale: float,
) -> mx.array:
    """Single-head attention decode step over PolarQuant-compressed KV cache.

    Args:
        q: Query vector, shape ``(D,)``, any float dtype.
        k_indices: uint8 key cache, shape ``(T_kv, D)``.
        k_norms: float16 key norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        v_indices: uint8 value cache, shape ``(T_kv, D)``.
        v_norms: float16 value norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        rotation: Orthogonal matrix ``(D, D)``, float32.
        centroids: Lloyd-Max codebook ``(n_levels,)``, float32.
        scale: Attention scale factor (typically ``1/sqrt(D)``).

    Returns:
        Attention output, shape ``(D,)``, float32.
    """
    from tqai.kernels import metal_aggregate_values, metal_score_keys

    # 1. Rotate query once (O(D²) matmul)
    q_rotated = rotation @ q.astype(mx.float32)  # (D,)

    # 2. Score all keys via fused Metal kernel
    k_norms_flat = mx.reshape(k_norms, (-1,))
    scores = metal_score_keys(q_rotated, k_indices, k_norms_flat, centroids)
    scores = scores * scale  # (T_kv,)

    # 3. Softmax
    weights = mx.softmax(scores, axis=-1)  # (T_kv,)

    # 4. Aggregate values in rotated space via fused Metal kernel
    v_norms_flat = mx.reshape(v_norms, (-1,))
    out_rotated = metal_aggregate_values(weights, v_indices, v_norms_flat, centroids)

    # 5. Inverse rotation (O(D²) matmul)
    return rotation.T @ out_rotated  # (D,)


# ---------------------------------------------------------------------------
# Single-head helpers (RotorQuant)
# ---------------------------------------------------------------------------


def fused_rotor_decode_step(
    q: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    block_mats: mx.array,
    centroids: mx.array,
    n_full: int,
    scale: float,
) -> mx.array:
    """Single-head attention decode step over RotorQuant-compressed KV cache.

    The rotation cost is O(D) instead of O(D²): each group of 3 dimensions
    is rotated independently by its 3×3 matrix.

    Args:
        q: Query vector, shape ``(D,)``, any float dtype.
        k_indices: uint8 key cache, shape ``(T_kv, D)``.
        k_norms: float16 key norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        v_indices: uint8 value cache, shape ``(T_kv, D)``.
        v_norms: float16 value norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        block_mats: Flat float32 ``(n_full * 9,)`` rotation matrices.
        centroids: Lloyd-Max codebook ``(n_levels,)``, float32.
        n_full: Number of complete 3-dim groups (``head_dim // 3``).
        scale: Attention scale factor.

    Returns:
        Attention output, shape ``(D,)``, float32.
    """
    from tqai.kernels import metal_aggregate_values, metal_score_keys

    D = q.shape[0]
    q_f32 = q.astype(mx.float32)  # (D,)

    # 1. Rotate query with block-diagonal rotors (O(D))
    q_rotated = _rotor_rotate_vec(q_f32, block_mats, n_full, D)  # (D,)

    # 2. Score all keys via fused Metal kernel
    k_norms_flat = mx.reshape(k_norms, (-1,))
    scores = metal_score_keys(q_rotated, k_indices, k_norms_flat, centroids)
    scores = scores * scale

    # 3. Softmax
    weights = mx.softmax(scores, axis=-1)

    # 4. Aggregate values in rotated space via fused Metal kernel
    v_norms_flat = mx.reshape(v_norms, (-1,))
    out_rotated = metal_aggregate_values(weights, v_indices, v_norms_flat, centroids)

    # 5. Inverse block-diagonal rotation (O(D))
    return _rotor_inverse_rotate_vec(out_rotated, block_mats, n_full, D)  # (D,)


def _rotor_rotate_vec(
    v: mx.array,
    block_mats: mx.array,
    n_full: int,
    D: int,
) -> mx.array:
    """Apply block-diagonal forward rotation to a single vector (MLX path).

    Each group of 3 dims is rotated by its 3×3 matrix stored in block_mats.
    Remainder dims (d >= n_full*3) pass through unchanged.
    """
    mats = mx.reshape(block_mats, (n_full, 3, 3))  # (n_full, 3, 3)
    rotated_part = v[: n_full * 3].reshape(n_full, 3)  # (n_full, 3)
    # Forward: y_g = R_g @ x_g  →  einsum "gi,gji->gj"  (matmul per group)
    rotated_part = mx.sum(
        mats * rotated_part[:, None, :], axis=-1
    )  # (n_full, 3)
    rotated_part = rotated_part.reshape(-1)  # (n_full*3,)

    if D > n_full * 3:
        return mx.concatenate([rotated_part, v[n_full * 3 :]])
    return rotated_part


def _rotor_inverse_rotate_vec(
    v: mx.array,
    block_mats: mx.array,
    n_full: int,
    D: int,
) -> mx.array:
    """Apply block-diagonal inverse rotation to a single vector (MLX path).

    Inverse of ``_rotor_rotate_vec``: applies R.T per group.
    """
    mats = mx.reshape(block_mats, (n_full, 3, 3))  # (n_full, 3, 3)
    rotated_part = v[: n_full * 3].reshape(n_full, 3)  # (n_full, 3)
    # Inverse: x_g = R_g.T @ y_g  →  transpose last two axes of mats, then matmul
    mats_T = mx.swapaxes(mats, -2, -1)  # (n_full, 3, 3) transposed per group
    inv_part = mx.sum(
        mats_T * rotated_part[:, None, :], axis=-1
    )  # (n_full, 3)
    inv_part = inv_part.reshape(-1)

    if D > n_full * 3:
        return mx.concatenate([inv_part, v[n_full * 3 :]])
    return inv_part


# ---------------------------------------------------------------------------
# Multi-head batched decode (PolarQuant)
# ---------------------------------------------------------------------------


def batched_fused_polar_decode(
    queries: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    rotation: mx.array,
    centroids: mx.array,
    scale: float,
    causal: bool = True,
) -> mx.array:
    """Multi-head single-query attention over PolarQuant-compressed KV cache.

    Processes each head independently via :func:`fused_polar_decode_step`.
    Intended for autoregressive decode (T_q=1) — prefill uses the standard
    MLX SDPA after full dequantization.

    Args:
        queries: Shape ``(B, n_q_heads, T_q, D)``.
        k_indices: uint8, shape ``(B, n_kv_heads, T_kv, D)``.
        k_norms: float16, shape ``(B, n_kv_heads, T_kv, 1)``.
        v_indices: uint8, shape ``(B, n_kv_heads, T_kv, D)``.
        v_norms: float16, shape ``(B, n_kv_heads, T_kv, 1)``.
        rotation: ``(D, D)`` float32 rotation matrix (shared across heads).
        centroids: ``(n_levels,)`` float32 codebook.
        scale: Attention scale.
        causal: If True, a causal mask is applied (for T_q > 1).

    Returns:
        Attention output, shape ``(B, n_q_heads, T_q, D)``.

    Note:
        For large T_q (prefill), dequantize K/V fully and use
        ``mx.fast.scaled_dot_product_attention`` instead — it is faster
        at high sequence concurrency than sequential head-by-head calls.
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = k_indices.shape[1]
    T_kv = k_indices.shape[2]

    # GQA: map query heads → kv heads
    repeats = n_q_heads // n_kv_heads

    outputs = []
    for b in range(B):
        for h_q in range(n_q_heads):
            h_kv = h_q // repeats
            for t in range(T_q):
                q_vec = queries[b, h_q, t, :]  # (D,)
                ki = k_indices[b, h_kv, :, :]   # (T_kv, D)
                kn = k_norms[b, h_kv, :, :]     # (T_kv, 1)
                vi = v_indices[b, h_kv, :, :]
                vn = v_norms[b, h_kv, :, :]

                if causal and T_q > 1:
                    # Causal masking: query at position t can only attend to
                    # KV positions <= (T_kv - T_q + t)
                    kv_end = T_kv - T_q + t + 1
                    ki = ki[:kv_end]
                    kn = kn[:kv_end]
                    vi = vi[:kv_end]
                    vn = vn[:kv_end]

                out = fused_polar_decode_step(
                    q_vec, ki, kn, vi, vn, rotation, centroids, scale
                )
                outputs.append(out)

    # Reassemble: (B * n_q_heads * T_q, D) → (B, n_q_heads, T_q, D)
    stacked = mx.stack(outputs, axis=0)
    return stacked.reshape(B, n_q_heads, T_q, D)


# ---------------------------------------------------------------------------
# Batched multi-head decode (v0.6) — 2 Metal dispatches total
# ---------------------------------------------------------------------------


def batched_fused_polar_decode_v2(
    queries: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    rotation_k: mx.array,
    rotation_v: mx.array,
    centroids_k: mx.array,
    centroids_v: mx.array,
    scale: float,
    sink_keys: mx.array | None = None,
    sink_values: mx.array | None = None,
) -> mx.array:
    """Batched multi-head decode over PolarQuant-compressed KV cache.

    Replaces the per-head Python loop with 2 total Metal dispatches
    (1 score + 1 aggregate), regardless of head count.

    Only supports B=1, T_q=1 (autoregressive decode hot path).

    Args:
        queries: ``(B, n_q_heads, 1, D)``.
        k_indices: uint8, ``(B, n_kv_heads, T_c, D)``.
        k_norms: float16, ``(B, n_kv_heads, T_c, 1)``.
        v_indices: uint8, ``(B, n_kv_heads, T_c, D)``.
        v_norms: float16, ``(B, n_kv_heads, T_c, 1)``.
        rotation_k: ``(D, D)`` float32 key rotation (shared across heads).
        rotation_v: ``(D, D)`` float32 value rotation (shared across heads).
        centroids_k: ``(n_levels,)`` float32 key codebook.
        centroids_v: ``(n_levels,)`` float32 value codebook.
        scale: Attention scale factor.
        sink_keys: ``(B, n_kv_heads, T_s, D)`` float32 or None.
        sink_values: ``(B, n_kv_heads, T_s, D)`` float32 or None.

    Returns:
        Attention output ``(B, n_q_heads, 1, D)`` float32.
    """
    from tqai.kernels import (
        metal_batched_aggregate_values,
        metal_batched_score_keys,
    )

    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = k_indices.shape[1]
    repeats = n_q_heads // n_kv_heads

    # Extract (n_q_heads, D) queries for batch=0, t=0
    q_all = queries[0, :, 0, :].astype(mx.float32)  # (n_q, D)

    # 1. Batch rotate all queries: q_rot = q @ R.T  (same as R @ q per-row)
    q_rotated = q_all @ rotation_k.T  # (n_q, D)

    has_sinks = sink_keys is not None and sink_keys.shape[2] > 0
    has_compressed = k_indices is not None and k_indices.shape[2] > 0

    score_parts = []

    # 2. Sink scoring — batched matmul (no rotation needed)
    if has_sinks:
        T_s = sink_keys.shape[2]
        # Expand sink keys for GQA: (n_kv, T_s, D) → (n_q, T_s, D)
        sk = sink_keys[0]  # (n_kv, T_s, D)
        if repeats > 1:
            sk = mx.repeat(sk, repeats, axis=0)  # (n_q, T_s, D)
        # (n_q, D) @ (n_q, D, T_s) → (n_q, T_s)
        sink_scores = mx.sum(
            q_all[:, None, :] * sk, axis=-1
        ) * scale  # (n_q, T_s)
        score_parts.append(sink_scores)

    # 3. Compressed scoring — 1 Metal dispatch
    if has_compressed:
        comp_scores = metal_batched_score_keys(
            q_rotated,
            k_indices[0],   # (n_kv, T_c, D)
            k_norms[0],     # (n_kv, T_c, 1)
            centroids_k,
            repeats,
        ) * scale  # (n_q, T_c)
        score_parts.append(comp_scores)

    if not score_parts:
        return mx.zeros_like(queries, dtype=mx.float32)

    # 4. Combined softmax
    all_scores = mx.concatenate(score_parts, axis=-1) if len(score_parts) > 1 else score_parts[0]
    weights = mx.softmax(all_scores, axis=-1)  # (n_q, T_total)

    output = mx.zeros((n_q_heads, D), dtype=mx.float32)
    w_idx = 0

    # 5. Sink value aggregation — batched matmul
    if has_sinks:
        T_s = sink_keys.shape[2]
        w_sink = weights[:, w_idx: w_idx + T_s]  # (n_q, T_s)
        sv = sink_values[0]  # (n_kv, T_s, D)
        if repeats > 1:
            sv = mx.repeat(sv, repeats, axis=0)  # (n_q, T_s, D)
        # (n_q, T_s, 1) * (n_q, T_s, D) → sum → (n_q, D)
        output = output + mx.sum(w_sink[:, :, None] * sv.astype(mx.float32), axis=1)
        w_idx += T_s

    # 6. Compressed value aggregation — 1 Metal dispatch
    if has_compressed:
        T_c = k_indices.shape[2]
        w_comp = weights[:, w_idx: w_idx + T_c]  # (n_q, T_c)
        out_rotated = metal_batched_aggregate_values(
            w_comp,
            v_indices[0],   # (n_kv, T_c, D)
            v_norms[0],     # (n_kv, T_c, 1)
            centroids_v,
            repeats,
        )  # (n_q, D) in rotated space

        # 7. Batch inverse rotation: R_v.T @ out_rotated (per-row)
        output = output + out_rotated @ rotation_v  # (n_q, D) @ (D, D)

    return output.reshape(1, n_q_heads, 1, D)


def batched_fused_rotor_decode_v2(
    queries: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    block_mats_k: mx.array,
    block_mats_v: mx.array,
    centroids_k: mx.array,
    centroids_v: mx.array,
    n_full: int,
    scale: float,
    sink_keys: mx.array | None = None,
    sink_values: mx.array | None = None,
) -> mx.array:
    """Batched multi-head decode over RotorQuant-compressed KV cache.

    Same as ``batched_fused_polar_decode_v2`` but uses O(D) block-diagonal
    rotation instead of O(D^2) full rotation.

    Only supports B=1, T_q=1.

    Args:
        queries: ``(B, n_q_heads, 1, D)``.
        k_indices, k_norms, v_indices, v_norms: Compressed cache buffers.
        block_mats_k: Flat ``(n_full * 9,)`` key rotation matrices.
        block_mats_v: Flat ``(n_full * 9,)`` value rotation matrices.
        centroids_k, centroids_v: Codebooks.
        n_full: Number of complete 3-dim groups (``D // 3``).
        scale: Attention scale factor.
        sink_keys, sink_values: Full-precision sink buffers or None.

    Returns:
        Attention output ``(B, n_q_heads, 1, D)`` float32.
    """
    from tqai.kernels import (
        metal_batched_aggregate_values,
        metal_batched_score_keys,
    )

    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = k_indices.shape[1]
    repeats = n_q_heads // n_kv_heads

    q_all = queries[0, :, 0, :].astype(mx.float32)  # (n_q, D)

    # 1. Batch rotate queries with block-diagonal rotors
    q_rotated = _batched_rotor_rotate(q_all, block_mats_k, n_full, D)

    has_sinks = sink_keys is not None and sink_keys.shape[2] > 0
    has_compressed = k_indices is not None and k_indices.shape[2] > 0

    score_parts = []

    # 2. Sink scoring
    if has_sinks:
        T_s = sink_keys.shape[2]
        sk = sink_keys[0]
        if repeats > 1:
            sk = mx.repeat(sk, repeats, axis=0)
        sink_scores = mx.sum(q_all[:, None, :] * sk, axis=-1) * scale
        score_parts.append(sink_scores)

    # 3. Compressed scoring — 1 Metal dispatch
    if has_compressed:
        comp_scores = metal_batched_score_keys(
            q_rotated, k_indices[0], k_norms[0], centroids_k, repeats,
        ) * scale
        score_parts.append(comp_scores)

    if not score_parts:
        return mx.zeros_like(queries, dtype=mx.float32)

    # 4. Combined softmax
    all_scores = mx.concatenate(score_parts, axis=-1) if len(score_parts) > 1 else score_parts[0]
    weights = mx.softmax(all_scores, axis=-1)

    output = mx.zeros((n_q_heads, D), dtype=mx.float32)
    w_idx = 0

    # 5. Sink value aggregation
    if has_sinks:
        T_s = sink_keys.shape[2]
        w_sink = weights[:, w_idx: w_idx + T_s]
        sv = sink_values[0]
        if repeats > 1:
            sv = mx.repeat(sv, repeats, axis=0)
        output = output + mx.sum(w_sink[:, :, None] * sv.astype(mx.float32), axis=1)
        w_idx += T_s

    # 6. Compressed value aggregation — 1 Metal dispatch
    if has_compressed:
        T_c = k_indices.shape[2]
        w_comp = weights[:, w_idx: w_idx + T_c]
        out_rotated = metal_batched_aggregate_values(
            w_comp, v_indices[0], v_norms[0], centroids_v, repeats,
        )
        # 7. Batch inverse rotation
        output = output + _batched_rotor_inverse_rotate(out_rotated, block_mats_v, n_full, D)

    return output.reshape(1, n_q_heads, 1, D)


# ---------------------------------------------------------------------------
# Batched rotor rotation helpers
# ---------------------------------------------------------------------------


def _batched_rotor_rotate(
    vecs: mx.array,
    block_mats: mx.array,
    n_full: int,
    D: int,
) -> mx.array:
    """Batch forward block-diagonal rotation for (n_q, D) vectors."""
    n_q = vecs.shape[0]
    mats = mx.reshape(block_mats, (n_full, 3, 3))  # (n_full, 3, 3)
    rotated_part = vecs[:, : n_full * 3].reshape(n_q, n_full, 3)  # (n_q, n_full, 3)
    # (n_full, 3, 3) applied to (n_q, n_full, 3): broadcast over n_q
    rotated_part = mx.sum(
        mats[None, :, :, :] * rotated_part[:, :, None, :], axis=-1
    )  # (n_q, n_full, 3)
    rotated_part = rotated_part.reshape(n_q, n_full * 3)

    if D > n_full * 3:
        return mx.concatenate([rotated_part, vecs[:, n_full * 3:]], axis=-1)
    return rotated_part


def _batched_rotor_inverse_rotate(
    vecs: mx.array,
    block_mats: mx.array,
    n_full: int,
    D: int,
) -> mx.array:
    """Batch inverse block-diagonal rotation for (n_q, D) vectors."""
    n_q = vecs.shape[0]
    mats = mx.reshape(block_mats, (n_full, 3, 3))
    mats_T = mx.swapaxes(mats, -2, -1)  # (n_full, 3, 3) transposed per group
    rotated_part = vecs[:, : n_full * 3].reshape(n_q, n_full, 3)
    inv_part = mx.sum(
        mats_T[None, :, :, :] * rotated_part[:, :, None, :], axis=-1
    )
    inv_part = inv_part.reshape(n_q, n_full * 3)

    if D > n_full * 3:
        return mx.concatenate([inv_part, vecs[:, n_full * 3:]], axis=-1)
    return inv_part
