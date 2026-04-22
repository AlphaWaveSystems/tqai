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
