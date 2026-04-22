"""Fused Metal kernels for PolarQuant and RotorQuant quantize/dequantize,
and fused dequant-attention scoring/aggregation.

Uses ``mx.fast.metal_kernel`` (MLX >= 0.16) to fuse L2-norm, rotation,
and codebook search into single GPU dispatches.  Falls back to the
Python/MLX path transparently when Metal is unavailable.

PolarQuant kernels:  ``metal_quantize``, ``metal_dequantize``
RotorQuant kernels:  ``metal_rotor_quantize``, ``metal_rotor_dequantize``
Attention kernels:   ``metal_score_keys``, ``metal_aggregate_values``
"""

from __future__ import annotations

import functools
from typing import Tuple

import mlx.core as mx

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def metal_available() -> bool:
    """Return True when fused Metal kernels can run."""
    try:
        from packaging.version import Version

        if Version(mx.__version__) < Version("0.16"):
            return False
    except ImportError:
        parts = mx.__version__.split(".")
        if int(parts[0]) == 0 and int(parts[1]) < 16:
            return False
    return mx.metal.is_available()


# ---------------------------------------------------------------------------
# MSL source templates — D and N_LEVELS baked in at compile time
# ---------------------------------------------------------------------------

_QUANTIZE_TEMPLATE = """
    uint gid = thread_position_in_grid.x;
    const uint D = {D};
    const uint N_LEVELS = {N_LEVELS};

    uint vec_idx = gid / D;
    uint d = gid % D;
    uint base = vec_idx * D;

    // 1. L2 norm (each thread computes full norm redundantly)
    float sum_sq = 0.0f;
    for (uint j = 0; j < D; j++) {{
        float v = static_cast<float>(x[base + j]);
        sum_sq += v * v;
    }}
    float norm_val = metal::sqrt(sum_sq);

    // 2. Thread 0 stores the norm as float16
    if (d == 0) {{
        norms_out[vec_idx] = static_cast<half>(norm_val);
    }}

    // 3. Rotate: y[d] = sum_j x_unit[j] * R[d, j]
    float safe_norm = norm_val + 1e-10f;
    float y_d = 0.0f;
    for (uint j = 0; j < D; j++) {{
        float x_unit_j = static_cast<float>(x[base + j]) / safe_norm;
        y_d += x_unit_j * R[d * D + j];
    }}

    // 4. Argmin over centroids for coordinate d
    float best_diff = INFINITY;
    uint8_t best_idx = 0;
    for (uint c = 0; c < N_LEVELS; c++) {{
        float diff = metal::abs(y_d - centroids[c]);
        if (diff < best_diff) {{
            best_diff = diff;
            best_idx = static_cast<uint8_t>(c);
        }}
    }}

    // 5. Store index
    indices[gid] = best_idx;
"""

_DEQUANTIZE_TEMPLATE = """
    uint gid = thread_position_in_grid.x;
    const uint D = {D};

    uint vec_idx = gid / D;
    uint d = gid % D;
    uint base = vec_idx * D;

    // 1. Inverse rotation: x_unit_hat[d] = sum_j y_hat[j] * R[j, d]
    float val = 0.0f;
    for (uint j = 0; j < D; j++) {{
        uint8_t idx = indices[base + j];
        float centroid_val = centroids[idx];
        val += centroid_val * R[j * D + d];
    }}

    // 2. Scale by norm
    float norm_val = static_cast<float>(norms_in[vec_idx]);
    out[gid] = val * norm_val;
"""


# ---------------------------------------------------------------------------
# Kernel builders — one compiled kernel per (D, n_levels) pair
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _get_quantize_kernel(D: int, n_levels: int):
    source = _QUANTIZE_TEMPLATE.format(D=D, N_LEVELS=n_levels)
    return mx.fast.metal_kernel(
        name=f"polar_quantize_{D}_{n_levels}",
        input_names=["x", "R", "centroids"],
        output_names=["indices", "norms_out"],
        source=source,
    )


@functools.lru_cache(maxsize=32)
def _get_dequantize_kernel(D: int):
    source = _DEQUANTIZE_TEMPLATE.format(D=D)
    return mx.fast.metal_kernel(
        name=f"polar_dequantize_{D}",
        input_names=["indices", "norms_in", "R", "centroids"],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def metal_quantize(
    x: mx.array,
    rotation: mx.array,
    centroids: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Fused Metal quantize: norm + normalize + rotate + argmin.

    Args:
        x: Input vectors, shape ``(..., D)``, any float dtype.
        rotation: Orthogonal matrix, shape ``(D, D)``, float32.
        centroids: Lloyd-Max centroids, shape ``(n_levels,)``, float32.

    Returns:
        ``(indices, norms)`` where indices is uint8 ``(..., D)``
        and norms is float16 ``(..., 1)``.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    if D & (D - 1) != 0:
        raise ValueError(f"Metal kernel requires power-of-2 head_dim, got {D}")

    n_levels = centroids.shape[0]
    num_vecs = x.size // D
    x_flat = mx.reshape(x.astype(mx.float32), (-1,))

    kernel = _get_quantize_kernel(D, n_levels)
    indices_flat, norms_flat = kernel(
        inputs=[x_flat, rotation, centroids],
        output_shapes=[(num_vecs * D,), (num_vecs,)],
        output_dtypes=[mx.uint8, mx.float16],
        grid=(num_vecs * D, 1, 1),
        threadgroup=(min(D, 1024), 1, 1),
    )

    indices = mx.reshape(indices_flat, orig_shape)
    norms = mx.reshape(norms_flat, orig_shape[:-1] + (1,))
    return indices, norms


def metal_dequantize(
    indices: mx.array,
    norms: mx.array,
    rotation: mx.array,
    centroids: mx.array,
) -> mx.array:
    """Fused Metal dequantize: gather + inverse-rotate + scale.

    Args:
        indices: uint8 ``(..., D)`` codebook indices.
        norms: float16 ``(..., 1)`` L2 norms.
        rotation: Orthogonal matrix ``(D, D)``, float32.
        centroids: Lloyd-Max centroids ``(n_levels,)``, float32.

    Returns:
        Reconstructed vectors, float32, shape ``(..., D)``.
    """
    orig_shape = indices.shape
    D = orig_shape[-1]
    num_vecs = indices.size // D

    indices_flat = mx.reshape(indices, (-1,))
    norms_flat = mx.reshape(norms, (-1,))

    kernel = _get_dequantize_kernel(D)
    (out_flat,) = kernel(
        inputs=[indices_flat, norms_flat, rotation, centroids],
        output_shapes=[(num_vecs * D,)],
        output_dtypes=[mx.float32],
        grid=(num_vecs * D, 1, 1),
        threadgroup=(min(D, 1024), 1, 1),
    )

    return mx.reshape(out_flat, orig_shape)


# ---------------------------------------------------------------------------
# RotorQuant MSL source templates
#
# Rotation model: block-diagonal 3×3 quaternion rotations (Clifford rotors).
# ``block_mats`` is a flat float32 array of shape (N_FULL * 9,) storing
# N_FULL row-major 3×3 rotation matrices.  Group g owns dimensions
# [g*3, g*3+3).  Remainder dimensions (d >= N_FULL*3) pass through unrotated.
#
# Forward (quantize):
#   y[g*3 + r] = R_g[r, :] · x_unit[g*3 : g*3+3]
#   block_mats[g*9 + r*3 + c] = R_g[r, c]
#
# Inverse (dequantize):
#   x_unit_hat[g*3 + r] = R_g[:, r] · y_hat[g*3 : g*3+3]
#   = R_g[0,r]*y0 + R_g[1,r]*y1 + R_g[2,r]*y2
#   = block_mats[g*9 + r] * y0 + block_mats[g*9+3 + r] * y1 + block_mats[g*9+6 + r] * y2
# ---------------------------------------------------------------------------

_ROTOR_QUANTIZE_TEMPLATE = """
    uint gid = thread_position_in_grid.x;
    const uint D       = {D};
    const uint N_LEVELS = {N_LEVELS};
    const uint N_FULL  = {N_FULL};

    uint vec_idx = gid / D;
    uint d       = gid % D;
    uint base    = vec_idx * D;

    // 1. L2 norm — computed redundantly per thread to avoid threadgroup sync
    float sum_sq = 0.0f;
    for (uint j = 0; j < D; j++) {{
        float v = (float)x[base + j];
        sum_sq += v * v;
    }}
    float norm_val = metal::sqrt(sum_sq);
    if (d == 0) {{
        norms_out[vec_idx] = (half)norm_val;
    }}
    float safe_norm = norm_val + 1e-10f;

    // 2. Block-diagonal rotor rotation
    // Each group of 3 dims is rotated independently by its 3x3 matrix.
    uint g = d / 3;
    uint r = d % 3;
    float y_d;
    if (g < N_FULL) {{
        // Forward: y_d = R_g[r, :] · x_unit[g*3 : g*3+3]
        uint mat_base = g * 9 + r * 3;
        y_d = block_mats[mat_base + 0] * ((float)x[base + g*3 + 0] / safe_norm)
            + block_mats[mat_base + 1] * ((float)x[base + g*3 + 1] / safe_norm)
            + block_mats[mat_base + 2] * ((float)x[base + g*3 + 2] / safe_norm);
    }} else {{
        // Remainder dim: no rotation
        y_d = (float)x[base + d] / safe_norm;
    }}

    // 3. Nearest centroid (argmin)
    float best_diff = INFINITY;
    uint8_t best_idx = 0;
    for (uint c = 0; c < N_LEVELS; c++) {{
        float diff = metal::abs(y_d - centroids[c]);
        if (diff < best_diff) {{
            best_diff = diff;
            best_idx = (uint8_t)c;
        }}
    }}
    indices[gid] = best_idx;
"""

_ROTOR_DEQUANTIZE_TEMPLATE = """
    uint gid = thread_position_in_grid.x;
    const uint D      = {D};
    const uint N_FULL = {N_FULL};

    uint vec_idx = gid / D;
    uint d       = gid % D;
    uint base    = vec_idx * D;

    // 1. Centroid lookup + inverse block-diagonal rotor rotation
    uint g = d / 3;
    uint r = d % 3;
    float x_unit_hat;
    if (g < N_FULL) {{
        float y0 = centroids[(uint)indices[base + g*3 + 0]];
        float y1 = centroids[(uint)indices[base + g*3 + 1]];
        float y2 = centroids[(uint)indices[base + g*3 + 2]];
        // Inverse: x_unit_hat[r] = R_g[:, r] · y_hat = col r of R_g
        // R_g[row, col] = block_mats[g*9 + row*3 + col]
        // col r: block_mats[g*9 + 0*3 + r], [g*9 + 1*3 + r], [g*9 + 2*3 + r]
        x_unit_hat = block_mats[g*9 + 0 + r] * y0
                   + block_mats[g*9 + 3 + r] * y1
                   + block_mats[g*9 + 6 + r] * y2;
    }} else {{
        x_unit_hat = centroids[(uint)indices[base + d]];
    }}

    // 2. Scale by norm
    out[gid] = x_unit_hat * (float)norms_in[vec_idx];
"""


# ---------------------------------------------------------------------------
# RotorQuant kernel builders — keyed on (D, n_levels, n_full) / (D, n_full)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _get_rotor_quantize_kernel(D: int, n_levels: int, n_full: int):
    source = _ROTOR_QUANTIZE_TEMPLATE.format(D=D, N_LEVELS=n_levels, N_FULL=n_full)
    return mx.fast.metal_kernel(
        name=f"rotor_quantize_{D}_{n_levels}_{n_full}",
        input_names=["x", "block_mats", "centroids"],
        output_names=["indices", "norms_out"],
        source=source,
    )


@functools.lru_cache(maxsize=32)
def _get_rotor_dequantize_kernel(D: int, n_full: int):
    source = _ROTOR_DEQUANTIZE_TEMPLATE.format(D=D, N_FULL=n_full)
    return mx.fast.metal_kernel(
        name=f"rotor_dequantize_{D}_{n_full}",
        input_names=["indices", "norms_in", "block_mats", "centroids"],
        output_names=["out"],
        source=source,
    )


# ---------------------------------------------------------------------------
# RotorQuant Python wrappers
# ---------------------------------------------------------------------------


def metal_rotor_quantize(
    x: mx.array,
    block_mats: mx.array,
    centroids: mx.array,
    n_full: int,
) -> tuple[mx.array, mx.array]:
    """Fused Metal quantize for RotorQuantizer: norm + rotor-rotate + argmin.

    Args:
        x: Input vectors ``(..., D)``, any float dtype.
        block_mats: Flat float32 array of shape ``(n_full * 9,)`` — the
            N_FULL row-major 3×3 rotation matrices packed contiguously.
        centroids: Lloyd-Max centroids ``(n_levels,)`` float32.
        n_full: Number of complete 3-dim groups (``D // 3``).

    Returns:
        ``(indices, norms)`` — uint8 ``(..., D)`` and float16 ``(..., 1)``.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    n_levels = centroids.shape[0]
    num_vecs = x.size // D

    x_flat = mx.reshape(x.astype(mx.float32), (-1,))
    kernel = _get_rotor_quantize_kernel(D, n_levels, n_full)

    indices_flat, norms_flat = kernel(
        inputs=[x_flat, block_mats, centroids],
        output_shapes=[(num_vecs * D,), (num_vecs,)],
        output_dtypes=[mx.uint8, mx.float16],
        grid=(num_vecs * D, 1, 1),
        threadgroup=(min(D, 1024), 1, 1),
    )

    indices = mx.reshape(indices_flat, orig_shape)
    norms = mx.reshape(norms_flat, orig_shape[:-1] + (1,))
    return indices, norms


def metal_rotor_dequantize(
    indices: mx.array,
    norms: mx.array,
    block_mats: mx.array,
    centroids: mx.array,
    n_full: int,
) -> mx.array:
    """Fused Metal dequantize for RotorQuantizer: gather + inverse-rotate + scale.

    Args:
        indices: uint8 ``(..., D)`` codebook indices.
        norms: float16 ``(..., 1)`` L2 norms.
        block_mats: Flat float32 ``(n_full * 9,)`` rotation matrices.
        centroids: Lloyd-Max centroids ``(n_levels,)`` float32.
        n_full: Number of complete 3-dim groups.

    Returns:
        Reconstructed float32 vectors ``(..., D)``.
    """
    orig_shape = indices.shape
    D = orig_shape[-1]
    num_vecs = indices.size // D

    indices_flat = mx.reshape(indices, (-1,))
    norms_flat = mx.reshape(norms, (-1,))

    kernel = _get_rotor_dequantize_kernel(D, n_full)

    (out_flat,) = kernel(
        inputs=[indices_flat, norms_flat, block_mats, centroids],
        output_shapes=[(num_vecs * D,)],
        output_dtypes=[mx.float32],
        grid=(num_vecs * D, 1, 1),
        threadgroup=(min(D, 1024), 1, 1),
    )

    return mx.reshape(out_flat, orig_shape)


# ---------------------------------------------------------------------------
# Fused dequant-attention kernels
#
# These kernels keep K and V in compressed (uint8 indices + float16 norms)
# form and fuse the dequantization into the attention score / aggregation
# passes.  This halves DRAM bandwidth at 4-bit vs float16 KV storage.
#
# Mathematical basis:
#   PolarQuant encodes: indices[d] = argmin_c |y[d] - c|  where y = R @ x_unit
#   Dequant: x_hat = norm * R.T @ centroids[indices]
#
#   Attention score for a query q and key k:
#     score = q · x_hat_k
#           = norm_k * q · (R.T @ centroids[indices_k])
#           = norm_k * (R @ q) · centroids[indices_k]
#
#   If we precompute q_rotated = R @ q (once per decode step), then:
#     score = norm_k * dot(q_rotated, centroids[indices_k])
#   — a simple gather-then-dot, no per-key rotation needed.
#
#   For the value aggregation:
#     output = sum_k weights_k * norm_k * R.T @ centroids[v_indices_k]
#            = R.T @ (sum_k weights_k * norm_k * centroids[v_indices_k])
#            = R.T @ output_rotated
#
#   Kernel computes output_rotated[d] per dimension, then the caller does
#   a single R.T @ output_rotated to recover the spatial output.
#
#   RotorQuant: same math, block-diagonal R (O(D) instead of O(D^2)).
# ---------------------------------------------------------------------------

_SCORE_KEYS_TEMPLATE = """
    // One thread per key position k.
    // scores_out[k] = dot(q_rotated, centroids[k_indices[k,:]]) * k_norms[k]
    uint k = thread_position_in_grid.x;
    const uint D = {D};

    float score = 0.0f;
    uint base = k * D;
    for (uint d = 0; d < D; d++) {{
        float q_d = q_rotated[d];
        uint8_t idx = k_indices[base + d];
        score += q_d * centroids[idx];
    }}
    scores_out[k] = score * (float)k_norms[k];
"""

_AGGREGATE_VALUES_TEMPLATE = """
    // One thread per output dimension d (in rotated space).
    // out_rotated[d] = sum_k weights[k] * v_norms[k] * centroids[v_indices[k,d]]
    uint d = thread_position_in_grid.x;
    const uint D = {D};
    const uint T_kv = t_kv[0];

    float acc = 0.0f;
    for (uint k = 0; k < T_kv; k++) {{
        float w = weights[k] * (float)v_norms[k];
        uint8_t idx = v_indices[k * D + d];
        acc += w * centroids[idx];
    }}
    out_rotated[d] = acc;
"""


@functools.lru_cache(maxsize=32)
def _get_score_keys_kernel(D: int, n_levels: int):
    source = _SCORE_KEYS_TEMPLATE.format(D=D)
    return mx.fast.metal_kernel(
        name=f"score_keys_{D}_{n_levels}",
        input_names=["q_rotated", "k_indices", "k_norms", "centroids"],
        output_names=["scores_out"],
        source=source,
    )


@functools.lru_cache(maxsize=32)
def _get_aggregate_values_kernel(D: int, n_levels: int):
    source = _AGGREGATE_VALUES_TEMPLATE.format(D=D)
    return mx.fast.metal_kernel(
        name=f"aggregate_values_{D}_{n_levels}",
        input_names=["weights", "v_norms", "v_indices", "centroids", "t_kv"],
        output_names=["out_rotated"],
        source=source,
    )


def metal_score_keys(
    q_rotated: mx.array,
    k_indices: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
) -> mx.array:
    """Fused gather-dot: score all compressed keys against a pre-rotated query.

    Computes ``scores[k] = dot(q_rotated, centroids[k_indices[k]]) * k_norms[k]``
    for every key position in a single Metal dispatch.  Avoids materializing
    the float32 key buffer — reads uint8 + float16 instead.

    Call ``R @ q`` (PolarQuant) or ``block_R @ q`` (RotorQuant) before this
    function to obtain ``q_rotated``.

    Args:
        q_rotated: Pre-rotated query, shape ``(D,)``, float32.
        k_indices: uint8 key cache indices, shape ``(T_kv, D)``.
        k_norms: float16 key norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        centroids: Lloyd-Max codebook, shape ``(n_levels,)``, float32.

    Returns:
        Raw attention scores (pre-softmax), shape ``(T_kv,)``, float32.
    """
    D = q_rotated.shape[0]
    n_levels = centroids.shape[0]
    T_kv = k_indices.shape[0]

    k_indices_flat = mx.reshape(k_indices, (-1,))
    k_norms_flat = mx.reshape(k_norms, (-1,))

    kernel = _get_score_keys_kernel(D, n_levels)
    (scores,) = kernel(
        inputs=[
            q_rotated.astype(mx.float32),
            k_indices_flat,
            k_norms_flat.astype(mx.float16),
            centroids.astype(mx.float32),
        ],
        output_shapes=[(T_kv,)],
        output_dtypes=[mx.float32],
        grid=(T_kv, 1, 1),
        threadgroup=(1, 1, 1),
    )
    return scores


def metal_aggregate_values(
    weights: mx.array,
    v_indices: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
) -> mx.array:
    """Fused weighted-gather: aggregate compressed values in rotated space.

    Computes ``out_rotated[d] = sum_k weights[k] * v_norms[k] * centroids[v_indices[k,d]]``
    for all output dimensions simultaneously.  Call ``R.T @ out_rotated``
    (PolarQuant) or the block-diagonal inverse rotation (RotorQuant) to
    recover the final spatial output.

    Args:
        weights: Softmax attention weights, shape ``(T_kv,)``, float32.
        v_indices: uint8 value cache indices, shape ``(T_kv, D)``.
        v_norms: float16 value norms, shape ``(T_kv,)`` or ``(T_kv, 1)``.
        centroids: Lloyd-Max codebook, shape ``(n_levels,)``, float32.

    Returns:
        Aggregated values in rotated space, shape ``(D,)``, float32.
    """
    D = v_indices.shape[-1]
    T_kv = v_indices.shape[0]
    n_levels = centroids.shape[0]

    v_indices_flat = mx.reshape(v_indices, (-1,))
    v_norms_flat = mx.reshape(v_norms, (-1,))
    t_kv_arr = mx.array([T_kv], dtype=mx.uint32)

    kernel = _get_aggregate_values_kernel(D, n_levels)
    (out_rotated,) = kernel(
        inputs=[
            weights.astype(mx.float32),
            v_norms_flat.astype(mx.float16),
            v_indices_flat,
            centroids.astype(mx.float32),
            t_kv_arr,
        ],
        output_shapes=[(D,)],
        output_dtypes=[mx.float32],
        grid=(D, 1, 1),
        threadgroup=(min(D, 1024), 1, 1),
    )
    return out_rotated
