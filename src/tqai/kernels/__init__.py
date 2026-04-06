"""Fused Metal kernels for PolarQuant quantize/dequantize.

Uses ``mx.fast.metal_kernel`` (MLX >= 0.16) to fuse L2-norm, rotation,
and codebook search into single GPU dispatches.  Falls back to the
Python/MLX path transparently when Metal is unavailable.
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
