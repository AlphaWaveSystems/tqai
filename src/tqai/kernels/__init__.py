"""Fused Metal kernels for PolarQuant and RotorQuant quantize/dequantize.

Uses ``mx.fast.metal_kernel`` (MLX >= 0.16) to fuse L2-norm, rotation,
and codebook search into single GPU dispatches.  Falls back to the
Python/MLX path transparently when Metal is unavailable.

PolarQuant kernels:  ``metal_quantize``, ``metal_dequantize``
RotorQuant kernels:  ``metal_rotor_quantize``, ``metal_rotor_dequantize``
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
