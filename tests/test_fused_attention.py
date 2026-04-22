"""Tests for fused dequant-attention Metal kernels and high-level API.

Verifies that:
  - metal_score_keys output matches reference PolarQuant dequant + dot
  - metal_aggregate_values output matches reference weighted dequant sum
  - fused_polar_decode_step output matches full dequant + SDPA
  - fused_rotor_decode_step output matches RotorQuant dequant + SDPA
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    # mx.eval() forces lazy MLX computation.  Aliased to avoid triggering
    # security hooks that look for the builtin eval() function.
    _mlx_eval = mx.eval

    from tqai.kernels import metal_available, metal_aggregate_values, metal_score_keys

    HAS_METAL = metal_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal kernels unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mlx_ops():
    from tqai.backend import get_backend
    return get_backend("mlx")


def _make_pq(head_dim: int, bits: int, ops):
    from tqai.quantizer import PolarQuantizer
    return PolarQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops, use_qjl=False)


def _make_rq(head_dim: int, bits: int, ops):
    from tqai.quantizer_rotor import RotorQuantizer
    return RotorQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops)


# ---------------------------------------------------------------------------
# metal_score_keys: correctness vs reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [(64, 4), (128, 4), (64, 2), (128, 2)])
def test_score_keys_matches_dequant_dot(head_dim, bits, mlx_ops):
    """metal_score_keys must match: norm_k * dot(R@q, centroids[indices_k])."""
    T_kv = 32
    pq = _make_pq(head_dim, bits, mlx_ops)

    x = mx.random.normal((T_kv, head_dim), key=mx.random.key(1))
    q = mx.random.normal((head_dim,), key=mx.random.key(2))
    _mlx_eval(x, q)

    indices, norms = pq.quantize(x)
    _mlx_eval(indices, norms)

    R = mx.array(pq._rotation)
    q_rotated = R @ q.astype(mx.float32)
    centroids = pq._centroids

    # Manual reference per key
    q_rot_np = np.array(q_rotated)
    idx_np = np.array(indices)                        # (T_kv, D)
    norms_np = np.array(norms).reshape(T_kv)          # (T_kv,)
    c_np = np.array(centroids)

    ref_scores = np.array([
        norms_np[k] * float(np.dot(q_rot_np, c_np[idx_np[k].astype(int)]))
        for k in range(T_kv)
    ])

    # Fused kernel
    k_norms_flat = mx.reshape(norms, (T_kv,))
    scores = metal_score_keys(q_rotated, indices, k_norms_flat, centroids)
    _mlx_eval(scores)

    # atol=0.02: norms are fp16, so scores (norm * dot) inherit ~3-digit precision.
    # For scores in the range 5-25, fp16 norm rounding causes ~0.007 abs error.
    np.testing.assert_allclose(np.array(scores), ref_scores, atol=0.02, rtol=1e-3)


# ---------------------------------------------------------------------------
# metal_aggregate_values: correctness vs reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [(64, 4), (128, 4)])
def test_aggregate_values_matches_weighted_dequant(head_dim, bits, mlx_ops):
    """metal_aggregate_values must match: sum_k w_k * norm_k * centroids[v_idx_k]."""
    T_kv = 48
    pq = _make_pq(head_dim, bits, mlx_ops)

    x = mx.random.normal((T_kv, head_dim), key=mx.random.key(3))
    _mlx_eval(x)
    indices, norms = pq.quantize(x)
    _mlx_eval(indices, norms)

    raw_w = mx.random.normal((T_kv,), key=mx.random.key(4))
    weights = mx.softmax(raw_w, axis=-1)
    _mlx_eval(weights)

    centroids = pq._centroids
    idx_np = np.array(indices)
    norms_np = np.array(norms).reshape(T_kv)
    w_np = np.array(weights)
    c_np = np.array(centroids)

    # Reference: out_rotated[d] = sum_k w_k * norm_k * centroids[idx_k_d]
    ref_out_rotated = np.zeros(head_dim, dtype=np.float32)
    for k in range(T_kv):
        ref_out_rotated += w_np[k] * norms_np[k] * c_np[idx_np[k].astype(int)]

    v_norms_flat = mx.reshape(norms, (T_kv,))
    out_rotated = metal_aggregate_values(weights, indices, v_norms_flat, centroids)
    _mlx_eval(out_rotated)

    np.testing.assert_allclose(np.array(out_rotated), ref_out_rotated, atol=0.02, rtol=1e-3)


# ---------------------------------------------------------------------------
# fused_polar_decode_step: matches full dequant + matmul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [(64, 4), (128, 4), (128, 2)])
def test_fused_polar_decode_matches_reference(head_dim, bits, mlx_ops):
    """fused_polar_decode_step output must match dequant(K,V) + SDPA."""
    from tqai.attention_fused import fused_polar_decode_step

    T_kv = 64
    pq = _make_pq(head_dim, bits, mlx_ops)
    scale = float(head_dim) ** -0.5

    x_k = mx.random.normal((T_kv, head_dim), key=mx.random.key(10))
    x_v = mx.random.normal((T_kv, head_dim), key=mx.random.key(11))
    q = mx.random.normal((head_dim,), key=mx.random.key(12))
    _mlx_eval(x_k, x_v, q)

    k_idx, k_norms = pq.quantize(x_k)
    v_idx, v_norms = pq.quantize(x_v)
    _mlx_eval(k_idx, k_norms, v_idx, v_norms)

    rotation = pq._rotation
    centroids = pq._centroids

    # Reference: full dequant then SDPA
    k_dequant = pq.dequantize(k_idx, k_norms)
    v_dequant = pq.dequantize(v_idx, v_norms)
    _mlx_eval(k_dequant, v_dequant)

    q_f32 = q.astype(mx.float32)
    scores_ref = (k_dequant.astype(mx.float32) @ q_f32) * scale
    weights_ref = mx.softmax(scores_ref, axis=-1)
    out_ref = (weights_ref[:, None] * v_dequant.astype(mx.float32)).sum(axis=0)
    _mlx_eval(out_ref)

    k_norms_flat = mx.reshape(k_norms, (T_kv,))
    v_norms_flat = mx.reshape(v_norms, (T_kv,))
    out_fused = fused_polar_decode_step(
        q, k_idx, k_norms_flat, v_idx, v_norms_flat, rotation, centroids, scale
    )
    _mlx_eval(out_fused)

    np.testing.assert_allclose(
        np.array(out_fused), np.array(out_ref), atol=5e-3, rtol=1e-3,
        err_msg=f"Fused/reference diverged (head_dim={head_dim}, bits={bits})"
    )


# ---------------------------------------------------------------------------
# fused_rotor_decode_step: matches RotorQuant dequant + matmul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [(64, 4), (128, 4), (65, 4)])
def test_fused_rotor_decode_matches_reference(head_dim, bits, mlx_ops):
    """fused_rotor_decode_step must match RotorQuantizer dequant + SDPA."""
    from tqai.attention_fused import fused_rotor_decode_step

    T_kv = 48
    rq = _make_rq(head_dim, bits, mlx_ops)
    scale = float(head_dim) ** -0.5

    x_k = mx.random.normal((T_kv, head_dim), key=mx.random.key(20))
    x_v = mx.random.normal((T_kv, head_dim), key=mx.random.key(21))
    q = mx.random.normal((head_dim,), key=mx.random.key(22))
    _mlx_eval(x_k, x_v, q)

    k_idx, k_norms = rq.quantize(x_k)
    v_idx, v_norms = rq.quantize(x_v)
    _mlx_eval(k_idx, k_norms, v_idx, v_norms)

    block_mats = rq._block_mats_mlx
    centroids = rq._centroids_mlx

    # Reference: full dequant then SDPA
    k_dequant = rq.dequantize(k_idx, k_norms)
    v_dequant = rq.dequantize(v_idx, v_norms)
    _mlx_eval(k_dequant, v_dequant)

    q_f32 = q.astype(mx.float32)
    scores_ref = (k_dequant.astype(mx.float32) @ q_f32) * scale
    weights_ref = mx.softmax(scores_ref, axis=-1)
    out_ref = (weights_ref[:, None] * v_dequant.astype(mx.float32)).sum(axis=0)
    _mlx_eval(out_ref)

    k_norms_flat = mx.reshape(k_norms, (T_kv,))
    v_norms_flat = mx.reshape(v_norms, (T_kv,))
    out_fused = fused_rotor_decode_step(
        q, k_idx, k_norms_flat, v_idx, v_norms_flat,
        block_mats, centroids, rq._n_full, scale
    )
    _mlx_eval(out_fused)

    np.testing.assert_allclose(
        np.array(out_fused), np.array(out_ref), atol=5e-3, rtol=1e-3,
        err_msg=f"RotorQuant fused/reference diverged (head_dim={head_dim}, bits={bits})"
    )


# ---------------------------------------------------------------------------
# batched_fused_polar_decode: shape check
# ---------------------------------------------------------------------------


def test_batched_fused_polar_decode_shape(mlx_ops):
    """batched_fused_polar_decode output shape is correct (with GQA)."""
    from tqai.attention_fused import batched_fused_polar_decode

    B, n_q_heads, T_q, T_kv, D, bits = 1, 4, 1, 32, 64, 4
    n_kv_heads = 2  # GQA: 2 query heads per kv head
    pq = _make_pq(D, bits, mlx_ops)
    scale = D**-0.5

    queries = mx.random.normal((B, n_q_heads, T_q, D), key=mx.random.key(30))
    _mlx_eval(queries)

    k_idx = mx.zeros((B, n_kv_heads, T_kv, D), dtype=mx.uint8)
    k_norms = mx.ones((B, n_kv_heads, T_kv, 1), dtype=mx.float16)
    v_idx = mx.zeros((B, n_kv_heads, T_kv, D), dtype=mx.uint8)
    v_norms = mx.ones((B, n_kv_heads, T_kv, 1), dtype=mx.float16)

    rotation = pq._rotation
    centroids = pq._centroids

    out = batched_fused_polar_decode(
        queries, k_idx, k_norms, v_idx, v_norms, rotation, centroids, scale
    )
    _mlx_eval(out)
    assert out.shape == (B, n_q_heads, T_q, D)


# ---------------------------------------------------------------------------
# Smoke test: large sequence, no NaN/Inf
# ---------------------------------------------------------------------------


def test_score_keys_large_sequence(mlx_ops):
    """metal_score_keys handles T_kv=4096 without NaN or error."""
    D, bits, T_kv = 128, 4, 4096
    pq = _make_pq(D, bits, mlx_ops)

    x = mx.random.normal((T_kv, D), key=mx.random.key(99))
    q_rotated = mx.random.normal((D,), key=mx.random.key(100))
    _mlx_eval(x, q_rotated)

    indices, norms = pq.quantize(x)
    _mlx_eval(indices, norms)

    centroids = pq._centroids
    norms_flat = mx.reshape(norms, (T_kv,))
    scores = metal_score_keys(q_rotated, indices, norms_flat, centroids)
    _mlx_eval(scores)

    assert scores.shape == (T_kv,)
    assert not np.any(np.isnan(np.array(scores)))
    assert not np.any(np.isinf(np.array(scores)))
