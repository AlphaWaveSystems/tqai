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


# ---------------------------------------------------------------------------
# Batched multi-head kernels (v0.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits,n_kv,repeats", [
    (64, 4, 2, 2),
    (128, 4, 4, 1),
    (64, 4, 2, 4),
])
def test_batched_score_keys_matches_per_head(head_dim, bits, n_kv, repeats, mlx_ops):
    """metal_batched_score_keys must match calling metal_score_keys per head."""
    from tqai.kernels import metal_batched_score_keys

    T_kv = 32
    n_q = n_kv * repeats
    pq = _make_pq(head_dim, bits, mlx_ops)
    centroids = pq._centroids
    R = pq._rotation

    # Generate per-head quantized data
    k_indices_all = []
    k_norms_all = []
    for h in range(n_kv):
        x = mx.random.normal((T_kv, head_dim), key=mx.random.key(200 + h))
        _mlx_eval(x)
        idx, nrm = pq.quantize(x)
        _mlx_eval(idx, nrm)
        k_indices_all.append(idx)
        k_norms_all.append(mx.reshape(nrm, (T_kv,)))

    k_indices = mx.stack(k_indices_all, axis=0)  # (n_kv, T_kv, D)
    k_norms = mx.stack(k_norms_all, axis=0)      # (n_kv, T_kv)

    # Generate rotated queries
    queries_raw = mx.random.normal((n_q, head_dim), key=mx.random.key(300))
    _mlx_eval(queries_raw)
    q_rotated = queries_raw @ mx.array(R).T  # (n_q, D)
    _mlx_eval(q_rotated)

    # Reference: per-head calls
    ref_scores = []
    for h_q in range(n_q):
        h_kv = h_q // repeats
        s = metal_score_keys(
            q_rotated[h_q], k_indices[h_kv], k_norms[h_kv], centroids
        )
        _mlx_eval(s)
        ref_scores.append(np.array(s))
    ref_scores = np.stack(ref_scores, axis=0)  # (n_q, T_kv)

    # Batched call
    batched_scores = metal_batched_score_keys(
        q_rotated, k_indices, k_norms, centroids, repeats
    )
    _mlx_eval(batched_scores)

    np.testing.assert_allclose(
        np.array(batched_scores), ref_scores, atol=1e-5, rtol=1e-5,
        err_msg="Batched score keys diverged from per-head"
    )


@pytest.mark.parametrize("head_dim,bits,n_kv,repeats", [
    (64, 4, 2, 2),
    (128, 4, 4, 1),
])
def test_batched_aggregate_values_matches_per_head(head_dim, bits, n_kv, repeats, mlx_ops):
    """metal_batched_aggregate_values must match per-head calls."""
    from tqai.kernels import metal_batched_aggregate_values

    T_kv = 32
    n_q = n_kv * repeats
    pq = _make_pq(head_dim, bits, mlx_ops)
    centroids = pq._centroids

    v_indices_all = []
    v_norms_all = []
    for h in range(n_kv):
        x = mx.random.normal((T_kv, head_dim), key=mx.random.key(400 + h))
        _mlx_eval(x)
        idx, nrm = pq.quantize(x)
        _mlx_eval(idx, nrm)
        v_indices_all.append(idx)
        v_norms_all.append(mx.reshape(nrm, (T_kv,)))

    v_indices = mx.stack(v_indices_all, axis=0)  # (n_kv, T_kv, D)
    v_norms = mx.stack(v_norms_all, axis=0)      # (n_kv, T_kv)

    # Random weights per query head
    weights_all = []
    for h_q in range(n_q):
        w = mx.softmax(mx.random.normal((T_kv,), key=mx.random.key(500 + h_q)), axis=-1)
        _mlx_eval(w)
        weights_all.append(w)
    weights = mx.stack(weights_all, axis=0)  # (n_q, T_kv)

    # Reference: per-head
    ref_out = []
    for h_q in range(n_q):
        h_kv = h_q // repeats
        o = metal_aggregate_values(
            weights[h_q], v_indices[h_kv], v_norms[h_kv], centroids
        )
        _mlx_eval(o)
        ref_out.append(np.array(o))
    ref_out = np.stack(ref_out, axis=0)  # (n_q, D)

    # Batched
    batched_out = metal_batched_aggregate_values(
        weights, v_indices, v_norms, centroids, repeats
    )
    _mlx_eval(batched_out)

    np.testing.assert_allclose(
        np.array(batched_out), ref_out, atol=1e-5, rtol=1e-5,
        err_msg="Batched aggregate values diverged from per-head"
    )


def test_batched_v2_matches_loop(mlx_ops):
    """batched_fused_polar_decode_v2 output matches the per-head loop."""
    from tqai.attention_fused import (
        batched_fused_polar_decode,
        batched_fused_polar_decode_v2,
    )

    B, n_q, T_q, D, bits = 1, 4, 1, 64, 4
    n_kv = 2
    T_kv = 32
    pq = _make_pq(D, bits, mlx_ops)
    scale = D**-0.5

    queries = mx.random.normal((B, n_q, T_q, D), key=mx.random.key(600))
    _mlx_eval(queries)

    # Build quantized cache
    k_indices_all = []
    k_norms_all = []
    v_indices_all = []
    v_norms_all = []
    for h in range(n_kv):
        xk = mx.random.normal((T_kv, D), key=mx.random.key(610 + h))
        xv = mx.random.normal((T_kv, D), key=mx.random.key(620 + h))
        _mlx_eval(xk, xv)
        ki, kn = pq.quantize(xk)
        vi, vn = pq.quantize(xv)
        _mlx_eval(ki, kn, vi, vn)
        k_indices_all.append(ki)
        k_norms_all.append(kn)
        v_indices_all.append(vi)
        v_norms_all.append(vn)

    k_idx = mx.stack(k_indices_all, axis=0)[None]  # (1, n_kv, T_kv, D)
    k_nrm = mx.stack(k_norms_all, axis=0)[None]    # (1, n_kv, T_kv, 1)
    v_idx = mx.stack(v_indices_all, axis=0)[None]
    v_nrm = mx.stack(v_norms_all, axis=0)[None]

    R = pq._rotation
    centroids = pq._centroids

    # v0.5 per-head loop
    out_loop = batched_fused_polar_decode(
        queries, k_idx, k_nrm, v_idx, v_nrm, R, centroids, scale
    )
    _mlx_eval(out_loop)

    # v0.6 batched
    out_batched = batched_fused_polar_decode_v2(
        queries, k_idx, k_nrm, v_idx, v_nrm,
        R, R, centroids, centroids, scale,
    )
    _mlx_eval(out_batched)

    np.testing.assert_allclose(
        np.array(out_batched), np.array(out_loop), atol=5e-3, rtol=1e-3,
        err_msg="v0.6 batched diverged from v0.5 per-head loop"
    )


def test_batched_v2_with_sinks(mlx_ops):
    """batched_fused_polar_decode_v2 handles sinks correctly."""
    from tqai.attention_fused import batched_fused_polar_decode_v2

    B, n_q, D, bits = 1, 4, 64, 4
    n_kv = 2
    T_c, T_s = 24, 4
    pq = _make_pq(D, bits, mlx_ops)
    scale = D**-0.5

    queries = mx.random.normal((B, n_q, 1, D), key=mx.random.key(700))
    sink_keys = mx.random.normal((B, n_kv, T_s, D), key=mx.random.key(710))
    sink_values = mx.random.normal((B, n_kv, T_s, D), key=mx.random.key(711))
    _mlx_eval(queries, sink_keys, sink_values)

    k_indices_all = []
    k_norms_all = []
    v_indices_all = []
    v_norms_all = []
    for h in range(n_kv):
        xk = mx.random.normal((T_c, D), key=mx.random.key(720 + h))
        xv = mx.random.normal((T_c, D), key=mx.random.key(730 + h))
        _mlx_eval(xk, xv)
        ki, kn = pq.quantize(xk)
        vi, vn = pq.quantize(xv)
        _mlx_eval(ki, kn, vi, vn)
        k_indices_all.append(ki)
        k_norms_all.append(kn)
        v_indices_all.append(vi)
        v_norms_all.append(vn)

    k_idx = mx.stack(k_indices_all, axis=0)[None]
    k_nrm = mx.stack(k_norms_all, axis=0)[None]
    v_idx = mx.stack(v_indices_all, axis=0)[None]
    v_nrm = mx.stack(v_norms_all, axis=0)[None]

    R = pq._rotation
    centroids = pq._centroids

    out = batched_fused_polar_decode_v2(
        queries, k_idx, k_nrm, v_idx, v_nrm,
        R, R, centroids, centroids, scale,
        sink_keys, sink_values,
    )
    _mlx_eval(out)

    assert out.shape == (B, n_q, 1, D)
    assert not np.any(np.isnan(np.array(out)))


def test_batched_v2_gqa(mlx_ops):
    """batched_fused_polar_decode_v2 handles GQA with repeats=4."""
    from tqai.attention_fused import batched_fused_polar_decode_v2

    B, n_q, D, bits = 1, 8, 64, 4
    n_kv = 2  # repeats = 4
    T_kv = 16
    pq = _make_pq(D, bits, mlx_ops)
    scale = D**-0.5

    queries = mx.random.normal((B, n_q, 1, D), key=mx.random.key(800))
    _mlx_eval(queries)

    k_indices_all = []
    k_norms_all = []
    v_indices_all = []
    v_norms_all = []
    for h in range(n_kv):
        xk = mx.random.normal((T_kv, D), key=mx.random.key(810 + h))
        xv = mx.random.normal((T_kv, D), key=mx.random.key(820 + h))
        _mlx_eval(xk, xv)
        ki, kn = pq.quantize(xk)
        vi, vn = pq.quantize(xv)
        _mlx_eval(ki, kn, vi, vn)
        k_indices_all.append(ki)
        k_norms_all.append(kn)
        v_indices_all.append(vi)
        v_norms_all.append(vn)

    k_idx = mx.stack(k_indices_all, axis=0)[None]
    k_nrm = mx.stack(k_norms_all, axis=0)[None]
    v_idx = mx.stack(v_indices_all, axis=0)[None]
    v_nrm = mx.stack(v_norms_all, axis=0)[None]

    R = pq._rotation
    centroids = pq._centroids

    out = batched_fused_polar_decode_v2(
        queries, k_idx, k_nrm, v_idx, v_nrm,
        R, R, centroids, centroids, scale,
    )
    _mlx_eval(out)

    assert out.shape == (B, n_q, 1, D)
    assert not np.any(np.isnan(np.array(out)))


def test_batched_rotor_v2_matches_loop(mlx_ops):
    """batched_fused_rotor_decode_v2 output matches per-head rotor decode."""
    from tqai.attention_fused import (
        fused_rotor_decode_step,
        batched_fused_rotor_decode_v2,
    )

    B, n_q, D, bits = 1, 4, 66, 4  # 66 = 22*3, all full groups
    n_kv = 2
    T_kv = 24
    rq = _make_rq(D, bits, mlx_ops)
    scale = D**-0.5
    repeats = n_q // n_kv

    queries = mx.random.normal((B, n_q, 1, D), key=mx.random.key(900))
    _mlx_eval(queries)

    k_indices_all = []
    k_norms_all = []
    v_indices_all = []
    v_norms_all = []
    for h in range(n_kv):
        xk = mx.random.normal((T_kv, D), key=mx.random.key(910 + h))
        xv = mx.random.normal((T_kv, D), key=mx.random.key(920 + h))
        _mlx_eval(xk, xv)
        ki, kn = rq.quantize(xk)
        vi, vn = rq.quantize(xv)
        _mlx_eval(ki, kn, vi, vn)
        k_indices_all.append(ki)
        k_norms_all.append(kn)
        v_indices_all.append(vi)
        v_norms_all.append(vn)

    k_idx = mx.stack(k_indices_all, axis=0)  # (n_kv, T_kv, D)
    k_nrm = mx.stack(k_norms_all, axis=0)
    v_idx = mx.stack(v_indices_all, axis=0)
    v_nrm = mx.stack(v_norms_all, axis=0)

    block_mats = rq._block_mats_mlx
    centroids = rq._centroids_mlx
    n_full = rq._n_full

    # Reference: per-head loop
    ref_outputs = []
    for h_q in range(n_q):
        h_kv = h_q // repeats
        out = fused_rotor_decode_step(
            queries[0, h_q, 0, :],
            k_idx[h_kv], mx.reshape(k_nrm[h_kv], (T_kv,)),
            v_idx[h_kv], mx.reshape(v_nrm[h_kv], (T_kv,)),
            block_mats, centroids, n_full, scale,
        )
        _mlx_eval(out)
        ref_outputs.append(np.array(out))
    ref = np.stack(ref_outputs, axis=0).reshape(B, n_q, 1, D)

    # v0.6 batched
    out_batched = batched_fused_rotor_decode_v2(
        queries,
        k_idx[None], k_nrm[None], v_idx[None], v_nrm[None],
        block_mats, block_mats, centroids, centroids,
        n_full, scale,
    )
    _mlx_eval(out_batched)

    np.testing.assert_allclose(
        np.array(out_batched), ref, atol=5e-3, rtol=1e-3,
        err_msg="v0.6 batched rotor diverged from per-head loop"
    )
