"""Tests for fused Metal rotor quantize/dequantize kernels.

Mirrors the structure of test_metal_kernels.py so that the rotor
kernels are held to the same correctness bar as the polar kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    # mx.eval() forces lazy MLX computation — aliased to avoid false-positive
    # security linters that flag Python's builtin eval().
    _mlx_eval = mx.eval

    from tqai.kernels import (
        metal_available,
        metal_rotor_dequantize,
        metal_rotor_quantize,
    )

    HAS_METAL = metal_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal kernels unavailable")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mlx_ops():
    from tqai.backend import get_backend

    return get_backend("mlx")


def _make_rq(head_dim: int, bits: int, ops):
    from tqai.quantizer_rotor import RotorQuantizer

    return RotorQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops)


def _python_quantize(rq, x):
    """Force the Python (non-Metal) quantize path."""
    prev = rq._use_metal
    rq._use_metal = False
    try:
        return rq.quantize(x)
    finally:
        rq._use_metal = prev


def _python_dequantize(rq, indices, norms):
    """Force the Python (non-Metal) dequantize path."""
    prev = rq._use_metal
    rq._use_metal = False
    try:
        return rq.dequantize(indices, norms)
    finally:
        rq._use_metal = prev


# ---------------------------------------------------------------------------
# Correctness: Metal output must match Python output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [
    (64, 2), (64, 4),
    (128, 2), (128, 4),
    (65, 4),   # non-divisible by 3 — remainder dims pass through unrotated
    (126, 4),  # exactly divisible by 3
])
class TestRotorQuantizeCorrectness:
    def test_indices_exact(self, head_dim, bits, mlx_ops):
        """Metal and Python paths must produce identical uint8 indices."""
        rq = _make_rq(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        _mlx_eval(x)

        py_idx, _ = _python_quantize(rq, x)
        m_idx, _ = metal_rotor_quantize(
            x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
        )
        _mlx_eval(py_idx, m_idx)

        py_np = np.array(py_idx)
        m_np = np.array(m_idx)

        if bits <= 4:
            # Well-separated centroids: expect bit-exact agreement
            np.testing.assert_array_equal(m_np, py_np)
        else:
            # 8-bit: 256 closely-spaced centroids allow off-by-1 from FP rounding
            match_rate = np.mean(py_np == m_np)
            assert match_rate > 0.98, f"Index match rate {match_rate:.4f} too low"
            assert np.max(np.abs(m_np.astype(int) - py_np.astype(int))) <= 1

    def test_norms_match(self, head_dim, bits, mlx_ops):
        """Norms from Metal and Python paths must be identical (stored as fp16)."""
        rq = _make_rq(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        _mlx_eval(x)

        _, py_norms = _python_quantize(rq, x)
        _, m_norms = metal_rotor_quantize(
            x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
        )
        _mlx_eval(py_norms, m_norms)

        np.testing.assert_array_equal(np.array(m_norms), np.array(py_norms))

    def test_dequantize_close(self, head_dim, bits, mlx_ops):
        """Metal dequantize must closely match the Python reference."""
        rq = _make_rq(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        _mlx_eval(x)

        indices, norms = _python_quantize(rq, x)
        _mlx_eval(indices, norms)

        py_recon = _python_dequantize(rq, indices, norms)
        m_recon = metal_rotor_dequantize(
            indices, norms, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
        )
        _mlx_eval(py_recon, m_recon)

        np.testing.assert_allclose(
            np.array(m_recon), np.array(py_recon), atol=2e-3, rtol=1e-3
        )


# ---------------------------------------------------------------------------
# Round-trip quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 65])
def test_roundtrip_cosine(head_dim, mlx_ops):
    """Metal round-trip should achieve high cosine similarity at 4-bit."""
    rq = _make_rq(head_dim, 4, mlx_ops)
    x = mx.random.normal((4, 8, 1, head_dim), key=mx.random.key(11))
    _mlx_eval(x)

    indices, norms = metal_rotor_quantize(
        x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    recon = metal_rotor_dequantize(
        indices, norms, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    _mlx_eval(recon)

    x_np = np.array(x).reshape(-1, head_dim).astype(np.float64)
    r_np = np.array(recon).reshape(-1, head_dim).astype(np.float64)
    cos = np.sum(x_np * r_np, axis=-1) / (
        np.linalg.norm(x_np, axis=-1) * np.linalg.norm(r_np, axis=-1) + 1e-12
    )
    assert np.all(cos > 0.90), f"Min cosine similarity: {cos.min():.4f}"


# ---------------------------------------------------------------------------
# Batch dimension handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    (2, 4, 128),
    (1, 8, 16, 128),
    (128,),
    (1, 1, 1, 128),
    (4, 65),      # non-divisible head_dim
])
def test_batch_dimensions(shape, mlx_ops):
    head_dim = shape[-1]
    rq = _make_rq(head_dim, 4, mlx_ops)
    x = mx.random.normal(shape, key=mx.random.key(3))
    _mlx_eval(x)

    indices, norms = metal_rotor_quantize(
        x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    _mlx_eval(indices, norms)
    assert indices.shape == shape
    assert norms.shape == shape[:-1] + (1,)

    recon = metal_rotor_dequantize(
        indices, norms, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    _mlx_eval(recon)
    assert recon.shape == shape


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_vector(mlx_ops):
    """Zero input must not produce NaN or Inf."""
    rq = _make_rq(128, 4, mlx_ops)
    x = mx.zeros((1, 1, 128))

    indices, norms = metal_rotor_quantize(
        x, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    recon = metal_rotor_dequantize(
        indices, norms, rq._block_mats_mlx, rq._centroids_mlx, rq._n_full
    )
    _mlx_eval(indices, norms, recon)

    assert not np.any(np.isnan(np.array(recon)))
    assert not np.any(np.isinf(np.array(recon)))


def test_fallback_when_metal_unavailable(mlx_ops, monkeypatch):
    """When metal_available() returns False, RotorQuantizer uses Python path."""
    import tqai.kernels as kernels_mod

    monkeypatch.setattr(kernels_mod, "metal_available", lambda: False)

    from tqai.quantizer_rotor import RotorQuantizer

    rq = RotorQuantizer(head_dim=128, bits=4, seed=42, ops=mlx_ops)
    assert rq._use_metal is False

    x = mx.random.normal((1, 4, 128), key=mx.random.key(5))
    _mlx_eval(x)
    indices, norms = rq.quantize(x)
    recon = rq.dequantize(indices, norms)
    _mlx_eval(indices, norms, recon)

    assert indices.shape == (1, 4, 128)
    assert norms.shape == (1, 4, 1)
    assert not np.any(np.isnan(np.array(recon)))


# ---------------------------------------------------------------------------
# High-level round-trip consistency (RotorQuantizer API vs raw kernel)
# ---------------------------------------------------------------------------


def test_high_level_uses_metal(mlx_ops):
    """RotorQuantizer.quantize() output must match the raw Metal kernel."""
    rq = _make_rq(128, 4, mlx_ops)
    assert rq._use_metal, "Precondition: Metal must be active"

    x = mx.random.normal((2, 8, 64, 128), key=mx.random.key(99))
    _mlx_eval(x)

    m_idx, m_norms = rq.quantize(x)
    m_recon = rq.dequantize(m_idx, m_norms)
    _mlx_eval(m_idx, m_norms, m_recon)

    py_idx, py_norms = _python_quantize(rq, x)
    py_recon = _python_dequantize(rq, py_idx, py_norms)
    _mlx_eval(py_idx, py_norms, py_recon)

    np.testing.assert_array_equal(np.array(m_idx), np.array(py_idx))
    np.testing.assert_array_equal(np.array(m_norms), np.array(py_norms))
    np.testing.assert_allclose(np.array(m_recon), np.array(py_recon), atol=2e-3)
