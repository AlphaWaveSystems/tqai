"""Tests for fused Metal quantize/dequantize kernels."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    from tqai.kernels import metal_available, metal_dequantize, metal_quantize

    HAS_METAL = metal_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal kernels unavailable")


def _python_quantize(pq, x):
    """Run quantize through the Python/MLX path (no Metal)."""
    old = pq._use_metal
    pq._use_metal = False
    try:
        result = pq.quantize(x)
    finally:
        pq._use_metal = old
    return result


def _python_dequantize(pq, indices, norms):
    """Run dequantize through the Python/MLX path (no Metal)."""
    old = pq._use_metal
    pq._use_metal = False
    try:
        result = pq.dequantize(indices, norms)
    finally:
        pq._use_metal = old
    return result


@pytest.fixture
def mlx_ops():
    from tqai.backend import get_backend

    return get_backend("mlx")


def _make_quantizer(head_dim, bits, ops):
    from tqai.quantizer import PolarQuantizer

    return PolarQuantizer(head_dim=head_dim, bits=bits, seed=42, ops=ops)


# ------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("bits", [2, 4, 8])
class TestQuantizeCorrectness:
    def test_indices_exact(self, head_dim, bits, mlx_ops):
        pq = _make_quantizer(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        mx.eval(x)

        py_idx, _ = _python_quantize(pq, x)
        m_idx, _ = metal_quantize(x, pq._rotation, pq._centroids)
        mx.eval(py_idx, m_idx)

        py_np = np.array(py_idx)
        m_np = np.array(m_idx)
        match_rate = np.mean(py_np == m_np)
        if bits <= 4:
            # Low-bit codebooks have well-separated centroids
            np.testing.assert_array_equal(m_np, py_np)
        else:
            # 8-bit: 256 closely-spaced centroids allow off-by-1 from FP rounding
            assert match_rate > 0.98, f"Index match rate {match_rate:.4f} too low"
            assert np.max(np.abs(m_np.astype(int) - py_np.astype(int))) <= 1

    def test_norms_match(self, head_dim, bits, mlx_ops):
        pq = _make_quantizer(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        mx.eval(x)

        _, py_norms = _python_quantize(pq, x)
        _, m_norms = metal_quantize(x, pq._rotation, pq._centroids)
        mx.eval(py_norms, m_norms)

        np.testing.assert_array_equal(np.array(m_norms), np.array(py_norms))

    def test_dequantize_close(self, head_dim, bits, mlx_ops):
        pq = _make_quantizer(head_dim, bits, mlx_ops)
        x = mx.random.normal((2, 4, 1, head_dim), key=mx.random.key(7))
        mx.eval(x)

        indices, norms = _python_quantize(pq, x)
        mx.eval(indices, norms)

        py_recon = _python_dequantize(pq, indices, norms)
        m_recon = metal_dequantize(indices, norms, pq._rotation, pq._centroids)
        mx.eval(py_recon, m_recon)

        np.testing.assert_allclose(
            np.array(m_recon), np.array(py_recon), atol=2e-3, rtol=1e-3
        )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_roundtrip_cosine(head_dim, mlx_ops):
    pq = _make_quantizer(head_dim, 4, mlx_ops)
    x = mx.random.normal((4, 8, 1, head_dim), key=mx.random.key(11))
    mx.eval(x)

    indices, norms = metal_quantize(x, pq._rotation, pq._centroids)
    recon = metal_dequantize(indices, norms, pq._rotation, pq._centroids)
    mx.eval(recon)

    x_np = np.array(x).reshape(-1, head_dim).astype(np.float64)
    r_np = np.array(recon).reshape(-1, head_dim).astype(np.float64)
    cos = np.sum(x_np * r_np, axis=-1) / (
        np.linalg.norm(x_np, axis=-1) * np.linalg.norm(r_np, axis=-1) + 1e-12
    )
    assert np.all(cos > 0.90), f"Min cosine similarity: {cos.min():.4f}"


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 128), (1, 8, 16, 128), (128,), (1, 1, 1, 128)],
)
def test_batch_dimensions(shape, mlx_ops):
    pq = _make_quantizer(128, 4, mlx_ops)
    x = mx.random.normal(shape, key=mx.random.key(3))
    mx.eval(x)

    indices, norms = metal_quantize(x, pq._rotation, pq._centroids)
    mx.eval(indices, norms)

    assert indices.shape == shape
    assert norms.shape == shape[:-1] + (1,)

    recon = metal_dequantize(indices, norms, pq._rotation, pq._centroids)
    mx.eval(recon)
    assert recon.shape == shape


def test_zero_vector(mlx_ops):
    pq = _make_quantizer(128, 4, mlx_ops)
    x = mx.zeros((1, 1, 128))

    indices, norms = metal_quantize(x, pq._rotation, pq._centroids)
    recon = metal_dequantize(indices, norms, pq._rotation, pq._centroids)
    mx.eval(indices, norms, recon)

    assert not np.any(np.isnan(np.array(recon)))
    assert not np.any(np.isinf(np.array(recon)))


def test_fallback_when_unavailable(mlx_ops, monkeypatch):
    """When metal_available returns False, PolarQuantizer uses Python path."""
    import tqai.kernels as kernels_mod

    monkeypatch.setattr(kernels_mod, "metal_available", lambda: False)

    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=128, bits=4, seed=42, ops=mlx_ops)
    assert pq._use_metal is False

    x = mx.random.normal((1, 4, 128), key=mx.random.key(5))
    mx.eval(x)
    indices, norms = pq.quantize(x)
    recon = pq.dequantize(indices, norms)
    mx.eval(indices, norms, recon)

    assert indices.shape == (1, 4, 128)
    assert norms.shape == (1, 4, 1)
