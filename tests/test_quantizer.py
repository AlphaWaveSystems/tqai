from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b", [2, 3, 4])
def test_roundtrip_shape(d, b, ops):
    """quantize -> dequantize preserves shape."""
    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
    x = ops.randn((2, 4, d), seed=99)  # (batch, heads, dim)
    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)
    assert ops.to_numpy(x_hat).shape == (2, 4, d)


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b", [3, 4])
def test_cosine_similarity(d, b, ops):
    """Round-trip should preserve direction well at 3+ bits."""
    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
    x = ops.randn((100, d), seed=7)
    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)

    x_np = ops.to_numpy(x)
    x_hat_np = ops.to_numpy(x_hat)

    # Per-vector cosine similarity
    dot = np.sum(x_np * x_hat_np, axis=-1)
    norm_x = np.linalg.norm(x_np, axis=-1)
    norm_xhat = np.linalg.norm(x_hat_np, axis=-1)
    cos_sim = dot / (norm_x * norm_xhat + 1e-10)

    mean_cos = np.mean(cos_sim)
    assert mean_cos > 0.95, f"Mean cosine similarity {mean_cos:.4f} too low for d={d}, b={b}"


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b", [2, 3, 4])
def test_mse_reasonable(d, b, ops):
    """MSE should decrease with more bits."""
    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
    x = ops.randn((200, d), seed=11)
    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)

    x_np = ops.to_numpy(x)
    x_hat_np = ops.to_numpy(x_hat)
    mse = np.mean((x_np - x_hat_np) ** 2)

    # MSE should be finite and positive
    assert 0 < mse < 10.0, f"MSE={mse} out of reasonable range"


@pytest.mark.parametrize("d", [64, 128])
def test_mse_decreases_with_bits(d, ops):
    """More bits should give lower MSE."""
    from tqai.quantizer import PolarQuantizer

    mses = []
    for b in [2, 3, 4]:
        pq = PolarQuantizer(head_dim=d, bits=b, seed=42, ops=ops)
        x = ops.randn((200, d), seed=11)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        x_np = ops.to_numpy(x)
        x_hat_np = ops.to_numpy(x_hat)
        mses.append(np.mean((x_np - x_hat_np) ** 2))

    assert mses[0] > mses[1] > mses[2], f"MSE should decrease: {mses}"


def test_deterministic_rotation(ops):
    """Same seed produces same rotation matrix."""
    from tqai.quantizer import PolarQuantizer

    pq1 = PolarQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    pq2 = PolarQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    npt.assert_array_equal(
        ops.to_numpy(pq1._rotation),
        ops.to_numpy(pq2._rotation),
    )


def test_different_seeds(ops):
    """Different seeds produce different rotation matrices."""
    from tqai.quantizer import PolarQuantizer

    pq1 = PolarQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    pq2 = PolarQuantizer(head_dim=64, bits=3, seed=99, ops=ops)
    assert not np.allclose(
        ops.to_numpy(pq1._rotation),
        ops.to_numpy(pq2._rotation),
    )


def test_zero_vector(ops):
    """Should not crash on zero-norm vectors."""
    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    x = ops.zeros((1, 64))
    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)
    x_hat_np = ops.to_numpy(x_hat)
    assert np.all(np.isfinite(x_hat_np))


@pytest.mark.parametrize("d", [64, 128])
def test_batch_dimensions(d, ops):
    """Works with higher-dimensional batch shapes."""
    from tqai.quantizer import PolarQuantizer

    pq = PolarQuantizer(head_dim=d, bits=3, seed=42, ops=ops)
    # Simulate (batch, n_kv_heads, seq_len, head_dim)
    x = ops.randn((2, 8, 16, d), seed=55)
    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)
    assert ops.to_numpy(x_hat).shape == (2, 8, 16, d)
