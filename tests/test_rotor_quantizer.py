"""Tests for RotorQuantizer — Clifford rotor block-diagonal KV compression.

Mirrors the structure of test_quantizer.py so that PolarQuantizer and
RotorQuantizer are held to the same quality and API contracts.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rq(head_dim: int, bits: int, ops, seed: int = 42):
    from tqai.quantizer_rotor import RotorQuantizer

    return RotorQuantizer(head_dim=head_dim, bits=bits, seed=seed, ops=ops)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two 2-D arrays."""
    dot = np.sum(a * b, axis=-1)
    return dot / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-10)


# ---------------------------------------------------------------------------
# Shape and dtype contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [64, 128, 65])  # 65 is not divisible by 3
@pytest.mark.parametrize("b", [2, 3, 4])
def test_roundtrip_shape(d, b, ops):
    """quantize → dequantize preserves shape."""
    rq = _make_rq(d, b, ops)
    x = ops.randn((2, 4, d), seed=99)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)
    assert ops.to_numpy(x_hat).shape == (2, 4, d)


@pytest.mark.parametrize("d", [64, 128, 65])
@pytest.mark.parametrize("b", [2, 3, 4])
def test_index_dtype_is_uint8(d, b, ops):
    rq = _make_rq(d, b, ops)
    x = ops.randn((4, d), seed=1)
    indices, _ = rq.quantize(x)
    arr = ops.to_numpy(indices)
    assert arr.dtype == np.uint8, f"Expected uint8 indices, got {arr.dtype}"


@pytest.mark.parametrize("d", [64, 128, 65])
def test_norms_shape(d, ops):
    """Norms output should be (..., 1)."""
    rq = _make_rq(d, 4, ops)
    x = ops.randn((3, 7, d), seed=2)
    _, norms = rq.quantize(x)
    assert ops.to_numpy(norms).shape == (3, 7, 1)


# ---------------------------------------------------------------------------
# Quality contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [64, 128, 65])
@pytest.mark.parametrize("b", [3, 4])
def test_cosine_similarity(d, b, ops):
    """Round-trip should preserve direction well at 3+ bits."""
    rq = _make_rq(d, b, ops)
    x = ops.randn((100, d), seed=7)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)

    x_np = ops.to_numpy(x).reshape(-1, d).astype(np.float64)
    x_hat_np = ops.to_numpy(x_hat).reshape(-1, d).astype(np.float64)

    mean_cos = float(np.mean(_cosine_sim(x_np, x_hat_np)))
    assert mean_cos > 0.95, f"Mean cosine {mean_cos:.4f} too low (d={d}, b={b})"


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b", [2, 3, 4])
def test_mse_reasonable(d, b, ops):
    """MSE should be finite and in a sane range."""
    rq = _make_rq(d, b, ops)
    x = ops.randn((200, d), seed=11)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)

    mse = float(np.mean((ops.to_numpy(x) - ops.to_numpy(x_hat)) ** 2))
    assert 0 < mse < 10.0, f"MSE={mse:.4f} out of range (d={d}, b={b})"


@pytest.mark.parametrize("d", [64, 128])
def test_mse_decreases_with_bits(d, ops):
    """More bits should always give lower MSE."""
    mses = []
    for b in [2, 3, 4]:
        rq = _make_rq(d, b, ops)
        x = ops.randn((200, d), seed=11)
        indices, norms = rq.quantize(x)
        x_hat = rq.dequantize(indices, norms)
        mses.append(float(np.mean((ops.to_numpy(x) - ops.to_numpy(x_hat)) ** 2)))
    assert mses[0] > mses[1] > mses[2], f"MSE not decreasing with bits: {mses}"


@pytest.mark.parametrize("d", [64, 128])
def test_quality_parity_with_polar(d, ops):
    """RotorQuantizer and PolarQuantizer should achieve similar CosSim at 4-bit."""
    from tqai.quantizer import PolarQuantizer
    from tqai.quantizer_rotor import RotorQuantizer

    # Use a different seed for data vs quantizers to avoid adversarial correlation
    x = ops.randn((200, d), seed=77)

    pq = PolarQuantizer(head_dim=d, bits=4, seed=42, ops=ops)
    idx_p, n_p = pq.quantize(x)
    x_hat_p = pq.dequantize(idx_p, n_p)

    rq = RotorQuantizer(head_dim=d, bits=4, seed=42, ops=ops)
    idx_r, n_r = rq.quantize(x)
    x_hat_r = rq.dequantize(idx_r, n_r)

    x_np = ops.to_numpy(x).reshape(-1, d).astype(np.float64)
    cos_p = float(np.mean(_cosine_sim(x_np, ops.to_numpy(x_hat_p).reshape(-1, d).astype(np.float64))))
    cos_r = float(np.mean(_cosine_sim(x_np, ops.to_numpy(x_hat_r).reshape(-1, d).astype(np.float64))))

    # Allow up to 2% difference — both should be near 0.995
    assert abs(cos_p - cos_r) < 0.02, (
        f"Quality gap too large: PolarQuant={cos_p:.4f}, RotorQuant={cos_r:.4f}"
    )


# ---------------------------------------------------------------------------
# Determinism and seed contracts
# ---------------------------------------------------------------------------


def test_deterministic_rotors(ops):
    """Same seed produces identical quaternions."""
    from tqai.quantizer_rotor import RotorQuantizer

    rq1 = RotorQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    rq2 = RotorQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    npt.assert_array_equal(rq1._quats, rq2._quats)


def test_different_seeds(ops):
    """Different seeds produce different quaternions."""
    from tqai.quantizer_rotor import RotorQuantizer

    rq1 = RotorQuantizer(head_dim=64, bits=3, seed=42, ops=ops)
    rq2 = RotorQuantizer(head_dim=64, bits=3, seed=99, ops=ops)
    assert not np.allclose(rq1._quats, rq2._quats)


# ---------------------------------------------------------------------------
# Edge-case inputs
# ---------------------------------------------------------------------------


def test_zero_vector(ops):
    """Should not crash on zero-norm vectors and return finite output."""
    rq = _make_rq(64, 3, ops)
    x = ops.zeros((1, 64))
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)
    assert np.all(np.isfinite(ops.to_numpy(x_hat)))


@pytest.mark.parametrize("d", [65, 127])  # d % 3 != 0
def test_remainder_dims(d, ops):
    """Dimensions not divisible by 3 are handled correctly."""
    from tqai.quantizer_rotor import RotorQuantizer

    rq = RotorQuantizer(head_dim=d, bits=4, seed=42, ops=ops)
    assert rq._remainder == d % 3
    assert rq._n_full == d // 3

    x = ops.randn((50, d), seed=5)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)

    assert ops.to_numpy(x_hat).shape == (50, d)
    assert np.all(np.isfinite(ops.to_numpy(x_hat)))


def test_very_small_dim(ops):
    """head_dim=3 (exactly one group, no remainder) must work."""
    rq = _make_rq(3, 4, ops)
    x = ops.randn((10, 3), seed=0)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)
    assert ops.to_numpy(x_hat).shape == (10, 3)


# ---------------------------------------------------------------------------
# Batch dimension handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [64, 128])
def test_batch_dimensions(d, ops):
    """Works with (batch, n_kv_heads, seq_len, head_dim) shaped input."""
    rq = _make_rq(d, 3, ops)
    x = ops.randn((2, 8, 16, d), seed=55)
    indices, norms = rq.quantize(x)
    x_hat = rq.dequantize(indices, norms)
    assert ops.to_numpy(x_hat).shape == (2, 8, 16, d)


# ---------------------------------------------------------------------------
# Quaternion orthogonality invariant
# ---------------------------------------------------------------------------


def test_rotation_matrices_are_orthogonal(ops):
    """Each 3x3 rotation matrix derived from a quaternion must be orthogonal."""
    from tqai.quantizer_rotor import RotorQuantizer

    rq = RotorQuantizer(head_dim=128, bits=4, seed=7, ops=ops)
    for i, R in enumerate(rq._mats):
        RRT = R @ R.T
        npt.assert_allclose(
            RRT, np.eye(3, dtype=np.float32), atol=1e-5,
            err_msg=f"Rotation matrix {i} is not orthogonal",
        )


# ---------------------------------------------------------------------------
# Metal backend detection
# ---------------------------------------------------------------------------


def test_metal_activated_on_mlx():
    """RotorQuantizer should activate Metal path when MLX + Metal are present."""
    try:
        import mlx.core as mx
        from tqai.backend import get_backend
        from tqai.kernels import metal_available
    except ImportError:
        pytest.skip("MLX not available")

    if not metal_available():
        pytest.skip("Metal not available")

    ops = get_backend("mlx")
    from tqai.quantizer_rotor import RotorQuantizer
    rq = RotorQuantizer(head_dim=128, bits=4, seed=42, ops=ops)
    assert rq._use_metal, "Metal path should be active on MLX+Metal"
    assert hasattr(rq, "_block_mats_mlx"), "_block_mats_mlx must be precomputed"
    assert hasattr(rq, "_centroids_mlx"), "_centroids_mlx must be precomputed"


def test_no_metal_on_torch():
    """RotorQuantizer should not activate Metal on the PyTorch backend."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not available")

    from tqai.backend import get_backend
    from tqai.quantizer_rotor import RotorQuantizer

    ops = get_backend("torch")
    rq = RotorQuantizer(head_dim=128, bits=4, seed=42, ops=ops)
    assert not rq._use_metal, "Metal path must not be active on PyTorch backend"
