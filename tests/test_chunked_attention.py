"""Tests for chunked scaled dot-product attention."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _reference_sdpa(q, k, v, scale, mask=None):
    """Full attention via mx.fast.scaled_dot_product_attention."""
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask
    )


@pytest.fixture
def rng():
    return mx.random.key(42)


@pytest.mark.parametrize("seq_len", [128, 512, 1024])
@pytest.mark.parametrize("chunk_size", [64, 256])
def test_matches_full_attention(seq_len, chunk_size, rng):
    """Chunked output matches full SDPA output."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, D = 1, 4, 64
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, H, seq_len, D), key=k1)
    k = mx.random.normal((B, H, seq_len, D), key=k2)
    v = mx.random.normal((B, H, seq_len, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    ref = _reference_sdpa(q, k, v, scale)
    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, chunk_size=chunk_size,
    )
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(chunked), np.array(ref), atol=5e-4, rtol=2e-3,
    )


def test_causal_mask(rng):
    """Chunked attention with causal masking matches full causal attention."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, T, D = 1, 2, 256, 64
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, H, T, D), key=k1)
    k = mx.random.normal((B, H, T, D), key=k2)
    v = mx.random.normal((B, H, T, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    ref = _reference_sdpa(q, k, v, scale, mask="causal")
    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal", chunk_size=64,
    )
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(chunked), np.array(ref), atol=5e-4, rtol=2e-3,
    )


def test_additive_mask(rng):
    """Chunked attention with additive mask."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, T, D = 1, 2, 128, 32
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, H, T, D), key=k1)
    k = mx.random.normal((B, H, T, D), key=k2)
    v = mx.random.normal((B, H, T, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    # Random additive mask
    rand_vals = mx.random.uniform(shape=(1, 1, T, T), key=mx.random.key(99))
    mask = mx.where(rand_vals > 0.3, mx.array(0.0), mx.array(-1e9))

    ref = _reference_sdpa(q, k, v, scale, mask=mask)
    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, chunk_size=32,
    )
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(chunked), np.array(ref), atol=5e-4, rtol=2e-3,
    )


def test_short_sequence_passthrough(rng):
    """Sequences shorter than chunk_size use native SDPA directly."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, T, D = 1, 4, 64, 32
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, H, T, D), key=k1)
    k = mx.random.normal((B, H, T, D), key=k2)
    v = mx.random.normal((B, H, T, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    ref = _reference_sdpa(q, k, v, scale)
    # chunk_size > T → should pass through
    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, chunk_size=256,
    )
    mx.synchronize()

    np.testing.assert_array_equal(np.array(chunked), np.array(ref))


def test_gqa_support(rng):
    """Grouped query attention (fewer KV heads than Q heads)."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, T, D = 1, 256, 64
    n_q_heads, n_kv_heads = 8, 2
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, n_q_heads, T, D), key=k1)
    k = mx.random.normal((B, n_kv_heads, T, D), key=k2)
    v = mx.random.normal((B, n_kv_heads, T, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    # Reference: repeat KV heads then full SDPA
    k_rep = mx.repeat(k, n_q_heads // n_kv_heads, axis=1)
    v_rep = mx.repeat(v, n_q_heads // n_kv_heads, axis=1)
    ref = _reference_sdpa(q, k_rep, v_rep, scale)

    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, chunk_size=64,
    )
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(chunked), np.array(ref), atol=5e-4, rtol=2e-3,
    )


def test_decode_step_causal(rng):
    """Regression test: T_q=1 (decode mode) with causal mask must produce
    the same output as full SDPA. This catches the case where a single
    query is at position T_kv-1 and should attend to all KV positions.
    """
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, D = 1, 4, 64
    T_kv = 256  # large enough to span multiple chunks at chunk_size=64
    chunk_size = 64

    q = mx.random.normal((B, H, 1, D), key=rng)  # T_q == 1 (decode)
    k = mx.random.normal((B, H, T_kv, D), key=rng)
    v = mx.random.normal((B, H, T_kv, D), key=rng)
    scale = 1.0 / (D ** 0.5)

    # Reference: query at position T_kv-1 should see all KV positions
    out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

    # Chunked
    out_chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal", chunk_size=chunk_size,
    )

    diff = mx.max(mx.abs(out_ref - out_chunked))
    assert float(diff) < 1e-3, f"Decode chunked SDPA differs from full SDPA: max_diff={float(diff)}"


def test_decode_step_multiple_positions(rng):
    """Decode mode with T_q > 1 (e.g., partial prefill continuation)."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, D = 1, 4, 64
    T_q = 8
    T_kv = 256
    chunk_size = 64

    q = mx.random.normal((B, H, T_q, D), key=rng)
    k = mx.random.normal((B, H, T_kv, D), key=rng)
    v = mx.random.normal((B, H, T_kv, D), key=rng)
    scale = 1.0 / (D ** 0.5)

    out_ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    out_chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, mask="causal", chunk_size=chunk_size,
    )

    diff = mx.max(mx.abs(out_ref - out_chunked))
    assert float(diff) < 1e-3, f"Multi-query decode differs from full SDPA: max_diff={float(diff)}"


@pytest.mark.parametrize("chunk_size", [32, 128, 512])
def test_various_chunk_sizes(chunk_size, rng):
    """Different chunk sizes all produce correct results."""
    from tqai.attention import chunked_scaled_dot_product_attention

    B, H, T, D = 1, 2, 256, 32
    k1, k2, k3 = mx.random.split(rng, 3)
    q = mx.random.normal((B, H, T, D), key=k1)
    k = mx.random.normal((B, H, T, D), key=k2)
    v = mx.random.normal((B, H, T, D), key=k3)
    scale = 1.0 / (D ** 0.5)

    ref = _reference_sdpa(q, k, v, scale)
    chunked = chunked_scaled_dot_product_attention(
        q, k, v, scale=scale, chunk_size=chunk_size,
    )
    mx.synchronize()

    np.testing.assert_allclose(
        np.array(chunked), np.array(ref), atol=5e-4, rtol=2e-3,
    )
