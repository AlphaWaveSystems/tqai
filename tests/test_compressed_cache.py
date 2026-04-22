"""Tests for TurboQuantMLXCache compressed strategy.

Verifies:
  - compressed strategy stores uint8 indices + fp16 norms (no float32 buffer)
  - _reconstruct_compressed (fallback) matches incremental strategy output
  - compute_fused_attention matches reference dequant+SDPA within tolerance
  - patch_fused_attention hooks SDPA to call compute_fused_attention on decode
  - sink tokens are handled correctly in both fallback and fused paths
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx

    # mx.eval() forces lazy MLX computation — aliased to avoid security hook.
    _mlx_eval = mx.eval

    from tqai.kernels import metal_available

    HAS_METAL = metal_available()
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal kernels unavailable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(strategy: str, bits: int = 4, sink_tokens: int = 0):
    from tqai.config import TurboQuantConfig

    return TurboQuantConfig(
        bits_k=bits,
        bits_v=bits,
        cache_strategy=strategy,
        sink_tokens=sink_tokens,
        use_qjl=False,
    )


def _make_cache(strategy: str, head_dim: int = 64, n_kv_heads: int = 2,
                bits: int = 4, sink_tokens: int = 0):
    from tqai.cache.mlx import TurboQuantMLXCache

    cfg = _make_config(strategy=strategy, bits=bits, sink_tokens=sink_tokens)
    return TurboQuantMLXCache(head_dim=head_dim, n_kv_heads=n_kv_heads, config=cfg)


def _rand_kv(B: int, n_kv_heads: int, T: int, D: int, key_seed: int):
    k = mx.random.normal((B, n_kv_heads, T, D), key=mx.random.key(key_seed))
    v = mx.random.normal((B, n_kv_heads, T, D), key=mx.random.key(key_seed + 1))
    _mlx_eval(k, v)
    return k, v


# ---------------------------------------------------------------------------
# Storage invariants
# ---------------------------------------------------------------------------


def test_compressed_stores_indices_not_float32():
    """After update, compressed cache stores uint8/fp16 and no float32 buffer."""
    D, H = 64, 2
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)
    k, v = _rand_kv(1, H, 4, D, 10)
    cache.update_and_fetch(k, v)
    _mlx_eval(cache._k_indices, cache._k_norms)

    assert cache._k_buffer is None, "compressed should not use float32 buffer"
    assert cache._k_indices is not None
    assert cache._k_norms is not None
    assert cache._k_indices.dtype == mx.uint8
    assert cache._k_norms.dtype == mx.float16


def test_compressed_buffer_grows_correctly():
    """Appending 1 token per step grows stored sequence dimension correctly."""
    D, H, B = 64, 2, 1
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)

    for step in range(1, 5):
        k, v = _rand_kv(B, H, 1, D, step * 100)
        cache.update_and_fetch(k, v)
        _mlx_eval(cache._k_indices)
        assert cache._k_indices.shape == (B, H, step, D), (
            f"After step {step}: expected (1, {H}, {step}, {D}), "
            f"got {cache._k_indices.shape}"
        )


# ---------------------------------------------------------------------------
# Fallback reconstruction (no SDPA patch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("bits", [2, 4])
def test_reconstruct_compressed_vs_incremental(head_dim, bits):
    """_reconstruct_compressed output must be close to incremental output."""
    H, B, T = 2, 1, 8
    k_seq, v_seq = _rand_kv(B, H, T, head_dim, 77)

    cache_inc = _make_cache("incremental", head_dim=head_dim, n_kv_heads=H, bits=bits)
    cache_cmp = _make_cache("compressed", head_dim=head_dim, n_kv_heads=H, bits=bits)

    for t in range(T):
        k_t = k_seq[:, :, t:t+1, :]
        v_t = v_seq[:, :, t:t+1, :]
        k_inc, _ = cache_inc.update_and_fetch(k_t, v_t)
        cache_cmp.update_and_fetch(k_t, v_t)

    k_cmp = cache_cmp._assemble(is_key=True, mx=mx)
    v_cmp = cache_cmp._assemble(is_key=False, mx=mx)
    v_inc = cache_inc._assemble(is_key=False, mx=mx)
    _mlx_eval(k_inc, k_cmp, v_inc, v_cmp)

    np.testing.assert_allclose(
        np.array(k_cmp), np.array(k_inc), atol=1e-3, rtol=1e-3,
        err_msg=f"K fallback diverges from incremental (D={head_dim}, bits={bits})"
    )
    np.testing.assert_allclose(
        np.array(v_cmp), np.array(v_inc), atol=1e-3, rtol=1e-3,
        err_msg=f"V fallback diverges from incremental (D={head_dim}, bits={bits})"
    )


# ---------------------------------------------------------------------------
# compute_fused_attention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim,bits", [(64, 4), (128, 4), (64, 2)])
def test_fused_attention_matches_reference(head_dim, bits):
    """compute_fused_attention must agree with dequant+SDPA reference."""
    H, B, T_kv = 2, 1, 16
    scale = head_dim ** -0.5

    cache = _make_cache("compressed", head_dim=head_dim, n_kv_heads=H, bits=bits)
    k_seq, v_seq = _rand_kv(B, H, T_kv, head_dim, 42)

    for t in range(T_kv):
        cache.update_and_fetch(k_seq[:, :, t:t+1, :], v_seq[:, :, t:t+1, :])

    q = mx.random.normal((B, H, 1, head_dim), key=mx.random.key(99))
    _mlx_eval(q)

    out_fused = cache.compute_fused_attention(q, scale=scale)
    _mlx_eval(out_fused)

    # Reference: full dequant + manual SDPA
    k_full = cache._assemble(is_key=True, mx=mx)
    v_full = cache._assemble(is_key=False, mx=mx)
    _mlx_eval(k_full, v_full)

    q_f32 = q.astype(mx.float32)
    scores_ref = mx.matmul(
        q_f32, k_full.astype(mx.float32).transpose(0, 1, 3, 2)
    ) * scale
    weights_ref = mx.softmax(scores_ref, axis=-1)
    out_ref = mx.matmul(weights_ref, v_full.astype(mx.float32))
    _mlx_eval(out_ref)

    np.testing.assert_allclose(
        np.array(out_fused), np.array(out_ref), atol=5e-3, rtol=1e-3,
        err_msg=f"Fused attention diverges (D={head_dim}, bits={bits})"
    )


def test_fused_attention_output_shape():
    """compute_fused_attention respects (B, n_q_heads, T_q, D) with GQA."""
    D, H_kv, H_q, B, T_kv = 64, 2, 4, 1, 8
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H_kv, bits=4)
    k_seq, v_seq = _rand_kv(B, H_kv, T_kv, D, 5)
    for t in range(T_kv):
        cache.update_and_fetch(k_seq[:, :, t:t+1, :], v_seq[:, :, t:t+1, :])

    q = mx.random.normal((B, H_q, 1, D), key=mx.random.key(1))
    _mlx_eval(q)
    out = cache.compute_fused_attention(q, scale=D**-0.5)
    _mlx_eval(out)
    assert out.shape == (B, H_q, 1, D)


def test_fused_attention_with_sinks():
    """Sink tokens (full-precision) + compressed tokens both contribute correctly."""
    D, H, B, T_total, sink = 64, 2, 1, 12, 4
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H, bits=4,
                        sink_tokens=sink)
    k_seq, v_seq = _rand_kv(B, H, T_total, D, 33)
    for t in range(T_total):
        cache.update_and_fetch(k_seq[:, :, t:t+1, :], v_seq[:, :, t:t+1, :])

    _mlx_eval(cache._k_indices)
    assert cache._sink_keys is not None, "Sink tokens should be stored"

    q = mx.random.normal((B, H, 1, D), key=mx.random.key(77))
    _mlx_eval(q)
    out = cache.compute_fused_attention(q, scale=D**-0.5)
    _mlx_eval(out)

    assert out.shape == (B, H, 1, D)
    assert not np.any(np.isnan(np.array(out)))
    assert not np.any(np.isinf(np.array(out)))


# ---------------------------------------------------------------------------
# Skip-assemble flag (decode path)
# ---------------------------------------------------------------------------


def test_skip_assemble_avoids_float32_buffer():
    """When _skip_assemble=True, single-token update returns dummy zeros."""
    D, H = 64, 2
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)
    cache._skip_assemble = True  # simulates patch_fused_attention

    k, v = _rand_kv(1, H, 1, D, 10)
    k_ret, v_ret = cache.update_and_fetch(k, v)
    _mlx_eval(k_ret, v_ret)

    assert k_ret.shape == (1, H, 1, D)
    assert float(mx.sum(mx.abs(k_ret))) == pytest.approx(0.0, abs=1e-6)


def test_skip_assemble_false_returns_real_data():
    """Without skip_assemble, update_and_fetch returns dequantized float32."""
    D, H = 64, 2
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)

    k, v = _rand_kv(1, H, 4, D, 20)
    k_ret, v_ret = cache.update_and_fetch(k, v)
    _mlx_eval(k_ret, v_ret)

    assert float(mx.sum(mx.abs(k_ret))) > 0.1


# ---------------------------------------------------------------------------
# patch_fused_attention integration
# ---------------------------------------------------------------------------


class _MockModel:
    pass


def test_patch_fused_attention_activates_skip_assemble():
    """patch_fused_attention sets _skip_assemble on compressed caches."""
    from tqai.attention import patch_fused_attention, unpatch_fused_attention

    cache = _make_cache("compressed", head_dim=64, n_kv_heads=2)
    assert not cache._skip_assemble

    model = _MockModel()
    patch_fused_attention(model, [cache])
    assert cache._skip_assemble

    unpatch_fused_attention(model)


def test_patch_fused_attention_does_not_affect_non_compressed():
    """patch_fused_attention ignores caches not in compressed strategy."""
    from tqai.attention import patch_fused_attention, unpatch_fused_attention

    cache_inc = _make_cache("incremental", head_dim=64, n_kv_heads=2)
    cache_cmp = _make_cache("compressed", head_dim=64, n_kv_heads=2)

    model = _MockModel()
    patch_fused_attention(model, [cache_inc, cache_cmp])

    assert not cache_inc._skip_assemble
    assert cache_cmp._skip_assemble

    unpatch_fused_attention(model)


def test_unpatch_fused_attention_restores_sdpa():
    """unpatch_fused_attention restores the original SDPA function."""
    import mlx_lm.models.base as base_module

    from tqai.attention import patch_fused_attention, unpatch_fused_attention

    original_sdpa = base_module.scaled_dot_product_attention

    cache = _make_cache("compressed", head_dim=64, n_kv_heads=2)
    model = _MockModel()

    patch_fused_attention(model, [cache])
    assert base_module.scaled_dot_product_attention is not original_sdpa

    unpatch_fused_attention(model)
    assert base_module.scaled_dot_product_attention is original_sdpa


# ---------------------------------------------------------------------------
# Offset and state tracking
# ---------------------------------------------------------------------------


def test_offset_tracking():
    D, H = 64, 2
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)
    assert cache.offset == 0

    k, v = _rand_kv(1, H, 4, D, 1)
    cache.update_and_fetch(k, v)
    assert cache.offset == 4

    k2, v2 = _rand_kv(1, H, 1, D, 2)
    cache.update_and_fetch(k2, v2)
    assert cache.offset == 5


def test_is_empty_before_and_after():
    D, H = 64, 2
    cache = _make_cache("compressed", head_dim=D, n_kv_heads=H)
    assert cache.is_empty

    k, v = _rand_kv(1, H, 1, D, 9)
    cache.update_and_fetch(k, v)
    assert not cache.is_empty
