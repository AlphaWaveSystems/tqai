from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture
def config():
    from tqai.config import TurboQuantConfig
    return TurboQuantConfig(bits_k=4, bits_v=2, backend="mlx")


@pytest.fixture
def cache(config):
    from tqai.cache.mlx import TurboQuantMLXCache
    return TurboQuantMLXCache(head_dim=64, n_kv_heads=4, config=config)


def _rand_kv(seq=8, dim=64, heads=4):
    k = mx.random.normal((1, heads, seq, dim))
    v = mx.random.normal((1, heads, seq, dim))
    return k, v


def test_update_returns_correct_shapes(cache):
    k, v = _rand_kv(seq=8, dim=64)
    all_k, all_v = cache.update_and_fetch(k, v)
    assert all_k.shape == (1, 4, 8, 64)
    assert all_v.shape == (1, 4, 8, 64)


def test_sequence_grows(cache):
    k1, v1 = _rand_kv(seq=4, dim=64)
    k2, v2 = _rand_kv(seq=3, dim=64)

    cache.update_and_fetch(k1, v1)
    all_k, all_v = cache.update_and_fetch(k2, v2)

    assert all_k.shape[2] == 7
    assert all_v.shape[2] == 7
    assert cache.offset == 7


def test_reconstruction_fidelity(cache):
    k, v = _rand_kv(seq=16, dim=64)
    all_k, all_v = cache.update_and_fetch(k, v)

    k_np = np.array(k)
    all_k_np = np.array(all_k)

    cos_sims = []
    for h in range(k_np.shape[1]):
        for s in range(k_np.shape[2]):
            orig = k_np[0, h, s]
            recon = all_k_np[0, h, s]
            cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
            cos_sims.append(cos)

    mean_cos = np.mean(cos_sims)
    assert mean_cos > 0.90, f"Mean cosine similarity {mean_cos:.4f} too low"


def test_sink_tokens():
    from tqai.cache.mlx import TurboQuantMLXCache
    from tqai.config import TurboQuantConfig

    config = TurboQuantConfig(bits_k=4, bits_v=2, sink_tokens=4, backend="mlx")
    cache = TurboQuantMLXCache(head_dim=64, n_kv_heads=4, config=config)

    k, v = _rand_kv(seq=8, dim=64)
    all_k, all_v = cache.update_and_fetch(k, v)

    npt.assert_array_equal(
        np.array(all_k[:, :, :4, :]),
        np.array(k[:, :, :4, :]),
    )


def test_state_property(cache):
    assert cache.is_empty
    k, v = _rand_kv(seq=5, dim=64)
    cache.update_and_fetch(k, v)
    assert not cache.is_empty
    state_k, state_v = cache.state
    assert state_k.shape[2] == 5


def test_qjl_cache_reconstruction_fidelity():
    """MLX cache with use_qjl=True should still reconstruct with reasonable fidelity."""
    from tqai.cache.mlx import TurboQuantMLXCache
    from tqai.config import TurboQuantConfig

    config = TurboQuantConfig(bits_k=4, bits_v=2, backend="mlx",
                               use_qjl=True, qjl_sketch_size=64)
    cache = TurboQuantMLXCache(head_dim=64, n_kv_heads=4, config=config)
    k, v = _rand_kv(seq=16, dim=64)
    all_k, _ = cache.update_and_fetch(k, v)

    k_np = np.array(k)
    all_k_np = np.array(all_k)
    cos_sims = []
    for h in range(k_np.shape[1]):
        for s in range(k_np.shape[2]):
            orig = k_np[0, h, s]
            recon = all_k_np[0, h, s]
            cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
            cos_sims.append(cos)

    mean_cos = np.mean(cos_sims)
    assert mean_cos > 0.85, f"QJL MLX cache cosine similarity too low: {mean_cos:.4f}"
