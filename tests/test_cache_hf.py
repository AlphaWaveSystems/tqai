from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import torch

from tqai.cache.hf import TurboQuantDynamicCache
from tqai.config import TurboQuantConfig


@pytest.fixture
def config():
    return TurboQuantConfig(bits_k=4, bits_v=2, backend="torch")


@pytest.fixture
def cache(config):
    return TurboQuantDynamicCache(config)


def _rand_kv(batch=1, heads=4, seq=8, dim=64):
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    return k, v


def test_update_returns_correct_shapes(cache):
    k, v = _rand_kv(seq=8, dim=64)
    all_k, all_v = cache.update(k, v, layer_idx=0)
    assert all_k.shape == (1, 4, 8, 64)
    assert all_v.shape == (1, 4, 8, 64)


def test_sequence_grows(cache):
    k1, v1 = _rand_kv(seq=4, dim=64)
    k2, v2 = _rand_kv(seq=3, dim=64)

    cache.update(k1, v1, layer_idx=0)
    all_k, all_v = cache.update(k2, v2, layer_idx=0)

    assert all_k.shape[2] == 7
    assert all_v.shape[2] == 7
    assert cache.get_seq_length(0) == 7


def test_multiple_layers(cache):
    k0, v0 = _rand_kv(seq=5, dim=64)
    k1, v1 = _rand_kv(seq=3, dim=64)

    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)

    assert cache.get_seq_length(0) == 5
    assert cache.get_seq_length(1) == 3


def test_reconstruction_fidelity(cache):
    """Reconstructed values should be close to originals."""
    k, v = _rand_kv(seq=16, dim=128)
    all_k, all_v = cache.update(k, v, layer_idx=0)

    k_np = k.numpy()
    all_k_np = all_k.detach().numpy()

    # Per-vector cosine similarity
    cos_sims = []
    for b in range(k_np.shape[0]):
        for h in range(k_np.shape[1]):
            for s in range(k_np.shape[2]):
                orig = k_np[b, h, s]
                recon = all_k_np[b, h, s]
                cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
                cos_sims.append(cos)

    mean_cos = np.mean(cos_sims)
    assert mean_cos > 0.90, f"Mean cosine similarity {mean_cos:.4f} too low"


def test_sink_tokens():
    config = TurboQuantConfig(bits_k=4, bits_v=2, sink_tokens=4, backend="torch")
    cache = TurboQuantDynamicCache(config)

    k, v = _rand_kv(seq=8, dim=64)
    all_k, all_v = cache.update(k, v, layer_idx=0)

    # First 4 tokens should be exact (sink)
    npt.assert_array_equal(
        all_k[:, :, :4, :].numpy(),
        k[:, :, :4, :].numpy(),
    )


def test_seq_length_tracking(cache):
    assert cache.get_seq_length(0) == 0
    cache.update(*_rand_kv(seq=3, dim=64), layer_idx=0)
    assert cache.get_seq_length(0) == 3
    cache.update(*_rand_kv(seq=5, dim=64), layer_idx=0)
    assert cache.get_seq_length(0) == 8


def test_qjl_cache_reconstruction_fidelity():
    """Cache with use_qjl=True should still reconstruct with reasonable fidelity."""
    config = TurboQuantConfig(bits_k=4, bits_v=2, backend="torch",
                               use_qjl=True, qjl_sketch_size=64)
    cache = TurboQuantDynamicCache(config)
    k, v = _rand_kv(seq=16, dim=128)
    all_k, all_v = cache.update(k, v, layer_idx=0)

    k_np = k.numpy()
    all_k_np = all_k.detach().numpy()
    cos_sims = []
    for b in range(k_np.shape[0]):
        for h in range(k_np.shape[1]):
            for s in range(k_np.shape[2]):
                orig = k_np[b, h, s]
                recon = all_k_np[b, h, s]
                cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
                cos_sims.append(cos)

    assert np.mean(cos_sims) > 0.85, f"QJL cache cosine similarity too low: {np.mean(cos_sims):.4f}"


def test_asymmetric_bits():
    """k3/v2 config should work and reconstruct with lower fidelity than k4/v2."""
    config_low = TurboQuantConfig(bits_k=3, bits_v=2, backend="torch")
    config_high = TurboQuantConfig(bits_k=4, bits_v=4, backend="torch")
    k, v = _rand_kv(seq=32, dim=128)

    cache_low = TurboQuantDynamicCache(config_low)
    cache_high = TurboQuantDynamicCache(config_high)

    all_k_low, _ = cache_low.update(k, v, layer_idx=0)
    all_k_high, _ = cache_high.update(k, v, layer_idx=0)

    def mean_cos(orig, recon):
        orig_np, recon_np = orig.numpy(), recon.detach().numpy()
        sims = []
        for b in range(orig_np.shape[0]):
            for h in range(orig_np.shape[1]):
                for s in range(orig_np.shape[2]):
                    o, r = orig_np[b, h, s], recon_np[b, h, s]
                    sims.append(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-10))
        return float(np.mean(sims))

    cos_low = mean_cos(k, all_k_low)
    cos_high = mean_cos(k, all_k_high)
    assert cos_low < cos_high, "Higher bits should give better cosine similarity"
