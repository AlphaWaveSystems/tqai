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
