from __future__ import annotations

import tqai
from tqai.cache.hf import TurboQuantDynamicCache


class FakeModel:
    """Minimal fake model for testing patch routing."""
    pass


def test_patch_returns_hf_cache():
    model = FakeModel()
    cache = tqai.patch(model, bits_k=4, bits_v=2, backend="torch")
    assert isinstance(cache, TurboQuantDynamicCache)


def test_patch_config_propagated():
    model = FakeModel()
    cache = tqai.patch(model, bits_k=3, bits_v=3, sink_tokens=8, backend="torch")
    assert cache.tq_config.bits_k == 3
    assert cache.tq_config.bits_v == 3
    assert cache.tq_config.sink_tokens == 8
