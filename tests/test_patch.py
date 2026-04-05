from __future__ import annotations

import tqai
from tqai.cache.hf import TurboQuantDynamicCache
from tqai.config import TurboQuantConfig


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


def test_patch_qjl_propagated():
    model = FakeModel()
    cache = tqai.patch(model, bits_k=4, bits_v=2, backend="torch",
                       use_qjl=True, qjl_sketch_size=32)
    assert cache.tq_config.use_qjl is True
    assert cache.tq_config.qjl_sketch_size == 32


def test_patch_forward_compression_propagated():
    model = FakeModel()
    cache = tqai.patch(model, bits_k=4, bits_v=2, backend="torch",
                       compress_hidden=True, bits_hidden=6,
                       compress_ffn=True, bits_ffn=8)
    assert cache.tq_config.compress_hidden is True
    assert cache.tq_config.bits_hidden == 6
    assert cache.tq_config.compress_ffn is True
    assert cache.tq_config.bits_ffn == 8


def test_unpatch_removes_hooks():
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

    model = TinyModel()
    tqai.patch(model, bits_k=4, bits_v=2, backend="torch",
               compress_hidden=True)
    assert hasattr(model, "_tqai_hooks")
    tqai.unpatch(model)
    assert not hasattr(model, "_tqai_hooks")


def test_unpatch_noop_if_not_patched():
    """unpatch on an unpatched model must not raise."""
    model = FakeModel()
    tqai.unpatch(model)  # should not raise


def test_config_has_forward_compression():
    cfg = TurboQuantConfig(compress_hidden=True)
    assert cfg.has_forward_compression is True

    cfg2 = TurboQuantConfig(compress_ffn=True)
    assert cfg2.has_forward_compression is True

    cfg3 = TurboQuantConfig()
    assert cfg3.has_forward_compression is False
