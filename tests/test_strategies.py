"""Tests for strategy modules."""

from __future__ import annotations

import pytest
import torch

from tqai.backend import get_backend
from tqai.pipeline.base import ScoredEntry
from tqai.quantizer import PolarQuantizer


@pytest.fixture
def ops():
    return get_backend("torch")


@pytest.fixture
def quantizer(ops):
    return PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)


class TestTieredStrategy:
    def _make_strategy(self, **kwargs):
        from tqai.strategies.tiered import TieredStrategy
        return TieredStrategy(**kwargs)

    def test_compress_with_scored_entry(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        entry = [ScoredEntry(data=x, score=0.8, tier=2, metadata={})]
        compressed, state = strategy.compress(entry, quantizer)
        assert compressed[0] == "tiered"
        assert compressed[1] is True  # high tier
        assert state["last_use_high"] is True

    def test_compress_low_score_uses_low_tier(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        entry = [ScoredEntry(data=x, score=0.1, tier=0, metadata={})]
        compressed, state = strategy.compress(entry, quantizer)
        assert compressed[1] is False  # low tier
        assert state["last_use_high"] is False

    def test_compress_raw_tensor(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = strategy.compress(x, quantizer)
        assert compressed[0] == "tiered"
        assert compressed[1] is True  # raw tensor defaults to high

    def test_roundtrip(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        entry = [ScoredEntry(data=x, score=0.5, tier=2, metadata={})]
        compressed, state = strategy.compress(entry, quantizer)
        recon = strategy.decompress(compressed, quantizer, state)
        assert recon.shape == x.shape

    def test_tier_counts_accumulate(self, quantizer):
        strategy = self._make_strategy(high_tier_threshold=0.5)
        x = torch.randn(1, 4, 1, 64)
        state = {}

        # High score
        entry_high = [ScoredEntry(data=x, score=0.8, tier=3, metadata={})]
        _, state = strategy.compress(entry_high, quantizer, state)

        # Low score
        entry_low = [ScoredEntry(data=x, score=0.2, tier=0, metadata={})]
        _, state = strategy.compress(entry_low, quantizer, state)

        assert state["tier_counts"] == [1, 1]

    def test_registration(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import get_strategy
        strategy = get_strategy("tiered")
        assert strategy.name == "tiered"
