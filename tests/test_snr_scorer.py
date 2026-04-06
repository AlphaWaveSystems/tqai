"""Tests for SNR scorer and Delta strategy."""

from __future__ import annotations

import math

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


class TestSNRScorer:
    def _make_scorer(self, **kwargs):
        from tqai.scorers.snr import SNRScorer
        return SNRScorer(**kwargs)

    def test_early_step_high_score(self):
        scorer = self._make_scorer(total_steps=50)
        result = scorer.score(torch.randn(1, 4, 8, 64), layer_idx=0, step=0)
        assert result[0].score > 0.8

    def test_late_step_low_score(self):
        scorer = self._make_scorer(total_steps=50)
        result = scorer.score(torch.randn(1, 4, 8, 64), layer_idx=0, step=49)
        assert result[0].score < 0.2

    def test_snr_from_context(self):
        scorer = self._make_scorer()
        result = scorer.score(
            torch.randn(1, 4, 8, 64), layer_idx=0, step=0,
            context={"snr": 10.0}
        )
        expected = 1.0 / (1.0 + 10.0)
        assert abs(result[0].score - expected) < 1e-6

    def test_cosine_schedule(self):
        scorer = self._make_scorer(schedule="cosine", total_steps=100)
        mid = scorer.score(torch.randn(1, 1, 1, 64), layer_idx=0, step=50)
        # Cosine at progress=0.5 → 0.5*(1+cos(π*0.5)) = 0.5
        assert abs(mid[0].score - 0.5) < 0.01

    def test_linear_schedule(self):
        scorer = self._make_scorer(schedule="linear", total_steps=100)
        mid = scorer.score(torch.randn(1, 1, 1, 64), layer_idx=0, step=50)
        assert abs(mid[0].score - 0.4949) < 0.02  # ~1 - 50/99

    def test_no_step_defaults_mid(self):
        scorer = self._make_scorer()
        result = scorer.score(torch.randn(1, 1, 1, 64), layer_idx=0)
        assert result[0].score == 0.5

    def test_registration(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import get_scorer
        scorer = get_scorer("snr", schedule="cosine")
        assert scorer.name == "snr"


class TestDeltaStrategy:
    def _make_strategy(self, **kwargs):
        from tqai.strategies.delta import DeltaStrategy
        return DeltaStrategy(**kwargs)

    def test_first_call_uses_full(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = strategy.compress(x, quantizer)
        assert compressed[0] == "delta"
        assert compressed[1] is False  # first call is always full

    def test_similar_input_uses_delta(self, quantizer):
        strategy = self._make_strategy(threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = strategy.compress(x, quantizer)

        # Second call with very similar input
        x_similar = x + torch.randn_like(x) * 0.01
        compressed, state = strategy.compress(x_similar, quantizer, state)
        assert compressed[1] is True  # delta used

    def test_different_input_uses_full(self, quantizer):
        strategy = self._make_strategy(threshold=0.01)
        x = torch.randn(1, 4, 8, 64)
        _, state = strategy.compress(x, quantizer)

        # Completely different input
        x_diff = torch.randn(1, 4, 8, 64) * 100
        compressed, state = strategy.compress(x_diff, quantizer, state)
        assert compressed[1] is False  # full used

    def test_roundtrip_full(self, quantizer):
        strategy = self._make_strategy()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = strategy.compress(x, quantizer)
        recon = strategy.decompress(compressed, quantizer, state)
        assert recon.shape == x.shape

    def test_roundtrip_delta(self, quantizer):
        strategy = self._make_strategy(threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = strategy.compress(x, quantizer)

        x_similar = x + torch.randn_like(x) * 0.01
        compressed, state = strategy.compress(x_similar, quantizer, state)
        assert compressed[1] is True
        recon = strategy.decompress(compressed, quantizer, state)
        assert recon.shape == x_similar.shape

    def test_delta_stats_tracked(self, quantizer):
        strategy = self._make_strategy(threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = strategy.compress(x, quantizer)

        x_similar = x + torch.randn_like(x) * 0.001
        _, state = strategy.compress(x_similar, quantizer, state)

        assert state["delta_stats"]["full_used"] == 1
        assert state["delta_stats"]["delta_used"] == 1

    def test_registration(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import get_strategy
        strategy = get_strategy("delta", threshold=0.2)
        assert strategy.name == "delta"
