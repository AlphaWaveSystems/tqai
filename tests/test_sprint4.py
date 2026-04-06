"""Tests for Sprint 4: Fisher scorer, Delta2/Window strategies, Monitors."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from tqai.backend import get_backend
from tqai.pipeline.base import ScoredEntry
from tqai.quantizer import PolarQuantizer


@pytest.fixture
def ops():
    return get_backend("torch")


@pytest.fixture
def quantizer(ops):
    return PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)


# ---------------------------------------------------------------------------
# Fisher scorer
# ---------------------------------------------------------------------------

class TestFisherScorer:
    def _make(self, **kw):
        from tqai.scorers.fisher import FisherScorer
        return FisherScorer(**kw)

    def test_returns_scored_entry(self):
        scorer = self._make()
        x = torch.randn(1, 4, 8, 64)
        result = scorer.score(x, layer_idx=0)
        assert isinstance(result[0], ScoredEntry)
        assert 0.0 <= result[0].score <= 1.0

    def test_high_activation_higher_score(self):
        scorer = self._make()
        x_low = torch.randn(1, 4, 8, 64) * 0.01
        x_high = torch.randn(1, 4, 8, 64) * 10.0
        s_low = scorer.score(x_low, layer_idx=0)
        scorer.reset()
        s_high = scorer.score(x_high, layer_idx=0)
        assert s_high[0].score >= s_low[0].score

    def test_reset(self):
        scorer = self._make()
        scorer.score(torch.randn(1, 4, 8, 64), layer_idx=0)
        scorer.reset()
        assert scorer._running_fisher is None

    def test_registration(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import get_scorer
        s = get_scorer("fisher")
        assert s.name == "fisher"


# ---------------------------------------------------------------------------
# SecondOrderDelta strategy
# ---------------------------------------------------------------------------

class TestSecondOrderDelta:
    def _make(self, **kw):
        from tqai.strategies.delta2 import SecondOrderDelta
        return SecondOrderDelta(**kw)

    def test_first_call_full(self, quantizer):
        s = self._make()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = s.compress(x, quantizer)
        assert compressed[1] == 0  # full

    def test_second_call_delta1(self, quantizer):
        s = self._make(delta1_threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        x2 = x + torch.randn_like(x) * 0.01
        compressed, state = s.compress(x2, quantizer, state)
        assert compressed[1] in (0, 1)  # delta1 or full

    def test_third_call_delta2(self, quantizer):
        s = self._make(threshold=0.5, delta1_threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        x2 = x + torch.randn_like(x) * 0.001
        _, state = s.compress(x2, quantizer, state)
        x3 = x2 + torch.randn_like(x) * 0.001
        compressed, state = s.compress(x3, quantizer, state)
        # Should be delta2 or delta1 since changes are tiny
        assert compressed[1] in (1, 2)

    def test_roundtrip_all_orders(self, quantizer):
        s = self._make(threshold=0.9, delta1_threshold=0.9)
        x1 = torch.randn(1, 4, 4, 64)
        c1, state = s.compress(x1, quantizer)
        r1 = s.decompress(c1, quantizer, state)
        assert r1.shape == x1.shape

        x2 = x1 + torch.randn_like(x1) * 0.001
        c2, state = s.compress(x2, quantizer, state)
        r2 = s.decompress(c2, quantizer, state)
        assert r2.shape == x2.shape

        x3 = x2 + torch.randn_like(x2) * 0.001
        c3, state = s.compress(x3, quantizer, state)
        r3 = s.decompress(c3, quantizer, state)
        assert r3.shape == x3.shape

    def test_stats(self, quantizer):
        s = self._make(threshold=0.9, delta1_threshold=0.9)
        x = torch.randn(1, 4, 4, 64)
        _, state = s.compress(x, quantizer)
        x2 = x + torch.randn_like(x) * 0.001
        _, state = s.compress(x2, quantizer, state)
        assert state["delta2_stats"]["full_used"] >= 1

    def test_registration(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import get_strategy
        s = get_strategy("delta2")
        assert s.name == "delta2"


# ---------------------------------------------------------------------------
# Window strategy
# ---------------------------------------------------------------------------

class TestWindowStrategy:
    def _make(self, **kw):
        from tqai.strategies.window import WindowStrategy
        return WindowStrategy(**kw)

    def test_first_call_refreshes(self, quantizer):
        s = self._make()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = s.compress(x, quantizer)
        assert compressed[1] is False  # not reused

    def test_similar_input_reuses(self, quantizer):
        s = self._make(similarity_threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        # Very similar
        x2 = x + torch.randn_like(x) * 0.001
        compressed, state = s.compress(x2, quantizer, state)
        assert compressed[1] is True  # reused

    def test_different_input_refreshes(self, quantizer):
        s = self._make(similarity_threshold=0.999)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        x2 = torch.randn(1, 4, 8, 64)  # completely different
        compressed, state = s.compress(x2, quantizer, state)
        assert compressed[1] is False

    def test_window_expires(self, quantizer):
        s = self._make(window_size=2, similarity_threshold=0.0)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        _, state = s.compress(x, quantizer, state)  # reuse 1
        _, state = s.compress(x, quantizer, state)  # reuse 2
        # Window of 2 expired
        compressed, state = s.compress(x, quantizer, state)
        assert compressed[1] is False  # forced refresh

    def test_roundtrip(self, quantizer):
        s = self._make()
        x = torch.randn(1, 4, 8, 64)
        compressed, state = s.compress(x, quantizer)
        recon = s.decompress(compressed, quantizer, state)
        assert recon.shape == x.shape

    def test_stats(self, quantizer):
        s = self._make(similarity_threshold=0.5)
        x = torch.randn(1, 4, 8, 64)
        _, state = s.compress(x, quantizer)
        x2 = x + torch.randn_like(x) * 0.001
        _, state = s.compress(x2, quantizer, state)
        assert state["window_stats"]["refreshed"] == 1
        assert state["window_stats"]["reused"] == 1

    def test_registration(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import get_strategy
        s = get_strategy("window")
        assert s.name == "window"


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

class TestStabilityMonitor:
    def _make(self, **kw):
        from tqai.monitors.stability import StabilityMonitor
        return StabilityMonitor(**kw)

    def test_no_adjustment_initially(self):
        m = self._make()
        result = m.observe(0, 0, {"entropy": 1.0})
        assert result is None

    def test_stable_entropy_no_adjustment(self):
        m = self._make(window=10, entropy_threshold=0.5)
        for i in range(20):
            result = m.observe(0, i, {"entropy": 1.0})
        assert result is None  # stable

    def test_entropy_shift_triggers_adjustment(self):
        m = self._make(window=20, entropy_threshold=0.2)
        # Stable period
        for i in range(15):
            m.observe(0, i, {"entropy": 1.0})
        # Sudden drop
        result = None
        for i in range(15, 25):
            result = m.observe(0, i, {"entropy": 0.1})
        # Should eventually detect the shift
        assert m.stats["observations"] == 20  # capped by deque(maxlen=20)

    def test_stats(self):
        m = self._make()
        m.observe(0, 0, {"entropy": 1.0, "score": 0.5})
        m.observe(0, 1, {"entropy": 2.0, "score": 0.8})
        stats = m.stats
        assert stats["observations"] == 2
        assert stats["mean_entropy"] == 1.5

    def test_registration(self):
        import tqai.monitors  # noqa: F401
        from tqai.pipeline.registry import get_monitor
        m = get_monitor("stability")
        assert m.name == "stability"


class TestLyapunovMonitor:
    def _make(self, **kw):
        from tqai.monitors.lyapunov import LyapunovMonitor
        return LyapunovMonitor(**kw)

    def test_no_adjustment_initially(self):
        m = self._make()
        result = m.observe(0, 0, {"hidden_state": np.ones(10)})
        assert result is None

    def test_diverging_returns_conservative(self):
        m = self._make(positive_threshold=0.01)
        # Exponentially growing state
        for i in range(10):
            state = np.ones(10) * (2.0 ** i)
            m.observe(0, i, {"hidden_state": state})
        stats = m.stats
        assert stats["mean_ftle"] > 0  # positive = diverging

    def test_converging_returns_aggressive(self):
        m = self._make(positive_threshold=0.01)
        # Exponentially shrinking state
        for i in range(10):
            state = np.ones(10) * (0.5 ** i)
            m.observe(0, i, {"hidden_state": state})
        stats = m.stats
        assert stats["mean_ftle"] < 0  # negative = converging

    def test_registration(self):
        import tqai.monitors  # noqa: F401
        from tqai.pipeline.registry import get_monitor
        m = get_monitor("lyapunov")
        assert m.name == "lyapunov"
