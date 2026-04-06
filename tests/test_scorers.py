"""Tests for scorer modules."""

from __future__ import annotations

import numpy as np
import torch

from tqai.pipeline.base import ScoredEntry


class TestPalmScorer:
    def _make_scorer(self, **kwargs):
        from tqai.scorers.palm import PalmScorer
        return PalmScorer(**kwargs)

    def test_score_returns_scored_entries(self):
        scorer = self._make_scorer()
        x = torch.randn(1, 4, 8, 64)
        result = scorer.score(x, layer_idx=0)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ScoredEntry)

    def test_score_during_warmup(self):
        scorer = self._make_scorer(warmup_steps=5)
        x = torch.randn(1, 4, 8, 64)
        for _ in range(5):
            result = scorer.score(x, layer_idx=0)
            assert result[0].tier == 2  # mid-tier during warmup

    def test_novelty_increases_with_different_input(self):
        scorer = self._make_scorer(warmup_steps=0)
        # Feed identical inputs to establish baseline
        x_same = torch.ones(1, 4, 1, 64)
        for _ in range(20):
            scorer.score(x_same, layer_idx=0)

        # Now feed very different input
        x_diff = torch.randn(1, 4, 1, 64) * 10
        result = scorer.score(x_diff, layer_idx=0)
        assert result[0].score > 0.0

    def test_get_bits_tiered(self):
        scorer = self._make_scorer()
        assert scorer.get_bits(0.05) == 2  # tier 0
        assert scorer.get_bits(0.2) == 3   # tier 1
        assert scorer.get_bits(0.5) == 4   # tier 2
        assert scorer.get_bits(0.9) == 8   # tier 3

    def test_reset_clears_state(self):
        scorer = self._make_scorer()
        x = torch.randn(1, 4, 8, 64)
        scorer.score(x, layer_idx=0)
        assert scorer._step_count == 1
        scorer.reset()
        assert scorer._step_count == 0

    def test_registration(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import get_scorer
        scorer = get_scorer("palm")
        assert scorer.name == "palm"


class TestPalmEMATracker:
    def test_first_update_returns_zero(self):
        from tqai.scorers.palm import _EMATracker
        tracker = _EMATracker(alpha=0.1)
        novelty = tracker.update(np.ones(10))
        assert novelty == 0.0

    def test_identical_inputs_low_novelty(self):
        from tqai.scorers.palm import _EMATracker
        tracker = _EMATracker(alpha=0.1)
        x = np.ones(10)
        tracker.update(x)
        for _ in range(10):
            novelty = tracker.update(x)
        assert novelty < 0.5

    def test_different_input_high_novelty(self):
        from tqai.scorers.palm import _EMATracker
        tracker = _EMATracker(alpha=0.1)
        x = np.ones(10)
        for _ in range(20):
            tracker.update(x)
        novelty = tracker.update(np.ones(10) * 100)
        assert novelty > 1.0
