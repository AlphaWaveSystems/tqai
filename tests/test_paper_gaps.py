"""Tests for paper gap implementations: Sheaf, BSA, copresheaf codebooks, layer protection."""

from __future__ import annotations

import numpy as np
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


@pytest.fixture
def quantizer_low(ops):
    return PolarQuantizer(head_dim=64, bits=2, seed=42, ops=ops)


# ---------------------------------------------------------------------------
# Sheaf scorer (arXiv:2601.21207)
# ---------------------------------------------------------------------------

class TestSheafScorer:
    def _make(self, **kw):
        from tqai.scorers.sheaf import SheafScorer
        return SheafScorer(**kw)

    def test_returns_scored_entry(self):
        scorer = self._make()
        x = torch.randn(1, 4, 16, 64)
        result = scorer.score(x, layer_idx=0)
        assert isinstance(result[0], ScoredEntry)
        assert 0.0 <= result[0].score <= 1.0

    def test_smooth_signal_high_harmonicity(self):
        scorer = self._make()
        # Smooth: constant across sequence
        x = torch.ones(1, 1, 32, 64)
        result = scorer.score(x, layer_idx=0)
        # Constant signal = perfectly harmonic = harmonicity=1 → score=0.5 (low info)
        assert result[0].score <= 0.5

    def test_noisy_signal_low_harmonicity(self):
        scorer = self._make()
        # Noisy: random across sequence
        x = torch.randn(1, 1, 32, 64) * 10
        result = scorer.score(x, layer_idx=0)
        # Random signal = non-harmonic → higher score
        assert result[0].score > 0.0

    def test_short_sequence_handled(self):
        scorer = self._make()
        x = torch.randn(1, 1, 2, 64)  # too short for Laplacian
        result = scorer.score(x, layer_idx=0)
        assert isinstance(result[0], ScoredEntry)

    def test_reset(self):
        scorer = self._make()
        scorer.score(torch.randn(1, 4, 16, 64), layer_idx=0)
        scorer.reset()
        assert scorer._running_harmonicity is None

    def test_registration(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import get_scorer
        s = get_scorer("sheaf")
        assert s.name == "sheaf"


# ---------------------------------------------------------------------------
# BSA scorer (arXiv:2509.01085)
# ---------------------------------------------------------------------------

class TestBSAScorer:
    def _make(self, **kw):
        from tqai.scorers.bsa import BSAScorer
        return BSAScorer(**kw)

    def test_returns_scored_entry(self):
        scorer = self._make()
        x = torch.randn(1, 4, 16, 64)
        result = scorer.score(x, layer_idx=0)
        assert isinstance(result[0], ScoredEntry)
        assert 0.0 <= result[0].score <= 1.0

    def test_uniform_blocks_low_saliency(self):
        scorer = self._make(block_size=8)
        # All tokens identical within blocks → distance from centroid ≈ 0
        x = torch.ones(1, 1, 16, 64)
        result = scorer.score(x, layer_idx=0)
        assert result[0].score < 0.1  # very low saliency

    def test_diverse_tokens_higher_saliency(self):
        scorer = self._make(block_size=8)
        x = torch.randn(1, 1, 16, 64) * 5
        result = scorer.score(x, layer_idx=0)
        assert result[0].score > 0.0

    def test_reset(self):
        scorer = self._make()
        scorer.score(torch.randn(1, 4, 16, 64), layer_idx=0)
        scorer.reset()
        assert scorer._running_saliency is None

    def test_registration(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import get_scorer
        s = get_scorer("bsa")
        assert s.name == "bsa"


# ---------------------------------------------------------------------------
# Tiered with quantizer_low (fixed bug)
# ---------------------------------------------------------------------------

class TestTieredDualQuantizer:
    def test_low_score_uses_different_quantizer(self, quantizer, quantizer_low):
        from tqai.pipeline.runner import CompressionPipeline
        from tqai.strategies.tiered import TieredStrategy

        strategy = TieredStrategy(high_tier_threshold=0.5)
        pipe = CompressionPipeline(
            quantizer=quantizer,
            quantizer_low=quantizer_low,
            strategy=strategy,
        )
        x = torch.randn(1, 4, 8, 64)

        # Force low score by passing raw tensor (defaults to 1.0)
        # We need to use ScoredEntry directly
        entry = [ScoredEntry(data=x, score=0.1, tier=0, metadata={})]
        compressed, state = strategy.compress(entry, quantizer, {"_quantizer_low": quantizer_low})
        assert compressed[1] is False  # low tier

        # Verify roundtrip works (even with different quantizer for decompress)
        recon = strategy.decompress(compressed, quantizer, {"_quantizer_low": quantizer_low})
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# Layer protection (arXiv:2504.10317)
# ---------------------------------------------------------------------------

class TestLayerProtection:
    def test_skip_layers_bypass_middleware(self, quantizer):
        from tqai.pipeline.runner import CompressionPipeline
        from tqai.scorers.palm import PalmScorer
        from tqai.strategies.tiered import TieredStrategy

        pipe = CompressionPipeline(
            quantizer=quantizer,
            scorer=PalmScorer(),
            strategy=TieredStrategy(),
        )
        # Mark layer 5 as non-sparse (must not be compressed with middleware)
        pipe._state["_skip_layers"] = {5}

        x = torch.randn(1, 4, 8, 64)

        # Layer 5: should bypass scorer/strategy, use direct quantize
        result_5 = pipe.compress(x, layer_idx=5)
        # Direct quantize returns (indices, norms) tuple, not ("tiered", ...)
        assert isinstance(result_5, tuple)
        assert not isinstance(result_5[0], str)  # not a tagged strategy output

        # Layer 0: should go through middleware
        result_0 = pipe.compress(x, layer_idx=0)
        assert isinstance(result_0, tuple)
        assert result_0[0] == "tiered"  # tagged strategy output

    def test_skip_layers_from_config(self, quantizer):
        from tqai.config import TurboQuantConfig
        from tqai.pipeline import build_pipeline

        import tqai.scorers  # noqa: F401
        import tqai.strategies  # noqa: F401

        config = TurboQuantConfig(
            bits_k=4, bits_v=2, backend="torch",
            pipeline={
                "scorer": "palm",
                "strategy": "tiered",
                "skip_layers": [3, 7, 15],
            },
        )
        pipe = build_pipeline(config, quantizer=quantizer)
        assert pipe._state.get("_skip_layers") == {3, 7, 15}


# ---------------------------------------------------------------------------
# Copresheaf codebook registry
# ---------------------------------------------------------------------------

class TestCopresheafCodebook:
    def test_generic_codebook_loads(self):
        from tqai.codebook.registry import CodebookRegistry
        reg = CodebookRegistry()
        centroids, boundaries = reg.load(64, 4, head_type="generic")
        assert len(centroids) == 16  # 2^4

    def test_unknown_head_type_falls_back_to_generic(self):
        from tqai.codebook.registry import CodebookRegistry
        reg = CodebookRegistry()
        # "spatial" codebook doesn't exist, should fall back to generic
        centroids, boundaries = reg.load(64, 4, head_type="spatial")
        assert len(centroids) == 16

    def test_codebook_filename_with_head_type(self):
        from tqai.codebook.registry import CodebookRegistry
        assert CodebookRegistry.codebook_filename(128, 4) == "d128_b4.npz"
        assert CodebookRegistry.codebook_filename(128, 4, "spatial") == "d128_b4_spatial.npz"
        assert CodebookRegistry.codebook_filename(128, 4, "generic") == "d128_b4.npz"

    def test_head_types_constant(self):
        from tqai.codebook.registry import CodebookRegistry
        assert "generic" in CodebookRegistry.HEAD_TYPES
        assert "spatial" in CodebookRegistry.HEAD_TYPES
        assert "temporal" in CodebookRegistry.HEAD_TYPES
        assert "cross_attn" in CodebookRegistry.HEAD_TYPES


# ---------------------------------------------------------------------------
# All plugins listed
# ---------------------------------------------------------------------------

class TestFullPluginList:
    def test_all_scorers_registered(self):
        import tqai.scorers  # noqa: F401
        from tqai.pipeline.registry import list_available
        available = list_available()
        expected = {"palm", "snr", "fisher", "sheaf", "bsa"}
        assert expected.issubset(set(available["scorers"]))

    def test_all_strategies_registered(self):
        import tqai.strategies  # noqa: F401
        from tqai.pipeline.registry import list_available
        available = list_available()
        expected = {"tiered", "delta", "delta2", "window"}
        assert expected.issubset(set(available["strategies"]))

    def test_all_monitors_registered(self):
        import tqai.monitors  # noqa: F401
        from tqai.pipeline.registry import list_available
        available = list_available()
        expected = {"stability", "lyapunov"}
        assert expected.issubset(set(available["monitors"]))

    def test_all_adapters_registered(self):
        import tqai.adapters  # noqa: F401
        from tqai.pipeline.registry import list_available
        available = list_available()
        expected = {"llm", "dit", "wan"}
        assert expected.issubset(set(available["adapters"]))
