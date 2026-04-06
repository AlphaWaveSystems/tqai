"""Tests for the tqai pipeline middleware system (Sprint 1)."""

from __future__ import annotations

import pytest
import torch

from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.pipeline import (
    CompressionPipeline,
    ScoredEntry,
    build_pipeline,
    list_available,
    register_scorer,
    register_strategy,
)
from tqai.pipeline.registry import _MONITORS, _SCORERS, _STRATEGIES
from tqai.quantizer import PolarQuantizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ops():
    return get_backend("torch")


@pytest.fixture
def quantizer(ops):
    return PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)


@pytest.fixture
def quantizer_low(ops):
    return PolarQuantizer(head_dim=64, bits=2, seed=42, ops=ops)


def _rand_tensor(batch=1, heads=4, seq=8, dim=64):
    return torch.randn(batch, heads, seq, dim)


def _rand_kv(batch=1, heads=4, seq=8, dim=64):
    return _rand_tensor(batch, heads, seq, dim), _rand_tensor(batch, heads, seq, dim)


# ---------------------------------------------------------------------------
# Stub implementations for testing
# ---------------------------------------------------------------------------

class _StubScorer:
    name = "stub"

    def __init__(self, fixed_score=0.5):
        self.fixed_score = fixed_score
        self._reset_count = 0

    def score(self, x, layer_idx, step=None, context=None):
        return [
            ScoredEntry(data=x, score=self.fixed_score, tier=1, metadata={})
        ]

    def reset(self):
        self._reset_count += 1


class _StubStrategy:
    """Pass-through strategy that delegates to the quantizer directly."""
    name = "stub"

    def compress(self, entry, quantizer, prev_state=None):
        if isinstance(entry, list):
            data = entry[0].data
        else:
            data = entry
        compressed = quantizer.quantize(data)
        return compressed, {"calls": (prev_state or {}).get("calls", 0) + 1}

    def decompress(self, compressed, quantizer, state=None):
        indices, norms = compressed[0], compressed[1]
        qjl = compressed[2] if len(compressed) > 2 else None
        return quantizer.dequantize(indices, norms, qjl)


class _StubMonitor:
    name = "stub"

    def __init__(self):
        self.observations = []

    def observe(self, layer_idx, step, attention_state):
        self.observations.append((layer_idx, step))
        return {"fixed_score": 0.9}  # adjust scorer param


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def setup_method(self):
        # Save and restore registry state
        self._saved_scorers = dict(_SCORERS)
        self._saved_strategies = dict(_STRATEGIES)
        self._saved_monitors = dict(_MONITORS)

    def teardown_method(self):
        _SCORERS.clear()
        _SCORERS.update(self._saved_scorers)
        _STRATEGIES.clear()
        _STRATEGIES.update(self._saved_strategies)
        _MONITORS.clear()
        _MONITORS.update(self._saved_monitors)

    def test_register_and_get_scorer(self):
        register_scorer("test_stub", _StubScorer)
        available = list_available()
        assert "test_stub" in available["scorers"]

        from tqai.pipeline.registry import get_scorer
        scorer = get_scorer("test_stub", fixed_score=0.7)
        assert scorer.fixed_score == 0.7

    def test_register_and_get_strategy(self):
        register_strategy("test_stub", _StubStrategy)
        available = list_available()
        assert "test_stub" in available["strategies"]

    def test_unknown_scorer_raises(self):
        from tqai.pipeline.registry import get_scorer
        with pytest.raises(ValueError, match="Unknown scorer"):
            get_scorer("nonexistent")

    def test_unknown_strategy_raises(self):
        from tqai.pipeline.registry import get_strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")

    def test_list_available(self):
        result = list_available()
        assert "scorers" in result
        assert "strategies" in result
        assert "monitors" in result
        assert "adapters" in result


# ---------------------------------------------------------------------------
# Pipeline runner tests
# ---------------------------------------------------------------------------

class TestCompressionPipeline:
    def test_no_middleware_matches_direct_quantize(self, quantizer):
        """Pipeline without middleware must produce identical output to PolarQuantizer."""
        pipe = CompressionPipeline(quantizer=quantizer)
        assert not pipe.has_middleware

        x = _rand_tensor(dim=64)
        torch.manual_seed(0)
        direct = quantizer.quantize(x)

        torch.manual_seed(0)
        via_pipe = pipe.compress(x)

        # indices and norms should be identical
        assert torch.equal(direct[0], via_pipe[0])
        assert torch.equal(direct[1], via_pipe[1])

    def test_no_middleware_decompress(self, quantizer):
        pipe = CompressionPipeline(quantizer=quantizer)
        x = _rand_tensor(dim=64)
        compressed = pipe.compress(x)
        recon = pipe.decompress(compressed)
        assert recon.shape == x.shape

    def test_scorer_only_still_quantizes(self, quantizer):
        scorer = _StubScorer()
        pipe = CompressionPipeline(quantizer=quantizer, scorer=scorer)
        assert pipe.has_middleware

        x = _rand_tensor(dim=64)
        compressed = pipe.compress(x, layer_idx=0, step=1)
        # With scorer but no strategy, falls through to standard quantize
        assert isinstance(compressed, tuple)
        assert len(compressed) >= 2

    def test_scorer_and_strategy(self, quantizer):
        scorer = _StubScorer()
        strategy = _StubStrategy()
        pipe = CompressionPipeline(
            quantizer=quantizer, scorer=scorer, strategy=strategy
        )
        assert pipe.has_middleware

        x = _rand_tensor(dim=64)
        compressed = pipe.compress(x, layer_idx=0, step=1)
        recon = pipe.decompress(compressed, layer_idx=0)
        assert recon.shape == x.shape

    def test_strategy_state_accumulates(self, quantizer):
        strategy = _StubStrategy()
        pipe = CompressionPipeline(quantizer=quantizer, strategy=strategy)

        x = _rand_tensor(dim=64)
        pipe.compress(x)
        pipe.compress(x)
        pipe.compress(x)
        assert pipe._state["calls"] == 3

    def test_reset_clears_state(self, quantizer):
        scorer = _StubScorer()
        strategy = _StubStrategy()
        pipe = CompressionPipeline(
            quantizer=quantizer, scorer=scorer, strategy=strategy
        )
        x = _rand_tensor(dim=64)
        pipe.compress(x)
        assert pipe._state.get("calls") == 1

        pipe.reset()
        assert pipe._state == {}
        assert scorer._reset_count == 1

    def test_monitor_adjusts_scorer(self, quantizer):
        scorer = _StubScorer(fixed_score=0.5)
        monitor = _StubMonitor()
        pipe = CompressionPipeline(
            quantizer=quantizer, scorer=scorer, monitor=monitor
        )
        pipe.observe(layer_idx=0, step=1, attention_state={})
        assert scorer.fixed_score == 0.9
        assert len(monitor.observations) == 1

    def test_decompress_invalid_format_raises(self, quantizer):
        pipe = CompressionPipeline(quantizer=quantizer)
        with pytest.raises(ValueError, match="Unknown compressed format"):
            pipe.decompress("not_a_tuple")


# ---------------------------------------------------------------------------
# build_pipeline tests
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def setup_method(self):
        self._saved_scorers = dict(_SCORERS)
        self._saved_strategies = dict(_STRATEGIES)
        self._saved_monitors = dict(_MONITORS)

    def teardown_method(self):
        _SCORERS.clear()
        _SCORERS.update(self._saved_scorers)
        _STRATEGIES.clear()
        _STRATEGIES.update(self._saved_strategies)
        _MONITORS.clear()
        _MONITORS.update(self._saved_monitors)

    def test_build_no_pipeline_config(self, quantizer):
        config = TurboQuantConfig(bits_k=4, bits_v=2, backend="torch")
        pipe = build_pipeline(config, quantizer=quantizer)
        assert not pipe.has_middleware

    def test_build_with_scorer(self, quantizer):
        register_scorer("test_stub", _StubScorer)
        config = TurboQuantConfig(
            bits_k=4, bits_v=2, backend="torch",
            pipeline={"scorer": "test_stub", "scorer_kwargs": {"fixed_score": 0.3}},
        )
        pipe = build_pipeline(config, quantizer=quantizer)
        assert pipe.has_middleware
        assert pipe._scorer.fixed_score == 0.3

    def test_build_with_scorer_and_strategy(self, quantizer):
        register_scorer("test_stub", _StubScorer)
        register_strategy("test_stub", _StubStrategy)
        config = TurboQuantConfig(
            bits_k=4, bits_v=2, backend="torch",
            pipeline={"scorer": "test_stub", "strategy": "test_stub"},
        )
        pipe = build_pipeline(config, quantizer=quantizer)
        assert pipe.has_middleware

        x = _rand_tensor(dim=64)
        compressed = pipe.compress(x, layer_idx=0)
        recon = pipe.decompress(compressed)
        assert recon.shape == x.shape

    def test_build_unknown_scorer_raises(self, quantizer):
        config = TurboQuantConfig(
            bits_k=4, bits_v=2, backend="torch",
            pipeline={"scorer": "nonexistent"},
        )
        with pytest.raises(ValueError, match="Unknown scorer"):
            build_pipeline(config, quantizer=quantizer)


# ---------------------------------------------------------------------------
# Cache integration tests (pipeline=None -> same as v0.3.1)
# ---------------------------------------------------------------------------

class TestCacheBackwardCompat:
    def test_hf_cache_no_pipeline(self):
        """HF cache with pipeline=None must behave identically to v0.3.1."""
        from tqai.cache.hf import TurboQuantDynamicCache

        config = TurboQuantConfig(bits_k=4, bits_v=2, backend="torch")
        cache = TurboQuantDynamicCache(config)

        k, v = _rand_kv(seq=8, dim=64)
        all_k, all_v = cache.update(k, v, layer_idx=0)

        assert all_k.shape == (1, 4, 8, 64)
        assert all_v.shape == (1, 4, 8, 64)
        assert cache.get_seq_length(0) == 8

    def test_hf_cache_no_pipeline_sequence_grows(self):
        from tqai.cache.hf import TurboQuantDynamicCache

        config = TurboQuantConfig(bits_k=4, bits_v=2, backend="torch")
        cache = TurboQuantDynamicCache(config)

        k1, v1 = _rand_kv(seq=4, dim=64)
        k2, v2 = _rand_kv(seq=3, dim=64)
        cache.update(k1, v1, layer_idx=0)
        all_k, all_v = cache.update(k2, v2, layer_idx=0)

        assert all_k.shape[2] == 7
        assert cache.get_seq_length(0) == 7

    def test_hf_cache_with_pipeline_stub(self):
        """HF cache with a stub pipeline still produces correct shapes."""
        register_scorer("test_compat", _StubScorer)
        register_strategy("test_compat", _StubStrategy)

        try:
            from tqai.cache.hf import TurboQuantDynamicCache

            config = TurboQuantConfig(
                bits_k=4, bits_v=2, backend="torch",
                pipeline={
                    "scorer": "test_compat",
                    "strategy": "test_compat",
                },
            )
            cache = TurboQuantDynamicCache(config)

            k, v = _rand_kv(seq=8, dim=64)
            all_k, all_v = cache.update(k, v, layer_idx=0)

            assert all_k.shape == (1, 4, 8, 64)
            assert all_v.shape == (1, 4, 8, 64)
        finally:
            _SCORERS.pop("test_compat", None)
            _STRATEGIES.pop("test_compat", None)

    def test_hf_cache_residual_with_pipeline_stub(self):
        """HF residual strategy with pipeline still works."""
        register_scorer("test_residual", _StubScorer)
        register_strategy("test_residual", _StubStrategy)

        try:
            from tqai.cache.hf import TurboQuantDynamicCache

            config = TurboQuantConfig(
                bits_k=4, bits_v=2, backend="torch",
                cache_strategy="residual", residual_window=4,
                pipeline={
                    "scorer": "test_residual",
                    "strategy": "test_residual",
                },
            )
            cache = TurboQuantDynamicCache(config)

            # Feed 10 tokens in 2 batches -> overflow triggers pipeline path
            k1, v1 = _rand_kv(seq=6, dim=64)
            k2, v2 = _rand_kv(seq=4, dim=64)
            cache.update(k1, v1, layer_idx=0)
            all_k, all_v = cache.update(k2, v2, layer_idx=0)

            assert all_k.shape[2] == 10
            assert cache.get_seq_length(0) == 10
        finally:
            _SCORERS.pop("test_residual", None)
            _STRATEGIES.pop("test_residual", None)
