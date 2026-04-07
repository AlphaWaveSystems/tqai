"""Tests for offline Fisher calibration + the static scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")


# ---------------------------------------------------------------------------
# Stub HF-style model for tests (no network, no downloads)
# ---------------------------------------------------------------------------


class _StubAttention(nn.Module):
    """A minimal HF-style attention module with q_proj/k_proj/v_proj/o_proj."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simple dot-product attention without scaling/masks (test stub)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return self.o_proj(torch.matmul(attn, v))


class _StubLayer(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.self_attn = _StubAttention(dim)
        self.mlp = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _StubModel(nn.Module):
    """A minimal HF-style causal LM with `model.layers[*].self_attn.{q,k,v}_proj`.

    The structure matches what `iter_attention_modules` looks for: a
    `layers` attribute on `self.model` containing modules with
    `self_attn.{q,k,v}_proj`.
    """

    def __init__(self, vocab_size: int = 64, dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # Mimic the HF "model.layers" structure
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_StubLayer(dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        logits = self.lm_head(x)
        if labels is None:
            # Return a mimic of HF CausalLMOutput
            class _Out:
                pass
            o = _Out()
            o.logits = logits
            o.loss = None
            return o
        # Compute next-token CE loss like HF does
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        out.loss = loss
        return out


class _StubTokenizer:
    """Minimal tokenizer interface: callable returning a dict with input_ids."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        # Hash the text deterministically into token ids
        tokens = [(hash(text) + i) % self.vocab_size for i in range(min(len(text), max_length or 32))]
        if not tokens:
            tokens = [0]
        return {"input_ids": torch.tensor([tokens])}

    def encode(self, text):
        return [(hash(text) + i) % self.vocab_size for i in range(len(text))]


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibrateFisher:
    def test_basic_calibration(self, tmp_path):
        from tqai.optimization.fisher_calibration import calibrate_fisher

        torch.manual_seed(42)
        model = _StubModel(vocab_size=64, dim=32, n_layers=3)
        tokenizer = _StubTokenizer(vocab_size=64)

        prompts = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning is the study of algorithms",
            "language models predict the next token",
            "fisher information measures parameter sensitivity",
        ]

        out_path = tmp_path / "fisher.json"
        cal = calibrate_fisher(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_path=out_path,
            max_length=16,
        )

        assert cal.num_layers == 3
        assert cal.num_samples == 4
        assert len(cal.layer_fisher_k) == 3
        assert len(cal.layer_fisher_v) == 3
        # All Fisher values should be positive after backward passes
        for v in cal.layer_fisher_k:
            assert v > 0, "K Fisher value should be positive after calibration"
        for v in cal.layer_fisher_v:
            assert v > 0, "V Fisher value should be positive after calibration"

        # File should exist and round-trip
        assert out_path.exists()
        loaded = type(cal).load(out_path)
        assert loaded.num_layers == cal.num_layers
        assert loaded.layer_fisher_k == cal.layer_fisher_k

    def test_calibration_no_prompts_raises(self):
        from tqai.optimization.fisher_calibration import calibrate_fisher

        model = _StubModel()
        tokenizer = _StubTokenizer()

        with pytest.raises(ValueError, match="At least one calibration prompt"):
            calibrate_fisher(model, tokenizer, prompts=[])

    def test_calibration_no_attention_raises(self):
        from tqai.optimization.fisher_calibration import calibrate_fisher

        # Bare module with no attention
        bare = nn.Sequential(nn.Linear(10, 10))
        tokenizer = _StubTokenizer()

        with pytest.raises(ValueError, match="No attention K/V projections found"):
            calibrate_fisher(bare, tokenizer, prompts=["hello"])

    def test_calibration_returns_to_eval_mode(self):
        from tqai.optimization.fisher_calibration import calibrate_fisher

        model = _StubModel()
        tokenizer = _StubTokenizer()
        model.train(True)

        calibrate_fisher(model, tokenizer, prompts=["test"], max_length=8)
        assert not model.training, "Model should be returned to eval mode"

    def test_calibration_different_layers_get_different_values(self):
        from tqai.optimization.fisher_calibration import calibrate_fisher

        torch.manual_seed(123)
        model = _StubModel(n_layers=4)
        tokenizer = _StubTokenizer()

        cal = calibrate_fisher(
            model, tokenizer,
            prompts=["alpha beta gamma", "delta epsilon zeta", "eta theta iota"],
            max_length=16,
        )

        # Different layers should generally have different values (not all equal)
        assert len(set(cal.layer_fisher_k)) > 1, \
            "All layers got identical Fisher values — calibration likely broken"


# ---------------------------------------------------------------------------
# FisherStaticScorer tests
# ---------------------------------------------------------------------------


class TestFisherStaticScorer:
    def _make_calibration_file(self, tmp_path: Path, k_vals: list[float], v_vals: list[float]) -> Path:
        path = tmp_path / "fisher.json"
        path.write_text(json.dumps({
            "model_id": "test-model",
            "timestamp": "2026-04-06T00:00:00",
            "num_samples": 8,
            "num_layers": len(k_vals),
            "layer_fisher_k": k_vals,
            "layer_fisher_v": v_vals,
            "notes": "test",
        }))
        return path

    def test_load_and_score(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(
            tmp_path,
            k_vals=[0.1, 0.5, 0.9],
            v_vals=[0.2, 0.4, 0.8],
        )

        scorer = FisherStaticScorer(calibration_path=path, kv_mode="k")
        assert scorer.num_layers == 3

        x = torch.randn(1, 4, 8, 32)
        # Layer 0: lowest Fisher → should normalize to 0.0
        result_0 = scorer.score(x, layer_idx=0)
        assert result_0[0].score == 0.0
        # Layer 2: highest Fisher → should normalize to 1.0
        result_2 = scorer.score(x, layer_idx=2)
        assert result_2[0].score == 1.0
        # Layer 1: middle → should be 0.5
        result_1 = scorer.score(x, layer_idx=1)
        assert result_1[0].score == 0.5

    def test_kv_mode_v(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(
            tmp_path,
            k_vals=[1.0, 1.0, 1.0],  # K is uniform → would all be 0.5
            v_vals=[0.1, 0.5, 0.9],  # V varies
        )

        scorer = FisherStaticScorer(calibration_path=path, kv_mode="v")
        result = scorer.score(torch.randn(1, 1, 1, 32), layer_idx=2)
        assert result[0].score == 1.0

    def test_kv_mode_max(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(
            tmp_path,
            k_vals=[0.0, 0.5, 0.0],
            v_vals=[0.0, 0.0, 1.0],
        )

        scorer = FisherStaticScorer(calibration_path=path, kv_mode="max")
        # max per-layer = [0.0, 0.5, 1.0] → normalizes to [0.0, 0.5, 1.0]
        assert scorer.score(torch.randn(1, 1, 1, 32), layer_idx=0)[0].score == 0.0
        assert scorer.score(torch.randn(1, 1, 1, 32), layer_idx=1)[0].score == 0.5
        assert scorer.score(torch.randn(1, 1, 1, 32), layer_idx=2)[0].score == 1.0

    def test_kv_mode_invalid_raises(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.1, 0.2], v_vals=[0.3, 0.4])
        with pytest.raises(ValueError, match="kv_mode must be one of"):
            FisherStaticScorer(calibration_path=path, kv_mode="bogus")

    def test_layer_idx_wraps_when_out_of_range(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.1, 0.5, 0.9], v_vals=[0.1, 0.5, 0.9])
        scorer = FisherStaticScorer(calibration_path=path)

        # layer_idx=5 should wrap to 5 % 3 = 2 → score 1.0
        result = scorer.score(torch.randn(1, 1, 1, 32), layer_idx=5)
        assert result[0].score == 1.0

    def test_uniform_calibration_gives_mid_tier(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.5, 0.5, 0.5], v_vals=[0.5, 0.5, 0.5])
        scorer = FisherStaticScorer(calibration_path=path)
        # All layers equal → all get 0.5
        for i in range(3):
            result = scorer.score(torch.randn(1, 1, 1, 32), layer_idx=i)
            assert result[0].score == 0.5

    def test_metadata_contains_calibration_info(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.1, 0.9], v_vals=[0.2, 0.8])
        scorer = FisherStaticScorer(calibration_path=path)
        result = scorer.score(torch.randn(1, 1, 1, 32), layer_idx=0)
        meta = result[0].metadata
        assert meta["layer_idx"] == 0
        assert meta["kv_mode"] == "k"
        assert meta["calibration_layers"] == 2
        assert "calibration_path" in meta

    def test_empty_calibration_raises(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = tmp_path / "empty.json"
        path.write_text(json.dumps({
            "model_id": "test",
            "timestamp": "2026-04-06T00:00:00",
            "num_samples": 0,
            "num_layers": 0,
            "layer_fisher_k": [],
            "layer_fisher_v": [],
        }))
        with pytest.raises(ValueError, match="zero layers"):
            FisherStaticScorer(calibration_path=path)

    def test_reset_no_op(self, tmp_path):
        from tqai.scorers.fisher_static import FisherStaticScorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.5], v_vals=[0.5])
        scorer = FisherStaticScorer(calibration_path=path)
        # reset() should not crash
        scorer.reset()

    def test_registration(self, tmp_path):
        import tqai.scorers  # noqa: F401  — triggers registration
        from tqai.pipeline.registry import get_scorer

        path = self._make_calibration_file(tmp_path, k_vals=[0.3, 0.7], v_vals=[0.4, 0.6])
        scorer = get_scorer("fisher_static", calibration_path=path)
        assert scorer.name == "fisher_static"
        assert scorer.num_layers == 2


# ---------------------------------------------------------------------------
# End-to-end: calibrate + load + score
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_calibrate_then_score(self, tmp_path):
        from tqai.optimization.fisher_calibration import calibrate_fisher
        from tqai.scorers.fisher_static import FisherStaticScorer

        torch.manual_seed(42)
        model = _StubModel(n_layers=3)
        tokenizer = _StubTokenizer()

        out_path = tmp_path / "e2e.json"
        cal = calibrate_fisher(
            model=model,
            tokenizer=tokenizer,
            prompts=["one", "two three", "four five six"],
            output_path=out_path,
            max_length=16,
        )

        # Now load via the scorer
        scorer = FisherStaticScorer(calibration_path=out_path)
        assert scorer.num_layers == 3

        # All layers should produce a score in [0, 1]
        for i in range(3):
            result = scorer.score(torch.randn(1, 1, 1, 16), layer_idx=i)
            assert 0.0 <= result[0].score <= 1.0
            assert result[0].metadata["layer_idx"] == i

    def test_calibration_via_pipeline_config(self, tmp_path):
        """The static scorer should work end-to-end via tqai.pipeline."""
        import tqai.scorers  # noqa: F401
        import tqai.strategies  # noqa: F401
        from tqai.backend import get_backend
        from tqai.config import TurboQuantConfig
        from tqai.optimization.fisher_calibration import calibrate_fisher
        from tqai.pipeline import build_pipeline
        from tqai.quantizer import PolarQuantizer

        torch.manual_seed(7)
        model = _StubModel(n_layers=2, dim=64)
        tokenizer = _StubTokenizer()
        out_path = tmp_path / "pipe.json"
        calibrate_fisher(
            model=model, tokenizer=tokenizer,
            prompts=["alpha beta", "gamma delta"],
            output_path=out_path, max_length=8,
        )

        ops = get_backend("torch")
        quantizer = PolarQuantizer(head_dim=64, bits=4, seed=42, ops=ops)

        config = TurboQuantConfig(
            bits_k=4, bits_v=2, backend="torch",
            pipeline={
                "scorer": "fisher_static",
                "scorer_kwargs": {"calibration_path": str(out_path)},
                "strategy": "tiered",
            },
        )
        pipe = build_pipeline(config, quantizer=quantizer)
        assert pipe.has_middleware

        x = torch.randn(1, 4, 8, 64)
        compressed = pipe.compress(x, layer_idx=0)
        recon = pipe.decompress(compressed, layer_idx=0)
        assert recon.shape == x.shape
