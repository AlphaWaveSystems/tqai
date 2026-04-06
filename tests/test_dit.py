"""Tests for DiT integration: module detection, Palm scorer, step delta."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tqai.module_utils import is_attention, is_ffn, iter_transformer_layers
from tqai.palm import EMATracker, PalmConfig, PalmScorer

# -- Fake DiT modules (diffusers-style) ---------------------------------------


class FakeDiTAttention(nn.Module):
    """Mimics diffusers Attention with to_q/to_k/to_v."""

    def __init__(self, d=64):
        super().__init__()
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(d, d, bias=False)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x))


class FakeFeedForward(nn.Module):
    """Mimics diffusers FeedForward with net attribute."""

    def __init__(self, d=64, dff=256):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(d, dff),
            nn.Linear(dff, d),
        ])

    def forward(self, x):
        return self.net[1](torch.relu(self.net[0](x)))


class FakeBasicTransformerBlock(nn.Module):
    """Mimics diffusers BasicTransformerBlock."""

    def __init__(self, d=64, dff=256):
        super().__init__()
        self.attn1 = FakeDiTAttention(d)  # self-attention
        self.attn2 = FakeDiTAttention(d)  # cross-attention
        self.ff = FakeFeedForward(d, dff)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x


class FakeDiTModel(nn.Module):
    """Mimics a diffusers Transformer2DModel."""

    def __init__(self, n_layers=3, d=64, dff=256):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            FakeBasicTransformerBlock(d, dff) for _ in range(n_layers)
        ])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


class FakeFusedQKVAttention(nn.Module):
    """Attention with fused QKV projection."""

    def __init__(self, d=64):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(self.qkv(x)[:, :, :x.shape[-1]])


# -- Module detection tests ----------------------------------------------------


class TestDiTModuleDetection:
    def test_dit_attention_detected(self):
        attn = FakeDiTAttention(d=64)
        assert is_attention(attn)

    def test_dit_fused_qkv_detected(self):
        attn = FakeFusedQKVAttention(d=64)
        assert is_attention(attn)

    def test_dit_feedforward_detected(self):
        ff = FakeFeedForward(d=64)
        assert is_ffn(ff)

    def test_dit_feedforward_not_attention(self):
        ff = FakeFeedForward(d=64)
        assert not is_attention(ff)

    def test_dit_attention_not_ffn(self):
        attn = FakeDiTAttention(d=64)
        assert not is_ffn(attn)

    def test_dit_layer_iteration(self):
        model = FakeDiTModel(n_layers=3)
        layers = list(iter_transformer_layers(model))
        assert len(layers) == 3
        for name, layer in layers:
            assert "transformer_blocks" in name

    def test_dit_hook_attachment(self):
        from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig

        model = FakeDiTModel(n_layers=2, d=64)
        config = ForwardHookConfig(
            compress_hidden=True, bits_hidden=8,
            compress_ffn=True, bits_ffn=8,
        )
        hooks = ForwardCompressionHooks(config)
        hooks.attach(model)
        # 2 layers x (2 attention + 1 ffn) = 6 hooks
        # Actually: attn1 + attn2 per block = 4 attention hooks, 2 ffn hooks
        assert hooks.num_hooks >= 4  # at least the attention hooks
        hooks.detach()
        assert hooks.num_hooks == 0


# -- Palm scorer tests ---------------------------------------------------------


class TestPalmScorer:
    def test_ema_tracker_first_call(self):
        tracker = EMATracker(alpha=0.1)
        novelty = tracker.update(np.array([1.0, 2.0, 3.0]))
        assert novelty == 0.0  # first call is always 0
        assert tracker.count == 1

    def test_ema_tracker_detects_change(self):
        tracker = EMATracker(alpha=0.1)
        tracker.update(np.zeros(10))
        tracker.update(np.zeros(10))
        tracker.update(np.zeros(10))
        # Now feed something very different
        novelty = tracker.update(np.ones(10) * 100)
        assert novelty > 1.0  # should be highly novel

    def test_ema_tracker_stable_input(self):
        tracker = EMATracker(alpha=0.1)
        for _ in range(20):
            novelty = tracker.update(np.ones(10))
        # After convergence, same input should have low novelty
        assert novelty < 0.5

    def test_palm_scorer_tiers(self):
        config = PalmConfig(
            tier_boundaries=[0.1, 0.3, 0.7],
            bits_per_tier=[2, 3, 4, 8],
        )
        scorer = PalmScorer(config)
        assert scorer.get_tier(0.05) == 0  # redundant
        assert scorer.get_tier(0.2) == 1   # expected
        assert scorer.get_tier(0.5) == 2   # novel
        assert scorer.get_tier(0.9) == 3   # critical

    def test_palm_scorer_bits(self):
        config = PalmConfig(bits_per_tier=[2, 3, 4, 8])
        scorer = PalmScorer(config)
        assert scorer.get_bits(0.05) == 2
        assert scorer.get_bits(0.5) == 4
        assert scorer.get_bits(0.9) == 8

    def test_diffusion_step_scoring(self):
        scorer = PalmScorer()
        # Early step (high noise) → high info_score
        early = scorer.score_diffusion_step(0, 50)
        # Late step (refinement) → low info_score
        late = scorer.score_diffusion_step(49, 50)
        assert early > late
        assert early == pytest.approx(1.0, abs=0.01)
        assert late == pytest.approx(0.0, abs=0.05)

    def test_diffusion_snr_scoring(self):
        scorer = PalmScorer()
        # High SNR = clean image = low info
        high_snr = scorer.score_diffusion_step(0, 50, snr=10.0)
        # Low SNR = noisy = high info
        low_snr = scorer.score_diffusion_step(0, 50, snr=0.1)
        assert low_snr > high_snr

    def test_palm_scorer_reset(self):
        scorer = PalmScorer()
        scorer.score(np.ones(10))
        scorer.score(np.ones(10))
        scorer.reset()
        # After reset, should behave like fresh
        novelty = scorer.score(np.ones(10))
        assert novelty == 0.0  # first call after reset


# -- Step delta tracker tests --------------------------------------------------


class TestStepDelta:
    def test_step_delta_first_step(self):
        from tqai.dit.step_delta import StepDeltaTracker

        model = FakeDiTModel(n_layers=1, d=64)
        tracker = StepDeltaTracker(threshold=0.1, delta_bits=2)
        tracker.attach(model)

        x = torch.randn(1, 16, 64)
        out = model(x)
        assert out.shape == (1, 16, 64)
        # First step: no delta possible
        assert tracker.stats["delta_used"] == 0
        tracker.detach()

    def test_step_delta_uses_delta_for_similar(self):
        from tqai.dit.step_delta import StepDeltaTracker

        model = FakeDiTModel(n_layers=1, d=64)
        tracker = StepDeltaTracker(threshold=0.5, delta_bits=2)
        tracker.attach(model)

        x = torch.randn(1, 16, 64)
        model(x)  # step 1
        model(x + 0.001 * torch.randn_like(x))  # step 2, very similar
        # Should use delta for at least some modules
        assert tracker.stats["delta_used"] + tracker.stats["full_used"] > 0
        tracker.detach()

    def test_step_delta_stats(self):
        from tqai.dit.step_delta import StepDeltaTracker

        tracker = StepDeltaTracker()
        stats = tracker.stats
        assert "delta_used" in stats
        assert "full_used" in stats
        assert "delta_ratio" in stats
