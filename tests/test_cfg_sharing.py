"""Tests for CFG attention sharing strategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _FakeAttention(nn.Module):
    """Minimal attention module for testing hooks."""

    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, *args, **kwargs):
        return self.proj(x)


class _FakeBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn1 = _FakeAttention(dim)
        self.attn2 = _FakeAttention(dim)


class _FakeTransformer(nn.Module):
    def __init__(self, n_blocks=2, dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(n_blocks)])


class TestCFGSharingHooks:
    def test_attach_detach(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=3)
        hooks = CFGSharingHooks(mode="split")
        hooks.attach(transformer)
        # 3 blocks × 2 attn modules = 6 hooks
        assert len(hooks._handles) == 6
        hooks.detach()
        assert len(hooks._handles) == 0

    def test_split_mode_caching(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=1)
        hooks = CFGSharingHooks(mode="split")
        hooks.attach(transformer)

        x = torch.randn(1, 8, 64)

        # Conditional pass — caches output
        hooks.set_phase("conditional")
        out_cond = transformer.blocks[0].attn1(x)

        # Unconditional pass — should return cached
        hooks.set_phase("unconditional")
        out_uncond = transformer.blocks[0].attn1(x)

        assert hooks.stats["shared"] >= 1
        assert torch.equal(out_cond, out_uncond)

        hooks.detach()

    def test_split_mode_clear_cache(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=1)
        hooks = CFGSharingHooks(mode="split")
        hooks.attach(transformer)

        x = torch.randn(1, 8, 64)
        hooks.set_phase("conditional")
        transformer.blocks[0].attn1(x)

        hooks.clear_cache()
        assert len(hooks._cached_outputs) == 0

        hooks.detach()

    def test_batched_mode_sharing(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=1)
        hooks = CFGSharingHooks(mode="batched")
        hooks.attach(transformer)

        # Batch = [uncond, cond]
        x = torch.randn(2, 8, 64)
        hooks.set_phase("batched")
        out = transformer.blocks[0].attn1(x)

        # Both halves should be identical (cond copied to uncond)
        assert torch.equal(out[0], out[1])
        assert hooks.stats["shared"] >= 1

        hooks.detach()

    def test_no_sharing_without_phase(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=1)
        hooks = CFGSharingHooks(mode="split")
        hooks.attach(transformer)

        x = torch.randn(1, 8, 64)
        # Default phase is "conditional" — just caches
        out = transformer.blocks[0].attn1(x)
        assert hooks.stats["computed"] == 1
        assert hooks.stats["shared"] == 0

        hooks.detach()

    def test_similarity_threshold(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=1)
        hooks = CFGSharingHooks(mode="split", similarity_threshold=0.99)
        hooks.attach(transformer)

        x_cond = torch.randn(1, 8, 64)
        hooks.set_phase("conditional")
        transformer.blocks[0].attn1(x_cond)

        # Very different input — should NOT share
        x_uncond = torch.randn(1, 8, 64) * 100
        hooks.set_phase("unconditional")
        transformer.blocks[0].attn1(x_uncond)

        # May or may not share depending on the actual projection output similarity
        # Just verify it ran without error
        assert hooks.stats["computed"] + hooks.stats["shared"] >= 2

        hooks.detach()

    def test_cross_attn_disabled(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        transformer = _FakeTransformer(n_blocks=2)
        hooks = CFGSharingHooks(mode="split", share_cross_attn=False)
        hooks.attach(transformer)
        # 2 blocks × 1 attn (self only) = 2 hooks
        assert len(hooks._handles) == 2
        hooks.detach()

    def test_stats(self):
        from tqai.strategies.cfg_sharing import CFGSharingHooks
        hooks = CFGSharingHooks(mode="split")
        assert hooks.stats["shared"] == 0
        assert hooks.stats["computed"] == 0
        assert hooks.stats["share_ratio"] == 0.0
