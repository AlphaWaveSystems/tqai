"""Tests for ForwardCompressionHooks: attach, detach, quantize-in-forward."""

from __future__ import annotations

import torch
import torch.nn as nn

from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig

# ---------------------------------------------------------------------------
# Fake modules matching patterns detected by module_utils
# ---------------------------------------------------------------------------

class FakeAttention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        return self.o_proj(q)


class FakeMLP(nn.Module):
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj   = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class FakeTransformerLayer(nn.Module):
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.self_attn = FakeAttention(d)
        self.mlp       = FakeMLP(d, dff)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.mlp(x)
        return x


class FakeModel(nn.Module):
    def __init__(self, n_layers=2, d=64, dff=128):
        super().__init__()
        self.layers = nn.ModuleList([FakeTransformerLayer(d, dff) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dummy_input(batch=1, seq=16, d=64):
    return torch.randn(batch, seq, d)


# ---------------------------------------------------------------------------
# Hook attachment / detachment
# ---------------------------------------------------------------------------

class TestHookAttachDetach:
    def test_attach_registers_handles_for_hidden(self):
        model = FakeModel(n_layers=2)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        assert hooks.num_hooks > 0
        hooks.detach()

    def test_attach_registers_handles_for_ffn(self):
        model = FakeModel(n_layers=2)
        cfg = ForwardHookConfig(compress_ffn=True, bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        assert hooks.num_hooks > 0
        hooks.detach()

    def test_attach_both_more_handles_than_single(self):
        model = FakeModel(n_layers=2)
        cfg_single = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
        cfg_both   = ForwardHookConfig(compress_hidden=True, bits_hidden=8,
                                       compress_ffn=True,   bits_ffn=8)
        h1 = ForwardCompressionHooks(cfg_single)
        h1.attach(model)
        n_single = h1.num_hooks
        h1.detach()

        h2 = ForwardCompressionHooks(cfg_both)
        h2.attach(model)
        n_both = h2.num_hooks
        h2.detach()

        assert n_both >= n_single

    def test_detach_removes_all_handles(self):
        model = FakeModel(n_layers=2)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8, compress_ffn=True, bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        assert hooks.num_hooks > 0
        hooks.detach()
        assert hooks.num_hooks == 0

    def test_no_hooks_when_all_disabled(self):
        model = FakeModel(n_layers=2)
        cfg = ForwardHookConfig()  # all False by default
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        assert hooks.num_hooks == 0
        hooks.detach()

    def test_detach_twice_is_safe(self):
        model = FakeModel(n_layers=2)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        hooks.detach()
        hooks.detach()  # should not raise


# ---------------------------------------------------------------------------
# Forward pass correctness with hooks
# ---------------------------------------------------------------------------

class TestHookedForward:
    def test_hidden_compression_forward_runs(self):
        """Model forward pass must succeed with hidden-state hooks attached."""
        model = FakeModel(n_layers=2, d=64)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)

        x = _dummy_input(batch=1, seq=8, d=64)
        out = model(x)

        assert out.shape == x.shape
        hooks.detach()

    def test_ffn_compression_forward_runs(self):
        model = FakeModel(n_layers=2, d=64)
        cfg = ForwardHookConfig(compress_ffn=True, bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)

        x = _dummy_input(batch=1, seq=8, d=64)
        out = model(x)

        assert out.shape == x.shape
        hooks.detach()

    def test_combined_compression_forward_runs(self):
        model = FakeModel(n_layers=2, d=64)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8,
                                compress_ffn=True,   bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)

        x = _dummy_input(batch=1, seq=8, d=64)
        out = model(x)

        assert out.shape == x.shape
        hooks.detach()

    def test_output_dtype_preserved(self):
        """Hooked forward should not change output dtype."""
        model = FakeModel(n_layers=1, d=64)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)

        x = _dummy_input(d=64).float()
        out = model(x)

        assert out.dtype == x.dtype
        hooks.detach()

    def test_output_shape_batch_seq(self):
        """Larger batch/seq should still produce correct shape."""
        model = FakeModel(n_layers=2, d=64)
        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8,
                                compress_ffn=True,   bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)

        x = _dummy_input(batch=4, seq=32, d=64)
        out = model(x)

        assert out.shape == x.shape
        hooks.detach()

    def test_output_without_hooks_not_identical_with_high_compression(self):
        """With low-bit compression (4-bit), output should differ from no-hook baseline."""
        model = FakeModel(n_layers=2, d=64)
        x = _dummy_input(batch=1, seq=8, d=64)

        out_baseline = model(x).detach()

        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=4,
                                compress_ffn=True,   bits_ffn=4)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        out_compressed = model(x).detach()
        hooks.detach()

        # Should differ (quantization noise) but not be all zeros
        assert not torch.allclose(out_baseline, out_compressed, atol=1e-6)
        assert out_compressed.abs().sum() > 0

    def test_8bit_output_close_to_baseline(self):
        """8-bit compression should produce output close to uncompressed baseline."""
        model = FakeModel(n_layers=2, d=128)
        x = _dummy_input(batch=1, seq=8, d=128)

        out_baseline = model(x).detach()

        cfg = ForwardHookConfig(compress_hidden=True, bits_hidden=8,
                                compress_ffn=True,   bits_ffn=8)
        hooks = ForwardCompressionHooks(cfg)
        hooks.attach(model)
        out_compressed = model(x).detach()
        hooks.detach()

        # 8-bit should be within 20% relative error on average
        rel_err = (out_baseline - out_compressed).abs().mean() / (out_baseline.abs().mean() + 1e-8)
        assert rel_err < 0.20, f"8-bit relative error too large: {rel_err:.4f}"


# ---------------------------------------------------------------------------
# tqai.patch integration
# ---------------------------------------------------------------------------

class TestPatchIntegration:
    def test_patch_attaches_hooks_and_unpatch_removes(self):
        """tqai.patch with compress_hidden=True should store hooks on model."""
        import tqai

        model = FakeModel(n_layers=2, d=64)
        tqai.patch(model, backend="torch", compress_hidden=True, bits_hidden=8)

        assert hasattr(model, "_tqai_hooks"), "hooks not stored on model"
        assert model._tqai_hooks.num_hooks > 0

        tqai.unpatch(model)
        assert not hasattr(model, "_tqai_hooks"), "hooks not removed after unpatch"

    def test_patch_no_hooks_when_compression_disabled(self):
        """Default tqai.patch (KV only) should NOT attach forward hooks."""
        import tqai

        model = FakeModel(n_layers=2, d=64)
        tqai.patch(model, backend="torch")

        assert not hasattr(model, "_tqai_hooks"), "unexpected hooks on model"
        tqai.unpatch(model)
