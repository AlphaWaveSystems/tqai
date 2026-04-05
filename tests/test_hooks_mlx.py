"""Tests for MLX forward compression hooks."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# -- Fake MLX model components ------------------------------------------------


class FakeAttention(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, mask=None, cache=None):
        return self.o_proj(self.q_proj(x))


class FakeMLP(nn.Module):
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.relu(self.gate_proj(x)) * self.up_proj(x))


class FakeTransformerLayer(nn.Module):
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.self_attn = FakeAttention(d)
        self.mlp = FakeMLP(d, dff)

    def __call__(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class FakeModel(nn.Module):
    def __init__(self, n_layers=2, d=64, dff=128):
        super().__init__()
        self.layers = [FakeTransformerLayer(d, dff) for _ in range(n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -- Tests ---------------------------------------------------------------------


def test_attach_creates_hooks():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=2, d=64)
    config = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    assert hooks.num_hooks == 2  # one attention per layer


def test_attach_ffn_hooks():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=2, d=64)
    config = ForwardHookConfig(compress_ffn=True, bits_ffn=8)
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    assert hooks.num_hooks == 2  # one MLP per layer


def test_attach_both():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=3, d=64)
    config = ForwardHookConfig(
        compress_hidden=True, bits_hidden=8,
        compress_ffn=True, bits_ffn=8,
    )
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    assert hooks.num_hooks == 6  # 3 attention + 3 MLP


def test_detach_restores_modules():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=2, d=64)
    original_attn_type = type(model.layers[0].self_attn)

    config = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)
    assert hooks.num_hooks == 2
    # Module replaced with wrapper
    assert type(model.layers[0].self_attn) is not original_attn_type

    hooks.detach()
    assert hooks.num_hooks == 0
    # Original module restored
    assert type(model.layers[0].self_attn) is original_attn_type


def test_forward_shape_preserved():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=2, d=64)
    x = mx.random.normal((1, 16, 64), key=mx.random.key(0))
    mx.synchronize()

    config = ForwardHookConfig(
        compress_hidden=True, bits_hidden=8,
        compress_ffn=True, bits_ffn=8,
    )
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    out = model(x)
    mx.synchronize()

    assert out.shape == (1, 16, 64)
    hooks.detach()


def test_8bit_close_to_baseline():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=1, d=64)
    x = mx.random.normal((1, 8, 64), key=mx.random.key(2))
    mx.synchronize()

    out_baseline = model(x)
    mx.synchronize()

    config = ForwardHookConfig(compress_hidden=True, bits_hidden=8)
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    out_compressed = model(x)
    mx.synchronize()
    hooks.detach()

    diff = np.max(np.abs(np.array(out_baseline) - np.array(out_compressed)))
    assert diff < 1.0, f"8-bit diff too large: {diff}"


def test_4bit_differs_from_baseline():
    from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

    model = FakeModel(n_layers=1, d=64)
    x = mx.random.normal((1, 8, 64), key=mx.random.key(3))
    mx.synchronize()

    out_baseline = model(x)
    mx.synchronize()

    config = ForwardHookConfig(compress_hidden=True, bits_hidden=4)
    hooks = MLXForwardCompressionHooks(config)
    hooks.attach(model)

    out_compressed = model(x)
    mx.synchronize()
    hooks.detach()

    diff = np.max(np.abs(np.array(out_baseline) - np.array(out_compressed)))
    assert diff > 0.001, f"4-bit should differ: {diff}"
