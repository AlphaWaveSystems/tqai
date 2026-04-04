"""Tests for module_utils: architecture-agnostic attention/FFN detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from tqai.module_utils import is_attention, is_ffn, iter_transformer_layers

# ---------------------------------------------------------------------------
# Fake modules that mimic common HuggingFace patterns
# ---------------------------------------------------------------------------

class FakeSwiGLU(nn.Module):
    """Mimics Llama/Qwen2 MLP (SwiGLU)."""
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj   = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)


class FakeClassicFFN(nn.Module):
    """Mimics GPT-NeoX / BERT-style FFN (fc1/fc2)."""
    def __init__(self, d=64, dff=128):
        super().__init__()
        self.fc1 = nn.Linear(d, dff)
        self.fc2 = nn.Linear(dff, d)


class FakeAttentionSeparate(nn.Module):
    """Mimics Llama/Qwen2 attention (q_proj, k_proj, v_proj, o_proj)."""
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class FakeAttentionFused(nn.Module):
    """Mimics GPT-2 attention (c_attn fused + c_proj)."""
    def __init__(self, d=64):
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)


class FakeLinear(nn.Module):
    """Plain linear layer — should NOT match attention or FFN."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64))


class FakeTransformerLayer(nn.Module):
    """One transformer block with attention + FFN."""
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttentionSeparate()
        self.mlp       = FakeSwiGLU()


class FakeModel(nn.Module):
    """Minimal model with model.layers list (HuggingFace-style nesting)."""
    def __init__(self, n=2):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([FakeTransformerLayer() for _ in range(n)])
        self.model = inner


# ---------------------------------------------------------------------------
# is_attention tests
# ---------------------------------------------------------------------------

class TestIsAttention:
    def test_separate_projections(self):
        assert is_attention(FakeAttentionSeparate())

    def test_fused_c_attn(self):
        assert is_attention(FakeAttentionFused())

    def test_swiglu_not_attention(self):
        assert not is_attention(FakeSwiGLU())

    def test_classic_ffn_not_attention(self):
        assert not is_attention(FakeClassicFFN())

    def test_plain_linear_not_attention(self):
        assert not is_attention(FakeLinear())

    def test_empty_module_not_attention(self):
        assert not is_attention(nn.Module())


# ---------------------------------------------------------------------------
# is_ffn tests
# ---------------------------------------------------------------------------

class TestIsFfn:
    def test_swiglu(self):
        assert is_ffn(FakeSwiGLU())

    def test_classic_fc1_fc2(self):
        assert is_ffn(FakeClassicFFN())

    def test_separate_attention_not_ffn(self):
        assert not is_ffn(FakeAttentionSeparate())

    def test_fused_attention_not_ffn(self):
        # c_attn present → should NOT match c_fc/c_proj check
        assert not is_ffn(FakeAttentionFused())

    def test_plain_linear_not_ffn(self):
        assert not is_ffn(FakeLinear())

    def test_empty_module_not_ffn(self):
        assert not is_ffn(nn.Module())


# ---------------------------------------------------------------------------
# iter_transformer_layers tests
# ---------------------------------------------------------------------------

class TestIterTransformerLayers:
    def test_finds_layers_via_model_attr(self):
        """Model with .model.layers list → yields 2 layers."""
        model = FakeModel(n=2)
        layers = list(iter_transformer_layers(model))
        assert len(layers) == 2

    def test_layer_names_contain_index(self):
        model = FakeModel(n=3)
        names = [name for name, _ in iter_transformer_layers(model)]
        assert any("0" in n for n in names)
        assert any("2" in n for n in names)

    def test_layer_objects_are_modules(self):
        model = FakeModel(n=2)
        for _, layer in iter_transformer_layers(model):
            assert isinstance(layer, nn.Module)

    def test_nested_model_with_direct_layers(self):
        """Model with .layers directly on root (no wrapping)."""
        class DirectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeTransformerLayer() for _ in range(3)])
        model = DirectModel()
        layers = list(iter_transformer_layers(model))
        assert len(layers) == 3
