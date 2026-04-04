"""Architecture-agnostic module detection for transformer models.

Detects attention and FFN modules across common HuggingFace architectures
(Llama, Qwen2, Mistral, Phi, Gemma, Falcon, GPT-NeoX, etc.) using
attribute-pattern matching rather than class-name checks.
"""

from __future__ import annotations

from typing import Iterator


def is_attention(module) -> bool:
    """Detect multi-head attention modules.

    Matches modules that have q_proj/k_proj/v_proj attributes (HuggingFace
    standard) or c_attn (GPT-2 style fused projection).
    """
    # Standard HuggingFace: separate q/k/v projections
    if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj")):
        return True
    # GPT-2 style: fused c_attn
    if hasattr(module, "c_attn") and hasattr(module, "c_proj"):
        return True
    return False


def is_ffn(module) -> bool:
    """Detect feed-forward / MLP modules.

    Matches SwiGLU (gate_proj + up_proj + down_proj) and classic FFN
    (fc1/fc2 or c_fc/c_proj) patterns.
    """
    # SwiGLU (Llama, Qwen2, Mistral, Gemma)
    if all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
        return True
    # Classic FFN with fc1/fc2 (GPT-NeoX, Falcon, BERT-family)
    if hasattr(module, "fc1") and hasattr(module, "fc2"):
        return True
    # GPT-2 MLP style
    if hasattr(module, "c_fc") and hasattr(module, "c_proj") and not hasattr(module, "c_attn"):
        return True
    return False


def iter_transformer_layers(model) -> Iterator[tuple[str, object]]:
    """Yield (name, module) pairs for the model's transformer layers.

    Handles common HuggingFace nesting patterns:
    - model.layers[i]  (Llama, Qwen2, Mistral, Gemma, Falcon)
    - model.model.layers[i]  (wrapped with CausalLM head)
    - model.transformer.h[i]  (GPT-2)
    - model.gpt_neox.layers[i]  (GPT-NeoX)
    - model.model.decoder.layers[i]  (OPT)
    """
    def _try_layers(root, prefix=""):
        for path in ("layers", "h", "blocks"):
            layers = _nested_get(root, path)
            if layers is not None:
                for i, layer in enumerate(layers):
                    name = f"{prefix}{path}.{i}" if prefix else f"{path}.{i}"
                    yield name, layer
                return

    # Unwrap outer model wrapper (CausalLM → inner model)
    inner = (
        _nested_get(model, "model")
        or _nested_get(model, "transformer")
        or _nested_get(model, "gpt_neox")
    )
    if inner is not None:
        yield from _try_layers(inner, prefix="model.")
    else:
        yield from _try_layers(model)


def _nested_get(obj, attr: str):
    """Get attribute from object if it exists, else None."""
    return getattr(obj, attr, None)


def get_hidden_dim(model) -> int | None:
    """Extract d_model (hidden dimension) from model config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("hidden_size", "d_model", "n_embd"):
        val = getattr(config, attr, None)
        if val is not None:
            return val
    return None


def get_ffn_dim(model) -> int | None:
    """Extract d_ff (intermediate/FFN dimension) from model config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("intermediate_size", "ffn_dim", "n_inner", "d_ff"):
        val = getattr(config, attr, None)
        if val is not None:
            return val
    return None


def iter_attention_modules(model) -> Iterator[tuple[str, object]]:
    """Yield (name, module) for every attention sub-module in transformer layers."""
    for layer_name, layer in iter_transformer_layers(model):
        for name, module in layer.named_modules():
            if is_attention(module):
                full_name = f"{layer_name}.{name}" if name else layer_name
                yield full_name, module


def iter_ffn_modules(model) -> Iterator[tuple[str, object]]:
    """Yield (name, module) for every FFN/MLP sub-module in transformer layers."""
    for layer_name, layer in iter_transformer_layers(model):
        for name, module in layer.named_modules():
            if is_ffn(module):
                full_name = f"{layer_name}.{name}" if name else layer_name
                yield full_name, module
