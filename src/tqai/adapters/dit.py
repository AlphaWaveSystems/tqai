"""Generic Diffusion Transformer (DiT) adapter.

Handles diffusers-style pipelines with a ``transformer`` attribute
(Stable Diffusion 3, Flux, and generic BasicTransformerBlock models).
"""

from __future__ import annotations

from typing import Any


class DiTAdapter:
    """Adapter for diffusers DiT models.

    Detects pipelines with a ``transformer`` attribute that uses
    ``BasicTransformerBlock`` layers (diffusers convention).
    """

    name = "dit"

    def detect(self, model: Any) -> bool:
        # Diffusers pipeline: has .transformer with .transformer_blocks or .blocks
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            return False
        if hasattr(transformer, "transformer_blocks"):
            return True
        if hasattr(transformer, "blocks"):
            return True
        return False

    def get_attention_modules(self, model: Any):
        transformer = getattr(model, "transformer", model)
        blocks = getattr(transformer, "transformer_blocks", None) or getattr(transformer, "blocks", [])
        for i, block in enumerate(blocks):
            # diffusers BasicTransformerBlock has attn1/attn2
            for attn_name in ("attn1", "attn2", "self_attn"):
                attn = getattr(block, attn_name, None)
                if attn is not None:
                    yield f"block.{i}.{attn_name}", attn

    def get_head_info(self, model: Any) -> dict:
        transformer = getattr(model, "transformer", model)
        config = getattr(transformer, "config", None)
        if config is None:
            return {"head_dim": None, "n_heads": None, "n_kv_heads": None, "n_layers": None}

        n_heads = getattr(config, "num_attention_heads", getattr(config, "attention_head_dim", None))
        inner_dim = getattr(config, "inner_dim", None) or getattr(config, "hidden_size", 0)
        head_dim = inner_dim // n_heads if n_heads and inner_dim else None
        n_layers = getattr(config, "num_layers", getattr(config, "num_hidden_layers", None))

        return {
            "head_dim": head_dim,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,  # DiT typically doesn't use GQA
            "n_layers": n_layers,
        }

    def patch(self, model: Any, pipeline: Any, config: Any) -> Any:
        from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig

        transformer = getattr(model, "transformer", model)
        hook_config = ForwardHookConfig(
            compress_hidden=True,
            bits_hidden=config.bits_k,
            compress_ffn=True,
            bits_ffn=config.bits_v,
            seed=config.seed,
        )
        hooks = ForwardCompressionHooks(hook_config)
        hooks.attach(transformer)
        model._tqai_hooks = hooks
        return None

    def unpatch(self, model: Any) -> None:
        if hasattr(model, "_tqai_hooks"):
            model._tqai_hooks.detach()
            del model._tqai_hooks
