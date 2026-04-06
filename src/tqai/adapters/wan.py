"""WAN 2.2 specific adapter.

Handles Alibaba's WAN 2.2 video generation model (TI2V-5B, A14B).
WAN uses a Flow Matching DiT with cross-attention to T5 text embeddings.
"""

from __future__ import annotations

from typing import Any


class WANAdapter:
    """Adapter for WAN 2.2 (Wan-AI/Wan2.2-TI2V-5B and variants).

    Detects WAN models by checking for the ``wan`` or ``WanTransformer``
    architecture identifiers.
    """

    name = "wan"

    def detect(self, model: Any) -> bool:
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            return False
        cls_name = type(transformer).__name__.lower()
        if "wan" in cls_name:
            return True
        config = getattr(transformer, "config", None)
        if config:
            model_type = getattr(config, "model_type", "")
            if "wan" in model_type.lower():
                return True
        return False

    def get_attention_modules(self, model: Any):
        transformer = getattr(model, "transformer", model)
        # WAN uses transformer_blocks with self-attn + cross-attn
        blocks = getattr(transformer, "transformer_blocks", [])
        if not blocks:
            blocks = getattr(transformer, "blocks", [])

        for i, block in enumerate(blocks):
            for attn_name in ("attn1", "attn2", "self_attn", "cross_attn"):
                attn = getattr(block, attn_name, None)
                if attn is not None:
                    yield f"block.{i}.{attn_name}", attn

    def get_head_info(self, model: Any) -> dict:
        transformer = getattr(model, "transformer", model)
        config = getattr(transformer, "config", None)
        if config is None:
            return {"head_dim": None, "n_heads": None, "n_kv_heads": None, "n_layers": None}

        n_heads = getattr(config, "num_attention_heads", None)
        hidden = getattr(config, "hidden_size", 0)
        head_dim = hidden // n_heads if n_heads and hidden else None
        n_layers = getattr(config, "num_layers", None)

        return {
            "head_dim": head_dim,
            "n_heads": n_heads,
            "n_kv_heads": n_heads,
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

        # Also set up text encoder caching if available
        text_enc = getattr(model, "text_encoder", None)
        if text_enc is not None:
            from tqai.dit.step_cache import TextEncoderCache

            cache = TextEncoderCache(bits=config.bits_k, seed=config.seed + 50000)
            cache.attach(text_enc)
            model._tqai_text_cache = cache

        return None

    def unpatch(self, model: Any) -> None:
        if hasattr(model, "_tqai_hooks"):
            model._tqai_hooks.detach()
            del model._tqai_hooks
        if hasattr(model, "_tqai_text_cache"):
            model._tqai_text_cache.detach()
            del model._tqai_text_cache
