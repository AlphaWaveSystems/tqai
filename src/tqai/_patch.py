from __future__ import annotations

from tqai.backend import detect_backend
from tqai.config import TurboQuantConfig


def _patch(model, config: TurboQuantConfig):
    detected = config.backend or detect_backend()

    if detected == "mlx":
        from tqai.cache.mlx import patch_mlx

        patch_mlx(model, config)
        # MLX forward hooks not yet supported (Phase 4)
        return None

    from tqai.cache.hf import TurboQuantDynamicCache

    cache = TurboQuantDynamicCache(config)

    # Attach forward-pass activation compression hooks (PyTorch only)
    if config.has_forward_compression:
        from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig

        hook_config = ForwardHookConfig(
            compress_hidden=config.compress_hidden,
            bits_hidden=config.bits_hidden,
            compress_ffn=config.compress_ffn,
            bits_ffn=config.bits_ffn,
            compress_attn_logits=config.compress_attn_logits,
            bits_attn=config.bits_attn,
            seed=config.seed,
        )
        hooks = ForwardCompressionHooks(hook_config)
        hooks.attach(model)
        # Store on model so _unpatch can remove them
        model._tqai_hooks = hooks

    return cache


def _unpatch(model):
    # Remove forward-pass hooks
    if hasattr(model, "_tqai_hooks"):
        model._tqai_hooks.detach()
        del model._tqai_hooks

    # Remove MLX cache patch
    if hasattr(model, "_tqai_original_make_prompt_cache"):
        import mlx_lm.models.cache as cache_module

        cache_module.make_prompt_cache = model._tqai_original_make_prompt_cache
        del model._tqai_original_make_prompt_cache
