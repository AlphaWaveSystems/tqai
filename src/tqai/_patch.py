from __future__ import annotations

from tqai.backend import detect_backend
from tqai.config import TurboQuantConfig


def _patch(model, config: TurboQuantConfig):
    detected = config.backend or detect_backend()

    if detected == "mlx":
        # Only patch the cache when KV compression is requested
        if config.kv_compression:
            from tqai.cache.mlx import patch_mlx

            patch_mlx(model, config)

        # Forward-pass activation compression (MLX)
        if config.has_forward_compression:
            from tqai.hooks import ForwardHookConfig, MLXForwardCompressionHooks

            hook_config = ForwardHookConfig(
                compress_hidden=config.compress_hidden,
                bits_hidden=config.bits_hidden,
                compress_ffn=config.compress_ffn,
                bits_ffn=config.bits_ffn,
                compress_attn_logits=config.compress_attn_logits,
                bits_attn=config.bits_attn,
                seed=config.seed,
            )
            hooks = MLXForwardCompressionHooks(hook_config)
            hooks.attach(model)
            model._tqai_hooks = hooks

        # Chunked attention for long sequences
        if config.chunk_attention:
            from tqai.attention import patch_chunked_attention

            patch_chunked_attention(model, config.attention_chunk_size)

        return None

    cache = None
    if config.kv_compression:
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
        model._tqai_hooks = hooks

    return cache


def _unpatch(model):
    # Remove forward-pass hooks
    if hasattr(model, "_tqai_hooks"):
        model._tqai_hooks.detach()
        del model._tqai_hooks

    # Remove chunked attention patch
    if hasattr(model, "_tqai_original_sdpa"):
        from tqai.attention import unpatch_chunked_attention

        unpatch_chunked_attention(model)

    # Remove MLX cache patch
    if hasattr(model, "_tqai_original_make_prompt_cache"):
        import mlx_lm.models.cache as cache_module

        cache_module.make_prompt_cache = model._tqai_original_make_prompt_cache
        del model._tqai_original_make_prompt_cache
