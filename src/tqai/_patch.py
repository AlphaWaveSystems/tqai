from __future__ import annotations

from tqai.backend import detect_backend
from tqai.config import TurboQuantConfig


def _detect_baked_config(model) -> dict | None:
    """Check model config directory for tqai_bake_config.json.

    Returns parsed dict if found, else None.
    """
    import json
    from pathlib import Path

    config = getattr(model, "config", None)
    if config is None:
        return None
    name_or_path = getattr(config, "_name_or_path", None)
    if not name_or_path:
        return None
    bake_cfg_path = Path(name_or_path) / "tqai_bake_config.json"
    if not bake_cfg_path.exists():
        return None
    try:
        return json.loads(bake_cfg_path.read_text())
    except Exception:
        return None


def _patch(model, config: TurboQuantConfig):
    # Auto-detect rotation-baked model
    baked = _detect_baked_config(model)
    if baked and baked.get("tqai_baked"):
        if not config.pre_rotated:
            config.pre_rotated = True
        # Respect baked bits/seed if caller didn't override them explicitly
        if config.bits_k == 4 and "bits_k" in baked:
            config.bits_k = baked["bits_k"]
        if config.bits_v == 2 and "bits_v" in baked:
            config.bits_v = baked["bits_v"]
        if config.seed == 42 and "seed" in baked:
            config.seed = baked["seed"]

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
