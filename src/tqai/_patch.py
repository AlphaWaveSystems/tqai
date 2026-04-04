from __future__ import annotations

from tqai.backend import detect_backend
from tqai.config import TurboQuantConfig


def _patch(model, config: TurboQuantConfig):
    detected = config.backend or detect_backend()

    if detected == "mlx":
        from tqai.cache.mlx import patch_mlx

        patch_mlx(model, config)
        return None

    from tqai.cache.hf import TurboQuantDynamicCache

    return TurboQuantDynamicCache(config)


def _unpatch(model):
    if hasattr(model, "_tqai_original_make_prompt_cache"):
        import mlx_lm.models.cache as cache_module

        cache_module.make_prompt_cache = model._tqai_original_make_prompt_cache
        del model._tqai_original_make_prompt_cache
