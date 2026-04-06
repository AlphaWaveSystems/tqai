"""Patch diffusers pipelines for CFG attention sharing.

Wraps the pipeline's denoising loop to automatically manage CFG sharing
phases (conditional/unconditional) without modifying the pipeline code.

Usage::

    from tqai.dit.cfg_patch import patch_cfg_sharing, unpatch_cfg_sharing

    patch_cfg_sharing(pipeline)
    video = pipeline("A cat on a surfboard", num_frames=81)
    unpatch_cfg_sharing(pipeline)
"""

from __future__ import annotations

from typing import Any

from tqai.strategies.cfg_sharing import CFGSharingHooks


def patch_cfg_sharing(
    pipeline: Any,
    share_cross_attn: bool = True,
    similarity_threshold: float = 0.0,
) -> CFGSharingHooks:
    """Patch a diffusers pipeline for CFG attention sharing.

    Detects the pipeline type (WAN split-pass vs LTX batched) and
    installs appropriate hooks.

    Args:
        pipeline: A diffusers pipeline (WanPipeline, LTX2Pipeline, etc.).
        share_cross_attn: Share cross-attention too (default True).
        similarity_threshold: Cosine similarity threshold (0.0 = always share).

    Returns:
        The CFGSharingHooks instance for stats inspection.
    """
    transformer = _find_transformer(pipeline)
    if transformer is None:
        raise ValueError("Cannot find transformer in pipeline")

    mode = _detect_cfg_mode(pipeline)

    hooks = CFGSharingHooks(
        mode=mode,
        share_cross_attn=share_cross_attn,
        similarity_threshold=similarity_threshold,
    )
    hooks.attach(transformer)
    pipeline._tqai_cfg_hooks = hooks

    if mode == "split":
        _patch_split_pass_pipeline(pipeline, hooks)

    return hooks


def unpatch_cfg_sharing(pipeline: Any) -> None:
    """Remove CFG sharing hooks from a pipeline."""
    if hasattr(pipeline, "_tqai_cfg_hooks"):
        pipeline._tqai_cfg_hooks.detach()
        del pipeline._tqai_cfg_hooks

    if hasattr(pipeline, "_tqai_original_call"):
        pipeline.__class__.__call__ = pipeline._tqai_original_call
        del pipeline._tqai_original_call


def _find_transformer(pipeline) -> Any | None:
    for attr in ("transformer", "unet", "dit"):
        model = getattr(pipeline, attr, None)
        if model is not None:
            return model
    return None


def _detect_cfg_mode(pipeline) -> str:
    """Detect whether the pipeline uses split-pass or batched CFG."""
    cls_name = type(pipeline).__name__.lower()

    # WAN pipelines use split-pass (two separate forward calls)
    if "wan" in cls_name:
        return "split"

    # LTX and most others batch cond+uncond into one forward
    return "batched"


def _patch_split_pass_pipeline(pipeline, hooks: CFGSharingHooks) -> None:
    """For split-pass pipelines, wrap __call__ to set phase before each pass.

    This uses a step callback to manage the phase transitions.
    """
    # For WAN, the phase switching happens via the cache_context("cond"/"uncond")
    # We hook into the transformer's cache_context to detect the phase
    transformer = _find_transformer(pipeline)
    if transformer is None:
        return

    original_cache_context = getattr(transformer, "cache_context", None)
    if original_cache_context is None:
        return

    hooks_ref = hooks

    class _PhaseSwitchingContext:
        """Context manager that sets CFG phase based on cache_context name."""

        def __init__(self, name):
            self._name = name
            self._original_ctx = original_cache_context(name)

        def __enter__(self):
            if "uncond" in self._name:
                hooks_ref.set_phase("unconditional")
            else:
                hooks_ref.set_phase("conditional")
                hooks_ref.clear_cache()
            return self._original_ctx.__enter__()

        def __exit__(self, *args):
            return self._original_ctx.__exit__(*args)

    def patched_cache_context(name):
        return _PhaseSwitchingContext(name)

    transformer.cache_context = patched_cache_context
    pipeline._tqai_original_cache_context = original_cache_context
