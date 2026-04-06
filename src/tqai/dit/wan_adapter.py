"""WAN 2.2 and generic DiT pipeline adapter for tqai compression.

Patches diffusers-style diffusion transformer pipelines to compress:
1. Text encoder KV: compress T5/CLIP output once, reuse across denoising steps
2. DiT self-attention: compress K/V activations per denoising step
3. Cross-attention K/V: compress text→DiT cross-attention cache

Supports WAN 2.2 (TI2V-5B, A14B), Stable Diffusion 3, Flux, and any
diffusers pipeline with a ``transformer`` attribute.

References:
    - TurboQuant: arXiv:2504.19874
    - WAN 2.2: Alibaba (2026), Flow Matching DiT with MoE
    - Diffusers BasicTransformerBlock: github.com/huggingface/diffusers
"""

from __future__ import annotations

from typing import Any

from tqai.hooks import ForwardCompressionHooks, ForwardHookConfig


def patch_dit(
    pipeline: Any,
    bits_hidden: int = 8,
    bits_ffn: int = 8,
    compress_hidden: bool = True,
    compress_ffn: bool = True,
    seed: int = 42,
    enable_step_cache: bool = True,
    enable_step_delta: bool = False,
    delta_threshold: float = 0.1,
    delta_bits: int = 2,
) -> None:
    """Patch a diffusers pipeline for tqai DiT compression.

    Args:
        pipeline: A diffusers pipeline with a ``transformer`` or ``unet``
            attribute (WAN 2.2, SD3, Flux, etc.).
        bits_hidden: Bits for hidden state compression (default 8).
        bits_ffn: Bits for FFN activation compression (default 8).
        compress_hidden: Compress residual stream entering attention blocks.
        compress_ffn: Compress hidden states entering FFN blocks.
        seed: RNG seed for quantizer.
        enable_step_cache: Cache text encoder output across denoising steps.
        enable_step_delta: Enable inter-step K/V delta compression.
        delta_threshold: Norm threshold for delta vs full storage.
        delta_bits: Bits for delta compression (1-4).
    """
    # Find the DiT/transformer component
    dit = _find_transformer(pipeline)
    if dit is None:
        raise ValueError(
            "Cannot find transformer in pipeline. "
            "Expected pipeline.transformer or pipeline.unet."
        )

    # Attach forward compression hooks to DiT attention/FFN blocks
    hook_config = ForwardHookConfig(
        compress_hidden=compress_hidden,
        bits_hidden=bits_hidden,
        compress_ffn=compress_ffn,
        bits_ffn=bits_ffn,
        seed=seed,
    )

    # Use PyTorch hooks (diffusers models are PyTorch)
    hooks = ForwardCompressionHooks(hook_config)
    hooks.attach(dit)
    pipeline._tqai_dit_hooks = hooks

    # Text encoder caching
    if enable_step_cache:
        from tqai.dit.step_cache import TextEncoderCache

        text_enc = _find_text_encoder(pipeline)
        if text_enc is not None:
            cache = TextEncoderCache(
                bits=bits_hidden, seed=seed + 50000
            )
            cache.attach(text_enc)
            pipeline._tqai_text_cache = cache

    # Inter-step delta compression
    if enable_step_delta:
        from tqai.dit.step_delta import StepDeltaTracker

        tracker = StepDeltaTracker(
            threshold=delta_threshold,
            delta_bits=delta_bits,
            seed=seed + 60000,
        )
        tracker.attach(dit)
        pipeline._tqai_step_delta = tracker


def unpatch_dit(pipeline: Any) -> None:
    """Remove all tqai patches from a diffusers pipeline."""
    if hasattr(pipeline, "_tqai_dit_hooks"):
        pipeline._tqai_dit_hooks.detach()
        del pipeline._tqai_dit_hooks

    if hasattr(pipeline, "_tqai_text_cache"):
        pipeline._tqai_text_cache.detach()
        del pipeline._tqai_text_cache

    if hasattr(pipeline, "_tqai_step_delta"):
        pipeline._tqai_step_delta.detach()
        del pipeline._tqai_step_delta


def _find_transformer(pipeline) -> Any | None:
    """Find the DiT/transformer component in a diffusers pipeline."""
    for attr in ("transformer", "unet", "dit"):
        model = getattr(pipeline, attr, None)
        if model is not None:
            return model
    return None


def _find_text_encoder(pipeline) -> Any | None:
    """Find the text encoder in a diffusers pipeline."""
    for attr in ("text_encoder", "text_encoder_2", "tokenizer"):
        enc = getattr(pipeline, attr, None)
        if enc is not None and hasattr(enc, "forward"):
            return enc
    return None
