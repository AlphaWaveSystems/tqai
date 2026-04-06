"""MPS (Apple Silicon) compatibility fixes for video diffusion pipelines.

Patches known float64 incompatibilities in diffusers models that prevent
them from running on Apple's MPS backend.

Usage::

    from tqai.dit.mps_fixes import patch_mps_compatibility

    pipe = LTX2Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("mps")
    patch_mps_compatibility(pipe)  # fixes float64 RoPE, etc.
"""

from __future__ import annotations

from typing import Any


def patch_mps_compatibility(pipeline: Any) -> int:
    """Patch all known MPS float64 incompatibilities in a diffusers pipeline.

    Currently fixes:
    - LTX-2 RoPE ``double_precision=True`` in connectors and transformer

    Args:
        pipeline: A diffusers pipeline loaded on MPS.

    Returns:
        Number of modules patched.
    """
    patched = 0

    # Fix 1: RoPE double_precision in LTX-2 connectors and transformer
    for component_name in ("connectors", "transformer", "unet"):
        component = getattr(pipeline, component_name, None)
        if component is None:
            continue
        for module in component.modules():
            if hasattr(module, "double_precision") and module.double_precision:
                module.double_precision = False
                patched += 1

    return patched
