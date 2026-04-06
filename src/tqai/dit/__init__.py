"""Diffusion Transformer (DiT) integration for tqai.

Provides compression hooks and adapters for video diffusion models
(WAN 2.2, Stable Diffusion 3, Flux, etc.) using TurboQuant quantization.

Key components:

- :func:`patch_dit` — Top-level API for patching diffusion pipelines
- :class:`TextEncoderCache` — Compress and cache text encoder output
- :class:`StepDeltaCache` — Inter-step delta compression for denoising
- :class:`PalmScorer` — Information-theoretic adaptive bit allocation

References:
    - TurboQuant: arXiv:2504.19874
    - Flash Attention: arXiv:2205.14135
    - KIVI: arXiv:2402.02750
"""

from tqai.dit.wan_adapter import patch_dit, unpatch_dit

__all__ = ["patch_dit", "unpatch_dit"]
