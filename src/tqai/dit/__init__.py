"""DiT (Diffusion Transformer) integration for tqai.

Provides CFG sharing, text encoder caching, VAE memory optimization,
and inter-step delta compression for diffusers-style video generation.
"""

from tqai.dit.cfg_patch import patch_cfg_sharing, unpatch_cfg_sharing
from tqai.dit.vae_memory import optimize_vae_memory, estimate_vae_memory
from tqai.dit.mps_fixes import patch_mps_compatibility

__all__ = [
    "patch_cfg_sharing",
    "unpatch_cfg_sharing",
    "optimize_vae_memory",
    "estimate_vae_memory",
    "patch_mps_compatibility",
]
