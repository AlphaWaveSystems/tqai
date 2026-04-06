"""CFG (Classifier-Free Guidance) attention sharing strategy.

During classifier-free guidance in diffusion models, the conditional and
unconditional branches compute nearly identical attention.  This strategy
caches attention K/V from the conditional pass and reuses them for the
unconditional pass, effectively halving the attention compute for CFG steps.

Works with two CFG architectures:
- **Split-pass** (WAN 2.2): Two separate forward passes; hook caches K/V
  from conditional and returns cached for unconditional.
- **Batched** (LTX-2): Single forward with [uncond, cond] concatenated;
  hook copies K/V from cond half to uncond half within the batch.

References:
    - DiTFastAttn CFG sharing: arXiv:2406.08552
    - Classifier-Free Guidance: arXiv:2207.12598
"""

from __future__ import annotations

from typing import Any

import torch


class CFGSharingHooks:
    """Attach post-hooks to attention modules for CFG K/V sharing.

    Usage::

        hooks = CFGSharingHooks(mode="split")
        hooks.attach(transformer)

        # Conditional pass — caches attention K/V
        hooks.set_phase("conditional")
        out_cond = transformer(...)

        # Unconditional pass — reuses cached K/V
        hooks.set_phase("unconditional")
        out_uncond = transformer(...)

        hooks.detach()

    For batched mode (LTX-2), call ``hooks.set_phase("batched")`` and the
    hook will copy cond→uncond within the batch dimension automatically.

    Args:
        mode: ``"split"`` for two-pass CFG (WAN 2.2) or ``"batched"``
            for single-pass batched CFG (LTX-2).
        similarity_threshold: Cosine similarity above which sharing is
            applied.  Set to 0.0 to always share (recommended for CFG
            where conditional/unconditional are provably similar).
        share_cross_attn: Whether to also share cross-attention K/V
            (default True — text conditioning is identical).
    """

    def __init__(
        self,
        mode: str = "split",
        similarity_threshold: float = 0.0,
        share_cross_attn: bool = True,
    ):
        self._mode = mode
        self._similarity_threshold = similarity_threshold
        self._share_cross_attn = share_cross_attn
        self._phase: str = "conditional"
        self._cached_outputs: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._stats = {"shared": 0, "computed": 0}

    def attach(self, transformer) -> None:
        """Attach post-hooks to all attention modules in the transformer."""
        blocks = getattr(transformer, "blocks", None)
        if blocks is None:
            blocks = getattr(transformer, "transformer_blocks", [])

        for i, block in enumerate(blocks):
            # Self-attention (always share)
            attn1 = getattr(block, "attn1", None) or getattr(block, "self_attn", None)
            if attn1 is not None:
                handle = attn1.register_forward_hook(
                    self._make_hook(f"block.{i}.attn1")
                )
                self._handles.append(handle)

            # Cross-attention (share if enabled)
            if self._share_cross_attn:
                attn2 = getattr(block, "attn2", None) or getattr(block, "cross_attn", None)
                if attn2 is not None:
                    handle = attn2.register_forward_hook(
                        self._make_hook(f"block.{i}.attn2")
                    )
                    self._handles.append(handle)

    def detach(self) -> None:
        """Remove all hooks and clear cache."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cached_outputs.clear()
        self._stats = {"shared": 0, "computed": 0}

    def set_phase(self, phase: str) -> None:
        """Set CFG phase: ``"conditional"``, ``"unconditional"``, or ``"batched"``."""
        self._phase = phase

    def clear_cache(self) -> None:
        """Clear cached outputs between denoising steps."""
        self._cached_outputs.clear()

    @property
    def stats(self) -> dict:
        total = self._stats["shared"] + self._stats["computed"]
        return {
            **self._stats,
            "share_ratio": self._stats["shared"] / total if total > 0 else 0.0,
        }

    def _make_hook(self, module_name: str):
        hooks_self = self
        mod_key = hash(module_name)

        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return output

            if hooks_self._mode == "batched":
                return hooks_self._batched_hook(mod_key, output)
            return hooks_self._split_hook(mod_key, output)

        return hook

    def _split_hook(self, mod_key: int, output: torch.Tensor) -> torch.Tensor:
        """Hook for split-pass CFG (WAN 2.2 style)."""
        if self._phase == "conditional":
            # Cache conditional output
            self._cached_outputs[mod_key] = output.detach()
            self._stats["computed"] += 1
            return output

        elif self._phase == "unconditional":
            cached = self._cached_outputs.get(mod_key)
            if cached is not None and cached.shape == output.shape:
                if self._similarity_threshold <= 0.0:
                    self._stats["shared"] += 1
                    return cached
                # Check similarity
                sim = _cosine_similarity_batched(output, cached)
                if sim >= self._similarity_threshold:
                    self._stats["shared"] += 1
                    return cached

            self._stats["computed"] += 1
            return output

        return output

    def _batched_hook(self, mod_key: int, output: torch.Tensor) -> torch.Tensor:
        """Hook for batched CFG (LTX-2 style).

        In batched mode, the first half of batch dim is unconditional,
        the second half is conditional.  Copy cond → uncond.
        """
        batch_size = output.shape[0]
        if batch_size < 2 or batch_size % 2 != 0:
            return output

        half = batch_size // 2
        cond = output[half:]  # conditional is second half
        uncond = output[:half]  # unconditional is first half

        if self._similarity_threshold <= 0.0:
            # Always share: replace uncond with cond
            output = torch.cat([cond, cond], dim=0)
            self._stats["shared"] += 1
        else:
            sim = _cosine_similarity_batched(uncond, cond)
            if sim >= self._similarity_threshold:
                output = torch.cat([cond, cond], dim=0)
                self._stats["shared"] += 1
            else:
                self._stats["computed"] += 1

        return output


def _cosine_similarity_batched(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors."""
    a_flat = a.detach().flatten().float()
    b_flat = b.detach().flatten().float()
    dot = (a_flat * b_flat).sum()
    norms = a_flat.norm() * b_flat.norm() + 1e-10
    return float(dot / norms)
