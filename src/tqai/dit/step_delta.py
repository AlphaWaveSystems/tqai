"""Inter-step delta compression for diffusion transformer denoising.

Adjacent denoising steps produce nearly identical K/V activations.  This
module tracks activations across steps and stores only the delta when it
is small (late refinement steps), falling back to full storage when the
delta is large (early high-noise steps).

The threshold follows the diffusion schedule naturally: early steps have
high noise and large deltas (full compression), late steps have low noise
and tiny deltas (delta-only at 1-2 bits).

This is analogous to Palm's novelty/surprise framework applied across
denoising steps rather than across tokens.

References:
    - Palm information-theoretic scoring (novelty/surprise)
    - Diffusion schedule SNR as stability proxy
    - TurboQuant: arXiv:2504.19874
"""

from __future__ import annotations

from typing import Any

import torch

from tqai.quantizer import PolarQuantizer


class StepDeltaTracker:
    """Track and compress K/V deltas between denoising steps.

    For each attention module, maintains the previous step's K/V values.
    On each new step:
    - Compute delta = current - previous
    - If ||delta|| / ||current|| < threshold: store delta at low bits
    - Else: store full current at standard bits
    - Reconstruct: previous + dequant(delta) or dequant(full)

    Args:
        threshold: Relative norm threshold for delta vs full (default 0.1).
        delta_bits: Bits for delta compression (default 2).
        full_bits: Bits for full compression fallback (default 4).
        seed: RNG seed for quantizers.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        delta_bits: int = 2,
        full_bits: int = 4,
        seed: int = 42,
    ):
        self._threshold = threshold
        self._delta_bits = delta_bits
        self._full_bits = full_bits
        self._seed = seed

        # Per-module state: {module_id: prev_output_tensor}
        self._prev_outputs: dict[int, torch.Tensor] = {}
        self._delta_quantizers: dict[int, PolarQuantizer] = {}
        self._full_quantizers: dict[int, PolarQuantizer] = {}
        self._handles: list[Any] = []
        self._stats = {"delta_used": 0, "full_used": 0}

    def attach(self, model) -> None:
        """Attach post-hooks to attention modules for delta tracking."""
        from tqai.module_utils import iter_attention_modules

        for _name, module in iter_attention_modules(model):
            handle = module.register_forward_hook(self._make_hook(id(module)))
            self._handles.append(handle)

    def detach(self) -> None:
        """Remove all hooks and clear state."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._prev_outputs.clear()
        self._delta_quantizers.clear()
        self._full_quantizers.clear()
        self._stats = {"delta_used": 0, "full_used": 0}

    def reset_step(self) -> None:
        """Call between denoising steps to prepare for delta tracking."""
        pass  # State persists across steps by design

    @property
    def stats(self) -> dict:
        """Return delta vs full usage statistics."""
        total = self._stats["delta_used"] + self._stats["full_used"]
        return {
            **self._stats,
            "delta_ratio": (
                self._stats["delta_used"] / total if total > 0 else 0.0
            ),
        }

    def _make_hook(self, mod_id: int):
        tracker = self

        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return output
            return tracker._process_output(mod_id, output)

        return hook

    def _process_output(
        self, mod_id: int, output: torch.Tensor
    ) -> torch.Tensor:
        """Compare with previous step, compress via delta or full."""
        prev = self._prev_outputs.get(mod_id)

        if prev is None or prev.shape != output.shape:
            # First step or shape change: store and return as-is
            self._prev_outputs[mod_id] = output.detach()
            return output

        # Compute delta
        delta = output - prev
        delta_norm = torch.norm(delta).item()
        output_norm = torch.norm(output).item() + 1e-10
        relative_delta = delta_norm / output_norm

        if relative_delta < self._threshold:
            # Small delta: compress delta at low bits, reconstruct
            compressed = self._compress(
                delta, mod_id, is_delta=True
            )
            reconstructed = prev + compressed
            self._stats["delta_used"] += 1
        else:
            # Large delta: compress full output
            reconstructed = self._compress(
                output, mod_id, is_delta=False
            )
            self._stats["full_used"] += 1

        self._prev_outputs[mod_id] = reconstructed.detach()
        return reconstructed

    def _compress(
        self, tensor: torch.Tensor, mod_id: int, is_delta: bool
    ) -> torch.Tensor:
        """Quantize-then-dequantize a tensor."""
        from tqai.backend import get_backend

        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        dim = orig_shape[-1]

        bits = self._delta_bits if is_delta else self._full_bits
        cache = self._delta_quantizers if is_delta else self._full_quantizers

        if mod_id not in cache:
            ops = get_backend("torch")
            cache[mod_id] = PolarQuantizer(
                head_dim=dim, bits=bits, seed=self._seed, ops=ops
            )

        pq = cache[mod_id]
        x_2d = tensor.reshape(-1, dim).float()
        indices, norms = pq.quantize(x_2d)
        x_hat = pq.dequantize(indices, norms)
        return x_hat.reshape(orig_shape).to(orig_dtype)
