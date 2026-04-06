"""Text encoder output caching across denoising steps.

In diffusion models, the text encoder produces fixed K/V tensors from the
prompt that are identical across all ~50 denoising steps.  This module
caches the encoder output after the first call and returns the cached
version on subsequent calls, optionally compressing it.

For WAN 2.2 with T5-XXL encoder: ~100MB savings from avoiding redundant
encoding, plus bandwidth reduction from compressed cache reads across
50 steps.

References:
    - TurboQuant: arXiv:2504.19874
    - T5 encoder: arXiv:1910.10683
"""

from __future__ import annotations

from typing import Any

import torch

from tqai.quantizer import PolarQuantizer


class TextEncoderCache:
    """Cache and optionally compress text encoder output.

    First call: run encoder normally, cache the output
    Subsequent calls: return cached output (skip encoder forward pass)

    If ``bits`` is set, the cached output is stored compressed via
    PolarQuantizer and dequantized on each read, saving memory at the
    cost of minor reconstruction error.
    """

    def __init__(self, bits: int = 8, seed: int = 42, compress: bool = True):
        self._bits = bits
        self._seed = seed
        self._compress = compress
        self._cached_output: Any = None
        self._compressed_data: tuple | None = None
        self._quantizer: PolarQuantizer | None = None
        self._original_forward: Any = None
        self._encoder: Any = None

    def attach(self, encoder) -> None:
        """Wrap the encoder's forward to cache output after first call."""
        self._encoder = encoder
        self._original_forward = encoder.forward
        cache_self = self

        def cached_forward(*args, **kwargs):
            if cache_self._cached_output is not None:
                return cache_self._get_cached()
            output = cache_self._original_forward(*args, **kwargs)
            cache_self._store(output)
            return output

        encoder.forward = cached_forward

    def detach(self) -> None:
        """Restore original encoder forward."""
        if self._encoder is not None and self._original_forward is not None:
            self._encoder.forward = self._original_forward
        self._encoder = None
        self._original_forward = None
        self._cached_output = None
        self._compressed_data = None
        self._quantizer = None

    def _store(self, output):
        """Store encoder output, optionally compressed."""
        if not self._compress:
            self._cached_output = output
            return

        # Handle different output types (tuple, BaseModelOutput, tensor)
        if isinstance(output, torch.Tensor):
            self._cached_output = self._compress_tensor(output)
        elif hasattr(output, "last_hidden_state"):
            # BaseModelOutput or similar
            compressed = self._compress_tensor(output.last_hidden_state)
            self._cached_output = (compressed, output)
        else:
            # Fall back to storing uncompressed
            self._cached_output = output

    def _get_cached(self):
        """Return cached output, dequantizing if compressed."""
        cached = self._cached_output
        if isinstance(cached, tuple) and len(cached) == 2:
            tensor, original_output = cached
            if isinstance(tensor, tuple):
                # Compressed: (indices, norms, orig_shape, orig_dtype, device)
                indices, norms, shape, dtype, device = tensor
                decompressed = self._quantizer.dequantize(indices, norms)
                decompressed = decompressed.reshape(shape).to(dtype).to(device)
                # Reconstruct the output object
                original_output.last_hidden_state = decompressed
                return original_output
        return cached

    def _compress_tensor(self, tensor: torch.Tensor):
        """Compress a tensor via PolarQuantizer round-trip."""
        from tqai.backend import get_backend

        orig_shape = tensor.shape
        orig_dtype = tensor.dtype
        dim = orig_shape[-1]

        if self._quantizer is None:
            ops = get_backend("torch", device="cpu")
            self._quantizer = PolarQuantizer(
                head_dim=dim, bits=self._bits, seed=self._seed, ops=ops
            )

        # Move to CPU for quantization (rotation matrix is on CPU)
        x_2d = tensor.detach().cpu().reshape(-1, dim).float()
        indices, norms = self._quantizer.quantize(x_2d)
        return (indices, norms, orig_shape, orig_dtype, tensor.device)

    def reset(self) -> None:
        """Clear cached output (e.g., between different prompts)."""
        self._cached_output = None
        self._compressed_data = None
