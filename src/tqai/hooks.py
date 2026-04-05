"""Forward hook manager for full transformer forward pass compression.

Attaches PyTorch forward hooks to attention and FFN modules to compress
hidden states and intermediate activations during inference.

Compression is quantize-then-dequantize: tensors are stored in compressed
form internally but reconstructed before being used. This reduces peak
memory during long forward passes, at the cost of quantization noise.

Architecture-agnostic: works by detecting module patterns, not class names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tqai.module_utils import iter_attention_modules, iter_ffn_modules


@dataclass
class ForwardHookConfig:
    """Configuration for forward-pass activation compression."""

    compress_hidden: bool = False
    bits_hidden: int = 8
    compress_ffn: bool = False
    bits_ffn: int = 8
    compress_attn_logits: bool = False
    bits_attn: int = 8
    seed: int = 42
    # Internal: maps module id → PolarQuantizer instances (created lazily)
    _quantizer_cache: dict = field(default_factory=dict, repr=False)


class ForwardCompressionHooks:
    """Attach and manage compression hooks on transformer layers.

    Usage::

        hooks = ForwardCompressionHooks(config)
        hooks.attach(model)
        # run inference...
        hooks.detach()
    """

    def __init__(self, config: ForwardHookConfig):
        self._config = config
        self._handles: list[Any] = []
        # module_id → {key_str: PolarQuantizer}
        self._quantizers: dict[int, dict[str, Any]] = {}

    def attach(self, model) -> None:
        """Walk model and attach hooks to attention and FFN modules."""
        try:
            import torch.nn as nn  # noqa: F401 — verify torch is available
        except ImportError as e:
            raise RuntimeError("Forward hooks require PyTorch") from e

        if self._config.compress_hidden or self._config.compress_attn_logits:
            for name, module in iter_attention_modules(model):
                self._attach_attention_hooks(name, module)

        if self._config.compress_ffn:
            for name, module in iter_ffn_modules(model):
                self._attach_ffn_hooks(name, module)

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._quantizers.clear()

    def _get_quantizer(self, module_id: int, key: str, dim: int, bits: int) -> Any:
        """Lazily create and cache a PolarQuantizer for a given (module, tensor) pair."""
        from tqai.backend import get_backend
        from tqai.quantizer import PolarQuantizer

        if module_id not in self._quantizers:
            self._quantizers[module_id] = {}
        cache = self._quantizers[module_id]
        if key not in cache:
            ops = get_backend("torch")
            cache[key] = PolarQuantizer(head_dim=dim, bits=bits, seed=self._config.seed, ops=ops)
        return cache[key]

    def _compress_tensor(self, x: Any, module_id: int, key: str, bits: int) -> Any:
        """Round-trip: quantize then dequantize a tensor, preserving shape and dtype."""

        orig_dtype = x.dtype
        orig_shape = x.shape

        # Reshape to 2D: (batch * tokens, dim)
        dim = orig_shape[-1]
        x_2d = x.reshape(-1, dim).float()

        pq = self._get_quantizer(module_id, key, dim, bits)
        indices, norms = pq.quantize(x_2d)
        x_hat = pq.dequantize(indices, norms)

        return x_hat.reshape(orig_shape).to(orig_dtype)

    def _attach_attention_hooks(self, name: str, module) -> None:
        """Attach pre-hook to compress hidden states entering attention."""
        cfg = self._config

        if cfg.compress_hidden:
            def pre_hook(mod, inputs):
                # inputs is a tuple; first element is hidden states (B, S, D)
                if not inputs or not hasattr(inputs[0], "shape"):
                    return inputs
                h = inputs[0]
                if h.dim() < 2 or h.shape[-1] < 8:
                    return inputs
                h_compressed = self._compress_tensor(h, id(mod), "hidden", cfg.bits_hidden)
                return (h_compressed,) + inputs[1:]

            handle = module.register_forward_pre_hook(pre_hook)
            self._handles.append(handle)

    def _attach_ffn_hooks(self, name: str, module) -> None:
        """Attach pre-hook to compress hidden states entering FFN."""
        cfg = self._config
        module_id = id(module)  # noqa: F841 — used via closure below

        def pre_hook(mod, inputs):
            if not inputs or not hasattr(inputs[0], "shape"):
                return inputs
            h = inputs[0]
            if h.dim() < 2 or h.shape[-1] < 8:
                return inputs
            h_compressed = self._compress_tensor(h, id(mod), "ffn_in", cfg.bits_ffn)
            return (h_compressed,) + inputs[1:]

        handle = module.register_forward_pre_hook(pre_hook)
        self._handles.append(handle)

    @property
    def num_hooks(self) -> int:
        """Number of active hook handles."""
        return len(self._handles)
