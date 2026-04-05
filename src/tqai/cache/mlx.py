"""MLX KV cache wrapper with TurboQuant compression for mlx-lm."""

from __future__ import annotations

from typing import Any

from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.quantizer import PolarQuantizer


class TurboQuantMLXCache:
    """Drop-in replacement for mlx-lm's ``KVCache`` with TurboQuant compression.

    Matches the ``KVCache`` interface: ``update_and_fetch``, ``state``,
    ``offset``, ``is_empty``.
    """

    def __init__(self, head_dim: int, n_kv_heads: int, config: TurboQuantConfig):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.config = config
        self._ops = get_backend("mlx")

        self._k_quantizer = PolarQuantizer(
            head_dim=head_dim,
            bits=config.bits_k,
            seed=config.seed,
            ops=self._ops,
            pre_rotated=config.pre_rotated,
            use_qjl=config.use_qjl,
            qjl_sketch_size=config.qjl_sketch_size,
        )
        self._v_quantizer = PolarQuantizer(
            head_dim=head_dim,
            bits=config.bits_v,
            seed=config.seed + 10000,
            ops=self._ops,
            pre_rotated=config.pre_rotated,
            use_qjl=config.use_qjl,
            qjl_sketch_size=config.qjl_sketch_size,
        )

        self._compressed_keys: list[tuple[Any, Any]] = []
        self._compressed_values: list[tuple[Any, Any]] = []
        self._sink_keys: Any | None = None
        self._sink_values: Any | None = None
        self.offset: int = 0
        self._input_dtype = None

        # For mlx-lm compatibility
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        """Append new KV states, return full history (dequantized).

        Args:
            keys: ``[1, n_kv_heads, new_seq, head_dim]``
            values: same shape

        Returns:
            ``(all_keys, all_values)`` with full sequence.
        """
        import mlx.core as mx

        if self._input_dtype is None:
            self._input_dtype = keys.dtype
        new_seq = keys.shape[2]
        sink = self.config.sink_tokens

        # Handle sink tokens
        if self.offset < sink:
            sink_end = min(new_seq, sink - self.offset)
            if self._sink_keys is None:
                self._sink_keys = keys[:, :, :sink_end, :]
                self._sink_values = values[:, :, :sink_end, :]
            else:
                self._sink_keys = mx.concatenate(
                    [self._sink_keys, keys[:, :, :sink_end, :]], axis=2
                )
                self._sink_values = mx.concatenate(
                    [self._sink_values, values[:, :, :sink_end, :]], axis=2
                )
            keys = keys[:, :, sink_end:, :]
            values = values[:, :, sink_end:, :]

        # Compress remaining
        if keys.shape[2] > 0:
            self._compressed_keys.append(self._k_quantizer.quantize(keys))
            self._compressed_values.append(self._v_quantizer.quantize(values))

        self.offset += new_seq

        all_keys = self._reconstruct(is_key=True)
        all_values = self._reconstruct(is_key=False)

        # Update state references for compatibility
        self.keys = all_keys
        self.values = all_values

        return all_keys, all_values

    def _reconstruct(self, is_key: bool):
        import mlx.core as mx

        sink = self._sink_keys if is_key else self._sink_values
        compressed = self._compressed_keys if is_key else self._compressed_values
        quantizer = self._k_quantizer if is_key else self._v_quantizer

        parts = []
        if sink is not None:
            parts.append(sink)
        for entry in compressed:
            indices, norms = entry[0], entry[1]
            qjl_data = entry[2] if len(entry) > 2 else None
            parts.append(quantizer.dequantize(indices, norms, qjl_data))

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim))

        result = mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
        if self._input_dtype is not None and result.dtype != self._input_dtype:
            result = result.astype(self._input_dtype)
        return result

    @property
    def state(self):
        if self.offset == 0:
            return None, None
        return self._reconstruct(True), self._reconstruct(False)

    @property
    def is_empty(self) -> bool:
        return self.offset == 0


def _detect_head_dim_and_kv_heads(model) -> tuple[int, int]:
    """Infer head_dim and n_kv_heads from an mlx-lm model."""
    if hasattr(model, "args"):
        args = model.args
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None and hasattr(args, "hidden_size"):
            head_dim = args.hidden_size // args.num_attention_heads
        n_kv_heads = getattr(args, "num_key_value_heads", getattr(args, "num_attention_heads", 1))
        if head_dim:
            return head_dim, n_kv_heads
    raise ValueError("Cannot detect head_dim from model. Pass head_dim explicitly.")


def patch_mlx(model, config: TurboQuantConfig):
    """Monkey-patch mlx-lm to use TurboQuant-compressed caches."""
    import mlx_lm.models.cache as cache_module

    head_dim, n_kv_heads = _detect_head_dim_and_kv_heads(model)
    original = cache_module.make_prompt_cache

    def patched_make_prompt_cache(model, max_kv_size=None, **kwargs):
        num_layers = len(model.layers) if hasattr(model, "layers") else 1
        return [
            TurboQuantMLXCache(
                head_dim=head_dim,
                n_kv_heads=n_kv_heads,
                config=config,
            )
            for _ in range(num_layers)
        ]

    cache_module.make_prompt_cache = patched_make_prompt_cache
    model._tqai_original_make_prompt_cache = original
