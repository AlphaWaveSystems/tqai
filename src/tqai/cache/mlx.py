"""MLX KV cache wrapper with TurboQuant compression for mlx-lm.

Implements incremental and residual cache strategies to avoid O(n²)
full-reconstruction overhead.

References:
    - TurboQuant: arXiv:2504.19874
    - KIVI (residual buffer strategy): arXiv:2402.02750
    - KVQuant (fused dequant-attention motivation): arXiv:2401.18079
"""

from __future__ import annotations

from typing import Any

from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.quantizer import PolarQuantizer


class TurboQuantMLXCache:
    """Drop-in replacement for mlx-lm's ``KVCache`` with TurboQuant compression.

    Matches the ``KVCache`` interface: ``update_and_fetch``, ``state``,
    ``offset``, ``is_empty``.

    Supports three cache strategies:

    - **incremental** (default): Maintains a running dequantized buffer.
      Only the new token is dequantized each step — O(1) per token.
    - **residual**: Keeps the last ``residual_window`` tokens uncompressed.
      Older tokens are quantized then dequantized into an incremental buffer.
      Combines the speed of incremental with zero error on recent tokens.
    - **full**: Legacy O(n) full-reconstruction path (dequantizes entire
      history every token).
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
            use_qjl=config.use_qjl,
            qjl_sketch_size=config.qjl_sketch_size,
        )
        self._v_quantizer = PolarQuantizer(
            head_dim=head_dim,
            bits=config.bits_v,
            seed=config.seed + 10000,
            ops=self._ops,
            use_qjl=config.use_qjl,
            qjl_sketch_size=config.qjl_sketch_size,
        )

        # Resolve strategy
        strategy = config.cache_strategy
        if strategy == "auto":
            strategy = "incremental"
        self._strategy = strategy

        # Sink tokens (uncompressed, kept for all strategies)
        self._sink_keys: Any | None = None
        self._sink_values: Any | None = None

        # Compressed storage (kept for memory savings / future use)
        self._compressed_keys: list[tuple[Any, ...]] = []
        self._compressed_values: list[tuple[Any, ...]] = []

        # Incremental dequantized buffer (strategies: incremental, residual)
        self._k_buffer: Any | None = None  # [1, H, seq, D] in input dtype
        self._v_buffer: Any | None = None

        # Recent uncompressed window (strategy: residual)
        self._recent_keys: Any | None = None  # [1, H, <=R, D] full precision
        self._recent_values: Any | None = None

        self.offset: int = 0
        self._input_dtype = None

        # For mlx-lm compatibility
        self.keys = None
        self.values = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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

        # Handle sink tokens (shared across all strategies)
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

        # Dispatch to strategy
        if keys.shape[2] > 0:
            if self._strategy == "residual":
                self._update_residual(keys, values, mx)
            elif self._strategy == "incremental":
                self._update_incremental(keys, values, mx)
            else:
                self._update_full(keys, values)

        self.offset += new_seq

        # Assemble full history
        all_keys = self._assemble(is_key=True, mx=mx)
        all_values = self._assemble(is_key=False, mx=mx)

        self.keys = all_keys
        self.values = all_values
        return all_keys, all_values

    # ------------------------------------------------------------------
    # Strategy: incremental — O(1) dequant per token
    # ------------------------------------------------------------------

    def _update_incremental(self, keys, values, mx):
        """Quantize, store compressed, dequantize new token, append to buffer."""
        k_entry = self._k_quantizer.quantize(keys)
        v_entry = self._v_quantizer.quantize(values)
        self._compressed_keys.append(k_entry)
        self._compressed_values.append(v_entry)

        # Dequantize only the new token(s)
        k_dequant = self._dequant_entry(self._k_quantizer, k_entry)
        v_dequant = self._dequant_entry(self._v_quantizer, v_entry)

        # Cast to input dtype
        if self._input_dtype is not None:
            k_dequant = k_dequant.astype(self._input_dtype)
            v_dequant = v_dequant.astype(self._input_dtype)

        # Append to running buffer
        if self._k_buffer is None:
            self._k_buffer = k_dequant
            self._v_buffer = v_dequant
        else:
            self._k_buffer = mx.concatenate([self._k_buffer, k_dequant], axis=2)
            self._v_buffer = mx.concatenate([self._v_buffer, v_dequant], axis=2)

    # ------------------------------------------------------------------
    # Strategy: residual — recent tokens uncompressed, old incremental
    # ------------------------------------------------------------------

    def _update_residual(self, keys, values, mx):
        """Append to recent window; overflow → compress → incremental buffer."""
        R = self.config.residual_window

        # Append new tokens to recent window
        if self._recent_keys is None:
            self._recent_keys = keys
            self._recent_values = values
        else:
            self._recent_keys = mx.concatenate(
                [self._recent_keys, keys], axis=2
            )
            self._recent_values = mx.concatenate(
                [self._recent_values, values], axis=2
            )

        # Overflow: compress oldest tokens beyond window
        while self._recent_keys.shape[2] > R:
            overflow = self._recent_keys.shape[2] - R
            # Extract overflow tokens from the front
            old_k = self._recent_keys[:, :, :overflow, :]
            old_v = self._recent_values[:, :, :overflow, :]
            # Trim recent window
            self._recent_keys = self._recent_keys[:, :, overflow:, :]
            self._recent_values = self._recent_values[:, :, overflow:, :]

            # Compress and immediately dequantize into buffer
            k_entry = self._k_quantizer.quantize(old_k)
            v_entry = self._v_quantizer.quantize(old_v)
            self._compressed_keys.append(k_entry)
            self._compressed_values.append(v_entry)

            k_dequant = self._dequant_entry(self._k_quantizer, k_entry)
            v_dequant = self._dequant_entry(self._v_quantizer, v_entry)

            if self._input_dtype is not None:
                k_dequant = k_dequant.astype(self._input_dtype)
                v_dequant = v_dequant.astype(self._input_dtype)

            if self._k_buffer is None:
                self._k_buffer = k_dequant
                self._v_buffer = v_dequant
            else:
                self._k_buffer = mx.concatenate(
                    [self._k_buffer, k_dequant], axis=2
                )
                self._v_buffer = mx.concatenate(
                    [self._v_buffer, v_dequant], axis=2
                )

    # ------------------------------------------------------------------
    # Strategy: full — legacy O(n) reconstruction
    # ------------------------------------------------------------------

    def _update_full(self, keys, values):
        """Store compressed only (reconstruction happens in _assemble)."""
        self._compressed_keys.append(self._k_quantizer.quantize(keys))
        self._compressed_values.append(self._v_quantizer.quantize(values))

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, is_key: bool, mx):
        """Build the full KV tensor from sink + buffer/recent or full reconstruction."""
        sink = self._sink_keys if is_key else self._sink_values

        if self._strategy == "full":
            return self._reconstruct_full(is_key, mx)

        buffer = self._k_buffer if is_key else self._v_buffer
        recent = (self._recent_keys if is_key else self._recent_values) if self._strategy == "residual" else None

        parts = []
        if sink is not None:
            parts.append(sink)
        if buffer is not None:
            parts.append(buffer)
        if recent is not None:
            parts.append(recent)

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim))

        result = mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
        if self._input_dtype is not None and result.dtype != self._input_dtype:
            result = result.astype(self._input_dtype)
        return result

    def _reconstruct_full(self, is_key: bool, mx):
        """Legacy full reconstruction — dequantizes entire history."""
        sink = self._sink_keys if is_key else self._sink_values
        compressed = self._compressed_keys if is_key else self._compressed_values
        quantizer = self._k_quantizer if is_key else self._v_quantizer

        parts = []
        if sink is not None:
            parts.append(sink)
        for entry in compressed:
            parts.append(self._dequant_entry(quantizer, entry))

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim))

        result = mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
        if self._input_dtype is not None and result.dtype != self._input_dtype:
            result = result.astype(self._input_dtype)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dequant_entry(quantizer, entry):
        """Dequantize a single compressed entry (handles QJL)."""
        indices, norms = entry[0], entry[1]
        qjl_data = entry[2] if len(entry) > 2 else None
        return quantizer.dequantize(indices, norms, qjl_data)

    @property
    def state(self):
        if self.offset == 0:
            return None, None
        import mlx.core as mx
        return self._assemble(True, mx), self._assemble(False, mx)

    @property
    def is_empty(self) -> bool:
        return self.offset == 0


def _detect_head_dim_and_kv_heads(model) -> tuple[int, int]:
    """Infer head_dim and n_kv_heads from an mlx-lm model.

    Handles both simple models (model.args) and composite models like
    Gemma 4 (model.language_model.args / model.text_model.args).
    Falls back to inspecting the first attention layer if args are nested.
    """
    # Try direct model.args first (Qwen, Llama, etc.)
    for source in [model, getattr(model, "language_model", None), getattr(model, "text_model", None)]:
        if source is None or not hasattr(source, "args"):
            continue
        args = source.args
        head_dim = getattr(args, "head_dim", None)
        if head_dim is None and hasattr(args, "hidden_size") and hasattr(args, "num_attention_heads"):
            head_dim = args.hidden_size // args.num_attention_heads
        n_kv_heads = getattr(args, "num_key_value_heads", getattr(args, "num_attention_heads", 1))
        if head_dim:
            return head_dim, n_kv_heads

    # Fallback: inspect first attention layer
    layers = getattr(model, "layers", [])
    if layers and hasattr(layers[0], "self_attn"):
        attn = layers[0].self_attn
        head_dim = getattr(attn, "head_dim", None)
        n_kv_heads = getattr(attn, "n_kv_heads", getattr(attn, "num_key_value_heads", 1))
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
