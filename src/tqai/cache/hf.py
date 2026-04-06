"""HuggingFace DynamicCache wrapper with TurboQuant compression.

Implements incremental and residual cache strategies to avoid O(n²)
full-reconstruction overhead.  See :class:`TurboQuantDynamicCache` for details.

References:
    - TurboQuant: arXiv:2504.19874
    - KIVI (residual buffer strategy): arXiv:2402.02750
    - KVQuant (fused dequant-attention motivation): arXiv:2401.18079
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.quantizer import PolarQuantizer


class TurboQuantDynamicCache(DynamicCache):
    """Drop-in replacement for ``DynamicCache`` that compresses KV states.

    Supports three cache strategies (configured via ``TurboQuantConfig``):

    - **incremental** (default): Maintains a running dequantized buffer per
      layer.  Only the new token is dequantized each step — O(1) per token.
    - **residual**: Keeps the last ``residual_window`` tokens uncompressed
      (KIVI-style, arXiv:2402.02750).  Older tokens are compressed then
      dequantized into an incremental buffer.
    - **full**: Legacy path — dequantizes the entire compressed history each
      token (O(n) per token, O(n²) total).
    """

    def __init__(self, config: TurboQuantConfig):
        super().__init__()
        self.tq_config = config
        self._ops = get_backend(config.backend, config.device)

        strategy = config.cache_strategy
        if strategy == "auto":
            strategy = "incremental"
        self._strategy = strategy

        self._key_quantizers: dict[int, PolarQuantizer] = {}
        self._value_quantizers: dict[int, PolarQuantizer] = {}
        self._compressed_keys: dict[int, list[tuple[Any, ...]]] = {}
        self._compressed_values: dict[int, list[tuple[Any, ...]]] = {}
        self._sink_keys: dict[int, torch.Tensor | None] = {}
        self._sink_values: dict[int, torch.Tensor | None] = {}
        self._tokens_seen: dict[int, int] = {}
        self._dtype: dict[int, torch.dtype] = {}

        # Incremental buffers (strategies: incremental, residual)
        self._k_buffers: dict[int, torch.Tensor | None] = {}
        self._v_buffers: dict[int, torch.Tensor | None] = {}

        # Recent uncompressed windows (strategy: residual)
        self._recent_keys: dict[int, torch.Tensor | None] = {}
        self._recent_values: dict[int, torch.Tensor | None] = {}

        # Pipeline middleware (v0.4)
        self._pipeline_k: dict[int, Any] = {}
        self._pipeline_v: dict[int, Any] = {}

    def _get_pipeline(self, layer_idx: int, head_dim: int, is_key: bool):
        """Lazily build a CompressionPipeline for this layer (if configured)."""
        store = self._pipeline_k if is_key else self._pipeline_v
        if layer_idx not in store:
            if self.tq_config.pipeline is not None:
                from tqai.pipeline import build_pipeline

                quantizer = self._get_quantizer(layer_idx, head_dim, is_key)
                store[layer_idx] = build_pipeline(
                    self.tq_config, quantizer=quantizer
                )
            else:
                store[layer_idx] = None
        return store[layer_idx]

    def _get_quantizer(self, layer_idx: int, head_dim: int, is_key: bool) -> PolarQuantizer:
        store = self._key_quantizers if is_key else self._value_quantizers
        if layer_idx not in store:
            bits = self.tq_config.bits_k if is_key else self.tq_config.bits_v
            store[layer_idx] = PolarQuantizer(
                head_dim=head_dim,
                bits=bits,
                seed=self.tq_config.seed + layer_idx + (0 if is_key else 10000),
                ops=self._ops,
                use_qjl=self.tq_config.use_qjl,
                qjl_sketch_size=self.tq_config.qjl_sketch_size,
            )
        return store[layer_idx]

    def _init_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._tokens_seen:
            self._tokens_seen[layer_idx] = 0
            self._compressed_keys[layer_idx] = []
            self._compressed_values[layer_idx] = []
            self._sink_keys[layer_idx] = None
            self._sink_values[layer_idx] = None
            self._k_buffers[layer_idx] = None
            self._v_buffers[layer_idx] = None
            self._recent_keys[layer_idx] = None
            self._recent_values[layer_idx] = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress incoming KV states and return full dequantized history."""
        self._init_layer(layer_idx)
        head_dim = key_states.shape[-1]
        new_tokens = key_states.shape[2]
        if layer_idx not in self._dtype:
            self._dtype[layer_idx] = key_states.dtype
        sink = self.tq_config.sink_tokens
        tokens_so_far = self._tokens_seen[layer_idx]

        # Handle sink tokens
        if tokens_so_far < sink:
            sink_end = min(new_tokens, sink - tokens_so_far)
            sink_k = key_states[:, :, :sink_end, :]
            sink_v = value_states[:, :, :sink_end, :]
            if self._sink_keys[layer_idx] is None:
                self._sink_keys[layer_idx] = sink_k
                self._sink_values[layer_idx] = sink_v
            else:
                self._sink_keys[layer_idx] = torch.cat(
                    [self._sink_keys[layer_idx], sink_k], dim=2
                )
                self._sink_values[layer_idx] = torch.cat(
                    [self._sink_values[layer_idx], sink_v], dim=2
                )
            key_states = key_states[:, :, sink_end:, :]
            value_states = value_states[:, :, sink_end:, :]

        # Compress remaining tokens via strategy
        if key_states.shape[2] > 0:
            if self._strategy == "residual":
                self._update_residual(layer_idx, key_states, value_states, head_dim)
            elif self._strategy == "incremental":
                self._update_incremental(layer_idx, key_states, value_states, head_dim)
            else:
                self._update_full(layer_idx, key_states, value_states, head_dim)

        self._tokens_seen[layer_idx] += new_tokens

        all_keys = self._assemble(layer_idx, is_key=True)
        all_values = self._assemble(layer_idx, is_key=False)
        return all_keys, all_values

    # ------------------------------------------------------------------
    # Strategy: incremental
    # ------------------------------------------------------------------

    def _update_incremental(self, layer_idx, key_states, value_states, head_dim):
        k_pipe = self._get_pipeline(layer_idx, head_dim, is_key=True)
        v_pipe = self._get_pipeline(layer_idx, head_dim, is_key=False)

        if k_pipe is not None and k_pipe.has_middleware:
            k_compressed = k_pipe.compress(key_states, layer_idx)
            v_compressed = v_pipe.compress(value_states, layer_idx)
            k_dequant = k_pipe.decompress(k_compressed, layer_idx)
            v_dequant = v_pipe.decompress(v_compressed, layer_idx)
        else:
            k_quant = self._get_quantizer(layer_idx, head_dim, is_key=True)
            v_quant = self._get_quantizer(layer_idx, head_dim, is_key=False)
            k_entry = k_quant.quantize(key_states)
            v_entry = v_quant.quantize(value_states)
            k_dequant = self._dequant_entry(k_quant, k_entry)
            v_dequant = self._dequant_entry(v_quant, v_entry)

        target_dtype = self._dtype.get(layer_idx)
        if target_dtype is not None:
            k_dequant = k_dequant.to(target_dtype)
            v_dequant = v_dequant.to(target_dtype)

        if self._k_buffers[layer_idx] is None:
            self._k_buffers[layer_idx] = k_dequant
            self._v_buffers[layer_idx] = v_dequant
        else:
            self._k_buffers[layer_idx] = torch.cat(
                [self._k_buffers[layer_idx], k_dequant], dim=2
            )
            self._v_buffers[layer_idx] = torch.cat(
                [self._v_buffers[layer_idx], v_dequant], dim=2
            )

    # ------------------------------------------------------------------
    # Strategy: residual
    # ------------------------------------------------------------------

    def _update_residual(self, layer_idx, key_states, value_states, head_dim):
        R = self.tq_config.residual_window

        if self._recent_keys[layer_idx] is None:
            self._recent_keys[layer_idx] = key_states
            self._recent_values[layer_idx] = value_states
        else:
            self._recent_keys[layer_idx] = torch.cat(
                [self._recent_keys[layer_idx], key_states], dim=2
            )
            self._recent_values[layer_idx] = torch.cat(
                [self._recent_values[layer_idx], value_states], dim=2
            )

        while self._recent_keys[layer_idx].shape[2] > R:
            overflow = self._recent_keys[layer_idx].shape[2] - R
            old_k = self._recent_keys[layer_idx][:, :, :overflow, :]
            old_v = self._recent_values[layer_idx][:, :, :overflow, :]
            self._recent_keys[layer_idx] = self._recent_keys[layer_idx][:, :, overflow:, :]
            self._recent_values[layer_idx] = self._recent_values[layer_idx][:, :, overflow:, :]

            k_pipe = self._get_pipeline(layer_idx, head_dim, is_key=True)
            v_pipe = self._get_pipeline(layer_idx, head_dim, is_key=False)

            if k_pipe is not None and k_pipe.has_middleware:
                k_compressed = k_pipe.compress(old_k, layer_idx)
                v_compressed = v_pipe.compress(old_v, layer_idx)
                k_dequant = k_pipe.decompress(k_compressed, layer_idx)
                v_dequant = v_pipe.decompress(v_compressed, layer_idx)
            else:
                k_quant = self._get_quantizer(layer_idx, head_dim, is_key=True)
                v_quant = self._get_quantizer(layer_idx, head_dim, is_key=False)
                k_entry = k_quant.quantize(old_k)
                v_entry = v_quant.quantize(old_v)
                k_dequant = self._dequant_entry(k_quant, k_entry)
                v_dequant = self._dequant_entry(v_quant, v_entry)

            target_dtype = self._dtype.get(layer_idx)
            if target_dtype is not None:
                k_dequant = k_dequant.to(target_dtype)
                v_dequant = v_dequant.to(target_dtype)

            if self._k_buffers[layer_idx] is None:
                self._k_buffers[layer_idx] = k_dequant
                self._v_buffers[layer_idx] = v_dequant
            else:
                self._k_buffers[layer_idx] = torch.cat(
                    [self._k_buffers[layer_idx], k_dequant], dim=2
                )
                self._v_buffers[layer_idx] = torch.cat(
                    [self._v_buffers[layer_idx], v_dequant], dim=2
                )

    # ------------------------------------------------------------------
    # Strategy: full (legacy)
    # ------------------------------------------------------------------

    def _update_full(self, layer_idx, key_states, value_states, head_dim):
        k_quant = self._get_quantizer(layer_idx, head_dim, is_key=True)
        v_quant = self._get_quantizer(layer_idx, head_dim, is_key=False)
        self._compressed_keys[layer_idx].append(k_quant.quantize(key_states))
        self._compressed_values[layer_idx].append(v_quant.quantize(value_states))

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, layer_idx: int, is_key: bool) -> torch.Tensor:
        sink = self._sink_keys[layer_idx] if is_key else self._sink_values[layer_idx]

        if self._strategy == "full":
            return self._reconstruct_full(layer_idx, is_key)

        buffer = self._k_buffers[layer_idx] if is_key else self._v_buffers[layer_idx]
        recent = None
        if self._strategy == "residual":
            recent = self._recent_keys[layer_idx] if is_key else self._recent_values[layer_idx]

        parts: list[torch.Tensor] = []
        if sink is not None:
            parts.append(sink)
        if buffer is not None:
            parts.append(buffer)
        if recent is not None:
            parts.append(recent)

        if not parts:
            return torch.empty(0)

        result = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        target_dtype = self._dtype.get(layer_idx)
        if target_dtype is not None and result.dtype != target_dtype:
            result = result.to(target_dtype)
        return result

    def _reconstruct_full(self, layer_idx: int, is_key: bool) -> torch.Tensor:
        """Legacy full reconstruction — dequantizes entire history."""
        sink = self._sink_keys[layer_idx] if is_key else self._sink_values[layer_idx]
        compressed = (
            self._compressed_keys[layer_idx]
            if is_key
            else self._compressed_values[layer_idx]
        )

        parts: list[torch.Tensor] = []
        if sink is not None:
            parts.append(sink)

        if compressed:
            head_dim = compressed[0][0].shape[-1]
            quant = self._get_quantizer(layer_idx, head_dim, is_key)
            for entry in compressed:
                parts.append(self._dequant_entry(quant, entry))

        if not parts:
            return torch.empty(0)

        result = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        target_dtype = self._dtype.get(layer_idx)
        if target_dtype is not None and result.dtype != target_dtype:
            result = result.to(target_dtype)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dequant_entry(quantizer, entry):
        indices, norms = entry[0], entry[1]
        qjl_data = entry[2] if len(entry) > 2 else None
        return quantizer.dequantize(indices, norms, qjl_data)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._tokens_seen.get(layer_idx, 0)
