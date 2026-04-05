"""HuggingFace DynamicCache wrapper with TurboQuant compression."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from tqai.backend import get_backend
from tqai.config import TurboQuantConfig
from tqai.quantizer import PolarQuantizer


class TurboQuantDynamicCache(DynamicCache):
    """Drop-in replacement for ``DynamicCache`` that compresses KV states.

    Inherits from ``DynamicCache`` for full interface compatibility.
    Overrides ``update`` to compress incoming KV states and reconstruct
    on read.
    """

    def __init__(self, config: TurboQuantConfig):
        super().__init__()
        self.tq_config = config
        self._ops = get_backend(config.backend, config.device)
        self._key_quantizers: dict[int, PolarQuantizer] = {}
        self._value_quantizers: dict[int, PolarQuantizer] = {}
        self._compressed_keys: dict[int, list[tuple[Any, Any]]] = {}
        self._compressed_values: dict[int, list[tuple[Any, Any]]] = {}
        self._sink_keys: dict[int, torch.Tensor | None] = {}
        self._sink_values: dict[int, torch.Tensor | None] = {}
        self._tokens_seen: dict[int, int] = {}
        self._dtype: dict[int, torch.dtype] = {}

    def _get_quantizer(self, layer_idx: int, head_dim: int, is_key: bool) -> PolarQuantizer:
        store = self._key_quantizers if is_key else self._value_quantizers
        if layer_idx not in store:
            bits = self.tq_config.bits_k if is_key else self.tq_config.bits_v
            store[layer_idx] = PolarQuantizer(
                head_dim=head_dim,
                bits=bits,
                seed=self.tq_config.seed + layer_idx + (0 if is_key else 10000),
                ops=self._ops,
                pre_rotated=self.tq_config.pre_rotated,
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

        # Handle sink tokens (keep in FP16)
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

        # Compress remaining tokens
        if key_states.shape[2] > 0:
            k_quant = self._get_quantizer(layer_idx, head_dim, is_key=True)
            v_quant = self._get_quantizer(layer_idx, head_dim, is_key=False)
            k_result = k_quant.quantize(key_states)
            v_result = v_quant.quantize(value_states)
            self._compressed_keys[layer_idx].append(k_result)
            self._compressed_values[layer_idx].append(v_result)

        self._tokens_seen[layer_idx] += new_tokens

        all_keys = self._reconstruct(layer_idx, is_key=True)
        all_values = self._reconstruct(layer_idx, is_key=False)
        return all_keys, all_values

    def _reconstruct(self, layer_idx: int, is_key: bool) -> torch.Tensor:
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
                indices, norms = entry[0], entry[1]
                qjl_data = entry[2] if len(entry) > 2 else None
                parts.append(quant.dequantize(indices, norms, qjl_data))

        if not parts:
            return torch.empty(0)

        result = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        target_dtype = self._dtype.get(layer_idx)
        if target_dtype is not None and result.dtype != target_dtype:
            result = result.to(target_dtype)
        return result

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._tokens_seen.get(layer_idx, 0)
