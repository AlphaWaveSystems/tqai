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

        # Compressed storage (only used by "full" strategy for reconstruction)
        self._compressed_keys: list[tuple[Any, ...]] = []
        self._compressed_values: list[tuple[Any, ...]] = []

        # Incremental dequantized buffer (strategies: incremental, residual)
        self._k_buffer: Any | None = None  # [1, H, seq, D] in input dtype
        self._v_buffer: Any | None = None

        # Recent uncompressed window (strategy: residual)
        self._recent_keys: Any | None = None  # [1, H, <=R, D] full precision
        self._recent_values: Any | None = None

        # Compressed KV buffer (strategy: compressed)
        # Stores indices + norms in GPU memory; never materialises float32 history.
        # Shape: [1, n_kv_heads, T_stored, head_dim] uint8 / [1, n_kv_heads, T_stored, 1] fp16
        self._k_indices: Any | None = None
        self._k_norms: Any | None = None
        self._v_indices: Any | None = None
        self._v_norms: Any | None = None
        # When True, update_and_fetch skips assembling a float32 K/V buffer during
        # decode — the patched SDPA calls compute_fused_attention instead.
        # Self-activates for compressed strategy; also set by patch_fused_attention.
        self._skip_assemble: bool = (self._strategy == "compressed")

        self.offset: int = 0
        self._input_dtype = None

        # Pipeline middleware (v0.4)
        self._k_pipeline = None
        self._v_pipeline = None
        if config.pipeline is not None:
            from tqai.pipeline import build_pipeline

            self._k_pipeline = build_pipeline(config, quantizer=self._k_quantizer)
            self._v_pipeline = build_pipeline(config, quantizer=self._v_quantizer)

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
            elif self._strategy == "compressed":
                self._update_compressed(keys, values, mx)
            else:
                self._update_full(keys, values)

        self.offset += new_seq

        # For the compressed + fused path: the patched SDPA calls
        # compute_fused_attention directly, so we skip assembling float32 K/V.
        # During prefill (new_seq > 1) or without the patch, fall through to
        # the normal assembly path so the model gets valid float32 tensors.
        if self._strategy == "compressed" and self._skip_assemble and new_seq == 1:
            # Return zero-shaped placeholder — patched SDPA ignores it.
            dummy = mx.zeros((1, self.n_kv_heads, self.offset, self.head_dim))
            self.keys = dummy
            self.values = dummy
            return dummy, dummy

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
        """Quantize, dequantize new token, append to buffer."""
        if self._k_pipeline is not None and self._k_pipeline.has_middleware:
            k_compressed = self._k_pipeline.compress(keys, layer_idx=0)
            v_compressed = self._v_pipeline.compress(values, layer_idx=0)
            k_dequant = self._k_pipeline.decompress(k_compressed, layer_idx=0)
            v_dequant = self._v_pipeline.decompress(v_compressed, layer_idx=0)
        else:
            k_entry = self._k_quantizer.quantize(keys)
            v_entry = self._v_quantizer.quantize(values)
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
            if self._k_pipeline is not None and self._k_pipeline.has_middleware:
                k_compressed = self._k_pipeline.compress(old_k, layer_idx=0)
                v_compressed = self._v_pipeline.compress(old_v, layer_idx=0)
                k_dequant = self._k_pipeline.decompress(k_compressed, layer_idx=0)
                v_dequant = self._v_pipeline.decompress(v_compressed, layer_idx=0)
            else:
                k_entry = self._k_quantizer.quantize(old_k)
                v_entry = self._v_quantizer.quantize(old_v)
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

        if self._strategy == "compressed":
            return self._reconstruct_compressed(is_key, mx)

        buffer = self._k_buffer if is_key else self._v_buffer
        recent = None
        if self._strategy == "residual":
            recent = self._recent_keys if is_key else self._recent_values

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

    # ------------------------------------------------------------------
    # Strategy: compressed — keeps indices+norms in GPU memory
    # ------------------------------------------------------------------

    def _update_compressed(self, keys, values, mx):
        """Quantize new tokens, append to compressed (uint8+fp16) buffer.

        Never stores a float32 history buffer.  The full K/V history is
        reconstructed on demand by ``_reconstruct_compressed`` (fallback)
        or consumed directly by ``compute_fused_attention`` (fast path).
        """
        k_idx, k_nrm = self._k_quantizer.quantize(keys)   # uint8, fp16
        v_idx, v_nrm = self._v_quantizer.quantize(values)

        if self._k_indices is None:
            self._k_indices = k_idx
            self._k_norms   = k_nrm
            self._v_indices = v_idx
            self._v_norms   = v_nrm
        else:
            self._k_indices = mx.concatenate([self._k_indices, k_idx], axis=2)
            self._k_norms   = mx.concatenate([self._k_norms,   k_nrm], axis=2)
            self._v_indices = mx.concatenate([self._v_indices, v_idx], axis=2)
            self._v_norms   = mx.concatenate([self._v_norms,   v_nrm], axis=2)

    def _reconstruct_compressed(self, is_key: bool, mx):
        """Fallback: dequantize the full compressed buffer → float32.

        Used when ``compute_fused_attention`` is not active (e.g. the SDPA
        patch is not installed, or during prefill with T_q > 1).
        """
        sink = self._sink_keys if is_key else self._sink_values
        indices = self._k_indices if is_key else self._v_indices
        norms   = self._k_norms   if is_key else self._v_norms
        quantizer = self._k_quantizer if is_key else self._v_quantizer

        parts = []
        if sink is not None:
            parts.append(sink)
        if indices is not None:
            dequant = quantizer.dequantize(indices, norms)
            parts.append(dequant)

        if not parts:
            return mx.zeros((1, self.n_kv_heads, 0, self.head_dim))

        result = mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
        if self._input_dtype is not None and result.dtype != self._input_dtype:
            result = result.astype(self._input_dtype)
        return result

    def compute_fused_attention(
        self,
        queries: Any,
        scale: float,
        causal: bool = True,
    ) -> Any:
        """Run fused dequant-attention over the compressed KV cache.

        Computes multi-head attention without materialising float32 K or V
        buffers.  Handles GQA, sink tokens (full-precision chunks), and
        compressed tokens (fused Metal kernels).

        For B=1, T_q=1 (decode), uses the batched v0.6 path — 2 total Metal
        dispatches regardless of head count.  Falls back to the per-head loop
        for prefill (T_q > 1) or multi-batch.

        Only valid when ``_strategy == "compressed"`` and Metal is available.
        Called by the patched SDPA in ``patch_fused_attention`` during decode.

        Args:
            queries: ``[B, n_q_heads, T_q, D]`` query tensor.
            scale: Attention scale (typically ``1/sqrt(D)``).
            causal: Apply causal mask when T_q > 1 (prefill).

        Returns:
            Attention output ``[B, n_q_heads, T_q, D]`` float32.
        """
        import mlx.core as mx

        B, n_q_heads, T_q, D = queries.shape

        # v0.6 fast path: batched multi-head Metal kernels (2 dispatches total)
        if B == 1 and T_q == 1:
            from tqai.attention_fused import batched_fused_polar_decode_v2

            R_k = self._k_quantizer._rotation
            R_v = self._v_quantizer._rotation
            centroids_k = self._k_quantizer._centroids
            centroids_v = self._v_quantizer._centroids

            return batched_fused_polar_decode_v2(
                queries,
                self._k_indices, self._k_norms,
                self._v_indices, self._v_norms,
                R_k, R_v,
                centroids_k, centroids_v,
                scale,
                self._sink_keys, self._sink_values,
            )

        # Fallback: per-head loop (prefill, multi-batch)
        return self._compute_fused_attention_loop(queries, scale, causal)

    def _compute_fused_attention_loop(
        self,
        queries: Any,
        scale: float,
        causal: bool = True,
    ) -> Any:
        """Per-head fused attention loop (fallback for T_q > 1 or B > 1)."""
        import mlx.core as mx
        from tqai.kernels import metal_aggregate_values, metal_score_keys

        B, n_q_heads, T_q, D = queries.shape
        n_kv_heads = self.n_kv_heads
        repeats = n_q_heads // n_kv_heads

        R_k = self._k_quantizer._rotation          # [D, D] float32
        R_v = self._v_quantizer._rotation
        centroids_k = self._k_quantizer._centroids  # [n_levels] float32
        centroids_v = self._v_quantizer._centroids

        outputs = []
        for b in range(B):
            for h_q in range(n_q_heads):
                h_kv = h_q // repeats
                for t in range(T_q):
                    q_vec = queries[b, h_q, t, :].astype(mx.float32)  # (D,)

                    # --- Scoring ---
                    parts_k = []
                    parts_v_float = []  # float32 V slices from sinks
                    v_comp_idx = None
                    v_comp_nrm = None

                    # Sink tokens: full-precision dot products
                    if self._sink_keys is not None:
                        k_sink = self._sink_keys[b, h_kv, :, :]    # (T_s, D)
                        v_sink = self._sink_values[b, h_kv, :, :]
                        T_s = k_sink.shape[0]

                        if causal and T_q > 1:
                            kv_end = self._sink_keys.shape[2] - T_q + t + 1
                            k_sink = k_sink[:kv_end]
                            v_sink = v_sink[:kv_end]
                            T_s = k_sink.shape[0]

                        if T_s > 0:
                            scores_s = (k_sink.astype(mx.float32) @ q_vec) * scale
                            parts_k.append(scores_s)
                            parts_v_float.append(v_sink.astype(mx.float32))

                    # Compressed tokens: fused Metal gather-dot
                    if self._k_indices is not None:
                        k_idx = self._k_indices[b, h_kv, :, :]   # (T_c, D)
                        k_nrm = mx.reshape(
                            self._k_norms[b, h_kv, :, :], (-1,)
                        )                                           # (T_c,)
                        T_c = k_idx.shape[0]

                        if causal and T_q > 1:
                            T_sink_total = (
                                self._sink_keys.shape[2] if self._sink_keys is not None else 0
                            )
                            kv_end = self.offset - T_q + t + 1 - T_sink_total
                            k_idx = k_idx[:kv_end]
                            k_nrm = k_nrm[:kv_end]
                            T_c = k_idx.shape[0]

                        if T_c > 0:
                            q_rotated = R_k @ q_vec  # (D,)
                            scores_c = metal_score_keys(
                                q_rotated, k_idx, k_nrm, centroids_k
                            ) * scale
                            parts_k.append(scores_c)
                            v_comp_idx = self._v_indices[b, h_kv, :T_c, :]
                            v_comp_nrm = mx.reshape(
                                self._v_norms[b, h_kv, :T_c, :], (-1,)
                            )

                    if not parts_k:
                        outputs.append(mx.zeros((D,), dtype=mx.float32))
                        continue

                    all_scores = mx.concatenate(parts_k) if len(parts_k) > 1 else parts_k[0]
                    weights = mx.softmax(all_scores, axis=-1)

                    # --- Value aggregation ---
                    output = mx.zeros((D,), dtype=mx.float32)
                    idx = 0

                    # Sink value weighted sum (full-precision)
                    for v_f in parts_v_float:
                        T_chunk = v_f.shape[0]
                        w_chunk = weights[idx: idx + T_chunk]
                        output = output + (w_chunk[:, None] * v_f).sum(axis=0)
                        idx += T_chunk

                    # Compressed value weighted sum (fused Metal kernel)
                    if v_comp_idx is not None:
                        T_c = v_comp_idx.shape[0]
                        w_comp = weights[idx: idx + T_c]
                        out_rotated = metal_aggregate_values(
                            w_comp, v_comp_idx, v_comp_nrm, centroids_v
                        )
                        output = output + R_v.T @ out_rotated

                    outputs.append(output)

        stacked = mx.stack(outputs, axis=0)           # (B*n_q*T_q, D)
        return stacked.reshape(B, n_q_heads, T_q, D)

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
    candidates = [
        model,
        getattr(model, "language_model", None),
        getattr(model, "text_model", None),
    ]
    for source in candidates:
        if source is None or not hasattr(source, "args"):
            continue
        args = source.args
        head_dim = getattr(args, "head_dim", None)
        has_dims = hasattr(args, "hidden_size") and hasattr(args, "num_attention_heads")
        if head_dim is None and has_dims:
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
