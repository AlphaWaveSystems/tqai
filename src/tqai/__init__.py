"""tqai — TurboQuant KV cache and forward-pass compression for local LLM inference."""

from __future__ import annotations

__version__ = "0.3.1"

from tqai.config import TurboQuantConfig


def patch(
    model,
    bits_k: int = 4,
    bits_v: int = 2,
    sink_tokens: int = 0,
    backend: str | None = None,
    device: str | None = None,
    config_path: str | None = None,
    # Forward-pass compression (v0.2)
    compress_hidden: bool = False,
    bits_hidden: int = 8,
    compress_ffn: bool = False,
    bits_ffn: int = 8,
    compress_attn_logits: bool = False,
    bits_attn: int = 8,
    # QJL Stage 2 (opt-in, research use)
    use_qjl: bool = False,
    qjl_sketch_size: int = 64,
    # Cache strategy (v0.3.1)
    cache_strategy: str = "auto",
    residual_window: int = 128,
    # Pipeline composition (v0.4)
    pipeline: dict | None = None,
    # Chunked attention (v0.4.0)
    chunk_attention: bool = False,
    attention_chunk_size: int = 4096,
):
    """Enable TurboQuant compression on a model.

    KV cache compression works with both PyTorch (HuggingFace) and MLX backends.
    Forward-pass activation compression (``compress_hidden``, ``compress_ffn``,
    ``compress_attn_logits``) is currently supported for PyTorch only.

    Args:
        model: HuggingFace model or mlx-lm model.
        bits_k: Bits per key coordinate (default 4).
        bits_v: Bits per value coordinate (default 2).
        sink_tokens: Number of initial tokens to keep uncompressed.
        backend: ``'torch'``, ``'mlx'``, or ``None`` (auto-detect).
        device: PyTorch device string (ignored for MLX).
        config_path: Path to a pre-converted tqai config directory
            (from ``tqai convert``). When provided, ``bits_k`` and ``bits_v``
            are read from the saved config.
        compress_hidden: Compress residual stream (hidden states) entering
            each attention and FFN block. PyTorch only.
        bits_hidden: Bits for hidden state compression (default 8).
        compress_ffn: Compress hidden states entering FFN/MLP blocks. PyTorch only.
        bits_ffn: Bits for FFN input compression (default 8).
        compress_attn_logits: Reserved for future use (attention logit compression).
        bits_attn: Bits for attention logit compression (default 8).
        use_qjl: If True, enable QJL Stage 2 residual sketch (default False).
            Adds a 1-bit JL sketch to each KV token for inner-product bias correction.
            NOTE: degrades softmax attention on average; use for research only.
        qjl_sketch_size: Number of 1-bit JL projections (default 64).
        cache_strategy: Cache reconstruction strategy. ``'auto'`` (default)
            selects ``'incremental'`` which maintains a running dequantized
            buffer for O(1) per-token cost. ``'residual'`` keeps the last
            ``residual_window`` tokens uncompressed for better quality.
            ``'full'`` uses the original O(n) full-reconstruction path.
        residual_window: Number of recent tokens to keep uncompressed when
            using the ``'residual'`` cache strategy (default 128).

    Returns:
        For HuggingFace: a ``TurboQuantDynamicCache`` to pass as ``past_key_values``.
        For mlx-lm: ``None`` (model is patched in-place).
    """
    from tqai._patch import _patch

    if config_path is not None:
        from tqai.convert import load_converted

        converted = load_converted(config_path)
        cfg = converted["config"]
        bits_k = cfg["bits_k"]
        bits_v = cfg["bits_v"]

    config = TurboQuantConfig(
        bits_k=bits_k,
        bits_v=bits_v,
        seed=42,
        sink_tokens=sink_tokens,
        backend=backend,
        device=device,
        config_path=config_path,
        compress_hidden=compress_hidden,
        bits_hidden=bits_hidden,
        compress_ffn=compress_ffn,
        bits_ffn=bits_ffn,
        compress_attn_logits=compress_attn_logits,
        bits_attn=bits_attn,
        use_qjl=use_qjl,
        qjl_sketch_size=qjl_sketch_size,
        cache_strategy=cache_strategy,
        residual_window=residual_window,
        pipeline=pipeline,
        chunk_attention=chunk_attention,
        attention_chunk_size=attention_chunk_size,
    )
    return _patch(model, config)


def unpatch(model):
    """Remove TurboQuant compression hooks and restore original cache behaviour."""
    from tqai._patch import _unpatch

    _unpatch(model)


__all__ = ["TurboQuantConfig", "patch", "unpatch", "__version__"]
