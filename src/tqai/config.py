from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache and forward-pass compression."""

    # KV cache compression (v0.1)
    bits_k: int = 4
    bits_v: int = 2
    seed: int = 42
    sink_tokens: int = 0
    backend: str | None = None
    device: str | None = None
    config_path: str | None = None

    # Forward-pass activation compression (v0.2)
    compress_hidden: bool = False
    bits_hidden: int = 8
    compress_ffn: bool = False
    bits_ffn: int = 8
    compress_attn_logits: bool = False
    bits_attn: int = 8

    # QJL Stage 2 (v0.2, opt-in)
    use_qjl: bool = False
    qjl_sketch_size: int = 64

    # Cache strategy (v0.3.1)
    cache_strategy: str = "auto"  # "auto"|"incremental"|"residual"|"full"
    residual_window: int = 128  # tokens to keep uncompressed (residual strategy)

    @property
    def has_forward_compression(self) -> bool:
        """True if any forward-pass compression is enabled."""
        return self.compress_hidden or self.compress_ffn or self.compress_attn_logits
