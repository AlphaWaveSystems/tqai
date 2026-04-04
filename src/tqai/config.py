from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""

    bits_k: int = 4
    bits_v: int = 2
    seed: int = 42
    sink_tokens: int = 0
    backend: str | None = None
    device: str | None = None
    config_path: str | None = None
