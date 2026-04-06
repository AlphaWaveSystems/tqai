"""Protocol definitions for the tqai composable pipeline.

Four extension points — Scorer, CompressionStrategy, Monitor, ModelAdapter —
each defined as a :class:`~typing.Protocol` so that new papers become additive
modules without modifying core code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ScoredEntry:
    """A KV entry annotated with compression metadata."""

    data: Any  # Original tensor
    score: float  # 0.0 = evict, 1.0 = critical
    tier: int  # Compression tier (0-3)
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """Scores KV entries to determine compression priority.

    One module per paper/approach.  Called before quantization to decide
    *how* to compress each entry.
    """

    name: str

    def score(
        self,
        x: Any,
        layer_idx: int,
        step: int | None = None,
        context: dict | None = None,
    ) -> list[ScoredEntry]: ...

    def reset(self) -> None: ...


@runtime_checkable
class CompressionStrategy(Protocol):
    """Determines how to compress an entry based on its score.

    The core ``PolarQuantizer`` handles the actual quantization.
    Strategies wrap it to implement tiered, delta, or skip logic.
    """

    name: str

    def compress(
        self,
        entry: Any,
        quantizer: Any,
        prev_state: dict | None = None,
    ) -> tuple[Any, dict]: ...

    def decompress(
        self,
        compressed: Any,
        quantizer: Any,
        state: dict | None = None,
    ) -> Any: ...


@runtime_checkable
class Monitor(Protocol):
    """Observes the pipeline and can adjust parameters at runtime.

    Called after each update with the latest attention state.
    Can modify scorer thresholds or strategy parameters dynamically.
    """

    name: str

    def observe(
        self,
        layer_idx: int,
        step: int,
        attention_state: dict,
    ) -> dict | None: ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Adapts tqai to different model architectures.

    Handles the specifics of finding attention modules, hooking into the
    forward pass, and managing architecture-specific caching patterns.
    """

    name: str

    def detect(self, model: Any) -> bool: ...

    def get_attention_modules(self, model: Any) -> Any: ...

    def get_head_info(self, model: Any) -> dict: ...

    def patch(self, model: Any, pipeline: Any, config: Any) -> Any: ...

    def unpatch(self, model: Any) -> None: ...
