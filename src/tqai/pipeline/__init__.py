"""Composable middleware pipeline for tqai compression.

Public API::

    from tqai.pipeline import build_pipeline, list_available

    pipeline = build_pipeline(config)          # from TurboQuantConfig
    compressed = pipeline.compress(x, layer_idx=0)
    x_hat = pipeline.decompress(compressed)
"""

from __future__ import annotations

from typing import Any

from tqai.config import TurboQuantConfig
from tqai.pipeline.base import (
    CompressionStrategy,
    ModelAdapter,
    Monitor,
    ScoredEntry,
    Scorer,
)
from tqai.pipeline.registry import (
    get_adapter,
    get_monitor,
    get_scorer,
    get_strategy,
    list_available,
    register_adapter,
    register_monitor,
    register_scorer,
    register_strategy,
)
from tqai.pipeline.runner import CompressionPipeline
from tqai.quantizer import PolarQuantizer


def build_pipeline(
    config: TurboQuantConfig,
    quantizer: PolarQuantizer,
    quantizer_low: PolarQuantizer | None = None,
) -> CompressionPipeline:
    """Build a :class:`CompressionPipeline` from ``config.pipeline``.

    When ``config.pipeline`` is ``None``, returns a bare pipeline that
    delegates directly to *quantizer* (v0.3.1-compatible).

    Args:
        config: Full tqai configuration.
        quantizer: Primary ``PolarQuantizer`` for this layer.
        quantizer_low: Optional lower-bit quantizer (used by tiered strategy).

    Returns:
        A ready-to-use :class:`CompressionPipeline`.
    """
    pipe_cfg: dict[str, Any] | None = config.pipeline

    if pipe_cfg is None:
        return CompressionPipeline(quantizer=quantizer)

    scorer = None
    if "scorer" in pipe_cfg:
        scorer = get_scorer(
            pipe_cfg["scorer"], **pipe_cfg.get("scorer_kwargs", {})
        )

    strategy = None
    if "strategy" in pipe_cfg:
        strategy = get_strategy(
            pipe_cfg["strategy"], **pipe_cfg.get("strategy_kwargs", {})
        )

    monitor = None
    if "monitor" in pipe_cfg:
        monitor = get_monitor(
            pipe_cfg["monitor"], **pipe_cfg.get("monitor_kwargs", {})
        )

    pipe = CompressionPipeline(
        quantizer=quantizer,
        quantizer_low=quantizer_low,
        scorer=scorer,
        strategy=strategy,
        monitor=monitor,
    )

    # Inject skip_layers into pipeline state (arXiv:2504.10317)
    skip_layers = pipe_cfg.get("skip_layers")
    if skip_layers:
        pipe._state["_skip_layers"] = set(skip_layers)

    return pipe


__all__ = [
    "CompressionPipeline",
    "ScoredEntry",
    "Scorer",
    "CompressionStrategy",
    "Monitor",
    "ModelAdapter",
    "build_pipeline",
    "list_available",
    "register_scorer",
    "register_strategy",
    "register_monitor",
    "register_adapter",
    "get_scorer",
    "get_strategy",
    "get_monitor",
    "get_adapter",
]
