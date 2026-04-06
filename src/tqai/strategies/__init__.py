"""Compression strategy modules."""

from tqai.pipeline.registry import register_strategy
from tqai.strategies.delta import DeltaStrategy
from tqai.strategies.delta2 import SecondOrderDelta
from tqai.strategies.tiered import TieredStrategy
from tqai.strategies.window import WindowStrategy

register_strategy("tiered", TieredStrategy)
register_strategy("delta", DeltaStrategy)
register_strategy("delta2", SecondOrderDelta)
register_strategy("window", WindowStrategy)
